"""
moe_ddpg_spatial_weighted_canny.py
====================================
Phase-Conditioned Multi-Head DDPG with full ablation-switch support.

Ablation flags (all passed as constructor arguments):
    num_experts     : 1 (baseline single-head) | K>1 (phase-conditioned)
    reward_mode     : set on env.Paint, not here (env.py branches on it)
    noise_mode      : 'isotropic' (baseline) | 'paint' (PaintNoise)
    temporal_critic : False (baseline ResNet_wobn) | True (TemporalCritic)
    loss_mode       : 'fixed' (baseline) | 'adaptive' (phase-scheduled)

Design principles
------------------
- When num_experts=1, MultiHeadActor reduces exactly to the baseline actor.
- When temporal_critic=False, a plain ResNet_wobn is used as the critic.
- When noise_mode='isotropic', select_action() matches ddpg.py line-for-line.
- When loss_mode='fixed', update_policy() uses w_gan=w_q=0.5 throughout.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from Renderer.model import FCN
from DRL.rpm import rpm
from DRL.actor import ResNet
from DRL.critic import ResNet_wobn
from DRL.wgan import cal_reward, update as update_discriminator, \
                     save_gan, load_gan
from utils.util import hard_update, soft_update, to_numpy, to_tensor

import DRL.ddpg as _ddpg_module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ddpg_module.Decoder.to(device)
_ddpg_module.coord = _ddpg_module.coord.to(device)

# ── Local Decoder & coord grid ────────────────────────────────────────────────
_Decoder = FCN()
_Decoder.load_state_dict(torch.load('../renderer.pkl'))
_Decoder.to(device).eval()

_coord = torch.zeros([1, 2, 128, 128])
for _i in range(128):
    for _j in range(128):
        _coord[0, 0, _i, _j] = _i / 127.
        _coord[0, 1, _i, _j] = _j / 127.
_coord = _coord.to(device)

_criterion = nn.MSELoss()


def decode(x, canvas, num_strokes=5):
    x = x.view(-1, 13)
    stroke = 1 - _Decoder(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, num_strokes, 1, 128, 128)
    color_stroke = color_stroke.view(-1, num_strokes, 3, 128, 128)
    for i in range(num_strokes):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  Phase weights  (deterministic — no learned parameters)
# ══════════════════════════════════════════════════════════════════════════════

def phase_weights(T_norm, num_heads, sigma=0.35):
    """
    T_norm  : (B, 1) float in [0,1]
    Returns : (B, num_heads) softmax-normalised weights
    """
    if num_heads == 1:
        return torch.ones(T_norm.shape[0], 1, device=T_norm.device)
    centers = torch.linspace(0.0, 1.0, num_heads, device=T_norm.device)
    diff    = T_norm - centers.unsqueeze(0)          # (B, num_heads)
    log_w   = -(diff ** 2) / (2 * sigma ** 2)
    return F.softmax(log_w, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-Head Actor
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadActor(nn.Module):
    """
    Shared ResNet-18 backbone with num_heads parallel output heads.

    When num_heads=1 this reduces exactly to the baseline ResNet actor:
    one FC(512 -> action_dim) with sigmoid output.

    When num_heads>1, the final action is the Gaussian-weighted mixture
    of head outputs, where weights are computed deterministically from T_norm.
    """

    def __init__(self, num_inputs, depth, num_outputs, num_heads, sigma=0.35):
        super().__init__()
        self.num_heads   = num_heads
        self.num_outputs = num_outputs
        self.sigma       = sigma

        from DRL.actor import cfg, BasicBlock
        block, num_blocks = cfg(depth)

        self.in_planes = 64
        self.conv1  = nn.Conv2d(num_inputs, 64, kernel_size=3,
                                stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        feat_dim = 512 * block.expansion

        self.heads = nn.ModuleList([
            nn.Linear(feat_dim, num_outputs) for _ in range(num_heads)
        ])

    def _make_layer(self, block, planes, num_blocks, stride):
        from DRL.actor import BasicBlock
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, T_norm):
        feat = F.relu(self.bn1(self.conv1(x)))
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        feat = F.avg_pool2d(feat, 4)
        feat = feat.view(feat.size(0), -1)              # (B, feat_dim)

        head_out = torch.stack(
            [h(feat) for h in self.heads], dim=1)       # (B, num_heads, action_dim)

        w      = phase_weights(T_norm, self.num_heads, self.sigma)  # (B, num_heads)
        action = (w.unsqueeze(-1) * head_out).sum(dim=1)            # (B, action_dim)
        return torch.sigmoid(action)


# ══════════════════════════════════════════════════════════════════════════════
#  Temporal Critic  (ablation: temporal_critic=True)
# ══════════════════════════════════════════════════════════════════════════════

class TemporalCritic(nn.Module):
    """
    ResNet_wobn backbone with T_norm concatenated explicitly after pooling.

    FC(512 + 1 -> 1) instead of FC(512 -> 1).
    This gives the critic an undiluted temporal feature, resolving the
    1/512 signal-to-noise ratio of the baseline spatial-channel encoding.
    """

    def __init__(self, num_inputs, depth, num_outputs):
        super().__init__()
        self.backbone = ResNet_wobn(num_inputs, depth, num_outputs)
        # Remove the backbone's own final FC; we replace it
        self.backbone.fc  = nn.Identity()
        feat_dim          = 512
        self.temporal_fc  = nn.Linear(feat_dim + 1, num_outputs)

    def forward(self, x, T_norm=None):
        # Run through all backbone layers up to avg-pool
        feat = self.backbone.relu_1(self.backbone.conv1(x))
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)
        feat = F.avg_pool2d(feat, 4)
        feat = feat.view(feat.size(0), -1)              # (B, 512)

        if T_norm is None:
            # Fallback: read T from channel 9 (constant spatial channel)
            T_norm = x[:, 9:10, 0, 0]                  # (B, 1)

        feat_t = torch.cat([feat, T_norm], dim=1)       # (B, 513)
        return self.temporal_fc(feat_t)                 # (B, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  OU Noise
# ══════════════════════════════════════════════════════════════════════════════

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu    = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


# ══════════════════════════════════════════════════════════════════════════════
#  PaintNoise  (ablation: noise_mode='paint')
# ══════════════════════════════════════════════════════════════════════════════

class PaintNoise:
    """
    Structured exploration for painting.

    Four innovations over the isotropic baseline:
    1. Per-parameter-group noise scales (position / control / width / color / opacity)
    2. Phase-scaled magnitude: scale * (1 - 0.8 * T_norm) — wide early, tight late
    3. OU temporal correlation at 0.3x weight
    4. Von Mises geometric guidance from the target edge-orientation field
       (applied separately in noise_action() when target_theta is available)
    """

    # Noise standard deviations per parameter group
    SCALES = {
        'position': 0.05,   # x0, y0, x2, y2
        'control':  0.08,   # x1_ctrl, y1_ctrl
        'width':    0.02,   # w0, w1, w2
        'color':    0.08,   # R, G, B
        'opacity':  0.03,   # alpha
    }

    # Parameter indices within each 13-dim stroke vector
    INDICES = {
        'position': [0, 1, 4, 5],
        'control':  [2, 3],
        'width':    [6, 7, 8],
        'color':    [9, 10, 11],
        'opacity':  [12],
    }

    def __init__(self, num_strokes=5):
        self.num_strokes = num_strokes
        self.action_dim  = num_strokes * 13
        self.ou          = OUNoise(self.action_dim)

    def reset(self):
        self.ou.reset()

    def noise(self, action, phase=0.5, noise_factor=1.0):
        """
        action       : (B, N*13) float32 in [0,1]
        phase        : float or (B,) array — T/M, 0=early 1=late
        noise_factor : global scale multiplier
        Returns      : (B, N*13) clipped to [0,1]
        """
        phase        = np.asarray(phase).reshape(-1, 1)        # (B, 1)
        phase_scale  = 1.0 - 0.8 * phase                      # (B, 1)

        noise = np.zeros_like(action)
        for group, scale in self.SCALES.items():
            for stroke_idx in range(self.num_strokes):
                offset = stroke_idx * 13
                for idx in self.INDICES[group]:
                    col = offset + idx
                    noise[:, col] = np.random.normal(
                        0,
                        scale * phase_scale.squeeze(1) * noise_factor
                    )

        # OU temporal correlation
        ou_noise  = self.ou.noise() * noise_factor * 0.3       # (action_dim,)
        noise    += ou_noise

        return np.clip(action + noise, 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  MoEDDPG  — Phase-Conditioned DDPG with full ablation support
# ══════════════════════════════════════════════════════════════════════════════

class MoEDDPG:
    """
    Parameters
    ----------
    num_experts      : number of phase heads (1 = baseline single-head)
    sigma            : Gaussian width of phase responsibility curves
    num_strokes      : strokes per step N
    noise_mode       : 'isotropic' (baseline) | 'paint' (PaintNoise)
    temporal_critic  : False (baseline) | True (TemporalCritic)
    loss_mode        : 'fixed' (baseline) | 'adaptive' (phase-scheduled)
    batch_size, env_batch, max_step, tau, discount, rmsize, writer,
    resume, output_path : same as original ddpg.py
    """

    def __init__(self,
                 num_experts=3,
                 sigma=0.35,
                 num_strokes=5,
                 noise_mode='paint',
                 temporal_critic=True,
                 loss_mode='adaptive',
                 batch_size=64,
                 env_batch=1,
                 max_step=40,
                 tau=0.001,
                 discount=0.9,
                 rmsize=800,
                 writer=None,
                 resume=None,
                 output_path=None,
                 # kept for argparse API compatibility
                 gate_entropy_coef=0.0,
                 diversity_coef=0.0,
                 direct_q_coef=0.0):

        self.num_experts      = num_experts
        self.sigma            = sigma
        self.num_strokes      = num_strokes
        self.action_dim       = num_strokes * 13
        self.noise_mode       = noise_mode
        self.temporal_critic  = temporal_critic
        self.loss_mode        = loss_mode
        self.max_step         = max_step
        self.env_batch        = env_batch
        self.batch_size       = batch_size
        self.tau              = tau
        self.discount         = discount
        self.writer           = writer
        self.output_path      = output_path
        self.log              = 0

        # ── Actor: multi-head (or single-head when num_experts=1) ────────────
        self.actor        = MultiHeadActor(9, 18, self.action_dim, num_experts, sigma)
        self.actor_target = MultiHeadActor(9, 18, self.action_dim, num_experts, sigma)

        # ── Critic: temporal or standard ─────────────────────────────────────
        if temporal_critic:
            self.critic        = TemporalCritic(3 + 9, 18, 1)
            self.critic_target = TemporalCritic(3 + 9, 18, 1)
        else:
            # Baseline: plain ResNet_wobn; T encoded as spatial channel
            self.critic        = ResNet_wobn(3 + 9, 18, 1)
            self.critic_target = ResNet_wobn(3 + 9, 18, 1)

        # ── Optimisers ────────────────────────────────────────────────────────
        self.actor_optim  = Adam(self.actor.parameters(),  lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-2)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        if resume is not None:
            self.load_weights(resume)

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.memory = rpm(rmsize * max_step)

        # ── Episode state ─────────────────────────────────────────────────────
        self.state       = [None] * env_batch
        self.action      = [None] * env_batch
        self.noise_level = np.zeros(env_batch)
        self.noise_generator = PaintNoise(num_strokes=num_strokes)

        self._move_to_device()

    # ── Device management ─────────────────────────────────────────────────────

    def _move_to_device(self):
        for m in [self.actor, self.actor_target,
                  self.critic, self.critic_target]:
            m.to(device)

    # ── Observation normalisation ─────────────────────────────────────────────

    def _to_device_tensor(self, state):
        if not isinstance(state, torch.Tensor):
            return torch.tensor(state, dtype=torch.float32, device=device)
        return state.to(device=device, dtype=torch.float32)

    def _norm_obs(self, state):
        """
        Returns (norm_img, T_norm):
            norm_img : (B, 9, 128, 128)
            T_norm   : (B, 1)
        """
        state  = self._to_device_tensor(state)
        n      = state.shape[0]
        T_raw  = state[:, 6:7, 0, 0]
        T_norm = T_raw / self.max_step

        norm_img = torch.cat([
            state[:, :6] / 255.0,
            state[:, 6:7] / self.max_step,
            _coord.expand(n, 2, 128, 128),
        ], dim=1)

        return norm_img, T_norm

    # ── Actor forward ─────────────────────────────────────────────────────────

    def play(self, state, target=False):
        norm_img, T_norm = self._norm_obs(state)
        actor = self.actor_target if target else self.actor
        return actor(norm_img, T_norm)

    # ── Exploration ───────────────────────────────────────────────────────────

    def noise_action(self, action, target_theta=None, noise_factor=1.0):
        """
        Von Mises geometric guidance aligned to the target edge-orientation
        field. Applied on top of the base noise from select_action().

        target_theta : dict with keys 'theta' (B,) and 'mag' (B,)
        """
        B = action.shape[0]

        # Isotropic Gaussian baseline component (always applied)
        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_level[:, np.newaxis],
            size=action.shape
        ).astype('float32')
        action = action + noise * noise_factor

        # Von Mises geometric guidance (PaintNoise path only)
        if target_theta is not None and self.noise_mode == 'paint':
            N       = action.shape[1] // 13
            ar      = action.reshape(B, N, 13)
            t_theta = to_numpy(target_theta['theta'])   # (B,)
            t_mag   = to_numpy(target_theta['mag'])     # (B,)

            for i in range(N):
                dx = ar[:, i, 4] - ar[:, i, 0]
                dy = ar[:, i, 5] - ar[:, i, 1]
                agent_theta = np.arctan2(dy, dx)

                # Axial symmetry: flip target by pi if agent stroke points
                # more than 90 deg away
                diff = (agent_theta - t_theta + np.pi) % (2 * np.pi) - np.pi
                adjusted_target = np.where(
                    np.abs(diff) > (np.pi / 2),
                    (t_theta + np.pi) % (2 * np.pi),
                    t_theta
                )

                # Saliency-gated Von Mises concentration
                kappa = np.clip(5.0 * t_mag / (noise_factor + 0.05), 0.5, 15.0)
                geo_samples = np.random.vonmises(mu=adjusted_target, kappa=kappa)

                # Re-apply sampled angle, preserve stroke length
                L = np.sqrt(dx ** 2 + dy ** 2)
                ar[:, i, 4] = ar[:, i, 0] + L * np.cos(geo_samples)
                ar[:, i, 5] = ar[:, i, 1] + L * np.sin(geo_samples)

            action = ar.reshape(B, -1)

        return np.clip(action, 0, 1)

    def select_action(self, state, noise_factor=0, target_theta=None, training=False):
        """
        noise_mode='isotropic': matches ddpg.py noise_action() exactly.
        noise_mode='paint':     uses PaintNoise per-group + phase decay,
                                then optionally Von Mises guidance.
        """
        self._set_mode(train=False)
        with torch.no_grad():
            action = to_numpy(self.play(state))

        if noise_factor > 0 and training:
            if self.noise_mode == 'isotropic':
                # ── Baseline: one random level per env, fixed at reset ────────
                noise = np.zeros(action.shape)
                for i in range(self.env_batch):
                    noise[i] = np.random.normal(
                        0, self.noise_level[i], action.shape[1:]
                    ).astype('float32')
                action = np.clip(action + noise, 0, 1)

            else:  # 'paint'
                # ── PaintNoise: per-group + phase decay + OU correlation ──────
                T_norm = (state[:, 6:7, 0, 0].float() / self.max_step).cpu().numpy()
                action = self.noise_generator.noise(
                    action, phase=T_norm, noise_factor=noise_factor)

                # Von Mises geometric guidance
                if target_theta is not None:
                    action = self.noise_action(action, target_theta, noise_factor)

        self._set_mode(train=True)
        self.action = action
        return action

    def reset(self, obs, factor):
        self.state       = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)
        self.noise_generator.reset()

    def observe(self, reward, state, done, step):
        s0 = torch.tensor(self.state, device='cpu')
        a  = to_tensor(self.action,             'cpu')
        r  = to_tensor(reward,                  'cpu')
        s1 = torch.tensor(state, device='cpu')
        d  = to_tensor(done.astype('float32'),  'cpu')
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    # ── GAN update ────────────────────────────────────────────────────────────

    def _update_gan(self, state):
        canvas = state[:, :3].float() / 255.0
        gt     = state[:, 3:6].float() / 255.0
        fake, real, penal = update_discriminator(canvas, gt)
        if self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/gan_fake',  fake,  self.log)
            self.writer.add_scalar('train_moe/gan_real',  real,  self.log)
            self.writer.add_scalar('train_moe/gan_penal', penal, self.log)

    # ── Critic forward ────────────────────────────────────────────────────────

    def _evaluate(self, state, action, target=False):
        T       = state[:, 6:7, 0, 0].float()          # (B, 1) raw step counter
        gt      = state[:, 3:6].float() / 255.0
        canvas0 = state[:, :3].float()  / 255.0
        canvas1 = decode(action, canvas0, num_strokes=self.num_strokes)

        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)

        n      = state.shape[0]
        coord_ = _coord.expand(n, 2, 128, 128)
        merged = torch.cat([
            canvas0, canvas1, gt,
            (T + 1).view(n, 1, 1, 1).expand(n, 1, 128, 128) / self.max_step,
            coord_,
        ], dim=1)                                       # (B, 12, 128, 128)

        T_norm      = T / self.max_step                 # (B, 1) normalised
        critic_net  = self.critic_target if target else self.critic

        if self.temporal_critic:
            Q = critic_net(merged, T_norm)
        else:
            # Baseline: critic receives T as spatial channel; T_norm unused
            Q = critic_net(merged)

        if not target and self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/expect_reward', Q.mean(),         self.log)
            self.writer.add_scalar('train_moe/gan_reward',    gan_reward.mean(), self.log)

        return Q + gan_reward, Q, gan_reward

    # ── Policy update ─────────────────────────────────────────────────────────

    def update_policy(self, lr, step=0, train_times=0):
        self.log += 1

        for pg in self.actor_optim.param_groups:  pg['lr'] = lr[1]
        for pg in self.critic_optim.param_groups: pg['lr'] = lr[0]

        state, action, reward, next_state, terminal = \
            self.memory.sample_batch(self.batch_size, device)

        self._update_gan(next_state)

        # ── Critic update (TD error) ──────────────────────────────────────────
        cur_q = self._evaluate(state, action)[1]        # Q only (no GAN)

        with torch.no_grad():
            next_action = self.play(next_state, target=True)
            next_q      = self._evaluate(next_state, next_action, target=True)[1]
            target_q    = torch.clamp(
                self.discount * ((1 - terminal.float()).view(-1, 1)) * next_q
                + reward.view(-1, 1),
                -100, 100
            )

        value_loss = _criterion(cur_q, target_q)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # ── Actor update ──────────────────────────────────────────────────────
        norm_img, T_norm   = self._norm_obs(state)
        action_pred        = self.actor(norm_img, T_norm)
        _, q_val, gan_rew  = self._evaluate(state.detach(), action_pred)

        # loss_mode switch ─────────────────────────────────────────────────────
        if self.loss_mode == 'adaptive':
            progress = step / max(train_times, 1)
            if progress < 0.3:
                w_gan, w_q = 0.9, 0.1      # early: learn to paint realistically
            elif progress < 0.7:
                w_gan, w_q = 0.6, 0.4      # mid: balanced
            else:
                w_gan, w_q = 0.3, 0.7      # late: refine pixel accuracy
        else:
            # Baseline: fixed equal weighting throughout
            w_gan, w_q = 0.5, 0.5

        loss = w_gan * (-gan_rew.mean()) + w_q * (-q_val.mean() * 0.2)

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        # ── Target sync ───────────────────────────────────────────────────────
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        # ── TensorBoard logging ───────────────────────────────────────────────
        if self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/policy_loss', loss.item(),       self.log)
            self.writer.add_scalar('train_moe/value_loss',  value_loss.item(), self.log)
            self.writer.add_scalar('train_moe/w_gan',       w_gan,             self.log)
            self.writer.add_scalar('train_moe/w_q',         w_q,               self.log)
            for k, head in enumerate(self.actor.heads):
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in head.parameters() if p.grad is not None
                ) ** 0.5
                self.writer.add_scalar(
                    f'train_moe/head{k}_grad_norm', grad_norm, self.log)

        return q_val.mean(), value_loss

    # ── Mode helpers ──────────────────────────────────────────────────────────

    def _set_mode(self, train=True):
        fn = 'train' if train else 'eval'
        for m in [self.actor, self.actor_target,
                  self.critic, self.critic_target]:
            getattr(m, fn)()

    def eval(self):  self._set_mode(train=False)
    def train(self): self._set_mode(train=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path):
        self.eval()
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(),  f'{path}/moe_actor.pkl')
        torch.save(self.critic.state_dict(), f'{path}/moe_critic.pkl')
        save_gan(path)
        self._move_to_device()
        self.train()

    def load_weights(self, path):
        try:
            print(f'[MoEDDPG] Loading actor from {path}')
            self.actor.load_state_dict(
                torch.load(f'{path}/moe_actor.pkl', map_location=device))
        except FileNotFoundError:
            print(f'[MoEDDPG] No actor checkpoint at {path}, fresh start.')
        try:
            self.critic.load_state_dict(
                torch.load(f'{path}/moe_critic.pkl', map_location=device))
            print('[MoEDDPG] Critic loaded.')
        except FileNotFoundError:
            print(f'[MoEDDPG] No critic checkpoint at {path}, fresh start.')
        try:
            load_gan(path)
        except Exception:
            pass

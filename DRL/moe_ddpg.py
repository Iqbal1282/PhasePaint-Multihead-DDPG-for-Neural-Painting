"""
Phase-Conditioned Multi-Head DDPG for Painting
===============================================

Root-cause analysis of why MoE v1/v2 underperformed
-----------------------------------------------------
The MoE approach had three structural problems that no amount of loss
tuning could fully fix:

  1. Routing instability: A learned gate must discover which expert to use
     *while* the experts are still learning. Early in training all experts
     are identical, so the gate gets no gradient signal to differentiate
     them. Late in training the gate dominates and some experts collapse.

  2. Train/inference mismatch: Gumbel-Softmax (used during training) and
     argmax (used during inference) sample different experts for the same
     state, creating an inconsistency that destabilises the critic.

  3. Competing gradients: The gate, diversity loss, direct-Q loss, and
     entropy bonus all push the actor parameters in different directions
     through a single optimiser step, causing gradient conflict.

The fundamental insight we missed
----------------------------------
The step counter T already tells us exactly what "phase" we are in.
T=0 → blank canvas → we should paint broad strokes.
T=39 → near-complete canvas → we should add fine detail.

We do NOT need to *learn* when to use which expert — T encodes it exactly.
This means the routing problem dissolves entirely.

Design: Phase-Conditioned Multi-Head Actor
-------------------------------------------
Architecture:
  ┌─────────────────────────────┐
  │   Shared ResNet-18 Backbone │   (same as baseline, input 9-ch)
  │   → 512-dim feature vector  │
  └──────────────┬──────────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
     Head 0   Head 1   Head 2      (N linear heads, each 512→65)
    (coarse) (medium)  (fine)
        │        │        │
        └────────┼────────┘
                 │ weighted by phase weights w(T)
                 ▼
           Final action (B, 65)

Phase weights w(T) are smooth, differentiable, and deterministic:
  - Computed from T alone (no learned gate)
  - Same at train time and inference time → zero train/inference gap
  - Each head receives gradient proportional to how much it is used at
    that step — heads used in early steps learn coarse strokes, heads
    used in late steps learn fine strokes, naturally.

Why this solves all three problems
------------------------------------
  Problem 1 (routing instability): No learned router. T is fixed and known.
  Problem 2 (train/inference mismatch): Phase weights are deterministic,
    identical at train and inference.
  Problem 3 (gradient conflict): No diversity loss, no entropy bonus, no
    Gumbel. Just one clean actor loss through a single optimiser.

Why it beats the single-head baseline
---------------------------------------
  The baseline actor has ONE FC head (512→65) that must produce both
  coarse and fine strokes from the same weights. Phase-conditioned heads
  allow early layers (shared backbone) to learn image representation while
  each head specialises on its time regime.  This is the same principle as
  why mixture-of-experts works in LLMs — not routing, but specialisation.

Phase weight computation
-------------------------
  For N heads and step t in [0, max_step):
    progress p = t / max_step   ∈ [0, 1]
    Each head k is centred at c_k = k / (N-1) with width σ.
    w_k(p) = exp(-(p - c_k)^2 / (2σ^2))
    Then softmax over k to get a valid mixture.

  This gives smooth "responsibility" curves:
    Head 0 fires strongly at t=0   (blank canvas, coarse strokes)
    Head 1 fires strongly at t=20  (mid-painting, medium strokes)
    Head 2 fires strongly at t=39  (near-done, fine detail)

Implementation
--------------
  Only moe_ddpg.py (this file) and train_moe.py are new.
  All original files (ddpg.py, actor.py, critic.py, env.py, etc.) unchanged.
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

# ── patch ddpg module-level Decoder onto device ───────────────────────────────
import DRL.ddpg as _ddpg_module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ddpg_module.Decoder.to(device)
_ddpg_module.coord = _ddpg_module.coord.to(device)

# ── local Decoder & coord grid ────────────────────────────────────────────────
_Decoder = FCN()
_Decoder.load_state_dict(torch.load('../renderer.pkl'))
_Decoder.to(device)
_Decoder.eval()

_coord = torch.zeros([1, 2, 128, 128])
for _i in range(128):
    for _j in range(128):
        _coord[0, 0, _i, _j] = _i / 127.
        _coord[0, 1, _i, _j] = _j / 127.
_coord = _coord.to(device)

_criterion = nn.MSELoss()


def decode(x, canvas):
    """Decode 5 strokes onto canvas. Identical to ddpg.decode."""
    x = x.view(-1, 13)
    stroke = 1 - _Decoder(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  Phase weights  (deterministic, no learned parameters)
# ══════════════════════════════════════════════════════════════════════════════

def phase_weights(T_norm: torch.Tensor, num_heads: int,
                  sigma: float = 0.35) -> torch.Tensor:
    """
    Compute smooth phase-mixture weights from normalised step progress.

    Parameters
    ----------
    T_norm    : (B, 1) float tensor in [0, 1]  — normalised step t / max_step
    num_heads : int — number of heads N
    sigma     : float — width of each head's Gaussian responsibility curve

    Returns
    -------
    weights   : (B, N) float tensor, rows sum to 1 (softmax-normalised)

    Each head k is centred at progress c_k = k / (N-1).
    w_k ∝ exp(-(p - c_k)^2 / (2σ^2))
    """
    if num_heads == 1:
        return torch.ones(T_norm.shape[0], 1, device=T_norm.device)

    centers = torch.linspace(0.0, 1.0, num_heads,
                             device=T_norm.device)           # (N,)
    # T_norm: (B, 1),  centers: (N,)  →  diff: (B, N)
    diff = T_norm - centers.unsqueeze(0)                     # (B, N)
    log_w = -(diff ** 2) / (2 * sigma ** 2)                 # (B, N)
    return F.softmax(log_w, dim=-1)                          # (B, N), sums to 1


# ══════════════════════════════════════════════════════════════════════════════
#  Phase-Conditioned Multi-Head Actor
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadActor(nn.Module):
    """
    Shared ResNet-18 backbone with N parallel output heads.

    The backbone is identical to the baseline actor.  The final FC layer
    is replaced by N separate linear heads (512 → 65 each).

    The step-progress T_norm is used to compute Gaussian phase weights
    over the heads.  The final action is the weighted sum:

        action = sum_k( w_k(T) * head_k(backbone_features) )

    All N heads receive gradient at every step, scaled by their phase weight.
    Early heads naturally specialise on coarse strokes; late heads on fine ones.
    """

    def __init__(self, num_inputs: int, depth: int,
                 num_outputs: int, num_heads: int, sigma: float = 0.35):
        super().__init__()
        self.num_heads  = num_heads
        self.num_outputs = num_outputs
        self.sigma       = sigma

        # ── Shared ResNet backbone (identical to baseline, minus final FC) ────
        from DRL.actor import ResNet as _ResNet, cfg, BasicBlock
        block, num_blocks = cfg(depth)

        self.in_planes = 64
        self.conv1  = nn.Conv2d(num_inputs, 64, kernel_size=3,
                                stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        feat_dim = 512 * block.expansion   # 512 for BasicBlock (ResNet-18/34)

        # ── N parallel output heads ───────────────────────────────────────────
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

    def forward(self, x: torch.Tensor,
                T_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : (B, num_inputs, 128, 128)  — normalised observation
        T_norm : (B, 1)                      — normalised step progress

        Returns
        -------
        action : (B, num_outputs)            — sigmoid-squashed mixture action
        """
        # ── Shared backbone ───────────────────────────────────────────────────
        feat = F.relu(self.bn1(self.conv1(x)))
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        feat = F.avg_pool2d(feat, 4)
        feat = feat.view(feat.size(0), -1)              # (B, feat_dim)

        # ── Per-head outputs ──────────────────────────────────────────────────
        head_out = torch.stack(
            [h(feat) for h in self.heads], dim=1)       # (B, N, 65)

        # ── Phase-weighted mixture ────────────────────────────────────────────
        w = phase_weights(T_norm, self.num_heads,
                          self.sigma)                    # (B, N)
        action = (w.unsqueeze(-1) * head_out).sum(dim=1)  # (B, 65)
        return torch.sigmoid(action)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase-Conditioned DDPG   (drop-in replacement for DDPG in ddpg.py)
# ══════════════════════════════════════════════════════════════════════════════

class MoEDDPG:
    """
    Phase-Conditioned Multi-Head DDPG.

    Drop-in compatible with the DDPG API used in train.py / train_moe.py.

    Parameters
    ----------
    num_experts   : number of phase heads (default 3)
    sigma         : Gaussian width of each head's responsibility curve.
                    Smaller → sharper specialisation (try 0.25–0.45).
    batch_size    : training batch size
    env_batch     : parallel environments
    max_step      : episode length
    tau           : soft-update rate
    discount      : reward discount
    rmsize        : replay buffer capacity multiplier
    writer        : TensorBoard writer or None
    resume        : checkpoint path or None
    output_path   : save directory
    """

    def __init__(self, num_experts=3, sigma=0.35,
                 batch_size=64, env_batch=1, max_step=40,
                 tau=0.001, discount=0.9, rmsize=800,
                 writer=None, resume=None, output_path=None,
                 # kept for API compatibility with train_moe.py argparse
                 gate_entropy_coef=0.0,
                 diversity_coef=0.0,
                 direct_q_coef=0.0):

        self.num_experts  = num_experts
        self.sigma        = sigma
        self.max_step     = max_step
        self.env_batch    = env_batch
        self.batch_size   = batch_size
        self.tau          = tau
        self.discount     = discount
        self.writer       = writer
        self.output_path  = output_path
        self.log          = 0

        # ── Multi-head actor + target ─────────────────────────────────────────
        self.actor        = MultiHeadActor(9, 18, 65, num_experts, sigma)
        self.actor_target = MultiHeadActor(9, 18, 65, num_experts, sigma)

        # ── Shared critic + target (identical to baseline) ────────────────────
        self.critic        = ResNet_wobn(3 + 9, 18, 1)
        self.critic_target = ResNet_wobn(3 + 9, 18, 1)

        # ── Optimisers ────────────────────────────────────────────────────────
        self.actor_optim  = Adam(self.actor.parameters(),  lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-2)

        # ── Target network init ───────────────────────────────────────────────
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

        self._move_to_device()

    # ── Device management ─────────────────────────────────────────────────────

    def _move_to_device(self):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

    # ── Observation handling ──────────────────────────────────────────────────

    def _to_device_tensor(self, state):
        if not isinstance(state, torch.Tensor):
            return torch.tensor(state, dtype=torch.float32, device=device)
        return state.to(device=device, dtype=torch.float32)

    def _norm_obs(self, state):
        """
        Returns (norm_img, T_norm):
          norm_img : (B, 9, 128, 128) — canvas+gt normalised, with coord
          T_norm   : (B, 1)           — step progress in [0, 1]
        """
        state = self._to_device_tensor(state)
        n     = state.shape[0]
        T_raw = state[:, 6:7, 0, 0]                  # (B, 1) step counter
        T_norm = T_raw / self.max_step                # normalised ∈ [0,1]

        norm_img = torch.cat([
            state[:, :6] / 255.0,
            state[:, 6:7] / self.max_step,
            _coord.expand(n, 2, 128, 128),
        ], dim=1)                                     # (B, 9, 128, 128)

        return norm_img, T_norm

    # ── Actor forward ─────────────────────────────────────────────────────────

    def play(self, state, target=False):
        """Return action (B, 65) using phase-conditioned multi-head actor."""
        norm_img, T_norm = self._norm_obs(state)
        actor = self.actor_target if target else self.actor
        return actor(norm_img, T_norm)

    # ── Public API (same as DDPG in ddpg.py) ─────────────────────────────────

    def select_action(self, state, noise_factor=0, return_fix=False):
        self._set_mode(train=False)
        with torch.no_grad():
            action = to_numpy(self.play(state))
        if noise_factor > 0:
            for i in range(self.env_batch):
                noise = np.random.normal(
                    0, self.noise_level[i], action.shape[1:]).astype('float32')
                action[i] = np.clip(action[i] + noise, 0, 1)
        self._set_mode(train=True)
        self.action = action
        return self.action

    def reset(self, obs, factor):
        self.state       = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def observe(self, reward, state, done, step):
        s0 = torch.tensor(self.state, device='cpu')
        a  = to_tensor(self.action,              'cpu')
        r  = to_tensor(reward,                   'cpu')
        s1 = torch.tensor(state,     device='cpu')
        d  = to_tensor(done.astype('float32'),   'cpu')
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    # ── GAN & Q evaluation ────────────────────────────────────────────────────

    def _update_gan(self, state):
        canvas = state[:, :3].float() / 255.0
        gt     = state[:, 3:6].float() / 255.0
        fake, real, penal = update_discriminator(canvas, gt)
        if self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/gan_fake',  fake,  self.log)
            self.writer.add_scalar('train_moe/gan_real',  real,  self.log)
            self.writer.add_scalar('train_moe/gan_penal', penal, self.log)

    def _evaluate(self, state, action, target=False):
        T       = state[:, 6:7, 0, 0].float()          # (B, 1) step value
        gt      = state[:, 3:6].float() / 255.0
        canvas0 = state[:, :3].float()  / 255.0
        canvas1 = decode(action, canvas0)

        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)

        n      = state.shape[0]
        coord_ = _coord.expand(n, 2, 128, 128)
        merged = torch.cat([
            canvas0, canvas1, gt,
            (T + 1).view(n, 1, 1, 1).expand(n, 1, 128, 128) / self.max_step,
            coord_,
        ], dim=1)

        Q = (self.critic_target if target else self.critic)(merged)

        if not target and self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/expect_reward', Q.mean(),         self.log)
            self.writer.add_scalar('train_moe/gan_reward',    gan_reward.mean(), self.log)

        return Q + gan_reward, gan_reward

    # ── Policy update ─────────────────────────────────────────────────────────

    def update_policy(self, lr):
        """
        Identical update structure to the baseline DDPG.
        No routing, no diversity loss, no entropy bonus — clean TD + policy gradient.
        """
        self.log += 1

        for pg in self.critic_optim.param_groups: pg['lr'] = lr[0]
        for pg in self.actor_optim.param_groups:  pg['lr'] = lr[1]

        state, action, reward, next_state, terminal = \
            self.memory.sample_batch(self.batch_size, device)

        # ── GAN update ────────────────────────────────────────────────────────
        self._update_gan(next_state)

        # ── Critic update (TD) ────────────────────────────────────────────────
        with torch.no_grad():
            next_action = self.play(next_state, target=True)
            target_q, _ = self._evaluate(next_state, next_action, target=True)
            target_q    = self.discount * \
                          ((1 - terminal.float()).view(-1, 1)) * target_q

        cur_q, step_reward = self._evaluate(state, action)
        target_q = target_q + step_reward.detach()

        value_loss = _criterion(cur_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        # ── Actor update (policy gradient) ────────────────────────────────────
        action_pred = self.play(state)
        pre_q, _    = self._evaluate(state.detach(), action_pred)
        policy_loss = -pre_q.mean()

        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # ── Soft-update target networks ───────────────────────────────────────
        soft_update(self.actor_target,  self.actor,  self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        # ── Logging ───────────────────────────────────────────────────────────
        if self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/policy_loss', policy_loss.item(), self.log)
            self.writer.add_scalar('train_moe/value_loss',  value_loss.item(),  self.log)
            # Log per-head gradient norms to verify specialisation is happening
            for k, head in enumerate(self.actor.heads):
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in head.parameters()
                    if p.grad is not None
                ) ** 0.5
                self.writer.add_scalar(
                    f'train_moe/head{k}_grad_norm', grad_norm, self.log)

        return -policy_loss, value_loss

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
            self.actor.load_state_dict(
                torch.load(f'{path}/moe_actor.pkl', map_location=device))
            print('[PhaseDDPG] Actor loaded.')
        except FileNotFoundError:
            print(f'[PhaseDDPG] No actor checkpoint at {path}, fresh start.')
        try:
            self.critic.load_state_dict(
                torch.load(f'{path}/moe_critic.pkl', map_location=device))
            print('[PhaseDDPG] Critic loaded.')
        except FileNotFoundError:
            print(f'[PhaseDDPG] No critic checkpoint at {path}, fresh start.')
        try:
            load_gan(path)
        except Exception:
            pass
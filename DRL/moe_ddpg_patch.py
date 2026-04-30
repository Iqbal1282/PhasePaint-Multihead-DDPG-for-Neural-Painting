"""
Phase-Conditioned Multi-Head DDPG  (v2 – four RL improvements)
==============================================================

Builds on the working phase-conditioned design and adds four targeted
RL techniques, each addressing a concrete weakness in the baseline.

──────────────────────────────────────────────────────────────────────
1. Prioritized Experience Replay (PER)
──────────────────────────────────────────────────────────────────────
Gap: The uniform replay buffer (rpm) treats all transitions equally.
     But transitions vary enormously in learning value: a stroke that
     suddenly closes a large MSE gap is far more informative than a
     stroke that makes no difference.  Uniform sampling rarely revisits
     the most instructive transitions.

Fix: Replace uniform sampling with a sum-tree priority queue.
     Priority of each transition = |TD error| + ε (small constant to
     ensure every transition has a non-zero chance of being sampled).
     Importance-sampling (IS) weights correct the distribution bias so
     gradients remain unbiased: w_i = (1/N · 1/P(i))^β,  
     where β anneals from 0.4 → 1.0 over training.

Why it helps for painting: early strokes (t=0..5) produce the largest
     MSE reduction. Uniform sampling buries them in 40× more late-step
     transitions. PER re-surfaces them automatically.

──────────────────────────────────────────────────────────────────────
2. N-step Returns
──────────────────────────────────────────────────────────────────────
Gap: The 1-step TD target uses only the immediate reward:
       y = r_t + γ * Q(s_{t+1}, π(s_{t+1}))
     A stroke at step t=1 that sets up a great stroke at step t=5
     receives zero credit for the downstream improvement — all of it
     accrues to step 5's transition.

Fix: Accumulate n consecutive rewards before bootstrapping:
       y = r_t + γr_{t+1} + γ²r_{t+2} + … + γ^{n-1}r_{t+n-1}
           + γ^n * Q(s_{t+n}, π(s_{t+n}))
     We use a circular n-step buffer that holds the last n transitions
     and emits a complete (s_t, a_t, R_t^n, s_{t+n}, done) tuple.

Why it helps for painting: n=3 means the critic "sees" 3 steps ahead,
     giving early strokes credit for the improvement they enable, not
     just their immediate delta.

──────────────────────────────────────────────────────────────────────
3. TD3 Double Critic
──────────────────────────────────────────────────────────────────────
Gap: DDPG's single critic systematically overestimates Q because the
     actor maximises Q and errors compound: the actor moves toward
     regions where Q is overestimated, the critic is then trained on
     inflated targets, further inflating Q.  This causes divergence or
     very slow convergence in practice.

Fix: Add a second critic (same architecture).  The TD target uses
       y = r + γ * min(Q1_target(s', a'), Q2_target(s', a'))
     The pessimistic min eliminates the upward bias.  Both critics are
     trained on the same target y.  The actor still optimises Q1 only
     (following TD3).  Delayed actor update (every d=2 critic steps)
     further stabilises training.

──────────────────────────────────────────────────────────────────────
4. Spatial Progress Bonus (intrinsic reward)
──────────────────────────────────────────────────────────────────────
Gap: The extrinsic reward r_t = (lastdis - dis) / ini_dis measures
     *global* MSE improvement.  Late in an episode, easy regions are
     already painted well; the remaining error is concentrated in hard
     regions (complex textures, fine detail).  Global MSE barely moves
     per step, so the reward signal becomes near-zero.  The agent has
     no incentive to tackle the hard regions.

Fix: Divide the canvas into a K×K grid of patches.  Track each patch's
     MSE separately.  The intrinsic bonus for a stroke is:
       r_intrinsic = α * mean_k( max(0, patch_error_before_k
                                         - patch_error_after_k) )
                         / ini_dis
     This bonus is non-zero even when global MSE barely changes,
     as long as *any patch* improved.  It encourages the agent to
     rotate through hard patches rather than ignoring them.

     Novelty: this is a spatial decomposition of the reward signal,
     not a standard RL technique — it is specific to the painting task.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque

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
#  1. Prioritized Experience Replay (sum-tree implementation)
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """
    Binary sum-tree for O(log N) priority sampling.

    Leaf nodes hold individual priorities; internal nodes hold sums.
    Position of the write pointer cycles through leaves.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity)   # internal + leaf nodes
        self.data     = [None] * capacity
        self.ptr      = 0
        self.size     = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority: float, data):
        leaf_idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(leaf_idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, idx: int, s: float) -> int:
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] \
               else self._retrieve(right, s - self.tree[left])

    def sample(self, s: float):
        leaf_idx = self._retrieve(0, s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Parameters
    ----------
    capacity  : max transitions stored
    alpha     : priority exponent  (0 = uniform, 1 = full prioritization)
    beta_start: IS-weight exponent start (anneals → 1.0 over training)
    beta_steps: number of update steps for full anneal
    eps       : small constant added to |TD error| for numerical stability
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_steps: int = 100_000,
                 eps: float = 1e-6):
        self.tree       = SumTree(capacity)
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.eps        = eps
        self._step      = 0
        self._max_prio  = 1.0   # tracks running maximum priority

    def append(self, transition):
        """Add a new transition with maximum current priority."""
        self.tree.add(self._max_prio ** self.alpha, transition)

    def size(self) -> int:
        return self.tree.size

    def sample_batch(self, batch_size: int, device):
        """
        Sample a batch proportional to priority.

        Returns standard (s, a, r, s', done) tensors PLUS
        IS weights tensor (B,) and leaf indices list for priority update.
        """
        self._step += 1
        beta = min(1.0,
                   self.beta_start + (1.0 - self.beta_start)
                   * self._step / self.beta_steps)

        segment = self.tree.total / batch_size
        indices, priorities, batch = [], [], []

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.sample(s)
            indices.append(idx)
            priorities.append(prio)
            batch.append(data)

        probs   = np.array(priorities) / (self.tree.total + 1e-8)
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()   # normalise so max weight = 1
        weights  = torch.tensor(weights, dtype=torch.float32, device=device)

        res = []
        for i in range(5):
            k = torch.stack([item[i] for item in batch], dim=0).to(device)
            res.append(k)
        return res[0], res[1], res[2], res[3], res[4], weights, indices

    def update_priorities(self, indices: list, td_errors: np.ndarray):
        """Update priorities after a learning step."""
        for idx, err in zip(indices, td_errors):
            prio = (abs(err) + self.eps) ** self.alpha
            self._max_prio = max(self._max_prio, prio)
            self.tree.update(idx, prio)


# ══════════════════════════════════════════════════════════════════════════════
#  2. N-step Return Buffer (per-environment circular buffer)
# ══════════════════════════════════════════════════════════════════════════════

class NStepBuffer:
    """
    Collects n consecutive transitions per environment and emits
    n-step return tuples: (s_t, a_t, R_t^n, s_{t+n}, done_t^n).

    R_t^n = sum_{k=0}^{n-1} gamma^k * r_{t+k}

    Parameters
    ----------
    n       : look-ahead steps
    gamma   : discount factor
    env_batch: number of parallel environments
    """

    def __init__(self, n: int, gamma: float, env_batch: int):
        self.n         = n
        self.gamma     = gamma
        self.env_batch = env_batch
        self.buffers   = [deque(maxlen=n) for _ in range(env_batch)]

    def push(self, states, actions, rewards, next_states, dones):
        """
        Push one step across all envs.
        Returns list of complete n-step tuples (one per env that has n steps),
        or empty list if not yet n steps accumulated.
        """
        ready = []
        for i in range(self.env_batch):
            self.buffers[i].append((
                states[i], actions[i], rewards[i],
                next_states[i], dones[i]
            ))
            if len(self.buffers[i]) == self.n:
                ready.append(self._make_nstep(i))
        return ready

    def _make_nstep(self, env_idx: int):
        buf   = self.buffers[env_idx]
        s0, a0, _, _, _ = buf[0]
        _, _, _, sn, dn = buf[-1]

        R = 0.0
        for k, (_, _, r_k, _, d_k) in enumerate(buf):
            R += (self.gamma ** k) * r_k
            if d_k:   # episode ended early; stop accumulating
                break

        done_n = buf[-1][4]
        return (torch.tensor(s0, dtype=torch.float32),
                torch.tensor(a0, dtype=torch.float32),
                torch.tensor([R],  dtype=torch.float32),
                torch.tensor(sn,   dtype=torch.float32),
                torch.tensor([done_n], dtype=torch.float32))

    def reset(self, env_idx: int):
        self.buffers[env_idx].clear()


# ══════════════════════════════════════════════════════════════════════════════
#  4. Spatial Progress Bonus (patch-level intrinsic reward)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialProgressBonus:
    """
    Divides the 128×128 canvas into a K×K grid of patches and computes
    per-patch MSE.  The intrinsic reward for a transition is the mean
    improvement across patches, scaled the same way as the extrinsic reward.

    Calling convention:
        bonus.before(canvas, gt)   – call BEFORE the stroke is applied
        r_intrinsic = bonus.after(canvas_new, gt, ini_dis)  – call AFTER

    Parameters
    ----------
    grid_k : number of patches per side (default 4 → 16 patches total)
    alpha  : weight of intrinsic bonus relative to extrinsic reward
    """

    def __init__(self, grid_k: int = 4, alpha: float = 0.3):
        self.K     = grid_k
        self.alpha = alpha
        self._patch_err_before = None   # (B, K, K)

    def _patch_mse(self, canvas: torch.Tensor,
                   gt: torch.Tensor) -> torch.Tensor:
        """
        canvas, gt: (B, 3, 128, 128) float in [0,1]
        Returns:    (B, K, K) patch-level MSE
        """
        B = canvas.shape[0]
        ps = 128 // self.K   # patch size
        err = ((canvas - gt) ** 2)   # (B, 3, 128, 128)
        # Mean over colour channels
        err = err.mean(dim=1, keepdim=True)   # (B, 1, 128, 128)
        # Reshape to patches
        err = err.view(B, 1, self.K, ps, self.K, ps)
        err = err.mean(dim=(3, 5))            # (B, 1, K, K)
        return err.squeeze(1)                 # (B, K, K)

    def before(self, canvas: torch.Tensor, gt: torch.Tensor):
        """Store patch MSE before the stroke."""
        self._patch_err_before = self._patch_mse(
            canvas.float() / 255, gt.float() / 255).detach()

    def after(self, canvas_new: torch.Tensor, gt: torch.Tensor,
              ini_dis: torch.Tensor) -> np.ndarray:
        """
        Compute intrinsic bonus = alpha * mean_patch_improvement / ini_dis.
        Returns numpy array shape (B,).
        """
        if self._patch_err_before is None:
            return np.zeros(canvas_new.shape[0])
        patch_after = self._patch_mse(
            canvas_new.float() / 255, gt.float() / 255).detach()
        improvement = (self._patch_err_before - patch_after).clamp(min=0)
        bonus = self.alpha * improvement.mean(dim=(1, 2)) \
                / (ini_dis.to(improvement.device) + 1e-8)
        return bonus.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  Phase weights (deterministic, no learned parameters)
# ══════════════════════════════════════════════════════════════════════════════

def phase_weights(T_norm: torch.Tensor, num_heads: int,
                  sigma: float = 0.35) -> torch.Tensor:
    if num_heads == 1:
        return torch.ones(T_norm.shape[0], 1, device=T_norm.device)
    centers = torch.linspace(0.0, 1.0, num_heads, device=T_norm.device)
    diff    = T_norm - centers.unsqueeze(0)
    return F.softmax(-(diff ** 2) / (2 * sigma ** 2), dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase-Conditioned Multi-Head Actor
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadActor(nn.Module):
    def __init__(self, num_inputs, depth, num_outputs, num_heads, sigma=0.35):
        super().__init__()
        self.num_heads   = num_heads
        self.num_outputs = num_outputs
        self.sigma       = sigma

        from DRL.actor import cfg, BasicBlock
        block, num_blocks = cfg(depth)

        self.in_planes = 64
        self.conv1  = nn.Conv2d(num_inputs, 64, 3, 2, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], 2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)

        feat_dim   = 512 * block.expansion
        self.heads = nn.ModuleList([
            nn.Linear(feat_dim, num_outputs) for _ in range(num_heads)])

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
        feat = self.layer1(feat); feat = self.layer2(feat)
        feat = self.layer3(feat); feat = self.layer4(feat)
        feat = F.avg_pool2d(feat, 4).view(feat.size(0), -1)
        head_out = torch.stack([h(feat) for h in self.heads], dim=1)
        w = phase_weights(T_norm, self.num_heads, self.sigma)
        return torch.sigmoid((w.unsqueeze(-1) * head_out).sum(dim=1))


# ══════════════════════════════════════════════════════════════════════════════
#  3. TD3 Double Critic
# ══════════════════════════════════════════════════════════════════════════════

class DoubleCritic(nn.Module):
    """Two independent ResNet_wobn critics; forward returns (Q1, Q2)."""

    def __init__(self, num_inputs, depth):
        super().__init__()
        self.Q1 = ResNet_wobn(num_inputs, depth, 1)
        self.Q2 = ResNet_wobn(num_inputs, depth, 1)

    def forward(self, x):
        return self.Q1(x), self.Q2(x)

    def Q1_only(self, x):
        return self.Q1(x)


# ══════════════════════════════════════════════════════════════════════════════
#  MoEDDPG  – Phase-conditioned + PER + N-step + TD3 + Spatial bonus
# ══════════════════════════════════════════════════════════════════════════════

class MoEDDPG:
    """
    Phase-Conditioned Multi-Head DDPG with four improvements.

    Parameters
    ----------
    num_experts     : phase heads (default 3)
    sigma           : Gaussian phase width (default 0.35)
    n_step          : n-step return horizon (default 3)
    per_alpha       : PER priority exponent (default 0.6)
    per_beta_start  : PER IS-weight start (default 0.4)
    actor_delay     : TD3 delayed actor update frequency (default 2)
    spatial_grid    : patch grid size K for spatial bonus (default 4)
    spatial_alpha   : spatial bonus weight (default 0.3)
    batch_size / env_batch / max_step / tau / discount / rmsize
    writer / resume / output_path
    """

    def __init__(self, num_experts=3, sigma=0.35,
                 n_step=3, per_alpha=0.6, per_beta_start=0.4,
                 actor_delay=2,
                 spatial_grid=4, spatial_alpha=0.3,
                 batch_size=64, env_batch=1, max_step=40,
                 tau=0.001, discount=0.9, rmsize=800,
                 writer=None, resume=None, output_path=None,
                 gate_entropy_coef=0.0,
                 diversity_coef=0.0,
                 direct_q_coef=0.0):

        self.num_experts   = num_experts
        self.sigma         = sigma
        self.n_step        = n_step
        self.actor_delay   = actor_delay
        self.max_step      = max_step
        self.env_batch     = env_batch
        self.batch_size    = batch_size
        self.tau           = tau
        self.discount      = discount
        self.writer        = writer
        self.output_path   = output_path
        self.log           = 0
        self._critic_updates = 0

        # ── Actor ─────────────────────────────────────────────────────────────
        self.actor        = MultiHeadActor(9, 18, 65, num_experts, sigma)
        self.actor_target = MultiHeadActor(9, 18, 65, num_experts, sigma)

        # ── Double Critic (TD3, Fix 3) ────────────────────────────────────────
        self.critic        = DoubleCritic(3 + 9, 18)
        self.critic_target = DoubleCritic(3 + 9, 18)

        # ── Optimisers ────────────────────────────────────────────────────────
        self.actor_optim  = Adam(self.actor.parameters(),   lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(),  lr=1e-2)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        if resume is not None:
            self.load_weights(resume)

        # ── PER buffer (Fix 1) ────────────────────────────────────────────────
        self.memory = PrioritizedReplayBuffer(
            capacity   = rmsize * max_step,
            alpha      = per_alpha,
            beta_start = per_beta_start,
            beta_steps = 200_000,
        )

        # ── N-step buffer (Fix 2) ─────────────────────────────────────────────
        self.nstep_buf = NStepBuffer(n_step, discount, env_batch)
        # Effective discount for TD target (discounts n steps ahead)
        self.discount_n = discount ** n_step

        # ── Spatial progress bonus (Fix 4) ────────────────────────────────────
        self.spatial_bonus = SpatialProgressBonus(spatial_grid, spatial_alpha)

        # ── Episode bookkeeping ───────────────────────────────────────────────
        self.state       = [None] * env_batch
        self.action      = [None] * env_batch
        self.noise_level = np.zeros(env_batch)

        self._move_to_device()

    # ── Device management ─────────────────────────────────────────────────────

    def _move_to_device(self):
        for m in (self.actor, self.actor_target,
                  self.critic, self.critic_target):
            m.to(device)

    # ── Observation normalisation ─────────────────────────────────────────────

    def _to_dev(self, state):
        if not isinstance(state, torch.Tensor):
            return torch.tensor(state, dtype=torch.float32, device=device)
        return state.to(device=device, dtype=torch.float32)

    def _norm_obs(self, state):
        state  = self._to_dev(state)
        n      = state.shape[0]
        T_norm = state[:, 6:7, 0, 0] / self.max_step
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

    # ── Public API ────────────────────────────────────────────────────────────

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
        for i in range(self.env_batch):
            self.nstep_buf.reset(i)

    def observe(self, reward, state, done, step):
        """
        Push to the n-step buffer; emit completed n-step tuples to PER.
        Also adds spatial intrinsic bonus to the reward before storage.
        """
        s0   = np.array(self.state.cpu())   # (B, 7, 128, 128)
        a0   = np.array(self.action)  # (B, 65)
        done = np.array(done)         # (B,)

        # Spatial bonus: before() was called in observe_before()
        # The intrinsic bonus is already added to `reward` before we get here
        # if the training loop calls observe_spatial_bonus().
        # We just store raw `reward` as-is.

        nstep_tuples = self.nstep_buf.push(s0, a0, reward, state, done)
        for tup in nstep_tuples:
            self.memory.append(tup)

        self.state = state

        # Reset n-step buffer for done environments
        for i in range(self.env_batch):
            if done[i]:
                self.nstep_buf.reset(i)

    def observe_spatial_before(self, canvas, gt):
        """Call immediately before env.step to record patch errors."""
        self.spatial_bonus.before(canvas, gt)

    def observe_spatial_after(self, canvas_new, gt, ini_dis) -> np.ndarray:
        """Call immediately after env.step; returns intrinsic bonus (B,)."""
        return self.spatial_bonus.after(canvas_new, gt, ini_dis)

    # ── Q + GAN evaluation ────────────────────────────────────────────────────

    def _update_gan(self, state):
        canvas = state[:, :3].float() / 255.0
        gt     = state[:, 3:6].float() / 255.0
        fake, real, penal = update_discriminator(canvas, gt)
        if self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/gan_fake',  fake,  self.log)
            self.writer.add_scalar('train_moe/gan_real',  real,  self.log)
            self.writer.add_scalar('train_moe/gan_penal', penal, self.log)

    def _merged_state(self, state, action):
        T       = state[:, 6:7, 0, 0].float()
        gt      = state[:, 3:6].float() / 255.0
        canvas0 = state[:, :3].float() / 255.0
        canvas1 = decode(action, canvas0)
        n       = state.shape[0]
        coord_  = _coord.expand(n, 2, 128, 128)
        merged  = torch.cat([
            canvas0, canvas1, gt,
            (T + 1).view(n, 1, 1, 1).expand(n, 1, 128, 128) / self.max_step,
            coord_,
        ], dim=1)
        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        return merged, gan_reward

    def _evaluate(self, state, action, target=False):
        merged, gan_reward = self._merged_state(state, action)
        if target:
            Q1, Q2 = self.critic_target(merged)
            Q = torch.min(Q1, Q2)            # TD3: pessimistic min
        else:
            Q1, Q2 = self.critic(merged)
            Q = Q1                           # actor optimises Q1 only
            if self.log % 20 == 0 and self.writer:
                self.writer.add_scalar('train_moe/Q1', Q1.mean(), self.log)
                self.writer.add_scalar('train_moe/Q2', Q2.mean(), self.log)
                self.writer.add_scalar('train_moe/gan_reward',
                                       gan_reward.mean(), self.log)
        return Q + gan_reward, gan_reward

    # ── Policy update ─────────────────────────────────────────────────────────

    def update_policy(self, lr):
        """
        One gradient step.  Uses PER-sampled batch with IS weights,
        n-step TD targets, double-critic min, delayed actor update.
        """
        if self.memory.size() < self.batch_size:
            return torch.tensor(0.), torch.tensor(0.)

        self.log             += 1
        self._critic_updates += 1

        for pg in self.critic_optim.param_groups: pg['lr'] = lr[0]
        for pg in self.actor_optim.param_groups:  pg['lr'] = lr[1]

        # ── PER batch (Fix 1) ─────────────────────────────────────────────────
        state, action, reward, next_state, terminal, \
            is_weights, tree_indices = \
            self.memory.sample_batch(self.batch_size, device)

        self._update_gan(next_state)

        # ── N-step TD target with double critic min (Fix 2 + Fix 3) ──────────
        with torch.no_grad():
            next_action = self.play(next_state, target=True)
            target_q, _ = self._evaluate(next_state, next_action, target=True)
            # n-step discount already baked into stored rewards
            target_q = reward.view(-1, 1) + \
                       self.discount_n * \
                       ((1 - terminal.float()).view(-1, 1)) * target_q

        cur_q, step_reward = self._evaluate(state, action)
        target_q = target_q + step_reward.detach()

        # IS-weighted TD error (Fix 1)
        td_errors  = (cur_q - target_q).detach().squeeze(1)
        value_loss = (is_weights * (cur_q - target_q).squeeze(1) ** 2).mean()

        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        # Update PER priorities with current TD errors
        self.memory.update_priorities(
            tree_indices,
            td_errors.abs().cpu().numpy())

        # ── Delayed actor update (TD3, Fix 3) ─────────────────────────────────
        policy_loss = torch.tensor(0., device=device)
        if self._critic_updates % self.actor_delay == 0:
            action_pred = self.play(state)
            pre_q, _    = self._evaluate(state.detach(), action_pred)
            policy_loss = -pre_q.mean()
            self.actor.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.actor_optim.step()

        soft_update(self.actor_target,  self.actor,  self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        if self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/value_loss',  value_loss.item(),  self.log)
            self.writer.add_scalar('train_moe/policy_loss', policy_loss.item(), self.log)
            self.writer.add_scalar('train_moe/per_beta',
                                   self.memory.beta_start
                                   + (1 - self.memory.beta_start)
                                   * min(self.memory._step / self.memory.beta_steps, 1),
                                   self.log)

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
        self.actor.cpu();  self.critic.Q1.cpu();  self.critic.Q2.cpu()
        torch.save(self.actor.state_dict(),    f'{path}/moe_actor.pkl')
        torch.save(self.critic.Q1.state_dict(), f'{path}/moe_critic1.pkl')
        torch.save(self.critic.Q2.state_dict(), f'{path}/moe_critic2.pkl')
        save_gan(path)
        self._move_to_device()
        self.train()

    def load_weights(self, path):
        for name, attr in [('actor', self.actor),
                            ('critic Q1', self.critic.Q1),
                            ('critic Q2', self.critic.Q2)]:
            fname = {'actor': 'moe_actor.pkl',
                     'critic Q1': 'moe_critic1.pkl',
                     'critic Q2': 'moe_critic2.pkl'}[name]
            try:
                attr.load_state_dict(
                    torch.load(f'{path}/{fname}', map_location=device))
                print(f'[PhaseDDPG] {name} loaded.')
            except FileNotFoundError:
                print(f'[PhaseDDPG] {name}: no checkpoint, fresh start.')
        try:
            load_gan(path)
        except Exception:
            pass
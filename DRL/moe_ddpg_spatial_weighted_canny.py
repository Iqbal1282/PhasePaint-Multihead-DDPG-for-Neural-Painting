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


def decode(x, canvas, num_strokes: int = 5):
    """
    Decode `num_strokes` bezier strokes onto canvas.
    x      : (B, num_strokes * 13)  stroke parameters in [0,1]
    canvas : (B, 3, W, W)           current canvas in [0,1]
    """
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
    is replaced by N separate MLP heads (feat_dim → 256 → num_outputs each).

    Changes vs. original
    --------------------
    1. adaptive_avg_pool2d(..., (2, 2)) instead of avg_pool2d(..., 4):
       After 4 stride-2 layers the feature map is 4×4.  The original kernel-4
       pool collapsed it to 1×1, discarding all spatial structure.  A 2×2
       adaptive pool keeps 4 spatial cells, giving feat_dim = 512*4 = 2048
       and preserving coarse spatial information about where on the canvas
       features are strongest.

    2. Each head is now a 2-layer MLP with LayerNorm instead of a single
       Linear layer.  A bare Linear(2048, 65) cannot model any non-linear
       relationship between the rich backbone features and stroke parameters.
       The hidden layer + LayerNorm lets each head specialise non-linearly
       for its phase regime without exploding gradients.

    The step-progress T_norm is used to compute Gaussian phase weights
    over the heads.  The final action is the weighted sum:

        action = sum_k( w_k(T) * head_k(backbone_features) )

    All N heads receive gradient at every step, scaled by their phase weight.
    Early heads naturally specialise on coarse strokes; late heads on fine ones.
    """

    # Hidden dim for each head MLP
    HEAD_HIDDEN = 256

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

        # FIX 1: adaptive_avg_pool2d((2,2)) keeps a 2×2 spatial grid instead
        # of collapsing to 1×1.  feat_dim = 512 * 2 * 2 = 2048.
        feat_dim = 512 * block.expansion * 2 * 2   # 2048 for BasicBlock

        # FIX 2: Replace bare Linear with a 2-layer MLP + LayerNorm per head.
        # Structure: feat_dim → HEAD_HIDDEN (LN + ReLU) → num_outputs
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, self.HEAD_HIDDEN),
                nn.LayerNorm(self.HEAD_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Linear(self.HEAD_HIDDEN, num_outputs),
            )
            for _ in range(num_heads)
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
        feat = F.adaptive_avg_pool2d(feat, (2, 2))      # (B, 512, 2, 2)
        feat = feat.view(feat.size(0), -1)              # (B, 2048)

        # ── Per-head outputs ──────────────────────────────────────────────────
        head_out = torch.stack(
            [h(feat) for h in self.heads], dim=1)       # (B, N, 65)

        # ── Phase-weighted mixture ────────────────────────────────────────────
        w = phase_weights(T_norm, self.num_heads,
                          self.sigma)                    # (B, N)
        action = (w.unsqueeze(-1) * head_out).sum(dim=1)  # (B, 65)
        return torch.sigmoid(action)


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL CRITIC  (FIX: Explicit T_norm concatenation before final FC)
# ══════════════════════════════════════════════════════════════════════════════

class TemporalCritic(nn.Module):
    """
    Wrapper around ResNet_wobn that concatenates T_norm as an explicit
    feature before the final FC layer.

    Problem with original: T was passed as a constant spatial channel.
    After convolutions, this becomes a single number lost in 8192 features.
    The critic cannot learn that "step 20 + quality 0.8" is different from
    "step 5 + quality 0.8".

    Fix: Extract conv features, flatten, concat T_norm, then FC.
    This forces the critic to use T_norm as a real input feature.
    """

    def __init__(self, num_inputs: int, depth: int, num_outputs: int):
        super().__init__()
        # Base ResNet without final FC
        self.backbone = ResNet_wobn(num_inputs, depth, num_outputs)
        
        # Replace final FC with one that accepts feat_dim + 1 (for T_norm)
        feat_dim = 512  # ResNet-18 output dimension
        self.backbone.fc = nn.Identity()  # Remove original FC
        
        self.temporal_fc = nn.Linear(feat_dim + 1, num_outputs)

    def forward(self, x: torch.Tensor, T_norm: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : (B, num_inputs, 128, 128)  — merged state [canvas0, canvas1, gt, T, coord]
        T_norm : (B, 1)                      — normalised step progress [0,1]

        Returns
        -------
        Q      : (B, 1)                      — state-action value
        """
        # Extract spatial features through backbone conv layers
        feat = self.backbone.relu_1(self.backbone.conv1(x))
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)
        feat = F.avg_pool2d(feat, 4)
        feat = feat.view(feat.size(0), -1)  # (B, feat_dim)

        # If T_norm not provided, extract from x (backward compatibility)
        if T_norm is None:
            # x contains T as channel 9 (after canvas0, canvas1, gt)
            # T channel is constant, so sample from corner
            T_norm = x[:, 9:10, 0, 0]  # (B, 1)

        # Concatenate T_norm as explicit feature
        feat_temporal = torch.cat([feat, T_norm], dim=1)  # (B, feat_dim + 1)

        # Final Q-value
        Q = self.temporal_fc(feat_temporal)
        return Q


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

class PaintNoise:
    """Structured noise for painting with parameter-specific scales"""
    
    # Noise scales per parameter group
    SCALES = {
        'position': 0.05,   # x0,y0,x2,y2
        'control':  0.08,   # x1,y1  
        'width':    0.02,   # w0,w1,w2
        'color':    0.08,   # r,g,b
        'opacity':  0.03,   # alpha
    }
    
    # Parameter indices in 13-dim stroke
    INDICES = {
        'position': [0, 1, 4, 5],
        'control':  [2, 3],
        'width':    [6, 7, 8],
        'color':    [9, 10, 11],
        'opacity':  [12],
    }
    
    def __init__(self, num_strokes=5):
        self.num_strokes = num_strokes
        self.action_dim = num_strokes * 13
        self.ou = OUNoise(self.action_dim)
        
    def reset(self):
        self.ou.reset()
        
    def noise(self, action, phase=0.5, noise_factor=1.0):
        """
        phase: float or (B,) array — 0=early, 1=late
        """
        # Ensure phase_scale is (B, 1) so it broadcasts over action columns
        phase = np.asarray(phase).reshape(-1, 1)          # (B, 1)
        phase_scale = 1.0 - 0.8 * phase                   # (B, 1)

        # Generate structured noise — shape (B, action_dim)
        noise = np.zeros_like(action)
        for group, scale in self.SCALES.items():
            indices = self.INDICES[group]
            for stroke_idx in range(self.num_strokes):
                stroke_offset = stroke_idx * 13
                for idx in indices:
                    col = stroke_offset + idx
                    # std is (B, 1) broadcast → one sample per env, no size= needed
                    noise[:, col] = np.random.normal(
                        0, scale * phase_scale.squeeze(1) * noise_factor
                    )

        # Add temporal correlation via OU process — shape (action_dim,) → broadcasts over B
        ou_noise = self.ou.noise() * noise_factor * 0.3
        noise += ou_noise

        return np.clip(action + noise, 0, 1)
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
                 num_strokes=5,
                 batch_size=64, env_batch=1, max_step=40,
                 tau=0.001, discount=0.9, rmsize=800,
                 writer=None, resume=None, output_path=None,
                 # kept for API compatibility with train_moe.py argparse
                 gate_entropy_coef=0.0,
                 diversity_coef=0.0,
                 direct_q_coef=0.0):

        self.num_experts  = num_experts
        self.sigma        = sigma
        self.num_strokes  = num_strokes
        self.action_dim   = num_strokes * 13   # N strokes × 13 params
        self.max_step     = max_step
        self.env_batch    = env_batch
        self.batch_size   = batch_size
        self.tau          = tau
        self.discount     = discount
        self.writer       = writer
        self.output_path  = output_path
        self.log          = 0

        # ── Multi-head actor + target ─────────────────────────────────────────
        # action_dim = num_strokes * 13  (configurable)
        self.actor        = MultiHeadActor(9, 18, self.action_dim, num_experts, sigma)
        self.actor_target = MultiHeadActor(9, 18, self.action_dim, num_experts, sigma)

        # ── Temporal Critic + target (FIXED: explicit T_norm feature) ─────────
        self.critic        = TemporalCritic(3 + 9, 18, 1)
        self.critic_target = TemporalCritic(3 + 9, 18, 1)


        # ── Optimisers ────────────────────────────────────────────────────────
        self.actor_optim  = Adam(self.actor.parameters(),  lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-2)

        # ── Target network init ───────────────────────────────────────────────
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        if resume is not None:
            self.load_weights(resume)
            #self.noise_generator.reset()

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



    def noise_action(self, action, target_theta=None, noise_factor=1.0):
        """
        Vectorized exploration with Axial Symmetry (Nematic) and Saliency Gating.
        """
        B = action.shape[0]
        # 1. Standard Gaussian Jitter for non-geometric parameters (color, pressure, etc.)
        noise = np.random.normal(
            loc=0.0, 
            scale=self.noise_level[:, np.newaxis], 
            size=action.shape
        ).astype('float32')
        action = action + (noise * noise_factor)

        # 2. Geometric Guided Exploration
        if target_theta is not None:
            N = action.shape[1] // 13  # Number of strokes (from action shape)
            action_reshaped = action.reshape(B, N, 13)
            
            # Extract target orientation and magnitude (confidence)
            # Assuming target_theta is passed as a dict or tensor from env
            t_theta = to_numpy(target_theta['theta']) # [B]
            t_mag = to_numpy(target_theta['mag'])     # [B]
            
            for i in range(N):
                # Calculate current stroke angle: atan2(y2-y0, x2-x0)
                dx = action_reshaped[:, i, 4] - action_reshaped[:, i, 0]
                dy = action_reshaped[:, i, 5] - action_reshaped[:, i, 1]
                agent_theta = np.arctan2(dy, dx)

                # --- AXIAL SYMMETRY LOGIC ---
                # Check if the opposite direction (target + pi) is closer to the agent's current stroke
                diff = agent_theta - t_theta
                # Wrap to [-pi, pi]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                
                # If diff > 90 deg, flip the target to follow the agent's axial intent
                adjusted_target = np.where(np.abs(diff) > (np.pi / 2), 
                                           (t_theta + np.pi) % (2 * np.pi), 
                                           t_theta)
                # --- SALIENCY GATING ---
                # Concentration kappa depends on edge confidence (mag)
                # If mag is low, kappa is low -> wide/random exploration
                # If mag is high, kappa is high -> tight structural exploration
                #kappa = 10.0 * t_mag * (1.0 / (noise_factor + 1e-6))
                kappa = np.clip(5.0 * t_mag / (noise_factor + 0.05), 0.5, 15.0)
                
                # Sample from Von Mises
                geo_samples = np.random.vonmises(mu=adjusted_target, kappa=kappa)
                
                # Apply new endpoints (Length L preserved from original action intent)
                L = np.sqrt(dx**2 + dy**2)
                action_reshaped[:, i, 4] = action_reshaped[:, i, 0] + L * np.cos(geo_samples)
                action_reshaped[:, i, 5] = action_reshaped[:, i, 1] + L * np.sin(geo_samples)
            
            action = action_reshaped.reshape(B, -1)

        return np.clip(action, 0, 1)
    
    
    def select_action(self, state, noise_factor=0, target_theta=None, training=False):
        """Always explore during training, with structured noise"""
        self._set_mode(train=False)
        with torch.no_grad():
            action = to_numpy(self.play(state))
        
        if noise_factor > 0 and training:
            # Extract phase from state
            T_norm = (state[:, 6:7, 0, 0].float() / self.max_step).cpu().numpy()
            action = self.noise_generator.noise(action, phase=T_norm, noise_factor=noise_factor)
        
        self._set_mode(train=True)
        self.action = action
        return action

    def reset(self, obs, factor):
        self.state       = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)
        self.noise_generator.reset()

    def observe(self, reward, state, done, step):
        s0 = torch.tensor(self.state, device='cpu')
        a  = to_tensor(self.action,              'cpu')
        r  = to_tensor(reward,                   'cpu')
        s1 = torch.tensor(state,     device='cpu')
        d  = to_tensor(done.astype('float32'),   'cpu')
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state
    

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
        canvas1 = decode(action, canvas0, num_strokes=self.num_strokes)

        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        #gan_reward = torch.tanh(cal_reward(canvas1, gt) - cal_reward(canvas0, gt))

        n      = state.shape[0]
        coord_ = _coord.expand(n, 2, 128, 128)
        merged = torch.cat([
            canvas0, canvas1, gt,
            (T + 1).view(n, 1, 1, 1).expand(n, 1, 128, 128) / self.max_step,
            coord_,
        ], dim=1)

        # ── FIX: Pass T_norm explicitly to TemporalCritic ─────────────────────
        T_norm = T / self.max_step  # (B, 1) normalised progress
        critic_net = self.critic_target if target else self.critic
        Q = critic_net(merged, T_norm)

        if not target and self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/expect_reward', Q.mean(),         self.log)
            self.writer.add_scalar('train_moe/gan_reward',    gan_reward.mean(), self.log)

        return Q + gan_reward, Q,  gan_reward

    # ── Policy update ─────────────────────────────────────────────────────────

    def update_policy(self, lr, step = 0, train_times = 0):
        self.log += 1

        for pg in self.actor_optim.param_groups: pg['lr'] = lr[1]
        for pg in self.critic_optim.param_groups: pg['lr'] = lr[0]

        state, action, reward, next_state, terminal = \
            self.memory.sample_batch(self.batch_size, device)

        # ── GAN (discriminator head update, encoder frozen) ───────────────
        self._update_gan(next_state)

        # ── Critic: TD error backprop through encoder ─────────────────────
        # Use _evaluate to handle the 12-channel concatenation internally
        cur_q  = self._evaluate(state, action)[1]   # Q only, no GAN

        with torch.no_grad():
            next_action = self.play(next_state, target=True)
            next_q      = self._evaluate(next_state, next_action, target=True)[1]
            target_q = torch.clamp(
                self.discount * ((1 - terminal.float()).view(-1, 1)) * next_q
                + reward.view(-1, 1),
                -100, 100
            )

        value_loss = _criterion(cur_q, target_q)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        norm_img, T_norm = self._norm_obs(state)
        action_pred = self.actor(norm_img, T_norm)
        _, q_val, gan_reward = self._evaluate(state.detach(), action_pred)
        
        # Phase-based dynamic weighting
        progress = step / max(train_times, 1)
        
        if progress < 0.3:
            # Early: GAN dominates (learn to paint realistically)
            w_gan, w_q = 0.9, 0.1
        elif progress < 0.7:
            # Mid: balanced
            w_gan, w_q = 0.6, 0.4
        else:
            # Late: Q dominates (refine accuracy)
            w_gan, w_q = 0.5, 0.5
        
        self.actor_optim.zero_grad()
        loss = w_gan * (-gan_reward.mean()) + w_q * (-q_val.mean()*0.1)
        loss.backward()
        self.actor_optim.step()

        # ── Target Sync & Logging ─────────────────────────────────────────
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        if self.log % 20 == 0 and self.writer:
            self.writer.add_scalar('train_moe/policy_loss', loss.item(), self.log)
            self.writer.add_scalar('train_moe/value_loss',  value_loss.item(),  self.log)
            for k, head in enumerate(self.actor.heads):
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in head.parameters() if p.grad is not None
                ) ** 0.5
                self.writer.add_scalar(f'train_moe/head{k}_grad_norm', grad_norm, self.log)

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
        #path = 'model\MoEPaint-run49'
        try:
            print(f'[PhaseDDPG] Loading actor and critic from {path}...')
            self.actor.load_state_dict(
                torch.load(f'{path}/moe_actor.pkl', map_location=device), strict=False)
            print('[PhaseDDPG] Actor loaded.')
        except FileNotFoundError:
            print(f'[PhaseDDPG] No actor checkpoint at {path}, fresh start.')
        try:
            self.critic.load_state_dict(
                torch.load(f'{path}/moe_critic.pkl', map_location=device), strict=False)
            print('[PhaseDDPG] Critic loaded.')
        except FileNotFoundError:
            print(f'[PhaseDDPG] No critic checkpoint at {path}, fresh start.')
        try:
            load_gan(path)
        except Exception:
            pass
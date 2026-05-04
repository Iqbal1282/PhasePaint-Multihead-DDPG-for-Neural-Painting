"""
ablation_configs.py
===================
Single source of truth for all seven ablation variants used in the paper.

Each config dict controls five independent axes:

    num_experts     : 1 = single-head (baseline), 3 = phase-conditioned heads
    sigma           : Gaussian width of each phase head
    reward_mode     : 'gan_only'  -> WGAN diff only, env returns zero (baseline)
                      'dense'     -> step-wise WMSE + quality + terminal + alignment
    noise_mode      : 'isotropic' -> uniform-scale Gaussian per env (baseline)
                      'paint'     -> PaintNoise: per-group + OU + phase decay + VonMises
    temporal_critic : False       -> T encoded as diluted spatial channel (baseline)
                      True        -> T_norm concatenated after backbone pooling
    loss_mode       : 'fixed'     -> w_gan=w_q=0.5 throughout (baseline)
                      'adaptive'  -> phase-scheduled GAN->Q weighting

Imported by:
    run_ablation.py          (launches all training runs)
    eval_ablation.py         (evaluates all checkpoints)
    print_ablation_table.py  (generates LaTeX table)
"""

ABLATION_CONFIGS = {

    # ── A0 ── Exact Learning to Paint baseline ────────────────────────────────
    'A0_baseline': {
        'num_experts':      1,
        'sigma':            0.35,
        'reward_mode':      'gan_only',
        'noise_mode':       'isotropic',
        'temporal_critic':  False,
        'loss_mode':        'fixed',
    },

    # ── A1 ── Add dense reward only ───────────────────────────────────────────
    'A1_dense_reward': {
        'num_experts':      1,
        'sigma':            0.35,
        'reward_mode':      'dense',
        'noise_mode':       'isotropic',
        'temporal_critic':  False,
        'loss_mode':        'fixed',
    },

    # ── A2 ── Add PaintNoise only ─────────────────────────────────────────────
    'A2_paint_noise': {
        'num_experts':      1,
        'sigma':            0.35,
        'reward_mode':      'gan_only',
        'noise_mode':       'paint',
        'temporal_critic':  False,
        'loss_mode':        'fixed',
    },

    # ── A3 ── Add Temporal Critic only ────────────────────────────────────────
    'A3_temporal_critic': {
        'num_experts':      1,
        'sigma':            0.35,
        'reward_mode':      'gan_only',
        'noise_mode':       'isotropic',
        'temporal_critic':  True,
        'loss_mode':        'fixed',
    },

    # ── A4 ── Add Phase Heads (K=3) only ──────────────────────────────────────
    'A4_phase_heads': {
        'num_experts':      3,
        'sigma':            0.35,
        'reward_mode':      'gan_only',
        'noise_mode':       'isotropic',
        'temporal_critic':  False,
        'loss_mode':        'fixed',
    },

    # ── A5 ── Architectural improvements combined ──────────────────────────────
    # Phase heads + Temporal Critic + adaptive loss (no dense reward, no PaintNoise)
    'A5_arch_combined': {
        'num_experts':      3,
        'sigma':            0.35,
        'reward_mode':      'gan_only',
        'noise_mode':       'isotropic',
        'temporal_critic':  True,
        'loss_mode':        'adaptive',
    },

    # ── A6 ── Full PhasePaint (all five contributions) ────────────────────────
    'A6_phasepaint_full': {
        'num_experts':      3,
        'sigma':            0.35,
        'reward_mode':      'dense',
        'noise_mode':       'paint',
        'temporal_critic':  True,
        'loss_mode':        'adaptive',
    },
}

# Canonical run order
ABLATION_ORDER = [
    'A0_baseline',
    'A1_dense_reward',
    'A2_paint_noise',
    'A3_temporal_critic',
    'A4_phase_heads',
    'A5_arch_combined',
    'A6_phasepaint_full',
]

# Human-readable labels for the paper table (LaTeX strings)
ABLATION_LABELS = {
    'A0_baseline':        'Baseline (L2P exact)',
    'A1_dense_reward':    '+~Dense Reward',
    'A2_paint_noise':     '+~PaintNoise',
    'A3_temporal_critic': '+~Temporal Critic',
    'A4_phase_heads':     r'+~Phase Heads ($K{=}3$)',
    'A5_arch_combined':   r'+~Arch.\ Combined',
    'A6_phasepaint_full': r'\textbf{PhasePaint (full)}',
}

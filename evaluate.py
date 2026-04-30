#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluates a trained MoEDDPG actor on the full test set and reports
per-image and aggregate metrics suitable for a paper results section.

Metrics computed
----------------
  PSNR  (dB)         — higher is better  (Peak Signal-to-Noise Ratio)
  SSIM              — higher is better  (Structural Similarity, range [0,1])
  L2 / MSE          — lower is better   (mean squared pixel error, [0,1] range)
  MAE               — lower is better   (mean absolute pixel error, [0,1] range)
  Edge MSE          — lower is better   (Canny-weighted MSE, same as training signal)

Outputs
-------
  results/metrics_summary.csv        — per-image metrics
  results/metrics_aggregate.csv      — mean ± std for every metric
  results/metrics_aggregate.txt      — human-readable table for copy-paste into LaTeX
  results/qualitative/img_<id>.png   — side-by-side (target | canvas | error) per image
                                       (first --save_qual images only, default 20)

Usage
-----
  python evaluate.py --model ./model/MoEPaint-run1
  python evaluate.py --model ./model/MoEPaint-run1 --num_experts 3 --sigma 0.2
  python evaluate.py --model ./model/MoEPaint-run1 --batch_size 32 --max_step 40
  python evaluate.py --model ./model/MoEPaint-run1 --max_images 500 --save_qual 50
  python evaluate.py --model ./model/MoEPaint-run1 --compare_model ./model/Baseline-run1

Arguments
---------
  --model         : path to model folder containing moe_actor.pkl  (required)
  --compare_model : optional second model folder to compare side-by-side
  --renderer      : path to renderer.pkl              (default: ../renderer.pkl)
  --max_step      : painting steps per image          (default: 40)
  --num_experts   : number of phase heads             (default: 3)
  --sigma         : Gaussian phase width              (default: 0.2)
  --batch_size    : images processed in parallel      (default: 32)
  --max_images    : cap on test images evaluated      (default: all 2001)
  --save_qual     : number of qualitative images saved (default: 20)
  --output_dir    : output folder                     (default: results/)
  --no_qual       : skip qualitative image saving
  --seed          : random seed                       (default: 0)
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
from skimage.metrics import (peak_signal_noise_ratio as psnr_fn,
                              structural_similarity  as ssim_fn)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Renderer.model import FCN
from DRL.moe_ddpg_spatial_weighted_canny import MultiHeadActor
from DRL.actor import ResNet

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='MoEDDPG test-set evaluation')
parser.add_argument('--model',         default='model/MoEPaint-run68', type=str)
parser.add_argument('--compare_model', default=None,              type=str,
                    help='Optional second model for side-by-side comparison')
parser.add_argument('--renderer',      default='../renderer.pkl', type=str)
parser.add_argument('--max_step',      default=40,                type=int)
parser.add_argument('--num_strokes',   default=5,                 type=int,
                    help='Strokes per step N (must match training config)')
parser.add_argument('--num_experts',   default=3,                 type=int)
parser.add_argument('--sigma',         default=0.2,               type=float)
parser.add_argument('--batch_size',    default=32,                type=int)
parser.add_argument('--dataset',       default='celeba',          type=str,
                    choices=['celeba', 'mnist', 'imagenet', 'cub200', 'stanford_cars'],
                    help='Dataset to evaluate on (default: celeba)')
parser.add_argument('--data_root',     default='./data',          type=str,
                    help='Root directory for dataset files')
parser.add_argument('--max_images',    default=2001,              type=int,
                    help='Cap on total test images')
parser.add_argument('--save_qual',     default=20,                type=int,
                    help='Number of qualitative comparison images to save')
parser.add_argument('--output_dir',    default='results',         type=str)
parser.add_argument('--no_qual',       action='store_true')
parser.add_argument('--seed',          default=0,                 type=int)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device      : {device}')
print(f'Model       : {args.model}')
print(f'Max images  : {args.max_images}')
print(f'Batch size  : {args.batch_size}')
print(f'Max steps   : {args.max_step}')
WIDTH = 128

os.makedirs(args.output_dir, exist_ok=True)
if not args.no_qual:
    os.makedirs(os.path.join(args.output_dir, 'qualitative'), exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading helpers
# ══════════════════════════════════════════════════════════════════════════════

Decoder = FCN()
Decoder.load_state_dict(torch.load(args.renderer, map_location=device))
Decoder = Decoder.to(device).eval()


def load_actor(model_path, num_experts, sigma, num_strokes=5):
    action_dim = num_strokes * 13
    if "actor.pkl" in os.listdir(model_path):
        print('Loading single-head actor (legacy model)')
        actor = ResNet(9, 18, action_dim)
        pkl = os.path.join(model_path, 'actor.pkl')
        actor.load_state_dict(torch.load(pkl, map_location=device))
        return actor.to(device).eval()
    actor = MultiHeadActor(9, 18, action_dim, num_experts, sigma)
    pkl = os.path.join(model_path, 'moe_actor.pkl')
    actor.load_state_dict(torch.load(pkl, map_location=device))
    return actor.to(device).eval()


coord = torch.zeros(1, 2, WIDTH, WIDTH, device=device)
for i in range(WIDTH):
    for j in range(WIDTH):
        coord[0, 0, i, j] = i / (WIDTH - 1.)
        coord[0, 1, i, j] = j / (WIDTH - 1.)


def decode(action_batch, canvas, num_strokes=5):
    """Decode num_strokes strokes from (B, num_strokes*13) action onto (B,3,H,W) canvas."""
    x = action_batch.view(-1, 13)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, WIDTH, WIDTH, 1)
    colour = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    colour = colour.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, num_strokes, 1, WIDTH, WIDTH)
    colour = colour.view(-1, num_strokes, 3, WIDTH, WIDTH)
    for i in range(num_strokes):
        canvas = canvas * (1 - stroke[:, i]) + colour[:, i]
    return canvas


def run_episode(actor, target_t, max_step):
    """
    Run one full painting episode.

    target_t : (B, 3, H, W) float32 [0,1] on device
    Returns  : (B, 3, H, W) float32 [0,1] final canvas
    """
    B = target_t.shape[0]
    canvas = torch.zeros_like(target_t)
    coord_b = coord.expand(B, 2, WIDTH, WIDTH)

    with torch.no_grad():
        for step in range(max_step):
            T_frac  = torch.full((B, 1, WIDTH, WIDTH), step / max_step,
                                 device=device, dtype=torch.float32)
            T_norm  = torch.full((B, 1), step / max_step,
                                 device=device, dtype=torch.float32)
            obs     = torch.cat([canvas, target_t, T_frac, coord_b], dim=1)
            if "moe_actor.pkl" in os.listdir(args.model):
                action  = actor(obs, T_norm)
            else:
                action  = actor(obs)
            canvas  = decode(action, canvas, num_strokes=args.num_strokes)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  Metric functions  (all operate on uint8 numpy arrays, H×W×3)
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(pred_u8, gt_u8):
    """PSNR in dB. data_range=255 since inputs are uint8."""
    return float(psnr_fn(gt_u8, pred_u8, data_range=255))


def compute_ssim(pred_u8, gt_u8):
    """SSIM in [0,1]. channel_axis=2 for HxWxC images."""
    return float(ssim_fn(gt_u8, pred_u8, channel_axis=2, data_range=255))


def compute_l2(pred_f, gt_f):
    """MSE in [0,1] pixel range. Inputs float32 [0,1] (H,W,3)."""
    return float(np.mean((pred_f - gt_f) ** 2))


def compute_mae(pred_f, gt_f):
    """MAE in [0,1] pixel range."""
    return float(np.mean(np.abs(pred_f - gt_f)))


def compute_edge_mse(pred_u8, gt_u8, edge_lambda=2.0):
    """
    Canny-weighted MSE — same formulation used during training.
    Pixel errors on edge regions are weighted by (1 + edge_lambda).
    """
    gray = cv2.cvtColor(gt_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0   # [0,1]
    pixel_err = ((pred_u8.astype(np.float32) - gt_u8.astype(np.float32)) / 255.) ** 2
    weight = 1.0 + edge_lambda * edges[:, :, np.newaxis]
    return float(np.mean(pixel_err * weight))


def metrics_for_pair(pred_u8, gt_u8):
    """Compute all metrics for one image pair. Returns a dict."""
    pred_f = pred_u8.astype(np.float32) / 255.
    gt_f   = gt_u8.astype(np.float32)   / 255.
    return {
        'psnr':     compute_psnr(pred_u8, gt_u8),
        'ssim':     compute_ssim(pred_u8, gt_u8),
        'l2':       compute_l2(pred_f, gt_f),
        'mae':      compute_mae(pred_f, gt_f),
        'edge_mse': compute_edge_mse(pred_u8, gt_u8),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading  (mirrors env.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  Data loading  (unified — works for all datasets)
# ══════════════════════════════════════════════════════════════════════════════

from datasets import DatasetLoader

print(f'\nLoading {args.dataset} test images …')
t0 = time.time()

_dl = DatasetLoader(dataset=args.dataset, width=WIDTH, data_root=args.data_root)
_dl.load()

# Evaluate on test split (CHW → HWC for downstream cv2 operations)
test_num  = _dl.test_num
img_test  = [np.transpose(_dl.get_test(i), (1, 2, 0))   # HWC BGR
             for i in range(test_num)]

print(f'Test images : {test_num}  (loaded in {time.time()-t0:.1f}s)')
n_eval = min(args.max_images, test_num)
print(f'Evaluating  : {n_eval} images\n')



# ══════════════════════════════════════════════════════════════════════════════
#  Qualitative image helper
# ══════════════════════════════════════════════════════════════════════════════

DISP = 256   # display size for each panel in the qualitative grid

def save_qualitative(img_id, gt_u8, pred_u8, pred2_u8=None, prefix='img'):
    """
    Save a side-by-side comparison image.
    gt_u8, pred_u8: (128,128,3) uint8 RGB
    """
    def panel(arr, label):
        p = cv2.resize(arr, (DISP, DISP), interpolation=cv2.INTER_LINEAR)
        p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
        cv2.putText(p, label, (8, DISP - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return p

    # Error map: amplified absolute difference, INFERNO colourmap
    err = np.abs(pred_u8.astype(np.float32) - gt_u8.astype(np.float32)).mean(axis=2)
    err = np.clip(err * 4, 0, 255).astype(np.uint8)
    err_colour = cv2.resize(err, (DISP, DISP))
    err_colour = cv2.applyColorMap(err_colour, cv2.COLORMAP_INFERNO)

    panels = [panel(gt_u8, 'Target')]
    panels.append(panel(pred_u8, 'Ours'))
    if pred2_u8 is not None:
        panels.append(panel(pred2_u8, 'Baseline'))
    panels.append(err_colour)

    # Metric overlay on error panel
    m = metrics_for_pair(pred_u8, gt_u8)
    txt = f"PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.3f}"
    cv2.putText(panels[-1], txt, (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    sep = np.zeros((DISP, 2, 3), dtype=np.uint8)
    row = np.concatenate(
        [p for pair in zip(panels, [sep]*len(panels)) for p in pair][:-1],
        axis=1)

    path = os.path.join(args.output_dir, 'qualitative', f'{prefix}_{img_id:04d}.png')
    cv2.imwrite(path, row)


# ══════════════════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(actor, model_label, compare_actor=None):
    rows = []
    n_batches = (n_eval + args.batch_size - 1) // args.batch_size
    saved_qual = 0
    t_start = time.time()

    for batch_idx in range(n_batches):
        start_id = batch_idx * args.batch_size
        end_id   = min(start_id + args.batch_size, n_eval)
        ids      = list(range(start_id, end_id))
        B        = len(ids)

        # Load images for this batch
        batch_bgr = [img_test[i] for i in ids]     # list of (128,128,3) BGR uint8

        # Convert to RGB float tensor  (B, 3, 128, 128) [0,1]
        target_np = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                               for im in batch_bgr], axis=0)   # (B,128,128,3)
        target_t  = torch.tensor(
            target_np.transpose(0, 3, 1, 2), dtype=torch.float32, device=device
        ) / 255.

        # Run episode
        canvas_t = run_episode(actor, target_t, args.max_step)


        # Optional comparison model
        canvas2_t = None
        if compare_actor is not None:
            canvas2_t = run_episode(compare_actor, target_t, args.max_step)

        # Convert results to uint8 numpy  (B,128,128,3) RGB
        canvas_np = (canvas_t.detach().cpu().numpy().transpose(0,2,3,1)
                     * 255).clip(0,255).astype(np.uint8)
        canvas2_np = None
        if canvas2_t is not None:
            canvas2_np = (canvas2_t.detach().cpu().numpy().transpose(0,2,3,1)
                          * 255).clip(0,255).astype(np.uint8)

        # Per-image metrics
        for local_i, img_id in enumerate(ids):
            pred_u8 = canvas_np[local_i]
            gt_u8   = target_np[local_i]
            m       = metrics_for_pair(pred_u8, gt_u8)
            m['img_id'] = img_id
            rows.append(m)

            # Qualitative saves
            if (not args.no_qual) and saved_qual < args.save_qual:
                pred2_u8 = canvas2_np[local_i] if canvas2_np is not None else None
                save_qualitative(img_id, gt_u8, pred_u8, pred2_u8)
                saved_qual += 1

        elapsed = time.time() - t_start
        imgs_done = end_id
        eta = (elapsed / imgs_done) * (n_eval - imgs_done) if imgs_done > 0 else 0
        print(f'\r  [{imgs_done:4d}/{n_eval}]  '
              f'PSNR {np.mean([r["psnr"] for r in rows]):.3f}  '
              f'SSIM {np.mean([r["ssim"] for r in rows]):.4f}  '
              f'L2 {np.mean([r["l2"] for r in rows]):.5f}  '
              f'ETA {eta:.0f}s   ', end='', flush=True)

    print()
    return rows


# ── Load actors ───────────────────────────────────────────────────────────────
print(f'Loading actor from {args.model}')
actor = load_actor(args.model, args.num_experts, args.sigma, args.num_strokes)

compare_actor = None
if args.compare_model:
    print(f'Loading comparison actor from {args.compare_model}')
    compare_actor = load_actor(args.compare_model, args.num_experts, args.sigma, args.num_strokes)

# ── Run evaluation ────────────────────────────────────────────────────────────
print(f'\nEvaluating …')
rows = evaluate_model(actor, 'ours', compare_actor)


# ══════════════════════════════════════════════════════════════════════════════
#  Save results
# ══════════════════════════════════════════════════════════════════════════════

df = pd.DataFrame(rows)
metric_cols = ['psnr', 'ssim', 'l2', 'mae', 'edge_mse']

# Per-image CSV
per_image_path = os.path.join(args.output_dir, 'metrics_per_image.csv')
df.to_csv(per_image_path, index=False)
print(f'\nPer-image results  → {per_image_path}')

# Aggregate stats
agg = {}
for col in metric_cols:
    agg[col] = {
        'mean':   df[col].mean(),
        'std':    df[col].std(),
        'median': df[col].median(),
        'min':    df[col].min(),
        'max':    df[col].max(),
    }

agg_df = pd.DataFrame(agg).T
agg_csv = os.path.join(args.output_dir, 'metrics_aggregate.csv')
agg_df.to_csv(agg_csv)
print(f'Aggregate results  → {agg_csv}')

# Human-readable table for paper
lines = []
lines.append('=' * 64)
lines.append(f'Evaluation results  —  {n_eval} test images')
lines.append(f'Model : {args.model}')
lines.append('=' * 64)
lines.append(f'{"Metric":<14} {"Mean":>10} {"Std":>10} {"Median":>10}')
lines.append('-' * 64)

METRIC_LABELS = {
    'psnr':     'PSNR  (dB) ↑',
    'ssim':     'SSIM      ↑',
    'l2':       'L2 / MSE  ↓',
    'mae':      'MAE       ↓',
    'edge_mse': 'Edge MSE  ↓',
}
for col in metric_cols:
    label  = METRIC_LABELS[col]
    mean   = agg[col]['mean']
    std    = agg[col]['std']
    median = agg[col]['median']
    lines.append(f'{label:<14} {mean:>10.4f} {std:>10.4f} {median:>10.4f}')

lines.append('=' * 64)

# LaTeX table snippet
lines.append('')
lines.append('LaTeX table row (mean ± std):')
lines.append('\\hline')
latex_vals = ' & '.join(
    f'{agg[c]["mean"]:.4f} $\\pm$ {agg[c]["std"]:.4f}'
    for c in metric_cols
)
lines.append(f'Ours & {latex_vals} \\\\')
lines.append('\\hline')

txt_path = os.path.join(args.output_dir, 'metrics_summary.txt')
report   = '\n'.join(lines)
with open(txt_path, 'w') as f:
    f.write(report)

print(f'Summary table      → {txt_path}')
print()
print(report)

# ── Histogram plots (optional, requires matplotlib) ───────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 3))
    for ax, col in zip(axes, metric_cols):
        ax.hist(df[col], bins=40, color='steelblue', edgecolor='none', alpha=0.85)
        ax.axvline(df[col].mean(), color='tomato', linewidth=1.5,
                   label=f'mean={df[col].mean():.4f}')
        ax.set_title(METRIC_LABELS[col], fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle(f'Metric distributions — {n_eval} test images', fontsize=10)
    fig.tight_layout()
    hist_path = os.path.join(args.output_dir, 'metric_distributions.png')
    fig.savefig(hist_path, dpi=150)
    plt.close()
    print(f'Distributions plot → {hist_path}')
except ImportError:
    print('(matplotlib not found — skipping distribution plots)')

print('\nDone.')
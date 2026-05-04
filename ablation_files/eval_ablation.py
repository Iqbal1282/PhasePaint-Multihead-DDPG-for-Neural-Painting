#!/usr/bin/env python3
"""
eval_ablation.py
================
Evaluates all seven ablation checkpoints on the test set and produces:

  results/ablation/
    per_variant/
      <variant_name>/
        metrics_per_image.csv        — per-image PSNR/SSIM/L2/MAE/EdgeMSE
        metrics_aggregate.csv        — mean ± std
        qualitative/img_NNNN.png     — target | canvas | error panels
    summary_table.csv                — one row per variant, all metrics
    summary_table.tex                — LaTeX table ready to paste into paper

Usage
-----
  # Evaluate all variants
  python eval_ablation.py

  # Evaluate one variant only
  python eval_ablation.py --only A4_phase_heads

  # Override paths
  python eval_ablation.py \\
      --model_root  ./model/ablation \\
      --output_root ./results/ablation \\
      --dataset     celeba \\
      --data_root   ./data \\
      --max_step    40 \\
      --num_strokes 5 \\
      --max_images  2001

Assumptions
-----------
  Each variant checkpoint is at:
      <model_root>/<variant_name>/moe_actor.pkl
  Renderer:
      ../renderer.pkl
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
from skimage.metrics import (peak_signal_noise_ratio as psnr_fn,
                              structural_similarity   as ssim_fn)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Renderer.model import FCN
from DRL.moe_ddpg_spatial_weighted_canny import MultiHeadActor
from DRL.actor import ResNet
from datasets import DatasetLoader
from ablation_configs import ABLATION_CONFIGS, ABLATION_ORDER, ABLATION_LABELS

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate all ablation variants')
    p.add_argument('--model_root',   default='./model/ablation', type=str)
    p.add_argument('--output_root',  default='./results/ablation', type=str)
    p.add_argument('--renderer',     default='../renderer.pkl', type=str)
    p.add_argument('--dataset',      default='celeba',          type=str)
    p.add_argument('--data_root',    default='./data',          type=str)
    p.add_argument('--num_strokes',  default=5,                 type=int)
    p.add_argument('--max_step',     default=40,                type=int)
    p.add_argument('--batch_size',   default=32,                type=int)
    p.add_argument('--max_images',   default=2001,              type=int)
    p.add_argument('--save_qual',    default=20,                type=int,
                   help='Number of qualitative images to save per variant')
    p.add_argument('--no_qual',      action='store_true')
    p.add_argument('--seed',         default=0,                 type=int)
    p.add_argument('--only',         default=None,              type=str,
                   help='Evaluate one variant only')
    return p.parse_args()

args = parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WIDTH  = 128
DISP   = 256  # qualitative panel size

# ── Renderer ──────────────────────────────────────────────────────────────────

Decoder = FCN()
Decoder.load_state_dict(torch.load(args.renderer, map_location=device))
Decoder = Decoder.to(device).eval()

coord = torch.zeros(1, 2, WIDTH, WIDTH, device=device)
for i in range(WIDTH):
    for j in range(WIDTH):
        coord[0, 0, i, j] = i / (WIDTH - 1.)
        coord[0, 1, i, j] = j / (WIDTH - 1.)


def decode(action_batch, canvas, num_strokes=5):
    x      = action_batch.view(-1, 13)
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


# ── Actor loading ─────────────────────────────────────────────────────────────

def load_actor(model_path, cfg, num_strokes):
    """Load actor for a given ablation config from a checkpoint directory."""
    action_dim  = num_strokes * 13
    num_experts = cfg['num_experts']
    sigma       = cfg['sigma']

    # Single-head configs (baseline, A1-A3) may have been saved with either
    # moe_actor.pkl (MultiHeadActor K=1) or actor.pkl (legacy ResNet).
    files = os.listdir(model_path) if os.path.isdir(model_path) else []

    if 'actor.pkl' in files:
        actor = ResNet(9, 18, action_dim)
        actor.load_state_dict(
            torch.load(os.path.join(model_path, 'actor.pkl'), map_location=device))
    elif 'moe_actor.pkl' in files:
        actor = MultiHeadActor(9, 18, action_dim, num_experts, sigma)
        actor.load_state_dict(
            torch.load(os.path.join(model_path, 'moe_actor.pkl'), map_location=device))
    else:
        raise FileNotFoundError(
            f'No actor checkpoint found in {model_path}\n'
            f'Expected moe_actor.pkl or actor.pkl')

    return actor.to(device).eval()


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(actor, target_t, max_step, num_strokes):
    """
    Run one full painting episode without any environment overhead.

    target_t : (B, 3, H, W) float32 [0,1] on device
    Returns  : (B, 3, H, W) float32 [0,1] final canvas
    """
    B       = target_t.shape[0]
    canvas  = torch.zeros_like(target_t)
    coord_b = coord.expand(B, 2, WIDTH, WIDTH)

    with torch.no_grad():
        for step in range(max_step):
            T_frac = torch.full((B, 1, WIDTH, WIDTH), step / max_step,
                                device=device, dtype=torch.float32)
            T_norm = torch.full((B, 1), step / max_step,
                                device=device, dtype=torch.float32)
            obs    = torch.cat([canvas, target_t, T_frac, coord_b], dim=1)

            # MultiHeadActor requires T_norm; legacy ResNet does not
            if isinstance(actor, MultiHeadActor):
                action = actor(obs, T_norm)
            else:
                action = actor(obs)

            canvas = decode(action, canvas, num_strokes=num_strokes)

    return canvas


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred_u8, gt_u8):
    """
    pred_u8, gt_u8 : (H, W, 3) uint8 RGB
    Returns dict with psnr / ssim / l2 / mae / edge_mse
    """
    pred_f = pred_u8.astype(np.float32) / 255.
    gt_f   = gt_u8.astype(np.float32)   / 255.

    psnr = float(psnr_fn(gt_u8, pred_u8, data_range=255))
    ssim = float(ssim_fn(gt_u8, pred_u8, channel_axis=2, data_range=255))
    l2   = float(np.mean((pred_f - gt_f) ** 2))
    mae  = float(np.mean(np.abs(pred_f - gt_f)))

    gray       = cv2.cvtColor(gt_u8, cv2.COLOR_RGB2GRAY)
    edges      = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.
    pixel_err  = ((pred_u8.astype(np.float32)
                   - gt_u8.astype(np.float32)) / 255.) ** 2
    weight     = 1.0 + 2.0 * edges[:, :, np.newaxis]
    edge_mse   = float(np.mean(pixel_err * weight))

    return dict(psnr=psnr, ssim=ssim, l2=l2, mae=mae, edge_mse=edge_mse)


# ── Qualitative image saver ───────────────────────────────────────────────────

def save_qualitative(img_id, gt_u8, pred_u8, out_dir):
    """Save a side-by-side Target | Canvas | Error panel."""
    def panel(arr, label):
        p = cv2.resize(arr, (DISP, DISP), interpolation=cv2.INTER_LINEAR)
        p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
        cv2.putText(p, label, (8, DISP - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return p

    err = np.abs(pred_u8.astype(np.float32)
                 - gt_u8.astype(np.float32)).mean(axis=2)
    err = np.clip(err * 4, 0, 255).astype(np.uint8)
    err = cv2.resize(err, (DISP, DISP))
    err_colour = cv2.applyColorMap(err, cv2.COLORMAP_INFERNO)

    m   = compute_metrics(pred_u8, gt_u8)
    txt = f"PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.3f}"
    cv2.putText(err_colour, txt, (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    sep = np.zeros((DISP, 2, 3), dtype=np.uint8)
    row = np.concatenate(
        [panel(gt_u8, 'Target'), sep,
         panel(pred_u8, 'Canvas'), sep,
         err_colour], axis=1)

    path = os.path.join(out_dir, f'img_{img_id:04d}.png')
    cv2.imwrite(path, row)


# ── Data loading ──────────────────────────────────────────────────────────────

print(f'\nLoading {args.dataset} test images …')
_dl = DatasetLoader(dataset=args.dataset, width=WIDTH, data_root=args.data_root)
_dl.load()
test_num = _dl.test_num
img_test = [np.transpose(_dl.get_test(i), (1, 2, 0))   # HWC BGR uint8
            for i in range(test_num)]
n_eval   = min(args.max_images, test_num)
print(f'Test images: {test_num}   Evaluating: {n_eval}\n')


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_variant(variant_name, cfg):
    model_path = os.path.join(args.model_root, variant_name)
    out_dir    = os.path.join(args.output_root, 'per_variant', variant_name)
    qual_dir   = os.path.join(out_dir, 'qualitative')

    os.makedirs(out_dir,  exist_ok=True)
    if not args.no_qual:
        os.makedirs(qual_dir, exist_ok=True)

    if not os.path.isdir(model_path):
        print(f'  [SKIP] Checkpoint not found: {model_path}')
        return None

    print(f'  Loading actor … ({model_path})')
    try:
        actor = load_actor(model_path, cfg, args.num_strokes)
    except FileNotFoundError as e:
        print(f'  [SKIP] {e}')
        return None

    rows       = []
    saved_qual = 0
    n_batches  = (n_eval + args.batch_size - 1) // args.batch_size
    t0         = time.time()

    for batch_idx in range(n_batches):
        start_id = batch_idx * args.batch_size
        end_id   = min(start_id + args.batch_size, n_eval)
        ids      = list(range(start_id, end_id))
        B        = len(ids)

        batch_bgr = [img_test[i] for i in ids]
        target_np = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                               for im in batch_bgr])           # (B,H,W,3) RGB
        target_t  = torch.tensor(
            target_np.transpose(0, 3, 1, 2), dtype=torch.float32, device=device
        ) / 255.

        canvas_t = run_episode(actor, target_t, args.max_step, args.num_strokes)

        canvas_np = (canvas_t.detach().cpu().numpy().transpose(0, 2, 3, 1)
                     * 255).clip(0, 255).astype(np.uint8)

        for local_i, img_id in enumerate(ids):
            pred_u8 = canvas_np[local_i]
            gt_u8   = target_np[local_i]
            m       = compute_metrics(pred_u8, gt_u8)
            m['img_id'] = img_id
            rows.append(m)

            if not args.no_qual and saved_qual < args.save_qual:
                save_qualitative(img_id, gt_u8, pred_u8, qual_dir)
                saved_qual += 1

        elapsed   = time.time() - t0
        remaining = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
        print(f'  Batch {batch_idx+1}/{n_batches}  '
              f'({end_id}/{n_eval} images)  '
              f'ETA {remaining:.0f}s', end='\r')

    print()

    # ── Save per-image CSV ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'metrics_per_image.csv'), index=False)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    metrics_cols = ['psnr', 'ssim', 'l2', 'mae', 'edge_mse']
    agg = {}
    for col in metrics_cols:
        agg[f'{col}_mean'] = df[col].mean()
        agg[f'{col}_std']  = df[col].std()

    agg_df = pd.DataFrame([agg])
    agg_df.to_csv(os.path.join(out_dir, 'metrics_aggregate.csv'), index=False)

    # ── Human-readable summary ────────────────────────────────────────────────
    print(f'\n  ── {variant_name} ──')
    print(f"  PSNR  : {agg['psnr_mean']:.3f} ± {agg['psnr_std']:.3f}")
    print(f"  SSIM  : {agg['ssim_mean']:.4f} ± {agg['ssim_std']:.4f}")
    print(f"  L2    : {agg['l2_mean']:.5f} ± {agg['l2_std']:.5f}")
    print(f"  MAE   : {agg['mae_mean']:.5f} ± {agg['mae_std']:.5f}")
    print(f"  EMSE  : {agg['edge_mse_mean']:.5f} ± {agg['edge_mse_std']:.5f}")
    print(f"  Time  : {time.time()-t0:.1f}s\n")

    return agg


# ── Run all variants ──────────────────────────────────────────────────────────

os.makedirs(args.output_root, exist_ok=True)

order    = [args.only] if args.only else ABLATION_ORDER
all_rows = []

for variant_name in order:
    if variant_name not in ABLATION_CONFIGS:
        print(f'[WARNING] Unknown variant: {variant_name}. Skipping.')
        continue

    cfg = ABLATION_CONFIGS[variant_name]
    print(f'\n{"="*55}')
    print(f'  Evaluating: {variant_name}')
    print(f'  Config    : {cfg}')
    print(f'{"="*55}')

    result = evaluate_variant(variant_name, cfg)
    if result is not None:
        result['variant'] = variant_name
        result['label']   = ABLATION_LABELS.get(variant_name, variant_name)
        all_rows.append(result)

# ── Summary table ─────────────────────────────────────────────────────────────

if all_rows:
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(
        os.path.join(args.output_root, 'summary_table.csv'), index=False)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    tex_path = os.path.join(args.output_root, 'summary_table.tex')
    with open(tex_path, 'w') as f:
        f.write('%% Generated by eval_ablation.py\n')
        f.write('%% Paste into the ablation table in main.tex\n\n')
        f.write('\\begin{tabular}{@{}l c c c c@{}}\n')
        f.write('\\toprule\n')
        f.write('\\textbf{Variant} & \\textbf{PSNR $\\uparrow$} '
                '& \\textbf{SSIM $\\uparrow$} '
                '& \\textbf{L2 $\\downarrow$} '
                '& \\textbf{EMSE $\\downarrow$} \\\\\n')
        f.write('\\midrule\n')

        for row in all_rows:
            label    = row['label']
            psnr     = f"{row['psnr_mean']:.2f}"
            ssim     = f"{row['ssim_mean']:.3f}"
            l2       = f"{row['l2_mean']:.4f}"
            edge_mse = f"{row['edge_mse_mean']:.4f}"

            # Bold the full model row
            is_full = 'full' in row['variant']
            if is_full:
                psnr     = f'\\textbf{{{psnr}}}'
                ssim     = f'\\textbf{{{ssim}}}'
                l2       = f'\\textbf{{{l2}}}'
                edge_mse = f'\\textbf{{{edge_mse}}}'
                f.write('\\midrule\n')

            f.write(f'{label} & {psnr} & {ssim} & {l2} & {edge_mse} \\\\\n')

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    print(f'\nSummary table written to:')
    print(f'  {os.path.join(args.output_root, "summary_table.csv")}')
    print(f'  {tex_path}')

    # ── Print to console ──────────────────────────────────────────────────────
    print('\n\n' + '='*65)
    print(f'{"Variant":<30}  {"PSNR":>6}  {"SSIM":>6}  {"L2":>7}  {"EMSE":>7}')
    print('='*65)
    for row in all_rows:
        print(f"{row['label']:<30}  "
              f"{row['psnr_mean']:>6.2f}  "
              f"{row['ssim_mean']:>6.3f}  "
              f"{row['l2_mean']:>7.4f}  "
              f"{row['edge_mse_mean']:>7.4f}")
    print('='*65)

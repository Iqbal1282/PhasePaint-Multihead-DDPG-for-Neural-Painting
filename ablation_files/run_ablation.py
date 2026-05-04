#!/usr/bin/env python3
"""
run_ablation.py
===============
Launches all seven ablation training runs sequentially (or prints SLURM
job scripts for parallel cluster submission).

Usage
-----
  # Sequential (single machine)
  python run_ablation.py

  # Dry-run: print commands without executing
  python run_ablation.py --dry_run

  # Run one specific variant only
  python run_ablation.py --only A4_phase_heads

  # Override dataset / steps / output dir
  python run_ablation.py --dataset celeba --num_strokes 5 --max_step 40 \\
      --train_times 2000000 --output_root ./model/ablation

  # Print SLURM job scripts instead of running
  python run_ablation.py --slurm --slurm_partition gpu --slurm_gpus 1

Outputs
-------
  ./model/ablation/<variant_name>/moe_actor.pkl
  ./model/ablation/<variant_name>/moe_critic.pkl
  ../train_log/ablation_<variant_name>/   (TensorBoard)
"""

import os
import sys
import argparse
import subprocess

from ablation_configs import ABLATION_CONFIGS, ABLATION_ORDER

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='PhasePaint ablation training runner')
    p.add_argument('--dataset',      default='celeba',          type=str)
    p.add_argument('--data_root',    default='./data',          type=str)
    p.add_argument('--num_strokes',  default=5,                 type=int)
    p.add_argument('--max_step',     default=40,                type=int)
    p.add_argument('--train_times',  default=2000000,           type=int)
    p.add_argument('--env_batch',    default=96,                type=int)
    p.add_argument('--batch_size',   default=96,                type=int)
    p.add_argument('--warmup',       default=400,               type=int)
    p.add_argument('--seed',         default=42,                type=int,
                   help='Fixed seed ensures fair comparison across all variants')
    p.add_argument('--output_root',  default='./model/ablation',type=str)
    p.add_argument('--only',         default=None,              type=str,
                   help='Run only this variant (e.g. A4_phase_heads)')
    p.add_argument('--dry_run',      action='store_true',
                   help='Print commands without executing')
    p.add_argument('--slurm',        action='store_true',
                   help='Print SLURM job scripts instead of running')
    p.add_argument('--slurm_partition', default='gpu',         type=str)
    p.add_argument('--slurm_gpus',   default=1,                type=int)
    return p.parse_args()


def build_cmd(variant_name, cfg, args):
    """Build the python command for one ablation variant."""
    out_dir = os.path.join(args.output_root, variant_name)
    return [
        sys.executable, 'train_moe_spatial_weighted_canny.py',
        '--dataset',         args.dataset,
        '--data_root',       args.data_root,
        '--num_strokes',     str(args.num_strokes),
        '--max_step',        str(args.max_step),
        '--train_times',     str(args.train_times),
        '--env_batch',       str(args.env_batch),
        '--batch_size',      str(args.batch_size),
        '--warmup',          str(args.warmup),
        '--seed',            str(args.seed),
        '--output',          out_dir,
        # ── ablation-specific flags ──
        '--num_experts',     str(cfg['num_experts']),
        '--sigma',           str(cfg['sigma']),
        '--reward_mode',     cfg['reward_mode'],
        '--noise_mode',      cfg['noise_mode'],
        '--temporal_critic', str(int(cfg['temporal_critic'])),
        '--loss_mode',       cfg['loss_mode'],
        '--ablation_name',   variant_name,
    ]


def slurm_script(variant_name, cmd, args):
    cmd_str = ' '.join(cmd)
    return f"""#!/bin/bash
#SBATCH --job-name=abl_{variant_name}
#SBATCH --partition={args.slurm_partition}
#SBATCH --gres=gpu:{args.slurm_gpus}
#SBATCH --output=logs/ablation_{variant_name}_%j.out
#SBATCH --error=logs/ablation_{variant_name}_%j.err

source activate phasepaint
cd {os.path.abspath('.')}
{cmd_str}
"""


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    if args.slurm:
        os.makedirs('logs', exist_ok=True)

    order = [args.only] if args.only else ABLATION_ORDER

    for variant_name in order:
        if variant_name not in ABLATION_CONFIGS:
            print(f'[WARNING] Unknown variant: {variant_name}. Skipping.')
            continue

        cfg = ABLATION_CONFIGS[variant_name]
        cmd = build_cmd(variant_name, cfg, args)

        if args.slurm:
            script = slurm_script(variant_name, cmd, args)
            script_path = f'slurm_{variant_name}.sh'
            with open(script_path, 'w') as f:
                f.write(script)
            print(f'Written: {script_path}')
            print(f'  Submit with: sbatch {script_path}')

        elif args.dry_run:
            print(f'\n=== DRY RUN: {variant_name} ===')
            print(' '.join(cmd))

        else:
            out_dir = os.path.join(args.output_root, variant_name)
            os.makedirs(out_dir, exist_ok=True)
            print(f'\n{"="*60}')
            print(f'  Running ablation: {variant_name}')
            print(f'  Config: {cfg}')
            print(f'  Output: {out_dir}')
            print(f'{"="*60}')
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f'[ERROR] {variant_name} exited with code {result.returncode}')
                print('  Continuing with next variant...')

    print('\nAll ablation runs complete.')


if __name__ == '__main__':
    main()

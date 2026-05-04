#!/usr/bin/env python3
"""
train_moe_spatial_weighted_canny.py
=====================================
Phase-Conditioned Multi-Head DDPG training script.

Supports both full training and ablation study via five flags:
    --num_experts      1 (baseline) | K>1 (phase-conditioned heads)
    --reward_mode      'gan_only' (baseline) | 'dense'
    --noise_mode       'isotropic' (baseline) | 'paint'
    --temporal_critic  0 (baseline) | 1
    --loss_mode        'fixed' (baseline) | 'adaptive'

These flags are set automatically by run_ablation.py.  For a normal
full-model training run, the defaults reproduce PhasePaint (full).

Usage examples
--------------
  # Full PhasePaint (default)
  python train_moe_spatial_weighted_canny.py --dataset celeba --debug

  # Exact baseline reproduction (A0)
  python train_moe_spatial_weighted_canny.py \\
      --num_experts 1 --reward_mode gan_only \\
      --noise_mode isotropic --temporal_critic 0 --loss_mode fixed

  # CUB-200, 8 strokes, 60 steps
  python train_moe_spatial_weighted_canny.py \\
      --dataset cub200 --num_strokes 8 --max_step 60 --num_experts 4

  # Resume from checkpoint
  python train_moe_spatial_weighted_canny.py \\
      --resume ./model/MoEPaint-run1 --dataset celeba
"""

import os
import random
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

from utils.util import prRed, prBlack, get_output_folder
from utils.tensorboard import TensorBoard
from DRL.evaluator import Evaluator

current_path = os.path.abspath('.')
exp          = os.path.basename(current_path)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='PhasePaint training (ablation-aware)')

    # ── Dataset / environment ─────────────────────────────────────────────────
    parser.add_argument('--dataset',      default='celeba',  type=str,
        choices=['celeba', 'mnist', 'imagenet', 'cub200', 'stanford_cars'])
    parser.add_argument('--data_root',    default='./data',  type=str)
    parser.add_argument('--num_strokes',  default=5,         type=int,
        help='Strokes per step N')
    parser.add_argument('--max_step',     default=40,        type=int,
        help='Max painting steps per episode M')

    # ── Ablation flags ────────────────────────────────────────────────────────
    parser.add_argument('--num_experts',      default=3,          type=int,
        help='Phase heads K (1 = baseline single-head)')
    parser.add_argument('--sigma',            default=0.35,       type=float,
        help='Gaussian width of phase responsibility curves')
    parser.add_argument('--reward_mode',      default='dense',    type=str,
        choices=['gan_only', 'dense'],
        help="'gan_only' = baseline WGAN diff; 'dense' = multi-component reward")
    parser.add_argument('--noise_mode',       default='paint',    type=str,
        choices=['isotropic', 'paint'],
        help="'isotropic' = baseline; 'paint' = PaintNoise")
    parser.add_argument('--temporal_critic',  default=1,          type=int,
        choices=[0, 1],
        help='0 = baseline spatial-channel critic; 1 = TemporalCritic')
    parser.add_argument('--loss_mode',        default='adaptive', type=str,
        choices=['fixed', 'adaptive'],
        help="'fixed' = baseline equal weights; 'adaptive' = phase-scheduled")
    parser.add_argument('--ablation_name',    default=None,       type=str,
        help='Ablation variant name (used for log directory naming)')

    # ── Training hyper-params ─────────────────────────────────────────────────
    parser.add_argument('--warmup',              default=400,     type=int)
    parser.add_argument('--discount',            default=0.995,   type=float)
    parser.add_argument('--batch_size',          default=96,      type=int)
    parser.add_argument('--rmsize',              default=800,     type=int)
    parser.add_argument('--env_batch',           default=96,      type=int)
    parser.add_argument('--tau',                 default=0.001,   type=float)
    parser.add_argument('--noise_factor',        default=0.05,    type=float)
    parser.add_argument('--validate_interval',   default=50,      type=int)
    parser.add_argument('--validate_episodes',   default=5,       type=int)
    parser.add_argument('--train_times',         default=2000000, type=int)
    parser.add_argument('--episode_train_times', default=10,      type=int)
    parser.add_argument('--resume',              default=None,    type=str)
    parser.add_argument('--output',              default='./model', type=str)
    parser.add_argument('--debug',               action='store_true')
    parser.add_argument('--seed',                default=1234,    type=int)

    # ── Kept for API compatibility ─────────────────────────────────────────────
    parser.add_argument('--spatial_grid',   default=4,   type=int)
    parser.add_argument('--spatial_alpha',  default=0.3, type=float)
    parser.add_argument('--gate_entropy',   default=0.0, type=float)
    parser.add_argument('--diversity_coef', default=0.0, type=float)
    parser.add_argument('--direct_q_coef',  default=0.0, type=float)

    return parser.parse_args()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(agent, env, evaluate, args, writer):
    train_times         = args.train_times
    validate_interval   = args.validate_interval
    max_step            = args.max_step
    debug               = args.debug
    episode_train_times = args.episode_train_times
    output              = args.output
    noise_factor        = args.noise_factor

    time_stamp    = time.time()
    step          = 0
    episode       = 0
    episode_steps = 0
    observation   = None
    target_theta  = None

    pbar = tqdm(total=train_times, initial=step, desc='Training')

    while step <= train_times:
        step          += 1
        episode_steps += 1
        pbar.update(1)

        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)

        action = agent.select_action(
            observation,
            noise_factor=noise_factor,
            target_theta=target_theta,
            training=True,
        )

        observation, reward, done, info = env.step(action)
        if info is not None:
            target_theta = info.get('target_theta')

        agent.observe(reward, observation, done, step)

        if step % 100 == 0 and writer:
            writer.add_scalar('Reward/Extrinsic',  info['extrinsic'],  step)
            writer.add_scalar('Reward/Alignment',  info['alignment'],  step)
            writer.add_scalar('Reward/Total',      info['total'],      step)
            writer.add_scalar('Reward/Quality',    info['quality'],    step)

        if episode_steps >= max_step and max_step:
            if step > args.warmup:
                if episode > 0 and validate_interval > 0 \
                        and episode % validate_interval == 0:
                    reward_eval, dist_eval = evaluate(
                        env, agent.select_action, debug=debug)
                    if debug:
                        prRed('Step_{:07d}: mean_reward:{:.3f}  '
                              'mean_dist:{:.3f}  var_dist:{:.3f}'.format(
                                  step - 1,
                                  np.mean(reward_eval),
                                  np.mean(dist_eval),
                                  np.var(dist_eval)))
                    if writer:
                        writer.add_scalar('validate/mean_reward',
                                          np.mean(reward_eval), step)
                        writer.add_scalar('validate/mean_dist',
                                          np.mean(dist_eval),   step)
                    agent.save_model(output)

            if   step < 10000 * max_step: lr = (3e-4, 1e-3)
            elif step < 20000 * max_step: lr = (1e-4, 3e-4)
            else:                         lr = (3e-5, 1e-4)

            train_time_interval = time.time() - time_stamp
            time_stamp          = time.time()
            tot_Q               = 0.
            tot_value_loss      = 0.

            if step > args.warmup:
                for _ in range(episode_train_times):
                    Q, value_loss   = agent.update_policy(lr, step, train_times)
                    tot_Q          += Q.detach().cpu().item()
                    tot_value_loss += value_loss.detach().cpu().item()

                if writer:
                    writer.add_scalar('train/critic_lr', lr[0], step)
                    writer.add_scalar('train/actor_lr',  lr[1], step)
                    writer.add_scalar('train/Q',
                                      tot_Q / episode_train_times, step)
                    writer.add_scalar('train/critic_loss',
                                      tot_value_loss / episode_train_times, step)

            if debug:
                prBlack('#{}: steps:{} interval:{:.2f}s  train:{:.2f}s'.format(
                    episode, step, train_time_interval,
                    time.time() - time_stamp))

            time_stamp    = time.time()
            observation   = None
            episode_steps = 0
            episode      += 1


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    # ── Log directory: encodes all ablation axes for easy TensorBoard comparison
    ablation_tag = args.ablation_name or (
        f'{args.dataset}_n{args.num_strokes}_m{args.max_step}'
        f'_K{args.num_experts}'
        f'_rwd-{args.reward_mode}'
        f'_noise-{args.noise_mode}'
        f'_tc{args.temporal_critic}'
        f'_loss-{args.loss_mode}'
    )
    log_dir = os.path.join('..', 'train_log', ablation_tag)
    os.makedirs(log_dir, exist_ok=True)
    writer = TensorBoard(log_dir)

    os.makedirs('./model', exist_ok=True)
    try:
        if os.name != 'nt':
            os.system(f'ln -sf ../train_log/{ablation_tag} {ablation_tag}')
    except Exception:
        pass

    args.output = get_output_folder(args.output, 'MoEPaint')

    # ── Seeds ─────────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

    # ── Environment ───────────────────────────────────────────────────────────
    from DRL.multi import fastenv
    fenv = fastenv(
        max_episode_length = args.max_step,
        env_batch          = args.env_batch,
        num_strokes        = args.num_strokes,
        dataset            = args.dataset,
        data_root          = args.data_root,
        reward_mode        = args.reward_mode,
        writer             = writer,
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    from DRL.moe_ddpg_spatial_weighted_canny import MoEDDPG
    agent = MoEDDPG(
        num_experts      = args.num_experts,
        sigma            = args.sigma,
        num_strokes      = args.num_strokes,
        noise_mode       = args.noise_mode,
        temporal_critic  = bool(args.temporal_critic),
        loss_mode        = args.loss_mode,
        batch_size       = args.batch_size,
        env_batch        = args.env_batch,
        max_step         = args.max_step,
        tau              = args.tau,
        discount         = args.discount,
        rmsize           = args.rmsize,
        writer           = writer,
        resume           = args.resume,
        output_path      = args.output,
    )

    evaluate = Evaluator(args, writer)

    print(f'\n{"="*55}')
    print(f'  Dataset        : {args.dataset}')
    print(f'  N strokes/step : {args.num_strokes}')
    print(f'  M max steps    : {args.max_step}')
    print(f'  Phase heads K  : {args.num_experts}  sigma={args.sigma}')
    print(f'  Reward mode    : {args.reward_mode}')
    print(f'  Noise mode     : {args.noise_mode}')
    print(f'  Temporal critic: {bool(args.temporal_critic)}')
    print(f'  Loss mode      : {args.loss_mode}')
    print(f'  Action space   : {fenv.action_space}  ({args.num_strokes}x13)')
    print(f'{"="*55}\n')

    train(agent, fenv, evaluate, args, writer)

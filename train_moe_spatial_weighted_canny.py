#!/usr/bin/env python3
"""
train_moe_spatial_weighted_canny.py  (robust multi-dataset version)
=====================================================================
Phase-Conditioned Multi-Head DDPG painting agent.

New flags vs. original
-----------------------
  --dataset        : celeba | mnist | imagenet | cub200 | stanford_cars
  --data_root      : root directory for dataset files  (default ./data)
  --num_strokes    : N strokes drawn per step          (default 5)
  --max_step       : M maximum steps per episode       (default 40)

All other flags are identical to the original train_moe.py.

Usage examples
--------------
  # Original CelebA, 5 strokes, 40 steps (backward compatible)
  python train_moe_spatial_weighted_canny.py --dataset celeba --debug

  # MNIST with 3 strokes and 20 steps
  python train_moe_spatial_weighted_canny.py \\
      --dataset mnist --num_strokes 3 --max_step 20 --debug

  # CUB-200 birds with 8 strokes and 60 steps
  python train_moe_spatial_weighted_canny.py \\
      --dataset cub200 --data_root ./data \\
      --num_strokes 8 --max_step 60 --num_experts 4

  # Stanford Cars 196
  python train_moe_spatial_weighted_canny.py \\
      --dataset stanford_cars --num_strokes 5 --max_step 40

  # ImageNet (large — specify max_train to cap memory usage)
  python train_moe_spatial_weighted_canny.py \\
      --dataset imagenet --num_strokes 5 --max_step 40

  # Resume from checkpoint
  python train_moe_spatial_weighted_canny.py \\
      --dataset celeba --num_strokes 5 --max_step 40 \\
      --resume ./model/MoEPaint-run1
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
        description='Phase-Conditioned Multi-Head Painting Agent (multi-dataset)')

    # ── Dataset / environment ─────────────────────────────────────────────────
    parser.add_argument('--dataset',      default='celeba',  type=str,
        choices=['celeba', 'mnist', 'imagenet', 'cub200', 'stanford_cars'],
        help='Training dataset (default: celeba)')
    parser.add_argument('--data_root',    default='./data',  type=str,
        help='Root directory for dataset files (default: ./data)')
    parser.add_argument('--num_strokes',  default=5,         type=int,
        help='Number of strokes drawn per step N (default: 5)')
    parser.add_argument('--max_step',     default=40,        type=int,
        help='Maximum painting steps per episode M (default: 40)')

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

    # ── Phase-head flags ──────────────────────────────────────────────────────
    parser.add_argument('--num_experts', default=3,   type=int,
        help='Number of phase heads (default 3)')
    parser.add_argument('--sigma',       default=0.2, type=float,
        help='Gaussian width of each head (default 0.2)')

    # ── Spatial reward ────────────────────────────────────────────────────────
    parser.add_argument('--spatial_grid',  default=4,   type=int)
    parser.add_argument('--spatial_alpha', default=0.3, type=float)

    # ── API compat stubs (ignored) ────────────────────────────────────────────
    parser.add_argument('--gate_entropy',   default=0.0, type=float)
    parser.add_argument('--diversity_coef', default=0.0, type=float)
    parser.add_argument('--direct_q_coef',  default=0.0, type=float)

    return parser.parse_args()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(agent, env, evaluate, args, writer):
    train_times         = args.train_times
    env_batch           = args.env_batch
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

    pbar = tqdm(total=train_times, initial=step, desc="Training")

    while step <= train_times:
        step          += 1
        episode_steps += 1
        pbar.update(1)

        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)

        epsilon = max(0.001, 0.05 * (1 - step / train_times))
        action  = agent.select_action(observation,
                                      noise_factor=noise_factor,
                                      target_theta=target_theta,
                                      training=True)

        observation, reward, done, info = env.step(action)
        if info is not None:
            target_theta = info.get('target_theta')

        agent.observe(reward, observation, done, step)

        if step % 100 == 0:
            writer.add_scalar('Reward/Extrinsic_MSE',         info['extrinsic'],   step)
            writer.add_scalar('Reward/Intrinsic_Curiosity',   info['intrinsic'],   step)
            writer.add_scalar('Reward/Structural_Alignment',  info['alignment'],   step)
            writer.add_scalar('Reward/Total_Combined',        info['total'],       step)

        if episode_steps >= max_step and max_step:
            if step > args.warmup:
                if episode > 0 and validate_interval > 0 \
                        and episode % validate_interval == 0:
                    reward_eval, dist_eval = evaluate(
                        env, agent.select_action, debug=debug)
                    if debug:
                        prRed('Step_{:07d}: mean_reward:{:.3f}  '
                              'mean_dist:{:.3f}  var_dist:{:.3f}'.format(
                                  step - 1, np.mean(reward_eval),
                                  np.mean(dist_eval), np.var(dist_eval)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward_eval), step)
                    writer.add_scalar('validate/mean_dist',   np.mean(dist_eval),   step)
                    writer.add_scalar('validate/var_dist',    np.var(dist_eval),    step)
                    agent.save_model(output)

            if step < 10000 * max_step:
                lr = (3e-4, 1e-3)
            elif step < 20000 * max_step:
                lr = (1e-4, 3e-4)
            else:
                lr = (3e-5, 1e-4)

            train_time_interval = time.time() - time_stamp
            time_stamp          = time.time()
            tot_Q          = 0.
            tot_value_loss = 0.

            if step > args.warmup:
                for _ in range(episode_train_times):
                    Q, value_loss   = agent.update_policy(lr, step, train_times)
                    tot_Q          += Q.detach().cpu().item()
                    tot_value_loss += value_loss.detach().cpu().item()

                writer.add_scalar('train/critic_lr',   lr[0], step)
                writer.add_scalar('train/actor_lr',    lr[1], step)
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

    # ── Log directory — encodes dataset + N + M for easy identification ───────
    run_tag = (f'{args.dataset}_n{args.num_strokes}_m{args.max_step}_'
               f'experts{args.num_experts}_{exp}')
    log_dir = os.path.join('..', 'train_log', run_tag)
    os.makedirs(log_dir, exist_ok=True)
    writer = TensorBoard(log_dir)

    os.makedirs('./model', exist_ok=True)
    try:
        if os.name != 'nt':
            os.system(f'ln -sf ../train_log/{run_tag}')
    except Exception as e:
        print(f'Note: could not create symlink: {e}')

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
        writer             = writer,
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    from DRL.moe_ddpg_spatial_weighted_canny import MoEDDPG
    agent = MoEDDPG(
        num_experts  = args.num_experts,
        sigma        = args.sigma,
        num_strokes  = args.num_strokes,
        batch_size   = args.batch_size,
        env_batch    = args.env_batch,
        max_step     = args.max_step,
        tau          = args.tau,
        discount     = args.discount,
        rmsize       = args.rmsize,
        writer       = writer,
        resume       = args.resume,
        output_path  = args.output,
    )

    evaluate = Evaluator(args, writer)

    print(f'\nobservation_space : {fenv.observation_space}')
    print(f'action_space      : {fenv.action_space}   '
          f'(= {args.num_strokes} strokes × 13)')
    print(f'Dataset           : {args.dataset}')
    print(f'Data root         : {args.data_root}')
    print(f'Max steps M       : {args.max_step}')
    print(f'Strokes per step N: {args.num_strokes}')
    print(f'Phase heads       : {args.num_experts}  sigma={args.sigma}\n')

    train(agent, fenv, evaluate, args, writer)

#!/usr/bin/env python3
import cv2
import random
import numpy as np
import argparse
import os
import torch
import time
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard

# --- FIX 1: Cross-platform Path Handling ---
# We get the folder name without full path to avoid "D:" issues
current_path = os.path.abspath('.')
exp = os.path.basename(current_path)

# Ensure the log directory exists before initializing TensorBoard
log_dir = os.path.join('..', 'train_log', exp)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

writer = TensorBoard(log_dir)

# --- FIX 2: Replace Linux 'ln' and 'mkdir' with Python native calls ---
# Create model directory safely
if not os.path.exists('./model'):
    os.makedirs('./model', exist_ok=True)

# Symbolic links ('ln -s') usually require Admin rights on Windows. 
# It's safer to just print the location or skip this on Windows.
try:
    if os.name == 'nt': # Windows
        print(f"Logs are being saved to: {log_dir}")
    else: # Linux/Mac
        os.system('ln -sf ../train_log/{} ./log'.format(exp))
except Exception as e:
    print(f"Note: Could not create symlink: {e}")

def train(agent, env, evaluate):
    train_times = args.train_times
    env_batch = args.env_batch
    validate_interval = args.validate_interval
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times
    resume = args.resume
    output = args.output
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = args.noise_factor
    
    while step <= train_times:
        step += 1
        episode_steps += 1
        
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)    
            
        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action)
        agent.observe(reward, observation, done, step)
        
        if (episode_steps >= max_step and max_step):
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward_eval, dist_eval = evaluate(env, agent.select_action, debug=debug)
                    if debug: 
                        prRed('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'
                              .format(step - 1, np.mean(reward_eval), np.mean(dist_eval), np.var(dist_eval)))
                    
                    writer.add_scalar('validate/mean_reward', np.mean(reward_eval), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist_eval), step)
                    writer.add_scalar('validate/var_dist', np.var(dist_eval), step)
                    agent.save_model(output)
            
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            
            if step > args.warmup:
                # Adjust Learning Rate based on progress
                if step < 10000 * max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                    
                for i in range(episode_train_times):
                    Q, value_loss = agent.update_policy(lr)
                    # Use .item() for modern PyTorch scalars
                    tot_Q += Q.detach().cpu().numpy()
                    tot_value_loss += value_loss.detach().cpu().numpy()
                
                writer.add_scalar('train/critic_lr', lr[0], step)
                writer.add_scalar('train/actor_lr', lr[1], step)
                writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
            
            if debug: 
                prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' 
                        .format(episode, step, train_time_interval, time.time()-time_stamp)) 
            
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint')

    # hyper-parameters
    parser.add_argument('--warmup', default=400, type=int, help='timesteps before training starts')
    parser.add_argument('--discount', default=0.95**5, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='episodes between validations')
    parser.add_argument('--validate_episodes', default=5, type=int, help='episodes per validation run')
    parser.add_argument('--train_times', default=2000000, type=int, help='total train steps')
    parser.add_argument('--episode_train_times', default=10, type=int, help='train iterations per episode')    
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path')
    parser.add_argument('--output', default='./model', type=str, help='Output path')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print debug info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    
    args = parser.parse_args()    
    args.output = get_output_folder(args.output, "Paint")
    
    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # Delayed imports to ensure seeds and writers are set
    from DRL.ddpg import DDPG
    from DRL.multi import fastenv
    
    fenv = fastenv(args.max_step, args.env_batch, writer)
    agent = DDPG(args.batch_size, args.env_batch, args.max_step, 
                 args.tau, args.discount, args.rmsize, 
                 writer, args.resume, args.output)
    evaluate = Evaluator(args, writer)
    
    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)
    train(agent, fenv, evaluate)
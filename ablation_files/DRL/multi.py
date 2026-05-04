"""
multi.py  — fastenv wrapper
============================
Wraps the Paint environment and adds TensorBoard image logging.
Passes reward_mode (ablation switch) through to Paint.
"""

import cv2
import torch
import numpy as np
from env import Paint
from utils.util import to_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fastenv:
    """
    Parameters
    ----------
    max_episode_length : int   — max painting steps M
    env_batch          : int   — parallel environments
    num_strokes        : int   — strokes per step N
    dataset            : str   — celeba|mnist|imagenet|cub200|stanford_cars
    data_root          : str   — path to dataset root
    reward_mode        : str   — 'gan_only' (baseline) | 'dense' (PhasePaint)
    writer             : TensorBoard writer or None
    dataset_kwargs     : dict  — extra args forwarded to DatasetLoader
    """

    def __init__(self,
                 max_episode_length=40,
                 env_batch=64,
                 num_strokes=5,
                 dataset='celeba',
                 data_root='./data',
                 reward_mode='dense',
                 writer=None,
                 dataset_kwargs=None):

        self.max_episode_length = max_episode_length
        self.env_batch          = env_batch
        self.num_strokes        = num_strokes
        self.reward_mode        = reward_mode

        self.env = Paint(
            batch_size=env_batch,
            max_step=max_episode_length,
            num_strokes=num_strokes,
            dataset=dataset,
            data_root=data_root,
            reward_mode=reward_mode,
            dataset_kwargs=dataset_kwargs or {},
        )
        self.env.load_data()

        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space

        self.writer = writer
        self.test   = False
        self.log    = 0

    def save_image(self, log, step):
        if self.writer is None:
            return
        for i in range(self.env_batch):
            if self.env.imgid[i] <= 10:
                canvas = cv2.cvtColor(
                    to_numpy(self.env.canvas[i].permute(1, 2, 0)),
                    cv2.COLOR_BGR2RGB)
                self.writer.add_image(
                    f'{self.env.imgid[i]}/canvas_{step}.png', canvas, log)
        if step == self.max_episode_length:
            for i in range(self.env_batch):
                if self.env.imgid[i] < 50:
                    gt = cv2.cvtColor(
                        to_numpy(self.env.gt[i].permute(1, 2, 0)),
                        cv2.COLOR_BGR2RGB)
                    canvas = cv2.cvtColor(
                        to_numpy(self.env.canvas[i].permute(1, 2, 0)),
                        cv2.COLOR_BGR2RGB)
                    iid = self.env.imgid[i]
                    self.writer.add_image(f'{iid}/_target.png', gt,     log)
                    self.writer.add_image(f'{iid}/_canvas.png', canvas, log)

    def step(self, action):
        with torch.no_grad():
            ob, r, d, info = self.env.step(torch.tensor(action).to(device))
        if d[0]:
            if not self.test and self.writer:
                dist = self.get_dist()
                for i in range(self.env_batch):
                    self.writer.add_scalar('train/dist', dist[i], self.log)
                    self.log += 1
        return ob, r, d, info

    def get_dist(self):
        return to_numpy(
            (((self.env.gt.float() - self.env.canvas.float()) / 255) ** 2
             ).mean(1).mean(1).mean(1))

    def reset(self, test=False, episode=0):
        self.test = test
        return self.env.reset(test, episode * self.env_batch)

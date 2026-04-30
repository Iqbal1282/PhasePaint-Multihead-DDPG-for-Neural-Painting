"""
multi.py  (robust version)
==========================
fastenv wraps the Paint environment and adds:
  - per-step canvas image logging to TensorBoard
  - dist metric at episode end
  - passes num_strokes and dataset config through to Paint
"""

import cv2
import torch
import numpy as np
from env import Paint
from utils.util import to_numpy
from DRL.ddpg import decode as _ddpg_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fastenv:
    """
    Parameters
    ----------
    max_episode_length : int   — max painting steps M
    env_batch          : int   — parallel environments
    num_strokes        : int   — strokes per step N  (default 5)
    dataset            : str   — celeba|mnist|imagenet|cub200|stanford_cars
    data_root          : str   — path to dataset root
    writer             : TensorBoard writer or None
    dataset_kwargs     : dict  — extra args for DatasetLoader
    """

    def __init__(self,
                 max_episode_length: int = 40,
                 env_batch: int = 64,
                 num_strokes: int = 5,
                 dataset: str = 'celeba',
                 data_root: str = './data',
                 writer=None,
                 dataset_kwargs: dict = None):

        self.max_episode_length = max_episode_length
        self.env_batch          = env_batch
        self.num_strokes        = num_strokes

        self.env = Paint(
            batch_size=env_batch,
            max_step=max_episode_length,
            num_strokes=num_strokes,
            dataset=dataset,
            data_root=data_root,
            dataset_kwargs=dataset_kwargs or {},
        )
        self.env.load_data()

        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space   # num_strokes * 13

        self.writer = writer
        self.test   = False
        self.log    = 0

    def save_image(self, log: int, step: int):
        for i in range(self.env_batch):
            if self.env.imgid[i] <= 10:
                canvas = cv2.cvtColor(
                    to_numpy(self.env.canvas[i].permute(1, 2, 0)),
                    cv2.COLOR_BGR2RGB)
                self.writer.add_image(
                    '{}/canvas_{}.png'.format(self.env.imgid[i], step),
                    canvas, log)
        if step == self.max_episode_length:
            for i in range(self.env_batch):
                if self.env.imgid[i] < 50:
                    gt     = cv2.cvtColor(
                        to_numpy(self.env.gt[i].permute(1, 2, 0)),
                        cv2.COLOR_BGR2RGB)
                    canvas = cv2.cvtColor(
                        to_numpy(self.env.canvas[i].permute(1, 2, 0)),
                        cv2.COLOR_BGR2RGB)
                    self.writer.add_image(
                        str(self.env.imgid[i]) + '/_target.png', gt,     log)
                    self.writer.add_image(
                        str(self.env.imgid[i]) + '/_canvas.png', canvas, log)

    def step(self, action):
        with torch.no_grad():
            ob, r, d, info = self.env.step(torch.tensor(action).to(device))
        if d[0]:
            if not self.test:
                dist = self.get_dist()
                for i in range(self.env_batch):
                    self.writer.add_scalar('train/dist', dist[i], self.log)
                    self.log += 1
        return ob, r, d, info

    def get_dist(self):
        return to_numpy(
            (((self.env.gt.float() - self.env.canvas.float()) / 255) ** 2
             ).mean(1).mean(1).mean(1))

    def reset(self, test: bool = False, episode: int = 0):
        self.test = test
        return self.env.reset(test, episode * self.env_batch)

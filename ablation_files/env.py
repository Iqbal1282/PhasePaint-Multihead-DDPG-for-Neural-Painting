"""
env.py  (robust multi-dataset + ablation-ready version)
=========================================================
Painting environment supporting:
  - celeba / mnist / imagenet / cub200 / stanford_cars
  - configurable N strokes per step  (num_strokes N, default 5)
  - configurable M max steps         (max_step M,    default 40)
  - reward_mode: 'gan_only' | 'dense'  <- ablation switch

reward_mode='gan_only'
    Mirrors original ddpg.py: env.step() returns zeros as the extrinsic
    reward; the full signal comes from the WGAN discriminator inside
    MoEDDPG._evaluate(). This is the exact baseline behaviour.

reward_mode='dense'
    r1  step-wise edge-weighted MSE improvement  (x10)
    r2  progress-scaled quality shaping          (x3)
    r3  terminal quality bonus                   (x10, last step only)
    r4  stroke-alignment reward                  (capped 0.1)
    penalty  -0.1 per step  |  floor 0.01
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets import DatasetLoader
from utils.util import to_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIDTH = 128


def _make_decoder():
    from Renderer.model import FCN
    import os
    for rpath in ('../renderer.pkl', './renderer.pkl', 'renderer.pkl'):
        if os.path.isfile(rpath):
            dec = FCN()
            dec.load_state_dict(torch.load(rpath, map_location=device))
            dec.to(device).eval()
            coord = torch.zeros([1, 2, WIDTH, WIDTH])
            for i in range(WIDTH):
                for j in range(WIDTH):
                    coord[0, 0, i, j] = i / (WIDTH - 1)
                    coord[0, 1, i, j] = j / (WIDTH - 1)
            return dec, coord.to(device)
    raise FileNotFoundError("renderer.pkl not found at ../renderer.pkl")

_Decoder, _coord = _make_decoder()


def decode(x, canvas, num_strokes=5):
    x = x.view(-1, 13)
    stroke = 1 - _Decoder(x[:, :10])
    stroke = stroke.view(-1, WIDTH, WIDTH, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke       = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke       = stroke.view(-1, num_strokes, 1, WIDTH, WIDTH)
    color_stroke = color_stroke.view(-1, num_strokes, 3, WIDTH, WIDTH)
    for i in range(num_strokes):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas


_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         device=device).float().view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         device=device).float().view(1, 1, 3, 3)


class Paint:
    def __init__(self, batch_size, max_step, num_strokes=5,
                 dataset='celeba', data_root='./data',
                 reward_mode='dense',
                 spatial_grid=4, spatial_alpha=0.3,
                 dataset_kwargs=None):

        self.batch_size  = batch_size
        self.max_step    = max_step
        self.num_strokes = num_strokes
        self.reward_mode = reward_mode

        self.action_space      = num_strokes * 13
        self.observation_space = (batch_size, WIDTH, WIDTH, 7)
        self.test              = False

        self.K           = spatial_grid
        self.alpha       = spatial_alpha
        self.edge_lambda = 10.0
        self.SOBEL_X     = _SOBEL_X
        self.SOBEL_Y     = _SOBEL_Y

        self._loader         = None
        self._dataset_name   = dataset
        self._data_root      = data_root
        self._dataset_kwargs = dataset_kwargs or {}

    def load_data(self):
        self._loader = DatasetLoader(
            dataset=self._dataset_name, width=WIDTH,
            data_root=self._data_root, **self._dataset_kwargs)
        self._loader.load()

    @property
    def train_num(self):
        return self._loader.train_num if self._loader else 0

    @property
    def test_num(self):
        return self._loader.test_num if self._loader else 0

    def pre_data(self, idx, test):
        if test:
            return self._loader.get_test(idx)
        return self._loader.get_train(idx, augment=True)

    def get_canny_mask(self, gt_images):
        gt_np = to_numpy(gt_images.permute(0, 2, 3, 1))
        masks = []
        for i in range(self.batch_size):
            img   = (gt_np[i] * 255).clip(0, 255).astype(np.uint8)
            gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            masks.append(edges / 255.0)
        mask_t = torch.from_numpy(np.array(masks)).float().to(device)
        return mask_t.unsqueeze(1)

    def weighted_mse(self, canvas, gt, mask):
        mse_map    = torch.mean((canvas - gt) ** 2, dim=1, keepdim=True)
        masked_err = mse_map * mask
        denom      = torch.sum(mask, dim=(1, 2, 3)) + 1e-8
        return torch.sum(masked_err, dim=(1, 2, 3)) / denom

    def _patch_mse(self, canvas, gt):
        ps  = WIDTH // self.K
        err = ((canvas.float() - gt.float()) / 255.0) ** 2
        err = err.mean(dim=1, keepdim=True)
        err = err.view(self.batch_size, 1, self.K, ps, self.K, ps)
        return err.mean(dim=(3, 5)).squeeze(1)

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2
                ).mean(1).mean(1).mean(1)

    def reset(self, test=False, begin_num=0):
        self.test   = test
        self.imgid  = [0] * self.batch_size
        self.gt     = torch.zeros([self.batch_size, 3, WIDTH, WIDTH],
                                   dtype=torch.uint8).to(device)
        for i in range(self.batch_size):
            idx = ((i + begin_num) % self.test_num) if test \
                  else np.random.randint(self.train_num)
            self.imgid[i] = idx
            self.gt[i]    = torch.tensor(self.pre_data(idx, test))

        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum    = 0
        self.canvas     = torch.zeros([self.batch_size, 3, WIDTH, WIDTH],
                                       dtype=torch.uint8).to(device)
        self.lastdis    = self.ini_dis = self.cal_dis()
        self.last_patch_err = self._patch_mse(self.canvas, self.gt)

        gt_t              = self.gt.float() / 255.0
        self.gt_edge_mask = self.get_canny_mask(gt_t)

        initial_wmse           = self.weighted_mse(
            self.canvas.float() / 255.0, gt_t, self.gt_edge_mask)
        self.best_wmse         = initial_wmse.clone()
        self.last_weighted_mse = initial_wmse

        with torch.no_grad():
            gray = self.gt.float().mean(dim=1, keepdim=True) / 255.0
            gx   = F.conv2d(gray, self.SOBEL_X, padding=1)
            gy   = F.conv2d(gray, self.SOBEL_Y, padding=1)
            self.gt_mag         = torch.sqrt(gx**2 + gy**2)
            self.gt_mag         = self.gt_mag / (self.gt_mag.max() + 1e-8)
            raw_ori             = torch.atan2(gx, -gy)
            self.gt_orientation = torch.where(
                self.gt_mag > 0.15, raw_ori,
                torch.tensor(0.0, device=device))

        return self.observation()

    def observation(self):
        T = torch.ones([self.batch_size, 1, WIDTH, WIDTH],
                        dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas, self.gt, T.to(device)), 1)

    def step(self, action):
        self.canvas = (
            decode(action, self.canvas.float() / 255,
                   num_strokes=self.num_strokes) * 255
        ).byte()

        canvas_t     = self.canvas.float() / 255.0
        gt_t         = self.gt.float()     / 255.0
        current_wmse = self.weighted_mse(canvas_t, gt_t, self.gt_edge_mask)

        # ── Stroke alignment signal (always computed; used only in dense mode)
        a0     = action[:, :13]
        x1_abs = a0[:, 0] + (a0[:, 4] - a0[:, 0]) * a0[:, 2]
        y1_abs = a0[:, 1] + (a0[:, 5] - a0[:, 1]) * a0[:, 3]
        mid_x  = 0.25 * a0[:, 0] + 0.5 * x1_abs + 0.25 * a0[:, 4]
        mid_y  = 0.25 * a0[:, 1] + 0.5 * y1_abs + 0.25 * a0[:, 5]
        px     = (mid_x * (WIDTH - 1)).long().clamp(0, WIDTH - 1)
        py     = (mid_y * (WIDTH - 1)).long().clamp(0, WIDTH - 1)
        bidx   = torch.arange(self.batch_size, device=device)
        t_theta = self.gt_orientation[bidx, 0, py, px]
        e_conf  = self.gt_mag[bidx, 0, py, px]
        dx, dy  = a0[:, 4] - a0[:, 0], a0[:, 5] - a0[:, 1]
        align   = torch.abs(torch.cos(torch.atan2(dy, dx) - t_theta))
        align_r = torch.clamp(0.05 * align * e_conf, 0, 0.1)

        # ─────────────────────────────────────────────────────────────────────
        if self.reward_mode == 'gan_only':
            # Baseline: environment reward = 0.
            # GAN discriminator reward is added inside MoEDDPG._evaluate(),
            # exactly mirroring the original ddpg.py behaviour.
            extrinsic = torch.zeros(self.batch_size, device=device)
            align_r   = torch.zeros(self.batch_size, device=device)

        else:  # 'dense'
            # r1: step-wise WMSE improvement (dense signal every step)
            step_improvement   = self.last_weighted_mse - current_wmse
            improvement_reward = torch.clamp(step_improvement, min=0.0) * 10.0

            # r2: progress-scaled quality shaping (prevents late regression)
            quality        = 1.0 - (current_wmse / (self.ini_dis + 1e-8))
            progress       = self.stepnum / self.max_step
            quality_reward = quality * progress * 3.0

            # r3: terminal quality bonus (last step only)
            terminal_reward = 0.0
            if self.stepnum == self.max_step - 1:
                terminal_reward = 10.0 * quality

            # time penalty
            extrinsic = improvement_reward + quality_reward \
                        + terminal_reward - 0.1
            extrinsic = torch.clamp(extrinsic, min=0.01)

        total = extrinsic + align_r

        self.last_weighted_mse = current_wmse.detach()
        self.best_wmse         = torch.min(self.best_wmse, current_wmse)
        self.stepnum          += 1
        done = np.array([self.stepnum == self.max_step] * self.batch_size)

        # quality for logging (may be 0-tensor in gan_only mode)
        quality_val = (1.0 - current_wmse / (self.ini_dis + 1e-8)).mean().item()

        info = {
            'target_theta':  {'theta': t_theta.detach(), 'mag': e_conf.detach()},
            'extrinsic':     extrinsic.mean().item() if torch.is_tensor(extrinsic) else 0.0,
            'intrinsic':     0.0,
            'alignment':     align_r.mean().item() if torch.is_tensor(align_r) else 0.0,
            'step_progress': self.stepnum / self.max_step,
            'total':         total.mean().item(),
            'quality':       quality_val,
        }
        return self.observation().detach(), total.cpu().numpy(), done, info

    def cal_reward(self):
        dis    = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)

    def cal_alignment_reward(self, action):
        a0     = action[:, :13]
        dx, dy = a0[:, 4] - a0[:, 0], a0[:, 5] - a0[:, 1]
        theta_s = torch.atan2(dy, dx)
        x1_abs  = a0[:, 0] + (a0[:, 4] - a0[:, 0]) * a0[:, 2]
        y1_abs  = a0[:, 1] + (a0[:, 5] - a0[:, 1]) * a0[:, 3]
        mid_x   = (0.25*a0[:,0] + 0.5*x1_abs + 0.25*a0[:,4]) * (WIDTH-1)
        mid_y   = (0.25*a0[:,1] + 0.5*y1_abs + 0.25*a0[:,5]) * (WIDTH-1)
        px      = mid_x.long().clamp(0, WIDTH-1)
        py      = mid_y.long().clamp(0, WIDTH-1)
        bidx    = torch.arange(self.batch_size, device=device)
        t_theta = self.gt_orientation[bidx, 0, py, px]
        return torch.abs(torch.cos(theta_s - t_theta))

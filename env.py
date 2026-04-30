"""
env.py  (robust multi-dataset version)
=======================================
Painting environment compatible with:
  - celeba / mnist / imagenet / cub200 / stanford_cars
  - configurable N strokes per step  (num_strokes N, default 5)
  - configurable M max steps         (max_step M,    default 40)

Key changes vs. original
-------------------------
1. Dataset loading delegated to `datasets.DatasetLoader`.
2. `num_strokes` is a first-class argument. action_space = num_strokes * 13.
3. decode() / env.step() both respect num_strokes.
4. All reward logic (canny, high-water-mark, alignment) is preserved.
5. Backward compatible: defaults reproduce the original (5 strokes, 40 steps, celeba).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets import DatasetLoader
from utils.util import to_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIDTH = 128   # Canvas resolution (fixed — renderer is trained at 128)


# ─────────────────────────────────────────────────────────────────────────────
#  Stroke decoder
# ─────────────────────────────────────────────────────────────────────────────

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
    raise FileNotFoundError("renderer.pkl not found. Expected at ../renderer.pkl")

_Decoder, _coord = _make_decoder()


def decode(x: torch.Tensor, canvas: torch.Tensor, num_strokes: int = 5) -> torch.Tensor:
    """
    Decode num_strokes bezier strokes onto canvas.
    x      : (B, num_strokes * 13) in [0,1]
    canvas : (B, 3, W, W)          in [0,1]
    """
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


# ─────────────────────────────────────────────────────────────────────────────
#  Sobel kernels
# ─────────────────────────────────────────────────────────────────────────────

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         device=device).float().view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         device=device).float().view(1, 1, 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
#  Paint environment
# ─────────────────────────────────────────────────────────────────────────────

class Paint:
    """
    Parameters
    ----------
    batch_size    : int   — parallel environments
    max_step      : int   — max painting steps M
    num_strokes   : int   — strokes per step N  (default 5)
    dataset       : str   — celeba|mnist|imagenet|cub200|stanford_cars
    data_root     : str   — path to dataset root directory
    spatial_grid  : int   — patch grid K for spatial bonus
    spatial_alpha : float — spatial bonus weight
    dataset_kwargs: dict  — forwarded to DatasetLoader (e.g. max_train for imagenet)
    """

    def __init__(self,
                 batch_size: int,
                 max_step: int,
                 num_strokes: int = 5,
                 dataset: str = 'celeba',
                 data_root: str = './data',
                 spatial_grid: int = 4,
                 spatial_alpha: float = 0.3,
                 dataset_kwargs: dict = None):

        self.batch_size  = batch_size
        self.max_step    = max_step
        self.num_strokes = num_strokes

        # action_space = N * 13  (N strokes, 13 params each)
        self.action_space      = num_strokes * 13
        self.observation_space = (batch_size, WIDTH, WIDTH, 7)
        self.test              = False

        self.K           = spatial_grid
        self.alpha       = spatial_alpha
        self.edge_lambda = 10.0

        self.SOBEL_X = _SOBEL_X
        self.SOBEL_Y = _SOBEL_Y

        self._loader: DatasetLoader = None
        self._dataset_name   = dataset
        self._data_root      = data_root
        self._dataset_kwargs = dataset_kwargs or {}

    # ── Data ──────────────────────────────────────────────────────────────────

    def load_data(self):
        self._loader = DatasetLoader(
            dataset=self._dataset_name,
            width=WIDTH,
            data_root=self._data_root,
            **self._dataset_kwargs,
        )
        self._loader.load()

    @property
    def train_num(self):
        return self._loader.train_num if self._loader else 0

    @property
    def test_num(self):
        return self._loader.test_num if self._loader else 0

    def pre_data(self, idx: int, test: bool) -> np.ndarray:
        if test:
            return self._loader.get_test(idx)
        return self._loader.get_train(idx, augment=True)

    # ── Canny importance mask ─────────────────────────────────────────────────

    def get_canny_mask(self, gt_images: torch.Tensor) -> torch.Tensor:
        gt_np = to_numpy(gt_images.permute(0, 2, 3, 1))
        masks = []
        for i in range(self.batch_size):
            img  = (gt_np[i] * 255).clip(0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            e_fine = cv2.Canny(gray, 50, 120)
            lap    = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
            lap    = cv2.convertScaleAbs(lap)
            _, e_crs = cv2.threshold(lap, 40, 255, cv2.THRESH_BINARY)
            comb   = cv2.bitwise_or(e_fine, e_crs)
            dist   = cv2.distanceTransform(255 - comb, cv2.DIST_L2, 3)
            dist_m = np.exp(-dist / 7.0)
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sm = np.sqrt(sx**2 + sy**2)
            sm = cv2.normalize(sm, None, 0, 1, cv2.NORM_MINMAX)
            imp = dist_m * (sm + 0.3)
            imp = imp / (imp.max() + 1e-8)
            masks.append(imp)
        mask_t = torch.from_numpy(np.array(masks)).float().to(gt_images.device)
        return mask_t.unsqueeze(1)

    # ── Loss helpers ──────────────────────────────────────────────────────────

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

    # ── Episode ───────────────────────────────────────────────────────────────

    def reset(self, test: bool = False, begin_num: int = 0):
        self.test   = test
        self.imgid  = [0] * self.batch_size
        self.gt     = torch.zeros([self.batch_size, 3, WIDTH, WIDTH],
                                   dtype=torch.uint8).to(device)
        for i in range(self.batch_size):
            if test:
                idx = (i + begin_num) % self.test_num
            else:
                idx = np.random.randint(self.train_num)
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

    def step(self, action: torch.Tensor):
        """
        action : (B, num_strokes * 13) in [0,1]
        """
        self.canvas = (
            decode(action, self.canvas.float() / 255,
                num_strokes=self.num_strokes) * 255
        ).byte()

        canvas_t     = self.canvas.float() / 255.0
        gt_t         = self.gt.float()     / 255.0
        current_wmse = self.weighted_mse(canvas_t, gt_t, self.gt_edge_mask)

        # ── REWARD 1: Step-wise improvement (dense signal) ─────────────────────
        # Compare to LAST step, not best ever. This gives signal every step.
        step_improvement = self.last_weighted_mse - current_wmse
        improvement_reward = torch.clamp(step_improvement, min=0.0) * 10.0

        # ── REWARD 2: Quality maintenance (prevents Q collapse) ────────────────
        # Reward for current canvas quality, scaled by progress.
        # Later steps get more reward for same quality → encourages finishing well.
        quality = 1.0 - (current_wmse / (self.ini_dis + 1e-8))
        progress = self.stepnum / self.max_step
        quality_reward = quality * progress * 3.0  # 0 at step 0, up to +3 at end

        # ── REWARD 3: Terminal bonus (ensures final steps matter) ──────────────
        terminal_reward = 0.0
        if self.stepnum == self.max_step - 1:  # last step
            terminal_reward = 20.0 * quality  # up to +20 for perfect finish

        # ── PENALTY: Time cost (prevents lazy behavior) ────────────────────────
        time_penalty = -0.1  # stronger than your -0.005

        # ── Alignment reward (unchanged) ───────────────────────────────────────
        a0      = action[:, :13]
        x1_abs  = a0[:, 0] + (a0[:, 4] - a0[:, 0]) * a0[:, 2]
        y1_abs  = a0[:, 1] + (a0[:, 5] - a0[:, 1]) * a0[:, 3]
        mid_x   = 0.25 * a0[:, 0] + 0.5 * x1_abs + 0.25 * a0[:, 4]
        mid_y   = 0.25 * a0[:, 1] + 0.5 * y1_abs + 0.25 * a0[:, 5]
        px      = (mid_x * (WIDTH - 1)).long().clamp(0, WIDTH - 1)
        py      = (mid_y * (WIDTH - 1)).long().clamp(0, WIDTH - 1)
        bidx    = torch.arange(self.batch_size, device=device)
        t_theta = self.gt_orientation[bidx, 0, py, px]
        e_conf  = self.gt_mag[bidx, 0, py, px]
        dx, dy  = a0[:, 4] - a0[:, 0], a0[:, 5] - a0[:, 1]
        align   = torch.abs(torch.cos(torch.atan2(dy, dx) - t_theta))
        align_r = torch.clamp(0.05 * align * e_conf, 0, 0.1)

        # ── Combine ────────────────────────────────────────────────────────────
        extrinsic = improvement_reward + quality_reward + terminal_reward + time_penalty
        extrinsic = torch.clamp(extrinsic, min=0.01)  # floor so Q never hits 0
        total     = extrinsic + align_r

        # ── Update state ───────────────────────────────────────────────────────
        self.last_weighted_mse = current_wmse.detach()
        self.best_wmse = torch.min(self.best_wmse, current_wmse)  # track but don't use
        self.stepnum += 1
        done = np.array([self.stepnum == self.max_step] * self.batch_size)

        info = {
            'target_theta':  {'theta': t_theta.detach(), 'mag': e_conf.detach()},
            'extrinsic':     extrinsic.mean().item(),
            'intrinsic':     0.0,
            'alignment':     align_r.mean().item(),
            'step_progress': progress,
            'total':         total.mean().item(),
            'quality':       quality.mean().item(),  # log for debugging
        }
        return self.observation().detach(), total.cpu().numpy(), done, info

    def cal_reward(self):
        dis    = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)

    def cal_alignment_reward(self, action):
        a0      = action[:, :13]
        dx      = a0[:, 4] - a0[:, 0]
        dy      = a0[:, 5] - a0[:, 1]
        theta_s = torch.atan2(dy, dx)
        x1_abs  = a0[:, 0] + (a0[:, 4] - a0[:, 0]) * a0[:, 2]
        y1_abs  = a0[:, 1] + (a0[:, 5] - a0[:, 1]) * a0[:, 3]
        mid_x   = (0.25*a0[:, 0] + 0.5*x1_abs + 0.25*a0[:, 4]) * (WIDTH-1)
        mid_y   = (0.25*a0[:, 1] + 0.5*y1_abs + 0.25*a0[:, 5]) * (WIDTH-1)
        px      = mid_x.long().clamp(0, WIDTH-1)
        py      = mid_y.long().clamp(0, WIDTH-1)
        bidx    = torch.arange(self.batch_size, device=device)
        t_theta = self.gt_orientation[bidx, 0, py, px]
        return torch.abs(torch.cos(theta_s - t_theta))

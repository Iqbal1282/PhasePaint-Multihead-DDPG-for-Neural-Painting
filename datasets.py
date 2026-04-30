"""
datasets.py
===========
Unified dataset loader for all supported datasets:
  - celeba      : CelebA (aligned, 200k images)
  - mnist       : MNIST (60k train / 10k test)  ← grayscale → RGB
  - imagenet    : ImageNet (ILSVRC, arbitrary subset)
  - cub200      : CUB-200-2011 Bird dataset
  - stanford_cars: Stanford Cars 196 dataset

All images are resized to `width × width` (default 128).

Usage
-----
    from datasets import DatasetLoader
    loader = DatasetLoader(dataset='celeba', width=128, data_root='./data')
    loader.load()
    img = loader.get_train(idx)    # → np.uint8 (3, W, W)
    img = loader.get_test(idx)     # → np.uint8 (3, W, W)
    print(loader.train_num, loader.test_num)

Expected directory layouts
--------------------------
CelebA:
    ./data/img_align_celeba/{000001..202599}.jpg

MNIST:
    auto-downloaded via torchvision to ./data/MNIST/

ImageNet:
    ./data/imagenet/train/<class>/<file>.JPEG
    ./data/imagenet/val/<class>/<file>.JPEG

CUB-200-2011:
    ./data/CUB_200_2011/images/<class>/<file>.jpg
    ./data/CUB_200_2011/train_test_split.txt  (1=train, 0=test)
    ./data/CUB_200_2011/images.txt

Stanford Cars:
    ./data/stanford_cars/cars_train/<file>.jpg
    ./data/stanford_cars/cars_test/<file>.jpg
    (Annotation CSV optional; if absent all train/ is used)
"""

import os
import cv2
import glob
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.datasets as dsets


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_resize(path: str, width: int) -> np.ndarray:
    """Load an image file, convert to RGB uint8, resize to (width, width).
    Returns np.ndarray shape (width, width, 3) uint8, or None on error."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.ndim == 2:                         # grayscale fallback
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (width, width), interpolation=cv2.INTER_LINEAR)
    return img   # BGR, uint8, (H, W, 3)


def _chw(img: np.ndarray) -> np.ndarray:
    """HWC → CHW."""
    return np.transpose(img, (2, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation (same as original codebase)
# ─────────────────────────────────────────────────────────────────────────────

_aug = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
])


def _augment(img_hwc: np.ndarray) -> np.ndarray:
    """Apply random horizontal flip (training only). img is HWC uint8."""
    img_pil = _aug(img_hwc)
    return np.asarray(img_pil)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-dataset loaders  (return lists of HWC BGR uint8 arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _load_celeba(data_root: str, width: int, max_images: int = 202599):
    """
    CelebA aligned images.
    First 2000 → test; remainder → train  (matches original env.py logic).
    """
    img_dir = os.path.join(data_root, 'img_align_celeba')
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"CelebA directory not found: {img_dir}\n"
            "Download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html "
            "and unzip into ./data/img_align_celeba/"
        )
    train, test = [], []
    for i in range(min(max_images, 202599)):
        img_id = '%06d' % (i + 1)
        path = os.path.join(img_dir, img_id + '.jpg')
        img = _load_resize(path, width)
        if img is None:
            continue
        if i >= 2000:
            train.append(img)
        else:
            test.append(img)
        if (i + 1) % 10000 == 0:
            print(f'  [celeba] loaded {i+1} images …')
    print(f'  [celeba] train={len(train)}  test={len(test)}')
    return train, test


def _load_mnist(data_root: str, width: int):
    """
    MNIST: auto-downloads if needed.
    Grayscale is replicated to 3 channels.
    Returns BGR uint8 arrays (consistent with OpenCV convention).
    """
    print('  [mnist] Loading (auto-download if missing) …')
    train_ds = dsets.MNIST(root=data_root, train=True,  download=True)
    test_ds  = dsets.MNIST(root=data_root, train=False, download=True)

    def _mnist_list(ds):
        imgs = []
        for pil_img, _ in ds:
            arr = np.asarray(pil_img.convert('RGB'))   # HWC RGB uint8
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            arr = cv2.resize(arr, (width, width), interpolation=cv2.INTER_NEAREST)
            imgs.append(arr)
        return imgs

    train = _mnist_list(train_ds)
    test  = _mnist_list(test_ds)
    print(f'  [mnist] train={len(train)}  test={len(test)}')
    return train, test


def _glob_images(directory: str, exts=('jpg', 'jpeg', 'png', 'JPEG', 'JPG')):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, '**', f'*.{ext}'),
                               recursive=True))
    paths.sort()
    return paths


def _load_imagenet(data_root: str, width: int, max_train: int = 0, max_val: int = 5000):
    """
    ImageNet ILSVRC.
    Expects:  <data_root>/imagenet/train/<class>/<file>.JPEG
              <data_root>/imagenet/val/<class>/<file>.JPEG
    max_train=0 means load all.
    """
    train_dir = os.path.join(data_root, 'imagenet', 'train')
    val_dir   = os.path.join(data_root, 'imagenet', 'val')

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet train dir not found: {train_dir}\n"
            "Download ILSVRC and place under ./data/imagenet/train/"
        )

    train_paths = _glob_images(train_dir)
    val_paths   = _glob_images(val_dir) if os.path.isdir(val_dir) else []

    if max_train > 0:
        random.shuffle(train_paths)
        train_paths = train_paths[:max_train]
    if max_val > 0:
        val_paths = val_paths[:max_val]

    def _load_list(paths, label):
        imgs = []
        for k, p in enumerate(paths):
            img = _load_resize(p, width)
            if img is not None:
                imgs.append(img)
            if (k + 1) % 50000 == 0:
                print(f'  [imagenet/{label}] loaded {k+1} …')
        return imgs

    train = _load_list(train_paths, 'train')
    test  = _load_list(val_paths,   'val')
    print(f'  [imagenet] train={len(train)}  test={len(test)}')
    return train, test


def _load_cub200(data_root: str, width: int):
    """
    CUB-200-2011 Birds.
    Expects:
        <data_root>/CUB_200_2011/images.txt          (id path)
        <data_root>/CUB_200_2011/train_test_split.txt (id 1=train 0=test)
        <data_root>/CUB_200_2011/images/<class>/<file>.jpg
    Falls back to glob if annotation files are missing.
    """
    cub_root   = os.path.join(data_root, 'CUB_200_2011')
    img_root   = os.path.join(cub_root, 'images')
    images_txt = os.path.join(cub_root, 'images.txt')
    split_txt  = os.path.join(cub_root, 'train_test_split.txt')

    if not os.path.isdir(cub_root):
        raise FileNotFoundError(
            f"CUB-200 directory not found: {cub_root}\n"
            "Download from https://www.vision.caltech.edu/datasets/cub_200_2011/"
        )

    # ── Use annotation files if available ────────────────────────────────────
    if os.path.isfile(images_txt) and os.path.isfile(split_txt):
        id2path = {}
        with open(images_txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    id2path[parts[0]] = parts[1]
        id2split = {}
        with open(split_txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    id2split[parts[0]] = int(parts[1])
        train, test = [], []
        for img_id, rel_path in id2path.items():
            full_path = os.path.join(img_root, rel_path)
            img = _load_resize(full_path, width)
            if img is None:
                continue
            if id2split.get(img_id, 1) == 1:
                train.append(img)
            else:
                test.append(img)
    else:
        # ── Fallback: glob all, 90/10 split ──────────────────────────────────
        print('  [cub200] annotation files missing — using 90/10 random split')
        paths = _glob_images(img_root)
        random.shuffle(paths)
        split = int(0.9 * len(paths))
        train_paths, test_paths = paths[:split], paths[split:]
        train = [img for p in train_paths for img in [_load_resize(p, width)] if img is not None]
        test  = [img for p in test_paths  for img in [_load_resize(p, width)] if img is not None]

    print(f'  [cub200] train={len(train)}  test={len(test)}')
    return train, test


def _load_stanford_cars(data_root: str, width: int):
    """
    Stanford Cars 196.
    Expects:
        <data_root>/stanford_cars/cars_train/*.jpg
        <data_root>/stanford_cars/cars_test/*.jpg
    (The devkit annotation files are optional.)
    """
    cars_root  = os.path.join(data_root, 'stanford_cars')
    train_dir  = os.path.join(cars_root, 'cars_train')
    test_dir   = os.path.join(cars_root, 'cars_test')

    if not os.path.isdir(cars_root):
        raise FileNotFoundError(
            f"Stanford Cars directory not found: {cars_root}\n"
            "Download from https://ai.stanford.edu/~jkrause/cars/car_dataset.html"
        )

    def _load_dir(d):
        if not os.path.isdir(d):
            return []
        paths = _glob_images(d)
        imgs = []
        for p in paths:
            img = _load_resize(p, width)
            if img is not None:
                imgs.append(img)
        return imgs

    train = _load_dir(train_dir)
    test  = _load_dir(test_dir)

    # If test split is missing, carve 10 % of train
    if len(test) == 0 and len(train) > 0:
        random.shuffle(train)
        split = max(1, len(train) // 10)
        test  = train[:split]
        train = train[split:]

    print(f'  [stanford_cars] train={len(train)}  test={len(test)}')
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_DATASETS = ('celeba', 'mnist', 'imagenet', 'cub200', 'stanford_cars')


class DatasetLoader:
    """
    Unified loader. After calling `.load()`:
        self.train_num  : int
        self.test_num   : int
        self.get_train(idx, augment=True) → np.ndarray CHW uint8
        self.get_test(idx)                → np.ndarray CHW uint8
    """

    def __init__(self, dataset: str = 'celeba',
                 width: int = 128,
                 data_root: str = './data',
                 **kwargs):
        dataset = dataset.lower().replace('-', '_')
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Supported: {SUPPORTED_DATASETS}"
            )
        self.dataset   = dataset
        self.width     = width
        self.data_root = data_root
        self.kwargs    = kwargs

        self._train: list = []
        self._test:  list = []
        self.train_num = 0
        self.test_num  = 0

    def load(self):
        print(f'\n[DatasetLoader] Loading {self.dataset} '
              f'(width={self.width}, root={self.data_root}) …')

        loaders = {
            'celeba':        _load_celeba,
            'mnist':         _load_mnist,
            'imagenet':      _load_imagenet,
            'cub200':        _load_cub200,
            'stanford_cars': _load_stanford_cars,
        }
        self._train, self._test = loaders[self.dataset](
            self.data_root, self.width, **self.kwargs)

        self.train_num = len(self._train)
        self.test_num  = len(self._test)

        if self.train_num == 0:
            raise RuntimeError(
                f"No training images found for dataset '{self.dataset}'. "
                "Check your data_root path and directory structure."
            )
        print(f'[DatasetLoader] Ready — '
              f'train={self.train_num}  test={self.test_num}\n')

    def get_train(self, idx: int, augment: bool = True) -> np.ndarray:
        """Return CHW uint8 array (3, W, W) for training image `idx`."""
        img = self._train[idx % self.train_num]        # HWC BGR
        if augment:
            img = _augment(img)
            img = np.asarray(img)
        return _chw(img)

    def get_test(self, idx: int) -> np.ndarray:
        """Return CHW uint8 array (3, W, W) for test image `idx`."""
        img = self._test[idx % self.test_num]          # HWC BGR
        return _chw(img)

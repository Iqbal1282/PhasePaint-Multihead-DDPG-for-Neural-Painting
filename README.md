# DRL Painting Agent — Robust Multi-Dataset Edition

Phase-Conditioned Multi-Head DDPG painting agent, extended to support
**5 datasets**, **N configurable strokes per step**, and **M configurable
max steps**.

---

## What changed vs. the original

| Component | Original | Robust version |
|---|---|---|
| Dataset | CelebA (hardcoded) | celeba / mnist / imagenet / cub200 / stanford_cars |
| Data loading | `env.py` (200k loop) | `datasets.py` → `DatasetLoader` |
| Strokes per step | Fixed 5 | `--num_strokes N` (any N ≥ 1) |
| Max steps | Fixed 40 | `--max_step M` (any M) |
| `action_space` | 65 (5×13) | `N × 13` (dynamic) |
| Hardcoded pretrained load | `MoEPaint-run55` always | Only via `--resume` |

All reward logic (Canny mask, high-water-mark extrinsic, alignment intrinsic,
PCGrad) is preserved exactly.

---

## File layout

```
datasets.py                       ← NEW: unified dataset loader
env.py                            ← REWRITTEN: N strokes, M steps, any dataset
DRL/
  multi.py                        ← REWRITTEN: passes num_strokes / dataset through
  moe_ddpg_spatial_weighted_canny.py  ← PATCHED: num_strokes arg + dynamic action_dim
train_moe_spatial_weighted_canny.py   ← REWRITTEN: all new flags
evaluate.py                       ← PATCHED: dataset + num_strokes flags
```

---

## Dataset directory layouts

### CelebA (default)
```
./data/img_align_celeba/
    000001.jpg  000002.jpg  …  202599.jpg
```
Download: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### MNIST
Auto-downloaded by `torchvision` into `./data/MNIST/`.
No manual setup needed.

### ImageNet
```
./data/imagenet/
    train/<class>/<file>.JPEG
    val/<class>/<file>.JPEG
```
Download ILSVRC-2012 and extract. For memory-limited setups, use
`--max_train <N>` to cap training images.

### CUB-200-2011 Birds
```
./data/CUB_200_2011/
    images.txt
    train_test_split.txt
    images/<class_name>/<file>.jpg
```
Download: https://www.vision.caltech.edu/datasets/cub_200_2011/

Falls back to a 90/10 random split if annotation files are missing.

### Stanford Cars 196
```
./data/stanford_cars/
    cars_train/<file>.jpg
    cars_test/<file>.jpg
```
Download: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

If `cars_test/` is missing, 10 % of `cars_train/` is used as test split.

---

## Training

### Quick start (CelebA, original settings — fully backward compatible)
```bash
python train_moe_spatial_weighted_canny.py --dataset celeba --debug
```

### MNIST, 3 strokes, 20 steps
```bash
python train_moe_spatial_weighted_canny.py \
    --dataset mnist \
    --num_strokes 3 \
    --max_step 20 \
    --debug
```

### CUB-200 Birds, 8 strokes, 60 steps, 4 phase heads
```bash
python train_moe_spatial_weighted_canny.py \
    --dataset cub200 \
    --data_root ./data \
    --num_strokes 8 \
    --max_step 60 \
    --num_experts 4
```

### Stanford Cars 196, default N/M
```bash
python train_moe_spatial_weighted_canny.py \
    --dataset stanford_cars \
    --num_strokes 5 \
    --max_step 40
```

### ImageNet (large dataset — optionally cap training images)
```bash
python train_moe_spatial_weighted_canny.py \
    --dataset imagenet \
    --num_strokes 5 \
    --max_step 40
```
> To limit memory: pass `--data_root ./data` and edit `datasets.py`
> `_load_imagenet` to set `max_train` (e.g. `50000`).

### Resume from checkpoint
```bash
python train_moe_spatial_weighted_canny.py \
    --dataset celeba \
    --num_strokes 5 \
    --max_step 40 \
    --resume ./model/MoEPaint-run1
```

---

## Full flag reference

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `celeba` | Dataset: celeba / mnist / imagenet / cub200 / stanford_cars |
| `--data_root` | `./data` | Root directory for dataset files |
| `--num_strokes` | `5` | **N** strokes drawn per environment step |
| `--max_step` | `40` | **M** maximum steps per episode |
| `--num_experts` | `3` | Phase heads in MultiHeadActor |
| `--sigma` | `0.2` | Gaussian width of phase responsibility curves |
| `--env_batch` | `96` | Parallel environments |
| `--batch_size` | `96` | Training batch size |
| `--warmup` | `400` | Steps before policy updates begin |
| `--train_times` | `2000000` | Total training steps |
| `--resume` | `None` | Path to checkpoint folder (moe_actor.pkl + moe_critic.pkl) |
| `--output` | `./model` | Directory for saved checkpoints |
| `--debug` | flag | Print per-episode stats |
| `--seed` | `1234` | Random seed |

---

## Evaluation

```bash
# CelebA, 5 strokes
python evaluate.py \
    --model ./model/MoEPaint-run1 \
    --dataset celeba \
    --num_strokes 5 \
    --max_step 40

# CUB-200 Birds, 8 strokes
python evaluate.py \
    --model ./model/MoEPaint-run2 \
    --dataset cub200 \
    --num_strokes 8 \
    --max_step 60 \
    --num_experts 4

# Side-by-side comparison of two models
python evaluate.py \
    --model ./model/MoEPaint-run1 \
    --compare_model ./model/MoEPaint-run2 \
    --dataset celeba \
    --num_strokes 5 \
    --max_step 40
```

Outputs:
- `results/metrics_summary.csv` — per-image PSNR / SSIM / L2 / MAE / EdgeMSE
- `results/metrics_aggregate.csv` — mean ± std
- `results/metrics_aggregate.txt` — LaTeX-ready table
- `results/qualitative/img_<id>.png` — target | canvas | error panels

---

## How `num_strokes` flows through the system

```
--num_strokes N
    │
    ├─► fastenv (multi.py)
    │       └─► Paint(num_strokes=N)               (env.py)
    │               └─► action_space = N * 13
    │
    └─► MoEDDPG(num_strokes=N)
            ├─► MultiHeadActor(num_outputs = N*13)
            ├─► decode(x, canvas, num_strokes=N)   (applies N strokes)
            └─► noise_action — infers N from action.shape[1] // 13
```

The **critic** (`ResNet_wobn`) and **observation** (7 channels: canvas 3 +
gt 3 + step 1) are **unchanged** by N — only the actor output dimension
and the decode loop scale with N.

---

## Log directories

Each run gets a unique log directory encoding dataset + N + M:
```
../train_log/{dataset}_n{N}_m{M}_experts{K}_{cwd}/
```
e.g. `../train_log/cub200_n8_m60_experts4_myproject/`

This makes it easy to compare runs with different configurations in
TensorBoard:
```bash
tensorboard --logdir ../train_log/
```

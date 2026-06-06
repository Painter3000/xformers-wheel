# xformers — Pre-built Wheel for PyTorch 2.8.0 + CUDA 12.8 (Blackwell)

## Why does this exist?

In May 2026, Hugging Face migrated ZeroGPU Spaces from NVIDIA H200 to the
**NVIDIA RTX Pro 6000 Blackwell** (sm_120, CUDA 12.8). This broke hundreds
of Spaces that rely on `xformers` for memory-efficient attention.

The problem is twofold:
1. Existing PyPI wheels for xformers (e.g. 0.0.29) were built against
   **older PyTorch versions** (2.5.1) and pull in a downgrade of torch
   when installed — breaking torch 2.8.0 environments silently.
2. ZeroGPU Spaces use `python:3.10` as base image —
   **no nvcc available at build time**, so source builds fail.

This repository provides a freshly compiled wheel for:
- Python 3.10
- PyTorch 2.8.0
- CUDA 12.8 (cu128)
- Linux x86_64
- CUDA architectures: sm_70, sm_75, sm_80, sm_86, sm_90, sm_120 (Blackwell)

## How to use on Hugging Face ZeroGPU Spaces

In your `app.py`:

```python
import subprocess
import sys

try:
    import xformers
    print("✅ xformers already installed")
except ImportError:
    print("⚙️ Installing xformers...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-deps",
        "https://github.com/Painter3000/xformers-wheel/releases/download/v1.0.0/xformers-0.0.32-cp310-cp310-linux_x86_64.whl"
    ])
    import xformers
    print("✅ xformers installed")
```

> ⚠️ The `--no-deps` flag is important! It prevents pip from
> trying to install a mismatched torch version.

## How the wheel was built

Built via GitHub Actions using:
- Base image: `nvidia/cuda:12.8.0-devel-ubuntu22.04`
- Python 3.10 explicitly installed
- PyTorch 2.8.0 from `https://download.pytorch.org/whl/cu128`
- Source: `git+https://github.com/facebookresearch/xformers.git@v0.0.32`
- `TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;12.0"`
- `FORCE_CUDA=1`
- `--no-build-isolation` to use the pre-installed torch

## Related repository

> 👉 Also check out the companion wheel for `diff-gaussian-rasterization`:
> [Painter3000/gaussian-wheels](https://github.com/Painter3000/gaussian-wheels)
> — the first publicly available pre-built wheel for
> `diff-gaussian-rasterization` compatible with Blackwell + torch 2.8.0.

## Background

This issue affects all Spaces using CUDA extensions that require
compilation, including `xformers`, `diff-gaussian-rasterization`,
`simple-knn`, and similar packages from the 3D Gaussian Splatting
and diffusion model ecosystem.

Greetings to the ~50-200 people worldwide who are dealing with exactly
this problem right now. You're not alone! 👋😄

## Related issues
- https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960
- https://github.com/Painter3000/gaussian-wheels

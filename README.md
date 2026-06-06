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
- CUDA architecture: **sm_120 (Blackwell only)**

## ✅ Tested & Verified

**Tested on Hugging Face ZeroGPU with FaceLift** — ✅ Works perfectly!
- `sparse_mv_attention` optimized for Blackwell
- Consistent multi-view 3D reconstruction
- All 6 camera angles render correctly with consistent geometry

## ⚠️ Update (Juni 2026)

Since xformers **0.0.35+**, Facebook Research provides official pre-built wheels
for PyTorch 2.8.0+ on PyPI. Before using this wheel, check if the official version
works for your use case:

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu128
```

**Use this wheel if:**
- The official wheel pulls in a wrong torch version as dependency
- You need explicit `sm_120` (Blackwell) optimization
- You are on HF ZeroGPU and need `--no-deps` installation

## Alternatives

| Option | When to use |
|--------|------------|
| Official PyPI wheel (0.0.35+) | torch 2.10.0+ environments |
| This wheel (0.0.33, sm_120) | torch 2.8.0 + Blackwell ZeroGPU |

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
        "https://github.com/Painter3000/xformers-wheel/releases/download/v0.0.33/xformers-0.0.33+ac00641.d20260606-cp39-abi3-linux_x86_64.whl"
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
- `TORCH_CUDA_ARCH_LIST="12.0"` (Blackwell only)
- `FORCE_CUDA=1`
- `--no-build-isolation` to use the pre-installed torch

**Wheel details:**
- `xformers-0.0.33+ac00641.d20260606-cp39-abi3-linux_x86_64.whl`
- **cp39-abi3**: Compatible with Python 3.9+ (stable ABI)
- Built from commit `ac0064145d87ae73e65bf93d982af64257c8a2af` (v0.0.32 tag)

> ⚠️ **Why only sm_120?**
> Building for multiple CUDA architectures simultaneously (sm_70, sm_75, sm_80, sm_86, sm_90, sm_120) 
> causes **Out of Memory (OOM)** errors on GitHub Actions runners (Exit code 137). 
> Since ZeroGPU Spaces use RTX Pro 6000 Blackwell (sm_120), this wheel is optimized 
> for that specific architecture to avoid compilation failures.

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

**Greetings to the ~50-200 people worldwide who are dealing with exactly
this problem right now. The solution exists now!** 👋😄

## Related issues
- https://discuss.huggingface.co/t/nvidia-rtx-pro-6000-instead-of-h200-for-zerogpu/175960
- https://github.com/Painter3000/gaussian-wheels

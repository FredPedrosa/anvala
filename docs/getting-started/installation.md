# Installation & GPU Setup

## Requirements

- **Python** `>= 3.11`
- Core dependencies (installed automatically):
  - `numpy >= 1.26`
  - `scipy >= 1.12`
  - `pandas >= 2.0`
  - `torch >= 2.2`
  - `igraph >= 0.11`

## Standard Installation (PyPI)

```bash
pip install anvala
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/FredPedrosa/anvala.git
```

## Enabling CUDA GPU Acceleration

`anvala` uses a **hybrid CPU--GPU architecture**:

- **Single-model point estimation** (`cfa`, `sem`, `cfa_multigroup`, `efa`, `polychoric`) runs in double-precision (`float64`) on the **CPU** to avoid host-to-device transfer and kernel-launch overhead while preserving exact numerical parity with R.
- **Resampling and batched tensor optimization** (`bootstrap_ci` and `boot_ega`) automatically dispatch 3D tensor batches to **CUDA-enabled GPUs** when available (`device="auto"`), with transparent fallback to vectorized CPU execution when CUDA is absent.

To verify whether PyTorch detects your NVIDIA GPU:

```python
import torch
import anvala as av

print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device     :", torch.cuda.get_device_name(0))
```

If `torch.cuda.is_available()` returns `False` on a machine with an NVIDIA GPU, install a CUDA-enabled PyTorch wheel matching your CUDA toolkit from [pytorch.org](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Running `anvala` on Google Colab (Free GPU)

Every tutorial in this documentation includes an **Open in Colab** badge. To test GPU-accelerated `bootstrap_ci` and `boot_ega` without local setup:

1. Click any **Open in Colab** badge.
2. In Colab, open **Runtime -> Change runtime type** and select **T4 GPU**.
3. Run the first cell (`pip install -q anvala`).

!!! tip "Controlling Device Selection"
    Both `bootstrap_ci(..., device="auto")` and `boot_ega(..., device="auto")` accept `"auto"`, `"cuda"`, or `"cpu"`. Note that `boot_ega` uses the CUDA batched ADMM solver for continuous (`corr="pearson"`, `corr="spearman"`, or `corr="auto"` on continuous data), and automatically falls back to multi-threaded CPU execution when `corr="polychoric"`.

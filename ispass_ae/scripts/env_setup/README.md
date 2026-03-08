# 🛠️ Environment Setup

Three Python virtual environments are required because different model families have incompatible dependencies. `mamba_ssm` needs custom CUDA kernels; Transformer-only models need only the HuggingFace stack.

> All environments use **Python 3.10**.

---

## Contents

- [Environment 1 — Transformers](#environment-1--transformers)
- [Environment 2 — Mamba Models](#environment-2--mamba-models-mamba_ssm)
- [Environment 3 — Falcon-H1 / Hybrid Models](#environment-3--falcon-h1--hybrid-models-torch_falcon_ispass)
- [Jetson Nano Orin (aarch64)](#jetson-nano-orin--jetpack-62-aarch64)
- [Quick Verification](#quick-verification)

---

## Environment 1 — Transformers

**Used for:** Qwen2.5, LLaMA-3.2, TinyLlama, GPT-Neo checkpoints.

```bash
python3 -m venv ~/.venvs/torch_transformers_ispass
source ~/.venvs/torch_transformers_ispass/bin/activate
pip install --upgrade pip
# Check your CUDA version with nvcc --version (here: 12.4) and 
# install the appropriate PyTorch version from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.52.3 accelerate pandas datasets matplotlib numpy
```

#### ▶ Activate
```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
```

---

## Environment 2 — Mamba Models (`mamba_ssm`)

**Used for:** `state-spaces/mamba-*` and `state-spaces/mamba2-*` checkpoints that require the compiled `mamba_ssm` and `causal-conv1d` CUDA kernels.

> **Why pinned wheels?**
> `mamba_ssm` and `causal_conv1d` contain fused CUDA kernels that must be compiled against a specific `(CUDA · PyTorch · cxx11-ABI · Python)` tuple.
> The wheels below target **CUDA 12.x · PyTorch 2.6 · cxx11 ABI=False · Python 3.10**.
> A mismatched build causes silent numerical errors or import failures.
>
> - **`mamba_ssm`** — custom CUDA kernels for the selective scan (core SSM op); exposes `MambaLMHeadModel`.
> - **`causal_conv1d`** — fast CUDA causal 1-D convolution; hard dependency of `mamba_ssm`.
>
> Browse available wheels:
> - <https://github.com/state-spaces/mamba/releases/tag/v2.2.4>
> - <https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.0.post8>

```bash
python3 -m venv ~/.venvs/torch_ssm_ispass
source ~/.venvs/torch_ssm_ispass/bin/activate
pip install --upgrade pip
# Check your CUDA version with nvcc --version (here: 12.4) and
# install the appropriate PyTorch version from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.52.3 accelerate pandas datasets matplotlib numpy
# Pre-built wheels pinned to CUDA 12, PyTorch 2.6, cxx11 ABI=False, Python 3.10
# Browse available wheels at:
#   https://github.com/state-spaces/mamba/releases/tag/v2.2.4
#   https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.0.post8
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### ▶ Activate
```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
```

---

## Environment 3 — Falcon-H1 / Hybrid Models (`torch_falcon_ispass`)

**Used for:** `tiiuae/Falcon-H1-*` (Hybrid SSM-Transformer), `Zyphra/Zamba2-*` (Hybrid SSM), and `nvidia/Hymba-*` (Hybrid SSM-Transformer). All models interleave Mamba-2 SSM layers with Transformer attention layers and require the same `mamba_ssm` and `causal_conv1d` CUDA kernels as Environment 2.

```bash
python3 -m venv ~/.venvs/torch_falcon_ispass
source ~/.venvs/torch_falcon_ispass/bin/activate
pip install --upgrade pip
# Check your CUDA version with nvcc --version (here: 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install accelerate pandas datasets matplotlib numpy transformers==4.57.3
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Hymba — Additional Packages

`nvidia/Hymba-*` additionally requires FlashAttention and tokenizer utilities:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
pip install sentencepiece==0.2.1 protobuf==6.33.1
```

#### ▶ Activate
```bash
source ~/.venvs/torch_falcon_ispass/bin/activate
```

---

## Jetson Nano Orin — JetPack 6.2 (aarch64)

**Device notes:**
- NVMe storage mounted at `/data`; virtual environments are placed under `/data/.venvs/` to avoid filling the eMMC.
- 16 GB swap configured from the NVMe (see [Mounting Swap — Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/ram-optimization/#mounting-swap)).
- MAXN power mode enabled.

Pre-built wheels for Jetson are sourced from **[https://pypi.jetson-ai-lab.io/jp6/cu126](https://pypi.jetson-ai-lab.io/jp6/cu126)** (JetPack 6.2, CUDA 12.6). Older patch versions of a package are available at the same index.

All environments use **Python 3.10** (ships with JetPack 6.2).

---

### Jetson — Environment 1 — Transformers

**Used for:** Qwen2.5 and other Transformer-only models like TinyLlama, Llama-3.2 checkpoints.

```bash
python3 -m venv /data/.venvs/torch_transformers_ispass --system-site-packages
source /data/.venvs/torch_transformers_ispass/bin/activate
pip install --upgrade pip
# PyTorch 2.8.0 pre-built for JetPack 6.2 / CUDA 12.6 / aarch64
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
pip install transformers==4.52.3 accelerate pandas datasets matplotlib numpy
```

#### Activate

```bash
source /data/.venvs/torch_transformers_ispass/bin/activate
```

---

### Jetson — Environment 2 — Mamba Models (`mamba_ssm`)

**Used for:** `state-spaces/mamba-*` and `state-spaces/mamba2-*` checkpoints.

Pre-built wheels are pinned to **JetPack 6.2 · CUDA 12.6 · PyTorch 2.8 · Python 3.10 · aarch64**.

```bash
python3 -m venv /data/.venvs/torch_ssm_ispass --system-site-packages
source /data/.venvs/torch_ssm_ispass/bin/activate
pip install --upgrade pip
# PyTorch 2.8.0 pre-built for JetPack 6.2 / CUDA 12.6 / aarch64
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
pip install transformers==4.52.3 accelerate pandas datasets matplotlib numpy
# mamba_ssm 2.2.5 and causal_conv1d 1.5.2 — pre-built for JetPack 6.2 / aarch64
# Browse available wheels at https://pypi.jetson-ai-lab.io/jp6/cu126
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/b8e/35eeb4d7f0ada/mamba_ssm-2.2.5-cp310-cp310-linux_aarch64.whl#sha256=b8e35eeb4d7f0ada87235c15db0408cded09863bf6798ac451d0f65a6035b4ba"
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/28a/11e19b7f9fd56/causal_conv1d-1.5.2-cp310-cp310-linux_aarch64.whl#sha256=28a11e19b7f9fd56f17347da18fa31e09ad2ac5e61b8ed5653f069cbe7e5177b"
# triton 3.4.0 — pre-built for JetPack 6.2 / aarch64
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/9da/4bcb8e8f0eba0/triton-3.4.0-cp310-cp310-linux_aarch64.whl#sha256=9da4bcb8e8f0eba00a097ad8c57b26102add499e520d67fb2d5362bebf976ca3"
```

#### Activate

```bash
source /data/.venvs/torch_ssm_ispass/bin/activate
```

---

### Jetson — Environment 3 — Falcon-H1 / Hybrid Models (`torch_falcon_ispass`)

**Used for:** `tiiuae/Falcon-H1-*` (Hybrid SSM-Transformer), `Zyphra/Zamba2-*` (Hybrid SSM), and `nvidia/Hymba-*` (Hybrid SSM-Transformer).

```bash
python3 -m venv /data/.venvs/torch_falcon_ispass --system-site-packages
source /data/.venvs/torch_falcon_ispass/bin/activate
pip install --upgrade pip
# PyTorch 2.8.0 pre-built for JetPack 6.2 / CUDA 12.6 / aarch64
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
pip install accelerate pandas datasets matplotlib numpy transformers==4.57.3
# mamba_ssm 2.2.5 and causal_conv1d 1.5.2 — pre-built for JetPack 6.2 / aarch64
# Browse available wheels at https://pypi.jetson-ai-lab.io/jp6/cu126
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/b8e/35eeb4d7f0ada/mamba_ssm-2.2.5-cp310-cp310-linux_aarch64.whl#sha256=b8e35eeb4d7f0ada87235c15db0408cded09863bf6798ac451d0f65a6035b4ba"
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/28a/11e19b7f9fd56/causal_conv1d-1.5.2-cp310-cp310-linux_aarch64.whl#sha256=28a11e19b7f9fd56f17347da18fa31e09ad2ac5e61b8ed5653f069cbe7e5177b"
# triton 3.4.0 — pre-built for JetPack 6.2 / aarch64
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/9da/4bcb8e8f0eba0/triton-3.4.0-cp310-cp310-linux_aarch64.whl#sha256=9da4bcb8e8f0eba00a097ad8c57b26102add499e520d67fb2d5362bebf976ca3"
```

#### Activate

```bash
source /data/.venvs/torch_falcon_ispass/bin/activate
```

---

## ✅ Quick Verification

After activating any environment, confirm the setup from `<repo_root>/src/`:

```bash
cd <repo_root>/src
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

For Environments 2 and 3, additionally verify the CUDA kernels loaded correctly:

```bash
python -c "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel; print('mamba_ssm ✓')"
python -c "import causal_conv1d; print('causal_conv1d ✓')"
```

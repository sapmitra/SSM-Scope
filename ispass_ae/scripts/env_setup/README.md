# Environment Setup

Separate Python virtual environments are required because different LLMs have unique dependencies. Mamba (`mamba_ssm`) needs CUDA kernels, while Transformer-based models only need the HuggingFace stack.

All environments use **Python 3.10** in our experiments.

---

## Environment 1 — Transformers

**Used for:** Qwen2.5 and other Transformer only models like TinyLlama, Llama-3.2 checkpoints.

```bash
python3 -m venv ~/.venvs/torch_transformers_ispass
source ~/.venvs/torch_transformers_ispass/bin/activate
pip install --upgrade pip
# Check your CUDA version with nvcc --version (here: 12.4) and 
# install the appropriate PyTorch version from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.52.3 accelerate pandas datasets matplotlib numpy
```

### Activate

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
```

---

## Environment 2 — Mamba Models (`mamba_ssm`)

**Used for:** `state-spaces/mamba-*` and `state-spaces/mamba2-*` checkpoints that require the compiled `mamba_ssm` and `causal-conv1d` CUDA kernels.

### Why `mamba_ssm` and `causal_conv1d`?

- **`mamba_ssm`**: Contains the custom CUDA kernels that implement the selective scan (the core SSM operation) and exposes `MambaLMHeadModel`. Pure PyTorch cannot replicate the fused CUDA kernels required for correct and efficient SSM inference — the `state-spaces/mamba*` checkpoints are designed to be loaded exclusively through this library.
- **`causal_conv1d`**: A fast CUDA implementation of the causal 1-D convolution used inside every Mamba layer. It is a hard dependency of `mamba_ssm` and **must** be compiled against the same CUDA toolkit, PyTorch version, and C++ ABI. Using a mismatched build causes silent numerical errors or import failures.

Both packages are distributed as pre-built wheels pinned to a specific `(CUDA, PyTorch, C++ ABI, Python)` tuple — here **CUDA 12.x · PyTorch 2.6 · cxx11 ABI = FALSE · Python 3.10** — which is why the wheel URLs are used directly instead of a plain `pip install mamba-ssm`.

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

### Activate

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
```

---

## Environment 3 — Falcon-H1 Models (`torch_falcon_ispass`)

**Used for:** `tiiuae/Falcon-H1-*` (Hybrid SSM-Transformer) and `Zyphra/Zamba2-*` (Hybrid SSM). Both models interleave Mamba-2 SSM layers with Transformer attention layers and require the same `mamba_ssm` and `causal_conv1d` CUDA kernels as Environment 2.

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

### Activate

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

### Jetson — Environment 3 — Falcon-H1 Models (`torch_falcon_ispass`)

**Used for:** `tiiuae/Falcon-H1-*` (Hybrid SSM-Transformer) and `Zyphra/Zamba2-*` (Hybrid SSM).

```bash
python3 -m venv /data/.venvs/torch_falcon_ispass --system-site-packages
source /data/.venvs/torch_falcon_ispass/bin/activate
pip install --upgrade pip
# PyTorch 2.8.0 pre-built for JetPack 6.2 / CUDA 12.6 / aarch64
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
pip install accelerate pandas datasets matplotlib numpy transformers==4.57.3
# causal_conv1d 1.5.2 — pre-built for JetPack 6.2 / aarch64
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/28a/11e19b7f9fd56/causal_conv1d-1.5.2-cp310-cp310-linux_aarch64.whl#sha256=28a11e19b7f9fd56f17347da18fa31e09ad2ac5e61b8ed5653f069cbe7e5177b"
# triton 3.4.0 — pre-built for JetPack 6.2 / aarch64
pip install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/9da/4bcb8e8f0eba0/triton-3.4.0-cp310-cp310-linux_aarch64.whl#sha256=9da4bcb8e8f0eba00a097ad8c57b26102add499e520d67fb2d5362bebf976ca3"
```

#### Activate

```bash
source /data/.venvs/torch_falcon_ispass/bin/activate
```

---

## Quick Verification

After activating the relevant environment, confirm the setup from inside `src/`:

```bash
cd <repo_root>/src
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

For Environments 2 and 3, additionally verify:

```bash
python -c "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel; print('mamba_ssm OK')"
python -c "import causal_conv1d; print('causal_conv1d OK')"
```

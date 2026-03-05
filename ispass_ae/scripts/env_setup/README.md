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
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Activate

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
```

---

## Quick Verification

After activating the relevant environment, confirm the setup from inside `src/`:

```bash
cd <repo_root>/src
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

For Environment 2, additionally verify:

```bash
python -c "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel; print('mamba_ssm OK')"
python -c "import causal_conv1d; print('causal_conv1d OK')"
```

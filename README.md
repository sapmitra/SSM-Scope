## LLM Long-Context Characterization Framework (ISPASS 2026)
State Space Model and SSM-Transformer Hybrid model characterization on consumer GPU and edge devices for very large context

* This repository contains the source code for the performance characterization framework introduced in the paper, *"Characterizing State Space Model and Hybrid Language Model Performance with Long Context"*. 

* The framework provides comprehensive benchmarking and profiling tools to evaluate Transformers, State Space Models (SSMs), and Hybrid models (such as Qwen2.5, Mamba-2, and Falcon-H1).



## Key Features

* **Computational Performance Tracking**: Measures end-to-end inference metrics across generation stages, including Time to First Token (TTFT), Time per Output Token (TPOT), and overall throughput.
* **Detailed Memory Analysis**: Captures system-level peak GPU memory usage reserved during inference, alongside fine-grained operator-level memory footprints.
* **Operator-Level Profiling**: Generates latency breakdowns to identify execution bottlenecks, separating GEMM, non-GEMM, and novel SSM-specific operators.
* **Energy Consumption Metrics**: Calculates energy usage over time based on power draw statistics, which is critical for edge deployment evaluation.

## Repository Structure

```
src/
├── __init__.py
├── profiling/                  # Core PyTorch profiler engine
│   ├── __init__.py
│   ├── eval.py                 # Operator-level profiler (TTFT, TPOT, energy, shapes)
│   └── power_logger.py         # nvidia-smi power log parser
├── models/                     # Model loaders and profiling entry points
│   ├── __init__.py
│   └── profile_runner.py       # LMProfile, MambaProfile, per-model CLI functions
├── memory/                     # Memory footprint analysis
│   ├── __init__.py
│   ├── mem_footprint.py        # Prefill / decode memory measurement (PyTorch)
│   └── vllm_oom.py             # vLLM OOM boundary sweep
└── visualization/              # Figure generation from profiling CSVs
    ├── __init__.py
    └── gen_figure_data.py      # Operator breakdown plots and summary CSVs
```

## Environment Setup

Two virtual environments are required — see [`ispass_ae/scripts/env_setup/README.md`](ispass_ae/scripts/env_setup/README.md) for full instructions.

All environments use **Python 3.10** in our experiments.

**Environment 1 — Transformers & HuggingFace SSMs** (Qwen2.5, Falcon-H1, Hymba, …)

```bash
python3 -m venv ~/.venvs/torch_transformers_ispass
source ~/.venvs/torch_transformers_ispass/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.52.3 accelerate pandas datasets matplotlib numpy
```

**Environment 2 — Native Mamba** (`mamba_ssm` + `causal_conv1d` CUDA kernels required for `state-spaces/mamba*`)

> `mamba_ssm` provides the fused selective-scan CUDA kernels and `MambaLMHeadModel` loader that the `state-spaces/mamba*` checkpoints depend on. `causal_conv1d` is a hard dependency that must match the same CUDA/PyTorch/ABI build. Pre-built wheels are used to guarantee ABI compatibility (CUDA 12 · PyTorch 2.6 · cxx11 ABI=False · Python 3.10).

```bash
python3 -m venv ~/.venvs/torch_ssm_ispass
source ~/.venvs/torch_ssm_ispass/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.52.3 accelerate pandas datasets matplotlib numpy
# Pre-built wheels pinned to CUDA 12, PyTorch 2.6, cxx11 ABI=False, Python 3.10
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Running the Profiling Framework

All scripts are run from within the `src/` directory:

```bash
# Operator-level profiling (prefill / TTFT)
cd src
python -m models.profile_runner --model_name mamba2 --batch_size 1 --seq_len 1024 --device cuda

# Memory footprint sweep
python -m memory.mem_footprint

# vLLM OOM boundary sweep
python -m memory.vllm_oom

# Figure generation from existing CSVs
python -m visualization.gen_figure_data
```


## Citation

If you use this framework in your research, please cite:

```bibtex
@article{mitra2025characterizing,
  title={Characterizing state space model (ssm) and ssm-transformer hybrid language model performance with long context length},
  author={Mitra, Saptarshi and Karami, Rachid and Xu, Haocheng and Huang, Sitao and Kwon, Hyoukjun},
  journal={arXiv preprint arXiv:2507.12442},
  year={2025}
}

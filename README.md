<div align="center">

# 🔬 SSM-Scope

### *A Characterization Framework for State Space Models & Hybrid LLMs in Long Context*

**ISPASS 2026 — Artifact Evaluation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![arXiv](https://img.shields.io/badge/arXiv-2507.12442-b31b1b.svg)](https://arxiv.org/abs/2507.12442)

</div>

---

> **Paper:** *"Characterizing State Space Model and Hybrid Language Model Performance with Long Context"*
> Saptarshi Mitra, Rachid Karami, Haocheng Xu, Sitao Huang, Hyoukjun Kwon · ISPASS 2026

---

## 🗺️ What Is This?

**SSM-Scope** provides comprehensive benchmarking and profiling tools to evaluate three families of language models across long context lengths on both a desktop GPU (RTX 4090) and an edge device (NVIDIA Jetson Nano Orin):

| Family | Representative Models |
|--------|-----------------------|
| **Transformer** | Qwen2.5-0.5B/1.5B, LLaMA-3.2-1B, Phi-3 |
| **SSM** | Mamba-130m/790m, Mamba2-130m/780m |
| **Hybrid (SSM + Attn)** | Falcon-H1-0.5B/1.5B, Zamba2-1.2B, Hymba-1.5B |


### Key Features

- **Computational Performance Tracking** — TTFT, TPOT, and overall throughput across generation stages.
- **Detailed Memory Analysis** — peak GPU memory decomposed into model weights, activations, and KV cache.
- **Operator-Level Profiling** — latency breakdowns separating GEMM, non-GEMM, and SSM-specific kernels.
- **Energy Consumption Metrics** — joules-per-prefill from `nvidia-smi` power logs, critical for edge evaluation.

---

## 📊 Paper Figures at a Glance

### 🔷 Motivation & Inference Performance

| Figure | What It Shows | README | Notebook |
|--------|--------------|--------|----------|
| **Fig 1** | TTFT & TPOT crossover: Qwen2.5-0.5B vs Mamba2-780m at short/long context | [Fig_1/README](ispass_ae/scripts/paper_figures/Fig_1/README.md) | [`plotting_intro_ttft_tpot.ipynb`](ispass_ae/notebooks/plotting_intro_ttft_tpot.ipynb) |
| **Fig 3** | Accuracy vs TTFT: Transformer vs SSM vs Hybrid (~1.5B models) | [Fig_3/README](ispass_ae/scripts/paper_figures/Fig_3/README.md) | [`plotting_accuracy_ttft.ipynb`](ispass_ae/notebooks/plotting_accuracy_ttft.ipynb) |
| **Fig 6a** | Prefill energy consumption vs sequence length | [Fig_6a/README](ispass_ae/scripts/paper_figures/Fig_6a/README.md) | [`plotting_energy_seq.ipynb`](ispass_ae/notebooks/plotting_energy_seq.ipynb) |
| **Fig 6b** | Overall throughput across sequence lengths | [Fig_6b/README](ispass_ae/scripts/paper_figures/Fig_6b/README.md) | [`plotting_throughput_seq.ipynb`](ispass_ae/notebooks/plotting_throughput_seq.ipynb) |

### 🔷 Memory Footprint

| Figure | What It Shows | README | Notebook |
|--------|--------------|--------|----------|
| **Fig 5a** | GPU memory footprint vs sequence length — RTX 4090 GPU | [Fig_5a/README](ispass_ae/scripts/paper_figures/Fig_5a/README.md) | [`plotting_mem_footprint.ipynb`](ispass_ae/notebooks/plotting_mem_footprint.ipynb) |
| **Fig 5b** | GPU memory footprint vs sequence length — NVIDIA Jetson Nano Orin | [Fig_5b/README](ispass_ae/scripts/paper_figures/Fig_5b/README.md) | [`plotting_mem_footprint_jetson.ipynb`](ispass_ae/notebooks/plotting_mem_footprint_jetson.ipynb) |

### 🔷 Operator-level Performance Breakdown

| Figure | What It Shows | README | Notebook |
|--------|--------------|--------|----------|
| **Fig 7** | Op breakdown: Mamba-130m vs Mamba2-130m (desktop) | [Fig_7/README](ispass_ae/scripts/paper_figures/Fig_7/README.md) | [`plotting_ops.ipynb`](ispass_ae/notebooks/plotting_ops.ipynb) |
| **Fig 8** | Op breakdown: Hymba-1.5B vs Zamba2-1.2B (desktop) | [Fig_8/README](ispass_ae/scripts/paper_figures/Fig_8/README.md) | [`ssm_hybrid_op_breakdown.ipynb`](ispass_ae/notebooks/ssm_hybrid_op_breakdown.ipynb) |
| **Fig 9a** | Op breakdown: Mamba-130m vs Mamba2-130m (Jetson Nano Orin) | [Fig_9a/README](ispass_ae/scripts/paper_figures/Fig_9a/README.md) | [`plotting_ops_jetson.ipynb`](ispass_ae/notebooks/plotting_ops_jetson.ipynb) |
| **Fig 9b** | Cross-device op breakdown: all model families (desktop vs Jetson) | [Fig_9b/README](ispass_ae/scripts/paper_figures/Fig_9b/README.md) | [`plotting_ops_cross_device.ipynb`](ispass_ae/notebooks/plotting_ops_cross_device.ipynb) |

---

## 🗂️ Repository Structure

```
SSM-characterization/
├── src/                          # Core profiling framework
│   ├── profiling/                #   PyTorch profiler engine (TTFT, TPOT, energy, op shapes)
│   ├── models/                   #   Model loaders and per-model profiling entry points
│   ├── memory/                   #   Memory footprint & vLLM OOM sweep
│   └── visualization/            #   Figure generation from profiling CSVs
│
├── profile_data/                 # Pre-collected profiling CSVs — desktop GPU
├── profile_data_jetson/          # Pre-collected profiling CSVs — Jetson Nano Orin
│
└── ispass_ae/
    ├── notebooks/                # Interactive Jupyter notebooks (plot from pre-collected data)
    └── scripts/
        ├── env_setup/            # Virtual environment setup instructions
        │   └── README.md  ◄──── START HERE for environment setup
        └── paper_figures/        # End-to-end scripts per figure
            ├── Fig_1/  Fig_3/  Fig_5a/  Fig_5b/
            └── Fig_6a/ Fig_6b/ Fig_7/   Fig_8/  Fig_9a/  Fig_9b/
```

---

## ⚡ Quick Start

### 1 — Set Up Environments

Three virtual environments cover all models. 👉 See **[`ispass_ae/scripts/env_setup/README.md`](ispass_ae/scripts/env_setup/README.md)** for full instructions.

| venv | Models |
|------|--------|
| `torch_transformers_ispass` | Qwen2.5, LLaMA-3.2, TinyLlama, GPT-Neo |
| `torch_ssm_ispass` | Mamba-130m/790m, Mamba2-130m/780m |
| `torch_falcon_ispass` | Falcon-H1, Zamba2, Hymba |

### 2 — Reproduce Any Figure

Every figure has a one-command end-to-end script:

```bash
# Example: reproduce Figure 7 (SSM op-breakdown)
bash ispass_ae/scripts/paper_figures/Fig_7/gen_fig7.sh
```
Or
Follow the instructions in the target figure's `README.md` for more details and tips.

Or open any notebook in [`ispass_ae/notebooks/`](ispass_ae/notebooks/) to plot directly from pre-collected data — **no GPU required**.

---

## 🖥️ Hardware Requirements

| Component | Desktop (required) | Jetson (Figs 5b, 9a, 9b) |
|-----------|--------------------|--------------------------|
| GPU | RTX 4090 GPU, ≥ 24 GB VRAM | Jetson Nano Orin (8 GB unified) |
| CUDA | 12.x | 12.6 (JetPack 6.2) |
| Storage | ~50 GB (model weights + logs) | NVMe recommended |
| Python | 3.10 | 3.10 |


---

## 📦 Pre-collected Data

Raw profiling CSVs are included — figures can be reproduced without re-running inference:

- **Desktop GPU:** [`profile_data/`](profile_data/)
- **Jetson Nano Orin:** [`profile_data_jetson/`](profile_data_jetson/)

---

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

---


## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{mitra2025characterizing,
  title   = {Characterizing State Space Model (SSM) and SSM-Transformer Hybrid
             Language Model Performance with Long Context Length},
  author  = {Mitra, Saptarshi and Karami, Rachid and Xu, Haocheng and
             Huang, Sitao and Kwon, Hyoukjun},
  journal = {arXiv preprint arXiv:2507.12442},
  year    = {2025}
}
```

---

# Figure 5b — GPU Memory Footprint: Transformer vs. SSM Models on NVIDIA Jetson

> **Tip:** The interactive notebook at [`plotting_mem_footprint_jetson.ipynb`](../../../notebooks/plotting_mem_footprint_jetson.ipynb) can regenerate the figure directly from a pre-collected CSV without running any profiling.

> **Quick start:** Run the script from the repo root (or from this directory) to collect all data and generate the figure in one step:
> ```bash
> # from repo root
> chmod +x ispass_ae/scripts/paper_figures/Fig_5b/gen_fig5b.sh
> bash ispass_ae/scripts/paper_figures/Fig_5b/gen_fig5b.sh
>
> # or from this directory
> chmod +x gen_fig5b.sh
> bash gen_fig5b.sh
> ```
> The script activates the correct venvs automatically and writes the output PNG to this directory.

---

This directory contains the scripts to **collect the raw data** and **reproduce Figure 5b** from the paper.

Figure 5b shows the **GPU memory footprint during the prefill phase** across increasing sequence lengths for six language model architectures on an **NVIDIA Jetson** platform (8 GB unified memory).  
Sequence lengths are shorter than Figure 5a due to the tighter memory budget.

| Model | Type | Size | Venv |
|-------|------|------|------|
| Qwen2.5-0.5B-Instruct | Transformer | ~0.5 B | `torch_transformers_ispass` |
| Llama-3.2-1B-Instruct | Transformer | ~1 B | `torch_transformers_ispass` |
| Mamba-790m | SSM (Mamba-1) | ~790 M | `torch_ssm_ispass` |
| Mamba2-780m | SSM (Mamba-2) | ~780 M | `torch_ssm_ispass` |
| Falcon-H1-0.5B-Base | Hybrid SSM | ~0.5 B | `torch_falcon_ispass` |
| Zamba2-1.2B-Instruct-v2 | Hybrid SSM | ~1.2 B | `torch_falcon_ispass` |

Memory is decomposed into three stacked components:
- **Model Size** — static parameter memory (constant across sequence lengths)
- **Activation Memory** — intermediate tensors during the forward pass
- **KV Cache** — key-value cache for attention layers (zero for pure SSMs)

---

## Files

| File | Purpose |
|------|---------|
| `gen_fig5b.sh` | End-to-end bash script — collects data and plots the figure in one command |
| `collect_fig5b_data.py` | Profiles one model at one sequence length; appends a row to `src/memory/memory_footprints_jetson.csv` |
| `plot_fig5b.py` | Reads the CSV and generates the publication-quality PNG (`memory_footprint_jetson_ispass.png`) |

---

## Sequence Lengths Profiled

| Model key | Sequence lengths |
|-----------|-----------------|
| `qwen` | 256, 512, 1024, 2048, 4096, 8192, 16384 |
| `llama3_2` | 256, 512, 1024, 2048, 4096, 8192 |
| `mamba_790m` | 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768 |
| `mamba2_780m` | 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768 |
| `falcon_h1` | 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768 |
| `zamba2` | 256, 512, 1024, 2048, 4096, 8192 |

---

## Step 1 — Collect Data

All `python` commands are run from `<repo_root>/src/`.  
The CSV is written (appended) to `src/memory/memory_footprints_jetson.csv`.

### 1a. Transformer models — `torch_transformers_ispass` venv

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

# Qwen2.5-0.5B-Instruct
for seq_len in 256 512 1024 2048 4096 8192 16384; do
    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \
        --model qwen --seq_len $seq_len --device cuda
done

# Llama-3.2-1B-Instruct
for seq_len in 256 512 1024 2048 4096 8192; do
    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \
        --model llama3_2 --seq_len $seq_len --device cuda
done
```

### 1b. Mamba models — `torch_ssm_ispass` venv

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>/src

# Mamba-790m
for seq_len in 256 512 1024 2048 4096 8192 16384 24576 32768; do
    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \
        --model mamba_790m --seq_len $seq_len --device cuda
done

# Mamba2-780m
for seq_len in 256 512 1024 2048 4096 8192 16384 24576 32768; do
    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \
        --model mamba2_780m --seq_len $seq_len --device cuda
done
```

### 1c. Falcon-H1-0.5B — `torch_falcon_ispass` venv

```bash
source ~/.venvs/torch_falcon_ispass/bin/activate
cd <repo_root>/src

for seq_len in 256 512 1024 2048 4096 8192 16384 24576 32768; do
    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \
        --model falcon_h1 --seq_len $seq_len --device cuda
done
```

### 1d. Zamba2 — `torch_falcon_ispass` venv

```bash
source ~/.venvs/torch_falcon_ispass/bin/activate
cd <repo_root>/src

for seq_len in 256 512 1024 2048 4096 8192; do
    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \
        --model zamba2 --seq_len $seq_len --device cuda
done
```

Output CSV:

| CSV | Contents |
|-----|---------|
| `src/memory/memory_footprints_jetson.csv` | `model_name`, `seq_len`, `model_size_mb`, `activation_memory_mb`, `kv_cache_mb`, `reserved_memory_mb`, `total_memory_mb` |

---

## Step 2 — Generate Figure

From anywhere with a venv that has `matplotlib` and `pandas` (e.g. `torch_transformers_ispass`):

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>

python ispass_ae/scripts/paper_figures/Fig_5b/plot_fig5b.py \
    --csv_path src/memory/memory_footprints_jetson.csv \
    --out_dir  ispass_ae/scripts/paper_figures/Fig_5b
```

Output file:

| File | Description |
|------|-------------|
| `memory_footprint_jetson_ispass.png` | Publication-quality stacked-bar chart (300 DPI) |

---

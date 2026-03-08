# Figure 5a — GPU Memory Footprint: Transformer vs. SSM Models

> 💡 **Tip:** The interactive notebook at [`plotting_mem_footprint.ipynb`](../../../notebooks/plotting_mem_footprint.ipynb) can regenerate the figure directly from a pre-collected CSV without running any profiling.

> ⚡ **Quick start:** Run the script from the repo root (or from this directory) to collect all data and generate the figure in one step:
> ```bash
> # from repo root
> chmod +x ispass_ae/scripts/paper_figures/Fig_5a/gen_fig5a.sh
> bash ispass_ae/scripts/paper_figures/Fig_5a/gen_fig5a.sh
>
> # or from this directory
> chmod +x gen_fig5a.sh
> bash gen_fig5a.sh
> ```
> The script activates the correct venvs automatically and writes the output PNG to this directory.

---

This directory contains the scripts to **collect the raw data** and **reproduce Figure 5a** from the paper.

Figure 5a shows the **GPU memory footprint during the prefill phase** across increasing sequence lengths for seven language model architectures on an NVIDIA RTX 4090 (24 GB VRAM):

| Model | Type | Size | Venv |
|-------|------|------|------|
| Phi-3-mini | Transformer | ~3.8 B | `torch_transformers_ispass` |
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
| `gen_fig5a.sh` | End-to-end bash script — collects data and plots the figure in one command |
| `collect_fig5a_data.py` | Profiles one model at one sequence length; appends a row to `src/memory/memory_footprints.csv` |
| `plot_fig5a.py` | Reads the CSV and generates the publication-quality PNG (`memory_footprint_rtx_ispass.png`) |

---

## Sequence Lengths Profiled

| Model key | Sequence lengths |
|-----------|-----------------|
| `phi3` | 1024, 2048, 4096 |
| `qwen` | 1024 … 57344 (10 points) |
| `llama3_2` | 1024 … 65536 (11 points) |
| `mamba_790m` | 1024 … 220000 (19 points) |
| `mamba2_780m` | 1024 … 220000 (19 points) |
| `falcon_h1` | 1024 … 163840 (17 points) |
| `zamba2` | 1024 … 49152 (9 points) |

---

## Step 1 — Collect Data

All `python` commands are run from `<repo_root>/src/`.  
The CSV is written (appended) to `src/memory/memory_footprints.csv`.

### 1a. Transformer models — `torch_transformers_ispass` venv

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

# Phi-3-mini
for seq_len in 1024 2048 4096; do
    python ../ispass_ae/scripts/paper_figures/Fig_5a/collect_fig5a_data.py \
        --model phi3 --seq_len $seq_len --device cuda
done

# Qwen2.5-0.5B-Instruct
for seq_len in 1024 2048 4096 8192 16384 24576 32768 40960 49152 57344; do
    python ../ispass_ae/scripts/paper_figures/Fig_5a/collect_fig5a_data.py \
        --model qwen --seq_len $seq_len --device cuda
done

# Llama-3.2-1B-Instruct
for seq_len in 1024 2048 4096 8192 16384 24576 32768 40960 49152 57344 65536; do
    python ../ispass_ae/scripts/paper_figures/Fig_5a/collect_fig5a_data.py \
        --model llama3_2 --seq_len $seq_len --device cuda
done
```

### 1b. Mamba models — `torch_ssm_ispass` venv

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>/src

# Mamba-790m
for seq_len in 1024 2048 4096 8192 16384 24576 32768 40960 49152 57344 \
               65536 81920 98304 114688 131072 147456 163840 180224 220000; do
    python ../ispass_ae/scripts/paper_figures/Fig_5a/collect_fig5a_data.py \
        --model mamba_790m --seq_len $seq_len --device cuda
done

# Mamba2-780m
for seq_len in 1024 2048 4096 8192 16384 24576 32768 40960 49152 57344 \
               65536 81920 98304 114688 131072 147456 163840 180224 220000; do
    python ../ispass_ae/scripts/paper_figures/Fig_5a/collect_fig5a_data.py \
        --model mamba2_780m --seq_len $seq_len --device cuda
done
```

### 1c. Falcon-H1-0.5B — `torch_falcon_ispass` venv

```bash
source ~/.venvs/torch_falcon_ispass/bin/activate
cd <repo_root>/src

for seq_len in 1024 2048 4096 8192 16384 24576 32768 40960 49152 57344 \
               65536 81920 98304 114688 131072 147456 163840; do
    python ../ispass_ae/scripts/paper_figures/Fig_5a/collect_fig5a_data.py \
        --model falcon_h1 --seq_len $seq_len --device cuda
done
```

### 1d. Zamba2 — `torch_falcon_ispass` venv

```bash
source ~/.venvs/torch_falcon_ispass/bin/activate
cd <repo_root>/src

for seq_len in 1024 2048 4096 8192 16384 24576 32768 40960 49152; do
    python ../ispass_ae/scripts/paper_figures/Fig_5a/collect_fig5a_data.py \
        --model zamba2 --seq_len $seq_len --device cuda
done
```

Output CSV is at (relative to `src/`):

| CSV | Contents |
|-----|---------|
| `memory/memory_footprints.csv` | `model_name`, `seq_len`, `model_size_mb`, `activation_memory_mb`, `kv_cache_mb`, `reserved_memory_mb`, `total_memory_mb` |

---

## Step 2 — Generate Figure

From anywhere with a venv that has `matplotlib` and `pandas` (e.g. `torch_transformers_ispass`):

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>

python ispass_ae/scripts/paper_figures/Fig_5a/plot_fig5a.py \
    --csv_path src/memory/memory_footprints.csv \
    --out_dir  ispass_ae/scripts/paper_figures/Fig_5a
```

Output file:

| File | Description |
|------|-------------|
| `memory_footprint_rtx_ispass.png` | Publication-quality stacked-bar chart (300 DPI) |

---
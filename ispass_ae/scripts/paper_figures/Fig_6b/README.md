# Figure 6b — Overall Throughput: Qwen2.5-0.5B vs Mamba2-780m vs Falcon-H1 0.5B

> 💡 **Tip:** The interactive notebook at [`plotting_throughput_seq.ipynb`](../../../notebooks/plotting_throughput_seq.ipynb) can regenerate the figure directly from hard-coded paper values without running any inference.

> ⚡ **Quick start:** Run the script from the repo root (or from this directory) to collect all data and generate the figure in one step:
> ```bash
> # from repo root
> chmod +x ispass_ae/scripts/paper_figures/Fig_6b/gen_fig6b.sh
> bash ispass_ae/scripts/paper_figures/Fig_6b/gen_fig6b.sh
>
> # or from this directory
> chmod +x gen_fig6b.sh
> bash gen_fig6b.sh
> ```
> The script activates the correct venvs automatically and writes the output PNGs to this directory.

#### For detailed instructions on how to set up the environments, collect the raw data, and generate the figure, see the rest of this README.
---

This directory contains the scripts to **collect the raw data** and **reproduce Figure 6b** from the paper.

Figure 6b shows **Overall Throughput** (total tokens / total inference time, in tokens/s) for:

- **Qwen2.5-0.5B-Instruct** — Transformer
- **Mamba2-780m** — State Space Model (SSM)
- **Falcon-H1 0.5B** — Hybrid SSM

across seven sequence lengths:

| Sequence Length | Output tokens |
|----------------|---------------|
| 1024           | 256           |
| 2048           | 256           |
| 4096           | 256           |
| 8192           | 256           |
| 16384          | 256           |
| 24576          | 256           |
| 32768          | 256           |

Overall throughput = `(input_seq_length + output_tokens) / total_time_seconds`

---

## Files

| File | Purpose |
|------|---------|
| `gen_fig6b.sh` | End-to-end bash script — collects data and plots the figure in one command |
| `collect_fig6b_data.py` | Runs generation profiling for one model / one sequence length and writes results to CSV |
| `plot_fig6b.py` | Reads the CSV and generates the publication-quality PNGs |

---

## Step 1 — Collect Data

All commands are run from `<repo_root>/src/`.

### 1a. Qwen2.5-0.5B-Instruct — Transformer venv

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

for SEQ in 1024 2048 4096 8192 16384 24576 32768; do
    python ../ispass_ae/scripts/paper_figures/Fig_6b/collect_fig6b_data.py \
        --model qwen --seq_len $SEQ --max_new_tokens 256 --device cuda
done
```

### 1b. Mamba2-780m — Mamba venv

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>/src

for SEQ in 1024 2048 4096 8192 16384 24576 32768; do
    python ../ispass_ae/scripts/paper_figures/Fig_6b/collect_fig6b_data.py \
        --model mamba2 --seq_len $SEQ --max_new_tokens 256 --device cuda
done
```

### 1c. Falcon-H1-0.5B-Base — Transformer venv

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

for SEQ in 1024 2048 4096 8192 16384 24576 32768; do
    python ../ispass_ae/scripts/paper_figures/Fig_6b/collect_fig6b_data.py \
        --model falcon --seq_len $SEQ --max_new_tokens 256 --device cuda
done
```

Output CSV (relative to `src/`):

| CSV | Contents |
|-----|----------|
| `throughput_logs/generation_times.csv` | TTFT, TPOT, decode_time, total_time, throughput per run |

---

## Step 2 — Plot Figure 6b

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>

python ispass_ae/scripts/paper_figures/Fig_6b/plot_fig6b.py \
    --throughput_csv src/throughput_logs/generation_times.csv \
    --out_dir  ispass_ae/scripts/paper_figures/Fig_6b
```

Two PNG files are produced in `out_dir`:

| File | Description |
|------|-------------|
| `overall_throughput_comparison.png` | Publication-quality figure (no axis labels, 300 DPI) |
| `overall_throughput_annotated.png`  | Annotated version with exact values on each bar (150 DPI) |

> 💡 **Tip:** If the CSV is not present, `plot_fig6b.py` falls back to hard-coded paper values so the figure can always be regenerated without running inference.

---
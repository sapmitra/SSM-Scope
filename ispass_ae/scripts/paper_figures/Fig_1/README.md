# Figure 1 — TTFT & TPOT: Qwen2.5-0.5B vs Mamba2-780m

> **Tip:** The interactive notebook [`ispass_ae/notebooks/plotting_intro_ttft_tpot.ipynb`](ispass_ae/notebooks/plotting_intro_ttft_tpot.ipynb) can regenerate the figure directly from hard-coded paper values without running any inference.

> **Quick start:** Run the single top-level script from the repo root to collect all data and generate the figure in one step:
> ```bash
> bash gen_fig1.sh
> ```
> The script activates the correct venvs automatically and writes the output PNGs to this directory.

#### For detailed instructions on how to set up the environments, collect the raw data, and generate the figure, see the rest of this README.
---

This directory contains the scripts to **collect the raw data** and **reproduce Figure 1** from the paper.

Figure 1 shows Time to First Token (TTFT) and Time Per Output Token (TPOT) for:

- **Qwen2.5-0.5B-Instruct** — Transformer
- **Mamba2-780m** — State Space Model (SSM)

across two inference scenarios:

| Scenario      | Input tokens | Output tokens |
|---------------|-------------|---------------|
| Short context | 1024       | 256           |
| Long context  | 32768      | 256           |

---

## Files

| File | Purpose |
|------|---------|
| `collect_fig1_data.py` | Runs inference profiling for one model / one context length and writes TTFT + TPOT to CSV |
| `plot_fig1.py` | Reads the CSVs and generates the publication-quality PNG (`intro_ttft_tpot.png`) |

---

## Step 1 — Collect Data

All commands are run from `<repo_root>/src/`.

### 1a. Qwen2.5-0.5B — Transformer venv

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

# Short context
python ../ispass_ae/scripts/paper_figures/Fig_1/collect_fig1_data.py \
    --model qwen --seq_len 1024 --max_new_tokens 256 --device cuda

# Long context
python ../ispass_ae/scripts/paper_figures/Fig_1/collect_fig1_data.py \
    --model qwen --seq_len 32768 --max_new_tokens 256 --device cuda
```

### 1b. Mamba2-780m — Mamba venv

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>/src

# Short context
python ../ispass_ae/scripts/paper_figures/Fig_1/collect_fig1_data.py \
    --model mamba2 --seq_len 1024 --max_new_tokens 256 --device cuda

# Long context
python ../ispass_ae/scripts/paper_figures/Fig_1/collect_fig1_data.py \
    --model mamba2 --seq_len 32768 --max_new_tokens 256 --device cuda
```

Output CSVs are written to (relative to `src/`):

| CSV | Contents |
|-----|----------|
| `tpot_logs/tpot_times.csv` | TTFT (`prefill_time_seconds`), TPOT (`tpot_seconds`), throughput |

---

## Step 2 — Plot Figure 1

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>

python ispass_ae/scripts/paper_figures/Fig_1/plot_fig1.py \
    --tpot_csv src/tpot_logs/tpot_times.csv \
    --out_dir  ispass_ae/scripts/paper_figures/Fig_1
```

Two PNG files are produced in `out_dir`:

| File | Description |
|------|-------------|
| `intro_ttft_tpot.png` | Publication-quality figure (no axis labels, 300 DPI) |
| `intro_ttft_tpot_annotated.png` | Annotated version with exact values on each bar |

> **Tip:** If the CSVs are not present, `plot_fig1.py` falls back to the hardcoded paper values so the figure can always be regenerated without running inference.

---


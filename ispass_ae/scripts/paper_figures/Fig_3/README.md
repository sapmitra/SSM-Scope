# Figure 3 — Accuracy vs TTFT: Transformer, SSM, and Hybrid (~1.5B Models)

> 💡 **Tip:** The interactive notebook at [`plotting_accuracy_ttft.ipynb`](../../../notebooks/plotting_accuracy_ttft.ipynb) can regenerate the figure directly from hard-coded paper values without running any inference.

> ⚡ **Quick start:** Run the script from the repo root (or from this directory) to collect all data and generate the figure in one step:
> ```bash
> # from repo root
> chmod +x ispass_ae/scripts/paper_figures/Fig_3/gen_fig3.sh
> bash ispass_ae/scripts/paper_figures/Fig_3/gen_fig3.sh
>
> # or from this directory
> chmod +x gen_fig3.sh
> bash gen_fig3.sh
> ```
> The script activates the correct venvs automatically and writes the output PNGs to this directory.

#### For detailed instructions on how to set up the environments, collect the raw data, and generate the figure, see the rest of this README.

---

This directory contains the scripts to **collect the raw data** and **reproduce Figure 3** from the paper.

Figure 3 shows benchmark **accuracy** and **Time to First Token (TTFT)** for three ~1.5B-parameter model architectures at ~57k tokens of input context:

- **Qwen2.5-1.5B-Instruct** — Transformer
- **Mamba2-1.3b** — State Space Model (SSM)
- **Falcon-H1-1.5B-Instruct** — Hybrid SSM-Transformer

Accuracy is evaluated across five benchmarks:

| Benchmark  | Description |
|------------|-------------|
| MMLU       | Massive Multitask Language Understanding |
| HellaSwag  | Commonsense NLI sentence completion |
| Winogrande | Commonsense reasoning via pronoun resolution |
| ARC-C      | AI2 Reasoning Challenge (Challenge set) |
| TruthfulQA | Tendency to reproduce human misconceptions |

TTFT is measured at ~57,344 input tokens to highlight the prefill latency advantage of SSMs and hybrid models at long context.

---

## Files

| File | Purpose |
|------|---------|
| `gen_fig3.sh` | End-to-end bash script — collects TTFT data and plots the figure in one command |
| `collect_fig3_data.py` | Runs inference profiling for one model at ~57k tokens and writes TTFT to CSV |
| `plot_fig3.py` | Reads the CSVs and generates the publication-quality PNGs |
| `accuracy_data.csv` | Hard-coded benchmark accuracy scores (no inference required) |


---

## Step 1 — Collect TTFT Data

All commands are run from `<repo_root>/src/`.

### 1a. Qwen2.5-1.5B — Transformer venv

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

python ../ispass_ae/scripts/paper_figures/Fig_3/collect_fig3_data.py \
    --model qwen25_1.5b --seq_len 57344 --device cuda
```

### 1b. Mamba2-1.3b — Mamba venv

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>/src

python ../ispass_ae/scripts/paper_figures/Fig_3/collect_fig3_data.py \
    --model mamba2_1.3b --seq_len 57344 --device cuda
```

### 1c. Falcon-H1-1.5B — Falcon venv

```bash
source ~/.venvs/torch_falcon_ispass/bin/activate
cd <repo_root>/src

python ../ispass_ae/scripts/paper_figures/Fig_3/collect_fig3_data.py \
    --model falcon_h1_1.5b --seq_len 57344 --device cuda
```

Profiling output is written to (relative to `src/`):

| CSV | Contents |
|-----|----------|
| `ttft_logs/iteration_times.csv` | per-iteration TTFT (`time_seconds`) at the given `seq_length` |

---

## Step 2 — Plot Figure 3

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>

python ispass_ae/scripts/paper_figures/Fig_3/plot_fig3.py \
    --ttft_csv     src/ttft_logs/iteration_times.csv \
    --accuracy_csv ispass_ae/scripts/paper_figures/Fig_3/accuracy_data.csv \
    --out_dir      ispass_ae/scripts/paper_figures/Fig_3
```

Two PNG files are produced in `out_dir`:

| File | Description |
|------|-------------|
| `accuracy_ttft.png` | Publication-quality figure (no axis tick labels, 300 DPI) |
| `accuracy_ttft_annotated.png` | Annotated version with labels and exact TTFT values |

> 💡 **Tip:** If the timing CSV is not present, `plot_fig3.py` falls back to hard-coded paper values so the figure can always be regenerated without running inference.

---


# Figure 7 — GPU Kernel Time Breakdown: Mamba-130m vs. Mamba2-130m

> **Tip:** The interactive notebook at
> [`plotting_ops.ipynb`](../../../notebooks/plotting_ops.ipynb) can regenerate
> the figure directly from pre-existing profiling data in `profile_data/`
> without running any new inference.

> **Quick start:** Run the script from the repo root (or from this directory)
> to collect all data and generate the figure in one step:
> ```bash
> # from repo root
> bash ispass_ae/scripts/paper_figures/Fig_7/gen_fig7.sh
>
> # or from this directory
> bash gen_fig7.sh
> ```
> The script activates the correct venv automatically, runs all 20 profiling
> jobs (2 models × 10 sequence lengths), and writes the output PNGs to this
> directory.  Profile CSVs are stored in `src/profile_logs/`.

#### For detailed step-by-step instructions see the rest of this README.

---

Figure 7 shows the **GPU kernel time breakdown** of two SSM architectures
across increasing prefill sequence lengths:

| Model | Architecture | Weights |
|-------|-------------|---------|
| Mamba-130m  | SSM (Mamba-1) | `state-spaces/mamba-130m`  |
| Mamba2-130m | SSM (Mamba-2) | `state-spaces/mamba2-130m` |

Execution time is decomposed into:

| Category | Description |
|----------|-------------|
| **GEMM** | Linear projections, convolutions |
| **SSM_Scan** | State-space recurrence kernel |
| **activation** | SiLU, GeLU, etc. |
| **arithmetic** | Element-wise and reduction ops |
| **memory** | Data movement ops (reshape, permute, …) |
| **nomralization** | Layer norm / RMS norm |
| **embedding** | Token embedding lookup |
| **logit_computation** | Output softmax |
| **other** | Miscellaneous |

---

## Files

| File | Purpose |
|------|---------|
| `gen_fig7.sh` | End-to-end bash script — collects all data and plots |
| `collect_fig7_data.py` | Profiles one model at one sequence length and writes operator-breakdown CSVs |
| `plot_fig7.py` | Reads the CSVs and generates publication-quality PNGs |

---

## Step 1 — Collect Data

All commands are run from `<repo_root>/src/`.  Only the **Mamba venv** is
needed because both models use `mamba_ssm`.

### Sequence lengths

10 sequence lengths are profiled for each model:
`256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072`

### 1a. Profile mamba-130m

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>/src

for SEQ_LEN in 256 512 1024 2048 4096 8192 16384 32768 65536 131072; do
    python ../ispass_ae/scripts/paper_figures/Fig_7/collect_fig7_data.py \
        --model mamba \
        --seq_len ${SEQ_LEN} \
        --device cuda
done
```

> **`--out_dir` defaults to `src/profile_logs`**, so no explicit flag is needed.
> Pass `--out_dir <path>` to override.

### 1b. Profile mamba2-130m

```bash
for SEQ_LEN in 256 512 1024 2048 4096 8192 16384 32768 65536 131072; do
    python ../ispass_ae/scripts/paper_figures/Fig_7/collect_fig7_data.py \
        --model mamba2 \
        --seq_len ${SEQ_LEN} \
        --device cuda
done
```

---

## Step 2 — Plot Figure 7

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>

python ispass_ae/scripts/paper_figures/Fig_7/plot_fig7.py \
    --out_dir ispass_ae/scripts/paper_figures/Fig_7
```

> **`--profile_data_dir` defaults to `src/profile_logs`**, so no explicit flag
> is needed when the default collection path was used.
> Pass `--profile_data_dir <path>` to override.

Output layout (relative to `<repo_root>/src/profile_logs/`):

```
src/profile_logs/
    mamba-130m_cuda_1_<seq_len>/
        mamba-130m_cuda_1_<seq_len>.csv   ← raw per-op timing
        gemm.csv
        non_gemm.csv
        ssm_scan.csv
        summary_mamba-130m_cuda_1_<seq_len>.csv
        pct_mamba-130m_cuda_1_<seq_len>.csv   ← % breakdown (needed by plot)
        gng_*.csv
        (optional) *.json  chrome trace
    mamba2-130m_cuda_1_<seq_len>/
        ...
```

Output files written to `out_dir`:

| File | Description |
|------|-------------|
| `fig7_ops_breakdown.png` | Side-by-side comparison, publication quality (300 DPI, no tick labels) |
| `fig7_ops_breakdown_annotated.png` | Same with axis labels, title, and legend (150 DPI) |
| `fig7_mamba_breakdown.png` | Mamba-130m individual breakdown (300 DPI) |
| `fig7_mamba_breakdown_annotated.png` | Mamba-130m with annotations (150 DPI) |
| `fig7_mamba2_breakdown.png` | Mamba2-130m individual breakdown (300 DPI) |
| `fig7_mamba2_breakdown_annotated.png` | Mamba2-130m with annotations (150 DPI) |

> **Tip:** If the profiling CSVs are not present, `plot_fig7.py` falls back to
> hard-coded paper values so the figure can always be regenerated without
> running inference.  Pass `--skip_summarize` if the summary CSVs already exist.

---

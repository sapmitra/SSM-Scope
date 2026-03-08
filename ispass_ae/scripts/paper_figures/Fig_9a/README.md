# Figure 9a — GPU Kernel Time Breakdown: Mamba-130m vs. Mamba2-130m (Jetson Nano Orin)

> **Tip:** The interactive notebook at
> [`plotting_ops_jetson.ipynb`](../../../notebooks/plotting_ops_jetson.ipynb)
> can regenerate the figure directly from pre-existing profiling data without
> running new inference.

> **Quick start:** Run the script from the repo root (or from this directory)
> to collect all data and generate the figure in one step:
> ```bash
> # from repo root
> chmod +x ispass_ae/scripts/paper_figures/Fig_9a/gen_fig9a.sh
> bash ispass_ae/scripts/paper_figures/Fig_9a/gen_fig9a.sh
>
> # or from this directory
> chmod +x gen_fig9a.sh
> bash gen_fig9a.sh
> ```
> The script activates the correct venv automatically, runs all 16 profiling
> jobs (2 models × 8 sequence lengths), and writes the output PNGs to this
> directory.  Profile CSVs are stored in `src/profile_logs/`.

#### For detailed step-by-step instructions see the rest of this README.

---

Figure 9a shows the **GPU kernel time breakdown** of two SSM architectures
across increasing prefill sequence lengths on **NVIDIA Jetson Nano**:

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

Sequence lengths are limited to **256 – 32 768** tokens to fit within the
Jetson Nano unified memory budget.

---

## Files

| File | Purpose |
|------|---------|
| `gen_fig9a.sh` | End-to-end bash script — collects all data and plots |
| `collect_fig9a_data.py` | Profiles one model at one sequence length, writes operator-breakdown CSVs |
| `plot_fig9a.py` | Reads the CSVs and generates publication-quality PNGs |

---

## Step 1 — Collect Data

All commands are run from `<repo_root>/src/`.  Only the **Mamba venv** is
needed because both models use `mamba_ssm`.

### Sequence lengths

8 sequence lengths are profiled for each model:
`256, 512, 1024, 2048, 4096, 8192, 16384, 32768`

### 1a. Profile mamba-130m

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>/src

for SEQ_LEN in 256 512 1024 2048 4096 8192 16384 32768; do
    python ../ispass_ae/scripts/paper_figures/Fig_9a/collect_fig9a_data.py \
        --model mamba \
        --seq_len ${SEQ_LEN} \
        --device cuda
done
```

> **`--out_dir` defaults to `src/profile_logs`**, so no explicit flag is needed.
> Pass `--out_dir <path>` to override.

### 1b. Profile mamba2-130m

```bash
for SEQ_LEN in 256 512 1024 2048 4096 8192 16384 32768; do
    python ../ispass_ae/scripts/paper_figures/Fig_9a/collect_fig9a_data.py \
        --model mamba2 \
        --seq_len ${SEQ_LEN} \
        --device cuda
done
```

---

## Step 2 — Plot Figure 9a

```bash
# from repo root, any venv with matplotlib + pandas
source ~/.venvs/torch_ssm_ispass/bin/activate

python ispass_ae/scripts/paper_figures/Fig_9a/plot_fig9a.py \
    --profile_data_dir src/profile_logs \
    --out_dir          ispass_ae/scripts/paper_figures/Fig_9a
```

### Output files

| File | Description |
|------|-------------|
| `fig9a_ops_breakdown.png` | Publication-quality side-by-side, 300 DPI (no axis labels) |
| `fig9a_ops_breakdown_annotated.png` | Same with labels, title, legend, 150 DPI |
| `fig9a_mamba_breakdown.png` | Mamba-130m panel only, 300 DPI |
| `fig9a_mamba_breakdown_annotated.png` | Mamba-130m panel with labels, 150 DPI |
| `fig9a_mamba2_breakdown.png` | Mamba2-130m panel only, 300 DPI |
| `fig9a_mamba2_breakdown_annotated.png` | Mamba2-130m panel with labels, 150 DPI |

---

## Profile data layout

Each invocation of `collect_fig9a_data.py` creates one subdirectory:

```
src/profile_logs/
    mamba-130m_cuda_1_<seq_len>/
        mamba-130m_cuda_1_<seq_len>.csv    ← raw per-op timing
        non_gemm.csv
        gemm.csv
        ssm_scan.csv
        pct_mamba-130m_cuda_1_<seq_len>.csv   ← % breakdown (read by plot_fig9a.py)
        summary_mamba-130m_cuda_1_<seq_len>.csv
        gng_mamba-130m_cuda_1_<seq_len>.csv
        gng_pct_mamba-130m_cuda_1_<seq_len>.csv
        gng_ssm_mamba-130m_cuda_1_<seq_len>.csv
        gng_ssm_pct_mamba-130m_cuda_1_<seq_len>.csv
    mamba2-130m_cuda_1_<seq_len>/
        …
```

---

# Figure 8 — GPU Kernel Time Breakdown: Hymba-1.5B vs. Zamba2-1.2B

> **Tip:** The interactive notebook at
> [`ssm_hybrid_op_breakdown.ipynb`](../../../notebooks/ssm_hybrid_op_breakdown.ipynb)
> can regenerate the figure directly from pre-existing profiling data. The profiling data is not present in
> `profile_data/` because of storage constraints.

> **Quick start:** Run the script from the repo root (or from this directory)
> to collect all data and generate the figure in one step:
> ```bash
> # from repo root
> chmod +x ispass_ae/scripts/paper_figures/Fig_8/gen_fig8.sh
> bash ispass_ae/scripts/paper_figures/Fig_8/gen_fig8.sh
>
> # or from this directory
> chmod +x gen_fig8.sh
> bash gen_fig8.sh
> ```
> The script activates the correct venv automatically, runs all 15 profiling
> jobs (7 Hymba + 8 Zamba2 sequence lengths), and writes the output PNGs to
> this directory.  Profile CSVs are stored in `src/profile_logs/`.

#### For detailed step-by-step instructions see the rest of this README.

---

Figure 8 shows the **GPU kernel time breakdown** of two hybrid SSM architectures
across increasing prefill sequence lengths:

| Model | Architecture | Weights |
|-------|-------------|---------|
| Hymba-1.5B | SSM + Attention (interleaved Mamba-2 + MHA per layer) | `nvidia/Hymba-1.5B-Instruct` |
| Zamba2-1.2B | SSM + Attention (shared transformer layers + Mamba-2 blocks) | `Zyphra/Zamba2-1.2B-Instruct-v2` |

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
| `gen_fig8.sh` | End-to-end bash script — collects all data and plots |
| `collect_fig8_data.py` | Profiles one model at one sequence length and writes operator-breakdown CSVs |
| `plot_fig8.py` | Reads the CSVs and generates publication-quality PNGs |

---

## Environment

Both models (Hymba and Zamba2) require the **Falcon/Hybrid venv** because they
interleave Mamba-2 SSM layers with attention and depend on `mamba_ssm` +
`transformers>=4.48`.

```bash
# Environment 3 — torch_falcon_ispass
source ~/.venvs/torch_falcon_ispass/bin/activate
```

See `ispass_ae/scripts/env_setup/README.md` for full setup instructions.

---

## Step 1 — Collect Data

All commands are run from `<repo_root>/src/`.

### Sequence lengths

| Model | Sequence lengths |
|-------|-----------------|
| Hymba-1.5B | 256, 512, 1024, 2048, 4096, 8192, 16384 |
| Zamba2-1.2B | 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 |

### 1a. Profile Hymba-1.5B

```bash
source ~/.venvs/torch_falcon_ispass/bin/activate
cd <repo_root>/src

for SEQ_LEN in 256 512 1024 2048 4096 8192 16384; do
    python ../ispass_ae/scripts/paper_figures/Fig_8/collect_fig8_data.py \
        --model hymba \
        --seq_len ${SEQ_LEN} \
        --device cuda
done
```

> **`--out_dir` defaults to `src/profile_logs`**, so no explicit flag is needed.
> Pass `--out_dir <path>` to override.

### 1b. Profile Zamba2-1.2B

```bash
for SEQ_LEN in 256 512 1024 2048 4096 8192 16384 32768; do
    python ../ispass_ae/scripts/paper_figures/Fig_8/collect_fig8_data.py \
        --model zamba2 \
        --seq_len ${SEQ_LEN} \
        --device cuda
done
```

---

## Step 2 — Plot Figure 8

```bash
source ~/.venvs/torch_falcon_ispass/bin/activate
cd <repo_root>

python ispass_ae/scripts/paper_figures/Fig_8/plot_fig8.py \
    --out_dir ispass_ae/scripts/paper_figures/Fig_8
```

> **`--profile_data_dir` defaults to `src/profile_logs`**, so no explicit flag
> is needed when the default collection path was used.
> Pass `--profile_data_dir <path>` to override.

Output layout (relative to `<repo_root>/src/profile_logs/`):

```
src/profile_logs/
    hymba_cuda_1_<seq_len>/
        hymba_cuda_1_<seq_len>.csv       ← raw per-op timing
        gemm.csv
        non_gemm.csv
        ssm_scan.csv
        summary_hymba_cuda_1_<seq_len>.csv
        pct_hymba_cuda_1_<seq_len>.csv   ← % breakdown (needed by plot)
        gng_*.csv
        (optional) *.json  chrome trace
    zamba2_cuda_1_<seq_len>/
        ...
```

Output files written to `out_dir`:

| File | Description |
|------|-------------|
| `fig8_ops_breakdown.png` | Side-by-side comparison, publication quality (300 DPI, no tick labels) |
| `fig8_ops_breakdown_annotated.png` | Same with axis labels, title, and legend (150 DPI) |
| `fig8_hymba_breakdown.png` | Hymba individual breakdown (300 DPI) |
| `fig8_hymba_breakdown_annotated.png` | Hymba with annotations (150 DPI) |
| `fig8_zamba2_breakdown.png` | Zamba2 individual breakdown (300 DPI) |
| `fig8_zamba2_breakdown_annotated.png` | Zamba2 with annotations (150 DPI) |

> **Tip:** If the profiling CSVs are not present, `plot_fig8.py` falls back to
> hard-coded paper values so the figure can always be regenerated without
> running inference.  Pass `--skip_summarize` if the summary CSVs already exist.

---


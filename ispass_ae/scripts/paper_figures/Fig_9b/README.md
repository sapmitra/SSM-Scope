# Figure 9b — Cross-Device GPU Kernel Time Breakdown

> **Tip:** The interactive notebook at
> [`plotting_ops_cross_device.ipynb`](../../../notebooks/plotting_ops_cross_device.ipynb)
> can regenerate the figure directly from pre-existing profiling CSVs in
> `profile_data/` (desktop) and `profile_data_jetson/` (Jetson) without
> running any new inference.

> **Quick start (desktop collection + plot):**
> ```bash
> # From repo root
> bash ispass_ae/scripts/paper_figures/Fig_9b/gen_fig9b.sh
>
> # Or from this directory
> bash gen_fig9b.sh
> ```
> The script activates the correct venv for each model family, profiles all
> nine models at seq_len=1024, and writes the output PNG to this directory.
> Profile CSVs are stored in `src/profile_logs/`.

> **Two-device workflow** (Desktop + Jetson):  See [Step 2 — Jetson](#step-2--collect-jetson-data) below.

#### For detailed step-by-step instructions see the rest of this README.

---

Figure 9b shows the **GPU kernel time breakdown** across all three model
families at a fixed sequence length (**1024 tokens**) on two hardware platforms:

| Platform | Device                          |
|----------|---------------------------------|
| Desktop  | NVIDIA GPU (RTX 4090)   |
| Edge     | NVIDIA Jetson Orin Nano         |

Models and architecture families:

| Family          | Model                  | HuggingFace checkpoint               |
|-----------------|------------------------|--------------------------------------|
| **Transformer** | GPT-Neo-125m           | `EleutherAI/gpt-neo-125m`            |
| **Transformer** | TinyLlama-1.1B         | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Transformer** | LLaMA-3.2-1B           | `meta-llama/Llama-3.2-1B-Instruct`   |
| **Transformer** | Qwen2.5-0.5B-Instruct  | `Qwen/Qwen2.5-0.5B-Instruct`         |
| **Transformer** | Qwen2.5-1.5B-Instruct  | `Qwen/Qwen2.5-1.5B-Instruct`         |
| **SSM**         | Mamba-130m             | `state-spaces/mamba-130m`            |
| **SSM**         | Mamba2-130m            | `state-spaces/mamba2-130m`           |
| **Hybrid**      | Zamba2-1.2B            | `Zyphra/Zamba2-1.2B-Instruct-v2`     |
| **Hybrid**      | Hymba-1.5B             | `nvidia/Hymba-1.5B-Instruct`         |

Execution time is decomposed into the same operator categories used in
[`plotting_ops_cross_device.ipynb`](../../../notebooks/plotting_ops_cross_device.ipynb):

| Category          | Description                                        |
|-------------------|----------------------------------------------------|
| **GEMM**          | Linear projections, convolutions                   |
| **SSM_Scan**      | State-space recurrence kernel                      |
| **activation**    | SiLU, GeLU, ReLU, etc.                             |
| **arithmetic**    | Element-wise and reduction ops                     |
| **memory**        | Data movement ops (reshape, permute, cat, …)       |
| **nomralization** | Layer norm / RMS norm                              |
| **embedding**     | Token embedding lookup                             |
| **logit_computation** | Output softmax                                |
| **other**         | Dropout, sort, topk, …                             |

---

## Files

| File                    | Purpose                                                                   |
|-------------------------|---------------------------------------------------------------------------|
| `gen_fig9b.sh`          | End-to-end bash script — collects all data **and** plots the figure       |
| `collect_fig9b_data.py` | Profiles **one model** at one sequence length; writes operator-breakdown CSVs |
| `plot_fig9b.py`         | Reads CSVs from both platforms and generates publication-quality PNGs     |

---

## Prerequisites

Three Python virtual environments are required.  All must have PyTorch and
the `torch_profiler` utilities from this repository installed.

| venv                        | Models                                                  | Extra requirement          |
|-----------------------------|---------------------------------------------------------|----------------------------|
| `~/.venvs/torch_transformers_ispass` | GPT-Neo, TinyLlama, LLaMA-3.2, Qwen2.5-0.5B, Qwen2.5-1.5B | standard HF Transformers |
| `~/.venvs/torch_ssm_ispass`          | Mamba-130m, Mamba2-130m                          | `mamba_ssm`               |
| `~/.venvs/torch_falcon_ispass`       | Hymba-1.5B, Zamba2-1.2B                          | `mamba_ssm` + `transformers>=4.48` |

---

## Step 1 — Collect Desktop Data

All commands are run from `<repo_root>/src/`.

### 1a. Transformer models

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

for MODEL in gpt-neo-125m tinyllama llama3_2 qwen25-instruct qwen25-1.5b-instruct; do
    python ../ispass_ae/scripts/paper_figures/Fig_9b/collect_fig9b_data.py \
        --model ${MODEL} --seq_len 1024 --device cuda
done
```

### 1b. SSM models

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>/src

for MODEL in mamba mamba2; do
    python ../ispass_ae/scripts/paper_figures/Fig_9b/collect_fig9b_data.py \
        --model ${MODEL} --seq_len 1024 --device cuda
done
```

### 1c. Hybrid models

```bash
source ~/.venvs/torch_falcon_ispass/bin/activate
cd <repo_root>/src

for MODEL in hymba zamba2; do
    python ../ispass_ae/scripts/paper_figures/Fig_9b/collect_fig9b_data.py \
        --model ${MODEL} --seq_len 1024 --device cuda
done
```

> **`--out_dir` defaults to `src/profile_logs`**, so no explicit flag is
> needed.  Pass `--out_dir <path>` to override.

---

## Step 2 — Collect Jetson Data

On the **Jetson board**, clone the repository (or transfer this directory),
then run the same three collection steps as Step 1.  Use `SKIP_PLOT=1` so the
script does not attempt to generate the comparison plot before the desktop data
is available:

```bash
# On the Jetson board
cd <repo_root>
SKIP_PLOT=1 bash ispass_ae/scripts/paper_figures/Fig_9b/gen_fig9b.sh
```

This writes CSVs to `src/profile_logs/` on the Jetson.

### Transfer to workstation

```bash
# On the workstation
rsync -avz jetson:<repo_root>/src/profile_logs/ \
          <local_repo_root>/src/profile_logs_jetson/
```

Or with `scp`:

```bash
scp -r jetson:<repo_root>/src/profile_logs/ \
       <local_repo_root>/src/profile_logs_jetson/
```

---

## Step 3 — Plot Figure 9b

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>

python ispass_ae/scripts/paper_figures/Fig_9b/plot_fig9b.py \
    --out_dir ispass_ae/scripts/paper_figures/Fig_9b
```

> **Default directories:**
> - `--desktop_dir` defaults to `src/profile_logs`
> - `--jetson_dir`  defaults to `src/profile_logs_jetson`
>
> Pass explicit paths if you used non-default collection directories.


---

## Output layout

Profile CSVs (relative to `<repo_root>/`):

```
src/profile_logs/                          ← Desktop data
    gpt-neo-125m_cuda_1_1024/
        gpt-neo-125m_cuda_1_1024.csv       ← raw per-op timing
        gemm.csv  non_gemm.csv  ssm_scan.csv
        pct_gpt-neo-125m_cuda_1_1024.csv   ← % breakdown (used by plot)
        summary_*.csv  gng_*.csv
    tinyllama_cuda_1_1024/  …
    llama3_2_cuda_1_1024/   …
    qwen25-instruct_cuda_1_1024/  …
    qwen25-1.5b-instruct_cuda_1_1024/  …
    mamba-130m_cuda_1_1024/   …
    mamba2-130m_cuda_1_1024/  …
    zamba2_cuda_1_1024/       …
    hymba_cuda_1_1024/        …

src/profile_logs_jetson/                   ← Jetson data (transferred)
    gpt-neo-125m_cuda_1_1024/   …
    …                           (same layout as above)
```

Output PNGs written to `out_dir` (defaults to this directory):

| File | Description |
|------|-------------|
| `fig9b_device_comparison_seq1024.png` | Publication-quality figure (300 DPI, no labels) |
| `fig9b_device_comparison_seq1024_annotated.png` | Same with axis labels, legend, category headers (150 DPI) |

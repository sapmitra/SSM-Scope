# Figure 6a — Prefill Energy Consumption vs Sequence Length

> 💡 **Tip:** The interactive notebook at [`plotting_energy_seq.ipynb`](../../../notebooks/plotting_energy_seq.ipynb) can regenerate the figure directly from hard-coded paper values without running any inference.

> ⚡ **Quick start:** Run the script from the repo root (or from this directory) to collect all data and generate the figure in one step:
> ```bash
> # from repo root
> chmod +x ispass_ae/scripts/paper_figures/Fig_6a/gen_fig6a.sh
> bash ispass_ae/scripts/paper_figures/Fig_6a/gen_fig6a.sh
>
> # or from this directory
> chmod +x gen_fig6a.sh
> bash gen_fig6a.sh
> ```
> The script activates the correct venvs automatically and writes the output PNGs to this directory.

#### For detailed instructions on how to set up the environments, collect the raw data, and generate the figure, see the rest of this README.
---

This directory contains the scripts to **collect the raw data** and **reproduce Figure 6a** from the paper.

Figure 6a shows GPU energy consumption (Joules) during the **prefill phase** for:

- **Qwen2.5-0.5B-Instruct** — Transformer
- **Mamba2-780m** — State Space Model (SSM)
- **Falcon-H1-0.5B-Base** — Hybrid SSM/Attention

across 10 sequence lengths:

| Sequence lengths (tokens) |
|--------------------------|
| 1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152, 57344 |

---

## Files

| File | Purpose |
|------|---------|
| `gen_fig6a.sh` | End-to-end bash script — collects data and plots the figure in one command |
| `collect_fig6a_data.py` | Profiles one model at one sequence length and writes energy (J) to CSV |
| `plot_fig6a.py` | Reads the CSV and generates publication-quality PNGs |
| `README.md` | This file |
> **Power log parsing** is handled internally by
> [`src/profiling/power_logger.py`](../../../../src/profiling/power_logger.py).
> That module exposes ``parse_energy_from_log(log_file, num_iterations)``
> which is called by `eval.py`; it can also be run standalone to inspect
> raw power logs (see [Re-parsing logs](#optional-re-parse-power-logs) below).
---

## How Energy is Measured

For each `(model, seq_len)` pair:

1. **Warmup** — 5 forward passes (not recorded).
2. **Measurement** — `N ≥ 50` forward passes under `nvidia-smi` power logging (100 ms sampling).
3. **Parse** — `src/profiling/power_logger.py::parse_energy_from_log(log_file, N)` reads
   the CSV, strips unit suffixes, and computes per-GPU values:
   ```
   energy_per_pass (J) = Σ(power_i [W] × 0.1 s) / N
   ```
   The function returns a dict with `gpu{i}_energy_joules`, `gpu{i}_avg_power_watts`,
   `total_energy_joules`, and `avg_power_watts`.
4. **Save** — `total_energy_joules` and `avg_power_watts` are appended to `src/energy_logs/energy_data.csv`.

Raw per-run nvidia-smi logs are stored in `src/power_logs/`.

---

## Optional: Re-parse Power Logs

If you already have raw power logs in `src/power_logs/` and want to recompute energy
without re-running inference, run `power_logger.py` directly:

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

python -m profiling.power_logger \
    --log_dir    power_logs \
    --num_iterations 50
```

This prints per-GPU average power (W) and energy per pass (J) for every `*.log` file found.

---

## Step 1 — Collect Data

All commands are run from `<repo_root>/src/`.

### 1a. Qwen2.5-0.5B-Instruct — Transformer venv

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

for SEQ_LEN in 1024 2048 4096 8192 16384 24576 32768 40960 49152 57344; do
    python ../ispass_ae/scripts/paper_figures/Fig_6a/collect_fig6a_data.py \
        --model qwen --seq_len ${SEQ_LEN} --device cuda
done
```

### 1b. Mamba2-780m — Mamba venv

```bash
source ~/.venvs/torch_ssm_ispass/bin/activate
cd <repo_root>/src

for SEQ_LEN in 1024 2048 4096 8192 16384 24576 32768 40960 49152 57344; do
    python ../ispass_ae/scripts/paper_figures/Fig_6a/collect_fig6a_data.py \
        --model mamba2 --seq_len ${SEQ_LEN} --device cuda
done
```

### 1c. Falcon-H1-0.5B-Base — Transformer venv

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>/src

for SEQ_LEN in 1024 2048 4096 8192 16384 24576 32768 40960 49152 57344; do
    python ../ispass_ae/scripts/paper_figures/Fig_6a/collect_fig6a_data.py \
        --model falcon --seq_len ${SEQ_LEN} --device cuda
done
```

Output CSV (relative to `src/`):

| File | Contents |
|------|----------|
| `energy_logs/energy_data.csv` | `model_name, seq_len, energy_joules, duration_per_pass_s, avg_power_watts, num_iterations, device, timestamp` |

---

## Step 2 — Plot Figure 6a

```bash
source ~/.venvs/torch_transformers_ispass/bin/activate
cd <repo_root>

python ispass_ae/scripts/paper_figures/Fig_6a/plot_fig6a.py \
    --energy_csv src/energy_logs/energy_data.csv \
    --out_dir    ispass_ae/scripts/paper_figures/Fig_6a
```

Two PNG files are produced in `out_dir`:

| File | Description |
|------|-------------|
| `energy_consumption.png` | Publication-quality figure (no axis labels, 300 DPI) |
| `energy_consumption_annotated.png` | Annotated version with exact values on each bar (150 DPI) |

> 💡 **Tip:** If the CSV is absent or incomplete, `plot_fig6a.py` falls back to the hard-coded paper values so the figure can always be regenerated without running inference.

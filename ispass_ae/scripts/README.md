# 📁 Scripts Directory

This directory contains all the scripts needed to reproduce the paper's figures and set up the required environments.

---

## 📂 Contents

| Directory | Purpose |
|-----------|---------|
| [`env_setup/`](env_setup/README.md) | ⬅️ **Start here** — set up the three Python virtual environments |
| [`paper_figures/Fig_1/`](paper_figures/Fig_1/README.md) | TTFT & TPOT crossover (Fig 1) |
| [`paper_figures/Fig_3/`](paper_figures/Fig_3/README.md) | Accuracy vs TTFT (Fig 3) |
| [`paper_figures/Fig_5a/`](paper_figures/Fig_5a/README.md) | Memory footprint — desktop GPU (Fig 5a) |
| [`paper_figures/Fig_5b/`](paper_figures/Fig_5b/README.md) | Memory footprint — Jetson Nano Orin (Fig 5b) |
| [`paper_figures/Fig_6a/`](paper_figures/Fig_6a/README.md) | Prefill energy consumption (Fig 6a) |
| [`paper_figures/Fig_6b/`](paper_figures/Fig_6b/README.md) | Overall throughput (Fig 6b) |
| [`paper_figures/Fig_7/`](paper_figures/Fig_7/README.md) | SSM op-breakdown: Mamba vs Mamba2 — desktop (Fig 7) |
| [`paper_figures/Fig_8/`](paper_figures/Fig_8/README.md) | Hybrid op-breakdown: Hymba vs Zamba2 — desktop (Fig 8) |
| [`paper_figures/Fig_9a/`](paper_figures/Fig_9a/README.md) | SSM op-breakdown: Mamba vs Mamba2 — Jetson (Fig 9a) |
| [`paper_figures/Fig_9b/`](paper_figures/Fig_9b/README.md) | Cross-device op-breakdown: desktop vs Jetson (Fig 9b) |

---

## 🔄 Recommended Workflow

1. **Set up environments** → [`env_setup/README.md`](env_setup/README.md)
2. **Reproduce a figure** — run the `gen_fig*.sh` script inside the target figure directory, or follow the step-by-step instructions in its `README.md`.
3. **Interactive exploration** — open the corresponding notebook from [`../notebooks/`](../notebooks/).

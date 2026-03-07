'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-06
 # @ Description: Reproduce Figure 5b (GPU Memory Footprint on NVIDIA Jetson) from the paper.
 #                See the README for usage instructions.
 '''

"""
Reproduce Figure 5b — GPU Memory Footprint: Transformer vs. SSM Models on NVIDIA Jetson.

Reads the memory profiling results produced by ``collect_fig5b_data.py``
(``src/memory/memory_footprints_jetson.csv``) and generates a publication-quality
stacked-bar PNG.

Models shown (ordered left-to-right within each sequence-length group):
    Llama-3.2-1B | Qwen2.5-0.5B | Mamba-790m |
    Mamba2-780m  | Falcon-H1-0.5B | Zamba2-1.2B

Each bar is decomposed into:
    - Model Size      — static parameter memory (constant)
    - Activation Mem  — intermediate tensors during the forward pass
    - KV Cache        — key-value cache (zero for pure SSMs)

Usage (from repo root, any venv with matplotlib + pandas):

    python ispass_ae/scripts/paper_figures/Fig_5b/plot_fig5b.py \\
        --csv_path src/memory/memory_footprints_jetson.csv \\
        --out_dir  ispass_ae/scripts/paper_figures/Fig_5b

Output file
-----------
``<out_dir>/memory_footprint_jetson_ispass.png``  — 300 DPI, no axis tick labels.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.normpath(os.path.join(_script_dir, "../../../../"))

DEFAULT_CSV     = os.path.join(_repo_root, "src", "memory", "memory_footprints_jetson.csv")
DEFAULT_OUT_DIR = _script_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce Figure 5b — Jetson GPU memory footprint stacked-bar chart.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv_path",
        default=DEFAULT_CSV,
        help="Path to memory_footprints_jetson.csv produced by collect_fig5b_data.py.",
    )
    p.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Directory in which to write memory_footprint_jetson_ispass.png.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_data(df: pd.DataFrame, seq_lengths: list) -> dict:
    """Return a per-seq_len dict of memory components; zeros for missing rows."""
    data: dict = {}
    for seq_len in seq_lengths:
        row = df[df["seq_len"] == seq_len]
        if not row.empty:
            data[seq_len] = {
                "model_size":  row["model_size_mb"].values[0],
                "activation":  row["activation_memory_mb"].values[0],
                "kv_cache":    row["kv_cache_mb"].values[0],
                "reserved":    row["reserved_memory_mb"].values[0],
                "total":       row["total_memory_mb"].values[0],
            }
        else:
            data[seq_len] = {
                "model_size": 0, "activation": 0, "kv_cache": 0,
                "reserved": 0, "total": 0,
            }
    return data


def create_arrays(model_data: dict, seq_lengths: list) -> dict:
    """Convert the per-seq_len dict into flat NumPy-friendly lists."""
    return {
        "model_sizes": [model_data[s]["model_size"] for s in seq_lengths],
        "activations": [model_data[s]["activation"]  for s in seq_lengths],
        "kv_caches":   [model_data[s]["kv_cache"]    for s in seq_lengths],
        "reserved":    [model_data[s]["reserved"]     for s in seq_lengths],
        "totals":      [model_data[s]["total"]        for s in seq_lengths],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Load CSV -----------------------------------------------------------
    if not os.path.isfile(args.csv_path):
        sys.exit(
            f"ERROR: CSV not found at '{args.csv_path}'.\n"
            "Run collect_fig5b_data.py for all models first (see gen_fig5b.sh)."
        )

    df = pd.read_csv(args.csv_path)

    # Drop duplicate (model_name, seq_len) rows — keep the last measurement.
    df = df.drop_duplicates(subset=["model_name", "seq_len"], keep="last")

    # --- Per-model dataframes -----------------------------------------------
    qwen_df      = df[df["model_name"] == "qwen25-instruct"].sort_values("seq_len")
    mamba_df     = df[df["model_name"] == "mamba-790m"].sort_values("seq_len")
    mamba2_df    = df[df["model_name"] == "mamba2-780m"].sort_values("seq_len")
    zamba2_df    = df[df["model_name"] == "zamba2"].sort_values("seq_len")
    llama3_2_df  = df[df["model_name"] == "llama3_2"].sort_values("seq_len")
    falcon_h1_df = df[df["model_name"] == "falcon-h1-0.5b"].sort_values("seq_len")

    # --- Sequence lengths actually present in the data ----------------------
    # Desired sequence lengths matching plotting_mem_footprint_jetson.ipynb
    desired_seq_lengths = [
        256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768,
    ]

    all_seq_raw = sorted(set(
        qwen_df["seq_len"].tolist() +
        mamba_df["seq_len"].tolist() +
        mamba2_df["seq_len"].tolist() +
        zamba2_df["seq_len"].tolist() +
        llama3_2_df["seq_len"].tolist() +
        falcon_h1_df["seq_len"].tolist()
    ))
    all_seq_lengths = [s for s in all_seq_raw if s in desired_seq_lengths]

    if not all_seq_lengths:
        sys.exit("ERROR: No data found in the CSV for the expected sequence lengths.")

    print(f"Sequence lengths in plot : {all_seq_lengths}")

    # Use the full union as the lookup key-space for the helper
    lookup_seq = all_seq_raw

    # --- Prepare per-model data arrays --------------------------------------
    x         = np.arange(len(all_seq_lengths)) * 1.2
    bar_width = 0.15

    qwen_data      = get_model_data(qwen_df,      lookup_seq)
    mamba_data     = get_model_data(mamba_df,     lookup_seq)
    mamba2_data    = get_model_data(mamba2_df,    lookup_seq)
    zamba2_data    = get_model_data(zamba2_df,    lookup_seq)
    llama3_2_data  = get_model_data(llama3_2_df,  lookup_seq)
    falcon_h1_data = get_model_data(falcon_h1_df, lookup_seq)

    qwen_arr      = create_arrays(qwen_data,      all_seq_lengths)
    mamba_arr     = create_arrays(mamba_data,     all_seq_lengths)
    mamba2_arr    = create_arrays(mamba2_data,    all_seq_lengths)
    zamba2_arr    = create_arrays(zamba2_data,    all_seq_lengths)
    llama3_2_arr  = create_arrays(llama3_2_data,  all_seq_lengths)
    falcon_h1_arr = create_arrays(falcon_h1_data, all_seq_lengths)

    # --- Color palettes (dark = model, mid = activation, light/kv = kv) -----
    qwen_c      = {"model": "#E41A1C", "act": "#FBB4AE", "kv": "#FB8072"}
    mamba_c     = {"model": "#377EB8", "act": "#B3CDE3", "kv": "#6BAED6"}
    zamba2_c    = {"model": "#FF7F00", "act": "#FFED97", "kv": "#FFA347"}
    llama3_2_c  = {"model": "#4DAF4A", "act": "#CCEBC5", "kv": "#95D840"}
    mamba2_c    = {"model": "#00BFC4", "act": "#B4E2E2", "kv": "#1DE9B6"}
    falcon_h1_c = {"model": "#829D9A", "act": "#B3D1CE", "kv": "#8EACA8"}

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(26, 10))

    def _stack_bar(offset, arr, colors):
        """Draw one model's three stacked bar layers."""
        bot_act = arr["model_sizes"]
        bot_kv  = [a + b for a, b in zip(arr["model_sizes"], arr["activations"])]
        ax.bar(x + offset, arr["model_sizes"], bar_width, color=colors["model"])
        ax.bar(x + offset, arr["activations"], bar_width, bottom=bot_act, color=colors["act"])
        ax.bar(x + offset, arr["kv_caches"],   bar_width, bottom=bot_kv,  color=colors["kv"])

    # Order: Llama | Qwen | Mamba | Mamba2 | Falcon-H1 | Zamba2
    _stack_bar(-2.5 * bar_width, llama3_2_arr,  llama3_2_c)
    _stack_bar(-1.5 * bar_width, qwen_arr,      qwen_c)
    _stack_bar(-0.5 * bar_width, mamba_arr,     mamba_c)
    _stack_bar(+0.5 * bar_width, mamba2_arr,    mamba2_c)
    _stack_bar(+1.5 * bar_width, falcon_h1_arr, falcon_h1_c)
    _stack_bar(+2.5 * bar_width, zamba2_arr,    zamba2_c)

    # Jetson 8 GB unified memory limit
    ax.axhline(y=8192, color="red", linestyle="-", alpha=0.3)

    # Axis formatting
    ax.set_xticks([])
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # --- Legend (three columns: model size | activation | kv cache) ---------
    def _patch(color, label):
        return mpatches.Patch(facecolor=color, label=label)

    legend_items = [
        # Model size
        _patch(llama3_2_c["model"],  "Llama3-2B: Model Size"),
        _patch(qwen_c["model"],      "Qwen 0.5B: Model Size"),
        _patch(mamba_c["model"],     "Mamba-790m: Model Size"),
        _patch(mamba2_c["model"],    "Mamba2-780m: Model Size"),
        _patch(falcon_h1_c["model"], "Falcon-H1-0.5B: Model Size"),
        _patch(zamba2_c["model"],    "Zamba2: Model Size"),
        # Activation memory
        _patch(llama3_2_c["act"],    "Llama3-2B: Activation"),
        _patch(qwen_c["act"],        "Qwen 0.5B: Activation"),
        _patch(mamba_c["act"],       "Mamba-790m: Activation"),
        _patch(mamba2_c["act"],      "Mamba2-780m: Activation"),
        _patch(falcon_h1_c["act"],   "Falcon-H1-0.5B: Activation"),
        _patch(zamba2_c["act"],      "Zamba2: Activation"),
        # KV cache
        _patch(llama3_2_c["kv"],     "Llama3-2B: KV Cache"),
        _patch(qwen_c["kv"],         "Qwen 0.5B: KV Cache"),
        _patch(mamba_c["kv"],        "Mamba-790m: KV Cache"),
        _patch(mamba2_c["kv"],       "Mamba2-780m: KV Cache"),
        _patch(falcon_h1_c["kv"],    "Falcon-H1-0.5B: KV Cache"),
        _patch(zamba2_c["kv"],       "Zamba2: KV Cache"),
        # Misc
        Line2D([0], [0], color="red", linestyle="-", alpha=0.3, label="8 GB Jetson Memory Limit"),
    ]

    ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])

    # --- Save ---------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "memory_footprint_jetson_ispass.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

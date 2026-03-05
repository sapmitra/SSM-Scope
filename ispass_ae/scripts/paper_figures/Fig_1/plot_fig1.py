'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-04 03:04:46
 # @ Description: Reproduce Figure 1 from the paper.  See the README for usage instructions.
 '''

"""
Reproduce Figure 1 from the paper.

Reads ``tpot_logs/tpot_times.csv`` produced by ``collect_fig1_data.py`` and
generates two PNG files: a publication-quality figure and an annotated version.

If the CSV is not available the script falls back to the hard-coded paper
values so the figure can always be regenerated without running inference.

Usage (from repo root, any venv that has matplotlib + pandas):

    python ispass_ae/scripts/paper_figures/Fig_1/plot_fig1.py \\
        --tpot_csv src/tpot_logs/tpot_times.csv \\
        --out_dir  ispass_ae/scripts/paper_figures/Fig_1

Output files
------------
``intro_ttft_tpot.png``
    Publication-quality figure (no axis tick labels, 300 DPI).
``intro_ttft_tpot_annotated.png``
    Same data with exact values annotated on each bar (150 DPI).
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import os
import warnings

# ---------------------------------------------------------------------------
# Third-party (matplotlib must be non-interactive for headless environments)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paper / fallback values
# (median of 10 runs, 5 warm-up iterations, NVIDIA A100 40 GB, CUDA 12.4)
# ---------------------------------------------------------------------------
PAPER_VALUES = {
    # (model_key, input_seq_len): (ttft_s, tpot_s)
    ("qwen",   1024):  (0.013823878000039258,  0.012523412535152545),
    ("qwen",  32768):  (1.3003163600005792,    0.04180457996875475),
    ("mamba2", 1024):  (0.0260057350005809,    0.013800665003898871),
    ("mamba2", 32768): (0.49099710100017546,   0.013697560640622441),
}

# Inference scenarios shown in the figure: (input_tokens, output_tokens)
SCENARIOS = [(1024, 256), (32768, 256)]

MODEL_KEYS   = ["qwen",         "mamba2"]
MODEL_LABELS = ["Qwen2.5-0.5B", "Mamba2-780m"]
COLORS       = ["#FF776E",      "#00C3C5"]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _model_key(name: str) -> str | None:
    """Map a raw model_name string from the CSV to 'qwen' or 'mamba2'."""
    name = name.lower()
    if "mamba2" in name:
        return "mamba2"
    if "qwen" in name:
        return "qwen"
    return None


def _load_tpot_csv(csv_path: str) -> dict:
    """
    Load TTFT and TPOT values from ``tpot_logs/tpot_times.csv``.

    The CSV is written by ``profile_model_generate`` /
    ``profile_model_mamba_generate`` in ``profiling/eval.py``.

    Relevant columns
    ----------------
    model_name          : str  – e.g. ``gen_qwen25-instruct_cuda_256_1024``
    input_seq_length    : int  – prefill length in tokens
    prefill_time_seconds: float – TTFT (seconds)
    tpot_seconds        : float – time per output token (seconds)

    Returns
    -------
    dict mapping ``(model_key, input_seq_len)`` →
        ``{"ttft": float, "tpot": float}``
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        result = {}
        valid_seq_lens = {s for s, _ in SCENARIOS}
        for _, row in df.iterrows():
            key = _model_key(str(row.get("model_name", "")))
            sl  = int(row.get("input_seq_length", 0))
            if key is None or sl not in valid_seq_lens:
                continue
            result[(key, sl)] = {
                "ttft": float(row.get("prefill_time_seconds", 0.0)),
                "tpot": float(row.get("tpot_seconds", 0.0)),
            }
        return result
    except Exception as exc:
        warnings.warn(f"Could not load TPOT CSV ({csv_path}): {exc}")
        return {}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def assemble_data(tpot_csv: str | None) -> tuple[dict, dict]:
    """
    Return (ttft, tpot) dicts mapping model_key → list of values per scenario.

    Falls back to PAPER_VALUES for any missing (model_key, seq_len) entry.
    """
    csv_data = (
        _load_tpot_csv(tpot_csv)
        if tpot_csv and os.path.isfile(tpot_csv)
        else {}
    )
    if not csv_data:
        warnings.warn("CSV not found or empty — using hard-coded paper values.")

    ttft: dict[str, list[float]] = {k: [] for k in MODEL_KEYS}
    tpot: dict[str, list[float]] = {k: [] for k in MODEL_KEYS}

    for key in MODEL_KEYS:
        for seq_len, _ in SCENARIOS:
            paper_ttft, paper_tpot = PAPER_VALUES[(key, seq_len)]
            entry = csv_data.get((key, seq_len), {})
            ttft[key].append(entry.get("ttft", paper_ttft))
            tpot[key].append(entry.get("tpot", paper_tpot))

    return ttft, tpot


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _make_data_sets(ttft: dict, tpot: dict) -> list[list[float]]:
    """Ordered list of [model0_val, model1_val] for each subplot panel."""
    return [
        [ttft[k][0] for k in MODEL_KEYS],   # TTFT short context
        [ttft[k][1] for k in MODEL_KEYS],   # TTFT long context
        [tpot[k][0] for k in MODEL_KEYS],   # TPOT short context
        [tpot[k][1] for k in MODEL_KEYS],   # TPOT long context
    ]


def plot_annotated(ttft: dict, tpot: dict, out_dir: str) -> None:
    """Annotated bar chart — useful for inspection and the README."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    bar_width  = 0.5
    x          = np.arange(len(MODEL_LABELS))

    titles = [
        f"TTFT ({SCENARIOS[0][0]}, {SCENARIOS[0][1]})",
        f"TTFT ({SCENARIOS[1][0]}, {SCENARIOS[1][1]})",
        f"TPOT ({SCENARIOS[0][0]}, {SCENARIOS[0][1]})",
        f"TPOT ({SCENARIOS[1][0]}, {SCENARIOS[1][1]})",
    ]
    # Format strings that match the magnitude of each panel
    fmts = [".4f", ".3f", ".4f", ".4f"]

    for ax, title, vals, fmt in zip(axes, titles, _make_data_sets(ttft, tpot), fmts):
        bars = ax.bar(x, vals, bar_width, color=["#CD817C", "#254E70"], alpha=0.8)
        ax.set_ylabel("Time (s)", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=7, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val,
                f"{val:{fmt}}s",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    path = os.path.join(out_dir, "intro_ttft_tpot_annotated.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved annotated figure  →  {path}")


def plot_paper(ttft: dict, tpot: dict, out_dir: str) -> None:
    """Clean publication figure — no axis tick labels, 300 DPI."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5))
    bar_width  = 0.35
    x          = np.arange(len(MODEL_LABELS))

    for ax, vals in zip(axes, _make_data_sets(ttft, tpot)):
        ax.bar(x, vals, bar_width, color=COLORS)
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis="y", which="both", left=False,  right=False, labelleft=False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.2, hspace=0.5)
    path = os.path.join(out_dir, "intro_ttft_tpot.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved publication figure →  {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Figure 1 (TTFT & TPOT, Qwen2.5-0.5B vs Mamba2-780m)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tpot_csv",
        default=None,
        help="Path to src/tpot_logs/tpot_times.csv produced by collect_fig1_data.py",
    )
    p.add_argument(
        "--out_dir",
        default=".",
        help="Directory where the PNG files will be saved",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ttft, tpot = assemble_data(args.tpot_csv)

    # Print summary table matching the README expected output
    print("\n=== Fig 1 Data ===")
    for ki, kl in zip(MODEL_KEYS, MODEL_LABELS):
        for i, (sl, ot) in enumerate(SCENARIOS):
            print(f"{kl:<18} TTFT ({sl:>5}, {ot}): {ttft[ki][i]:.4f} s")
    for ki, kl in zip(MODEL_KEYS, MODEL_LABELS):
        for i, (sl, ot) in enumerate(SCENARIOS):
            print(f"{kl:<18} TPOT ({sl:>5}, {ot}): {tpot[ki][i]:.4f} s")
    print()

    plot_annotated(ttft, tpot, args.out_dir)
    plot_paper(ttft, tpot, args.out_dir)


if __name__ == "__main__":
    main()

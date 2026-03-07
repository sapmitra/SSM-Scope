'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-06
 # @ Description: Collect GPU memory-footprint data for Figure 5b (Jetson).
 #                See the README for usage instructions.
 '''

"""
Collect GPU memory footprint data for Figure 5b (Memory Footprint Analysis on NVIDIA Jetson).

Each invocation profiles *one model* at *one sequence length* during the
**prefill phase** and appends a row to ``src/memory/memory_footprints_jetson.csv``.

Run all combinations listed below (grouped by required venv) before plotting.

Sequence lengths per model (constrained by 8 GB Jetson unified memory)
-----------------------------------------------------------------------
qwen        : 256 … 16384       (7 points)
llama3_2    : 256 … 8192        (6 points)
mamba_790m  : 256 … 32768       (9 points)
mamba2_780m : 256 … 32768       (9 points)
falcon_h1   : 256 … 32768       (9 points)
zamba2      : 256 … 8192        (6 points)

Usage (all commands from ``<repo_root>/src/``)
----------------------------------------------

### Transformer venv  (qwen, llama3_2)
    source ~/.venvs/torch_transformers_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \\
        --model qwen --seq_len 256 --device cuda

    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \\
        --model llama3_2 --seq_len 256 --device cuda

### Mamba venv  (mamba_790m, mamba2_780m)
    source ~/.venvs/torch_ssm_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \\
        --model mamba_790m --seq_len 256 --device cuda

    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \\
        --model mamba2_780m --seq_len 256 --device cuda

### Falcon venv  (falcon_h1, zamba2)
    source ~/.venvs/torch_falcon_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \\
        --model falcon_h1 --seq_len 256 --device cuda

    python ../ispass_ae/scripts/paper_figures/Fig_5b/collect_fig5b_data.py \\
        --model zamba2 --seq_len 256 --device cuda

Output
------
``src/memory/memory_footprints_jetson.csv``
    Rows are *appended*; re-running the same (model, seq_len) combination
    adds a duplicate row — filter or de-duplicate when plotting if needed.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — make ``src/`` importable regardless of the cwd at call time.
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_script_dir, "../../../../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

# Ensure cwd is src/ so that intra-package relative paths in profile_runner
# and profiling.eval resolve correctly.
os.chdir(_src)

from memory.mem_footprint import model_prefill  # noqa: E402  (import after sys.path edit)


# ---------------------------------------------------------------------------
# CSV output filename (Jetson-specific, separate from the RTX CSV)
# ---------------------------------------------------------------------------
JETSON_CSV = "memory_footprints_jetson.csv"


# ---------------------------------------------------------------------------
# Model registry  (model_key → (csv_model_name, HF/hub model config))
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # Transformer models — torch_transformers_ispass venv
    "qwen":        ("qwen25-instruct",  "Qwen/Qwen2.5-0.5B-Instruct"),
    "llama3_2":    ("llama3_2",         "meta-llama/Llama-3.2-1B-Instruct"),
    # Pure SSM models — torch_ssm_ispass venv
    "mamba_790m":  ("mamba-790m",       "state-spaces/mamba-790m"),
    "mamba2_780m": ("mamba2-780m",      "state-spaces/mamba2-780m"),
    # Hybrid SSM — torch_falcon_ispass venv
    "falcon_h1":   ("falcon-h1-0.5b",   "tiiuae/Falcon-H1-0.5B-Base"),
    # Hybrid SSM — torch_falcon_ispass venv  (same venv as Falcon-H1)
    "zamba2":      ("zamba2",           "Zyphra/Zamba2-1.2B-Instruct-v2"),
}

# Sequence lengths used in the paper for each model on the Jetson (8 GB)
PAPER_SEQ_LENGTHS: dict[str, list[int]] = {
    "qwen":
        [256, 512, 1024, 2048, 4096, 8192, 16384],
    "llama3_2":
        [256, 512, 1024, 2048, 4096, 8192],
    "mamba_790m":
        [256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768],
    "mamba2_780m":
        [256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768],
    "falcon_h1":
        [256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768],
    "zamba2":
        [256, 512, 1024, 2048, 4096, 8192],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect memory footprint for Figure 5b / Jetson (one model × one seq_len).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help=(
            "Model to profile. "
            "Transformer venv: qwen | llama3_2. "
            "Mamba venv: mamba_790m | mamba2_780m. "
            "Falcon venv: falcon_h1 | zamba2."
        ),
    )
    p.add_argument(
        "--seq_len",
        type=int,
        required=True,
        help=(
            "Prefill (input) sequence length in tokens. "
            "See PAPER_SEQ_LENGTHS in this script for the per-model ranges used in the paper."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size — keep at 1 for single-sample profiling.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Compute device ('cuda' or 'cpu').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name, model_config = MODEL_REGISTRY[args.model]

    print("=" * 60)
    print("Fig 5b — collect GPU memory footprint (Jetson)")
    print("=" * 60)
    print(f"  model key    : {args.model}")
    print(f"  model_name   : {model_name}  (CSV key)")
    print(f"  model_config : {model_config}")
    print(f"  seq_len      : {args.seq_len}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  device       : {args.device}")
    print(f"  csv_output   : {JETSON_CSV}")
    print()

    model_prefill(
        model_name=model_name,
        model_config=model_config,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        csv_filename=JETSON_CSV,
    )

    print()
    csv_out = os.path.join(_src, "memory", JETSON_CSV)
    print(f"Row appended → {csv_out}")


if __name__ == "__main__":
    main()

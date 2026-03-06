'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-06
 # @ Description: Collect throughput data for Figure 6b of the paper.  See the README for usage.
 '''

"""
Collect overall throughput data for Figure 6b.

Each invocation profiles one model at one sequence length and appends a row to
``src/throughput_logs/generation_times.csv``.  Run all combinations (three
models × seven sequence lengths) before plotting. Qwen2.5-0.5B and Falcon-H1
require the Transformer venv; Mamba2 requires the Mamba venv.

Usage (from ``<repo_root>/src/``):

    # --- Transformer venv -----------------------------------------------
    source ~/.venvs/torch_transformers_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_6b/collect_fig6b_data.py \\
        --model qwen --seq_len 1024 --max_new_tokens 256 --device cuda

    python ../ispass_ae/scripts/paper_figures/Fig_6b/collect_fig6b_data.py \\
        --model falcon --seq_len 1024 --max_new_tokens 256 --device cuda

    # --- Mamba venv ---------------------------------------------------------
    source ~/.venvs/torch_ssm_ispass/bin/activate
    cd <repo_root>/src

    python ../ispass_ae/scripts/paper_figures/Fig_6b/collect_fig6b_data.py \\
        --model mamba2 --seq_len 1024 --max_new_tokens 256 --device cuda

Output CSV (relative to where the script is invoked, i.e. ``src/``):

    throughput_logs/generation_times.csv
        model_name, input_seq_length, output_tokens,
        prefill_time_seconds (= TTFT),
        decode_time_seconds, total_time_seconds,
        tpot_seconds (= TPOT),
        throughput_tokens_per_sec, device, timestamp
"""

import argparse
import sys
import os

# Allow running from src/ or from repo root: insert src/ onto sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_script_dir, "../../../../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


SEQ_LENGTHS = [1024, 2048, 4096, 8192, 16384, 24576, 32768]


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect throughput data for Figure 6b",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["qwen", "mamba2", "falcon"],
        help=(
            "Model to profile: "
            "'qwen' → Qwen2.5-0.5B-Instruct, "
            "'mamba2' → Mamba2-780m, "
            "'falcon' → Falcon-H1-0.5B-Base"
        ),
    )
    p.add_argument(
        "--seq_len",
        type=int,
        required=True,
        help=f"Prefill (input) sequence length in tokens. Typical values: {SEQ_LENGTHS}",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Number of tokens to generate (output length)",
    )
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    p.add_argument(
        "--weights",
        default=None,
        help="Optional path / HF hub ID to override the default model weights",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.model == "qwen":
        from models.profile_runner import qwen25_instruct_generate_throughput

        print(
            f"[Fig6b] Profiling Qwen2.5-0.5B-Instruct"
            f"  seq_len={args.seq_len}"
            f"  max_new_tokens={args.max_new_tokens}"
            f"  device={args.device}"
        )
        qwen25_instruct_generate_throughput(
            seq_len=args.seq_len,
            max_num_tokens=args.max_new_tokens,
            device=args.device,
            weights=args.weights,
        )

    elif args.model == "mamba2":
        from models.profile_runner import mamba2_generate_throughput

        print(
            f"[Fig6b] Profiling Mamba2-780m"
            f"  seq_len={args.seq_len}"
            f"  max_new_tokens={args.max_new_tokens}"
            f"  device={args.device}"
        )
        mamba2_generate_throughput(
            seq_len=args.seq_len,
            max_num_tokens=args.max_new_tokens,
            device=args.device,
            weights=args.weights,
        )

    else:  # falcon
        from models.profile_runner import falcon_h1_generate_throughput

        print(
            f"[Fig6b] Profiling Falcon-H1-0.5B-Base"
            f"  seq_len={args.seq_len}"
            f"  max_new_tokens={args.max_new_tokens}"
            f"  device={args.device}"
        )
        falcon_h1_generate_throughput(
            seq_len=args.seq_len,
            max_num_tokens=args.max_new_tokens,
            device=args.device,
            weights=args.weights,
        )

    print("[Fig6b] Done.  Results appended to throughput_logs/generation_times.csv")


if __name__ == "__main__":
    main()

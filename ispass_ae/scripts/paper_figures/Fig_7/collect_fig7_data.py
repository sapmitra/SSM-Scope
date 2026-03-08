'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-07
 # @ Description: Collect per-operator GPU kernel time breakdown for Figure 7.
 #                Profiles one model at one sequence length and writes the
 #                summary CSVs consumed by plot_fig7.py.
 '''

"""
Collect operator-breakdown profiling data for Figure 7.

Each invocation profiles one model (mamba-130m or mamba2-130m) at one
sequence length and writes per-operator CSV files to ``--out_dir``.

The output directory layout mirrors the existing ``profile_data/`` structure::

    <out_dir>/
        <model>_cuda_1_<seq_len>/
            <model>_cuda_1_<seq_len>.csv        (raw per-op timing)
            gemm.csv
            non_gemm.csv
            ssm_scan.csv
            summary_<model>_cuda_1_<seq_len>.csv
            pct_<model>_cuda_1_<seq_len>.csv
            gng_<model>_cuda_1_<seq_len>.csv
            gng_pct_<model>_cuda_1_<seq_len>.csv
            gng_ssm_<model>_cuda_1_<seq_len>.csv
            gng_ssm_pct_<model>_cuda_1_<seq_len>.csv
            (optional) <model>_cuda_1_<seq_len>.json   chrome trace

Usage (from ``<repo_root>/src/``):

    source ~/.venvs/torch_ssm_ispass/bin/activate
    cd <repo_root>/src

    # mamba-130m at seq_len=1024
    python ../ispass_ae/scripts/paper_figures/Fig_7/collect_fig7_data.py \\
        --model mamba --seq_len 1024 --device cuda \\
        --out_dir ../fig7_profile_data

    # mamba2-130m at seq_len=1024
    python ../ispass_ae/scripts/paper_figures/Fig_7/collect_fig7_data.py \\
        --model mamba2 --seq_len 1024 --device cuda \\
        --out_dir ../fig7_profile_data

See ``gen_fig7.sh`` to run all models / sequence lengths in one command.
"""

import argparse
import os
import sys

# Allow running from src/ or from the Fig_7 directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_script_dir, "../../../../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Sequence lengths profiled in the paper
# ---------------------------------------------------------------------------
SEQ_LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

MODEL_WEIGHTS = {
    "mamba":  "state-spaces/mamba-130m",
    "mamba2": "state-spaces/mamba2-130m",
}

MODEL_PROFILE_KEY = {
    "mamba":  "mamba-ops-profile",
    "mamba2": "mamba2-ops-profile",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect operator-breakdown profiling data for Figure 7.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_WEIGHTS.keys()),
        help="Model to profile: 'mamba' → mamba-130m, 'mamba2' → mamba2-130m",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        required=True,
        choices=SEQ_LENGTHS,
        help="Prefill (input) sequence length in tokens",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to local model weights (optional, overrides default HuggingFace checkpoint)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_src, "profile_logs"),
        help="Root output directory for operator-breakdown CSV and chrome trace files",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (must be 1 for mamba_ssm models)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model_key = args.model
    seq_len   = args.seq_len
    device    = args.device
    weights   = args.weights or MODEL_WEIGHTS[model_key]
    out_dir   = os.path.abspath(args.out_dir)
    batch_size = args.batch_size

    os.makedirs(out_dir, exist_ok=True)

    print(
        f"=== Fig 7 data collection ===\n"
        f"  model    : {model_key} ({weights})\n"
        f"  seq_len  : {seq_len}\n"
        f"  device   : {device}\n"
        f"  out_dir  : {out_dir}\n"
    )

    # Change to src/ so relative paths inside profile_model_mamba resolve correctly
    os.chdir(_src)

    from models.profile_runner import MambaProfile, custom_ops, NUM_RUNS, EXPORT

    model_name = "mamba-130m" if model_key == "mamba" else "mamba2-130m"
    profile = MambaProfile(model_name, weights, device)
    profile.eval_profile(
        seq_len=seq_len,
        batch_size=batch_size,
        num_runs=NUM_RUNS,
        export=EXPORT,
        custom_ops=custom_ops,
        profile_out_dir=out_dir,
    )
    del profile
    print(f"\nDone. Profile data written to {out_dir}/{model_name}_{device}_{batch_size}_{seq_len}/")


if __name__ == "__main__":
    main()

'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-07
 # @ Description: Collect per-operator GPU kernel time breakdown for Figure 8.
 #                Profiles one hybrid SSM model at one sequence length and writes
 #                the summary CSVs consumed by plot_fig8.py.
 '''

"""
Collect operator-breakdown profiling data for Figure 8.

Each invocation profiles one model (hymba or zamba2) at one sequence length
and writes per-operator CSV files to ``--out_dir``.

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

    source ~/.venvs/torch_falcon_ispass/bin/activate
    cd <repo_root>/src

    # Hymba at seq_len=1024
    python ../ispass_ae/scripts/paper_figures/Fig_8/collect_fig8_data.py \\
        --model hymba --seq_len 1024 --device cuda \\
        --out_dir ../src/profile_logs

    # Zamba2 at seq_len=1024
    python ../ispass_ae/scripts/paper_figures/Fig_8/collect_fig8_data.py \\
        --model zamba2 --seq_len 1024 --device cuda \\
        --out_dir ../src/profile_logs

See ``gen_fig8.sh`` to run all models / sequence lengths in one command.
"""

import argparse
import os
import sys

# Allow running from src/ or from the Fig_8 directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_script_dir, "../../../../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Models and their available sequence lengths (from the paper notebook)
# ---------------------------------------------------------------------------
MODEL_SEQ_LENGTHS = {
    "hymba":  [256, 512, 1024, 2048, 4096, 8192, 16384],
    "zamba2": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
}

# All unique sequence lengths across both models
ALL_SEQ_LENGTHS = sorted(set(sl for sls in MODEL_SEQ_LENGTHS.values() for sl in sls))

MODEL_WEIGHTS = {
    "hymba":  "nvidia/Hymba-1.5B-Instruct",
    "zamba2": "Zyphra/Zamba2-1.2B-Instruct-v2",
}

MODEL_PROFILE_KEY = {
    "hymba":  "hymba-ops-profile",
    "zamba2": "zamba2-ops-profile",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect operator-breakdown profiling data for Figure 8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_WEIGHTS.keys()),
        help="Model to profile: 'hymba' → Hymba-1.5B-Instruct, 'zamba2' → Zamba2-1.2B-Instruct-v2",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        required=True,
        choices=ALL_SEQ_LENGTHS,
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
        help="Batch size",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model_key  = args.model
    seq_len    = args.seq_len
    device     = args.device
    weights    = args.weights or MODEL_WEIGHTS[model_key]
    out_dir    = os.path.abspath(args.out_dir)
    batch_size = args.batch_size

    # Validate: warn if the seq_len is not in the paper list for this model
    if seq_len not in MODEL_SEQ_LENGTHS[model_key]:
        print(
            f"  [warn] seq_len={seq_len} is not in the standard paper list "
            f"for {model_key} ({MODEL_SEQ_LENGTHS[model_key]}). Proceeding anyway."
        )

    os.makedirs(out_dir, exist_ok=True)

    print(
        f"=== Fig 8 data collection ===\n"
        f"  model    : {model_key} ({weights})\n"
        f"  seq_len  : {seq_len}\n"
        f"  device   : {device}\n"
        f"  out_dir  : {out_dir}\n"
    )

    # Change to src/ so relative paths inside profile_model resolve correctly
    os.chdir(_src)

    from models.profile_runner import LMProfile, custom_ops, NUM_RUNS, EXPORT

    profile = LMProfile(model_key, weights, device)
    profile.eval_profile(
        seq_len=seq_len,
        batch_size=batch_size,
        num_runs=NUM_RUNS,
        export=EXPORT,
        custom_ops=custom_ops,
        profile_out_dir=out_dir,
    )
    del profile
    print(f"\nDone. Profile data written to {out_dir}/{model_key}_{device}_{batch_size}_{seq_len}/")


if __name__ == "__main__":
    main()

'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-08
 # @ Description: Collect per-operator GPU kernel time breakdown for Figure 9b.
 #                Profiles one model at a fixed sequence length and writes the
 #                summary CSVs consumed by plot_fig9b.py.
 #                Covers all model families: Transformer, SSM, and Hybrid.
 '''

"""
Collect operator-breakdown profiling data for Figure 9b.

Each invocation profiles **one model** at **one sequence length** and writes
per-operator CSV files to ``--out_dir``.

Figure 9b compares the GPU kernel-time breakdown across **all model families**
(Transformer, SSM, Hybrid) on two devices (Desktop GPU and NVIDIA Jetson Orin).
This script is run identically on both devices — the only difference is the
``--out_dir`` (or implicitly, which machine it runs on).

Workflow
--------
1. **Desktop:** run this script for every model → ``src/profile_logs/``
2. **Jetson:**  run this script for every model → ``src/profile_logs/``
3. **Transfer:** ``scp -r jetson:src/profile_logs/ workstation:src/profile_logs_jetson/``
4. **Plot:** ``python plot_fig9b.py``

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

Model families and required venvs
----------------------------------
- Transformer (``~/.venvs/torch_transformers_ispass``):
    gpt-neo-125m, tinyllama, llama3_2, qwen25-instruct, qwen25-1.5b-instruct
- SSM (``~/.venvs/torch_ssm_ispass``):
    mamba, mamba2
- Hybrid (``~/.venvs/torch_falcon_ispass``):
    hymba, zamba2

Usage (from ``<repo_root>/src/``):

    # gpt-neo-125m at seq_len=1024
    source ~/.venvs/torch_transformers_ispass/bin/activate
    python ../ispass_ae/scripts/paper_figures/Fig_9b/collect_fig9b_data.py \\
        --model gpt-neo-125m --seq_len 1024 --device cuda

    # mamba-130m at seq_len=1024
    source ~/.venvs/torch_ssm_ispass/bin/activate
    python ../ispass_ae/scripts/paper_figures/Fig_9b/collect_fig9b_data.py \\
        --model mamba --seq_len 1024 --device cuda

    # hymba at seq_len=1024
    source ~/.venvs/torch_falcon_ispass/bin/activate
    python ../ispass_ae/scripts/paper_figures/Fig_9b/collect_fig9b_data.py \\
        --model hymba --seq_len 1024 --device cuda

See ``gen_fig9b.sh`` to run all models in one command.
"""

import argparse
import os
import sys

# Allow running from src/ or from the Fig_9b directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_script_dir, "../../../../src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Sequence length for the cross-device comparison (single fixed point)
DEFAULT_SEQ_LEN = 1024

# HuggingFace model IDs for each model key
MODEL_WEIGHTS = {
    # Transformer models
    "gpt-neo-125m":         "EleutherAI/gpt-neo-125m",
    "tinyllama":            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3_2":             "meta-llama/Llama-3.2-1B-Instruct",
    "qwen25-instruct":      "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen25-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    # SSM models (require mamba_ssm — use torch_ssm_ispass venv)
    "mamba":  "state-spaces/mamba-130m",
    "mamba2": "state-spaces/mamba2-130m",
    # Hybrid models (require torch_falcon_ispass venv, transformers>=4.48)
    "hymba":  "nvidia/Hymba-1.5B-Instruct",
    "zamba2": "Zyphra/Zamba2-1.2B-Instruct-v2",
}

# Output directory name used by profile_runner (model_name field)
MODEL_OUTPUT_NAME = {
    "gpt-neo-125m":         "gpt-neo-125m",
    "tinyllama":            "tinyllama",
    "llama3_2":             "llama3_2",
    "qwen25-instruct":      "qwen25-instruct",
    "qwen25-1.5b-instruct": "qwen25-1.5b-instruct",
    "mamba":                "mamba-130m",
    "mamba2":               "mamba2-130m",
    "hymba":                "hymba",
    "zamba2":               "zamba2",
}

# Models that use MambaProfile (mamba_ssm library)
MAMBA_MODELS = {"mamba", "mamba2"}

# Models that use LMProfile (HuggingFace transformers)
LM_MODELS = set(MODEL_WEIGHTS.keys()) - MAMBA_MODELS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect operator-breakdown profiling data for Figure 9b.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_WEIGHTS.keys()),
        help=(
            "Model to profile.  Transformer models require torch_transformers_ispass venv; "
            "SSM models (mamba/mamba2) require torch_ssm_ispass venv; "
            "Hybrid models (hymba/zamba2) require torch_falcon_ispass venv."
        ),
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Prefill (input) sequence length in tokens.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on.",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Path to local model weights (optional).  "
            "Overrides the default HuggingFace checkpoint."
        ),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_src, "profile_logs"),
        help=(
            "Root output directory for operator-breakdown CSV and chrome trace files.  "
            "Use the same value when running on both Desktop and Jetson so both sets "
            "of CSVs share an identical layout (differ only by device performance)."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (must be 1 for mamba_ssm models).",
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
    model_name = MODEL_OUTPUT_NAME[model_key]

    os.makedirs(out_dir, exist_ok=True)

    print(
        f"=== Fig 9b data collection ===\n"
        f"  model      : {model_key} ({weights})\n"
        f"  output name: {model_name}\n"
        f"  seq_len    : {seq_len}\n"
        f"  device     : {device}\n"
        f"  out_dir    : {out_dir}\n"
    )

    # Change to src/ so relative imports inside profile_runner resolve correctly
    os.chdir(_src)

    from models.profile_runner import (
        LMProfile, MambaProfile, custom_ops, NUM_RUNS, EXPORT,
    )

    if model_key in MAMBA_MODELS:
        profile = MambaProfile(model_name, weights, device)
        profile.eval_profile(
            seq_len=seq_len,
            batch_size=batch_size,
            num_runs=NUM_RUNS,
            export=EXPORT,
            custom_ops=custom_ops,
            profile_out_dir=out_dir,
        )
    else:
        # LM models (Transformer and Hybrid)
        profile = LMProfile(model_name, weights, device)
        profile.eval_profile(
            seq_len=seq_len,
            batch_size=batch_size,
            num_runs=NUM_RUNS,
            export=EXPORT,
            custom_ops=custom_ops,
            profile_out_dir=out_dir,
        )

    del profile
    print(
        f"\nDone.  Profile data written to "
        f"{out_dir}/{model_name}_{device}_{batch_size}_{seq_len}/"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# gen_fig5b.sh — End-to-end script to collect memory-footprint data
#                on an NVIDIA Jetson and reproduce Figure 5b.
#
# Usage (from repo root):
#   bash ispass_ae/scripts/paper_figures/Fig_5b/gen_fig5b.sh
#
# Or from this directory:
#   bash gen_fig5b.sh
#
# Steps:
#   1. Transformer venv → profile Qwen2.5-0.5B, Llama-3.2-1B
#   2. Mamba venv       → profile Mamba-790m, Mamba2-780m
#   3. Falcon venv      → profile Falcon-H1-0.5B and Zamba2-1.2B
#   4. Transformer venv → generate the PNG figure
#
# The CSV is written (appended) to:
#   <repo_root>/src/memory/memory_footprints_jetson.csv
#
# The output PNG is written to this directory:
#   ispass_ae/scripts/paper_figures/Fig_5b/memory_footprint_jetson_ispass.png
#
# Notes:
#   - Each model is profiled once per sequence length.
#   - Re-running appends duplicate rows; run on a fresh CSV or de-duplicate.
#   - A single sequence length can be collected with collect_fig5b_data.py
#     directly (see its --help for options).
#   - Zamba2 runs in the same `torch_falcon_ispass` venv as Falcon-H1.
#   - Sequence lengths are shorter than Fig_5a due to the 8 GB Jetson memory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

COLLECT_SCRIPT="${SCRIPT_DIR}/collect_fig5b_data.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_fig5b.py"
CSV_PATH="${REPO_ROOT}/src/memory/memory_footprints_jetson.csv"
OUT_DIR="${SCRIPT_DIR}"

TRANSFORMER_VENV="${HOME}/.venvs/torch_transformers_ispass"
MAMBA_VENV="${HOME}/.venvs/torch_ssm_ispass"
FALCON_VENV="${HOME}/.venvs/torch_falcon_ispass"

# ---------------------------------------------------------------------------
# Per-model sequence-length arrays (match plotting_mem_footprint_jetson.ipynb)
# Shorter than Fig_5a due to the 8 GB Jetson unified-memory constraint.
# ---------------------------------------------------------------------------
QWEN_SEQ_LENS="256 512 1024 2048 4096 8192 16384"

LLAMA3_2_SEQ_LENS="256 512 1024 2048 4096 8192"

MAMBA_790M_SEQ_LENS="256 512 1024 2048 4096 8192 16384 24576 32768"

MAMBA2_780M_SEQ_LENS="256 512 1024 2048 4096 8192 16384 24576 32768"

FALCON_H1_SEQ_LENS="256 512 1024 2048 4096 8192 16384 24576 32768"

ZAMBA2_SEQ_LENS="256 512 1024 2048 4096 8192"

# ---------------------------------------------------------------------------
# Helper: profile one model across a list of sequence lengths
# ---------------------------------------------------------------------------
profile_model() {
    local model_key="$1"
    local seq_lens="$2"
    for seq_len in $seq_lens; do
        echo "  [${model_key}] seq_len=${seq_len} ..."
        python "${COLLECT_SCRIPT}" \
            --model  "${model_key}" \
            --seq_len "${seq_len}" \
            --device cuda
    done
}

# ---------------------------------------------------------------------------
# Step 1 — Transformer models
# ---------------------------------------------------------------------------
echo "================================================================"
echo "Step 1: Transformer models (Qwen2.5-0.5B, Llama-3.2-1B)"
echo "        venv: ${TRANSFORMER_VENV}"
echo "================================================================"
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

echo ""
echo "--- 1a. Qwen2.5-0.5B-Instruct ---"
profile_model "qwen" "${QWEN_SEQ_LENS}"

echo ""
echo "--- 1b. Llama-3.2-1B-Instruct ---"
profile_model "llama3_2" "${LLAMA3_2_SEQ_LENS}"

deactivate

# ---------------------------------------------------------------------------
# Step 2 — Mamba models
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 2: Mamba models (Mamba-790m, Mamba2-780m)"
echo "        venv: ${MAMBA_VENV}"
echo "================================================================"
source "${MAMBA_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

echo ""
echo "--- 2a. Mamba-790m ---"
profile_model "mamba_790m" "${MAMBA_790M_SEQ_LENS}"

echo ""
echo "--- 2b. Mamba2-780m ---"
profile_model "mamba2_780m" "${MAMBA2_780M_SEQ_LENS}"

deactivate

# ---------------------------------------------------------------------------
# Step 3 — Falcon-H1 and Zamba2 (hybrid SSMs)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 3: Falcon-H1-0.5B-Base and Zamba2-1.2B"
echo "        venv: ${FALCON_VENV}"
echo "================================================================"
source "${FALCON_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

echo ""
echo "--- 3a. Falcon-H1-0.5B ---"
profile_model "falcon_h1" "${FALCON_H1_SEQ_LENS}"

echo ""
echo "--- 3b. Zamba2 ---"
profile_model "zamba2" "${ZAMBA2_SEQ_LENS}"

deactivate

# ---------------------------------------------------------------------------
# Step 4 — Plot
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 4: Generate Figure 5b"
echo "        venv: ${TRANSFORMER_VENV}"
echo "================================================================"
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}"

python "${PLOT_SCRIPT}" \
    --csv_path "${CSV_PATH}" \
    --out_dir  "${OUT_DIR}"

deactivate

echo ""
echo "Done.  Output:"
echo "  ${OUT_DIR}/memory_footprint_jetson_ispass.png"

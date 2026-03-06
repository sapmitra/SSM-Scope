#!/usr/bin/env bash
# gen_fig6b.sh — End-to-end script to collect data and reproduce Figure 6b.
#
# Usage (from repo root):
#   bash ispass_ae/scripts/paper_figures/Fig_6b/gen_fig6b.sh
#
# Or from this directory:
#   bash gen_fig6b.sh
#
# The script:
#   1. Activates the Transformer venv and profiles Qwen2.5-0.5B-Instruct across
#      all 7 sequence lengths.
#   2. Activates the Mamba venv and profiles Mamba2-780m across all 7 sequence
#      lengths.
#   3. Re-activates the Transformer venv and profiles Falcon-H1-0.5B-Base across
#      all 7 sequence lengths.
#   4. Generates the final PNG files.
#
# Output PNGs are written to the same directory as this script:
#   ispass_ae/scripts/paper_figures/Fig_6b/overall_throughput_comparison.png
#   ispass_ae/scripts/paper_figures/Fig_6b/overall_throughput_annotated.png
#
# Throughput data CSV is written to:
#   src/throughput_logs/generation_times.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

COLLECT_SCRIPT="${SCRIPT_DIR}/collect_fig6b_data.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_fig6b.py"
THROUGHPUT_CSV="${REPO_ROOT}/src/throughput_logs/generation_times.csv"
OUT_DIR="${SCRIPT_DIR}"

TRANSFORMER_VENV="${HOME}/.venvs/torch_transformers_ispass"
MAMBA_VENV="${HOME}/.venvs/torch_ssm_ispass"

SEQ_LENGTHS=(1024 2048 4096 8192 16384 24576 32768)
MAX_NEW_TOKENS=256

# ---------------------------------------------------------------------------
echo "=== Step 1a: Qwen2.5-0.5B-Instruct — Transformer venv ==="
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ in "${SEQ_LENGTHS[@]}"; do
    echo "  [Qwen] seq_len=${SEQ} ..."
    python "${COLLECT_SCRIPT}" \
        --model qwen \
        --seq_len "${SEQ}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --device cuda
done

deactivate

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1b: Mamba2-780m — Mamba venv ==="
source "${MAMBA_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ in "${SEQ_LENGTHS[@]}"; do
    echo "  [Mamba2] seq_len=${SEQ} ..."
    python "${COLLECT_SCRIPT}" \
        --model mamba2 \
        --seq_len "${SEQ}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --device cuda
done

deactivate

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1c: Falcon-H1-0.5B-Base — Transformer venv ==="
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ in "${SEQ_LENGTHS[@]}"; do
    echo "  [Falcon-H1] seq_len=${SEQ} ..."
    python "${COLLECT_SCRIPT}" \
        --model falcon \
        --seq_len "${SEQ}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --device cuda
done

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Plot Figure 6b ==="
cd "${REPO_ROOT}"
python "${PLOT_SCRIPT}" \
    --throughput_csv "${THROUGHPUT_CSV}" \
    --out_dir "${OUT_DIR}"

deactivate

echo ""
echo "=== Done — PNGs written to ${OUT_DIR} ==="

/**
 * @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 * @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 * @ License: MIT License
 * @ Create Time: 2026-03-05 02:55:15
 * @ Description:
 */

#!/usr/bin/env bash
# gen_fig1.sh — End-to-end script to collect data and reproduce Figure 1.
#
# Usage (from repo root):
#   bash gen_fig1.sh
#
# The script:
#   1. Activates the Transformer venv and profiles Qwen2.5-0.5B (short + long context).
#   2. Activates the Mamba venv and profiles Mamba2-780m (short + long context).
#   3. Activates the Transformer venv and generates the final PNG files.
#
# Output PNGs are written to:
#   ispass_ae/scripts/paper_figures/Fig_1/intro_ttft_tpot.png
#   ispass_ae/scripts/paper_figures/Fig_1/intro_ttft_tpot_annotated.png

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_SCRIPT="${REPO_ROOT}/ispass_ae/scripts/paper_figures/Fig_1/collect_fig1_data.py"
PLOT_SCRIPT="${REPO_ROOT}/ispass_ae/scripts/paper_figures/Fig_1/plot_fig1.py"
TPOT_CSV="${REPO_ROOT}/src/tpot_logs/tpot_times.csv"
OUT_DIR="${REPO_ROOT}/ispass_ae/scripts/paper_figures/Fig_1"

TRANSFORMER_VENV="${HOME}/.venvs/torch_transformers_ispass"
MAMBA_VENV="${HOME}/.venvs/torch_ssm_ispass"

echo "=== Step 1a: Qwen2.5-0.5B — Transformer venv ==="
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

echo "  [Qwen] short context (seq_len=1024) ..."
python "${COLLECT_SCRIPT}" --model qwen --seq_len 1024  --max_new_tokens 256 --device cuda

echo "  [Qwen] long context  (seq_len=32768) ..."
python "${COLLECT_SCRIPT}" --model qwen --seq_len 32768 --max_new_tokens 256 --device cuda

deactivate

echo ""
echo "=== Step 1b: Mamba2-780m — Mamba venv ==="
source "${MAMBA_VENV}/bin/activate"

echo "  [Mamba2] short context (seq_len=1024) ..."
python "${COLLECT_SCRIPT}" --model mamba2 --seq_len 1024  --max_new_tokens 256 --device cuda

echo "  [Mamba2] long context  (seq_len=32768) ..."
python "${COLLECT_SCRIPT}" --model mamba2 --seq_len 32768 --max_new_tokens 256 --device cuda

deactivate

echo ""
echo "=== Step 2: Plot Figure 1 ==="
source "${TRANSFORMER_VENV}/bin/activate"
cd "${REPO_ROOT}"

python "${PLOT_SCRIPT}" \
    --tpot_csv "${TPOT_CSV}" \
    --out_dir  "${OUT_DIR}"

deactivate

echo ""
echo "Done. Output files:"
echo "  ${OUT_DIR}/intro_ttft_tpot.png"
echo "  ${OUT_DIR}/intro_ttft_tpot_annotated.png"

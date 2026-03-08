#!/usr/bin/env bash
# gen_fig8.sh — End-to-end script to collect profiling data and reproduce Figure 8.
#
# Usage (from repo root):
#   bash ispass_ae/scripts/paper_figures/Fig_8/gen_fig8.sh
#
# Or from this directory:
#   bash gen_fig8.sh
#
# The script:
#   1. Activates the Falcon/Hybrid venv and profiles Hymba-1.5B across all
#      sequence lengths.
#   2. Activates the Falcon/Hybrid venv and profiles Zamba2-1.2B across all
#      sequence lengths.
#   3. Activates the Falcon/Hybrid venv and generates the final PNG files.
#
# Output profile CSVs are written to:
#   <REPO_ROOT>/src/profile_logs/
#
# Output PNGs are written to the same directory as this script:
#   ispass_ae/scripts/paper_figures/Fig_8/fig8_ops_breakdown.png
#   ispass_ae/scripts/paper_figures/Fig_8/fig8_ops_breakdown_annotated.png
#   … (per-model variants — see plot_fig8.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

COLLECT_SCRIPT="${SCRIPT_DIR}/collect_fig8_data.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_fig8.py"

# Profile data is written to src/profile_logs
PROFILE_DATA_DIR="${REPO_ROOT}/src/profile_logs"

OUT_DIR="${SCRIPT_DIR}"

# Falcon/Hybrid venv — required for Hymba and Zamba2 (needs mamba_ssm + transformers>=4.48)
FALCON_VENV="${HOME}/.venvs/torch_falcon_ispass"

# Sequence lengths for each model (matching the notebook)
HYMBA_SEQ_LENGTHS=(256 512 1024 2048 4096 8192 16384)
ZAMBA2_SEQ_LENGTHS=(256 512 1024 2048 4096 8192 16384 32768)

mkdir -p "${PROFILE_DATA_DIR}"

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Profile Hymba-1.5B-Instruct — Falcon/Hybrid venv ==="
source "${FALCON_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ_LEN in "${HYMBA_SEQ_LENGTHS[@]}"; do
    echo "  [hymba] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" \
        --model hymba \
        --seq_len "${SEQ_LEN}" \
        --device cuda \
        --out_dir "${PROFILE_DATA_DIR}"
done

deactivate

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Profile Zamba2-1.2B-Instruct-v2 — Falcon/Hybrid venv ==="
source "${FALCON_VENV}/bin/activate"
cd "${REPO_ROOT}/src"

for SEQ_LEN in "${ZAMBA2_SEQ_LENGTHS[@]}"; do
    echo "  [zamba2] seq_len=${SEQ_LEN} ..."
    python "${COLLECT_SCRIPT}" \
        --model zamba2 \
        --seq_len "${SEQ_LEN}" \
        --device cuda \
        --out_dir "${PROFILE_DATA_DIR}"
done

deactivate

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Generate Figure 8 ==="
source "${FALCON_VENV}/bin/activate"
cd "${REPO_ROOT}"

python "${PLOT_SCRIPT}" \
    --profile_data_dir "${PROFILE_DATA_DIR}" \
    --out_dir "${OUT_DIR}"

deactivate

echo ""
echo "Done.  PNGs written to ${OUT_DIR}/"

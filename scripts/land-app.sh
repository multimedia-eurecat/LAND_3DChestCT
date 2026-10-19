#!/bin/sh
set -eu

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
[ ! -r /usr/local/lib/land-runtime-env.sh ] || . /usr/local/lib/land-runtime-env.sh
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${PYTHONPATH:-/opt/app/src}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export USE_TF="${USE_TF:-0}"
export LAND_HOST="${LAND_HOST:-0.0.0.0}"
export LAND_PORT="${LAND_PORT:-7860}"
export LAND_MODEL_PATH="${LAND_MODEL_PATH:-/opt/app/data/ckpts/2025-09-10_17-51-07_256_bsz1_lr1e-5_nodule+lung_mask}"
export LAND_MASK_DATASET="${LAND_MASK_DATASET:-/opt/app/data/masks}"
export LAND_OUTPUTS_DIR="${LAND_OUTPUTS_DIR:-/opt/app/web_outputs}"

mkdir -p "$MPLCONFIGDIR" "$LAND_OUTPUTS_DIR"
cd /opt/app
python_bin="${LAND_PYTHON:-/opt/venv/bin/python3}"
if [ ! -x "$python_bin" ]; then
	python_bin=/usr/bin/python3
fi
exec "$python_bin" /opt/app/src/inference_ldm_app.py

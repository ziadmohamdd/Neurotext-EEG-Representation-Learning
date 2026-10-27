#!/usr/bin/env bash
set -euo pipefail
cd /workspace/New_trends_AI_project1
export UV_PYTHON_INSTALL_DIR=/workspace/eeg-env/python
export UV_CACHE_DIR=/workspace/eeg-env/uv-cache
export PIP_CACHE_DIR=/workspace/eeg-env/pip-cache
export HF_HOME=/workspace/eeg-env/huggingface
export MPLCONFIGDIR=/workspace/eeg-env/matplotlib
uv python install --no-bin 3.11.17
if [ ! -x /workspace/eeg-env/py311/bin/python ]; then
  /workspace/eeg-env/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11 -m venv /workspace/eeg-env/py311
fi
/workspace/eeg-env/py311/bin/python -m ensurepip --upgrade
/workspace/eeg-env/py311/bin/python -m pip install --disable-pip-version-check -r requirements-lock-linux-py311.txt --extra-index-url https://download.pytorch.org/whl/cpu
/workspace/eeg-env/py311/bin/python -m pip install --disable-pip-version-check --no-deps -e .
/workspace/eeg-env/py311/bin/python -m pip check
/workspace/eeg-env/py311/bin/python -c 'from eeg_decoding.models.pretrained_eeg import acquire_official; acquire_official("/workspace/eeg-env/labram")'
EEG_LABRAM_CACHE=/workspace/eeg-env/labram /workspace/eeg-env/py311/bin/python -m pytest -q
/workspace/eeg-env/py311/bin/python -m eeg_decoding.cli device

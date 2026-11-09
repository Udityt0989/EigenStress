#!/bin/sh
# macOS / Linux: ./run_app.sh  - installs everything inside this folder (.venv, pip-cache)
cd "$(dirname "$0")"
export PIP_CACHE_DIR="$PWD/pip-cache"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt || exit 1
fi
.venv/bin/python -m streamlit run app.py

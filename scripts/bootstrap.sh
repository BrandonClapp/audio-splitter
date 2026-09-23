#!/usr/bin/env bash
# Create .venv, install audio-splitter, and prefetch the htdemucs_6s weights so the
# first real run doesn't stall on a few-hundred-MB download.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "==> creating .venv with $("$PYTHON" --version)"
  "$PYTHON" -m venv .venv
fi

echo "==> installing audio-splitter (editable) + dev extras"
./.venv/bin/python -m pip install --upgrade pip >/dev/null
./.venv/bin/python -m pip install -e '.[dev]'

echo "==> prefetching htdemucs_6s weights"
./.venv/bin/python - <<'PY'
from demucs.api import Separator

sep = Separator(model="htdemucs_6s", device="cpu")
model = getattr(sep, "model", None) or sep._model
print("sources:", list(model.sources))
PY

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then try:             split-solo --help"

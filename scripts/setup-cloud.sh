#!/usr/bin/env bash
# Recreate the validated Linux/Python 3.12 geometry stack in this checkout.
set -euo pipefail
stlstep_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$stlstep_repo_root"
bootstrap_python="${STLSTEP_BOOTSTRAP_PYTHON:-python3.12}"
"$bootstrap_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "The validated cloud lock requires Python 3.12"'
if [ ! -x .venv/bin/python ]; then
  "$bootstrap_python" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing .venv must use Python 3.12; select a separate checkout or recreate it explicitly"'
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.lock.txt
.venv/bin/python -m pip check
# shellcheck source=scripts/activate.sh
source scripts/activate.sh
python -c 'import build123d, trimesh; from OCP.STEPControl import STEPControl_Writer; print("CAD kernel imports OK")'
python -m unittest discover -s tests -v
printf '\nReady. Activate with: source scripts/activate.sh\n'

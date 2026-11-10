#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$PWD/out/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
mkdir -p "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
python_bin="${STLSTEP_PYTHON:-$PWD/.venv/bin/python}"
"$python_bin" -m pip check
"$python_bin" -m unittest discover -s tests -v
if [ "$#" -eq 0 ]; then
  "$python_bin" tools/reconstruction_pipeline.py --fixture channel --fixture shaft --fixture adapter --fixture vslot --fixture battery_cover --fixture bearing --fixture crossblock --fixture filletblock --fixture manifold --fixture diff_case --fixture threadring
  "$python_bin" tools/reconstruction_pipeline.py --stl tests/fixtures/battery_cover.stl --script tests/reconstructions/battery_cover.py
  "$python_bin" tools/reconstruction_pipeline.py --stl tests/fixtures/threadring.stl --script tests/reconstructions/threadring_nominal.py --thread-band z,0,0,0,12,4.138,5.05
else
  "$python_bin" tools/reconstruction_pipeline.py "$@"
fi

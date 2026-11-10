#!/usr/bin/env bash
# Source this file from any directory; paths follow this checkout.
stlstep_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$stlstep_repo_root/.venv/bin/activate"
export PYTHONPATH="$stlstep_repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$stlstep_repo_root/out/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
mkdir -p "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
cd "$stlstep_repo_root"
unset stlstep_repo_root

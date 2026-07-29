#!/usr/bin/env bash
# stl2step setup — finds a suitable Python (3.10–3.14), creates .venv, installs
# dependencies, and verifies the CAD kernel imports. Idempotent: safe to re-run.
set -euo pipefail

say()  { printf '%s\n' "$*"; }
fail() { printf '\n[setup] ERROR: %s\n' "$*" >&2; exit 1; }

# --- 1. find a Python 3.10–3.14 interpreter -------------------------------
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
    major=${ver%%.*}; minor=${ver##*.}
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 14 ]; then
      PY="$cand"; break
    fi
    LAST_SEEN="$cand ($ver)"
  fi
done

if [ -z "$PY" ]; then
  say "[setup] No suitable Python found. This project needs Python 3.10–3.14"
  say "        (the OCCT CAD kernel ships prebuilt wheels for those versions)."
  [ -n "${LAST_SEEN:-}" ] && say "        Found only: ${LAST_SEEN}"
  say ""
  say "  Install one, then re-run ./setup.sh :"
  case "$(uname -s)" in
    Darwin) say "    macOS:   brew install python@3.12      (https://brew.sh)" ;;
    Linux)  say "    Debian/Ubuntu:  sudo apt install python3.12 python3.12-venv"
            say "    Fedora:         sudo dnf install python3.12" ;;
    *)      say "    Windows: install Python 3.12 from https://python.org (or 'winget install Python.Python.3.12')" ;;
  esac
  say "    any OS:  pyenv install 3.12  (https://github.com/pyenv/pyenv)"
  exit 1
fi
say "[setup] Using $PY ($($PY -c 'import sys; print(sys.version.split()[0])'))"

[ -n "${SETUP_CHECK_ONLY:-}" ] && { say "[setup] check-only: interpreter OK"; exit 0; }

# --- 2. venv (reuse if compatible) ----------------------------------------
if [ -x .venv/bin/python ]; then
  vver=$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  case "$vver" in
    3.1[0-4]) say "[setup] Reusing existing .venv (python $vver)" ;;
    *) say "[setup] Existing .venv uses python $vver — recreating"; rm -rf .venv ;;
  esac
fi
[ -x .venv/bin/python ] || { say "[setup] Creating .venv ..."; "$PY" -m venv .venv; }

# --- 3. dependencies -------------------------------------------------------
say "[setup] Installing dependencies (the CAD kernel is ~150 MB; first run takes a few minutes) ..."
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

# --- 4. smoke test ---------------------------------------------------------
say "[setup] Verifying the geometry stack ..."
./.venv/bin/python - <<'EOF'
import build123d, trimesh, shapely, numpy, scipy, matplotlib
from OCP.STEPControl import STEPControl_Writer  # the OCCT kernel
print(f"[setup]   build123d {build123d.__version__}, trimesh {trimesh.__version__}, OCCT kernel OK")
EOF

say ""
say "[setup] Done. Next:"
if command -v claude >/dev/null 2>&1; then
  say "  run 'claude' in this directory, then ask:  draw this stl: ~/Downloads/part.stl"
else
  say "  install Claude Code (https://claude.com/claude-code), then run 'claude'"
  say "  in this directory and ask:  draw this stl: ~/Downloads/part.stl"
fi

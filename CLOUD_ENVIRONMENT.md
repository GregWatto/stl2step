# Reusable Codex cloud environment

The code, dependency lock and setup commands in this checkout are self-contained.
They do not require `/workspace/cloud-setup`, Claude Code, or model API keys.
Creating these files does not itself save or publish a Codex environment.

## Codex environment settings

Create an environment named **STL2STEP CAD** in Codex's environment settings:

- Repository: your fork of `nickpagz/stl2step` containing this work.
- Branch: `codex/cloud-reconstruction` (until merged into your default branch).
- Runtime: Linux with Python 3.12 and working `venv`/`pip`; this session used
  Python 3.12.14. The lock describes this platform, not every OS/Python version.
- Setup command, from the checked-out repository root:

  ```bash
  bash scripts/setup-cloud.sh
  ```

- Runtime environment variable: `MPLBACKEND=Agg`. Source imports and writable
  caches are also configured by `source scripts/activate.sh` and `scripts/test.sh`.
- No application secrets are required. Repository checkout may need your GitHub
  connection. Dependency installation needs `pypi.org` and `files.pythonhosted.org`;
  cloning/fetching needs `github.com`. The tests themselves need no network access.
- Use at least 8 GiB RAM and run CAD pipelines sequentially.

Use the settings flow's **Save draft** and **Publish** actions, when offered.
Publishing the environment is separate from pushing Git commits. This chat's
available tools cannot perform those settings actions; publication is confirmed
only when environment status reports a saved configuration/version.

After publication, select this environment for a new Codex task and verify:

```bash
source scripts/activate.sh
python -m pip check
python -m unittest discover -s tests -v
python tools/reconstruction_pipeline.py --all-fixtures --out out/fresh-cloud-validation
```

The validated source accepts 14/14 native fixtures and passes 73 unit tests.
Threadring uses an identified nominal M10x1.5 bore with its measured relief band,
not a claim of exact helical geometry. Every geometric gate remains enforced.
See `tests/THREADS.md` and `TESTING.md` for scope, evidence and pipeline details.

## Work in an existing checkout

```bash
bash scripts/setup-cloud.sh
source scripts/activate.sh
python tools/reconstruction_pipeline.py --stl /path/to/part.stl --out out/my-part
```

The setup command reuses a compatible `.venv` and installs the pinned lock. It
does not delete an incompatible environment or overwrite source STL fixtures.
`STLSTEP_BOOTSTRAP_PYTHON` can select another Python 3.12 executable.

## Offline Git backup

A Git bundle preserves the source and its history without installed packages or
large generated reports. To restore a supplied `stl2step.bundle`:

```bash
git clone -b codex/cloud-reconstruction stl2step.bundle stl2step
cd stl2step
bash scripts/setup-cloud.sh
```

Push this branch to your repository before configuring a new cloud checkout.
Generated STEP files and reports remain under ignored `out/`; the compact
`validation/native-fixtures.json` records the tested geometry source hashes and
native results in Git. A separate report archive can preserve the full evidence.

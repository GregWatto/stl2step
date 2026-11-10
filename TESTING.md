# Reproducible reconstruction checks

The upstream snapshot (ae32fab1f8d081b125d6ad8727c11bf5a0947fd4)
had no discoverable unit tests: `python -m unittest discover -s tests -v`
ran zero tests. `tests/make_*.py` are fixture generators with author-specific
absolute output paths, not test runners. Do not run them as tests or overwrite
committed fixtures. `eval/run_corpus.py` requires an external paired CAD corpus
which is not included; it measures recognizer coverage, not full gate acceptance.

## Environment

Validated on cloud Linux, Python 3.12.14. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
./scripts/test.sh
```

`requirements.lock.txt` pins the full installed dependency graph, including the
OCCT kernel. Upstream `requirements.txt` remains unchanged. The lock describes
this Python/Linux environment; it is not a portable wheel/hash lock. All tests
run without Claude Code, Anthropic SDK, API credentials, or a model API.
The test launcher configures source imports, headless rendering, and writable
caches. `STLSTEP_PYTHON` can select a different interpreter.

## Commands

```bash
# Unit regressions, channel/shaft/adapter/vslot/battery_cover/bearing/crossblock/filletblock/manifold/diff_case, plus saved battery-cover and threadring recipes
./scripts/test.sh

# Full native fixture inventory
./scripts/test.sh --all-fixtures --out out/full-run-01

# A particular deterministic recognizer proposal; repeat --fixture as needed
.venv/bin/python tools/reconstruction_pipeline.py --fixture channel --out out/channel-01

# Validate a Codex-authored build123d script (leave BuildPart in variable `part`)
.venv/bin/python tools/reconstruction_pipeline.py --stl /path/part.stl \
  --script /path/attempt.py --out out/attempt-01
```

Output directories must be new. Each stage has a configurable `--timeout`
(default 120 seconds). `--seed` defaults to 0; workers use separate processes,
fixed Python/NumPy seeds, a repeatable sequence of explicit trimesh surface
sampling seeds, and single-threaded BLAS. The sampling adjustment is confined to
test worker processes; no production reconstruction or gate code is changed.
Reproducibility means the same inputs, versions, sampling sequence, and semantic
verdict in this environment, not identical STEP headers, timestamps, or PNG bytes.

## Acceptance and artifacts

Each case runs a deterministic draw proposal (or copies the supplied script),
`compose brief`, `compose run`, and `compose compare`. It retains:

- Input/script SHA-256, source-module hashes, Git revision, pipeline hash,
  resolved packages, platform/Python versions, commands, exit codes, and logs.
- The exact candidate, `analysis.json`, exported `part.step`, editable
  `feature_tree.py`, and `metrics.json`.
- Both deviation maps, three-axis occupancy overlays, and `final_compare.png`.
- `gate-execution.json`, per-case `result.json`, and aggregate `summary.json`.

Exit 0 means all requested cases have complete passing geometric validation and
all required artifacts. Dependency failures, timeouts, absent candidates,
missing metrics, and failed gates return nonzero. A failed model can still export
a STEP upstream: that file is diagnostic and is not an accepted reconstruction.

The checked pipeline preserves upstream tolerances and decisions:

- Validity and closure; both sampled p99 surface deviations.
- `featurecheck.size_tol`: max(0.10 mm, 0.15% of bbox diagonal).
- Editable feature completeness, with upstream advisory chamfer/thread semantics.
- Slice occupancy on all three axes, missing and extra material each <=3%.
- Per-opening shape/boundary mismatches must be empty.

`compose run` now records validation errors in `metrics.json` and fails when
feature analysis is missing, feature or occupancy validation raises, or occupancy
and opening results are incomplete/non-finite. Opening-match and candidate-face
inspection errors propagate to that verdict. The worker additionally wraps
feature, occupancy, and opening-match calls to record execution, then rejects
missing, incomplete, or failed calls even if a future CLI regression prints PASS. These wrappers
do not change the functions' geometry calculations or acceptance thresholds.
They detect exceptions at those function boundaries; they do not prove that every
lower-level recovery inside the geometry stack succeeded.

The reverse field called `reverse_max_mm` is sampled p99 in this implementation,
not a global maximum. Volume is reported and occupancy is enforced; no additional
arbitrary scalar volume threshold is introduced. Thread-band exclusions are
accepted only as explicit evidence-backed workflow inputs, using repeatable
`--thread-band 'axis,cu,cv,t0,t1,r0,r1'`. Automatic identified bands from the
fresh brief retain their upstream behavior. Record the thread specification and
inspect the masks; the harness cannot establish the truth of a manual declaration.

Numeric success is distinct from visual acceptance: every automated result has
`visual_review: pending`. Codex must inspect both deviation maps, sections, and
the final comparison, name any visible structural differences, and refine before
claiming completion. A recognizer PASS alone never shortcuts these checks.

## Baseline and regressions

See `tests/BASELINE.md` for the unmodified recognizer results and the full compose
inventory. The default smoke suite deliberately exercises known-good cases;
`--all-fixtures` remains a strict assessment and does not mark known failures as
expected passes. Tests also verify actual geometric rejection of a filled open
channel and a square substituted for a round hole, plus failures for missing
checks, suppressed exceptions, NaN metrics, and missing editable features.

The initial Codex workflow adaptation left reconstruction unchanged. Subsequent
improvements add normal evidence to cylinder/fillet detection, preserve analytic
constant bores through lofted parts, and prevent rounding gaps in emitted slabs.
Geometric acceptance tolerances remain unchanged. See `tests/IMPROVEMENTS.md`.

The default suite also validates `tests/reconstructions/battery_cover.py`, a
measured, editable refinement of the automatic slab proposal. The automatic
`--all-fixtures` run intentionally continues to test native recognition; it does
not substitute this recipe or hide native battery-cover failures.

On the current 8 GiB cloud machine, run one pipeline process at a time. An
experimental overlap of the full inventory and smoke suite triggered a confirmed
cgroup OOM kill (signal 9) during crossblock conversion. The pipeline reports that
as a stage failure; it must not be interpreted as a geometric result. CI runs
these cases sequentially within a single job.


The continuation in `tests/CONTINUATION.md` adds automatic battery-cover and
bearing regression cases. Candidate selection now uses the worse of forward and
reverse sampled p99, matching the directionality required by the full pipeline.
The candidate occupancy gate sections the CAD solid directly, then triangulates
the planar section at 0.01 mm for its filled 2D region. Target STL sections and
all material/opening thresholds are unchanged. This avoids depending on open
seams in a display tessellation. Section errors still fail validation.

Exact distance queries now run in batches of 512 points. The points, distances,
and percentiles are unchanged; tests compare batched and unbatched queries.


`draw:machined` now proposes rectangular stock plus confidently detected simple
bosses and bores on their measured axes. It is tried after the simpler extrusion,
revolve, and prismatic paths fail tolerance, and before a general loft. Dominant
outward planar faces define the stock; feature positions are high endpoints,
so cylinders explicitly use MIN alignment. The current scope is axis-aligned
stock with simple measured cylinders. Ambiguous recovered bores, pockets,
counterbores, threads, and end treatments are left to the existing composition
workflow. No fixture names or stored recipes are used. See `tests/MACHINING.md`.


`draw:edgeblock` recognizes a rounded rectangular extrusion with four equal
longitudinal fillets, one uniform 45-degree end chamfer, and optional constant
circular through-bores. It compares measured outlines and bores at eleven global axial
stations plus four targeted bevel-band stations before proposing explicit stock/fillet/chamfer/cut operations. It does
not substitute a fixture recipe or relax the geometric gates. See
`tests/EDGEBLOCK.md` for its scope and regression results.


The machining recognizer now supports measured rectangular pockets and two-stage
counterbores with an explicitly measured low/high opening end. It can propose
from a winding-consistent source that is not watertight; acceptance still requires
closed valid CAD and every geometric gate. Non-watertight target sections use
oriented segment winding and reject unbalanced open intersections rather than
inventing closing edges. See `tests/MANIFOLD.md`. Manifold is included in the
default sequential smoke suite.


The default suite also runs the measured nominal `threadring_nominal.py` recipe
with its explicitly declared, probe-confirmed M10×1.5 thread band. This is a
separate case; native `--all-fixtures` never substitutes that recipe or its mask.
`refined_stack` proposals measure circular outer tapers and constant excess
profiles between planar supports, then run the unchanged full gate stack.
See `tests/REMAINING.md` for scope, negative tests and retained failures.


Native threadring is now in the default smoke suite. `threaded_ring` verifies a
complete plain cylindrical ring, the probe signature, whole-wall single-start
helical phase, triangular flank shape, and ISO metric dimensions before emitting
a nominal internal minor-diameter bore. Its measured relief band is recorded in
both the feature analysis and recipe; no manual `--thread-band` is required.
The feature gate independently requires the nominal bore's diameter, axis,
concavity, full circumference, and axial span. See `tests/THREADS.md`. The earlier
saved recipe remains a separate regression, and native inventory still performs
no recipe substitution.

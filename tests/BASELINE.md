# Baseline recorded before algorithm changes

Date: 2026-11-09. Upstream commit:
`ae32fab1f8d081b125d6ad8727c11bf5a0947fd4`.
Python 3.12.14; build123d 0.12.0; cadquery-ocp 7.9.3.1.1;
trimesh 5.2.0. No changes to `src/` or the original Claude skill.

## Existing checks and dependencies

- `pip check`: PASS; geometry/kernel imports: PASS.
- Original `unittest discover`: 0 tests discovered (coverage gap).
- Fixture generators contain `/Users/nickp/...` output paths and were inspected,
  not executed against committed fixture files.
- The external paired corpus for `eval/run_corpus.py` is unavailable here;
  corpus coverage evaluation was not run. The included `.ground_truth.step`
  files were not used as geometric oracles by the mesh-based gate suite.
- Claude Code and optional `anthropic` SDK are absent. They are not needed for
  Codex in-session composition or the deterministic checks; the API-driven
  `--method llm/full` paths were not exercised.

## Unmodified recognizer inventory

Ran all 14 committed STL fixtures through `convert --method draw --report ...`
in separate processes with a 90-second timeout. Nine returned PASS, four APPROX,
and one had no recognized model. These are recognizer outcomes, not full compose
acceptance. Initial recognizer metrics used upstream sampling defaults and can
vary slightly; the new pipeline fixes the test sampling sequence.

| Fixture | Recognizer outcome | Reported forward p99 (mm) |
| --- | --- | --- |
| adapter.stl | PASS | 0.0893 |
| battery_cover.stl | PASS | 0.1983 |
| bearing.stl | PASS | 0.0798 |
| bracket.stl | PASS | 0.0417 |
| channel.stl | PASS | 0.0 |
| crossblock.stl | APPROX | 3.8559 |
| diff_case.stl | APPROX | 0.3832 |
| filletblock.stl | NO MODEL | — |
| manifold.stl | APPROX | 6.3622 |
| shaft.stl | PASS | 0.0457 |
| sleeve.stl | PASS | 0.0366 |
| threadring.stl | APPROX | 0.6444 |
| vslot.stl | PASS | 0.0119 |
| ytower.stl | PASS | 0.0 |

Raw initial logs and summary: `out/baseline/`. The checked reproduction command
is `./scripts/test.sh --all-fixtures --out out/full-run-01`.

## Workflow and validation changes

The original skill described a 0.1%-diagonal tolerance and allowed an early exit
on recognizer PASS. The implementation uses max(0.10 mm, 0.15%-diagonal), and
recognizer PASS does not run full feature/occupancy checks. The Codex adaptation
corrects the documentation and requires the full stack.

Some upstream gate exceptions are caught and ignored. The test worker observes
feature, occupancy, and opening-match calls and rejects incomplete execution.
No geometry calculations or thresholds were changed. Feature detector findings
may include false positives; this baseline records failures without waiving them.

## New regression validation

Eleven tests pass: two real geometry rejection cases and nine validation-boundary
cases. The channel/shaft end-to-end smoke suite passes. A deliberately solid-filled
channel script returns nonzero, with 229.9% extra occupied volume. Its exported
STEP is retained only as a failed diagnostic (`out/pipeline-negative/`).

GitHub Actions is configured to run the pinned environment and smoke suite and
upload artifacts. The workflow has been validated locally, not run on GitHub.

## Full checked fixture inventory
All 14 fixtures completed with no stage timeouts. Five pass the full numeric
gate stack; nine fail or lack an editable candidate. The inventory returns exit 1.
No failed case is treated as an expected pass. Visual review remains pending
in the automated results.
| Fixture | Full geometric gates | Findings |
| --- | --- | --- |
| adapter.stl | FAIL | feature completeness |
| battery_cover.stl | FAIL | surface deviation |
| bearing.stl | FAIL | feature completeness, opening shapes |
| bracket.stl | PASS | all numerical gates passed |
| channel.stl | PASS | all numerical gates passed |
| crossblock.stl | FAIL | surface deviation, feature completeness, occupancy, opening shapes |
| diff_case.stl | FAIL | surface deviation, feature completeness, opening shapes |
| filletblock.stl | FAIL | no recognized editable model |
| manifold.stl | FAIL | surface deviation, feature completeness, occupancy, opening shapes |
| shaft.stl | PASS | all numerical gates passed |
| sleeve.stl | PASS | all numerical gates passed |
| threadring.stl | FAIL | surface deviation, feature completeness, occupancy, opening shapes |
| vslot.stl | FAIL | feature completeness |
| ytower.stl | PASS | all numerical gates passed |

Raw artifact bundle: `out/pipeline-all/summary.json` and its per-case directories.
Full-gate failures after recognizer PASS include adapter, battery_cover, bearing,
and vslot. Feature detection can be approximate; these findings are recorded as
unresolved, rather than asserted to be proven missing geometry in every case.
Threadring was tested with automatic analysis only, without manually declared
thread bands; no exclusions were fabricated to make it pass.

The repeated seeded channel and shaft runs produced identical `metrics.json`
content (`out/pipeline-smoke/` versus `out/pipeline-confirmed/`). Their four-view
comparison overlays were inspected: no colored silhouette discrepancy was visible;
channel shows some shaded triangulation artifacts. The automated visual-review
field is deliberately unchanged: this does not constitute visual sign-off of all
14 inventory cases.

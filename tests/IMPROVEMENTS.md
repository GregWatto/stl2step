# Reconstruction improvements

The pre-change dependency check and 11 regression tests passed. The three
investigated cases still failed the full gates: adapter and vslot on feature
completeness, battery_cover on reverse deviation. See BASELINE.md for the original
14-fixture inventory. No committed STL was changed.

## Feature evidence and analytic bores

Vslot's four apparent Ø7.1 bosses were positional cylinder fits through flat
profile facets. Vertex positions lay close to a circle, but the triangle normals
did not point radially (5th-percentile agreement ~0.777). The actual bore's
agreement was ~0.9997. Cylinder descriptions now include area-weighted normal
support: at least 95% of face area must agree with radial normals within 10°.
Unsupported fits stay in `low_confidence_fits` for inspection, instead of creating
hard hole/boss requirements. Raw surface descriptions are retained.

Adapter's lofted exterior also yielded local cylinder fits wrongly interpreted
as fillets. Supported fillet bands now require at least 30° of angular extent;
short or poorly supported fits remain explicit `low_confidence_fillets` hints.
The real rounded-block fixture still detects its four fillets. These are feature
confidence criteria; surface, occupancy, and opening acceptance limits are
unchanged. Shallow/ambiguous rounds still require visual inspection.

Adapter's real Ø12 straight bore was reconstructed by lofting two effectively
identical circles. OCCT represented that wall as a spline, causing the analytic
feature check to reject it. Constant coaxial circular sections now produce a
cylindrical subtractive extrusion in both the runtime builder and emitted recipe.
A 0.0001 mm absolute equality check only absorbs fitting noise; tapered,
noncircular, or offset bores retain the loft path. Tests check genuine cylinders,
rejection of flat facets, real fillets, and tapered/offset bore behavior.

## Battery cover and slab continuity

The original emitted prismatic script independently rounded slab offsets and
lengths to six significant digits. For example, 6.96659 + 13.5 ends at 20.46659,
while the next sketch starts at 20.4666. Those tiny gaps create extra solids and
internal faces, corrupting reverse deviation even when volume is almost right.
The emitter now uses shared full-precision endpoints and computes lengths from
the same endpoint literals. A regression checks connectedness and volume parity
between the built model and emitted script along all three axes.

That general fix reduces native battery_cover reverse p99 from ~0.770 to ~0.338
mm, but the automatic model still fails the unchanged ~0.210 mm tolerance.

`tests/reconstructions/battery_cover.py` is a separate editable refinement of
that automatic proposal. It restores the full front slab and clips the ends to
measured sections: the XY footprint has R5 corners (circle-fit residual <0.002
mm), and the XZ clip section has R1 root/bottom rounds. It passes the full gates
with forward/reverse p99 ~0.0167/0.0180 mm. Occupancy and opening checks also pass.

The four-view comparison and both deviation maps were inspected: no large
structural silhouette differences remain. Local residuals near the clip and
advisory chamfer findings remain visible; p99 success is not exact surface
identity. The recipe is included in the default smoke suite. The full native
fixture inventory does not substitute it for automatic recognition.

## Explicit validation failures

`compose run` now returns failure and records `validation_errors` for missing or
invalid feature analysis, feature/occupancy exceptions, incomplete three-axis
occupancy, missing opening results, or non-finite occupancy metrics. Candidate
face inspection and per-opening matching exceptions propagate into this report.
Feature/edge-analysis failures also propagate from brief generation instead of
being replaced by an empty feature list. The independent pipeline observers
remain as an additional execution check.

The existing 0.10 mm / 0.15%-diagonal tolerance, bidirectional sampled p99 tests,
validity/closure requirements, 3% missing/extra occupancy limits, opening-shape
checks, and evidence-based thread semantics remain unchanged.

## Reproduce

```bash
./scripts/test.sh
./scripts/test.sh --all-fixtures --out out/full-improved-run
```

The default suite runs 20 unit regressions, native channel/shaft/adapter/vslot,
and the refined battery-cover recipe. GitHub Actions uses this suite and retains
artifacts; it has not been run remotely in this session.

Artifacts from this session:

- `out/improvements/features/`: first adapter/vslot passes.
- `out/improvements/battery-native/`: native battery failure after continuity fix.
- `out/improvements/battery-joined/`: passing measured battery recipe.
- `out/improvements/final-inventory/`: native fixture inventory after all source fixes.

Intermediate battery attempts and an interrupted inventory are retained under
`out/improvements/` for diagnosis; they are not passing final results.

## Final validation results

The default command completed successfully: 20 tests and all five end-to-end
cases passed. The native fixture inventory improves from 5/14 to 7/14 numerical
passes. Adapter and vslot are the two newly passing native cases; every previous
numeric pass is retained. Seven native cases remain unresolved, including the
automatic battery cover (the separate refined recipe passes).

| Native fixture | Result | Findings |
| --- | --- | --- |
| adapter.stl | PASS | all numeric gates passed |
| battery_cover.stl | FAIL | gates: 1; upstream validity/closure/deviation/feature verdict failed |
| bearing.stl | FAIL | gates: 1; upstream validity/closure/deviation/feature verdict failed; missing editable features; opening shape check failed or missing |
| bracket.stl | PASS | all numeric gates passed |
| channel.stl | PASS | all numeric gates passed |
| crossblock.stl | FAIL | convert: -9; gates: 1; upstream validity/closure/deviation/feature verdict failed; missing editable features; occupancy exceeds upstream 3% limit; opening shape check failed or missing |
| diff_case.stl | FAIL | gates: 1; upstream validity/closure/deviation/feature verdict failed; missing editable features; opening shape check failed or missing |
| filletblock.stl | FAIL | no editable candidate; recognizer miss/error (see convert.log) |
| manifold.stl | FAIL | gates: 1; upstream validity/closure/deviation/feature verdict failed; missing editable features; occupancy exceeds upstream 3% limit; opening shape check failed or missing |
| shaft.stl | PASS | all numeric gates passed |
| sleeve.stl | PASS | all numeric gates passed |
| threadring.stl | FAIL | gates: 1; upstream validity/closure/deviation/feature verdict failed; occupancy exceeds upstream 3% limit; opening shape check failed or missing |
| vslot.stl | PASS | all numeric gates passed |
| ytower.stl | PASS | all numeric gates passed |

The signal-9 conversion entry in the full inventory came from running it at the
same time as the smoke suite. The cgroup reported `oom_kill=1` with an 8 GiB
memory limit. A separate crossblock-only rerun is retained to distinguish this
resource failure from its actual geometric failures. Do not run multiple CAD
pipelines concurrently on this cloud machine.

The expanded smoke artifacts are `out/pipeline-20261109-183531/` (four native
parts) and `out/pipeline-20261109-183719/` (refined battery cover).

The isolated crossblock rerun completed conversion, brief, gates, and comparison
without a resource failure (`convert=1` for APPROX, `brief=0`, `gates=1`,
`compare=0`). Its
remaining failures are geometric: deviation, feature completeness, occupancy,
and opening shapes. Evidence: `out/improvements/crossblock-serial/`.

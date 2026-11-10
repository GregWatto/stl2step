# Automatic reconstruction and cloud-runtime follow-up

Baseline before these changes: dependencies and all 20 tests passed; seven native
fixtures passed the full gates, while battery_cover required a saved refinement
recipe. No committed input mesh or acceptance threshold was changed.

## Root causes addressed

- **Tessellation-dependent profile fitting.** A fixed 4-degree turn criterion
  collapsed finely tessellated curves into straight chords. Recursive geometric
  fitting now limits line/arc residuals to the existing 0.02 mm profile resolution,
  checks arc direction consistently, and retains a shape-preserving polyline
  fallback. Tests cover coarse/fine quarter-circles and the no-arcs fallback.
- **Lost analytic bearing bores.** Tiny radial/axial drift turned straight walls
  into cones. Near-axial profile lines are aligned within the profile resolution
  only when their slope is <=0.001; shared endpoints move together. Tests verify
  that real tapers remain tapered.
- **Unsupported material in prismatic proposals.** When a slab model misses
  bidirectional tolerance, the generator can intersect it with three measured
  orthogonal projection envelopes. Each envelope unions projected mesh triangles,
  preserves concavities/holes, and emits editable line/arc sketches and extrusions.
  It keeps the result only if bidirectional surface error improves. This supplies
  battery-cover end constraints automatically, without a fixture lookup, saved
  recipe, or hard-coded battery dimensions. Projection cost is bounded to meshes
  of <=100,000 faces and <=32 projected components; skipping this proposal never
  skips geometric validation.
- **False occupancy loss from display meshes.** The automatic cover is a valid
  closed CAD solid, but its tessellation has small seam gaps. Mesh slicing lost
  whole candidate sections and reported ~15% missing volume despite <0.2% actual
  volume difference. Candidate occupancy now sections the CAD solid first and
  triangulates the filled planar section at 0.01 mm. STL sections, three axes,
  occupancy limits, opening-shape rules, and failure handling remain unchanged.
  Tests verify that CAD sections preserve holes and still reject a square
  substituted for a circular opening.
- **Counterbore counting.** The feature gate used to demand twice the through-hole
  cylinder count at the *same* radius. It now independently verifies the through
  and recess diameters. Both a valid four-hole pattern and a missing-counterbore
  pattern are tested.
- **False circular pockets.** Circular-feature detection now checks edge
  midpoints too, preventing rectangle corners on a fitted circle from being
  mistaken for circular walls. True circular boundaries remain detectable.
- **Peak-memory pressure.** All production nearest-surface queries use batches
  of 512 query points. They retain every sample and use the same exact trimesh
  query; tests verify identical positions, distances, and triangle indices against
  unbatched results. This bounds intermediate candidate-triangle allocations
  instead of reducing sample counts to avoid the earlier OOM failure.

## Verification

Thirty unit regressions pass. The automatic cover and bearing both pass the full
pipeline, with separate logs and images in `out/continuation/cad-occupancy/`.
A final native inventory is retained in `out/continuation/final-inventory/`.
Exploratory runs in this directory are distinct from final passing results.

The default `./scripts/test.sh` includes native battery_cover and bearing in
addition to the previous channel, shaft, adapter, and vslot cases, followed by
the saved battery-cover recipe. The full native inventory does not substitute
hand-authored recipes for automatic reconstruction.

## Final native inventory

Nine of fourteen native fixtures pass, up from seven before this continuation.
Both newly passing cases use automatic recognition: battery_cover and bearing.
All seven previous passes are retained. The five unresolved cases remain failed,
not marked as expected passes. The full run completed with no memory kills or
stage timeouts; its nonzero exit reflects the remaining geometric failures.

| Fixture | Result | Findings |
| --- | --- | --- |
| adapter.stl | PASS | all full gates passed |
| battery_cover.stl | PASS | all full gates passed |
| bearing.stl | PASS | all full gates passed |
| bracket.stl | PASS | all full gates passed |
| channel.stl | PASS | all full gates passed |
| crossblock.stl | FAIL | gates: 1; upstream validity/closure/deviation/feature verdict failed; missing editable features; occupancy exceeds upstream 3% limit |
| diff_case.stl | FAIL | gates: 1; upstream validity/closure/deviation/feature verdict failed; opening shape check failed or missing |
| filletblock.stl | FAIL | gates: 1; upstream validity/closure/deviation/feature verdict failed; missing editable features |
| manifold.stl | FAIL | no editable candidate; recognizer miss/error (see convert.log) |
| shaft.stl | PASS | all full gates passed |
| sleeve.stl | PASS | all full gates passed |
| threadring.stl | FAIL | gates: 1; upstream validity/closure/deviation/feature verdict failed; occupancy exceeds upstream 3% limit; opening shape check failed or missing |
| vslot.stl | PASS | all full gates passed |
| ytower.stl | PASS | all full gates passed |

Automatic cover forward/reverse p99 is ~0.0139/0.0151 mm (limit ~0.210 mm),
with ~0.2% missing and ~0.0% extra occupied volume. Bearing is ~0.0665/0.0680
mm (limit 0.100 mm), with ~0.2% missing and ~1.0% extra volume. Both have no
opening mismatches or hard feature gaps. The cover retains advisory chamfer
findings and local residuals near its clip; neither result claims exact surface
identity. The comparison renders were inspected: the structural overlays are
gray; bearing's display triangulation is coarser than its source mesh.

Remaining work is feature composition for cross-axis bores/pockets and edge
treatments (crossblock, manifold, filletblock), the more intricate diff_case
profile, and an evidence-backed nominal-thread workflow for threadring. A probe
confirmed an M10-class internal thread (~1.452 mm measured pitch, nearest
standard M10x1.5); no thread exclusions were applied to the native inventory.

The final default command completed with exit 0: all 30 unit tests
and all seven end-to-end cases passed (six native cases plus the saved recipe).
Elapsed time was 196.6 seconds; maximum child-process RSS
reported by Python `resource.getrusage` was 702.3 MiB.
This is a process high-water mark, not total concurrent cgroup memory. The cgroup
OOM-kill counter did not increase during these final checks. Logs and resource
metadata are in `out/continuation/default-suite*`. The optional GNU time binary
was unavailable, so this measurement uses Python's standard library.

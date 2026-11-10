# Rectangular pockets and counterbores

Before algorithm changes, dependency checks and all 41 existing unit tests passed.
The previous native inventory passed 11/14 fixtures; manifold was unresolved.
No input fixture or acceptance threshold was changed.

## Diagnosis and implementation

Manifold's measured body is 60 × 40 × 28 mm, with a 44 × 24 mm rectangular
pocket opening upward from its floor at z=18 mm. It has a Ø14 cross bore at
z=9 mm and four Ø6 vertical bores at (±24, ±15), with Ø10 counterbores over
the upper 4 mm. The counterbores touch the exterior and intersect the pocket.

The repaired source is winding-consistent but not watertight. An explicit
measured CAD proposal initially passed surface/feature checks but failed the
occupancy gate: 12.7% missing material. Some target sections incorrectly filled
the pocket where tangent openings created branching contour nodes. A diagnostic
oriented intersection recovered areas matching the actual CAD sections.

`sections.py` now reconstructs sections of non-watertight, consistently wound
meshes from directed plane/triangle intersections. It polygonizes the linework
and selects cells by signed winding. Every intersection vertex must have balanced
incoming/outgoing segments; missing boundaries raise an error. Coordinates are
noded at 1e-9 of the bounding diagonal (minimum 1e-9 mm), far below geometric
acceptance tolerances. It creates no closing edges or replacement source faces.
The existing closed-mesh section path remains unchanged.

The machining recognizer now:

- Selects outermost sufficiently supported stock planes, so a large recessed
  pocket floor cannot replace the stock top while small boss caps remain excluded.
- Accepts rectangular pockets only after checking the union of the measured
  floor triangles against the rectangle. Profiled/unknown pockets are rejected.
  The emitted cut starts at the measured floor and follows its opening normal.
- Records counterbore opening end, stage count, axial continuity, and depth.
  Only two contiguous stages with an unambiguous low/high wider end are supported.
  Pattern grouping preserves differences in depths and opening ends.
- Deduplicates recovered bore hints against both measured cylinder diameters
  using perpendicular distance to their common axis line.
- Allows proposals from a winding-consistent source, then requires the complete
  unchanged validity, closure, surface, feature, occupancy, and opening gates.

There are no fixture-name branches or saved-recipe substitutions. Oblique stock,
profiled pockets, threads, countersinks, disconnected/multiple/ambiguous stages,
and unresolved recovered hole hints remain outside this recognizer's scope.

## Validation

All 50 unit tests pass; dependency checks pass. New coverage checks all three
section axes, tangent counterbores, actual open-gap rejection, inconsistent
winding, both counterbore ends on all axes, negative-direction pocket cuts,
ambiguous/disconnected stages, distinct pattern signatures, and manifold under
translation, scaling and axis permutation. Existing rejection tests remain active.

Automatic manifold artifacts: `out/manifold-native/00-manifold/`:

- Valid, closed single solid; 31 analytic faces.
- Forward/reverse sampled p99: approximately 0.02274/0.02274 mm, against 0.116 mm.
- No missing hard features, opening-shape mismatches, or validation exceptions.
- Mean section IoU ~0.9999; missing/extra material fractions ~0.012%/~0.003%.
- CAD volume ~43462.58 mm³. Source volume is not asserted for this open mesh.

Both deviation maps, source cross-sections and four-view final comparison were
inspected. The pocket, intersecting counterbores, and cross bore agree visibly;
remaining curved-surface bands are on the ~0.02 mm tessellation scale. This is a
sampled geometric validation, not proof of exact surface identity. The pipeline
keeps its automatic visual-review field pending; this document records the
separate session review.

The default CI smoke suite now includes native manifold. Reproduce with:

```bash
./scripts/test.sh --fixture manifold --out out/manifold-repeat
# Strict complete native inventory (known unresolved parts remain failures):
./scripts/test.sh --all-fixtures --out out/manifold-inventory-repeat
```

## Final inventory

The final-source run in `out/manifold-final-inventory/summary.json` passes 12/14
native fixtures, up from 11/14, retaining all previous passes. Only `diff_case`
(deviation/opening checks) and `threadring` (deviation/occupancy/opening checks)
remain unresolved. The strict inventory returns exit status 1 for these failures;
they are not expected-pass exemptions. Dependencies pass and there were no stage
timeouts or memory kills.

The final manifold recipe in `out/manifold-final-inventory/08-manifold/` is
byte-identical to the visually reviewed native recipe. The directory contains
`part.step`, `candidate.py`, metrics, gate-execution evidence, deviation maps,
all-axis section overlays, and a separate `visual-review.json` record. Source
hashes in the inventory match the final geometry implementation.

The saved battery-cover recipe also passes the complete pipeline with the final
implementation (`out/manifold-battery-recipe/`). Shell syntax and whitespace
checks pass. Remote CI was not run during this session.

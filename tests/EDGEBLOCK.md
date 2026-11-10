# Measured fillets and end chamfers

Before changes, dependencies and all 35 tests passed. Filletblock's native loft
proposal failed at ~0.92/0.94 mm forward/reverse p99 versus ~0.101 mm tolerance,
and did not contain editable cylindrical bore/fillet faces. No input STL was
modified.

## Reconstruction

`stlstep.edgeblock` adds a conservative recognizer for a rounded rectangular
extrusion with four equal longitudinal fillets, one uniform 45-degree end
chamfer, and optional constant circular through-bores. It reconstructs stock,
then calls actual `fillet()` and `chamfer()` operations before cutting bores.
There are no fixture-name branches or saved-recipe substitutions.

Measurements come from cross-sections:

- A middle section must match the stock footprint and an analytic rounded
  rectangle. All four corner centres and fitted radii must agree.
- Near-cap sections identify which end is inset. All four sides must have the
  same inset, and their outlines must match the inward offset of the body.
- Interior and near-cap sections must match the same constant body and linear
  inset model. Hole centres/radii must remain constant through every section.
- Both ends treated, continuously drafted walls, tapered bores, unequal corner
  radii, and nonuniform end treatments are outside this narrow recognizer.

The current geometry-fit resolution is max(0.02 mm, 0.01% of the mesh diagonal).
It supports axis-aligned extrusion directions, arbitrary translations, scaling,
and either a high- or low-end chamfer. The chamfer must be smaller than the
corner radius and 45% of the extrusion height. Proposal generation is limited
to closed meshes with at most 100,000 triangles. Unsupported cases retain the
existing fallback paths.

The recognizer is tried after extrusion/revolve/prismatic and simple machining
proposals, before general lofting. It is still measured bidirectionally, then
subjected to the unchanged validity, closure, feature, occupancy, and opening
checks in the pipeline. The fit resolution is not an acceptance threshold.

## Filletblock result

The automatic model has a measured ~50x40x20 body, R6.0006 longitudinal fillets,
a 3.0000 mm upper chamfer, and an approximately Ø11.9988 through-bore. Those
values are measured fits rather than dimensions loaded from a generator script.
The resulting CAD has 19 analytic faces, including the cylindrical rounds and
conical chamfer corners.

- Forward/reverse sampled p99: ~0.0236/0.0236 mm; tolerance ~0.1006 mm.
- Valid, closed single solid; no missing hard features or opening mismatches.
- Occupancy mean IoU ~1.0000, missing/extra volume ~0.5/0.9 mm³.
- CAD volume ~36385.0 mm³; source mesh volume ~36384.6 mm³.

Both deviation maps, all-axis slice overlays, and four-view comparison were
inspected. The overlays have no visible structural discrepancies. Cylindrical
bands in deviation maps are at a ~0.02 mm scale; sampled p99 success does not
claim exact surface identity. The advisory chamfer finding was confirmed by the
emitted operation, analytic conical faces, measured sections, and visual evidence.

First complete artifacts: `out/filletblock-edge-first/00-filletblock/`.
Native regression inventory: `out/filletblock-edge-final/`.

## Regression coverage

Tests check the original fixture under translation and a scaled axis permutation
with the opposite chamfer end. Kernel geometry is checked on all three axes and
both ends, including analytic round/bore faces. Negative tests reject continuous
draft, tapered bores, and two treated ends. An additional thin non-45-degree
bevel test exposed a gap in fixed-station sampling; its pre-fix failure is retained
in `out/edgeblock-bevel-before.log`.

Filletblock has been added to the default smoke suite used by CI:

```bash
./scripts/test.sh
```

## Final validation

The full native inventory improves from 10/14 to 11/14 passes and retains every
previous pass. Only diff_case, manifold, and threadring remain unresolved. The
full run completed without stage timeouts or new memory kills.

After that inventory, four dedicated measurement stations were added inside and
just beyond the inferred bevel band. The thin unequal-leg bevel regression now
fails recognition as intended. All 41 unit tests pass, and a fresh full-pipeline
filletblock run passes in `out/filletblock-bevel-verified/`. Its emitted recipe is
byte-identical to the visually inspected first result. This targeted follow-up
records the final source hashes; the earlier full-inventory snapshot predates
only the additional rejection safeguard.

The global validity, closure, bidirectional p99, feature, 3% occupancy, and opening
acceptance rules were not loosened. The updated default smoke configuration adds
filletblock; CI has not been run remotely during this session.

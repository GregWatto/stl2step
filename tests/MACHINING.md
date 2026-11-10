# Measured block machining

Crossblock's previous proposal used section stacks/lofts and omitted the cross
bore or removed material across the whole body. The baseline failed surface,
feature, and occupancy checks. Dependencies and all 30 pre-change tests passed.

## Reconstruction

`stlstep.machining` adds a conservative proposal generator. It uses six dominant
outward planar support levels to identify rectangular stock. Coplanar fragments
are grouped and area-weighted, so a small boss end cap does not replace the
larger block shoulder. This currently supports axis-aligned stock and arbitrary
translations; axis permutations and uniform scaling are tested.

The existing segmenter, cylindrical normal evidence, and feature grouping supply
bosses and simple bores. The generated script builds stock, adds bosses, then
cuts each bore along its own measured axis. Feature positions are high axial
endpoints; each cylinder starts at `high - depth*axis` with explicit MIN
alignment. This avoids the earlier scaffold convention of treating an endpoint
as a midpoint and shifting a boss away from its supporting face.

A fractional-translation regression exposed the feature table's 0.01 mm position
rounding: a nominally touching boss could become a separate solid. Axial boss
starts are now reconciled to a measured stock support level only within that
recorded 0.01 mm resolution, while preserving the reported high endpoint. Larger
gaps are not closed by guessing; disconnected or invalid proposals are rejected.

There are no fixture-name branches or saved reconstruction lookups. The narrow
recognizer rejects nonplanar stock, open target meshes, ambiguous recovered
bores, pockets, counterbores, threads, and end treatments; these remain jobs for
feature composition. It bounds proposal generation to 100,000 triangles and 64
feature instances. Failing a proposal never waives validation.

`draw()` tries this path after simpler extrusion, revolve, and prismatic attempts
fail tolerance, and before lofting. It still measures both surface directions.
The checked pipeline still requires validity, closure, feature completeness,
three-axis occupancy, opening-shape checks, and all required diagnostic artifacts.
No acceptance thresholds were changed.

## Crossblock result

The automatically emitted feature tree contains a 50x40x20 measured block,
Ø18x10 top boss, four Ø5 mounting holes, and an Ø12 cross-axis bore. The CAD has
13 analytic faces rather than a loft approximation.

- Forward/reverse sampled p99: ~0.0230/0.0229 mm (tolerance ~0.1061 mm).
- Valid, closed solid; all detected features present; no opening mismatches.
- Three-axis occupancy mean IoU ~0.9999; missing/extra volumes ~2.2/1.6 mm³.
- Model volume 36450.0 mm³ versus source mesh 36451.5 mm³.

Both deviation maps, section overlays, and the four-view comparison were
inspected. Colored bands on cylindrical walls in deviation maps are at a
~0.02 mm scale and reflect tessellation; no large structural discrepancy is
visible in the comparison overlays. Sampled p99 is not an exact surface bound.

First complete artifacts: `out/crossblock-feature-first/00-crossblock/`.
Final native inventory: `out/crossblock-machining-final/`.

## Regression coverage

Thirty-five tests cover the previous gates plus detection after translation,
scaling, and axis permutation; high-endpoint cylinder placement along all three
axes; refusal to assume nonplanar stock is a box; refusal to guess unsupported feature stages; fractional-coordinate attachment;
and rejection of disconnected bosses. A test initially called build123d's Boolean `is_valid` property
as a method; it was corrected, and all 35 tests pass.

Crossblock is included in the default smoke suite used by CI. Run:

```bash
./scripts/test.sh
```

The full native inventory remains strict and records unresolved cases instead
of labeling them expected passes.

## Full inventory

The complete native run improved from 9/14 to 10/14 passes and retained all nine
previous passes. The unresolved cases remain diff_case, filletblock, manifold,
and threadring. No tolerance or validation gate was loosened.

That full inventory preceded the final fractional-coordinate attachment
correction. The correction is covered by the transformed-input and disconnected
boss regressions, and a fresh full-pipeline crossblock run is recorded in
`out/crossblock-precision-final/`. The fixture's dimensions are unaffected by
this correction; it changes how rounded translated feature tables attach bosses.

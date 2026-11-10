# Remaining-case reconstruction work

Update: automatic threadring reconstruction is now implemented and the native
inventory passes 14/14. See [THREADS.md](THREADS.md). The results below record
the preceding 13/14 stage and its explicit nominal recipe.

The starting point passed dependency checks and all 50 unit tests, with 12/14
native fixtures accepted. `diff_case` failed bidirectional deviation and opening
checks; `threadring` failed deviation, occupancy and opening checks. The original
STLs and all geometric acceptance thresholds are unchanged.

## Automatic diff_case refinement

The old slab proposal loses two side recesses and approximates a circular tapered
rim with a midpoint extrusion. A weak circle-fit rule also allowed polygons with
nearly circular vertices to lose their flat sides. The circle classifier now
checks vertices and edge midpoints against the fitted circumference.

`stlstep.refinement` adds a proposal stage after an unsuccessful prismatic model:

- Circular outer bands are measured at five stations per mesh ring interval.
  Centres must agree, radii must vary linearly, and interior profiles must stay
  constant. Actual ruled lofts replace the midpoint outer approximation. A
  varying bore or topology change rejects replacement of that band.
- Excess material between measured planar supports is inspected at nine stations,
  including near both ends. Only a stationary profile may become an extruded cut.
  The fitted tool is sectioned independently and checked against source material
  at every station before it is used. This operation cannot fill missing material.
- The proposal must form one valid connected solid and meet the same bidirectional
  sampled deviation bar. It then goes through every compose gate in the pipeline.

These are measurement checks, not acceptance exemptions. The local fit resolution
is max(0.02 mm, 1e-5 of the part diagonal); bands thinner than twice this resolution
are left unresolved. The initial part-size cutoff missed 0.1 mm-wide recess ends;
a regression now covers those thin, measured bands. Cost is bounded to closed
meshes with at most 10,000 triangles, at most 16 planar supports per axis, at most
34 taper boundaries per band, and at most eight stationary cuts. Other models
retain existing fallback paths. No fixture names or stored recipes are consulted.

The first native result is retained in `out/diff-refinement-native/00-diff_case/`.
It passed all gates at approximately 0.05 mm forward/reverse sampled p99, with
no opening mismatches. Both deviation maps, all-axis slice overlays and final
comparison were inspected: the tapered rim, octagonal cavity, small axial bores
and side recesses are present. Remaining local bands reflect profile fitting and
tessellation; this sampled result does not establish exact surface identity.
The final inventory includes the additional thin-band recovery.

## Explicit nominal threadring model

The original native proposal is an off-centre average circular bore, which is
neither the actual helix nor a concentric tap-drill representation.

The existing reconstruction skill calls for identifying a thread, then supplying
a nominal hole/stud with the confirmed relief band declared to validation. A
retained probe identifies internal M10×1.5: measured pitch 1.452 mm, minor/major
10th/90th percentile diameters 8.539/9.837 mm, zero measured taper. Vertex-envelope
inspection supports a relief radius approximately 4.188..5.0 mm over z=0..12 mm.
The saved recipe uses the standard M10×1.5 internal minor diameter, 8.376 mm,
inside the measured Ø16 × 12 mm stock. Reapply the thread operation in downstream
CAD; this recipe intentionally does not reproduce its helical faces.

Evidence: `tests/reconstructions/threadring_probe.json`.
Recipe: `tests/reconstructions/threadring_nominal.py`.

```bash
source /workspace/cloud-setup/activate.sh
python -m stlstep.probe --json thread tests/fixtures/threadring.stl \
  --axis z --center=0,0 --range 0,12 --r 4.6
python tools/reconstruction_pipeline.py --stl tests/fixtures/threadring.stl \
  --script tests/reconstructions/threadring_nominal.py \
  --thread-band z,0,0,0,12,4.138,5.05 --out out/threadring-repeat
```

The declared radial band includes the existing 0.05 mm tessellation chord allowance
on each side of the measured relief envelope. The first narrower band, starting
at 4.17 mm, left nominal-bore tessellation samples outside the mask and correctly
failed reverse deviation (~0.534 mm). That failed attempt remains in
`out/threadring-nominal-first/`. The corrected band is explicit and confined to
the identified thread; no non-thread geometry or acceptance threshold is excluded.

`out/threadring-nominal-verified/00-threadring/` passes all gates: valid closed
solid, forward/reverse sampled p99 ~0.0238/0.0237 mm outside the declared band,
mean section IoU ~0.9988, and no opening mismatches. The mask excludes 15,285
forward and 6,068 reverse samples. Raw deviation maps still show the intended
helix-to-nominal difference, up to approximately 0.8 mm; these raw differences
must not be presented as a globally 0.024 mm match. Visual comparison confirms
matching stock and a concentric nominal bore. The chamfer advisory corresponds
to thread-flank geometry; no separate stock-edge chamfer is evident in the source.

The native recognizer does not yet make this evidence-backed semantic choice.
The native inventory continues to exercise it without the saved recipe or thread
mask, keeping its failure visible. The default smoke suite includes both the
native diff_case case and the separate nominal threadring recipe.

## Regression coverage

All 59 unit tests pass and dependency checks pass. Added tests preserve polygon
sides and genuine circles, reconstruct circular tapers on all three translated
axes, preserve their bores, recover stationary blind voids and thin cuts, and
reject varying bores/voids. Existing feature, occupancy, opening, incomplete-gate
and non-finite-result rejection tests remain active.


The first complete-inventory attempt found an adapter regression: tighter circle
classification chose a more detailed outer loft whose constant bore had tiny
section-to-section chord noise. It became a spline and was rejected by the hard
feature check. That failed run is retained in `out/remaining-final-inventory/`;
it was stopped after diagnosis rather than mixed with results from changed code.
The fix measures the original wall vertices: a complete constant-radius wall
must support one circle within the unchanged 1e-4 mm constancy bound across its
full axial span. A 0.001 mm real taper remains a varying bore in the regression.
A fresh adapter full-pipeline run passes with an analytic Ø12 cylinder.

## Final native inventory

`out/remaining-verified-inventory/summary.json` accepts 13/14 native fixtures,
up from 12/14, retaining every previous pass. Only native threadring remains
unresolved; it is tested without a thread mask or recipe substitution. Its
feature, deviation, occupancy and opening failures keep the strict inventory
exit code at 1. There were no stage timeouts or memory kills in the final run.
The tested source hashes match the final geometry implementation.

Final diff_case artifacts: `out/remaining-verified-inventory/06-diff_case/`.
Its recipe is byte-identical to the visually reviewed final targeted result:
77 analytic faces, four measured cut bands covering two side recesses, valid
closed single solid, no hard feature gaps or opening mismatches, forward/reverse
sampled p99 ~0.02455/0.02464 mm against 0.10 mm. Missing/extra material fractions
are ~0.064%/~0.018%. The visual-review record is separate from the pipeline's
automatic pending field.

The earlier interrupted inventory and first narrow thread-band failure remain
available as diagnostics; neither is mixed into this final native count. The
full native inventory is not described as 14/14 merely because an explicitly
composed nominal threadring model can pass separately.

Both saved recipes pass the complete pipeline on the final source snapshot:
`out/remaining-battery-recipe/` and `out/remaining-threadring-recipe/`.
The final threadring recipe has identical parsed operations to the visually
reviewed nominal proposal; its comments additionally record the exact declared
band. Source hashes for both recipe runs match the final native inventory.
Shell syntax and whitespace checks pass. Remote CI was not run in this session.

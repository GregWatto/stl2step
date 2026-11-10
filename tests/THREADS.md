# Automatic nominal threaded rings

Before algorithm changes, dependencies and all 59 unit tests passed. The native
inventory accepted 13/14 fixtures; threadring remained unresolved. The saved
M10×1.5 nominal recipe already passed when its confirmed relief band was supplied
explicitly. No fixture STL or global geometric tolerance was changed.

## Identification and representation

`stlstep.threaded` recognizes a narrow, complete geometry: a plain cylindrical
ring with two flat annular caps and a full-height, single-start triangular ISO
metric internal thread. It supports all three world axes, translation, either
hand, and the metric standards in the existing table. It is not a general
fastener, pipe-thread, partial-thread, oblique-axis or multi-start recognizer.

Evidence is measured from the source:

1. The complete mesh must be closed, consistently wound and connected, with
   only a circular exterior wall, inward-facing bore wall and end caps. Every
   source face must belong to those measured surfaces. Extra stock features or
   damaged/open geometry reject this proposal.
2. The same `thread_signature` routine used by `probe thread` supplies a pitch,
   relief and candidate specification. Its angular frame now follows the axis,
   rather than assuming that every thread points along world Z.
3. Whole-wall phase fitting checks one radial cycle per revolution and axial
   pitch. The fundamental must explain at least 95% of radial variance, then a
   triangular flank must explain at least 99.5% on both the fitting subset and
   the entire wall. This rejects annular grooves,
   multi-start relief and smooth sinusoidal grooves. Both hands are tested.
4. Full angular/axial sampling and matching relief envelopes in every axial
   quarter are required. Partial threads and taper are rejected. Refined pitch
   must agree with the standard within 2%; measured crest/root diameters must
   agree with the standard's internal minor/major diameters within the recorded
   identification resolution. These are proposal checks, not relaxed gate bars.

For the fixture, the existing probe reports M10, pitch 1.452 mm and zero taper.
The additional phase evidence measures 1.499973 mm pitch and right hand;
triangular flank R² is approximately 0.99999984. Source vertex relief radii are
approximately 4.188000..5.000000 mm over z=0..12 mm. The emitted model uses the
standard minor diameter rounded to 0.001 mm: Ø8.376 mm, inside measured Ø16 ×
12 mm stock. `M10x1.5` and hand are recorded in the editable recipe and analysis.
Reapply the thread operation in downstream CAD; the STEP intentionally contains
a nominal bore rather than helical faces.

There are no fixture-name branches, saved-recipe substitutions, model APIs or
external services. Costs are bounded to 200,000 source triangles, with at most
4,096 wall vertices used for phase fitting; the fitted phase is then checked
against every inner-wall vertex after whole-mesh structural checks.
Unsupported cases retain the existing proposal paths.

## Explicit evidence and strict validation

The proposal and independently regenerated brief carry the same measured band:

- Axis point at the upper cap; t = [-height, 0], with no broad axial padding.
- Radius = measured relief envelope plus the existing 0.05 mm tessellation
  chord allowance, also covering the specified nominal minor radius.
- For this fixture: z=0..12 mm and radius approximately 4.138..5.05 mm.

`thread_bands` consumes that exact evidence for the recognized ring. Existing
band behavior for other previously identified features is unchanged. Both raw
and band-excluded proposal deviations are reported, with an explicit scope.
The complete compose pipeline still executes validity, closure, bidirectional
sampled p99, editable-feature, all-axis occupancy and opening checks.

Because a relief mask intentionally hides helix-to-nominal differences, the
feature gate now independently inspects the actual BREP for this representation.
It requires a concave analytic cylinder with the specified nominal radius,
measured axis and cap span within 1e-4 mm (or 1e-6 of the radius if larger), full
circumference, and the complete expected cylindrical surface area. Missing wall
patches, an average-radius bore, eccentricity or a blind floor fail this hard
check, even when ordinary masked surface and occupancy metrics look good.
This strengthens validation and does not loosen existing acceptance thresholds.

## Regressions and retained failures

All 73 unit tests pass; dependency checks pass. New coverage includes translated
proper rotations on all axes, reflected left-hand geometry, a different M8×1
size/stock/height, measured band bounds, nominal CAD faces, and rejection of
annular grooves, multiple starts, sine relief, plain/partial/tapered walls,
unknown dimensions, damaged stock and open meshes. Geometry tests reject wrong
nominal diameters, eccentric or blind bores, a trimmed bore-wall patch, invalid
band data and wrong outer stock that lies beyond the thread band.

`out/thread-auto-wrong-bore/` retains a deliberately incorrect Ø9.2 average bore.
Its masked distance/occupancy metrics pass, but the new hard nominal-feature
check correctly fails the complete pipeline. This is an expected rejection test,
not a suppressed reconstruction failure.

`out/thread-auto-first/` retains the first native automatic pass, without a
manual band or saved recipe. Both raw deviation maps, source sections, all-axis
occupancy overlays and final comparison were reviewed. Stock and concentric
nominal bore agree; visible thread flanks are deliberately replaced by the
nominal representation. Raw helix differences reach about 0.8 mm. A masked
~0.024 mm result must not be described as a global surface match to the helix.
The advisory chamfer finding comes from the thread-flank geometry; a separate
stock-edge chamfer is not evident in the source.

Two partial inventories were stopped while finalizing the bore-area and
whole-wall phase safeguards; their artifacts remain in `out/thread-auto-inventory/`
and `out/thread-auto-final/`. The final run uses a fresh source snapshot and
output directory.

## Reproduce

```bash
# Native proposal + independent analysis + every gate; no manual thread band:
./scripts/test.sh --fixture threadring --out out/thread-native-repeat

# Strict complete native inventory:
./scripts/test.sh --all-fixtures --out out/thread-inventory-repeat
```

The default smoke suite now includes native threadring. The older explicit
nominal recipe remains a separate regression to preserve that workflow too.

## Final results

`out/thread-auto-complete/summary.json` passes **14/14 native fixtures**, up from
13/14, preserving every previous pass. Dependencies pass and there are no stage
timeouts or memory kills. The source hashes match the final implementation.
An audit of every fresh analysis confirms that only threadring declares a thread;
no other fixture acquired a new exclusion.

Final automatic threadring artifacts: `out/thread-auto-complete/11-threadring/`:

- Valid, closed single solid with four analytic faces.
- Nominal M10×1.5 right-hand Ø8.376 bore; measured stock radius ~8.0000002 mm.
- Forward/reverse sampled p99 outside the measured relief band:
  ~0.02378/~0.02371 mm, against 0.10 mm.
- No hard feature gaps, opening mismatches or validation exceptions.
- Masked missing/extra material ~0.007%/~0.125%; mean section IoU ~0.9988.
- The band excludes 15,285 forward and 6,062 reverse samples.

The exact script, STEP, feature analysis, phase/probe evidence, metrics, source
hashes, gate execution receipts, raw deviation maps and comparisons are retained.
The final output was visually reviewed; `visual-review.json` records that review
separately from the pipeline's automatic pending field. Nominal validation is
not a claim of global geometric identity with the helical mesh. No manual band
or recipe substitution was used in this native run.

Both saved-recipe regressions also pass with the final source snapshot:
`out/thread-auto-battery-recipe/` and `out/thread-auto-saved-recipe/`. Thus the
previous explicit nominal workflow remains supported as well as native automatic
identification. Shell syntax and whitespace checks pass. Remote CI was not run.

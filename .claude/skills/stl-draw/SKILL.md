---
name: stl-draw
description: >
  Convert an STL mesh into clean, editable CAD — a build123d feature tree plus a
  STEP file — by having Claude "draw" the part: study a dimensioned analysis and
  rendered views of the mesh, compose build123d operations (sketches, extrude,
  revolve, holes, fillets), verify the result against the mesh, and refine from a
  deviation colormap. This is the img2threejs gated generate→verify→refine loop
  applied to reverse-engineering CAD. Use when the user wants to turn an STL into
  editable CAD / a STEP file, or when `convert --method auto` fell back to a
  faceted BREP solid and they want a real feature model. Claude is the composer —
  no API key is needed.
---

# stl-draw — Claude draws an STL as editable CAD

You (this Claude Code session) are the CAD modeller. Deterministic scripts do the
part analysis, run the geometry kernel (build123d/OCCT), and verify the result
against the mesh. **You** supply the judgement the scripts can't: reading the
part, composing the feature operations, and fixing what the gate flags. The goal
is a dimensionally accurate, **editable** feature tree — not faceted triangle soup.

## Environment

Run everything from the project root with the project venv:

```
PYTHONPATH=src ./.venv/bin/python -m stlstep.<...>
```

Pick a `WORKDIR` for scratch renders/scripts, e.g. `/tmp/<partname>_draw`.

## Procedure

### 1. Try the deterministic recognizers first (instant, exact when they apply)

```
PYTHONPATH=src ./.venv/bin/python -m stlstep.convert <PART.stl> -o <OUT.step> --report <WORKDIR>
```

If it prints `RESULT: PASS`, you're done: a single extrusion / revolve / prismatic
stack / loft (continuously-varying section, e.g. a square-to-round adapter) was
recognized and `<WORKDIR>/feature_tree.py` is the editable recipe. Show the user
the metrics and the feature tree — you're done.

If it prints `RESULT: APPROX`, a recognizer built a valid, **editable** feature
model that captures the structure but is over the 0.1 mm bar (deviation is
printed) — the tool surfaces it instead of silently dropping to BREP, because the
accept/refine decision is YOURS. Read the colormaps it wrote
(`<WORKDIR>/draw_missing.png` = red where a feature is missing, `draw_extra.png` =
red where material is extra) and decide:
- deviation is tiny (≲ a few tenths mm) or the residual is inherent detail
  (fillets, a rim) → run the **fitter** (step 5b) on `feature_tree.py`; it usually
  clears the last bit. Keep it if it's faithful.
- a specific feature is missing/extra (a clear red blob — a bore, a pocket) →
  start from `feature_tree.py` as your base and **refine** via the compose loop
  (steps 3–5): add that one feature, re-verify.
- the model is poor (large scattered red, wrong structure) → fall back to the
  exact BREP: `convert ... --method solid`.

Continue to the compose path below when convert fell back to BREP
(`method: solid`), when you judged an `APPROX` model needs feature-level
refinement, or when you want a cleaner feature model than BREP gives.

### 2. Build the brief (analysis + rendered views)

```
PYTHONPATH=src ./.venv/bin/python -m stlstep.compose brief <PART.stl> <WORKDIR>
```

Then **study the part**:
- Start from **`features`** (the grouped view): named holes / counterbores /
  bosses with through-vs-blind, depths, and PATTERNS (a hole with `count`>1 and
  its `positions_mm` — model with one `with Locations(*positions):`). `counterbore`
  = two-diameter hole. **`low_confidence_fits`** lists analytic fits to IGNORE
  (mis-fit sphere/torus). The raw dimensioned `surfaces` list (planes, cylinders,
  cones, tori, freeform) is the fallback for whatever `features` doesn't cover
  (pockets/slots — read those off the sections). The command also prints the bbox,
  volume, and what the deterministic recognizers saw.
- **Read the shaded render** with the Read tool: `<WORKDIR>/part_views.png` (four
  views). Confirm the arrangement of features matches the numbers.
- **Read the cross-sections**: `<WORKDIR>/part_sections.png` (two axial + two
  transverse slices). These reveal the INTERNAL geometry the shaded views hide —
  cavity floor heights, counterbore steps, wall thicknesses, chamfers. Read exact
  depths and radii off them; they are usually the fastest way to get the interior
  right.
- Treat analytic fits as hints, not truth: a large-radius sphere/torus is often a
  degenerate fit to a nearly-flat or stepped region — trust the cross-section.
- Full detail is in `<WORKDIR>/analysis.json` if you need it.

### 2b. Measure exact geometry with the probe (don't guess)

When you need a real number the render/sections can't give you precisely — a
feature's footprint at some height, whether a gap is solid or void, a revolve's
radius profile — use the read-only `probe` tool instead of eyeballing or ad-hoc
`python -c` snippets (those get gated on every run). All subcommands just print
numbers; add `--json` for machine-readable output.

```
# overall bounds / volume / watertight
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe bbox <PART.stl>
# loops at planes ⟂ an axis — extents + SHAPE per loop (<CIRCLE Ød> /
# <rect w x h> / <profiled (~N sides)>). The rect tag is boundary-verified, so
# trust it — but a <profiled> loop is NEVER a box (an HDMI port, a stadium slot).
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe sections <PART.stl> --axis z --at 0.05,0.15,0.5 --frac
# WALL OPENINGS — do NOT re-draw them by hand, EVER. Hand-transcription from
# bboxes/labels is lossy and has shipped round holes as squares and an HDMI
# port as a plain box. `cutouts` extracts every opening in a wall as its EXACT
# fitted profile (lines + arcs) and emits paste-ready Mode.SUBTRACT cuts with
# the measured wall depth — paste them; your job is placement, not tracing:
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe cutouts <PART.stl> --axis y --at=63 --min-area 2
# material extent in one plane, optionally inside a window (= form for negatives)
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe slice <PART.stl> --axis y --at 14.5 --window=-76,-54,0,20
# radius vs axial position — reads off a revolve's silhouette + bore
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe profile <PART.stl> --axis x --n 40
# inside/outside solid at points — through vs blind, void checks
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe point <PART.stl> 0,6,22 0,50,22
# INTERNAL STRUCTURE: dimension the cavities — dense void scan along an axis,
# tracked + split into ~constant segments (span/Ø/volume/open-vs-SEALED), with
# ready-to-paste cuts (--snippets) and an internal-silhouette atlas (--atlas).
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe voids <PART.stl> --axis x --snippets --atlas <WORKDIR>/voids_x.png
# ZOOM: render section DRAWINGS at chosen coords or densely across a band —
# grey=material, white=INTERNAL VOID. Use on any band the 4 default sections
# step over (a head, a cavity zone, wherever INTERNAL CAVITY was flagged).
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe slicepics <PART.stl> --axis x --range=-76,-55 --n 8 --out <WORKDIR>/zoom.png
# faceted vs curved: cluster face normals (few clusters = flat facets + fillets)
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe normals <PART.stl>
# extract a cross-section as facets + fillet arcs and EMIT an extrude snippet
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe profile2d <PART.stl> --axis z --snippet
```

When a colormap later flags a region, come back here: probe that exact region's
real geometry and match it, rather than iterating on a guessed shape.

**Threads: identify the SPEC, model the NOMINAL.** If a hole/stud wall shows a
sawtooth in the sections or a helical band (the `slicepics` zoom makes them
obvious), identify it:

```
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe thread <PART.stl> --axis z --center=-64.5,6 --range 0.5,8 --r 6.3   # add --stud for external
```

It measures minor/major Ø, pitch, and taper, and matches ISO metric + common
UNC/UNF (a tapered result means a pipe thread — NPT/BSPT). Then follow real CAD
convention — a thread's mesh geometry is the *evidence*, the spec is the
*intent*:
- **Threaded hole** → model a plain hole at the **minor Ø** (tap-drill
  representation), and record the spec as a comment in attempt.py
  (`# M10x1.5 threaded hole — modelled at tap Ø8.5`) and in your final summary.
  The engineer re-applies Thread(M10x1.5) in Fusion in seconds *because the spec
  is known*.
- **Threaded stud** → plain cylinder at the **major (nominal) Ø**, spec recorded.
- **Never surface-fit the helix.** After identifying the spec, declare the band
  to the verify step so the gates don't punish the correct nominal model:

  ```
  ... -m stlstep.compose run <PART.stl> <WORKDIR> attempt.py --thread-band "z,cu,cv,t0,t1,r0,r1"
  ```

  (axis letter, axis position cu,cv in the perpendicular plane, axial band
  t0,t1, radial shell r0,r1 spanning minor/2−0.4 to major/2+0.4; repeatable for
  multiple threads.) Deviation samples and occupancy area inside declared bands
  are excluded — with the band declared, a correct nominal model of a threaded
  part PASSES; without it, the helix residual fails the gates. State the spec +
  band in the final summary. Modeled 3D threads (for printing) are possible via
  `bd_warehouse`'s IsoThread if the user asks — it is not installed by default.

The brief helps you find threads: hole/boss lines print `THREADED <spec>` when
identified automatically, and `possible THREAD near [x,y,z]` marks clusters of
tiny cylinder fragments (a helical wall shatters into these under segmentation)
— always run `probe thread` there before modelling. With threads recognized,
fillet demands from the feature check become advisory (helix flanks can read as
fillets).

**"Curves" are usually flat facets + fillets.** Before you approximate a curved
region with boxes, run `normals`: if a handful of clusters cover ~90% of the area,
the surface is flat facets meeting at angles, joined by small fillets — NOT a true
curve. Then run `profile2d` on the relevant cross-section: it fits the section to
straight facets + fillet arcs (and bores as circles) and prints a paste-ready
`extrude` snippet in world coords. For a part that's a constant-section prism this
reconstructs to <0.1 mm on its own (it rebuilt a 52-facet aluminium extrusion to
0.012 mm); for a part that's a prism plus end features, paste the profile as the
base body and add the caps/bosses/holes on top. `profile2d` sketches on the axis
minimum and extrudes the full span by default — pass `--height` to cap it, and
`--at`/`--frac` to pick which station's section to read (use a station where the
cross-section is representative). Thin-walled OPEN shells now work directly: the
section fit preserves a thin wall band (it no longer collapses to a filled
silhouette), so `profile2d` on a thin-wall part gives the actual walls with the
open side intact — extrude it as-is (a constant-section open channel / tray even
reconstructs automatically via `convert`). For a multi-level open shell (a base
plate + thin walls of differing height, like a corner holder), take a section in
each level — the thin-wall band, the base plate, the post — and stack the
extrudes; each `point`-check the front to confirm which sides are walled.

### 2c. Seed a scaffold from detected features (`compose emit`)

Before hand-writing anything, let the deterministic side emit what it already
knows, so you start from geometry-true structure instead of transcribing numbers
by hand (which mis-places holes):

```bash
PYTHONPATH=src ./.venv/bin/python -m stlstep.compose emit <PART.stl> <WORKDIR>
# -> WORKDIR/scaffold.py   (reuses WORKDIR/analysis.json if `brief` already ran)
```

The scaffold is the **profile2d base (max cross-section, extruded along the
prismatic axis) + every detected hole auto-placed** (all axes, exact position +
axis + Ø + through/depth — the error-prone part), bosses/pockets where known, and
explicit `TODO` comments for what the analysis can't place (fillets, chamfers,
under-placed pockets, and open-saddle candidates). Recovered **open-bores / rod
channels** (a slotted bore the closed-ring pass misses) are auto-cut only when
their axis ∥ the extrude axis and the base is solid there; otherwise they're left
as a note for you to confirm and place from a section.

Copy `scaffold.py` to `attempt.py`, then **verify it immediately** (§4) to see the
starting deviation, and refine from there. The scaffold is a seed, not an answer —
its base FORM is a single-station extrude, so level-varying parts still need you.

### 3. Compose a build123d script (this is the "drawing")

Start from `<WORKDIR>/scaffold.py` (§2c) when it exists; otherwise write
`<WORKDIR>/attempt.py` from scratch:
- Start with `from build123d import *`, and leave the solid in
  `with BuildPart() as part:` (the harness looks for `part`).
- Work in millimetres, in the **same world coordinates as the analysis**, using
  the reported axes / diameters / centres / extents so the result overlays the mesh.
- **build123d gotcha — `Plane.XZ` normal is (0,−1,0)**: `offset(+d)` lands at
  world **y=−d**, and a positive extrude goes toward −Y (XY and YZ behave as
  expected). All tool-EMITTED code (cutouts/profile2d/voids/feature trees)
  already accounts for this. In YOUR own hand-written sketches on y-planes,
  either negate (`Plane.XZ.offset(-y)` + negative amounts), or sidestep planes
  entirely with `Locations((x, y, z))`-positioned `Box`/`Cylinder` — and trust
  the verify colormap to catch a mirror (a y-wall cut in empty space shows as
  missing-opening flags).
- Model the **base first** (the dominant extrude or revolve — the biggest planes
  or the rotational profile), then add features: holes as `Circle(...) ,
  mode=Mode.SUBTRACT` extrudes on the right face/plane, bosses as adds, fillets
  and chamfers with `fillet`/`chamfer` on the relevant edges.
- **INTERNAL CAVITIES — the biggest trap.** `extrude`/`revolve` build the OUTER
  form; they do NOT create internal structure. A revolve of the outer silhouette
  discards EVERY internal feature — an off-axis chamber, a swirl/volute, a
  counterbored seat. When the brief prints `INTERNAL CAVITY`, it also prints the
  **dimensioned cavity table** (each cavity's shape, Ø/bbox, axial span, volume,
  centre, and open-vs-SEALED) and writes `void_atlas.png` — Read it: the top
  chart is the internal silhouette (each cavity's radius vs station — the drill
  profile), the slices below show the void in orange at every change point.
  **Model every cavity in the table as a CUT**, working from its numbers, not
  from a guess: `probe voids <PART.stl> --axis <a> --snippets` prints
  ready-to-paste `Mode.SUBTRACT` cuts for constant circular/annular cavities;
  run it on EACH axis (an inlet crosses the main axis; a cavity appearing as
  "profiled" on one axis is often circular on another). A SEALED cavity cannot
  be cut from an end face — cut it before capping, or cut deeper and re-cap.
  The occupancy gate measures internal void and will FAIL a solid model of a
  hollow part: the inside is as much the part as the outside.
- **Fillets & chamfers**: the brief's `edge_treatments` flags approximate fillet
  radii/counts (the segmenter hides them in freeform, so they're a hint — often
  ~10% high). Fillets show as ARCS in the cross-sections, chamfers as short
  DIAGONAL cuts — read the exact size there. Apply them AFTER the base, e.g.
  `fillet(part.edges().filter_by(Axis.Z), radius=r)` (vertical edges) or
  `chamfer(part.faces().sort_by(Axis.Z)[-1].edges(), length=w)` (top perimeter);
  fillet before chamfer. The fitter (step 5b) snaps the radii.
- A cylinder tagged `hole/bore (concave)` is a cut; `boss/shaft (convex)` is added
  material. Its axis + endpoints tell you the plane to sketch on and the depth.
- Don't invent features that aren't in the analysis or visible in the render.

### 4. Verify against the mesh

```
PYTHONPATH=src ./.venv/bin/python -m stlstep.compose run <PART.stl> <WORKDIR> <WORKDIR>/attempt.py --tag iter0 --out <OUT.step>
```

Read the printed metrics, then **Read both colormaps**:
- `<WORKDIR>/iter0_missing.png` — the target mesh coloured by distance to your
  model. **Red = a feature you did NOT model** (an omitted hole/pocket/step).
- `<WORKDIR>/iter0_extra.png` — your model coloured by distance to the mesh.
  **Red = material you added that isn't there** (wrong size/position).

The metric alone can lie — always look at BOTH colormaps and the volume, not one
number. A valid, closed solid with matching volume can still be missing a feature.

**`run` also prints SLICE OCCUPANCY — the volumetric-truth check.** Surface
deviation measures how close the skins are; occupancy slices BOTH solids along
all three axes and intersects the filled regions, so it measures whether the same
3D space is actually solid — catching filled-in internal cavities, dropped
pockets, and oversize/undersize material that surface metrics under-report (and
it works on non-watertight meshes). It prints mean IoU, missing/extra material as
% of the part's volume, an explicit "internal cavity filled solid" callout when
your model is solid where the part is hollow, and the **worst slices with their
coordinates and 2D region bounds** — then writes `<tag>_slices.png` (grey=match,
red=missing material, blue=extra). **Read that image**; fix the worst slices
first, using `probe sections --axis <a> --at=<coord>` at exactly those
coordinates to read the true geometry there. PASS requires missing AND extra
under 3% of the part's volume, AND zero `OPENING MISMATCHES`: a **scale-free
per-loop check** matches every hole/opening in the mesh sections against your
model's — by SHAPE, however small. Two levels: gross (IoU < 0.85 — a Ø4 circle
modelled as a 4×4 square scores 0.755) and detail (boundary deviation > ~0.35mm
at matching area — an HDMI trapezoid modelled as its bounding box scores IoU
0.95 but boundary-deviates ~2mm at the corner cuts; `profile detail mismatch`).
`missing opening` = your model is solid where the part has a hole; `extra
opening` = you cut one the part doesn't have. Fix each named opening at its
printed centre — with `probe cutouts`, never by re-drawing.

**Zoom on demand.** The uniform pass samples ~40 stations per axis and can step
over a small feature; when it flags a band (or you suspect one), request a dense
targeted re-analysis:
```
# target vs YOUR MODEL, slice-by-slice across the band, every slice rendered:
PYTHONPATH=src ./.venv/bin/python -m stlstep.compose slices <PART.stl> <WORKDIR> <WORKDIR>/attempt.py --axis x --range=-76,-55 --n 8
# target only (what IS the true internal structure here?):
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe slicepics <PART.stl> --axis x --range=-76,-55 --n 8 --out <WORKDIR>/zoom.png
```
On a threaded vortex head this revealed, per station: the swirl-vane end face,
an annular ring chamber around the bore, and internal thread reliefs — none of
which appear in the default 4 sections. Read the zoom image, then model exactly
what each station shows.

**Deviation cannot see small features, so `run` also prints a feature check.**
Because a missing fillet/hole/chamfer barely moves the distance metric, the verify
step enumerates the part's real features (from `analysis.json`) and checks your
model actually contains them, printing `MISSING FEATURES` for any hole/boss/fillet/
chamfer that isn't modelled. **This is the anti-"stop too soon" gate:** a matching
*shape* is worthless to an engineer who needs to edit those features. `STATUS: PASS`
now requires deviation under a **size-relative** tolerance (0.1% of the bbox
diagonal — printed each run — so 1 mm counts as huge on a small part) AND no
missing hole/boss/fillet. (Chamfers are flagged as a reminder, not a hard block;
the feature detector is approximate, so if a flagged feature is verifiably NOT on
the part — check the render/sections — note that and move on. But never ignore a
real one.)

### 5. Refine — model the FEATURES, don't settle for the shape

Fix the largest red regions AND clear every `MISSING FEATURES` line — add the
holes (`Hole`/`Circle`+SUBTRACT), bosses, and rounded edges (`fillet(...)`,
`chamfer(...)`) as real, editable operations. Bump the `--tag` (iter1, iter2, …)
and re-run step 4. Done = `STATUS: PASS`: deviation and reverse both under the
printed size-relative tolerance, a valid closed solid, volume matching, and no
un-modelled features. A "faithful-looking" solid that still lists missing features
is NOT done — keep going.

### 5b. Snap dimensions with the parameter fitter

Get the STRUCTURE right (the correct set/arrangement of features); you do NOT need
to nail every dimension by eye. Once the shape is right but deviation is still
~0.2–3 mm from imperfect numbers, let math finish it:

```
PYTHONPATH=src ./.venv/bin/python -m stlstep.compose fit <PART.stl> <WORKDIR> <WORKDIR>/attempt.py --out <OUT.step>
```

This treats every numeric literal in your script as a free parameter and optimises
them (Powell) to minimise deviation to the mesh — snapping radii, positions,
depths, and offsets to their true values (it fixed a plate+hole from 4.0 mm to
0.014 mm, recovering an exact hole position from a 3.6 mm error). It writes
`attempt_fitted.py` and verifies + exports it. If the structure is right, this is
usually what takes you under the size-relative tolerance. If it does NOT help
much, your feature structure is wrong (a missing/extra feature) — go back to step
5, not fit. Fit snaps dimensions; it cannot add a feature you never modelled, so
clear the `MISSING FEATURES` list first.

### 6. Final visual gate (structural check before you declare done)

The colormap is the per-iteration judge; run this ONCE at the end as the
human-equivalent "does it look like the part" check. It catches STRUCTURAL errors
the colormap can under-report — a wrong cross-section (box vs hex), a flipped or
mis-proportioned body — and it still works when the mesh is **non-watertight**
(where the deviation/`contains` passes are unreliable).

```
PYTHONPATH=src ./.venv/bin/python -m stlstep.compose compare <PART.stl> <WORKDIR> <WORKDIR>/attempt.py --tag final
```

Read `<WORKDIR>/final_compare.png`: for each view it shows TARGET | CANDIDATE |
overlay. In the overlay, **RED = material in the part you did NOT model (a missing
feature); BLUE = material you added that isn't there**; grey = agreement (a 1-px
edge fringe is filtered out, so any coloured *patch* is real). Do NOT rubber-stamp
— actively LIST what differs per view and NAME the shape difference, then if a
solid red/blue patch remains, go back to step 5 (measure it with `probe` first).
Only when the overlay is essentially all-grey is the structure right.

### 7. Finish

When `STATUS: PASS`, the `run --out` step has exported the STEP and written
`<WORKDIR>/feature_tree.py` — the editable feature recipe. Show the user the final
metrics, the feature tree, and where the STEP is.

If it won't converge after several iterations, first ask WHY before falling back:
almost every mechanical part IS buildable from base features (extrude / revolve /
loft / hole / fillet / chamfer + cuts) — "hard" usually means a feature you
haven't measured yet, not an unbuildable part. BREP is acceptable ONLY for a
genuinely organic/freeform region (sculpted blends, scanned surfaces) — never as
a substitute for modelling ordinary features. The whole-part fallback is:
`PYTHONPATH=src ./.venv/bin/python -m stlstep.convert <PART.stl> -o <OUT.step> --method solid --report <WORKDIR>`

## Principles (from this project's hard-won lessons)

- **The dimensional metric lies** — cross-check the colormaps, the volume, and
  topology validity, not a single number. A dropped feature can hide behind a
  good average.
- **Prefer the simplest faithful feature tree** — fewer, cleaner operations an
  engineer can actually edit in Fusion.
- **Verify, don't assume** — every attempt is measured against the real mesh; keep
  a change only if the gate confirms it helped.
- **Measure, don't guess** — when a colormap flags a region, extract that region's
  real geometry with the `probe` tool (step 2b: `sections` / `slice` / `profile` /
  `point`) and match the numbers. Do not eyeball a shape and iterate on guesses
  (the vortex foot was wrong for 3 iterations that way; one `sections` probe — a
  20.5 mm rectangle, not the 30 mm assumed — fixed it).

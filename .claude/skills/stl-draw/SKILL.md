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
stack was recognized and `<WORKDIR>/feature_tree.py` is the editable recipe. Show
the user the metrics and the feature tree. Only continue below if it fell back to
BREP (`method: solid`) or you want a cleaner feature model than BREP gives.

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
# loops (with X/Y/Z extents) at planes ⟂ an axis — footprints at heights
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe sections <PART.stl> --axis z --at 0.05,0.15,0.5 --frac
# material extent in one plane, optionally inside a window (= form for negatives)
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe slice <PART.stl> --axis y --at 14.5 --window=-76,-54,0,20
# radius vs axial position — reads off a revolve's silhouette + bore
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe profile <PART.stl> --axis x --n 40
# inside/outside solid at points — through vs blind, void checks
PYTHONPATH=src ./.venv/bin/python -m stlstep.probe point <PART.stl> 0,6,22 0,50,22
```

When a colormap later flags a region, come back here: probe that exact region's
real geometry and match it, rather than iterating on a guessed shape.

### 3. Compose a build123d script (this is the "drawing")

Write a build123d script to `<WORKDIR>/attempt.py`:
- Start with `from build123d import *`, and leave the solid in
  `with BuildPart() as part:` (the harness looks for `part`).
- Work in millimetres, in the **same world coordinates as the analysis**, using
  the reported axes / diameters / centres / extents so the result overlays the mesh.
- Model the **base first** (the dominant extrude or revolve — the biggest planes
  or the rotational profile), then add features: holes as `Circle(...) ,
  mode=Mode.SUBTRACT` extrudes on the right face/plane, bosses as adds, fillets
  and chamfers with `fillet`/`chamfer` on the relevant edges.
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

### 5. Refine

Fix the largest red regions, bump the `--tag` (iter1, iter2, …), and re-run step 4.
Target: max deviation and reverse both `< 0.1 mm`, valid closed solid, volume
matching the mesh.

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
usually what takes you across the `< 0.1 mm` line. If it does NOT help much, your
feature structure is wrong (a missing/extra feature) — go back to step 5, not fit.

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

If it won't converge after ~4 iterations (e.g. a genuinely freeform/organic body),
say so plainly and fall back to the BREP solid:
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

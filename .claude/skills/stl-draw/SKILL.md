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
- The command prints a dimensioned list of every surface (planes with normals;
  cylinders labelled hole-vs-boss with Ø, axis, length, endpoints; cones; tori;
  freeform patches), the bounding box, volume, and what the deterministic
  recognizers saw.
- **Read the render** with the Read tool: `<WORKDIR>/part_views.png` (four shaded
  views). Confirm the arrangement of features matches the numbers.
- Full detail is in `<WORKDIR>/analysis.json` if you need it.

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

### 6. Finish

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

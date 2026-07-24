# Hard-won lessons (don't rediscover these)

Technical gotchas and dead ends learned the expensive way this project. Read
before touching the relevant area.

## The dimensional metric LIES — always cross-check

The single most important lesson (it also killed v1). A low deviation/chamfer
number does NOT mean the shape is right:
- **The sleeve "donut"**: an early analytic `shell` reported 0.16mm deviation
  while the render was an unrecognizable donut of spiraling ribbons (RANSAC had
  hallucinated 24 tori on a 10mm ring). The number was fine; the shape was garbage.
- **The bearing**: a mis-fit torus gave correct *volume* but 2.6mm deviation.
- **Volume match is especially weak**: the stepped shaft reconstructed at 99.96%
  volume while built *axis-flipped* (5mm surface deviation) — volume integrates
  out the error. And the vslot BREP `solid` reported volume 8151 when the true
  mesh volume is 17181 (2x wrong) and the gate didn't catch it.

Therefore: gate on **per-region** deviation (not a single aggregate), a **visual
colormap render**, **topology validity/closedness**, AND **bidirectional**
distance (surface→mesh catches a surface that passes through the vertices but
bulges elsewhere). Verify draw results by exporting build123d STL and measuring
max deviation, not just volume.

## Face-stitching (the BREP path) — why it's hard

Confirmed by experiment (see `docs/topology-experiments.md`):
- A triangle edge on a cylinder is an **arc**; its neighbour in a flat region
  draws it as a **chord**. They don't coincide → no sewing tolerance closes a
  watertight analytic solid. **Arc edges** merge within a region but won't sew
  across regions; **chord edges** sew across regions but sit off the curved
  surface so faces are invalid and won't merge. The only geometry on both
  surfaces at once is the true **surface-surface intersection curve**
  (`GeomAPI_IntSS`) — no shortcut.
- Full periodic faces (360° cylinders) need a **seam**; build them with the
  uv-bounded constructor `BRepBuilderAPI_MakeFace(surf, umin, umax, vmin, vmax)`,
  which adds the seam automatically. Building from two circle edges = invalid
  (missing seam).
- **Clean segmentation is the root cause of everything downstream.**
  Region-growing (`segment_rg`) >> global RANSAC peeling: 24 fake tori on a ring
  → 5 real regions. Grow smooth patches by adjacent-normal continuity (25°),
  cut at sharp edges; large genuinely-smooth regions become freeform.
- **Graceful degradation is essential**: guard every analytic merge with
  area + BRepCheck-validity + bidirectional-deviation, and fall back to faceting.
  A bad fit must never corrupt the solid — the vslot BREP failures all traced to
  an unguarded bad fit.

## The pivot: "draw it" beats face-stitching

The realization that reframed the project: building at the **feature level**
(sketch + extrude/revolve) makes topology **correct by construction** — no seams,
no trimming, no sewing. The vslot that broke face-stitching (4330 faces, wrong
volume) draws trivially as extrude(profile, 100). For feature-based mechanical
parts (most of them), draw it; only fall back to BREP for genuinely freeform.

## Draw-path gotchas (draw.py)

- **Tangent-arc radius accuracy**: group profile points by **curved vertices**
  (turning >= curve_deg), NOT by edges. Edge-based grouping includes a flanking
  straight point that pulls the circle fit (a true r=3 rounded rect fit as
  r=3.95). Vertex-grouping → the arc's endpoints are the tangent junctions →
  exact radius.
- **Line vs arc classification** is genuinely fiddly with tangent transitions and
  multi-scale features. Single-arc-per-run mis-fits 90° L-corners as arcs
  (chord≈diameter, r≈3.5, wrong). Greedy circle-growth swallows tangent straight
  edges. Curvature-based (by turning rate) works. Backstop it with
  **verify-and-fallback**: fit with arcs, verify the built solid vs mesh, drop
  arcs if worse. A 90° L-corner has chord ≈ 2r and sagitta ≈ r — looks arc-like
  but isn't; the verify guard catches it.
- **ThreePointArc(p0, mid, p1)**, not CenterArc: uses the exact polyline
  endpoints (shared with neighbours) so the wire closes; CenterArc recomputes
  endpoints from the fitted circle and leaves gaps.
- **Dedup** near-coincident section points (`_dedup`, tol 1e-2mm) — mesh sections
  emit doubled vertices that become zero-length edges after rounding.
- **Revolve frame handedness**: on `Plane(origin, x_dir=radial_dir, z_dir=normal)`
  the sketch's local Y = normal × radial_dir. Use `radial_dir = axis × normal`
  so local Y = +axis; otherwise the profile builds flipped along the axis
  (correct volume, huge surface deviation).
- **Emitted-script float bug**: `f"{tuple(np.round(v,4))}"` renders numpy scalars
  as `np.float64(...)`, which needs `import numpy` and breaks the standalone
  script. Convert to plain `float` before formatting.
- **build123d API**: `export_stl(part, path, tolerance=, angular_tolerance=)` and
  `export_step(part, path)` are module functions, not methods. `part.wrapped` is
  the TopoDS_Shape for OCP interop.

## OCCT / OCP notes

- `edge.Reversed()` returns a generic `TopoDS_Shape`, not `TopoDS_Edge`; cast with
  `TopoDS.Edge_s(...)` before passing to `MakeWire`.
- `occ_util.shape_to_mesh` tessellates per-face → the trimesh isn't watertight
  even for a closed solid; fine for deviation sampling, misleading for watertight
  checks. Use BREP-level `TopAbs_SOLID` + `BRepCheck_Analyzer` for validity.
- A solid can be BRepCheck-invalid at the *solid* level while every face/edge is
  fine (a strict global self-intersection check); a `ShapeFix` pass usually
  clears it. Negative `solid_volume` = inward orientation → `.Reversed()`.
- OCCT leaks process-wide state on big batch runs; v1 needed subprocess-per-part.

## Composing complex parts by hand (the skill loop) — process lessons

Hard-won on the live `vortex.stl` test (a threaded vortex tube), where the same
mistake was made twice:

- **MEASURE the region the colormap flags — never GUESS its shape.** The revolve
  body was near-exact because its radial profile was *extracted* from the mesh and
  matched. The mounting foot was wrong for 3 iterations because it was *eyeballed*
  (box → wider box → hexagon) from a couple of width samples. One measurement of
  the actual footprint (a 20.5×21mm rectangle) fixed it immediately. When a region
  is red, section it / slice it horizontally / extract its profile, then match.
- **Read the colormap for the RIGHT cause.** The foot's persistent ~8mm reverse
  spike was its LENGTH (it overhung ~10mm under the neck), not its cross-section
  shape — but the error appeared "near the foot" so it got misattributed to shape.
  Check extent/position before fiddling with shape.
- **Don't stop while a red block is on screen.** Twice the loop was halted with a
  premature "good enough, concept proven" — leaving a glaring, fixable fault. The
  deviation colormap literally shows where to look; keep going until it's down or
  the residual is provably inherent (thread crests, faceting).
- **Decompose a complex part into primitives.** vortex became tractable once split
  into revolve(outer silhouette) + revolve-subtract(bore profile) + foot + inlet —
  not attacked as one blob. Extract the radial profile r_outer(X), r_bore(X) for
  anything shaft/tube-like; revolve about the found axis (here X @ y=6.06,z=25.01).
- **Threads: model as a spec, not a surface fit.** For a "decent interpretation"
  smooth a threaded region to a plain cylinder at its effective Ø; for a faithful
  one, add build123d `IsoThread`/`TrapezoidalThread` from the thread spec. Never
  try to reverse the helix geometrically (even commercial tools use thread libs).
- **The `fit.py` "didn't help" signal is diagnostic.** On Mold-bottom the fitter
  didn't move 0.16mm → that meant the pin STRUCTURE (shape) was wrong, not the
  numbers. Measured the pin profile (r2.0 cylinder + hemisphere) → 0.018mm PASS.

## How the loop "judges" (vs img2threejs)

img2threejs judges perceptually: render the generated scene → a vision model
compares the screenshot to the target IMAGE → iterate. Our target is a MESH, so we
judge GEOMETRICALLY: surface-to-surface deviation + the two colormaps (missing =
mesh coloured by dist to model; extra = model coloured by dist to mesh). The
colormap is the better *per-iteration* judge — it pinpoints WHERE and by how much.
A vision gate (render candidate vs target from matched angles) is best reserved as
a FINAL check or for genuinely ambiguous geometry — it's not built yet.

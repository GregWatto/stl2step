# stl-step-v2 — agent context (read this first)

Master context for resuming work after a context compaction. Companion files:
- `lessons.md` — hard-won technical gotchas (expensive to rediscover)
- `roadmap.md` — what's next, especially the LLM-composed multi-feature path
Also see repo `README.md`, `docs/v1-lessons.md`, `docs/topology-experiments.md`.

## The goal (the essence)

Convert STL meshes (triangle soup) into **clean, dimensionally-accurate, editable
CAD models** — real analytic surfaces / a feature tree — NOT the highly-faceted
STEP that free converters and Fusion's free "Convert Mesh" produce. The user is a
**mechanical engineer on Fusion 360 (Mac, Apple Silicon)**. No feature *history*
was originally required, but building one (a parametric recipe) turned out to be
both achievable and the best path, so we do.

This is **v2**. v1 (`~/Documents/Projects/stl-to-step`) was abandoned: a
RANSAC + face-stitching pipeline that produced dimensionally-plausible but
visually broken STEPs. See `docs/v1-lessons.md`.

## The guiding idea (img2threejs)

Inspired by hoainho/img2threejs: a **gated generate → verify → refine loop** where
deterministic scripts enforce gates and the model does only the judgment it's
uniquely good at. Transferred here as:
1. **Dimensional QC gates** at every stage (deviation vs the mesh, topology
   validity) — because the numeric metric alone *lies* (see lessons).
2. **"The LLM draws the part"** — reconstruct by composing CAD operations
   (sketch/extrude/revolve/…) that a kernel executes, verified dimensionally.
   This sidesteps the face-stitching topology hell entirely.
3. **verify-and-fallback** everywhere: generate, verify against the mesh, keep
   only if it's actually better; otherwise fall back to a safe exact version.

## The architecture (two strategies + a dispatcher)

**A. Draw it (parametric feature model)** — `src/stlstep/draw.py`. Recognize how
the part was built and re-draw it with build123d feature ops, so topology is
correct by construction and the output is an editable **feature-tree script**.
- `detect_extrusion` → constant cross-section along an axis → sketch + `extrude`
- `detect_revolve` → rotational symmetry → (radial, axial) profile + `revolve`
- profile fitting: `fit_loop` → clean lines / arcs / circles
- `draw(mesh)` dispatcher: try each recognizer, verify, return best or None

**B. BREP fallback** — for parts that are NOT a single extrude/revolve. Pipeline:
`ingest → segment_rg → surfaces/edges/nurbs → topology → verify`. Produces a
watertight solid with analytic surfaces and clean intersection-curve edges.
Region-by-region fallback (clean → polyline merge → NURBS drape → facet), guarded.

**Dispatcher** — `convert.py` `method="auto"` (default): draw a feature model;
if the part isn't drawable, fall back to the `solid` BREP method.

## Current state (all committed, latest 31464a3)

Working end-to-end via one CLI. Per-fixture results:

| fixture | route | deviation | notes |
|---|---|---|---|
| vslot | draw:extrude | 0.012mm | V-slot extrusion, 91 faces (was 4330 broken via BREP) |
| shaft | draw:revolve | 0.002mm | synthetic stepped shaft |
| sleeve | draw:revolve | 0.04mm | ring/washer |
| bearing | draw:revolve | 0.08mm | races become arcs in the profile |
| battery_cover | BREP solid | exact | curved cover body → BREP fallback |
| diff_case | BREP solid | exact | complex multi-feature → BREP fallback |

Drawable parts emit a runnable `feature_tree.py` (build123d recipe) that
round-trips. BREP parts are valid, watertight, correct-volume.

## Key modules (src/stlstep/)

- `convert.py` — orchestrator + CLI; `convert()`, `main()`, `_finish_draw()`
- `draw.py` — the draw path (detect_extrusion/revolve, fit_loop, draw_*, `draw`)
- `ingest.py` — load + repair mesh (reuses v1 mesh_repair), face adjacency
- `segment_rg.py` — region-growing segmentation (the good segmenter)
- `segment.py` + `_detector.py` — old RANSAC segmenter (kept for A/B; inferior)
- `surfaces.py` — per-region face building helpers (region boundary loops)
- `edges.py` — clean analytic edges (uv-bounded faces, GeomAPI_IntSS curves)
- `nurbs.py` — B-spline drape for gentle freeform regions (height-field only)
- `topology.py` — `build_solid` (methods: facets/merged/shell/solid), merge guards
- `occ_util.py` — OCCT/OCP idioms (surfaces, sew, STEP I/O, tessellate, validity)
- `verify.py` — the honest gate (per-region deviation + topology + colormap PNG)

## How to run / test

Env: venv at `.venv` (Python 3.11, Apple Silicon). Deps in `requirements.txt`:
cadquery-ocp, trimesh, numpy, scipy, rtree, pyransac3d, matplotlib, networkx,
shapely, build123d.

```bash
cd /Users/nickp/Documents/Projects/stl-step-v2
PYTHONPATH=src ./.venv/bin/python -m stlstep.convert <part.stl> -o <out.step> --report <dir>
```

Fixtures: `tests/fixtures/{vslot,shaft,sleeve,bearing,battery_cover,diff_case}.stl`
(+ `.ground_truth.step` for the corpus ones). Big test corpus of (mesh, authored
STEP) pairs at `~/my-project-data/stl-step-data-v1/` (4431 STL / 1621 STEP).

Common verification idiom (draw parts): export the built part to STL via
build123d `export_stl` and compare to the source with trimesh proximity — do NOT
trust `occ_util.shape_to_mesh` for watertightness (it tessellates per-face).

## Working agreements

- **Do not auto-commit** — the user reviews diffs first. (They do explicitly ask
  to commit at milestones; this session's work is committed.)
- Wrap shell paths containing spaces in double quotes.
- Prefer editing existing files over creating new ones.
- Generated output (STEP/reports) goes to /tmp or the user's ~/Downloads for
  review in Fusion; root-level `*.step` / `*_report/` are gitignored.

## Next step

The **LLM-composed multi-feature path** — see `roadmap.md`. For parts that are
base + holes + pockets + bosses + fillets (most real mechanical parts), the LLM
composes the build123d operations, guided by the analysis, verified by the gate.
That turns the BREP-fallback parts (battery_cover, diff_case) into clean feature
trees. This is the main event remaining.

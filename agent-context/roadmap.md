# Roadmap

## Done

- BREP pipeline: region-growing segmentation → analytic surfaces (plane/cylinder/
  cone/sphere/torus) + NURBS drape + clean intersection-curve edges → guarded
  watertight solid. Honest per-region gate.
- Draw path: **extrusion**, **revolve**, and **prismatic-stack (2.5D)**
  recognizers → build123d feature-tree scripts, with accurate line/arc/circle
  fitting + verify-and-fallback.
- `auto` dispatcher (default): draw if possible, else BREP `solid`.
- **LLM composition path** (`compose.py`): `analyze` → dimensioned feature report,
  `render_views`, `run_candidate` (exec+verify+two colormaps), `compose_loop`
  (generate→verify→refine). Driven keyless by the **`stl-draw` skill** (Claude is
  the composer, like img2threejs) or headless by `AnthropicComposer`
  (`--method llm`). Proven on `crossblock` (cross-axis hole, 0.02mm, first pass);
  the refine loop catches a dropped feature (9mm + wrong volume).

## Next: extend the composition path

Remaining to make the LLM path cover the hardest real parts:
- **Fillet/chamfer support**: detect fillet edges in `analyze` (they currently
  read as freeform/torus regions) and prompt the composer to add `fillet`/
  `chamfer`. Verify radii via the colormap loop.
- **Live refine runs**: drive the `stl-draw` skill on `battery_cover`/`diff_case`
  to turn the last BREP-fallback parts into feature trees (or confirm freeform).
- **Headless smoke**: run `--method llm` once with a real key to exercise the
  multi-turn `AnthropicComposer` end-to-end.

## Original plan: the LLM-composed multi-feature path (now largely built above)

**Why**: parts that are `base + holes + pockets + bosses + fillets` (most real
mechanical parts) are neither a single extrusion nor a single revolve, so today
they take the BREP fallback (battery_cover, diff_case). The prize is drawing
*these* as clean feature trees too.

**The idea** (full img2threejs loop): the deterministic analysis identifies
candidate features; the **LLM composes** the build123d operation sequence; the
kernel executes; the gate verifies dimensionally and drives refinement.

**Concrete shape of it:**
1. **Analysis → structured feature report.** Reuse `segment_rg` (planes,
   cylinders, etc.) plus new detectors: through/blind holes (cylinder + parallel
   caps — v1 had `feature_detect.py` for this, reusable), pockets, bosses, fillet
   edges, hole patterns (linear/circular), the dominant base (biggest
   extrude/revolve). Emit a machine-readable description with exact dims + a few
   rendered views of the mesh.
2. **LLM composes build123d ops.** Given the report + views, the LLM writes a
   build123d script: base sketch+extrude/revolve, then `.holes()`, pockets
   (sketch+extrude(mode=SUBTRACT)), bosses, `fillet`/`chamfer`. Start with a
   deterministic composer for the common cases (base + N holes) to prove the
   pipe, then hand the ambiguous composition to the LLM.
3. **Execute + verify.** Build the script → STEP. Run the gate (deviation vs
   mesh, topology validity, colormap). If deviation > tol or a region is
   unexplained, feed that back (which feature is off / what's missing) and let
   the LLM refine — bounded iterations, then fall back to BREP `solid`.
4. **Output**: the feature-tree script + STEP, same as the extrude/revolve path.

**De-risking order**: start with "base extrude + through-holes" on a real part
(e.g. a bolted flange / plate with a hole pattern). Prove analysis→compose→verify
end-to-end deterministically, THEN introduce the LLM for the composition/refine
step where judgment is actually needed.

**Key design principle** (from the whole project): the LLM never emits geometry
directly — it plans/composes CAD operations; the kernel (build123d/OCCT) executes
deterministically; the gate verifies. Keep the LLM to the judgment it's uniquely
good at (which features, how they compose, is this render right).

## Smaller follow-ups (as needed)

- **More recognizers** in the draw path: cone/sphere/torus as primitives; simple
  sweeps.
- **Corpus eval**: run `auto` over a sample of `~/my-project-data/stl-step-data-v1/`
  (subprocess-per-part; OCCT leaks) → scoreboard of draw-rate, deviation,
  feature counts, vs the authored STEPs. `eval/run_corpus.py` is a stub.
- **BREP refinements** (only if BREP-fallback quality matters): partial-cylinder
  reconstruction (curved walls in diff_case build slightly wrong → faceted);
  bearing-race torus fit accuracy; multi-patch NURBS for wrapping freeform
  (battery_cover body).
- **Packaging**: `pyproject.toml` for `pip install -e .` → a plain `stlstep`
  command (currently needs `PYTHONPATH=src`).
- **Vision-in-the-loop gate**: an actual vision model judging the colormap/render
  to catch qualitative failures the metric misses (the "donut" class) — the
  purest img2threejs transfer, still not built.

## Build-vs-buy reality (keep honest)

Clean mesh→editable-CAD is solved commercially (Geomagic Design X, QuickSurface,
Fusion's paid Convert Mesh). The BREP path competes with the free faceted
converters and wins on prismatic parts; the draw path is genuinely differentiated
(editable feature trees). If the user just needs results on a given part, Fusion's
paid Mesh workspace is the pragmatic alternative — worth re-checking against real
parts before large further investment.

# stl-step-v2

A scoped, **gated** STL → STEP converter: turn a triangle mesh into a
dimensionally accurate BREP solid with analytic surfaces (planes, cylinders,
cones, tori) where they exist and freeform NURBS patches elsewhere — more
editable than an STL, without a feature-history tree.

This is a from-scratch second attempt. The first
(`~/Documents/Projects/stl-to-step`, abandoned) produced dimensionally-plausible
but visually broken STEPs. v2 is organized around fixing its three specific
failure modes — see [`docs/v1-lessons.md`](docs/v1-lessons.md).

## The three ideas

1. **Coverage guarantee.** Every mesh triangle lands in exactly one region —
   analytic if a primitive fits, freeform NURBS otherwise. Nothing is ever
   dropped, so missed features can't become slits or pie-wedges.
2. **Real edges.** Adjacent surfaces meet along the true surface-surface
   intersection curve (`GeomAPI_IntSS`), not a mesh-boundary trace.
3. **An honest gate.** QC is reported **per region** (worst-case p95 deviation),
   plus topology validity, face-count reduction, and a deviation-colormap render.
   No single aggregate number that can hide a dropped feature.

## Pipeline

```
ingest  →  segment  →  topology  →  verify
(repair)   (100%       (faceted     (per-region deviation +
           coverage)    or merged)   topology + colormap render)
```

## Usage

```bash
python -m stlstep.convert part.stl -o part.step --report out/ [--method merged]
```

`--report out/` writes `report.json` (the full gate verdict) and `deviation.png`
(the colormap). Exit code is non-zero when the gate fails.

Methods (see [`docs/topology-experiments.md`](docs/topology-experiments.md)):
- `facets`: one BRep face per triangle. Dimensionally exact, watertight, 0% reduction.
- `merged`: watertight planar-merge (coplanar faces collapsed); curved regions faceted.
- `shell`: analytic faces, non-watertight (reference only).
- **`solid` (the main output)**: watertight solid with **analytic surfaces and
  clean analytic edges**. Each region → an analytic face (cylinder/plane/cone/
  sphere/torus); curved faces built via the uv-bounded seam constructor, plane
  boundaries from real `GeomAPI_IntSS` intersection curves (circles/lines). Bad
  fits fall back region-by-region (clean → polyline merge → facet), guarded by an
  area+validity+bidirectional-deviation check, so the solid is always valid and
  correct-volume.

## Status (region-growing segmentation + `solid`)

| Part | Faces | Valid | Watertight | Volume Δ | Deviation | File |
|---|---|---|---|---|---|---|
| sleeve | 5 | ✅ | ✅ | 0.05 mm³ | 0.013 mm | 239 KB |
| diff_case | 176 | ✅ | ✅ | 1.05 mm³ | 0.005 mm | 1.9 MB |
| battery_cover | 2602 | ✅ | ✅ | 0.00 | 0.000 | 6.3 MB |
| bearing | 14294 | ✅ | ✅ | 0.00 | 0.000 | 37 MB |

Clean edges cut file size 28–54× on well-reconstructed parts (sleeve 13 MB → 239 KB).

**Remaining limits** — big files come from regions that stay *faceted*:
- battery_cover has a large genuinely-curved body (normals span 90°) → needs a
  **NURBS drape** (not yet built).
- bearing races are toroidal but the torus fit is geometrically off → they fall
  back to faceting. Needs a better torus fit (or NURBS).
- Occasional analytic region (e.g. sleeve's keyed flat) falls back to a polyline
  merge rather than clean edges.

## Layout

- `src/stlstep/` — pipeline stages (`ingest`, `segment`, `surfaces`, `topology`,
  `verify`, `convert`) plus `occ_util` (OCCT idioms) and `_detector`/`_mesh_repair`
  (lifted verbatim from v1).
- `tests/fixtures/` — four parts with ground-truth STEPs, incl. the three that
  beat v1 (diff_case ribs, sleeve fillet, battery_cover flat).
- `eval/` — corpus scoring harness.
- `docs/v1_ref/` — v1 source kept for reference.

# Topology experiments: what closes a watertight analytic solid (and what doesn't)

Turning a segmented mesh into a watertight solid with *analytic* faces
(cylinders, tori, planes as single faces) is the crux of mesh→CAD. This records
the approaches tried, with the exact failure point of each, so the wall isn't
re-hit blindly. All were run on the four fixtures.

## The core tension (proven, not assumed)

A triangle edge can be represented two ways:

- **Arc edge** — a straight segment in the surface's (u,v), whose 3D curve lies
  *on* the analytic surface. Adjacent triangles *of the same region* share an
  identical arc, so `UnifySameDomain` merges them into one analytic face.
  ✅ within-region merge works (13 cylinder triangles → 1 cylindrical face).
  ❌ **across regions it fails**: a plane region draws the shared boundary edge
  as a straight chord, the cylinder region draws it as an arc — different 3D
  curves, so sewing can't join them. Sleeve stays at 6 shells / 325 free edges
  at any tolerance.

- **Chord edge** — a straight 3D segment between the two mesh vertices, with a
  pcurve attached on the surface. All shared edges are identical chords, so they
  coincide across regions and sew.
  ✅ cross-region sewing connects.
  ❌ **within-region merge fails**: the chord sits up to a sagitta *off* the
  curved surface, so `SameParameter` fails, the face is invalid, and
  `UnifySameDomain` refuses to merge. Faces stay at 632 → 632.

You cannot have both with polygonal edges. The only representation that lies on
*both* adjacent surfaces at once is the **true surface-surface intersection
curve** (`GeomAPI_IntSS`). That is the real fix and the real cost.

## Attempts

1. **Rebuild analytic face from region boundary loop** (v1's approach). The
   boundary polyline, projected to the fitted plane, **self-intersects**
   (`BRepCheck_SelfIntersectingWire`) because RANSAC regions grow to include
   faces up to 2 mm off-plane. Ragged segmentation boundaries defeat it.

2. **Per-triangle-on-surface + `UnifySameDomain`** — merges beautifully *within*
   a region, fragments *between* regions (arc-vs-chord above).

3. **Shared `TopoDS_Edge` per mesh edge (chord) + pcurves** — all faces build,
   deviation ~0.08 mm, but `UnifySameDomain` won't merge (chords off-surface →
   invalid) and it isn't watertight.

## What ships instead

- `merged` (**watertight**): start from the faceted solid (already watertight)
  and let `UnifySameDomain` merge coplanar faces while preserving topology.
  Robust, valid, exact, 46–74% face reduction. Curved regions stay faceted.

- `shell` (**editable, not watertight**): per-region analytic merge (approach 2
  within each region). Real cylinder/torus/plane faces, ~76% reduction on
  sleeve, deviation 0.1–0.35 mm (the segmentation fit residual). Free edges
  remain between regions.

## The path to watertight analytic (future work, large)

For each pair of adjacent regions, compute `GeomAPI_IntSS`, build a shared edge
on the true intersection curve, and trim *both* faces to it. This is the correct
solution and what commercial tools do. Two caveats make it a major effort:

- It is intricate OCCT topology (v1 spent weeks here and abandoned it).
- Its quality is capped by **segmentation**: RANSAC over-fits (diff_case → 559
  tiny cylinders, 746 regions). Better segmentation (region-growing pre-RANSAC,
  or CGAL Efficient RANSAC) is a prerequisite for the intersection pipeline to
  produce few, correct faces.

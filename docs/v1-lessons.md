# What killed v1 (and how v2 answers it)

v1 (`~/Documents/Projects/stl-to-step`) ran end-to-end and produced STEP files,
but they were visually broken. Its post-mortem pins three causes. v2 is built to
answer each.

### 1. Dropped/misassigned regions → slits, pie-wedges, jagged walls
The RANSAC detector missed small features (ribs, screw holes) and over-fit others
(three tori on one fillet). Anything unassigned was dropped; boundary-tracing drew
the gaps as slits. Battery Cover came out a "flat rectangle with no borders."

**v2:** `segment.py` guarantees 100% coverage — leftover faces are clustered into
freeform regions and NURBS-draped, never discarded. (Confirmed: battery_cover's
missing border/fillet faces now form a freeform region instead of vanishing.)

### 2. Mesh-boundary edges → degenerate topology on curved surfaces
Edges were traced from mesh boundaries and trimmed with a u/v bounding box.
A fillet band on a torus has two boundary circles, not outer+hole — v1 produced
degenerate faces.

**v2:** `topology.py` (merged method) computes real surface-surface intersection
curves (`GeomAPI_IntSS`) for shared edges and trims to them; sews + `ShapeFix` to
a watertight solid.

### 3. A single aggregate chamfer number hid everything
Mean sample distance over the whole part stayed low even when features were
dropped, because big flat/cylindrical regions dominate. "The failure mode only
became obvious when we opened a STEP in Fusion."

**v2:** `verify.py` reports deviation **per region** (worst-case p95 + the single
worst region), plus topology validity, face-count reduction, and a
deviation-colormap render — the "open it in Fusion" check, automated.

## Reused from v1 (it was never the problem)
- `_mesh_repair.py` — trimesh repair, verbatim.
- `_detector.py` — RANSAC primitive detection (plane/cylinder/sphere/torus),
  face-adjacency growing, scale-adaptive thresholds. Kept as the segmentation
  front-end; pluggable so CGAL Efficient RANSAC can replace it later.
- OCCT idioms (surface builders, sewing, STEP write) — consolidated in `occ_util.py`.

## Known bottleneck carried into v2
The detector still over-fits (diff_case → 559 tiny cylinders, 746 regions for a
3.7k-face mesh). The coverage guarantee stops this from producing *holes*, but
merged topology will only be as clean as the segmentation. Improving segmentation
(region-growing before RANSAC, or CGAL) is the highest-leverage future work — the
gate now measures it honestly per region.

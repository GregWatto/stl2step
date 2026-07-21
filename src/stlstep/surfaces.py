"""Stage 2/3 helper — build one BRep face per region.

The unit of work is a region: a connected set of mesh triangles plus a fitted
surface. We extract the region's boundary as ordered vertex loops directly from
the mesh, then build a trimmed face on the fitted surface.

Watertightness strategy (before real intersection curves exist): a region's
boundary loop is built from the *same* mesh vertices its neighbours use, so
shared edges coincide exactly and `BRepBuilderAPI_Sewing` stitches them. Edges
are therefore polygonal for now — the merged solid is already one face per
region (huge reduction, editable surfaces); replacing polylines with analytic
`GeomAPI_IntSS` curves is the next refinement and does not change topology.

Every builder returns `None` on any difficulty so the caller can fall back to
faceting that one region — the pipeline never regresses to a crash.
"""

from __future__ import annotations

import numpy as np

from OCP.gp import gp_Pnt
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeFace,
)
from OCP.BRepCheck import BRepCheck_Analyzer


# ----------------------------------------------------------------------------
# Boundary loop extraction (surface-type agnostic).
# ----------------------------------------------------------------------------

def build_edge_face_map(mesh) -> dict:
    """frozenset({a, b}) -> list of face ids sharing that undirected edge."""
    F = np.asarray(mesh.faces)
    emap: dict = {}
    for fid, tri in enumerate(F):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            emap.setdefault(frozenset((u, v)), []).append(fid)
    return emap


def region_boundary_loops(mesh, face_ids, edge_face_map):
    """Ordered, oriented vertex loops bounding a region.

    A boundary edge is one whose neighbouring face is outside the region (or a
    mesh border). Directed as it appears in its triangle, boundary edges keep
    the region interior on their left → outer loops come out CCW and holes CW
    w.r.t. the region normal, which is exactly OCCT's convention.

    Returns (loops, ok). Each loop is a list of vertex indices (open, i.e. last
    != first). `ok` is False if the boundary can't be chained into simple loops
    (region pinch / non-manifold) — caller should fall back to faceting.
    """
    F = np.asarray(mesh.faces)
    fset = set(int(f) for f in face_ids)

    # Directed boundary edges: successor[a] = b.
    succ: dict[int, list[int]] = {}
    n_edges = 0
    for f in fset:
        a, b, c = (int(F[f][0]), int(F[f][1]), int(F[f][2]))
        for u, v in ((a, b), (b, c), (c, a)):
            others = [g for g in edge_face_map.get(frozenset((u, v)), [])
                      if g != f]
            inside = any(g in fset for g in others)
            if not inside:
                succ.setdefault(u, []).append(v)
                n_edges += 1

    if n_edges == 0:
        return [], False

    loops: list[list[int]] = []
    used = 0
    # Walk chains. A well-formed manifold region gives each boundary vertex one
    # outgoing edge; branches (len>1) signal a pinch -> bail.
    remaining = {k: list(v) for k, v in succ.items()}
    guard = n_edges + 5
    while any(remaining.values()):
        start = next(k for k, v in remaining.items() if v)
        loop = [start]
        cur = start
        steps = 0
        while True:
            steps += 1
            if steps > guard:
                return [], False
            outs = remaining.get(cur)
            if not outs:
                return [], False
            nxt = outs.pop()
            used += 1
            if nxt == start:
                break
            if nxt in loop:
                return [], False
            loop.append(nxt)
            cur = nxt
        if len(loop) >= 3:
            loops.append(loop)
    if used != n_edges:
        return [], False
    return loops, True


def _loop_signed_area(pts: np.ndarray, normal: np.ndarray) -> float:
    """Signed area of a planar-ish 3D polygon w.r.t. `normal` (Newell/shoelace)."""
    acc = np.zeros(3)
    n = len(pts)
    for i in range(n):
        acc += np.cross(pts[i], pts[(i + 1) % n])
    return 0.5 * float(np.dot(acc, normal))


# ----------------------------------------------------------------------------
# Planar face.
# ----------------------------------------------------------------------------

def _wire_from_loop(mesh, loop):
    V = np.asarray(mesh.vertices)
    poly = BRepBuilderAPI_MakePolygon()
    for vid in loop:
        p = V[vid]
        poly.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    poly.Close()
    if not poly.IsDone():
        return None
    return poly.Wire()


def planar_face(geom_plane, normal, mesh, loops):
    """One trimmed planar face from boundary loops. Largest-area loop is the
    outer boundary; the rest are holes. Returns TopoDS_Face or None."""
    V = np.asarray(mesh.vertices)
    if not loops:
        return None
    areas = [abs(_loop_signed_area(V[np.asarray(lp)], normal)) for lp in loops]
    order = np.argsort(areas)[::-1]  # outer first
    outer = loops[order[0]]
    outer_wire = _wire_from_loop(mesh, outer)
    if outer_wire is None:
        return None
    mk = BRepBuilderAPI_MakeFace(geom_plane, outer_wire, True)
    if not mk.IsDone():
        return None
    for idx in order[1:]:
        hole_wire = _wire_from_loop(mesh, loops[idx])
        if hole_wire is not None:
            mk.Add(hole_wire)
    if not mk.IsDone():
        return None
    face = mk.Face()
    if not BRepCheck_Analyzer(face).IsValid():
        return None
    return face

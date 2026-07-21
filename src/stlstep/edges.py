"""Step 3 — clean analytic edges via surface-surface intersection.

The `solid` method closes a watertight body, but every face is trimmed by a
polyline of hundreds of tiny mesh-boundary edges (a cylinder's rim is a polygon,
not a circle). That bloats files ~100-1000x and isn't dimensionable.

Here we rebuild each analytic region's boundary from real geometry: walk the
region's mesh-boundary loop, group consecutive edges by which neighbour region
they border, and replace each contiguous A|B interface with ONE analytic edge on
the true intersection curve `GeomAPI_IntSS(surf_A, surf_B)` (a circle, line, or
ellipse). Interface edges are shared between the two faces (built once, reversed
for the neighbour) so the solid stays watertight, and node vertices at interface
transitions are shared too.

Everything degrades gracefully: a boundary against a freeform/open region, or any
interface where the clean construction fails, falls back to polyline edges from
the same mesh vertices — so sewing still closes and nothing regresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from OCP.gp import gp_Pnt
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
)
from OCP.GeomAPI import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.TopoDS import TopoDS
from OCP.BRepCheck import BRepCheck_Analyzer

from . import occ_util as ou
from . import surfaces as su


@dataclass
class Context:
    mesh: object
    segmentation: object
    region_of_face: dict
    edge_face_map: dict
    surfaces: dict = field(default_factory=dict)   # region_id -> Geom_Surface
    vertices: dict = field(default_factory=dict)   # vid -> TopoDS_Vertex
    iedges: dict = field(default_factory=dict)     # key -> (edge, v0, v1)
    stats: dict = field(default_factory=lambda: {"analytic_edges": 0,
                                                 "polyline_edges": 0,
                                                 "intss_fail": 0})


def build_context(mesh, segmentation) -> Context:
    ctx = Context(
        mesh=mesh,
        segmentation=segmentation,
        region_of_face=segmentation.region_of_face(),
        edge_face_map=su.build_edge_face_map(mesh),
    )
    for r in segmentation.regions:
        ctx.surfaces[r.id] = (ou.surface_from_region(r.kind, r.params)
                              if r.is_analytic else None)
    return ctx


def _vertex(ctx, vid):
    v = ctx.vertices.get(vid)
    if v is None:
        p = ctx.mesh.vertices[vid]
        v = BRepBuilderAPI_MakeVertex(gp_Pnt(float(p[0]), float(p[1]),
                                             float(p[2]))).Vertex()
        ctx.vertices[vid] = v
    return v


def _neighbor_region(ctx, region_id, u, v):
    """Region on the far side of mesh edge (u, v), or None if open/freeform."""
    faces = ctx.edge_face_map.get(frozenset((u, v)), [])
    for f in faces:
        rid = ctx.region_of_face.get(int(f))
        if rid is not None and rid != region_id:
            return rid
    return None


def _loop_segments(ctx, region_id, loop):
    """Split an ordered boundary loop into (neighbor_region, [vids]) segments —
    maximal runs of edges bordering the same neighbour. Segment vids include
    both endpoints; consecutive segments share a node vertex."""
    n = len(loop)
    edges = [(loop[i], loop[(i + 1) % n]) for i in range(n)]
    nbrs = [_neighbor_region(ctx, region_id, u, v) for u, v in edges]
    # rotate so a neighbour change is at index 0 (avoids splitting a run across
    # the wrap). If all edges share one neighbour, start anywhere.
    start = 0
    for i in range(n):
        if nbrs[i] != nbrs[i - 1]:
            start = i
            break
    segs = []
    cur_nbr = None
    cur = []
    for k in range(n):
        i = (start + k) % n
        u, v = edges[i]
        if nbrs[i] != cur_nbr and cur:
            segs.append((cur_nbr, cur))
            cur = [u]
        if not cur:
            cur = [u]
        cur.append(v)
        cur_nbr = nbrs[i]
    if cur:
        segs.append((cur_nbr, cur))
    return segs


def _intersection_curve(surfA, surfB, near_pts, tol=1e-4):
    """Best intersection curve of two surfaces near the mesh boundary points."""
    try:
        ints = GeomAPI_IntSS(surfA, surfB, tol)
    except Exception:
        return None
    if not ints.IsDone() or ints.NbLines() == 0:
        return None
    best = None
    for i in range(1, ints.NbLines() + 1):
        curve = ints.Line(i)
        d = 0.0
        for p in near_pts:
            proj = GeomAPI_ProjectPointOnCurve(
                gp_Pnt(float(p[0]), float(p[1]), float(p[2])), curve)
            if proj.NbPoints() > 0:
                d += proj.LowerDistance()
            else:
                d += 1e6
        d /= max(len(near_pts), 1)
        if best is None or d < best[1]:
            best = (curve, d)
    if best is None:
        return None
    curve, d = best
    return curve if d < 1.0 else None  # reject if no curve is near the boundary


def _interface_edge(ctx, region_id, nbr, seg_vids):
    """One analytic edge for an A|B interface segment, shared + cached. Falls
    back to a polyline of the segment's mesh edges when a clean edge can't be
    built. Returns a list of oriented TopoDS_Edges for this face's wire."""
    v_start, v_end = seg_vids[0], seg_vids[-1]
    surfA = ctx.surfaces.get(region_id)
    surfB = ctx.surfaces.get(nbr) if nbr is not None else None

    if surfA is not None and surfB is not None and v_start != v_end:
        key = (min(region_id, nbr), max(region_id, nbr),
               min(v_start, v_end), max(v_start, v_end))
        cached = ctx.iedges.get(key)
        if cached is not None:
            edge, cs, ce = cached
            return [edge if (v_start, v_end) == (cs, ce)
                    else TopoDS.Edge_s(edge.Reversed())]
        near = ctx.mesh.vertices[np.asarray(seg_vids)]
        curve = _intersection_curve(surfA, surfB, near)
        if curve is not None:
            try:
                mk = BRepBuilderAPI_MakeEdge(curve, _vertex(ctx, v_start),
                                             _vertex(ctx, v_end))
                if mk.IsDone():
                    edge = mk.Edge()
                    ctx.iedges[key] = (edge, v_start, v_end)
                    ctx.stats["analytic_edges"] += 1
                    return [edge]
            except Exception:
                pass
        ctx.stats["intss_fail"] += 1

    # fallback: polyline edges from the segment's mesh vertices
    return _polyline_edges(ctx, seg_vids)


def _polyline_edges(ctx, seg_vids):
    edges = []
    for a, b in zip(seg_vids[:-1], seg_vids[1:]):
        if a == b:
            continue
        mk = BRepBuilderAPI_MakeEdge(_vertex(ctx, a), _vertex(ctx, b))
        if mk.IsDone():
            edges.append(mk.Edge())
    ctx.stats["polyline_edges"] += len(edges)
    return edges


def clean_face(ctx, region):
    """Build one clean analytic face for `region`. Curved surfaces use the
    uv-bounded constructor (clean circle edges + automatic periodic seam);
    planes use a wire of intersection-curve edges (no seam). Returns a
    TopoDS_Face or None (caller falls back to the polyline merge)."""
    surf = ctx.surfaces.get(region.id)
    if surf is None:
        return None
    if region.kind in ("cylinder", "cone", "sphere", "torus"):
        return _uv_bounded_face(ctx, region, surf)
    if region.kind == "plane":
        return _plane_wire_face(ctx, region, surf)
    return None


def _uv_bounded_face(ctx, region, surf):
    """Clean curved face over the region's uv extent. OCCT adds the seam for a
    full-period surface, so a full cylinder comes out as two circle edges + seam
    rather than a polyline."""
    sa = ShapeAnalysis_Surface(surf)
    F = np.asarray(ctx.mesh.faces)
    V = np.asarray(ctx.mesh.vertices)
    verts = np.unique(F[np.asarray(region.face_ids)].reshape(-1))
    us = []; vs = []
    for p in V[verts]:
        q = sa.ValueOfUV(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), 1e-2)
        us.append(q.X()); vs.append(q.Y())
    us = np.asarray(us); vs = np.asarray(vs)
    up, vp = ou.surface_periods(region.kind)
    umin, umax = _uv_range(us, up)
    vmin, vmax = _uv_range(vs, vp)
    if umax - umin < 1e-6 or vmax - vmin < 1e-9:
        return None
    mk = BRepBuilderAPI_MakeFace(surf, umin, umax, vmin, vmax, 1e-6)
    if not mk.IsDone():
        return None
    face = mk.Face()
    return face if BRepCheck_Analyzer(face).IsValid() else None


def _uv_range(vals, period):
    """Parametric range for a face bound.

    For a periodic axis the naive [min, max] is wrong when the arc straddles the
    u=0/2pi seam (min~0, max~2pi looks like a full turn). Find the largest gap in
    the circularly-sorted values: the arc is its complement. A tiny largest gap
    means the data covers the full period; a gap at the wrap means the arc
    doesn't cross the seam; otherwise the arc crosses the seam and is unwrapped."""
    lo, hi = float(vals.min()), float(vals.max())
    if not period:
        return lo, hi
    sv = np.sort(vals)
    gaps = np.append(np.diff(sv), (sv[0] + period) - sv[-1])
    imax = int(gaps.argmax())
    gmax = float(gaps[imax])
    if gmax < 0.1 * period:            # covers ~all of the circle -> full turn
        return 0.0, period
    if imax == len(sv) - 1:            # largest gap is the wrap -> no seam cross
        return lo, hi
    return float(sv[imax + 1]), float(sv[imax] + period)  # arc crosses the seam


def _plane_wire_face(ctx, region, surf):
    loops, ok = su.region_boundary_loops(ctx.mesh, region.face_ids,
                                         ctx.edge_face_map)
    if not ok or not loops:
        return None
    wires = []
    for loop in loops:
        segs = _loop_segments(ctx, region.id, loop)
        edges = []
        # a loop bordering a single neighbour is a closed curve (full circle)
        if len(segs) == 1 and segs[0][0] is not None:
            e = _closed_edge(ctx, region.id, segs[0][0], loop)
            if e is None:
                edges = _polyline_edges(ctx, loop + [loop[0]])
            else:
                edges = [e]
        else:
            for nbr, seg_vids in segs:
                edges.extend(_interface_edge(ctx, region.id, nbr, seg_vids))
        if not edges:
            return None
        mkw = BRepBuilderAPI_MakeWire()
        for e in edges:
            mkw.Add(e)
        if not mkw.IsDone():
            return None
        wires.append((mkw.Wire(), loop))

    V = np.asarray(ctx.mesh.vertices)
    areas = []
    for _w, loop in wires:
        pts = V[np.asarray(loop)]
        c = pts.mean(0)
        areas.append(np.linalg.norm(
            np.cross(pts - c, np.roll(pts - c, -1, 0)).sum(0)))
    order = np.argsort(areas)[::-1]
    mk = BRepBuilderAPI_MakeFace(surf, wires[order[0]][0], True)
    if not mk.IsDone():
        return None
    for idx in order[1:]:
        mk.Add(wires[idx][0])
    if not mk.IsDone():
        return None
    face = mk.Face()
    return face if BRepCheck_Analyzer(face).IsValid() else None


def _closed_edge(ctx, region_id, nbr, loop):
    """A single closed analytic edge (full circle/line) for a loop bordering one
    neighbour. Not shared — geometrically identical to the neighbour's own edge,
    so tolerant sewing merges them."""
    surfA = ctx.surfaces.get(region_id)
    surfB = ctx.surfaces.get(nbr)
    if surfA is None or surfB is None:
        return None
    near = ctx.mesh.vertices[np.asarray(loop)]
    curve = _intersection_curve(surfA, surfB, near)
    if curve is None:
        ctx.stats["intss_fail"] += 1
        return None
    try:
        mk = BRepBuilderAPI_MakeEdge(curve)
        if mk.IsDone():
            ctx.stats["analytic_edges"] += 1
            return mk.Edge()
    except Exception:
        pass
    return None

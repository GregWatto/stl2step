"""brep_stitch.py

Turn a primitive_graph.json + repaired mesh into a STEP file by building
one trimmed BRep face per detected primitive. The trim is the u/v
bounding box of the assigned face vertices projected onto the primitive
surface — cheap, honest, first-cut.

Unassigned faces are dropped from the STEP output. The gap between the
input mesh area and the STEP-covered area IS the residual that Step 5
(feature recognition) will need to fill.

CLI:
    .venv/bin/python scripts/pipeline/brep_stitch.py \\
        <in.stl> <out_dir> [--samples 10000]

Library usage:
    from pipeline.brep_stitch import stitch_from_graph
    stitch_from_graph(mesh, graph, out_step_path)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import trimesh

# OCP is Open Cascade Python bindings, shipped by cadquery-ocp.
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCP.BRepLib import BRepLib
from OCP.GCE2d import GCE2d_MakeSegment
from OCP.gp import (
    gp_Ax3,
    gp_Dir,
    gp_Pnt,
    gp_Pnt2d,
)
from OCP.Geom import (
    Geom_CylindricalSurface,
    Geom_Plane,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
)
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.TopoDS import TopoDS_Compound, TopoDS_Face, TopoDS_Shape

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.mesh_repair import repair_mesh  # noqa: E402
from pipeline.primitive_detect import (  # noqa: E402
    _primitive_normal_at,
    _primitive_residuals_at,
    compute_thresholds,
    detect_primitives,
)


# ----------------------------------------------------------------------------
# Building OCCT surfaces from our primitive dicts.
# ----------------------------------------------------------------------------

def _ax3_from_axis(center: np.ndarray, axis: np.ndarray) -> gp_Ax3:
    """Build a gp_Ax3 (position + main direction + x-direction).
    The x-direction is arbitrary but must be perpendicular to the axis.
    """
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    # Pick an x-direction not parallel to axis.
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    xdir = tmp - (tmp @ axis) * axis
    xdir = xdir / max(np.linalg.norm(xdir), 1e-12)
    return gp_Ax3(
        gp_Pnt(float(center[0]), float(center[1]), float(center[2])),
        gp_Dir(float(axis[0]), float(axis[1]), float(axis[2])),
        gp_Dir(float(xdir[0]), float(xdir[1]), float(xdir[2])),
    )


def _surface_from_primitive(prim: dict):
    """Build the appropriate Geom_* surface. Return None if unsupported."""
    p = prim["params"]
    t = prim["type"]
    if t == "plane":
        n = np.asarray(p["normal"], dtype=float)
        # A point on the plane: -offset * normal (for a normalized normal).
        pt = -float(p["offset_mm"]) * n
        ax3 = _ax3_from_axis(pt, n)
        return Geom_Plane(ax3)
    if t == "cylinder":
        c = np.asarray(p["center_mm"], dtype=float)
        a = np.asarray(p["axis"], dtype=float)
        return Geom_CylindricalSurface(_ax3_from_axis(c, a),
                                       float(p["radius_mm"]))
    if t == "sphere":
        c = np.asarray(p["center_mm"], dtype=float)
        # Any axis works for a sphere.
        ax3 = _ax3_from_axis(c, np.array([0.0, 0.0, 1.0]))
        return Geom_SphericalSurface(ax3, float(p["radius_mm"]))
    if t == "torus":
        c = np.asarray(p["center_mm"], dtype=float)
        a = np.asarray(p["axis"], dtype=float)
        return Geom_ToroidalSurface(_ax3_from_axis(c, a),
                                    float(p["major_radius_mm"]),
                                    float(p["minor_radius_mm"]))
    return None


# ----------------------------------------------------------------------------
# u/v bounding + closed-parameter handling.
# ----------------------------------------------------------------------------

def _project_uv(surface, points_3d: np.ndarray) -> np.ndarray:
    """Return (N, 2) uv-coordinates for each 3D point via
    ShapeAnalysis_Surface.ValueOfUV. Points that hang far from the
    surface are silently included — we're only using them for a bbox,
    so a stray point won't crash things."""
    sa = ShapeAnalysis_Surface(surface)
    tol = 1e-3
    out = np.empty((len(points_3d), 2), dtype=float)
    for i, p in enumerate(points_3d):
        uv = sa.ValueOfUV(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), tol)
        out[i, 0] = uv.X()
        out[i, 1] = uv.Y()
    return out


def _uv_bounds(uv: np.ndarray, is_periodic_u: bool,
               is_periodic_v: bool, period_u: float,
               period_v: float) -> tuple[float, float, float, float]:
    """Compute u/v bounding box, being smart about periodic coordinates.

    For periodic parameters (θ on a cylinder is 2π-periodic), a range of
    [5°, 355°] almost certainly means the samples wrap around 0, not
    that they span the whole cylinder. Detect this via the size of the
    biggest angular gap: if that gap is larger than what's covered,
    unwrap by shifting the small side up by one period.
    """
    u = uv[:, 0]
    v = uv[:, 1]
    if is_periodic_u and len(u) > 1:
        u_min, u_max = _periodic_bounds(u, period_u)
    else:
        u_min, u_max = float(u.min()), float(u.max())
    if is_periodic_v and len(v) > 1:
        v_min, v_max = _periodic_bounds(v, period_v)
    else:
        v_min, v_max = float(v.min()), float(v.max())
    return u_min, u_max, v_min, v_max


def _periodic_bounds(x: np.ndarray, period: float) -> tuple[float, float]:
    """Find the tightest period-aware interval covering `x` values.
    Sort by (x mod period), then look for the biggest gap; the arc
    outside that gap is the tightest wrap-around cover.
    """
    xw = np.mod(x, period)
    xs = np.sort(xw)
    gaps = np.diff(xs)
    # Also consider the wrap-around gap between last and first + period.
    wrap_gap = xs[0] + period - xs[-1]
    all_gaps = np.concatenate([gaps, [wrap_gap]])
    max_gap_idx = int(np.argmax(all_gaps))
    if max_gap_idx == len(xs) - 1:
        # Biggest gap is the wrap. The tight interval is [xs[0], xs[-1]].
        return float(xs[0]), float(xs[-1])
    # Biggest gap is internal — unwrap: use xs[max_gap_idx+1] up to
    # xs[max_gap_idx] + period as the covered arc.
    lo = float(xs[max_gap_idx + 1])
    hi = float(xs[max_gap_idx]) + period
    return lo, hi


def _make_face_trimmed(surface, u_min, u_max, v_min, v_max) -> TopoDS_Face | None:
    """Trim a Geom_* surface to (u_min..u_max, v_min..v_max) and return a
    BRep face. Returns None if OCCT rejects the trim."""
    try:
        maker = BRepBuilderAPI_MakeFace(surface, u_min, u_max, v_min, v_max,
                                        1e-4)
        if not maker.IsDone():
            return None
        return maker.Face()
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# Per-triangle face construction on smooth primitive surfaces.
# ----------------------------------------------------------------------------

def _unwrap_uv(uv_triangle: np.ndarray, period_u: float,
               period_v: float, is_u_periodic: bool,
               is_v_periodic: bool) -> np.ndarray:
    """Shift periodic parameters so the 3 vertices of a triangle sit in
    the same "sheet". A cylinder returns θ in [0, 2π]; a triangle
    straddling θ=0 comes back as (0.05, 6.20, 0.10) which if drawn
    naively becomes a wire going the long way around. Detect that: if
    any pair differs by more than half the period, shift the smaller
    value up by one period."""
    uv = uv_triangle.copy()
    if is_u_periodic and period_u > 0:
        u_max = uv[:, 0].max()
        for i in range(3):
            if u_max - uv[i, 0] > period_u * 0.5:
                uv[i, 0] += period_u
    if is_v_periodic and period_v > 0:
        v_max = uv[:, 1].max()
        for i in range(3):
            if v_max - uv[i, 1] > period_v * 0.5:
                uv[i, 1] += period_v
    return uv


def _make_face_on_plane_from_triangle(surface, v0, v1, v2) -> TopoDS_Face | None:
    """Build a triangular face on a Geom_Plane using a 3D polygon wire.
    Planes accept 3D wires directly — OCCT builds PCurves automatically.
    """
    try:
        poly = BRepBuilderAPI_MakePolygon(
            gp_Pnt(float(v0[0]), float(v0[1]), float(v0[2])),
            gp_Pnt(float(v1[0]), float(v1[1]), float(v1[2])),
            gp_Pnt(float(v2[0]), float(v2[1]), float(v2[2])),
            True,  # close
        )
        if not poly.IsDone():
            return None
        wire = poly.Wire()
        maker = BRepBuilderAPI_MakeFace(surface, wire, True)
        if not maker.IsDone():
            return None
        face = maker.Face()
        BRepLib.BuildCurves3d_s(face)
        return face
    except Exception:  # noqa: BLE001
        return None


def _make_face_on_curved_from_triangle(surface, uv: np.ndarray
                                       ) -> TopoDS_Face | None:
    """Build a triangular face on a curved Geom_* surface, given the
    2D u/v coordinates of the 3 triangle vertices. Uses 2D segments in
    parametric space (via GCE2d_MakeSegment) with the surface handle."""
    try:
        p0 = gp_Pnt2d(float(uv[0, 0]), float(uv[0, 1]))
        p1 = gp_Pnt2d(float(uv[1, 0]), float(uv[1, 1]))
        p2 = gp_Pnt2d(float(uv[2, 0]), float(uv[2, 1]))
        seg_01 = GCE2d_MakeSegment(p0, p1).Value()
        seg_12 = GCE2d_MakeSegment(p1, p2).Value()
        seg_20 = GCE2d_MakeSegment(p2, p0).Value()
        e01 = BRepBuilderAPI_MakeEdge(seg_01, surface).Edge()
        e12 = BRepBuilderAPI_MakeEdge(seg_12, surface).Edge()
        e20 = BRepBuilderAPI_MakeEdge(seg_20, surface).Edge()
        wire = BRepBuilderAPI_MakeWire(e01, e12, e20).Wire()
        maker = BRepBuilderAPI_MakeFace(surface, wire, True)
        if not maker.IsDone():
            return None
        face = maker.Face()
        # Build the 3D curves for each edge from its 2D PCurve; without
        # this the STEP mesher can't tessellate the face.
        BRepLib.BuildCurves3d_s(face)
        return face
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# Main stitching entry point.
# ----------------------------------------------------------------------------

def _periodicity_for(prim_type: str) -> tuple[bool, bool, float, float]:
    """Return (u_periodic, v_periodic, period_u, period_v) for a primitive."""
    twopi = 2 * math.pi
    if prim_type == "cylinder":
        return True, False, twopi, 0.0  # θ periodic
    if prim_type == "sphere":
        return True, False, twopi, 0.0  # longitude periodic; latitude is bounded
    if prim_type == "torus":
        return True, True, twopi, twopi  # both angles
    return False, False, 0.0, 0.0  # plane


# ----------------------------------------------------------------------------
# Merged-face construction: one BRep face per (primitive × connected
# component of assigned mesh triangles). Boundary tracing is done on the
# mesh face-adjacency graph, then projected to the primitive's u/v space.
# ----------------------------------------------------------------------------

# Growing thresholds for the geometric per-face reassignment. The
# factor is intentionally tight — a torus is a "fat" surface (any point
# within its minor-radius tube is close to it), so at 5x the RANSAC
# threshold cylinder-wall faces near a torus get pulled onto the torus
# and inflate it into a lumpy blob. 2x + tight normal alignment keeps
# faces on the primitive whose surface they actually lie on.
_REASSIGN_DIST_FACTOR = 2.0
_REASSIGN_MIN_NORMAL_DOT = 0.9


def reassign_faces_by_geometry(mesh: trimesh.Trimesh,
                               primitives: list[dict],
                               thresholds: dict) -> dict[int, int]:
    """For each mesh triangle, find the primitive whose surface is
    closest AND whose normal matches. Returns {face_id: prim_idx}.

    Replaces sample-vote face assignment (which speckles at
    primitive-primitive boundaries). Faces that don't fit any primitive
    are dropped from the result — those become the "unassigned" set
    that Step 4c or Step 5 can pick up.
    """
    if not primitives:
        return {}

    centroids = np.asarray(mesh.vertices[mesh.faces].mean(axis=1), dtype=float)
    face_normals = np.asarray(mesh.face_normals, dtype=float)
    n_faces = len(centroids)

    diag = float(thresholds["bbox_diagonal_mm"])
    dist_floor = diag * 5e-4
    type_thresh = {
        "plane": max(thresholds["plane_mm"] * _REASSIGN_DIST_FACTOR, dist_floor),
        "cylinder": max(thresholds["cylinder_mm"] * _REASSIGN_DIST_FACTOR,
                        dist_floor),
        "torus": max(thresholds["torus_mm"] * _REASSIGN_DIST_FACTOR,
                     dist_floor),
        "sphere": max(thresholds["sphere_mm"] * _REASSIGN_DIST_FACTOR,
                      dist_floor),
    }

    # Per-primitive residual + normal-alignment matrices, then argmin.
    n_prims = len(primitives)
    dist_matrix = np.full((n_prims, n_faces), np.inf)
    valid_mask = np.zeros((n_prims, n_faces), dtype=bool)
    for p_i, prim in enumerate(primitives):
        d = _primitive_residuals_at(prim, centroids)
        pn = _primitive_normal_at(prim, centroids)
        dot = np.abs(np.sum(face_normals * pn, axis=1))
        thr = type_thresh.get(prim["type"], np.inf)
        ok = (d <= thr) & (dot >= _REASSIGN_MIN_NORMAL_DOT)
        dist_matrix[p_i] = np.where(ok, d, np.inf)
        valid_mask[p_i] = ok

    # For each face, the winning primitive is argmin of the residual
    # among the ones that passed the normal / threshold gate.
    best_prim = np.argmin(dist_matrix, axis=0)
    any_valid = valid_mask.any(axis=0)

    assignment: dict[int, int] = {}
    for f_i in range(n_faces):
        if any_valid[f_i]:
            assignment[f_i] = int(best_prim[f_i])
    return assignment


def _connected_components(face_ids: list[int],
                          mesh: trimesh.Trimesh) -> list[list[int]]:
    """Split face_ids into connected components via mesh.face_adjacency
    (restricted to the assigned set)."""
    assigned = set(int(f) for f in face_ids)
    if not assigned:
        return []
    parent = {f: f for f in assigned}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in np.asarray(mesh.face_adjacency, dtype=np.int64):
        a, b = int(a), int(b)
        if a in assigned and b in assigned:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for f in assigned:
        groups[find(f)].append(f)
    return list(groups.values())


def _boundary_directed_edges(comp_face_ids: list[int],
                             mesh: trimesh.Trimesh
                             ) -> list[tuple[int, int]]:
    """Return oriented boundary edges (v0 → v1) such that the component
    lies on the left of each edge in the mesh winding.

    A mesh half-edge is on the component's boundary iff its opposite
    half-edge is either absent (mesh-open boundary) or belongs to a
    face outside the component."""
    comp = set(int(f) for f in comp_face_ids)
    faces = np.asarray(mesh.faces)

    # For fast opposite-face lookup: map each ordered half-edge (v0, v1)
    # to its owning face.
    half_edge_to_face: dict[tuple[int, int], int] = {}
    for face_id in range(len(faces)):
        f = faces[face_id]
        for i in range(3):
            he = (int(f[i]), int(f[(i + 1) % 3]))
            half_edge_to_face[he] = face_id

    boundary: list[tuple[int, int]] = []
    for face_id in comp:
        f = faces[face_id]
        for i in range(3):
            v0 = int(f[i])
            v1 = int(f[(i + 1) % 3])
            # Opposite half-edge (v1 → v0) belongs to the neighbor face.
            opp_face = half_edge_to_face.get((v1, v0))
            if opp_face is None or opp_face not in comp:
                boundary.append((v0, v1))
    return boundary


def _chain_loops(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Chain directed edges into closed vertex loops."""
    if not edges:
        return []
    from collections import defaultdict
    out_edges: dict[int, list[int]] = defaultdict(list)
    for v0, v1 in edges:
        out_edges[v0].append(v1)

    loops: list[list[int]] = []
    for v_start, _ in edges:
        if not out_edges[v_start]:
            continue
        loop = [v_start]
        cur = v_start
        while out_edges[cur]:
            nxt = out_edges[cur].pop()
            loop.append(nxt)
            cur = nxt
            if cur == v_start:
                break
        # Only keep closed loops with at least 3 vertices.
        if loop[0] == loop[-1] and len(loop) >= 4:
            loops.append(loop[:-1])
    return loops


def _project_loop_to_uv(loop_vertex_ids: list[int],
                        mesh: trimesh.Trimesh,
                        surface,
                        prim_type: str,
                        tol: float = 1e-3) -> np.ndarray:
    """Project a boundary loop's 3D vertices to (u, v) on the surface,
    unwrapping periodic parameters so successive points stay in the
    same "sheet"."""
    verts = np.asarray(mesh.vertices, dtype=float)
    sa = ShapeAnalysis_Surface(surface)
    u_per, v_per, period_u, period_v = _periodicity_for(prim_type)

    uv = np.empty((len(loop_vertex_ids), 2), dtype=float)
    for i, vid in enumerate(loop_vertex_ids):
        v = verts[int(vid)]
        p = sa.ValueOfUV(gp_Pnt(float(v[0]), float(v[1]), float(v[2])), tol)
        uv[i] = (p.X(), p.Y())

    # Unwrap: for each successive point, shift by ±period if that keeps
    # it closer to the previous point.
    if u_per and period_u > 0:
        for i in range(1, len(uv)):
            while uv[i, 0] - uv[i - 1, 0] > period_u * 0.5:
                uv[i, 0] -= period_u
            while uv[i, 0] - uv[i - 1, 0] < -period_u * 0.5:
                uv[i, 0] += period_u
    if v_per and period_v > 0:
        for i in range(1, len(uv)):
            while uv[i, 1] - uv[i - 1, 1] > period_v * 0.5:
                uv[i, 1] -= period_v
            while uv[i, 1] - uv[i - 1, 1] < -period_v * 0.5:
                uv[i, 1] += period_v
    return uv


def _signed_area_uv(uv_loop: np.ndarray) -> float:
    """Signed area of a 2D polygon (shoelace). Positive = CCW."""
    x = uv_loop[:, 0]
    y = uv_loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _make_wire_from_uv_loop(surface, uv_loop: np.ndarray):
    """Build a closed BRep wire from a u/v polygon on the given surface."""
    n = len(uv_loop)
    if n < 3:
        return None
    edges = []
    for i in range(n):
        p0 = gp_Pnt2d(float(uv_loop[i, 0]), float(uv_loop[i, 1]))
        p1 = gp_Pnt2d(float(uv_loop[(i + 1) % n, 0]),
                      float(uv_loop[(i + 1) % n, 1]))
        try:
            seg = GCE2d_MakeSegment(p0, p1).Value()
            edge = BRepBuilderAPI_MakeEdge(seg, surface).Edge()
        except Exception:  # noqa: BLE001
            return None
        edges.append(edge)
    try:
        wm = BRepBuilderAPI_MakeWire()
        for e in edges:
            wm.Add(e)
        if not wm.IsDone():
            return None
        return wm.Wire()
    except Exception:  # noqa: BLE001
        return None


def _make_wire_from_3d_loop(loop_pts: np.ndarray):
    """Build a closed BRep wire from a 3D polygon (used for planes)."""
    n = len(loop_pts)
    if n < 3:
        return None
    try:
        poly = BRepBuilderAPI_MakePolygon()
        for p in loop_pts:
            poly.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        poly.Close()
        if not poly.IsDone():
            return None
        return poly.Wire()
    except Exception:  # noqa: BLE001
        return None


def _make_merged_face(surface, prim_type: str, loops: list[np.ndarray],
                      loops_3d: list[np.ndarray] | None,
                      min_inner_area_ratio: float = 0.02
                      ) -> TopoDS_Face | None:
    """Build a single BRep face with the largest-area loop as the outer
    boundary and the rest as inner (hole) wires.

    Inner loops smaller than `min_inner_area_ratio` of the outer's area
    are dropped — they're usually rib-tops or other small features
    poking through the primitive that we haven't detected, and drawing
    them as slits makes the STEP look terrible. Treating them as noise
    gives a clean primitive-region face.

    For planes: use 3D wires directly (loops_3d).
    For curved surfaces: use 2D u/v wires (loops).
    """
    if not loops:
        return None

    # Rank by |area|. Largest is outer, others are holes.
    order = sorted(range(len(loops)), key=lambda i: -abs(_signed_area_uv(loops[i])))
    outer_idx = order[0]
    outer_abs_area = abs(_signed_area_uv(loops[outer_idx]))
    min_inner_abs_area = outer_abs_area * min_inner_area_ratio
    inner_idxs = [i for i in order[1:]
                  if abs(_signed_area_uv(loops[i])) >= min_inner_abs_area]

    outer_uv = loops[outer_idx]
    outer_signed = _signed_area_uv(outer_uv)
    # Outer must be CCW (positive area). Inner must be CW (negative).
    if outer_signed < 0:
        outer_uv = outer_uv[::-1]
    if prim_type == "plane" and loops_3d is not None:
        outer_wire = _make_wire_from_3d_loop(
            loops_3d[outer_idx] if outer_signed >= 0
            else loops_3d[outer_idx][::-1]
        )
    else:
        outer_wire = _make_wire_from_uv_loop(surface, outer_uv)
    if outer_wire is None:
        return None

    try:
        mf = BRepBuilderAPI_MakeFace(surface, outer_wire, True)
        for idx in inner_idxs:
            inner_uv = loops[idx]
            inner_signed = _signed_area_uv(inner_uv)
            if inner_signed > 0:
                inner_uv = inner_uv[::-1]
            if prim_type == "plane" and loops_3d is not None:
                inner_wire = _make_wire_from_3d_loop(
                    loops_3d[idx] if inner_signed <= 0
                    else loops_3d[idx][::-1]
                )
            else:
                inner_wire = _make_wire_from_uv_loop(surface, inner_uv)
            if inner_wire is not None:
                mf.Add(inner_wire)
        if not mf.IsDone():
            return None
        face = mf.Face()
        BRepLib.BuildCurves3d_s(face)
        return face
    except Exception:  # noqa: BLE001
        return None


def _face_from_boundary(surface, prim_type: str, comp: list[int],
                        mesh: trimesh.Trimesh) -> TopoDS_Face | None:
    """Boundary-tracing path for planes: build a single face bounded by
    the traced outer loop, with any smaller loops as inner holes."""
    edges = _boundary_directed_edges(comp, mesh)
    loops_vid = _chain_loops(edges)
    if not loops_vid:
        return None
    loops_uv = [_project_loop_to_uv(l, mesh, surface, prim_type)
                for l in loops_vid]
    loops_3d = [np.asarray(mesh.vertices[l], dtype=float) for l in loops_vid]
    return _make_merged_face(surface, prim_type, loops_uv, loops_3d)


def _face_from_uv_bbox(surface, prim_type: str, comp: list[int],
                       mesh: trimesh.Trimesh) -> TopoDS_Face | None:
    """u/v bounding-box trim for curved surfaces. Correctly handles
    arc-strips (cylinder walls, fillet bands on tori) that would break
    the boundary-tracing outer+hole logic.

    Uses the u/v projections of the vertices ON the boundary of the
    component so we don't over-trim into interior features."""
    # Vertex set of the component (all vertices of assigned triangles).
    face_ids = np.asarray(comp, dtype=np.int64)
    vid_set = np.unique(mesh.faces[face_ids].reshape(-1))
    pts = np.asarray(mesh.vertices[vid_set], dtype=float)

    sa = ShapeAnalysis_Surface(surface)
    tol = 1e-3
    uv = np.empty((len(pts), 2), dtype=float)
    for i, p in enumerate(pts):
        q = sa.ValueOfUV(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), tol)
        uv[i] = (q.X(), q.Y())

    u_per, v_per, period_u, period_v = _periodicity_for(prim_type)
    u_min, u_max = _periodic_bounds(uv[:, 0], period_u) if u_per else (
        float(uv[:, 0].min()), float(uv[:, 0].max()))
    v_min, v_max = _periodic_bounds(uv[:, 1], period_v) if v_per else (
        float(uv[:, 1].min()), float(uv[:, 1].max()))

    # A tiny pad avoids zero-area faces for single-strip patches.
    u_span = max(u_max - u_min, 1e-3)
    v_span = max(v_max - v_min, 1e-3)
    u_min -= u_span * 0.005
    u_max += u_span * 0.005
    v_min -= v_span * 0.005
    v_max += v_span * 0.005

    return _make_face_trimmed(surface, u_min, u_max, v_min, v_max)


def stitch_from_graph_merged(mesh: trimesh.Trimesh, graph: dict,
                             step_path: str, sew: bool = True,
                             min_component_faces: int = 5,
                             reassign: bool = True) -> dict:
    """Build ONE BRep face per (primitive × connected component of
    assigned mesh triangles), trimmed to the traced boundary. Sew and
    export as STEP.

    This is what "STL → clean STEP" wants: a plane gets one large
    rectangular-ish face, a cylinder wall gets one arc-strip face, etc.

    `reassign=True` (default): override the sample-vote face_ids from
    the primitive graph with per-face geometric fit (each triangle goes
    to the primitive whose surface is nearest AND whose normal
    matches). Without this, sample vote leaves speckle that fragments
    connected regions and produces spurious boundary loops.

    Components smaller than `min_component_faces` are still skipped.
    """
    stats = Counter()
    faces_written = 0

    if sew:
        sewing = BRepBuilderAPI_Sewing(1e-3, True, True, True, False)
    else:
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)

    prims_list = graph.get("primitives", [])

    if reassign and prims_list:
        thresholds = graph.get("thresholds") or compute_thresholds(mesh)
        new_assignment = reassign_faces_by_geometry(mesh, prims_list, thresholds)
        prim_to_faces: dict[int, list[int]] = {i: [] for i in range(len(prims_list))}
        for face_id, prim_i in new_assignment.items():
            prim_to_faces[prim_i].append(face_id)
    else:
        prim_to_faces = {i: list(p.get("face_ids", []))
                         for i, p in enumerate(prims_list)}

    for p_i, prim in enumerate(prims_list):
        surface = _surface_from_primitive(prim)
        if surface is None:
            stats["unsupported"] += 1
            continue

        face_ids = prim_to_faces.get(p_i, [])
        if not face_ids:
            continue
        prim_type = prim["type"]

        # Split assigned faces into connected components; drop speckle.
        components = _connected_components(face_ids, mesh)
        components = [c for c in components if len(c) >= min_component_faces]
        if not components:
            stats[f"all_speckle_{prim_type}"] += 1
            continue
        for comp in components:
            # Planes: trace boundary loops → face with holes. Works for
            # generic planar regions bounded by arbitrary shapes.
            if prim_type == "plane":
                face = _face_from_boundary(surface, prim_type, comp, mesh)
            else:
                # Curved surfaces (cylinder/torus/sphere): use u/v
                # bounding box of assigned face vertices. This picks up
                # arc-strips, spherical caps, etc. correctly, whereas
                # loop-tracing miscategorises two-circle boundaries
                # (fillet bands) as "outer + hole" and produces
                # degenerate topology.
                face = _face_from_uv_bbox(surface, prim_type, comp, mesh)
            if face is None:
                stats[f"face_failed_{prim_type}"] += 1
                continue

            if sew:
                sewing.Add(face)
            else:
                builder.Add(compound, face)
            faces_written += 1
            stats[f"face_ok_{prim_type}"] += 1

    if sew:
        sewing.Perform()
        shape: TopoDS_Shape = sewing.SewedShape()
    else:
        shape = compound

    os.makedirs(os.path.dirname(step_path), exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(step_path)
    ok = status == IFSelect_RetDone

    return {
        "step_path": step_path if ok else None,
        "faces_written": faces_written,
        "primitives_total": len(graph.get("primitives", [])),
        "step_write_ok": bool(ok),
        "stats": dict(stats),
    }


def stitch_from_graph(mesh: trimesh.Trimesh, graph: dict, step_path: str,
                      sew: bool = True) -> dict:
    """Build one BRep face per assigned mesh triangle, projected onto its
    primitive's smooth surface. Sew all faces into a shell (or compound
    if sew=False). Export as STEP.
    """
    stats = Counter()
    triangles_written = 0
    triangles_attempted = 0

    if sew:
        sewing = BRepBuilderAPI_Sewing(1e-3, True, True, True, False)
    else:
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)

    face_vertex_positions = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.faces)

    for prim in graph.get("primitives", []):
        surface = _surface_from_primitive(prim)
        if surface is None:
            stats["unsupported"] += 1
            continue

        face_ids = prim.get("face_ids", [])
        if not face_ids:
            stats["no_faces"] += 1
            continue

        prim_type = prim["type"]
        u_per, v_per, period_u, period_v = _periodicity_for(prim_type)
        sa = ShapeAnalysis_Surface(surface) if prim_type != "plane" else None
        tol = 1e-3

        for face_id in face_ids:
            triangles_attempted += 1
            v0, v1, v2 = face_vertex_positions[mesh_faces[int(face_id)]]

            if prim_type == "plane":
                face = _make_face_on_plane_from_triangle(surface, v0, v1, v2)
            else:
                # Project each vertex to (u, v).
                uv = np.empty((3, 2), dtype=float)
                for i, v in enumerate((v0, v1, v2)):
                    p = sa.ValueOfUV(
                        gp_Pnt(float(v[0]), float(v[1]), float(v[2])), tol
                    )
                    uv[i] = (p.X(), p.Y())
                # Handle periodic wrap so the triangle isn't drawn the
                # long way around.
                uv = _unwrap_uv(uv, period_u, period_v, u_per, v_per)
                face = _make_face_on_curved_from_triangle(surface, uv)

            if face is None:
                stats[f"tri_failed_{prim_type}"] += 1
                continue
            if sew:
                sewing.Add(face)
            else:
                builder.Add(compound, face)
            triangles_written += 1
            stats[f"tri_ok_{prim_type}"] += 1

    # Assemble the shape.
    if sew:
        sewing.Perform()
        shape: TopoDS_Shape = sewing.SewedShape()
    else:
        shape = compound

    # Write STEP.
    os.makedirs(os.path.dirname(step_path), exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(step_path)
    ok = status == IFSelect_RetDone

    return {
        "step_path": step_path if ok else None,
        "triangles_attempted": triangles_attempted,
        "triangles_written": triangles_written,
        "primitives_total": len(graph.get("primitives", [])),
        "step_write_ok": bool(ok),
        "stats": dict(stats),
    }


# ----------------------------------------------------------------------------
# STEP → mesh (roundtrip helper for scoring).
# ----------------------------------------------------------------------------

def step_to_mesh(step_path: str, linear_deflection_mm: float = 0.1
                 ) -> trimesh.Trimesh | None:
    """Load a STEP file, mesh it via OCCT's incremental mesher, return a
    trimesh Trimesh. Returns None on any read/mesh failure."""
    import tempfile
    # Deferred imports so eval.py doesn't need OCP just to *skip* STEP scoring.
    from OCP.STEPControl import STEPControl_Reader
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    try:
        r = STEPControl_Reader()
        if r.ReadFile(step_path) != IFSelect_RetDone:
            return None
        r.TransferRoots()
        shape = r.OneShape()
        BRepMesh_IncrementalMesh(shape, linear_deflection_mm, False, 0.5, True)
        tmp = tempfile.mktemp(suffix=".stl")
        StlAPI_Writer().Write(shape, tmp)
        m = trimesh.load(tmp, force="mesh", process=True)
        os.remove(tmp)
        if not hasattr(m, "vertices") or len(m.vertices) == 0:
            return None
        return m
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# eval.py pipeline adapter.
# ----------------------------------------------------------------------------

def pipeline_brep_stitch(in_path: str, out_dir: str, design_id: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    mesh, _repair_report = repair_mesh(in_path)
    graph = detect_primitives(mesh)
    graph_path = os.path.join(out_dir, f"{design_id}.primitive_graph.json")
    with open(graph_path, "w") as fp:
        json.dump(graph, fp, indent=2, default=str)
    step_path = os.path.join(out_dir, f"{design_id}.step")
    stitched = stitch_from_graph_merged(mesh, graph, step_path)
    stitched["triangles_written"] = stitched.get(
        "faces_written", 0
    )  # legacy key for eval.py summary
    return {
        "output_mesh_path": None,
        "step_path": stitched["step_path"],
        "primitive_graph_path": graph_path,
        "notes": (f"prims={stitched['primitives_total']}, "
                  f"BRep_faces={stitched['triangles_written']}, "
                  f"step_ok={stitched['step_write_ok']}"),
    }


# ----------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stitch STL into a STEP via primitives.")
    p.add_argument("in_path")
    p.add_argument("out_dir")
    p.add_argument("--id", default=None,
                   help="Output basename. Defaults to input stem.")
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["merged", "per_triangle"],
                   default="merged",
                   help="merged (default): one BRep face per primitive-region "
                   "with the traced boundary. per_triangle: one BRep face per "
                   "mesh triangle, projected to the primitive surface.")
    args = p.parse_args(argv)

    design_id = args.id or os.path.splitext(os.path.basename(args.in_path))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    mesh, _ = repair_mesh(args.in_path)
    graph = detect_primitives(mesh, n_samples=args.samples, seed=args.seed)
    step_path = os.path.join(args.out_dir, f"{design_id}.step")
    if args.mode == "merged":
        result = stitch_from_graph_merged(mesh, graph, step_path)
    else:
        result = stitch_from_graph(mesh, graph, step_path)
    result["elapsed_s"] = time.time() - t0
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["step_write_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

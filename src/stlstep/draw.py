"""The 'draw it' path — reconstruct a part as a parametric feature model.

Instead of rebuilding a solid face-by-face and stitching (which fights topology),
recognize the part's construction and re-draw it with feature operations. The
topology is then correct by construction, exactly as the original was modelled.

This module implements the first and most common case: EXTRUSIONS. It detects a
constant cross-section along an axis, extracts and fits that 2D profile into
clean lines + arcs, and emits a build123d feature script (sketch -> extrude) plus
the resulting solid. The generated script IS the feature tree.

Non-extrusion cases (revolves, multi-feature parts) are where an LLM composes the
build123d operations; this deterministic extruder is the proof of concept and the
template for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


# ----------------------------------------------------------------------------
# Dispatcher: try each feature reconstruction, keep the best that verifies.
# ----------------------------------------------------------------------------

@dataclass
class DrawResult:
    method: str               # "extrude" | "revolve"
    part: object              # build123d Part
    script: str               # feature-tree script
    summary: dict             # segment kind counts
    deviation: float          # p99 distance to the mesh (mm)


def draw(mesh, *, dev_tol=0.1):
    """Try to reconstruct `mesh` as a parametric feature model. Attempts each
    recognizer (extrusion, revolve), builds + verifies against the mesh, and
    returns the lowest-deviation result that meets `dev_tol`. Returns None if
    the part isn't a clean single-feature solid (caller falls back to BREP)."""
    candidates = []
    for name, detect, build in (("extrude", detect_extrusion, draw_extrusion),
                                ("revolve", detect_revolve, draw_revolve)):
        try:
            spec = detect(mesh)
            if spec is None:
                continue
            part, script, summary = build(spec, mesh=mesh)
            if part is None:
                continue
            dev = _part_deviation(part, mesh)
            candidates.append(DrawResult(name, part, script, summary, dev))
        except Exception:  # noqa: BLE001 — a recognizer failing is not fatal
            continue
    passing = [c for c in candidates if c.deviation <= dev_tol]
    if not passing:
        return None
    return min(passing, key=lambda c: c.deviation)


# ----------------------------------------------------------------------------
# Extrusion detection + profile extraction.
# ----------------------------------------------------------------------------

@dataclass
class Extrusion:
    axis: int                 # 0=x, 1=y, 2=z
    length: float
    loops: list               # [outer, hole1, ...]; each an (N,2) array of profile pts
    origin_offset: float      # coordinate of the base cap along the axis


def detect_extrusion(mesh: trimesh.Trimesh, *, area_tol=0.01):
    """Return an Extrusion if the mesh has a constant cross-section along a world
    axis, else None. Slices at several stations and checks the areas agree."""
    for axis in (2, 1, 0):  # try Z first (most parts are extruded 'up')
        normal = np.zeros(3); normal[axis] = 1.0
        lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
        length = hi - lo
        if length <= 1e-6:
            continue
        areas = []
        loops_mid = None
        ok = True
        for i, frac in enumerate((0.2, 0.5, 0.8)):
            origin = np.zeros(3); origin[axis] = lo + length * frac
            sec = mesh.section(plane_origin=origin, plane_normal=normal)
            if sec is None:
                ok = False; break
            p2d, to_3d = sec.to_2D()
            areas.append(p2d.area)
            if i == 1:
                loops_mid = _polygons_to_loops(p2d, to_3d, axis)
        if not ok or not areas:
            continue
        a = np.array(areas)
        if a.max() > 0 and (a.max() - a.min()) / a.max() < area_tol and loops_mid:
            return Extrusion(axis=axis, length=length, loops=loops_mid,
                             origin_offset=lo)
    return None


def _polygons_to_loops(p2d, to_3d, axis):
    """Cross-section rings in WORLD coords of the two in-plane axes, so the
    rebuilt sketch lands where the original part is. `to_3d` maps the section's
    2D frame back to 3D world."""
    u, v = [c for c in (0, 1, 2) if c != axis]
    loops = []
    for poly in p2d.polygons_full:
        for ring in [poly.exterior] + list(poly.interiors):
            pts2d = np.asarray(ring.coords)
            pts3d = trimesh.transformations.transform_points(
                np.column_stack([pts2d, np.zeros(len(pts2d))]), to_3d)
            loops.append(pts3d[:, [u, v]])
    return loops


# ----------------------------------------------------------------------------
# Revolve detection + profile extraction.
# ----------------------------------------------------------------------------

@dataclass
class Revolution:
    axis: np.ndarray          # unit axis direction
    point: np.ndarray         # a point on the axis
    normal: np.ndarray        # section-plane normal (perp to axis)
    radial_dir: np.ndarray    # in-plane radial direction (perp to axis)
    loop: np.ndarray          # (N,2) profile in (radial, axial), radial >= 0


def detect_revolve(mesh, *, angles=(37.0, 79.0, 151.0), tol_frac=2e-3):
    """Return a Revolution if the mesh is rotationally symmetric about a world
    axis through its centre, else None. Checks that rotating by several angles
    leaves the shape unchanged, then extracts the (radial, axial) profile."""
    c = np.asarray(mesh.bounding_box.centroid, dtype=float)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    for axis in (np.array([0.0, 0, 1]), np.array([0.0, 1, 0]),
                 np.array([1.0, 0, 0])):
        if _rot_symmetric(mesh, axis, c, angles, tol_frac * diag):
            rev = _extract_profile(mesh, axis, c)
            if rev is not None:
                return rev
    return None


def _rot_symmetric(mesh, axis, pt, angles, tol):
    import trimesh
    pts, _ = trimesh.sample.sample_surface(mesh, 4000)
    pts = np.asarray(pts)
    for a in angles:
        R = trimesh.transformations.rotation_matrix(np.radians(a), axis, pt)
        rot = mesh.copy(); rot.apply_transform(R)
        _, d, _ = trimesh.proximity.closest_point(rot, pts)
        if float(d.max()) > tol:
            return False
    return True


def _extract_profile(mesh, axis, pt):
    from shapely.geometry import Polygon, box
    axis = axis / np.linalg.norm(axis)
    # a normal perpendicular to the axis (the section plane contains the axis)
    seed = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1, 0])
    normal = np.cross(axis, seed); normal /= np.linalg.norm(normal)
    # radial_dir = axis x normal so that, on Plane(x_dir=radial_dir, z_dir=normal),
    # the sketch's local Y (= normal x radial_dir) is +axis, not -axis (which
    # would build the profile flipped along the axis).
    radial_dir = np.cross(axis, normal); radial_dir /= np.linalg.norm(radial_dir)

    sec = mesh.section(plane_origin=pt, plane_normal=normal)
    if sec is None or not sec.discrete:
        return None
    loop3d = np.asarray(max(sec.discrete, key=len))
    rel = loop3d - pt
    radial = rel @ radial_dir
    axial = rel @ axis
    poly = Polygon(np.column_stack([radial, axial]))
    if not poly.is_valid:
        poly = poly.buffer(0)
    half = box(0.0, axial.min() - 1, radial.max() + 1, axial.max() + 1)
    prof = poly.intersection(half)
    if prof.is_empty or prof.geom_type != "Polygon":
        return None
    loop = np.asarray(prof.exterior.coords)
    return Revolution(axis=axis, point=pt, normal=normal, radial_dir=radial_dir,
                      loop=loop)


def draw_revolve(rev: Revolution, mesh=None, *, dev_tol=0.1):
    """Build a revolved parametric solid + feature-tree script from a profile."""
    from build123d import (BuildPart, BuildSketch, BuildLine, Plane, Axis,
                           Line, ThreePointArc, make_face, revolve)
    from OCP.gp import gp_Ax1, gp_Pnt, gp_Dir

    segs = fit_loop(rev.loop, arcs=True)
    if not segs:
        return None, None, None

    def build(segs_):
        plane = Plane(origin=tuple(rev.point), x_dir=tuple(rev.radial_dir),
                      z_dir=tuple(rev.normal))
        axis = Axis(tuple(rev.point), tuple(rev.axis))
        with BuildPart() as part:
            with BuildSketch(plane):
                with BuildLine():
                    for s in segs_:
                        if s.kind == "line":
                            Line(s.p0, s.p1)
                        elif s.kind == "arc":
                            ThreePointArc(s.p0, s.mid, s.p1)
                make_face()
            revolve(axis=axis)
        return part.part

    part = build(segs)
    if mesh is not None and any(s.kind == "arc" for s in segs):
        if _part_deviation(part, mesh) > dev_tol:
            segs2 = fit_loop(rev.loop, arcs=False)
            part2 = build(segs2)
            if _part_deviation(part2, mesh) < _part_deviation(part, mesh):
                segs, part = segs2, part2

    script = _emit_revolve_script(rev, segs)
    from collections import Counter
    return part, script, dict(Counter(s.kind for s in segs))


def _emit_revolve_script(rev, segs):
    def f(x):
        return f"{float(x):.6g}"

    def vec(v):
        return "(" + ", ".join(f(x) for x in v) + ")"

    L = ["from build123d import *", "",
         f"# Reconstructed revolve about axis {[round(float(x),3) for x in rev.axis]} "
         f"through {[round(float(x),3) for x in rev.point]}", "",
         "with BuildPart() as part:",
         f"    with BuildSketch(Plane(origin={vec(rev.point)}, "
         f"x_dir={vec(rev.radial_dir)}, z_dir={vec(rev.normal)})):",
         "        with BuildLine():"]
    for s in segs:
        if s.kind == "line":
            L.append(f"            Line(({f(s.p0[0])}, {f(s.p0[1])}), "
                     f"({f(s.p1[0])}, {f(s.p1[1])}))")
        elif s.kind == "arc":
            L.append(f"            ThreePointArc(({f(s.p0[0])}, {f(s.p0[1])}), "
                     f"({f(s.mid[0])}, {f(s.mid[1])}), "
                     f"({f(s.p1[0])}, {f(s.p1[1])}))")
    L.append("        make_face()")
    L.append(f"    revolve(axis=Axis({vec(rev.point)}, {vec(rev.axis)}))")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# 2D profile fitting: polyline -> lines + arcs.
# ----------------------------------------------------------------------------

@dataclass
class Seg:
    kind: str                 # "line" | "arc" | "circle"
    p0: tuple
    p1: tuple
    center: tuple = None
    radius: float = None
    mid: tuple = None         # arc: a point on the arc between p0 and p1


def fit_loop(pts, *, arcs=True, corner_deg=18.0, arc_min_pts=4, line_tol=0.02,
             arc_resid_frac=0.06):
    """Fit a closed polyline into clean line/arc segments. A loop with no sharp
    corners that fits a circle is a full circle; otherwise split at corners and
    classify each run between corners as line(s) or a circular arc. With
    `arcs=False`, curved runs are kept as an exact polyline instead (the safe
    fallback the caller verifies against)."""
    P = np.asarray(pts, dtype=float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    P = _dedup(P)
    n = len(P)
    if n < 3:
        return None

    # full circle: closed smooth curve with no big angular gap about its centre
    cfit = _circle_fit_2d(P)
    if cfit is not None and cfit[3] < 0.05 * cfit[2] and _spans_circle(P, cfit):
        return [Seg("circle", tuple(P[0]), tuple(P[0]),
                    center=(cfit[0], cfit[1]), radius=cfit[2])]

    e = np.roll(P, -1, axis=0) - P
    ang = np.arctan2(e[:, 1], e[:, 0])
    turn = np.abs(_wrap(ang - np.roll(ang, 1)))
    corners = [i for i in range(n) if np.degrees(turn[i]) > corner_deg]
    if not corners:
        corners = [0]
    segs = []
    for a, b in zip(corners, corners[1:] + [corners[0] + n]):
        idx = [i % n for i in range(a, b + 1)]
        segs.extend(_curvature_fit(P[idx], arc_min_pts, line_tol,
                                   arc_resid_frac, arcs))
    return segs


def _polyline_segs(pts):
    return [Seg("line", tuple(pts[k]), tuple(pts[k + 1]))
            for k in range(len(pts) - 1)]


def _curvature_fit(run, arc_min_pts, line_tol, arc_resid_frac, arcs,
                   curve_deg=4.0):
    """Split an inter-corner run into lines and arcs by turning rate. A straight
    edge turns ~0 per vertex; a smooth arc turns by a small, consistent amount.

    Group by *curved vertices* (not edges): an arc spans the maximal run of
    turning vertices, so its endpoints are the tangent junction points and the
    circle fit uses only on-arc points — a flanking straight point can't pull the
    radius. Straight stretches between arcs become single lines that share those
    junction points."""
    run = np.asarray(run, dtype=float)
    m = len(run)
    if m < 3:
        return [Seg("line", tuple(run[0]), tuple(run[-1]))]
    d = run[1:] - run[:-1]
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs(_wrap(np.diff(ang)))                 # turning at vertices 1..m-2
    vc = np.zeros(m, dtype=bool)
    vc[1:m - 1] = turn >= np.radians(curve_deg)        # curved vertices

    segs = []
    prev = 0                                            # index up to which emitted
    v = 1
    while v < m - 1:
        if not vc[v]:
            v += 1
            continue
        s = v
        while v < m - 1 and vc[v]:
            v += 1
        t = v - 1                                       # last curved vertex
        if s > prev:                                    # straight lead-in
            segs.append(Seg("line", tuple(run[prev]), tuple(run[s])))
        grp = run[s:t + 1]
        seg = (_fit_arc_group(grp, line_tol, arc_resid_frac)
               if arcs and len(grp) >= arc_min_pts else None)
        segs.extend([seg] if seg else _polyline_segs(grp))
        prev = t
    if prev < m - 1:                                    # straight tail
        segs.append(Seg("line", tuple(run[prev]), tuple(run[m - 1])))
    return segs


def _fit_arc_group(grp, line_tol, arc_resid_frac):
    fit = _circle_fit_2d(grp)
    if fit is None:
        return None
    cx, cy, r, resid = fit
    chord = float(np.linalg.norm(grp[-1] - grp[0]))
    sag = _max_sagitta(grp)
    if r < 1e4 and chord < 2.02 * r and resid < arc_resid_frac * r \
            and sag > line_tol * chord:
        return Seg("arc", tuple(grp[0]), tuple(grp[-1]), center=(cx, cy),
                   radius=r, mid=tuple(grp[len(grp) // 2]))
    return None


def _spans_circle(P, cfit):
    cx, cy, r, _ = cfit
    a = np.sort(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
    gaps = np.append(np.diff(a), (a[0] + 2 * np.pi) - a[-1])
    return float(gaps.max()) < np.radians(40)


# ----------------------------------------------------------------------------
# Draw: fitted profile -> build123d feature model + emitted script.
# ----------------------------------------------------------------------------

_PLANE = {0: "YZ", 1: "XZ", 2: "XY"}


def draw_extrusion(ext: Extrusion, mesh=None, *, dev_tol=0.1):
    """Build the parametric solid and the feature-tree script for an extrusion.

    Fits the profile with arcs, then — if a mesh is given — verifies the result
    against it and, if arcs made it worse than the tolerance, refits without
    arcs (exact polyline for curved runs) and keeps whichever matches better.
    This is the generate->verify->keep-if-better guard: arcs are used only when
    they genuinely help. Returns (part, script, fit_summary)."""
    loops = [fit_loop(lp, arcs=True) for lp in ext.loops]
    loops = [s for s in loops if s]
    if not loops:
        return None, None, None
    part = _build_part(ext, loops)

    if mesh is not None and _has_arcs(loops):
        dev = _part_deviation(part, mesh)
        if dev > dev_tol:
            loops2 = [fit_loop(lp, arcs=False) for lp in ext.loops]
            loops2 = [s for s in loops2 if s]
            part2 = _build_part(ext, loops2)
            if _part_deviation(part2, mesh) < dev:
                loops, part = loops2, part2

    script = _emit_script(ext, loops)
    from collections import Counter
    return part, script, dict(Counter(s.kind for loop in loops for s in loop))


def _has_arcs(loops):
    return any(s.kind == "arc" for loop in loops for s in loop)


def _part_deviation(part, mesh):
    """Max distance from mesh sample points to the built part's surface."""
    from . import occ_util as ou
    import trimesh
    rv, rf = ou.shape_to_mesh(part.wrapped, 0.1)
    if len(rf) == 0:
        return float("inf")
    gm = trimesh.Trimesh(vertices=rv, faces=rf, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, 20000)
    _, d, _ = trimesh.proximity.closest_point(gm, np.asarray(pts))
    return float(np.quantile(d, 0.99))


def _build_part(ext, loops):
    from build123d import (BuildPart, BuildSketch, BuildLine, Plane, Circle,
                           Locations, Line, ThreePointArc, make_face, extrude,
                           Mode)
    plane = getattr(Plane, _PLANE[ext.axis]).offset(ext.origin_offset)
    with BuildPart() as part:
        with BuildSketch(plane):
            for i, segs in enumerate(loops):
                mode = Mode.ADD if i == 0 else Mode.SUBTRACT
                if len(segs) == 1 and segs[0].kind == "circle":
                    c = segs[0]
                    with Locations((c.center[0], c.center[1])):
                        Circle(c.radius, mode=mode)
                else:
                    with BuildLine():
                        for s in segs:
                            if s.kind == "line":
                                Line(s.p0, s.p1)
                            elif s.kind == "arc":
                                ThreePointArc(s.p0, s.mid, s.p1)
                    make_face(mode=mode)
        extrude(amount=ext.length)
    return part.part


def _emit_script(ext, loops):
    """Emit a readable, runnable build123d feature script — the feature tree."""
    def f(x):
        return f"{x:.6g}"
    L = ["from build123d import *", "",
         f"# Reconstructed extrusion along {_PLANE[ext.axis]} normal, "
         f"length {f(ext.length)} mm", "",
         "with BuildPart() as part:",
         f"    with BuildSketch(Plane.{_PLANE[ext.axis]}"
         f".offset({f(ext.origin_offset)})):"]
    for i, segs in enumerate(loops):
        mode = "" if i == 0 else ", mode=Mode.SUBTRACT"
        tag = "outer profile" if i == 0 else "hole"
        L.append(f"        # {tag}")
        if len(segs) == 1 and segs[0].kind == "circle":
            c = segs[0]
            L.append(f"        with Locations(({f(c.center[0])}, "
                     f"{f(c.center[1])})):")
            L.append(f"            Circle({f(c.radius)}{mode})")
        else:
            L.append("        with BuildLine():")
            for s in segs:
                if s.kind == "line":
                    L.append(f"            Line(({f(s.p0[0])}, {f(s.p0[1])}), "
                             f"({f(s.p1[0])}, {f(s.p1[1])}))")
                else:
                    L.append(f"            ThreePointArc(({f(s.p0[0])}, "
                             f"{f(s.p0[1])}), ({f(s.mid[0])}, {f(s.mid[1])}), "
                             f"({f(s.p1[0])}, {f(s.p1[1])}))")
            L.append(f"        make_face({mode.lstrip(', ')})")
    L.append(f"    extrude(amount={f(ext.length)})")
    return "\n".join(L)


def _circle_fit_2d(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, cc = sol
    r2 = cc + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    resid = float(np.abs(np.hypot(x - cx, y - cy) - r).mean())
    return float(cx), float(cy), r, resid


def _max_sagitta(run):
    a, b = run[0], run[-1]
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-9:
        return 0.0
    d = d / L
    perp = np.abs((run - a) @ np.array([-d[1], d[0]]))
    return float(perp.max())


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _dedup(P, tol=1e-2):
    """Drop near-coincident consecutive points (mesh-section artefacts) that
    would otherwise become zero-length edges after rounding."""
    keep = [P[0]]
    for p in P[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    P = np.asarray(keep)
    if len(P) > 1 and np.linalg.norm(P[-1] - P[0]) <= tol:
        P = P[:-1]
    return P

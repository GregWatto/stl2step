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


def fit_loop(pts, *, corner_deg=18.0, arc_min_pts=4, line_tol=0.02,
             arc_resid_frac=0.12):
    """Fit a closed polyline into clean line/arc segments. A loop with no sharp
    corners that fits a circle is a full circle; otherwise split at corners and
    classify each run between corners as a line or a circular arc."""
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
        segs.extend(_fit_run(P[idx], arc_min_pts, line_tol, arc_resid_frac))
    return segs


def _fit_run(run, arc_min_pts, line_tol, arc_resid_frac):
    """Classify one inter-corner run as a single line, a single arc, or (when
    neither cleanly fits) an exact polyline — so a channel/mouth that isn't a
    clean arc stays dimensionally exact instead of being forced into a bad arc."""
    p0, p1 = tuple(run[0]), tuple(run[-1])
    chord = float(np.linalg.norm(run[-1] - run[0]))
    sag = _max_sagitta(run)
    if len(run) < arc_min_pts or sag < line_tol * max(chord, 1.0):
        return [Seg("line", p0, p1)]
    fit = _circle_fit_2d(run)
    if fit is not None:
        cx, cy, r, resid = fit
        # a real arc: endpoints lie on the circle (chord <= diameter), tight fit
        if chord < 1.98 * r and resid < arc_resid_frac * r and r < 1e4:
            return [Seg("arc", p0, p1, center=(cx, cy), radius=r,
                        mid=tuple(run[len(run) // 2]))]
    # neither a clean line nor a clean arc -> exact polyline (no corner-cutting)
    return [Seg("line", tuple(run[k]), tuple(run[k + 1]))
            for k in range(len(run) - 1)]


def _spans_circle(P, cfit):
    cx, cy, r, _ = cfit
    a = np.sort(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
    gaps = np.append(np.diff(a), (a[0] + 2 * np.pi) - a[-1])
    return float(gaps.max()) < np.radians(40)


# ----------------------------------------------------------------------------
# Draw: fitted profile -> build123d feature model + emitted script.
# ----------------------------------------------------------------------------

_PLANE = {0: "YZ", 1: "XZ", 2: "XY"}


def draw_extrusion(ext: Extrusion):
    """Build the parametric solid and the feature-tree script for an extrusion.
    Returns (build123d_part, script_text, fit_summary)."""
    loops = [fit_loop(lp) for lp in ext.loops]
    loops = [s for s in loops if s]
    if not loops:
        return None, None, None
    part = _build_part(ext, loops)
    script = _emit_script(ext, loops)
    from collections import Counter
    summary = Counter(s.kind for loop in loops for s in loop)
    return part, script, dict(summary)


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

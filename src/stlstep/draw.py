"""The 'draw it' path — reconstruct a part as a parametric feature model.

Instead of rebuilding a solid face-by-face and stitching (which fights topology),
recognize the part's construction and re-draw it with feature operations. The
topology is then correct by construction, exactly as the original was modelled.

The deterministic recognizers, tried cheapest-first by `draw()`:
  - EXTRUSION   — constant cross-section along an axis (fit to lines + arcs).
  - REVOLVE     — a profile swept about an axis.
  - PRISMATIC   — a stack of constant-section slabs (base + boss + pocket + step).
  - LOFT        — a cross-section that varies continuously along an axis between
                  flat end caps (square-to-round adapter, drafted/tapered body).
Each extracts and fits the real profile(s), emits a build123d feature script, and
is verified against the mesh; the generated script IS the feature tree.

Parts no recognizer fits (organic/open-shell/multi-axis bodies) fall to the LLM
compose path or the BREP solid; these deterministic recognizers are the fast,
exact path when they apply.
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
    """Try to reconstruct `mesh` as a parametric feature model, cheapest model
    first. Tier 1 is a single feature (extrusion / revolve) — the simplest, most
    editable result, preferred whenever it fits. Tier 2 is a multi-level
    prismatic stack (base + boss + pocket + step + holes). Every candidate is
    built and verified against the mesh; returns the best passing result, or None
    if nothing fits within `dev_tol` (caller falls back to BREP)."""
    # Tier 1: single-feature recognizers.
    single = []
    for name, detect, build in (("extrude", detect_extrusion, draw_extrusion),
                                ("revolve", detect_revolve, draw_revolve)):
        try:
            spec = detect(mesh)
            if spec is None:
                continue
            part, script, summary = build(spec, mesh=mesh)
            if part is None:
                continue
            single.append(DrawResult(name, part, script, summary,
                                     _part_deviation(part, mesh)))
        except Exception:  # noqa: BLE001 — a recognizer failing is not fatal
            continue
    passing = [c for c in single if c.deviation <= dev_tol]
    if passing:
        return min(passing, key=lambda c: c.deviation)

    # Tier 2: multi-feature prismatic stack. Try each candidate layering axis,
    # build+verify, and return the first that passes (candidates are ordered so
    # the most-decomposed stack is tried first).
    try:
        for pr in _prismatic_candidates(mesh):
            part, script, summary = draw_prismatic(pr, mesh=mesh, dev_tol=dev_tol)
            if part is None:
                continue
            dev = _part_deviation(part, mesh)
            if dev <= dev_tol:
                return DrawResult("prismatic", part, script, summary, dev)
    except Exception:  # noqa: BLE001
        pass

    # Tier 3: lofted (continuously-varying cross-section).
    try:
        lf = detect_lofted(mesh)
        if lf is not None:
            part, script, summary = draw_lofted(lf, mesh=mesh, dev_tol=dev_tol)
            if part is not None:
                dev = _part_deviation(part, mesh)
                if dev <= dev_tol:
                    return DrawResult("lofted", part, script, summary, dev)
    except Exception:  # noqa: BLE001
        pass
    return None


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
# Prismatic (2.5D) multi-feature detection.
#
# The natural generalization of a single extrusion: a part built as a stack of
# extruded layers along one axis — base plate + raised boss + blind pocket +
# steps + through-holes. We find the axis heights where the cross-section changes
# (the axis-perpendicular flat caps: plate faces, pocket floors, step shoulders),
# slice each constant-section slab, and rebuild as a UNION of per-slab extrudes.
# Union reproduces bosses (a slab adds material), blind pockets (the slab below
# the floor fills in), and through-holes (absent from every slab) automatically —
# no explicit add-vs-subtract classification of features is needed.
# ----------------------------------------------------------------------------

@dataclass
class Prismatic:
    axis: int                 # 0=x, 1=y, 2=z (the layering axis)
    levels: list              # [(lo, hi, polys), ...] bottom->top; polys is a
                              # list of (exterior (N,2), [interior (M,2), ...])


def detect_prismatic(mesh, *, angle_tol_deg=2.0):
    """Return the best Prismatic (a stack of >=2 constant-section slabs along an
    axis) or None. See `_prismatic_candidates` for how the layering axis is
    chosen — crucially by whether each slab's section is actually constant, not
    just by flat-cap area (which mis-picks the axis for parts like an open
    corner shell whose big vertical walls dominate the cap area)."""
    cands = _prismatic_candidates(mesh, angle_tol_deg=angle_tol_deg)
    return cands[0] if cands else None


def _prismatic_candidates(mesh, *, angle_tol_deg=2.0):
    """All axes that read as a genuine prismatic stack, best first. An axis
    qualifies only if it has >=2 slabs AND each slab's cross-section is constant
    within the slab (the definition of a prismatic level) — this is what rejects
    a wrong layering axis, whose 'slabs' span continuously-varying geometry.
    Ordered by slab count then cap area; the dispatcher builds and verifies each,
    so ordering only affects which is tried first."""
    fn = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    cos_tol = np.cos(np.radians(angle_tol_deg))
    cap_area = [float(areas[np.abs(fn[:, a]) > cos_tol].sum()) for a in (0, 1, 2)]
    cands = []
    for axis in (0, 1, 2):
        levels = _prismatic_levels(mesh, axis, cos_tol)
        if levels is not None:
            cands.append((Prismatic(axis=axis, levels=levels),
                          len(levels), cap_area[axis]))
    cands.sort(key=lambda t: (-t[1], -t[2]))
    return [c[0] for c in cands]


def _section_total_area(mesh, axis, coord):
    polys = _section_polys(mesh, axis, coord)
    if not polys:
        return None
    return float(sum(_poly_area(ext) for ext, _ in polys))


def _prismatic_levels(mesh, axis, cos_tol, *, min_slab_frac=0.01,
                      const_tol=0.15):
    """Slab boundaries along `axis`: the axis coordinates of the flat caps
    (faces whose normal is ~parallel to the axis). Requires >=1 interior break
    (a genuine multi-level part, not a plain extrusion) AND that each slab's
    cross-section is constant within the slab (else `axis` isn't the layering
    axis — its 'slabs' vary continuously, so reject it)."""
    fn = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    length = hi - lo
    if length <= 1e-6:
        return None
    coord_tol = max(1e-3 * length, 0.02)

    perp = np.abs(fn[:, axis]) > cos_tol
    if not perp.any():
        return None
    cap_coord = V[F[perp]][:, :, axis].mean(axis=1)   # per-cap axis coordinate
    breaks = _cluster_coords(cap_coord, coord_tol, areas[perp])
    interior = sorted(b for b in breaks if lo + coord_tol < b < hi - coord_tol)

    bounds = [lo]
    for b in interior + [hi]:
        if b - bounds[-1] > coord_tol:
            bounds.append(b)
    if len(bounds) < 3:            # no interior break -> a plain extrusion
        return None

    levels = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < max(coord_tol, min_slab_frac * length):
            continue
        polys = _section_polys(mesh, axis, 0.5 * (a + b))
        if not polys:
            return None
        # constancy: sample the slab interior; a real prismatic level holds its
        # cross-section, so a big area swing means `axis` is not the stack axis.
        if b - a > 4 * coord_tol:
            aa = [_section_total_area(mesh, axis, a + f * (b - a))
                  for f in (0.25, 0.5, 0.75)]
            if any(x is None or x <= 0 for x in aa):
                return None
            if (max(aa) - min(aa)) / max(aa) > const_tol:
                return None
        levels.append((a, b, polys))
    return levels if len(levels) >= 2 else None


def _cluster_coords(coords, tol, weights, *, min_weight_frac=0.02):
    """1D clustering: merge sorted coordinates within `tol`, area-weighted.
    Drops clusters whose total area is a negligible fraction of the largest
    (stray triangles) so they don't create phantom breakpoints."""
    coords = np.asarray(coords, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(coords) == 0:
        return []
    order = np.argsort(coords)
    coords, weights = coords[order], weights[order]
    centers, cw = [], []
    cur_c, cur_w = [coords[0]], [weights[0]]
    for c, w in zip(coords[1:], weights[1:]):
        if c - cur_c[-1] <= tol:
            cur_c.append(c); cur_w.append(w)
        else:
            centers.append(np.average(cur_c, weights=cur_w)); cw.append(sum(cur_w))
            cur_c, cur_w = [c], [w]
    centers.append(np.average(cur_c, weights=cur_w)); cw.append(sum(cur_w))
    thresh = min_weight_frac * max(cw)
    return [ct for ct, w in zip(centers, cw) if w >= thresh]


def _section_polys(mesh, axis, coord):
    """Cross-section at `coord` along `axis` as a list of polygons, each
    (exterior (N,2), [interior (M,2), ...]) in world (u, v) coords. None if the
    plane misses the mesh."""
    normal = np.zeros(3); normal[axis] = 1.0
    origin = np.zeros(3); origin[axis] = coord
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return None
    p2d, to_3d = sec.to_2D()
    u, v = [c for c in (0, 1, 2) if c != axis]

    def world_uv(ring):
        pts2d = np.asarray(ring.coords)
        pts3d = trimesh.transformations.transform_points(
            np.column_stack([pts2d, np.zeros(len(pts2d))]), to_3d)
        return pts3d[:, [u, v]]

    polys = [(world_uv(poly.exterior), [world_uv(r) for r in poly.interiors])
             for poly in p2d.polygons_full]
    return polys or None


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
# Draw: prismatic stack -> union of per-slab extrudes + emitted script.
# ----------------------------------------------------------------------------

def draw_prismatic(pr: Prismatic, mesh=None, *, dev_tol=0.1):
    """Build the multi-level prismatic solid + feature-tree script. Fits each
    slab's cross-section (outer + holes) into lines/arcs/circles and builds the
    part as a union of per-slab extrudes. verify-and-fallback: if arcs push the
    result past tolerance, refit the curved runs as exact polylines."""
    fitted = _fit_prismatic(pr, arcs=True)
    if fitted is None:
        return None, None, None
    part = _build_prismatic(pr, fitted)
    if part is None:
        return None, None, None

    if mesh is not None and _prismatic_has_arcs(fitted):
        dev = _part_deviation(part, mesh)
        if dev > dev_tol:
            fitted2 = _fit_prismatic(pr, arcs=False)
            part2 = _build_prismatic(pr, fitted2) if fitted2 else None
            if part2 is not None and _part_deviation(part2, mesh) < dev:
                fitted, part = fitted2, part2

    script = _emit_prismatic_script(pr, fitted)
    from collections import Counter
    kinds = Counter(s.kind for _, _, lvl in fitted for es, hs in lvl
                    for grp in [es, *hs] for s in grp)
    summary = dict(kinds); summary["levels"] = len(fitted)
    return part, script, summary


def _fit_prismatic(pr, *, arcs):
    """Fit every slab's loops. Returns [(lo, hi, [(ext_segs, [hole_segs...]), ...])]
    or None if any exterior loop fails to fit."""
    fitted = []
    for lo, hi, polys in pr.levels:
        lvl = []
        for ext, ints in polys:
            es = fit_loop(ext, arcs=arcs)
            if not es:
                return None
            hs = [h for h in (fit_loop(hole, arcs=arcs) for hole in ints) if h]
            lvl.append((es, hs))
        fitted.append((lo, hi, lvl))
    return fitted


def _prismatic_has_arcs(fitted):
    return any(s.kind == "arc" for _, _, lvl in fitted for es, hs in lvl
              for grp in [es, *hs] for s in grp)


def _build_prismatic(pr, fitted):
    from build123d import BuildPart, BuildSketch, Plane, Mode, extrude
    base = getattr(Plane, _PLANE[pr.axis])
    try:
        with BuildPart() as part:
            for lo, hi, lvl in fitted:
                with BuildSketch(base.offset(lo)):
                    for es, hs in lvl:
                        _sketch_loop(es, Mode.ADD)
                        for h in hs:
                            _sketch_loop(h, Mode.SUBTRACT)
                extrude(amount=hi - lo)   # ADD: unions with the stack so far
        return part.part
    except Exception:  # noqa: BLE001 — a bad slab must not crash the dispatcher
        return None


def _emit_prismatic_script(pr, fitted):
    pl = _PLANE[pr.axis]
    L = ["from build123d import *", "",
         f"# Reconstructed prismatic stack along {pl} normal, "
         f"{len(fitted)} levels (union of per-slab extrudes)", "",
         "with BuildPart() as part:"]
    for k, (lo, hi, lvl) in enumerate(fitted):
        L.append(f"    # level {k + 1}: {pl} offset {_fmt(lo)} -> {_fmt(hi)}")
        L.append(f"    with BuildSketch(Plane.{pl}.offset({_fmt(lo)})):")
        for j, (es, hs) in enumerate(lvl):
            for grp, suffix in [(es, "")] + [(h, ", mode=Mode.SUBTRACT")
                                             for h in hs]:
                L.extend(_emit_loop_lines(grp, suffix, 8))
        L.append(f"    extrude(amount={_fmt(hi - lo)})")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Lofted (multi-section) detection.
#
# The case between an extrusion (constant section) and a revolve (rotational): a
# section that changes CONTINUOUSLY along an axis — a square-to-round adapter, a
# drafted/tapered body, a transition duct. We find the axis with flat end caps
# and a section that varies (not constant -> not an extrude; not stepwise ->
# not a prismatic stack), sample the section at several stations, fit each to
# lines/arcs, and rebuild as a LOFT through the sampled profiles. A section-wide
# through-hole is lofted and subtracted too.
# ----------------------------------------------------------------------------

@dataclass
class Lofted:
    axis: int                 # 0=x, 1=y, 2=z (the loft/sweep axis)
    stations: list            # [(coord, ext(N,2), [hole(M,2), ...]), ...] end->end


def _poly_area(pts):
    x, y = np.asarray(pts)[:, 0], np.asarray(pts)[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def detect_lofted(mesh, *, n_stations=8, vary_tol=0.03, cap_frac=0.02,
                  angle_tol_deg=2.0):
    """Return a Lofted if the part is a single body whose cross-section varies
    continuously along an axis with flat caps at both ends, else None. Ordered so
    a plain extrusion (constant section) and a prismatic stack are preferred by
    the dispatcher before this is tried."""
    fn = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
    cos_tol = np.cos(np.radians(angle_tol_deg))
    total_area = float(areas.sum())

    for axis in (2, 1, 0):
        lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
        length = hi - lo
        if length <= 1e-6:
            continue
        eps = max(1e-3 * length, 0.05)
        # flat end caps: perpendicular face area near both extremes
        perp = np.abs(fn[:, axis]) > cos_tol
        if not perp.any():
            continue
        cc = V[F[perp]][:, :, axis].mean(axis=1)
        pa = areas[perp]
        bot = float(pa[cc < lo + 3 * eps].sum())
        top = float(pa[cc > hi - 3 * eps].sum())
        if bot < cap_frac * total_area or top < cap_frac * total_area:
            continue

        # sample interior sections (largest polygon at each), place ends at lo/hi
        fracs = np.linspace(0.0, 1.0, n_stations)
        stations, sec_areas, ok = [], [], True
        for k, fr in enumerate(fracs):
            coord = lo + eps + fr * (length - 2 * eps)   # sample just inside caps
            polys = _section_polys(mesh, axis, coord)
            if not polys:
                ok = False; break
            ext, ints = max(polys, key=lambda pi: _poly_area(pi[0]))
            place = lo if k == 0 else (hi if k == len(fracs) - 1 else coord)
            stations.append((place, ext, ints))
            sec_areas.append(_poly_area(ext))
        if not ok or len(stations) < 3:
            continue
        a = np.asarray(sec_areas)
        if a.max() <= 0 or (a.max() - a.min()) / a.max() < vary_tol:
            continue     # ~constant section -> an extrusion, not a loft
        return Lofted(axis=axis, stations=stations)
    return None


def _fit_lofted(spec, *, arcs):
    """Fit each station's exterior (+ a single through-hole if present at every
    station) into line/arc/circle segments. Returns (outer, holes) or None."""
    outer, holes, all_have_hole = [], [], True
    for coord, ext, ints in spec.stations:
        es = fit_loop(ext, arcs=arcs)
        if not es:
            return None
        outer.append((coord, es))
        if ints:
            hs = fit_loop(max(ints, key=_poly_area), arcs=arcs)
            holes.append((coord, hs)) if hs else holes.append((coord, None))
        else:
            all_have_hole = False
    if not (all_have_hole and holes and all(h for _, h in holes)):
        holes = None
    return outer, holes


def draw_lofted(spec: Lofted, mesh=None, *, dev_tol=0.1):
    """Build the lofted solid + feature-tree script.

    OCCT's ThruSections can only loft wires of compatible topology, so two shapes
    of loft need different section sets: a same-topology varying loft (a barrel /
    curved taper — every section a circle or every section a square) lofts cleanly
    through ALL sampled stations, while a topology-changing ruled loft (square ->
    round) only lofts between its two clean end caps. Try the full stack first,
    fall back to the two ends, and keep whichever builds and verifies best."""
    n = len(spec.stations)
    idx_options = [list(range(n)), [0, n - 1]]
    best = None                        # (dev, part, fitted, idx)
    for arcs in (True, False):
        fitted = _fit_lofted(spec, arcs=arcs)
        if fitted is None:
            continue
        for idx in idx_options:
            part = _build_lofted(spec, fitted, idx)
            if part is None:
                continue
            dev = _part_deviation(part, mesh) if mesh is not None else 0.0
            if best is None or dev < best[0]:
                best = (dev, part, fitted, idx)
            if dev <= dev_tol:
                break
        if best is not None and best[0] <= dev_tol:
            break
    if best is None:
        return None, None, None
    dev, part, fitted, idx = best
    script = _emit_lofted_script(spec, fitted, idx)
    outer, holes = fitted
    summary = {"stations": len(idx), "lofted_hole": bool(holes)}
    return part, script, summary


def _build_lofted(spec, fitted, idx):
    from build123d import (BuildPart, BuildSketch, Plane, Mode, loft)
    outer, holes = fitted
    base = getattr(Plane, _PLANE[spec.axis])
    try:
        with BuildPart() as part:
            for i in idx:
                coord, es = outer[i]
                with BuildSketch(base.offset(coord)):
                    _sketch_loop(es, Mode.ADD)
            loft()
            if holes:
                for i in idx:
                    coord, hs = holes[i]
                    with BuildSketch(base.offset(coord)):
                        _sketch_loop(hs, Mode.ADD)
                loft(mode=Mode.SUBTRACT)
        return part.part
    except Exception:  # noqa: BLE001
        return None


def _emit_lofted_script(spec, fitted, idx):
    outer, holes = fitted
    pl = _PLANE[spec.axis]
    L = ["from build123d import *", "",
         f"# Reconstructed loft along {pl} normal through {len(idx)} sections",
         "", "with BuildPart() as part:"]
    for i in idx:
        coord, es = outer[i]
        L.append(f"    with BuildSketch(Plane.{pl}.offset({_fmt(coord)})):")
        L.extend(_emit_loop_lines(es, "", 8))
    L.append("    loft()")
    if holes:
        L.append("    # through-hole lofted and subtracted")
        for i in idx:
            coord, hs = holes[i]
            L.append(f"    with BuildSketch(Plane.{pl}.offset({_fmt(coord)})):")
            L.extend(_emit_loop_lines(hs, "", 8))
        L.append("    loft(mode=Mode.SUBTRACT)")
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

    # Thin-band guard. The corner→segment logic assumes a simple filled outline;
    # given a THIN out-and-back band (an open-shell wall section, whose inner
    # return path hugs the outer one) it "cuts across" and fills the silhouette.
    # Detect that by a gross change in enclosed area and instead keep a
    # shape-preserving simplified polyline, so thin walls survive intact.
    raw_area = _poly_area(P)
    if raw_area > 1e-6 and len(segs) >= 3:
        fit_area = _poly_area([s.p0 for s in segs])
        if abs(fit_area - raw_area) > 0.25 * raw_area:
            diag = float(np.hypot(*(P.max(0) - P.min(0))))
            simp = _rdp_closed(P, max(0.15, 3e-3 * diag))
            if len(simp) >= 3:
                return _polyline_segs(np.vstack([simp, simp[0]]))
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
    from build123d import BuildPart, BuildSketch, Plane, Mode, extrude
    plane = getattr(Plane, _PLANE[ext.axis]).offset(ext.origin_offset)
    with BuildPart() as part:
        with BuildSketch(plane):
            for i, segs in enumerate(loops):
                _sketch_loop(segs, Mode.ADD if i == 0 else Mode.SUBTRACT)
        extrude(amount=ext.length)
    return part.part


def _sketch_loop(segs, mode):
    """Add one fitted loop (a circle, or a lines/arcs wire) to the currently
    open BuildSketch, in the given ADD/SUBTRACT mode. build123d's builder stack
    picks up the active sketch, so this composes inside a `with BuildSketch(..)`."""
    from build123d import (BuildLine, Locations, Circle, Line, ThreePointArc,
                           make_face)
    if len(segs) == 1 and segs[0].kind == "circle":
        c = segs[0]
        with Locations((c.center[0], c.center[1])):
            Circle(c.radius, mode=mode)
    else:
        # Pass the wire explicitly: build123d's argument-less make_face() relies
        # on caller-frame inspection and finds nothing when called from a helper.
        with BuildLine() as bl:
            for s in segs:
                if s.kind == "line":
                    Line(s.p0, s.p1)
                elif s.kind == "arc":
                    ThreePointArc(s.p0, s.mid, s.p1)
        make_face(bl.line, mode=mode)


def _fmt(x):
    x = float(x)
    if abs(x) < 1e-9:            # snap numerical dust to a clean 0
        x = 0.0
    return f"{x:.6g}"


def _emit_loop_lines(segs, mode_suffix, indent):
    """Emit the sketch body for one loop (Circle, or BuildLine + make_face) as
    source lines at the given indent. `mode_suffix` is "" or ", mode=Mode.SUBTRACT"."""
    pad = " " * indent
    L = []
    if len(segs) == 1 and segs[0].kind == "circle":
        c = segs[0]
        L.append(f"{pad}with Locations(({_fmt(c.center[0])}, "
                 f"{_fmt(c.center[1])})):")
        L.append(f"{pad}    Circle({_fmt(c.radius)}{mode_suffix})")
    else:
        L.append(f"{pad}with BuildLine():")
        for s in segs:
            if s.kind == "line":
                L.append(f"{pad}    Line(({_fmt(s.p0[0])}, {_fmt(s.p0[1])}), "
                         f"({_fmt(s.p1[0])}, {_fmt(s.p1[1])}))")
            else:
                L.append(f"{pad}    ThreePointArc(({_fmt(s.p0[0])}, "
                         f"{_fmt(s.p0[1])}), ({_fmt(s.mid[0])}, {_fmt(s.mid[1])}), "
                         f"({_fmt(s.p1[0])}, {_fmt(s.p1[1])}))")
        L.append(f"{pad}make_face({mode_suffix.lstrip(', ')})")
    return L


def _emit_script(ext, loops):
    """Emit a readable, runnable build123d feature script — the feature tree."""
    L = ["from build123d import *", "",
         f"# Reconstructed extrusion along {_PLANE[ext.axis]} normal, "
         f"length {_fmt(ext.length)} mm", "",
         "with BuildPart() as part:",
         f"    with BuildSketch(Plane.{_PLANE[ext.axis]}"
         f".offset({_fmt(ext.origin_offset)})):"]
    for i, segs in enumerate(loops):
        mode = "" if i == 0 else ", mode=Mode.SUBTRACT"
        L.append(f"        # {'outer profile' if i == 0 else 'hole'}")
        L.extend(_emit_loop_lines(segs, mode, 8))
    L.append(f"    extrude(amount={_fmt(ext.length)})")
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


def _rdp_closed(P, eps):
    """Douglas-Peucker simplification of a CLOSED loop, preserving shape to
    within `eps`. Anchors at the point farthest from the centroid (a definite
    corner) so the open-polyline pass has good, stable endpoints."""
    P = np.asarray(P, dtype=float)
    if len(P) < 4:
        return P
    c = P.mean(axis=0)
    start = int(np.argmax(((P - c) ** 2).sum(axis=1)))
    Q = np.roll(P, -start, axis=0)
    Q = np.vstack([Q, Q[0]])            # open polyline start..start
    keep = np.zeros(len(Q), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(Q) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = Q[b] - Q[a]
        L = float(np.hypot(seg[0], seg[1]))
        pts = Q[a + 1:b]
        if L < 1e-12:
            d = np.hypot(pts[:, 0] - Q[a][0], pts[:, 1] - Q[a][1])
        else:
            nrm = np.array([-seg[1], seg[0]]) / L
            d = np.abs((pts - Q[a]) @ nrm)
        if len(d) == 0:
            continue
        m = int(np.argmax(d))
        if d[m] > eps:
            k = a + 1 + m
            keep[k] = True
            stack.append((a, k)); stack.append((k, b))
    return Q[keep][:-1]                  # drop the duplicated closing point


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

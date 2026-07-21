"""Region-growing segmentation (the CGAL Region_Growing algorithm, in numpy).

v1/v2's original detector peels primitives from a global RANSAC sample, which
over-fits catastrophically — 24 tori (many single-triangle) on a 10 mm ring.
That garbage is faithfully rendered by the analytic `shell` method as a donut of
spiraling ribbons.

Region growing fixes the root cause: grow *contiguous* patches over the mesh,
cutting only at sharp edges (normal discontinuities), then fit exactly one best
primitive per patch. A cylinder becomes one patch → one cylinder, not 30
fragments; a flat becomes one plane; a fillet that fits nothing becomes one
freeform patch (drape-able) rather than a swarm of fake tori.

Same output type as `segment.segment` (a Segmentation of Regions), so it drops
into topology/verify unchanged.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ._detector import compute_thresholds
from .segment import Region, Segmentation


# Planar pass: a face joins only if its normal is within PLANAR_ANGLE_DEG of the
# region's *running average* normal. Comparing to the region normal (not the
# neighbour) stops a patch at a roundover instead of drifting through it, so
# flats don't merge into freeform blobs.
# Local adjacent-normal continuity grows a cylinder/sphere fully (a 32-facet
# cylinder is ~11 deg/step) while a sharp edge (>=45 deg) cuts the patch. Large
# genuinely-smooth regions (a curved cover) stay whole and become freeform.
GROW_ANGLE_DEG = 25.0


def segment_rg(mesh: trimesh.Trimesh, neighbors: list[list[int]],
               *, grow_angle_deg: float = GROW_ANGLE_DEG) -> Segmentation:
    fn = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    thr = compute_thresholds(mesh)
    # Single-pass smooth growth: local adjacent-normal continuity, cut at sharp
    # edges. Clean and robust across parts; large genuinely-smooth regions
    # (e.g. a curved cover) stay whole and become freeform for NURBS drape.
    patches = _grow_smooth(fn, list(range(len(fn))), neighbors, grow_angle_deg)

    regions: list[Region] = []
    rid = 0
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    for faces in patches:
        pts, norms = _patch_points(V, F, fn, faces)
        kind, params, resid = _fit_best(pts, norms, thr)
        regions.append(Region(
            id=rid, kind=kind, face_ids=sorted(int(f) for f in faces),
            params=params or {},
            residual_mm_mean=(float(resid) if resid is not None else None),
        ))
        rid += 1

    graph = {"segmenter": "region_growing", "n_patches": len(patches),
             "grow_angle_deg": grow_angle_deg}
    return Segmentation(regions=regions, graph=graph, face_count=int(len(F)))


# ----------------------------------------------------------------------------
# Growth.
# ----------------------------------------------------------------------------

def _grow_smooth(fn, faces, neighbors, grow_angle_deg):
    """Grow smooth patches over `faces` by local adjacent-normal continuity,
    cutting at sharp edges. Restricted to the given face subset."""
    cos_tol = np.cos(np.radians(grow_angle_deg))
    subset = set(int(f) for f in faces)
    seen: set[int] = set()
    patches: list[list[int]] = []
    for start in faces:
        start = int(start)
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            nf = fn[f]
            for nb in neighbors[f]:
                if nb in subset and nb not in seen and \
                        float(np.dot(nf, fn[nb])) >= cos_tol:
                    seen.add(nb)
                    stack.append(nb)
        patches.append(comp)
    return patches


def _patch_points(V, F, fn, faces):
    faces = np.asarray(faces)
    tri = F[faces]
    # vertices of the patch triangles + centroids as the fitting point set
    verts = V[tri].reshape(-1, 3)
    cents = V[tri].mean(axis=1)
    pts = np.vstack([verts, cents])
    norms = np.repeat(fn[faces], 4, axis=0)  # 3 verts + 1 centroid per face
    return pts, norms


# ----------------------------------------------------------------------------
# Per-patch primitive fitting. One shot, best of {plane, cylinder, sphere}.
# ----------------------------------------------------------------------------

# Mild preference for simpler surfaces so a higher-DOF primitive (torus) only
# wins when it fits clearly better, not on a tie — avoids a torus stealing a
# cylinder/sphere it merely approximates.
_COMPLEXITY = {"plane": 1.0, "cylinder": 1.05, "cone": 1.1,
               "sphere": 1.1, "torus": 1.2}


def _fit_best(pts, norms, thr):
    diag = thr["bbox_diagonal_mm"]
    candidates = []
    p = _fit_plane(pts)
    if p is not None:
        candidates.append(("plane", p[0], p[1], thr["plane_mm"]))
    c = _fit_cylinder(pts, norms)
    if c is not None:
        candidates.append(("cylinder", c[0], c[1], thr["cylinder_mm"]))
    s = _fit_sphere(pts)
    if s is not None:
        candidates.append(("sphere", s[0], s[1], thr["sphere_mm"]))
    co = _fit_cone(pts, norms)
    if co is not None:
        candidates.append(("cone", co[0], co[1], thr["cylinder_mm"]))
    t = _fit_torus(pts, diag)
    if t is not None:
        candidates.append(("torus", t[0], t[1], thr["torus_mm"]))

    best = None
    for kind, params, resid, tol in candidates:
        if resid > tol:
            continue
        score = (resid / tol) * _COMPLEXITY[kind]
        if best is None or score < best[0]:
            best = (score, kind, params, resid)
    if best is None:
        return "freeform", None, None
    return best[1], best[2], best[3]


def _fit_plane(pts):
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    normal = vt[-1]
    d = -float(normal @ c)
    resid = float(np.abs((pts - c) @ normal).mean())
    return {
        "normal": normal.tolist(),
        "offset_mm": d,
        "raw_equation": [normal[0], normal[1], normal[2], d],
    }, resid


def _fit_cylinder(pts, norms):
    # axis = direction most orthogonal to all surface normals (smallest s.v.)
    nn = norms / np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-12)
    u, s, vt = np.linalg.svd(nn, full_matrices=False)
    axis = vt[-1]
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    # build a basis perpendicular to axis
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, tmp); e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(axis, e1)
    x = pts @ e1
    y = pts @ e2
    # algebraic (Kasa) circle fit in the axis-perpendicular plane
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
    radius = float(np.sqrt(r2))
    if radius <= 1e-6 or radius > 1e5:
        return None
    # center point on the axis (place at mean axial coord)
    axial = pts @ axis
    center = cx * e1 + cy * e2 + float(axial.mean()) * axis
    # residual
    v = pts - center
    perp = v - (v @ axis)[:, None] * axis
    resid = float(np.abs(np.linalg.norm(perp, axis=1) - radius).mean())
    return {
        "center_mm": center.tolist(),
        "axis": axis.tolist(),
        "radius_mm": radius,
    }, resid


def _axis_basis(axis):
    tmp = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, tmp); e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(axis, e1)
    return e1, e2


def _circle_fit(x, y):
    """Kasa algebraic circle fit -> (cx, cy, R) or None."""
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
    return float(cx), float(cy), float(np.sqrt(r2))


def _fit_torus(pts, diag):
    """Fit a ring torus (bearing races, revolved fillets). For each axis
    candidate, project points onto the axis-perpendicular plane and circle-fit
    the ring to locate the true centre + major radius (the centroid is offset
    for a one-sided groove, which was giving wrong tori); minor radius is the
    mean distance to that major circle."""
    c0 = pts.mean(axis=0)
    axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    try:
        _, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
        axes += [vt[0], vt[1], vt[2]]
    except np.linalg.LinAlgError:
        pass
    best = None
    for axis in axes:
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        e1, e2 = _axis_basis(axis)
        x = pts @ e1; y = pts @ e2; t = pts @ axis
        fit = _circle_fit(x, y)
        if fit is None:
            continue
        cx, cy, R = fit
        if R <= 1e-6 or R > 2.0 * diag:
            continue
        tc = float(t.mean())
        rho = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dist_to_ring = np.sqrt((rho - R) ** 2 + (t - tc) ** 2)
        r = float(dist_to_ring.mean())
        if r <= 1e-6 or R <= r * 1.05:
            continue
        resid = float(np.abs(dist_to_ring - r).mean())
        if best is None or resid < best[0]:
            center = cx * e1 + cy * e2 + tc * axis
            best = (resid, axis, center, R, r)
    if best is None:
        return None
    resid, axis, center, R, r = best
    return {"center_mm": center.tolist(), "axis": axis.tolist(),
            "major_radius_mm": float(R), "minor_radius_mm": float(r)}, resid


def _fit_cone(pts, norms):
    """Fit a cone: its surface normals make a constant angle with the axis, so
    the axis is the small-eigenvalue direction of the normal covariance and
    sin(half-angle) = |axis . mean_normal|. Reject near-cylinder (angle ~ 0) and
    near-plane (angle ~ 90) so it doesn't poach those types."""
    nn = norms / np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-12)
    nbar = nn.mean(axis=0)
    cov = (nn - nbar).T @ (nn - nbar)
    try:
        w, V = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    axis = V[:, 0]  # smallest-variance direction of normals
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    d = float(np.clip(np.abs(axis @ nbar), 0.0, 1.0))
    half = np.arcsin(d)
    if half < np.radians(8) or half > np.radians(82):
        return None  # ~cylinder or ~plane; let those types handle it
    # radius(t) = (t - t_apex) * tan(half); fit apex axial position + ref radius
    t = pts @ axis
    perp = pts - t[:, None] * axis
    # axis line position: centroid of perp components
    axis_pt = perp.mean(axis=0)
    rho = np.linalg.norm(perp - axis_pt, axis=1)
    tan = np.tan(half)
    # least-squares apex param t0:  rho ≈ (t - t0)*tan  -> t0 = t - rho/tan (avg)
    t0 = float(np.mean(t - rho / max(tan, 1e-9)))
    resid = float(np.abs(rho - (t - t0) * tan).mean())
    ref_t = float(t.mean())
    ref_radius = (ref_t - t0) * tan
    if ref_radius <= 1e-6:
        return None
    apex = axis_pt + t0 * axis
    ref_pt = axis_pt + ref_t * axis
    return {"apex_mm": apex.tolist(), "ref_point_mm": ref_pt.tolist(),
            "axis": axis.tolist(), "semi_angle_rad": float(half),
            "ref_radius_mm": float(ref_radius)}, resid


def _fit_sphere(pts):
    # algebraic sphere fit: x^2+y^2+z^2 = 2cx x + 2cy y + 2cz z + k
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    A = np.column_stack([2 * x, 2 * y, 2 * z, np.ones_like(x)])
    b = x * x + y * y + z * z
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = sol[:3]
    r2 = sol[3] + center @ center
    if r2 <= 0:
        return None
    radius = float(np.sqrt(r2))
    if radius <= 1e-6 or radius > 1e5:
        return None
    resid = float(np.abs(np.linalg.norm(pts - center, axis=1) - radius).mean())
    return {"center_mm": center.tolist(), "radius_mm": radius}, resid

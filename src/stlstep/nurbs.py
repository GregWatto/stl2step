"""NURBS drape — fit a smooth B-spline surface to a region that no analytic
primitive captures (a genuinely curved body, or an analytic fit the deviation
guard rejected).

The region is treated as a height field over its best-fit plane: project the
vertices to a (u, v) frame, interpolate their height onto a regular grid, and
fit a bicubic B-spline through the grid with `GeomAPI_PointsToBSplineSurface`.
The caller then merges the region's triangles onto this surface (same machinery
as analytic regions) so a many-facet freeform patch collapses to one smooth
face. A region that folds over (not a height field) fits poorly and is caught by
the region-match guard downstream, so it falls back to faceting.
"""

from __future__ import annotations

import numpy as np

from OCP.TColgp import TColgp_Array2OfPnt
from OCP.gp import gp_Pnt
from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCP.GeomAbs import GeomAbs_C2


def fit_bspline_surface(mesh, region, *, grid=24):
    """Fit a B-spline surface to a region (height field over its PCA plane).
    Returns a Geom_BSplineSurface, or None if it can't be built."""
    from scipy.interpolate import griddata

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    vids = np.unique(F[np.asarray(region.face_ids)].reshape(-1))
    pts = V[vids]
    if len(pts) < 9:
        return None

    origin = pts.mean(axis=0)
    # PCA frame: e1,e2 span the patch, n is the (thin) height direction.
    u, s, vt = np.linalg.svd(pts - origin, full_matrices=False)
    e1, e2, n = vt[0], vt[1], vt[2]

    # Height-field pre-check: a single B-spline patch only represents a region
    # that is thin along n (a gentle bump/fillet). A curved wall or a surface
    # that wraps (battery-cover body, a 90 deg fillet) has a large n-extent, is
    # not single-valued over the plane, and would fit poorly — bail fast rather
    # than spend minutes projecting triangles onto a bad surface. Those need
    # multi-patch surfacing or an analytic fix, not this.
    if s[2] > 0.15 * s[0]:
        return None

    a = (pts - origin) @ e1
    b = (pts - origin) @ e2
    h = (pts - origin) @ n

    amin, amax = a.min(), a.max()
    bmin, bmax = b.min(), b.max()
    if amax - amin < 1e-9 or bmax - bmin < 1e-9:
        return None
    ga = np.linspace(amin, amax, grid)
    gb = np.linspace(bmin, bmax, grid)
    GA, GB = np.meshgrid(ga, gb, indexing="ij")
    GH = griddata((a, b), h, (GA, GB), method="linear")
    # fill the grid corners outside the sample hull with nearest heights
    nan = np.isnan(GH)
    if nan.any():
        GH[nan] = griddata((a, b), h, (GA[nan], GB[nan]), method="nearest")
    if np.isnan(GH).any():
        return None

    arr = TColgp_Array2OfPnt(1, grid, 1, grid)
    for i in range(grid):
        for j in range(grid):
            p = origin + GA[i, j] * e1 + GB[i, j] * e2 + GH[i, j] * n
            arr.SetValue(i + 1, j + 1,
                         gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    try:
        maker = GeomAPI_PointsToBSplineSurface(arr, 3, 8, GeomAbs_C2, 1e-3)
        if not maker.IsDone():
            return None
        return maker.Surface()
    except Exception:  # noqa: BLE001
        return None

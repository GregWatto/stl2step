"""primitive_detect.py

Sample surface points from a (repaired) mesh, run iterative RANSAC via
pyransac3d, and produce a primitive graph — the input to Step 4 (topology
extraction).

Coverage is the headline metric: what fraction of mesh faces got assigned
to a detected primitive (plane / cylinder / sphere)? Mean residual per
primitive tells us fit tightness.

Library usage:
    from pipeline.primitive_detect import detect_primitives
    graph = detect_primitives(mesh, ...)

CLI:
    .venv/bin/python scripts/pipeline/primitive_detect.py \\
        <in.stl> <out_dir> [--samples 10000]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pyransac3d as pyrsc
import trimesh

from ._mesh_repair import repair_mesh  # noqa: E402


SCHEMA_VERSION = "primitive-graph-v1"

# ----------------------------------------------------------------------------
# Sampling.
# ----------------------------------------------------------------------------

def sample_surface_with_face_ids(mesh: trimesh.Trimesh, n_samples: int,
                                 rng: np.random.Generator):
    """Return (points, face_ids). Deterministic for a given rng seed.
    Seeds numpy and Python `random` — the latter is used by pyransac3d
    internally, so without it RANSAC is nondeterministic run-to-run."""
    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed)
    random.seed(seed)
    points, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    return np.asarray(points), np.asarray(face_ids)


# ----------------------------------------------------------------------------
# Thresholds. Scale-adaptive so we don't fail on a 500 mm part or over-fit
# a 5 mm one.
# ----------------------------------------------------------------------------

def _bbox_diagonal(mesh: trimesh.Trimesh) -> float:
    ext = mesh.bounds[1] - mesh.bounds[0]
    return float(np.linalg.norm(ext))


def compute_thresholds(mesh: trimesh.Trimesh) -> dict:
    diag = _bbox_diagonal(mesh)
    return {
        "bbox_diagonal_mm": diag,
        # Plane fits are tight (STL facets ARE plane samples).
        "plane_mm": max(0.05, diag * 1e-4),
        # Cylinders/spheres need a wider band because faceted meshes
        # approximate the curve with flat triangles — samples can be up
        # to r*(1 - cos(pi/N)) away from the true surface. For a typical
        # 32-64 sided cylinder discretization that's ~0.5% of radius.
        "cylinder_mm": max(0.20, diag * 5e-4),
        "sphere_mm": max(0.20, diag * 5e-4),
        # Tori (fillets on curved edges) have the same discretization
        # penalty as cylinders on the minor radius direction.
        "torus_mm": max(0.20, diag * 5e-4),
    }


# ----------------------------------------------------------------------------
# Residual computations. Distance from a point to a fitted primitive.
# ----------------------------------------------------------------------------

def _plane_residuals(pts: np.ndarray, eq) -> np.ndarray:
    a, b, c, d = [float(v) for v in eq]
    denom = np.sqrt(a * a + b * b + c * c)
    return np.abs(pts @ np.array([a, b, c]) + d) / max(denom, 1e-12)


def _cylinder_residuals(pts: np.ndarray, center, axis, radius) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    v = pts - center
    perp = v - (v @ axis)[:, None] * axis
    return np.abs(np.linalg.norm(perp, axis=1) - float(radius))


def _sphere_residuals(pts: np.ndarray, center, radius) -> np.ndarray:
    center = np.asarray(center, dtype=float).reshape(3)
    return np.abs(np.linalg.norm(pts - center, axis=1) - float(radius))


def _torus_residuals(pts: np.ndarray, center, axis, R, r) -> np.ndarray:
    """Distance from each point to the surface of a torus with the given
    center (any point on the axis), axis direction, major radius R, and
    minor (tube) radius r.

    For a point P, let t = axial coordinate, rho = radial distance from
    axis. Then the point is on the torus iff (rho - R)^2 + t^2 = r^2.
    """
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    v = pts - center
    t = v @ axis
    u = v - t[:, None] * axis
    rho = np.linalg.norm(u, axis=1)
    return np.abs(np.sqrt((rho - float(R)) ** 2 + t ** 2) - float(r))


# ----------------------------------------------------------------------------
# Primitive surface: residuals and outward-normals at arbitrary points.
# Used by the face-adjacency growing pass.
# ----------------------------------------------------------------------------

def _primitive_residuals_at(prim: dict, points: np.ndarray) -> np.ndarray:
    p = prim["params"]
    t = prim["type"]
    if t == "plane":
        return _plane_residuals(points, p["raw_equation"])
    if t == "cylinder":
        return _cylinder_residuals(points, p["center_mm"], p["axis"],
                                   p["radius_mm"])
    if t == "sphere":
        return _sphere_residuals(points, p["center_mm"], p["radius_mm"])
    if t == "torus":
        return _torus_residuals(points, p["center_mm"], p["axis"],
                                p["major_radius_mm"], p["minor_radius_mm"])
    raise ValueError(f"unknown primitive type: {t}")


def _primitive_normal_at(prim: dict, points: np.ndarray) -> np.ndarray:
    """Outward-pointing surface normal of `prim` at each row of `points`.
    Sign is arbitrary — callers use |n_face · n_prim|."""
    p = prim["params"]
    t = prim["type"]
    if t == "plane":
        n = np.asarray(p["normal"], dtype=float)
        return np.broadcast_to(n, points.shape).copy()
    if t == "cylinder":
        c = np.asarray(p["center_mm"], dtype=float)
        a = np.asarray(p["axis"], dtype=float)
        a = a / max(np.linalg.norm(a), 1e-12)
        v = points - c
        perp = v - (v @ a)[:, None] * a
        norms = np.linalg.norm(perp, axis=1)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return perp / norms[:, None]
    if t == "sphere":
        c = np.asarray(p["center_mm"], dtype=float)
        v = points - c
        norms = np.linalg.norm(v, axis=1)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return v / norms[:, None]
    if t == "torus":
        c = np.asarray(p["center_mm"], dtype=float)
        a = np.asarray(p["axis"], dtype=float)
        a = a / max(np.linalg.norm(a), 1e-12)
        R = float(p["major_radius_mm"])
        v = points - c
        t_axial = v @ a
        u = v - t_axial[:, None] * a
        rho = np.linalg.norm(u, axis=1)
        rho_safe = np.where(rho > 1e-12, rho, 1.0)
        ring = c + (u / rho_safe[:, None]) * R
        v_ring = points - ring
        n = np.linalg.norm(v_ring, axis=1)
        n_safe = np.where(n > 1e-12, n, 1.0)
        return v_ring / n_safe[:, None]
    raise ValueError(f"unknown primitive type: {t}")


# Growing thresholds are scale-adaptive: a face centroid can be up to
# ~half the local edge length off a curved primitive even when the face
# genuinely belongs to it. 5x the RANSAC threshold plus a bbox-floor
# handles both fine-detail parts (0.05 mm plane threshold) and coarse
# assemblies without either over-claiming or under-claiming.
GROW_DIST_FACTOR = 5.0
GROW_MIN_NORMAL_DOT = 0.85


def grow_face_adjacency(mesh: trimesh.Trimesh,
                        face_assignment: dict[int, int],
                        primitives: list[dict],
                        thresholds: dict,
                        min_normal_dot: float = GROW_MIN_NORMAL_DOT
                        ) -> tuple[dict[int, int], int, int]:
    """Grow face_assignment outward: unassigned faces adjacent to assigned
    ones get claimed if their geometry matches the neighbor's primitive.

    Returns (updated_assignment, n_grown, n_iterations).
    """
    if not primitives or not face_assignment:
        return dict(face_assignment), 0, 0

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return dict(face_assignment), 0, 0

    n_faces = int(len(mesh.faces))
    neighbors: list[list[int]] = [[] for _ in range(n_faces)]
    for a, b in adjacency:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))

    centroids = np.asarray(mesh.vertices[mesh.faces].mean(axis=1),
                           dtype=float)
    face_normals = np.asarray(mesh.face_normals, dtype=float)

    diag = float(thresholds["bbox_diagonal_mm"])
    dist_floor = diag * 5e-4
    grow_thresh = {
        "plane": max(thresholds["plane_mm"] * GROW_DIST_FACTOR, dist_floor),
        "cylinder": max(thresholds["cylinder_mm"] * GROW_DIST_FACTOR,
                        dist_floor),
        "torus": max(thresholds["torus_mm"] * GROW_DIST_FACTOR, dist_floor),
        "sphere": max(thresholds["sphere_mm"] * GROW_DIST_FACTOR, dist_floor),
    }

    assigned: dict[int, int] = dict(face_assignment)
    n_grown_total = 0
    n_iter = 0
    # Cap iterations defensively — the frontier can only advance a finite
    # number of face-hops, but bad geometry could in principle oscillate
    # around numerical ties.
    max_iter = 64

    # Incremental frontier maintenance: `frontier[fid]` = set of candidate
    # primitives for the unassigned face `fid`. Rebuilding this from
    # scratch each iteration is O(|assigned|) and dominates runtime on
    # large meshes; instead we only update entries touched by
    # newly-assigned faces.
    frontier: dict[int, set[int]] = {}
    for fid, prim_i in assigned.items():
        for nb in neighbors[fid]:
            if nb in assigned:
                continue
            frontier.setdefault(nb, set()).add(prim_i)

    while n_iter < max_iter and frontier:
        n_iter += 1
        pair_fids: list[int] = []
        pair_pids: list[int] = []
        for fid, prim_set in frontier.items():
            for prim_i in prim_set:
                pair_fids.append(fid)
                pair_pids.append(prim_i)
        if not pair_fids:
            break

        fids = np.asarray(pair_fids, dtype=np.int64)
        pids = np.asarray(pair_pids, dtype=np.int64)
        n_pairs = len(fids)
        dists = np.full(n_pairs, np.inf)
        dots = np.zeros(n_pairs)
        thresh_per_pair = np.zeros(n_pairs)

        # Vectorize per primitive: gather rows for one prim, compute
        # residuals + normals in one shot, scatter back.
        for prim_i in np.unique(pids):
            prim = primitives[int(prim_i)]
            row_idx = np.where(pids == prim_i)[0]
            pts = centroids[fids[row_idx]]
            dists[row_idx] = _primitive_residuals_at(prim, pts)
            pn = _primitive_normal_at(prim, pts)
            fn = face_normals[fids[row_idx]]
            dots[row_idx] = np.abs(np.sum(fn * pn, axis=1))
            thresh_per_pair[row_idx] = grow_thresh.get(prim["type"], np.inf)

        ok = (dists <= thresh_per_pair) & (dots >= min_normal_dot)
        if not ok.any():
            break
        ok_fids = fids[ok]
        ok_pids = pids[ok]
        ok_dists = dists[ok]

        # Per-face min-dist pick: sort by (face_id, dist), keep the first
        # row of each face_id group.
        order = np.lexsort((ok_dists, ok_fids))
        sorted_fids = ok_fids[order]
        sorted_pids = ok_pids[order]
        first = np.concatenate((
            [True], np.diff(sorted_fids) != 0
        ))
        grown_fids = sorted_fids[first]
        grown_pids = sorted_pids[first]

        newly_assigned: list[tuple[int, int]] = []
        for fid, prim_i in zip(grown_fids.tolist(), grown_pids.tolist()):
            assigned[int(fid)] = int(prim_i)
            newly_assigned.append((int(fid), int(prim_i)))
        n_grown_total += int(len(grown_fids))

        # Advance the frontier: drop faces we just assigned, and add
        # their still-unassigned neighbors with the appropriate
        # candidate primitive.
        for fid, _ in newly_assigned:
            frontier.pop(fid, None)
        for fid, prim_i in newly_assigned:
            for nb in neighbors[fid]:
                if nb in assigned:
                    continue
                frontier.setdefault(nb, set()).add(prim_i)

    return assigned, n_grown_total, n_iter


# ----------------------------------------------------------------------------
# Per-primitive fit wrappers. Each returns a normalized dict or None.
# ----------------------------------------------------------------------------

def _fit_plane(pts: np.ndarray, thresh: float, min_inliers: int,
               max_iter: int) -> dict | None:
    try:
        eq, _inliers = pyrsc.Plane().fit(pts, thresh=thresh,
                                         minPoints=min_inliers,
                                         maxIteration=max_iter)
    except Exception:
        return None
    if eq is None or len(eq) < 4:
        return None
    # Grow: recruit every point within threshold, not just the ones
    # RANSAC happened to sample. This turns "found the plane" into
    # "found all inliers of the plane."
    all_residuals = _plane_residuals(pts, eq)
    inliers = np.where(all_residuals <= thresh)[0]
    if len(inliers) < min_inliers:
        return None
    residuals = all_residuals[inliers]
    # Quality gate: legit planar surfaces fit *tight*. A "plane" whose
    # residuals push near the threshold is almost always a small tangent
    # patch of a curved surface (cylinder wall / fillet). Reject those so
    # cylinders get their turn on the same points.
    if residuals.mean() > thresh * 0.3:
        return None
    a, b, c, d = [float(v) for v in eq]
    n = np.array([a, b, c])
    n_norm = float(np.linalg.norm(n)) or 1.0
    return {
        "type": "plane",
        "params": {
            "normal": (n / n_norm).tolist(),
            "offset_mm": d / n_norm,
            "raw_equation": [a, b, c, d],
        },
        "inlier_indices": np.asarray(inliers, dtype=np.int64),
        "residual_mm_mean": float(residuals.mean()),
        "residual_mm_p95": float(np.quantile(residuals, 0.95)),
    }


def _fit_cylinder(pts: np.ndarray, thresh: float, min_inliers: int,
                  max_iter: int, max_radius_mm: float) -> dict | None:
    try:
        center, axis, radius, _inliers = pyrsc.Cylinder().fit(
            pts, thresh=thresh, maxIteration=max_iter
        )
    except Exception:
        return None
    if radius is None or center is None or axis is None:
        return None
    if float(radius) > max_radius_mm or float(radius) <= 0:
        return None
    # Grow: pyransac3d's cylinder RANSAC often stops at a partial fit;
    # recruit all points within threshold using the fitted parameters.
    all_residuals = _cylinder_residuals(pts, center, axis, radius)
    inliers = np.where(all_residuals <= thresh)[0]
    if len(inliers) < min_inliers:
        return None
    residuals = all_residuals[inliers]
    if residuals.mean() > thresh * 0.5:
        return None
    return {
        "type": "cylinder",
        "params": {
            "center_mm": np.asarray(center, dtype=float).tolist(),
            "axis": np.asarray(axis, dtype=float).tolist(),
            "radius_mm": float(radius),
        },
        "inlier_indices": np.asarray(inliers, dtype=np.int64),
        "residual_mm_mean": float(residuals.mean()),
        "residual_mm_p95": float(np.quantile(residuals, 0.95)),
    }


def _fit_sphere(pts: np.ndarray, thresh: float, min_inliers: int,
                max_iter: int, max_radius_mm: float) -> dict | None:
    try:
        center, radius, _inliers = pyrsc.Sphere().fit(
            pts, thresh=thresh, maxIteration=max_iter
        )
    except Exception:
        return None
    if center is None or radius is None:
        return None
    if float(radius) > max_radius_mm or float(radius) <= 0:
        return None
    all_residuals = _sphere_residuals(pts, center, radius)
    inliers = np.where(all_residuals <= thresh)[0]
    if len(inliers) < min_inliers:
        return None
    residuals = all_residuals[inliers]
    # Spheres are the most frequent false positive (small patches of
    # cylinders/fillets locally look spherical). Tighter quality gate.
    if residuals.mean() > thresh * 0.3:
        return None
    return {
        "type": "sphere",
        "params": {
            "center_mm": list(map(float, center)),
            "radius_mm": float(radius),
        },
        "inlier_indices": np.asarray(inliers, dtype=np.int64),
        "residual_mm_mean": float(residuals.mean()),
        "residual_mm_p95": float(np.quantile(residuals, 0.95)),
    }


def _torus_lsq_Rr(pts: np.ndarray, center: np.ndarray,
                  axis: np.ndarray) -> tuple[float, float] | None:
    """Given an axis direction and any point on that axis (or near it),
    solve for major radius R and minor radius r via linear least squares.

    Derivation: on a torus, (rho - R)^2 + t^2 = r^2, where rho is radial
    distance from axis and t is axial position. Expanding:
        rho^2 + t^2 = 2R * rho + (r^2 - R^2)
    which is linear in (a=2R, b=r^2-R^2). Robust to axial offset of the
    reference center; sensitive to perpendicular offset (kept small by
    using multiple center candidates).
    """
    v = pts - center
    t = v @ axis
    u = v - t[:, None] * axis
    rho = np.linalg.norm(u, axis=1)
    A = np.column_stack([rho, np.ones(len(rho))])
    y = rho * rho + t * t
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b = float(sol[0]), float(sol[1])
    R = a / 2.0
    r_sq = b + R * R
    if R <= 0 or r_sq <= 0:
        return None
    return R, float(np.sqrt(r_sq))


def _fit_torus(pts: np.ndarray, normals: np.ndarray,
               thresh: float, min_inliers: int, max_iter: int,
               max_radius_mm: float, rng: np.random.Generator) -> dict | None:
    """Torus fit by *axis-candidate search*.

    General 7-DoF torus RANSAC via PCA-of-normals doesn't work: for a
    local patch on a torus the smallest-variance normal direction is
    the patch's own outward direction, not the torus axis. Since the
    LSQ for (R, r) is very sensitive to axis error, that approach
    diverges.

    Instead, enumerate a small set of *plausible* axes for mechanical
    parts (world XYZ + point-cloud PCA principal axes), and for each
    axis try several center candidates. This targets axis-aligned
    fillets (the common case in Fusion-designed parts). It won't find
    tori at arbitrary angles — those are deferred.

    Kept: the normals argument for future-proofing (not used yet).
    """
    _ = normals
    n = len(pts)
    if n < min_inliers:
        return None

    # Axis candidates: world axes + PCA principal axes of positions.
    axes: list[np.ndarray] = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    v = pts - pts.mean(axis=0)
    try:
        _, evecs = np.linalg.eigh(v.T @ v)
        for i in range(3):
            axes.append(evecs[:, i] / max(np.linalg.norm(evecs[:, i]), 1e-12))
    except np.linalg.LinAlgError:
        pass

    # Center candidates per axis: mean, plus random sample points.
    n_centers = min(20, max(5, max_iter // 20))
    best = None

    for axis in axes:
        centers = [pts.mean(axis=0)]
        centers += [pts[int(rng.integers(0, n))] for _ in range(n_centers)]
        for center in centers:
            Rr = _torus_lsq_Rr(pts, center, axis)
            if Rr is None:
                continue
            R, r = Rr
            if R > max_radius_mm or r > max_radius_mm:
                continue
            if R < r * 1.05:  # ring torus only
                continue
            residuals = _torus_residuals(pts, center, axis, R, r)
            n_in = int((residuals <= thresh).sum())
            if n_in >= min_inliers and (best is None
                                        or n_in > best["n_in"]):
                best = {
                    "axis": axis, "center": center, "R": R, "r": r,
                    "residuals": residuals, "n_in": n_in,
                }

    if best is None:
        return None

    # Refit R and r with all inliers of the winning candidate.
    inliers0 = np.where(best["residuals"] <= thresh)[0]
    Rr = _torus_lsq_Rr(pts[inliers0], best["center"], best["axis"])
    if Rr is None:
        return None
    R_ref, r_ref = Rr
    if (R_ref > max_radius_mm or r_ref > max_radius_mm
            or R_ref < r_ref * 1.05):
        return None

    residuals = _torus_residuals(pts, best["center"], best["axis"],
                                 R_ref, r_ref)
    inliers = np.where(residuals <= thresh)[0]
    if len(inliers) < min_inliers:
        return None
    resid_in = residuals[inliers]
    if resid_in.mean() > thresh * 0.5:
        return None

    return {
        "type": "torus",
        "params": {
            "center_mm": best["center"].tolist(),
            "axis": best["axis"].tolist(),
            "major_radius_mm": R_ref,
            "minor_radius_mm": r_ref,
        },
        "inlier_indices": inliers,
        "residual_mm_mean": float(resid_in.mean()),
        "residual_mm_p95": float(np.quantile(resid_in, 0.95)),
    }


# ----------------------------------------------------------------------------
# Iterative detection loop.
# ----------------------------------------------------------------------------

def _peel(fit_fn, remaining_idx: np.ndarray, pts_all: np.ndarray,
          thresh: float, min_inliers: int, max_iter: int,
          max_primitives: int) -> tuple[list[dict], np.ndarray]:
    """Iteratively fit `fit_fn` on the current remaining subset; peel off
    each accepted primitive's inliers. Returns (primitives, still_remaining),
    where each primitive's `inlier_indices` is mapped back to the *global*
    pts_all indexing."""
    out: list[dict] = []
    working = remaining_idx.copy()
    while len(working) >= min_inliers and len(out) < max_primitives:
        prim = fit_fn(pts_all[working], thresh, min_inliers, max_iter)
        if prim is None:
            break
        local = np.asarray(prim["inlier_indices"], dtype=np.int64)
        prim["inlier_indices"] = working[local]
        out.append(prim)
        mask = np.ones(len(working), dtype=bool)
        mask[local] = False
        working = working[mask]
    return out, working


def detect_primitives(mesh: trimesh.Trimesh, *, n_samples: int = 10000,
                      seed: int = 0, min_inlier_frac: float = 0.005,
                      min_inlier_abs: int = 50,
                      max_primitives_per_type: int = 40,
                      grow_faces: bool = True) -> dict:
    """Run RANSAC on `mesh` and return a primitive graph dict.

    Sequential three-pass detection (plane → cylinder → sphere). Each
    fit has an "inlier grow" step: after RANSAC finds approximate
    parameters, all points within threshold are recruited as inliers
    (so cylinders with slow-converging RANSAC still get their full
    inlier set). Strict per-type quality gates prevent low-quality
    tangent fits from stealing curved-surface points before the right
    primitive gets its pass.
    """
    started = time.time()
    rng = np.random.default_rng(seed)

    thresholds = compute_thresholds(mesh)
    min_inliers = max(min_inlier_abs, int(n_samples * min_inlier_frac))
    diag = thresholds["bbox_diagonal_mm"]
    max_cyl_radius = diag * 2.0
    max_sph_radius = diag * 1.0

    pts, face_ids = sample_surface_with_face_ids(mesh, n_samples, rng)
    # Per-sample normals — needed for torus axis estimation (via PCA of
    # sample normals). Face normals are constant per triangle, which is
    # fine for the smooth curvature analysis torus fitting relies on.
    normals = np.asarray(mesh.face_normals[face_ids])
    n_pts = len(pts)

    remaining = np.arange(n_pts, dtype=np.int64)

    plane_prims, remaining = _peel(
        lambda p, t, m, i: _fit_plane(p, t, m, i),
        remaining, pts,
        thresholds["plane_mm"], min_inliers, 500,
        max_primitives_per_type,
    )

    # Torus pass runs *before* cylinder because cylinder RANSAC eagerly
    # fits patches of torus surfaces as bogus cylinders. Torus fitting
    # is fast (axis-candidate search + LSQ) so this is a cheap prepass.
    torus_prims: list[dict] = []
    if len(remaining) >= 2 * min_inliers:
        working = remaining.copy()
        while (len(working) >= min_inliers
               and len(torus_prims) < max_primitives_per_type):
            prim = _fit_torus(
                pts[working], normals[working],
                thresholds["torus_mm"], min_inliers,
                max_iter=200, max_radius_mm=diag * 1.0, rng=rng,
            )
            if prim is None:
                break
            local = np.asarray(prim["inlier_indices"], dtype=np.int64)
            prim["inlier_indices"] = working[local]
            torus_prims.append(prim)
            mask = np.ones(len(working), dtype=bool)
            mask[local] = False
            working = working[mask]
        remaining = working

    cyl_prims, remaining = _peel(
        lambda p, t, m, i: _fit_cylinder(p, t, m, i, max_cyl_radius),
        remaining, pts,
        thresholds["cylinder_mm"], min_inliers, 3000,
        max_primitives_per_type,
    )

    # Skip sphere pass if only fillet-sized remnants remain — sphere
    # RANSAC produces mostly false positives on small curved patches.
    if len(remaining) >= 5 * min_inliers:
        sph_prims, remaining = _peel(
            lambda p, t, m, i: _fit_sphere(p, t, m, i, max_sph_radius),
            remaining, pts,
            thresholds["sphere_mm"], min_inliers, 500,
            max_primitives_per_type,
        )
    else:
        sph_prims = []

    primitives = plane_prims + torus_prims + cyl_prims + sph_prims

    # Per-point primitive assignment (index into `primitives` or -1).
    point_assignment = -np.ones(n_pts, dtype=np.int64)
    for i, prim in enumerate(primitives):
        point_assignment[prim["inlier_indices"]] = i

    # Per-face majority vote. A face's primitive is whichever one wins
    # among the sample points that landed on it.
    face_assignment: dict[int, int] = {}
    face_bucket: dict[int, Counter] = {}
    for pt_i, face_id in enumerate(face_ids):
        prim_i = int(point_assignment[pt_i])
        if prim_i < 0:
            continue
        c = face_bucket.setdefault(int(face_id), Counter())
        c[prim_i] += 1
    for face_id, counter in face_bucket.items():
        prim_i, _ = counter.most_common(1)[0]
        face_assignment[face_id] = prim_i

    total_faces = int(len(mesh.faces))
    assigned_samples = int((point_assignment >= 0).sum())
    faces_from_samples = len(face_assignment)

    # Face-adjacency growing: recruit edge/tangent triangles that RANSAC's
    # sample points missed but whose geometry matches an adjacent
    # primitive. Improves coverage on parts where the primitives are
    # correct but small unclaimed strips of faces bleed the score.
    grow_stats = {"grown": 0, "iterations": 0}
    if grow_faces and primitives:
        grown_assignment, n_grown, n_iter = grow_face_adjacency(
            mesh, face_assignment, primitives, thresholds
        )
        face_assignment = grown_assignment
        grow_stats = {"grown": int(n_grown), "iterations": int(n_iter)}

    prim_to_faces: dict[int, list[int]] = {i: [] for i in range(len(primitives))}
    for face_id, prim_i in face_assignment.items():
        prim_to_faces[prim_i].append(face_id)

    assigned_faces = len(face_assignment)

    # Emit primitive records (drop the fat inlier_indices arrays; keep
    # inlier_count and face lists).
    prim_records = []
    for i, prim in enumerate(primitives):
        faces = sorted(prim_to_faces[i])
        prim_records.append({
            "id": i,
            "type": prim["type"],
            "params": prim["params"],
            "inlier_count": int(len(prim["inlier_indices"])),
            "residual_mm_mean": prim["residual_mm_mean"],
            "residual_mm_p95": prim["residual_mm_p95"],
            "face_count": len(faces),
            "face_ids": faces,
        })

    unassigned_faces = sorted(
        set(range(total_faces)) - set(face_assignment.keys())
    )

    graph = {
        "schema_version": SCHEMA_VERSION,
        "sampling": {"n_samples": n_pts, "seed": seed},
        "thresholds": thresholds,
        "mesh_stats": {
            "face_count": total_faces,
            "vertex_count": int(len(mesh.vertices)),
            "is_watertight": bool(mesh.is_watertight),
            "body_count": int(mesh.body_count),
        },
        "primitives": prim_records,
        "counts_by_type": {
            "plane": sum(1 for p in prim_records if p["type"] == "plane"),
            "cylinder": sum(1 for p in prim_records if p["type"] == "cylinder"),
            "torus": sum(1 for p in prim_records if p["type"] == "torus"),
            "sphere": sum(1 for p in prim_records if p["type"] == "sphere"),
        },
        "unassigned_face_ids": unassigned_faces,
        "unassigned_face_count": len(unassigned_faces),
        "unassigned_sample_count": n_pts - assigned_samples,
        "face_coverage_pct": (assigned_faces / total_faces * 100.0
                              if total_faces else 0.0),
        "face_coverage_pct_pre_grow": (faces_from_samples / total_faces * 100.0
                                       if total_faces else 0.0),
        "face_grow_stats": grow_stats,
        "sample_coverage_pct": (assigned_samples / n_pts * 100.0
                                if n_pts else 0.0),
        "residual_mm_median_over_primitives": float(np.median(
            [p["residual_mm_mean"] for p in prim_records]
        )) if prim_records else None,
        "elapsed_s": time.time() - started,
    }
    return graph


# ----------------------------------------------------------------------------
# eval.py pipeline adapter.
# ----------------------------------------------------------------------------

def pipeline_primitive_detect(in_path: str, out_dir: str,
                              design_id: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    mesh, repair_report = repair_mesh(in_path)
    graph = detect_primitives(mesh)
    graph_path = os.path.join(out_dir, f"{design_id}.primitive_graph.json")
    with open(graph_path, "w") as fp:
        json.dump(graph, fp, indent=2, default=str)
    return {
        "output_mesh_path": None,
        "step_path": None,
        "primitive_graph_path": graph_path,
        "notes": (f"{len(graph['primitives'])} primitives, "
                  f"{graph['face_coverage_pct']:.1f}% face coverage, "
                  f"repair.became_watertight={repair_report['became_watertight']}"),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Detect primitives in an STL.")
    p.add_argument("in_path")
    p.add_argument("out_dir")
    p.add_argument("--id", default=None,
                   help="Output basename. Defaults to input stem.")
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    design_id = args.id or os.path.splitext(os.path.basename(args.in_path))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    mesh, _repair_report = repair_mesh(args.in_path)
    graph = detect_primitives(mesh, n_samples=args.samples, seed=args.seed)
    out_path = os.path.join(args.out_dir, f"{design_id}.primitive_graph.json")
    with open(out_path, "w") as fp:
        json.dump(graph, fp, indent=2, default=str)

    print(f"wrote {out_path}")
    print(f"  primitives: plane={graph['counts_by_type']['plane']} "
          f"cyl={graph['counts_by_type']['cylinder']} "
          f"sph={graph['counts_by_type']['sphere']}")
    print(f"  face_coverage_pct: {graph['face_coverage_pct']:.1f}")
    print(f"  sample_coverage_pct: {graph['sample_coverage_pct']:.1f}")
    if graph['residual_mm_median_over_primitives'] is not None:
        print(f"  residual_mm median: "
              f"{graph['residual_mm_median_over_primitives']:.4f}")
    print(f"  elapsed_s: {graph['elapsed_s']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

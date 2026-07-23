"""Feature grouping — turn the flat surface list into named, composable features.

`analyze()` produces a soup of surfaces (13 cylinders, 29 planes, ...). The
composer wants FEATURES: "a counterbored hole (Ø6 through, Ø10 c'bore 4 deep),
four of them at these positions", not thirteen anonymous cylinders. This module
groups the cylinder surfaces into holes / counterbores / bosses, detects
patterns (linear / circular / rectangular), and flags degenerate analytic fits
(the giant sphere on diff_case) so the composer stops trusting them.

Grouping is geometric and robust to the segmenter splitting or duplicating a
cylinder: coaxial cylinders (same axis line) merge; same-diameter stages merge by
axial span; different diameters become counterbore stages; equal-signature holes
with parallel axes become a pattern. Pockets (plane walls + floor) are not grouped
yet — noted for the composer to read off the cross-sections.
"""

from __future__ import annotations

import numpy as np


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _axis_key(a):
    """Canonical axis direction (sign-normalised so the dominant component is
    positive) rounded — coaxial/parallel cylinders share it."""
    a = _unit(a)
    if a[int(np.argmax(np.abs(a)))] < 0:
        a = -a
    return tuple(np.round(a, 2))


def _perp_key(point, axis):
    """The axis LINE's identity: a point's component perpendicular to the axis is
    the same for every point on the line, so it keys coaxial cylinders together."""
    p = np.asarray(point, float)
    return tuple(np.round(p - (p @ axis) * axis, 1))


def group_features(mesh, surfaces, bbox_size) -> dict:
    diag = float(np.linalg.norm(bbox_size))
    cyls = []
    for s in surfaces:
        if s["kind"] != "cylinder":
            continue
        axis = _unit(s["axis"])
        cyls.append({"dia": float(s["diameter_mm"]), "axis": axis,
                     "ea": np.asarray(s["end_a_mm"], float),
                     "eb": np.asarray(s["end_b_mm"], float),
                     "orient": s["orientation"]})

    # 1) bucket cylinders by axis line (coaxial) and hole-vs-boss
    buckets: dict = {}
    for c in cyls:
        akey = _axis_key(c["axis"])
        mid = 0.5 * (c["ea"] + c["eb"])
        key = (akey, _perp_key(mid, c["axis"]),
               "hole" if "hole" in c["orient"] else
               "boss" if "boss" in c["orient"] else "?")
        buckets.setdefault(key, []).append(c)

    holes, bosses = [], []
    for (akey, _pk, kind), cs in buckets.items():
        axis = _unit(np.asarray(akey, float))
        loc = _collapse_coaxial(cs, axis, mesh)
        (holes if kind == "hole" else bosses if kind == "boss" else holes).append(loc)

    # 2) merge equal-signature holes/bosses into patterns
    features = {
        "holes": _patternize(holes),
        "bosses": _patternize(bosses),
        "low_confidence_fits": _degenerate(surfaces, diag),
        "notes": ["pockets/slots are not grouped yet — read them off the "
                  "cross-sections and the plane list"],
    }
    return features


def _collapse_coaxial(cs, axis, mesh) -> dict:
    """One axis line's cylinders -> a hole/boss with counterbore stages."""
    stages_by_dia: dict = {}
    for c in cs:
        d = round(c["dia"], 1)
        ta, tb = float(c["ea"] @ axis), float(c["eb"] @ axis)
        lo, hi = min(ta, tb), max(ta, tb)
        s = stages_by_dia.setdefault(d, [lo, hi])
        s[0], s[1] = min(s[0], lo), max(s[1], hi)
    stages = sorted(({"dia": d, "t0": t[0], "t1": t[1]}
                     for d, t in stages_by_dia.items()), key=lambda s: s["t0"])
    t_lo = min(s["t0"] for s in stages)
    t_hi = max(s["t1"] for s in stages)
    ref = cs[0]["ea"]
    foot = ref - (ref @ axis) * axis                 # axis line anchor
    at = foot + t_hi * axis                          # a reference point (hi end)
    feat = {"axis": [round(float(x), 3) for x in axis],
            "depth_mm": round(t_hi - t_lo, 3),
            "diameter_mm": round(min(s["dia"] for s in stages), 3),
            "through": _through(foot, axis, t_lo, t_hi, mesh),
            "_at": (foot + t_lo * axis, at)}          # (lo end, hi end) world pts
    if len(stages) >= 2:
        wide = max(stages, key=lambda s: s["dia"])
        feat["counterbore"] = {"diameter_mm": round(wide["dia"], 3),
                               "depth_mm": round(wide["t1"] - wide["t0"], 3)}
    return feat


def _through(foot, axis, t_lo, t_hi, mesh) -> bool | None:
    """Both ends exit the solid? Needs a watertight mesh; else None (unknown)."""
    if not mesh.is_watertight:
        return None
    eps = 0.4
    p_hi = foot + (t_hi + eps) * axis
    p_lo = foot + (t_lo - eps) * axis
    try:
        inside = mesh.contains(np.array([p_hi, p_lo]))
    except Exception:  # noqa: BLE001
        return None
    return not bool(inside[0]) and not bool(inside[1])


def _patternize(locs: list[dict]) -> list[dict]:
    """Merge holes/bosses with identical signature into one feature carrying all
    positions + an arrangement label (single/linear/circular/rectangular/group)."""
    groups: dict = {}
    for f in locs:
        sig = (tuple(f["axis"]), round(f["diameter_mm"], 1), f.get("through"),
               round(f.get("counterbore", {}).get("diameter_mm", 0), 1))
        groups.setdefault(sig, []).append(f)

    out = []
    for sig, fs in groups.items():
        axis = _unit(np.asarray(sig[0], float))
        pts = np.array([f["_at"][1] for f in fs])      # hi-end reference points
        base = dict(fs[0]); base.pop("_at", None)
        base["count"] = len(fs)
        base["positions_mm"] = [[round(float(x), 2) for x in p] for p in pts]
        base["arrangement"], extra = _arrangement(pts, axis)
        base.update(extra)
        out.append(base)
    out.sort(key=lambda f: -f["count"])
    return out


def _arrangement(pts, axis):
    """Classify a set of parallel-axis positions in the plane perpendicular to
    the axis."""
    if len(pts) == 1:
        return "single", {}
    e1 = _unit(np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0]))
    e2 = np.cross(axis, e1)
    uv = np.column_stack([pts @ e1, pts @ e2])
    c = uv.mean(axis=0)
    r = np.linalg.norm(uv - c, axis=1)
    if len(pts) >= 3 and r.mean() > 1e-6 and r.std() / r.mean() < 0.06:
        ang = np.sort(np.arctan2(uv[:, 1] - c[1], uv[:, 0] - c[0]))
        gaps = np.diff(np.append(ang, ang[0] + 2 * np.pi))
        if gaps.std() < np.radians(8):               # equal angular spacing
            center3 = pts.mean(axis=0) - ((pts.mean(axis=0)) @ axis) * axis
            return "circular", {"circle": {
                "radius_mm": round(float(r.mean()), 3),
                "center_mm": [round(float(x), 2) for x in center3]}}
        return "rectangular", {}                     # equal radius, unequal angle
    # collinear?
    d = uv - c
    if len(pts) >= 2:
        _, _, vt = np.linalg.svd(d)
        perp = np.abs(d @ vt[1])
        span = np.ptp(d @ vt[0])
        if span > 1e-6 and perp.max() < 0.05 * span:
            return "linear", {}
    return "group", {}


def _degenerate(surfaces, diag) -> list[dict]:
    """Analytic fits too big to be real features — a sphere/torus with radius on
    the order of the whole part is almost always a mis-fit of a flat or gently
    curved region. Flag them so the composer trusts the sections instead."""
    out = []
    for s in surfaces:
        if s["kind"] == "sphere" and float(s["radius_mm"]) > 0.4 * diag:
            out.append({"kind": "sphere", "radius_mm": s["radius_mm"],
                        "reason": "radius ~ part size; likely a flat/curved mis-fit"})
        elif s["kind"] == "torus" and float(s["major_radius_mm"]) > 0.4 * diag:
            out.append({"kind": "torus", "major_radius_mm": s["major_radius_mm"],
                        "reason": "major radius ~ part size; likely a mis-fit"})
    return out

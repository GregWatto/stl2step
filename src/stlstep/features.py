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


def _pockets(mesh, seg, diag) -> list[dict]:
    """Recessed enclosed cavities. A pocket needs three things a step or a solid
    block lacks: a FLOOR (recessed planar face), an OPENING (a larger parallel
    face further along the floor's outward normal), and VOID between them
    (verified by a cross-section — robust without a watertight mesh), enclosed by
    perpendicular WALLS on 3+ sides. Prismatic (planar-walled) pockets only; the
    composer reads the exact outline off the sections."""
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
    fn = np.asarray(mesh.face_normals)
    part_size = np.asarray(mesh.bounds[1]) - np.asarray(mesh.bounds[0])
    P = []
    for r in seg.regions:
        if r.kind != "plane" or not r.face_ids:
            continue
        n = _unit(fn[r.face_ids].mean(axis=0))       # OUTWARD (seg normals have arbitrary sign)
        pts = V[F[r.face_ids]].reshape(-1, 3)
        P.append({"n": n, "off": float(np.median(pts @ n)), "pts": pts,
                  "cen": pts.mean(axis=0)})

    out = []
    for fl in P:
        n = fl["n"]
        if np.max(np.abs(n)) < 0.9:                  # principal-axis floors only
            continue
        f = fl["off"]
        e1 = _unit(np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]))
        e2 = np.cross(n, e1)
        fu, fv = fl["pts"] @ e1, fl["pts"] @ e2
        ub, vb = (fu.min(), fu.max()), (fv.min(), fv.max())
        mu, mv = 0.1 * (ub[1] - ub[0]) + 1e-6, 0.1 * (vb[1] - vb[0]) + 1e-6
        # walls: perpendicular faces sitting on the floor (min coord ~ f), near
        # the footprint — they rise from the floor to the opening (the rim).
        walls_top, sides = [], set()
        for w in P:
            if abs(float(w["n"] @ n)) > 0.25:
                continue
            wt = w["pts"] @ n
            if abs(float(wt.min()) - f) > 0.06 * diag or float(wt.max()) <= f + 0.02 * diag:
                continue
            wu, wv = float(np.mean(w["pts"] @ e1)), float(np.mean(w["pts"] @ e2))
            if (ub[0] - mu <= wu <= ub[1] + mu) and (vb[0] - mv <= wv <= vb[1] + mv):
                walls_top.append(float(wt.max()))
                sides.add(tuple(np.round(w["n"], 1)))
        if len(sides) < 3 or not walls_top:
            continue
        opening = float(np.median(walls_top))
        depth = opening - f
        # a floor much narrower than it is deep is really a WALL read side-on.
        if depth < 0.02 * diag or min(ub[1] - ub[0], vb[1] - vb[0]) < 0.5 * depth:
            continue
        # the opening must be at the part's outer surface along +n OVER the
        # footprint (a mis-read side wall's "opening" is an internal wall — the
        # part continues past it; a real pocket opens to the outside). Local, not
        # global, so a tall boss elsewhere doesn't disqualify a shallow pocket.
        ua, va = V @ e1, V @ e2
        mask = ((ua >= ub[0] - mu) & (ua <= ub[1] + mu)
                & (va >= vb[0] - mv) & (va <= vb[1] + mv))
        local_max = float((V[mask] @ n).max()) if mask.any() else opening
        if opening < local_max - 0.05 * diag:
            continue
        # ...and the space must be void mid-depth AND right up at the opening.
        if not (_is_void(mesh, fl["pts"], n, f + 0.5 * depth)
                and _is_void(mesh, fl["pts"], n, opening - min(0.5, 0.2 * depth))):
            continue
        # in-plane centre of the footprint, lifted to the floor level in 3D — so the pocket is
        # PLACEABLE (a size without a centre can't be emitted). e1,e2 span the floor; n is depth.
        cu, cv = 0.5 * (ub[0] + ub[1]), 0.5 * (vb[0] + vb[1])
        centre = cu * e1 + cv * e2 + f * n
        out.append({"opening_normal": [round(float(x), 3) for x in n],
                    "opening_level_mm": round(opening, 2),
                    "floor_level_mm": round(float(f), 2),
                    "depth_mm": round(depth, 2),
                    "size_mm": [round(float(ub[1] - ub[0]), 1),
                                round(float(vb[1] - vb[0]), 1)],
                    "center_mm": [round(float(x), 2) for x in centre],
                    "plane_axes": [[round(float(x), 4) for x in e1],
                                   [round(float(x), 4) for x in e2]]})
    return out


def _is_void(mesh, floor_pts, n, level) -> bool:
    """True if the space over the floor at `level` (along n) is mostly NOT inside
    material — the pocket void, vs a solid block or an open step read as solid.
    Cross-section + point-in-polygon over MANY floor points (a single centroid can
    fluke into a hole); robust to non-watertight meshes."""
    import trimesh
    from shapely.geometry import Point
    try:
        sec = mesh.section(plane_origin=(n * level).tolist(), plane_normal=n.tolist())
        if sec is None:
            return True                               # nothing there -> void
        p2d, to_3d = sec.to_2D()
        to_2d = np.linalg.inv(to_3d)
        idx = np.linspace(0, len(floor_pts) - 1, min(20, len(floor_pts))).astype(int)
        samp = floor_pts[idx]
        samp = samp.mean(axis=0) + 0.6 * (samp - samp.mean(axis=0))   # shrink off walls
        pts2 = trimesh.transformations.transform_points(samp, to_2d)[:, :2]
        inside = sum(any(poly.contains(Point(p)) for poly in p2d.polygons_full)
                     for p in pts2)
        return inside < 0.5 * len(pts2)               # majority not in material
    except Exception:  # noqa: BLE001
        return False


def group_features(mesh, seg, surfaces, bbox_size) -> dict:
    diag = float(np.linalg.norm(bbox_size))
    cyls = []
    for s in surfaces:
        if s["kind"] != "cylinder":
            continue
        axis = _unit(s["axis"])
        cyls.append({"dia": float(s["diameter_mm"]), "axis": axis,
                     "ea": np.asarray(s["end_a_mm"], float),
                     "eb": np.asarray(s["end_b_mm"], float),
                     "area": float(s.get("area_mm2", 0) or 0),
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

    holes, bosses, micro = [], [], []
    for (akey, _pk, kind), cs in buckets.items():
        axis = _unit(np.asarray(akey, float))
        loc = _collapse_coaxial(cs, axis, mesh)
        # fragment fits: tiny/shallow cylinders, or fits whose segmented area
        # covers only a sliver of their own cylinder surface, are fragments of
        # helical/noisy walls — not real features. They'd pollute the brief with
        # dozens of junk holes; divert them to the thread-candidate pool: where
        # they CLUSTER is where a thread (or knurl) lives.
        cov_area = sum(c["area"] for c in cs)
        full_area = np.pi * loc["diameter_mm"] * max(loc["depth_mm"], 1e-6)
        if (loc["diameter_mm"] < 1.0 or loc["depth_mm"] < 1.2
                or cov_area < 0.3 * full_area):
            micro.append(loc)
            continue
        try:
            sig = thread_signature(
                mesh, np.asarray(loc["axis"], float), loc["_at"][0],
                t_range=(-max(4.0, 0.75 * loc["depth_mm"]),
                         loc["depth_mm"] + max(4.0, 0.75 * loc["depth_mm"])),
                r_hint=0.5 * loc["diameter_mm"],
                internal=(kind == "hole"))
            if sig is not None:
                loc["threaded"] = sig
        except Exception:  # noqa: BLE001 — thread check must not break grouping
            pass
        (holes if kind == "hole" else bosses if kind == "boss" else holes).append(loc)

    try:
        pockets = _pockets(mesh, seg, diag)
    except Exception:  # noqa: BLE001 — pockets are best-effort
        pockets = []

    # 2) merge equal-signature holes/bosses into patterns
    grouped_holes = _patternize(holes)
    # 3) section-loop pass: recover small round holes segmentation missed
    #    (thin-wall holes have too few triangles to fit a cylinder region, but
    #    are unmissable as circular interior rings in cross-sections)
    try:
        grouped_holes.extend(_section_loop_holes(mesh, grouped_holes))
    except Exception:  # noqa: BLE001 — recall pass is best-effort
        pass
    # 4) open-bore pass: recover SLOTTED bores / rod saddles that a thin exterior
    #    slot hides from the closed-ring section pass (a y-rod holder's Ø8 channel)
    saddle_notes = []
    try:
        bores, saddle_notes = _open_bore_holes(mesh, grouped_holes, diag)
        grouped_holes.extend(bores)
    except Exception:  # noqa: BLE001 — recall pass is best-effort
        pass
    features = {
        "holes": grouped_holes,
        "bosses": _patternize(bosses),
        "pockets": pockets,
        "thread_candidates": _thread_candidates(micro, diag),
        "low_confidence_fits": _degenerate(surfaces, diag),
        "notes": ["round/oblique pockets and slots aren't grouped — read those "
                  "off the cross-sections and the plane list"] + saddle_notes,
    }
    return features


def _thread_candidates(micro: list[dict], diag: float) -> list[dict]:
    """Spatial clusters of micro cylinder fits = likely thread (or knurl) zones.
    A helical wall shatters into dozens of tiny cylinder fragments under
    segmentation; individually they're junk, but where they cluster is exactly
    where a thread lives. Emit the cluster centre + extent so the agent runs
    `probe thread` there."""
    if len(micro) < 3:
        return []
    pts = np.array([0.5 * (m["_at"][0] + m["_at"][1]) for m in micro])
    dias = np.array([m["diameter_mm"] for m in micro])
    # generous link radius: fragments of one thread scatter across its whole
    # bore diameter + length; over-merging HINTS is harmless (they're pointers
    # for `probe thread`, not features)
    link = max(20.0, 0.12 * diag)
    labels = -np.ones(len(pts), dtype=int)
    nl = 0
    for i in range(len(pts)):
        if labels[i] >= 0:
            continue
        stack = [i]; labels[i] = nl
        while stack:
            j = stack.pop()
            near = np.where((labels < 0) &
                            (np.linalg.norm(pts - pts[j], axis=1) < link))[0]
            labels[near] = nl
            stack.extend(near.tolist())
        nl += 1
    out = []
    for k in range(nl):
        m = labels == k
        # >=5 fragments, or >=3 that agree on diameter (crest slivers of one
        # thread share the crest Ø — strong evidence even in a small cluster)
        same_dia = (m.sum() >= 3 and dias[m].mean() > 1e-6
                    and dias[m].std() < 0.1 * dias[m].mean())
        if m.sum() < 5 and not same_dia:
            continue
        p = pts[m]
        out.append({"center_mm": [round(float(x), 1) for x in p.mean(axis=0)],
                    "extent_mm": [round(float(x), 1)
                                  for x in (p.max(axis=0) - p.min(axis=0))],
                    "fragments": int(m.sum()),
                    "note": "cluster of tiny cylinder fragments — likely a "
                            "THREAD (or knurl): run `probe thread` here"})
    out.sort(key=lambda c: -c["fragments"])
    return out[:4]


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


# ----------------------------------------------------------------------------
# Thread identification.
#
# A thread in a mesh is a helical wall: the radius about the axis oscillates
# between minor and major with period = pitch. The INTENT, though, is a spec
# (M10x1.5) — real CAD models threads as a spec'd feature on a nominal hole/
# stud, not as helical geometry. So the job here is identification: measure
# depth (r spread in the wall band), pitch (axial spacing of crest passes in
# fixed angular sectors — robust, no unwrapping), and taper (pipe threads),
# then match against standard tables. Modelling policy lives in the skill:
# nominal representation + recorded spec.
# ----------------------------------------------------------------------------

# (nominal D mm, pitch mm, name) — ISO metric coarse + common fine.
ISO_THREADS = [
    (2.0, 0.4, "M2"), (2.5, 0.45, "M2.5"), (3.0, 0.5, "M3"), (4.0, 0.7, "M4"),
    (5.0, 0.8, "M5"), (6.0, 1.0, "M6"), (8.0, 1.25, "M8"), (8.0, 1.0, "M8x1"),
    (10.0, 1.5, "M10"), (10.0, 1.25, "M10x1.25"), (10.0, 1.0, "M10x1"),
    (12.0, 1.75, "M12"), (12.0, 1.5, "M12x1.5"), (12.0, 1.25, "M12x1.25"),
    (14.0, 2.0, "M14"), (14.0, 1.5, "M14x1.5"), (16.0, 2.0, "M16"),
    (16.0, 1.5, "M16x1.5"), (20.0, 2.5, "M20"), (24.0, 3.0, "M24"),
]
# (major D mm, pitch mm, name) — a few common inch threads (UNC/UNF).
INCH_THREADS = [
    (6.35, 1.270, '1/4-20 UNC'), (6.35, 0.907, '1/4-28 UNF'),
    (7.94, 1.411, '5/16-18 UNC'), (9.53, 1.588, '3/8-16 UNC'),
    (12.70, 1.954, '1/2-13 UNC'), (12.70, 1.270, '1/2-20 UNF'),
]


def thread_signature(mesh, axis_dir, axis_pt, *, t_range=None, r_hint=None,
                     internal=None):
    """Measure a thread signature on a cylindrical wall band. Returns None if
    the wall reads as plain (no consistent helical oscillation), else a dict
    with measured minor/major diameter, pitch, taper, and the closest standard
    spec (or measured-only if nothing matches)."""
    d = np.asarray(axis_dir, dtype=float); d /= np.linalg.norm(d)
    p0 = np.asarray(axis_pt, dtype=float)
    V = np.asarray(mesh.vertices)
    rel = V - p0
    t = rel @ d
    radial = rel - t[:, None] * d[None]
    r = np.linalg.norm(radial, axis=1)

    sel = np.ones(len(V), dtype=bool)
    if t_range is not None:
        sel &= (t >= t_range[0]) & (t <= t_range[1])
    if r_hint is not None:
        sel &= np.abs(r - r_hint) < max(1.5, 0.25 * r_hint)
    else:
        sel &= r > 1e-6
    if sel.sum() < 120:
        return None
    rb, tb = r[sel], t[sel]
    # wall band: mode of the radius histogram, expanded while populated — keeps
    # the thread band, drops chamfers/counterbores at other radii
    binw = 0.15
    bins = np.arange(rb.min(), rb.max() + binw, binw)
    hist, edges = np.histogram(rb, bins=bins)
    k = int(np.argmax(hist))
    # steep cutoff (0.35 of peak): the thread band is a dense plateau; adjacent
    # features (a counterbore wall, a recess) show as separate spikes that a
    # looser threshold would wrongly merge into the band
    lo_k, hi_k = k, k
    while lo_k > 0 and hist[lo_k - 1] > 0.35 * hist[k]:
        lo_k -= 1
    while hi_k < len(hist) - 1 and hist[hi_k + 1] > 0.35 * hist[k]:
        hi_k += 1
    wall = (rb >= edges[lo_k]) & (rb < edges[hi_k + 1])
    if wall.sum() < 120:
        return None
    rw, tw = rb[wall], tb[wall]
    rad_w = radial[sel][wall]
    th = np.arctan2(rad_w[:, 1], rad_w[:, 0])
    # robust 10/90 percentiles: entry chamfers / an adjacent recess pollute
    # the tails and would fake a too-deep thread
    r_lo, r_hi = float(np.percentile(rw, 10)), float(np.percentile(rw, 90))
    depth = r_hi - r_lo
    if depth < 0.15:
        return None                       # plain wall, no thread relief

    # pitch: within a fixed angular sector, crest passes sit at z spaced by
    # exactly P. Estimate the period per sector from the FIRST strong peak of
    # the pairwise z-difference histogram (robust for both dense grids and
    # sparse decimated meshes — no clustering or unwrapping needed).
    peaks, all_dd = [], []
    for kk in range(8):
        a0 = -np.pi + kk * np.pi / 4
        sec = (th >= a0) & (th < a0 + np.pi / 4) & (rw < r_lo + 0.25 * depth)
        zs = tw[sec]
        if len(zs) < 4:
            continue
        if len(zs) > 300:
            zs = np.random.default_rng(0).choice(zs, 300, replace=False)
        dd = np.abs(zs[:, None] - zs[None, :])[np.triu_indices(len(zs), 1)]
        dd = dd[(dd > 0.25) & (dd < 10.0)]
        if len(dd) < 6:
            continue
        all_dd.append(dd)
        hist, edges = np.histogram(dd, bins=np.arange(0.25, 10.0, 0.05))
        if hist.max() < 3:
            continue
        idx = np.where(hist >= 0.6 * hist.max())[0]
        first = int(idx[0])                     # fundamental, not a multiple
        lo_b, hi_b = max(0, first - 2), first + 3
        w = hist[lo_b:hi_b].astype(float)
        c = 0.5 * (edges[lo_b:hi_b] + edges[lo_b + 1:hi_b + 1])
        peaks.append(float((w * c).sum() / w.sum()))
    if len(peaks) < 3:      # tolerate partial circumferential exposure
        return None
    peaks = np.asarray(peaks)
    p0 = float(np.median(peaks))
    spread = float(np.percentile(peaks, 75) - np.percentile(peaks, 25))
    if p0 <= 0 or spread > 0.2 * p0:
        return None                       # sectors disagree -> not a helix
    # refine with the harmonics: the pairwise-difference histogram peaks at
    # every multiple of P; the centroid at k*P divided by k averages away the
    # single-peak bias, giving pitch to ~1-2%
    dd = np.concatenate(all_dd)
    refined = []
    for kmul in (1, 2, 3, 4):
        w = dd[np.abs(dd - kmul * p0) < 0.3 * p0]
        if len(w) >= 6:
            refined.append(float(np.mean(w)) / kmul)
    pitch = float(np.median(refined)) if refined else p0
    # sanity: ISO thread depth ~ 0.54..0.65 * P
    if not (0.15 * pitch < depth < 1.4 * pitch):
        return None

    # taper (pipe threads): slope of the outer envelope along the axis
    hi_band = rw > r_hi - 0.3 * depth
    taper_deg = 0.0
    if hi_band.sum() > 10 and np.ptp(tw[hi_band]) > 1e-6:
        slope = float(np.polyfit(tw[hi_band], rw[hi_band], 1)[0])
        taper_deg = float(np.degrees(np.arctan(slope)))

    major = 2 * r_hi
    out = {"minor_diameter_mm": round(2 * r_lo, 3),
           "major_diameter_mm": round(major, 3),
           "pitch_mm": round(pitch, 3),
           "depth_mm": round(depth, 3),
           "taper_deg": round(taper_deg, 2),
           "internal": internal,
           "spec": None}
    if abs(taper_deg) > 1.0:
        out["spec_note"] = ("tapered — likely a PIPE thread (NPT/BSPT); "
                            "match by hand")
        return out
    best = None
    for D, P, name in ISO_THREADS + INCH_THREADS:
        if abs(P - pitch) > 0.12 * P:
            continue
        dD = abs(D - major)
        if dD < 0.6 and (best is None or dD < best[0]):
            best = (dD, D, P, name)
    if best:
        out["spec"] = best[3]
        out["spec_major_mm"] = best[1]
        out["spec_pitch_mm"] = best[2]
    return out


# ----------------------------------------------------------------------------
# Section-loop hole detection — recall independent of segmentation.
#
# A small round hole through a thin wall has a few dozen triangles; region
# growing routinely absorbs or fragments it, so the cylinder-based grouping
# above never sees it — and the feature gate can only demand what was detected
# (an enclosure with 3 such holes shipped with them SQUARED because of exactly
# this). But the same hole is unmissable in cross-section: a circular interior
# ring of the material region. Slice densely along all three axes, circle-fit
# every interior ring, cluster across stations, and emit the results as hole
# features — arming the brief AND the feature gate.
# ----------------------------------------------------------------------------

def _section_loop_holes(mesh, existing_holes, *, n=56) -> list[dict]:
    from shapely.geometry import Polygon
    from .draw import _circle_fit_2d
    cands = []
    for axis in (0, 1, 2):
        lo = float(mesh.bounds[0][axis]); hi = float(mesh.bounds[1][axis])
        if hi - lo <= 1e-6:
            continue
        u, v = [i for i in (0, 1, 2) if i != axis]
        from .featurecheck import _slice_region
        for c in np.linspace(lo + 1e-3, hi - 1e-3, n):
            region = _slice_region(mesh, axis, float(c))
            if region is None:
                continue
            for g in getattr(region, "geoms", [region]):
                for ring in getattr(g, "interiors", []):
                    pts = np.asarray(ring.coords)[:, :2]
                    if len(pts) < 8:
                        continue
                    try:
                        area = abs(Polygon(ring).area)
                    except Exception:  # noqa: BLE001
                        continue
                    if area < 1.0:
                        continue
                    cf = _circle_fit_2d(pts)
                    if cf is None:
                        continue
                    cx, cy, r, resid = cf
                    b = pts.min(axis=0), pts.max(axis=0)
                    w, h = b[1] - b[0]
                    aspect = max(w, h) / max(min(w, h), 1e-6)
                    # strict circularity: tight fit, square bbox, Ø ≈ bbox,
                    # area ≈ πr² — a rectangular vent fails all of these
                    if (resid > 0.05 * r or aspect > 1.2
                            or abs(2 * r - max(w, h)) > 0.3 * max(w, h)
                            or abs(area - np.pi * r * r) > 0.2 * np.pi * r * r):
                        continue
                    cands.append({"axis": axis, "t": float(c),
                                  "cu": float(cx), "cv": float(cy),
                                  "r": float(r), "u": u, "v": v})
    # cluster candidates: same axis, same centre (0.8mm), same r (12%)
    holes = []
    used = [False] * len(cands)
    for i, a in enumerate(cands):
        if used[i]:
            continue
        grp = [a]; used[i] = True
        for j in range(i + 1, len(cands)):
            bnd = cands[j]
            if used[j] or bnd["axis"] != a["axis"]:
                continue
            if (abs(bnd["cu"] - a["cu"]) < 0.8 and abs(bnd["cv"] - a["cv"]) < 0.8
                    and abs(bnd["r"] - a["r"]) < 0.12 * max(a["r"], 0.5)):
                grp.append(bnd); used[j] = True
        r = float(np.median([g["r"] for g in grp]))
        cu = float(np.median([g["cu"] for g in grp]))
        cv = float(np.median([g["cv"] for g in grp]))
        ts = [g["t"] for g in grp]
        pos = [0.0, 0.0, 0.0]
        pos[a["axis"]] = round(max(ts), 2)
        pos[a["u"]], pos[a["v"]] = round(cu, 2), round(cv, 2)
        axis_vec = [0.0, 0.0, 0.0]; axis_vec[a["axis"]] = 1.0
        holes.append({"axis": axis_vec, "diameter_mm": round(2 * r, 2),
                      "depth_mm": round(max(max(ts) - min(ts), 0.5), 2),
                      "through": None, "count": 1, "positions_mm": [pos],
                      "arrangement": "single", "source": "section-loop"})
    # dedupe vs cylinder-derived holes (same axis line + similar Ø)
    out = []
    for hnew in holes:
        dup = False
        for h in existing_holes:
            if abs(h["diameter_mm"] - hnew["diameter_mm"]) > 0.3 * max(
                    h["diameter_mm"], hnew["diameter_mm"]):
                continue
            if abs(float(np.dot(h["axis"], hnew["axis"]))) < 0.9:
                continue
            for p in h.get("positions_mm", []):
                d = np.linalg.norm(np.asarray(p) - np.asarray(
                    hnew["positions_mm"][0]))
                if d < max(2.0, h["diameter_mm"]):
                    dup = True
        if not dup:
            out.append(hnew)
    # merge equal-signature section holes into patterns
    merged = {}
    for h in out:
        key = (tuple(h["axis"]), round(h["diameter_mm"], 1))
        if key in merged:
            m = merged[key]
            m["count"] += 1
            m["positions_mm"].append(h["positions_mm"][0])
            m["arrangement"] = "group"
        else:
            merged[key] = h
    return list(merged.values())


def _robust_circle_fit(pts, iters=4):
    """Iterative inlier-trimmed circle fit — rejects the straight side walls of a
    saddle/slot so the fit locks onto the true circular arc. Returns
    (cx, cy, r, resid, inlier_mask) or None."""
    from .draw import _circle_fit_2d
    keep = np.ones(len(pts), bool)
    cf = _circle_fit_2d(pts)
    for _ in range(iters):
        if cf is None or keep.sum() < 6:
            return None if cf is None else (*cf, keep)
        cx, cy, r, _ = cf
        err = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        thr = max(0.12 * r, float(np.median(err)) * 1.5)
        nk = err < thr
        if nk.sum() < 6 or nk.sum() == keep.sum():
            keep = nk
            break
        keep = nk
        cf = _circle_fit_2d(pts[keep])
    return None if cf is None else (*cf, keep)


def _axial_void_extent(mesh, axis, cu, cv, lo, hi, *, n=60):
    """Longest contiguous void run along the line through (cu, cv) parallel to
    `axis`, by point-in-solid testing. Returns (t_lo, t_hi), or (None, None) if
    the containment test is unavailable (non-watertight mesh)."""
    try:
        ts = np.linspace(lo + 0.2, hi - 0.2, n)
        pts = np.zeros((n, 3))
        u, v = [i for i in (0, 1, 2) if i != axis]
        pts[:, u] = cu; pts[:, v] = cv; pts[:, axis] = ts
        void = ~mesh.contains(pts)
        if not void.any():
            return None, None
        # longest run of consecutive void samples
        best_lo = best_hi = None; best_len = 0; run_lo = None
        for i, isvoid in enumerate(void):
            if isvoid and run_lo is None:
                run_lo = i
            if (not isvoid or i == n - 1) and run_lo is not None:
                run_hi = i if not isvoid else i
                if ts[run_hi] - ts[run_lo] > best_len:
                    best_len = ts[run_hi] - ts[run_lo]
                    best_lo, best_hi = ts[run_lo], ts[run_hi]
                run_lo = None
        return best_lo, best_hi
    except Exception:  # noqa: BLE001 — containment can fail on open meshes
        return None, None


def _open_bore_holes(mesh, existing_holes, diag, *, n=48) -> tuple[list, list]:
    """Recall pass for SLOTTED / OPEN bores and rod saddles — the single biggest
    class of missed feature on real parts (e.g. a y-rod holder's Ø8 channel).

    `_section_loop_holes` only recovers CLOSED interior rings; a rod channel is a
    bore that a thin slot opens to the outside, so its cross-section is a concave
    arc in the OUTER boundary, not a closed ring — invisible to that pass. Here, in
    cross-sections ⟂ each axis, we find exterior concavities (``hull.difference`` of
    the section), keep only those whose void **connects to the exterior** (a real
    chord shared with the convex hull — this is the discriminator vs. a bounded
    counterbore/annulus, which `_section_loop_holes` already handles), and
    robust-fit a circle to the arc (side walls trimmed as outliers). Coaxial fits
    are clustered across slices.

    Returns ``(holes, notes)``:
      * holes — near-closed bores (arc span ≥ 330°): a slot broke the interior
        ring, so these are true holes the section-loop pass missed. Same schema +
        dedupe as `_section_loop_holes`; ``source="open-bore"``.
      * notes — medium-confidence open saddles (150°–330°): genuinely
        underspecified (half-pipe vs. fillet is ambiguous), left for the LLM to
        confirm and place, as free-text lines.
    """
    from shapely.geometry import Point, Polygon
    from .featurecheck import _slice_region
    cands = []
    for axis in (0, 1, 2):
        lo = float(mesh.bounds[0][axis]); hi = float(mesh.bounds[1][axis])
        if hi - lo <= 1e-6:
            continue
        u, v = [i for i in (0, 1, 2) if i != axis]
        for c in np.linspace(lo + 0.5, hi - 0.5, n):
            region = _slice_region(mesh, axis, float(c))
            if region is None:
                continue
            for g in getattr(region, "geoms", [region]):
                try:
                    poly = Polygon(g.exterior)
                    if poly.area < 40:
                        continue
                    hull = poly.convex_hull
                    hb = hull.exterior
                    defect = hull.difference(poly)
                except Exception:  # noqa: BLE001 — one bad slice must not kill the pass
                    continue
                for d in getattr(defect, "geoms", [defect]):
                    if d.is_empty or d.area < 4.0:
                        continue
                    # EXTERIOR-CONNECTED GUARD: the void must share a real chord
                    # with the hull. Interior holes (bounded voids) share none —
                    # they're the section-loop pass's job, not ours.
                    chord = d.exterior.intersection(hb).length
                    if chord < 0.8:
                        continue
                    bpts = np.asarray(d.exterior.coords)[:, :2]
                    arc = bpts[np.array([hb.distance(Point(*p)) > 0.15
                                         for p in bpts])]
                    if len(arc) < 8:
                        continue
                    rf = _robust_circle_fit(arc)
                    if rf is None:
                        continue
                    cx, cy, r, resid, keep = rf
                    if r < 1.5 or r > 0.45 * diag or resid > 0.05 * r:
                        continue
                    a = arc[keep]
                    span = float(np.degrees(np.ptp(np.unwrap(
                        np.sort(np.arctan2(a[:, 1] - cy, a[:, 0] - cx))))))
                    if span < 150 or chord > 1.4 * 2 * r:
                        continue
                    cands.append({"axis": axis, "t": float(c), "cu": float(cx),
                                  "cv": float(cy), "r": float(r), "span": span,
                                  "u": u, "v": v})
    # cluster coaxial candidates: same axis, centre (1.2mm), r (15%)
    holes, notes = [], []
    used = [False] * len(cands)
    for i, a in enumerate(cands):
        if used[i]:
            continue
        grp = [a]; used[i] = True
        for j in range(i + 1, len(cands)):
            b = cands[j]
            if used[j] or b["axis"] != a["axis"]:
                continue
            if (abs(b["cu"] - a["cu"]) < 1.2 and abs(b["cv"] - a["cv"]) < 1.2
                    and abs(b["r"] - a["r"]) < 0.15 * max(a["r"], 0.5)):
                grp.append(b); used[j] = True
        lo = float(mesh.bounds[0][a["axis"]]); hi = float(mesh.bounds[1][a["axis"]])
        ext = max(hi - lo, 1e-6)
        if len(grp) < max(3, int(0.08 * n)):
            continue
        r = float(np.median([g["r"] for g in grp]))
        cu = float(np.median([g["cu"] for g in grp]))
        cv = float(np.median([g["cv"] for g in grp]))
        max_span = max(g["span"] for g in grp)
        ts = [g["t"] for g in grp]
        run = (max(ts) - min(ts)) / ext
        axis_vec = [0.0, 0.0, 0.0]; axis_vec[a["axis"]] = 1.0
        pos = [0.0, 0.0, 0.0]
        pos[a["u"]], pos[a["v"]] = round(cu, 2), round(cv, 2)
        if max_span >= 330.0:
            # a near-closed bore: emit as a hole. The clean-fit t-span
            # under-measures depth (end-cap features contaminate the fit near the
            # faces), so probe the axis line for the TRUE void extent instead.
            v_lo, v_hi = _axial_void_extent(mesh, a["axis"], cu, cv, lo, hi)
            if v_hi is None:                       # probe failed -> fall back
                v_lo, v_hi = min(ts), max(ts)
            through = (v_hi - v_lo) >= 0.8 * (hi - lo)
            depth = max(v_hi - v_lo, 0.5)
            pos[a["axis"]] = round(hi if through else v_hi, 2)
            holes.append({"axis": axis_vec, "diameter_mm": round(2 * r, 2),
                          "depth_mm": round(depth, 2),
                          "through": (True if through else None), "count": 1,
                          "positions_mm": [pos], "arrangement": "single",
                          "source": "open-bore"})
        else:
            notes.append(
                f"OPEN SADDLE Ø{2 * r:.1f} at ({cu:.1f},{cv:.1f}) axis "
                f"{'xyz'[a['axis']]} (arc {max_span:.0f}°, ~{run * 100:.0f}% of "
                f"length) — half-open rod rest/scallop; confirm & place from a "
                f"probe section")
    # dedupe recovered bores vs cylinder- and section-derived holes
    out = []
    for hnew in holes:
        dup = False
        for h in existing_holes:
            if abs(h["diameter_mm"] - hnew["diameter_mm"]) > 0.3 * max(
                    h["diameter_mm"], hnew["diameter_mm"]):
                continue
            if abs(float(np.dot(h["axis"], hnew["axis"]))) < 0.9:
                continue
            for p in h.get("positions_mm", []):
                d = np.linalg.norm(np.asarray(p) - np.asarray(
                    hnew["positions_mm"][0]))
                if d < max(2.0, h["diameter_mm"]):
                    dup = True
        if not dup:
            out.append(hnew)
    return out, notes

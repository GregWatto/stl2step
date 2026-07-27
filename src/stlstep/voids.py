"""Void-structure analysis — dimension the INSIDE of the part.

Everything else describes the solid's surfaces; internal reconstruction keeps
failing because nobody hands the composer a dimensioned description of the
CAVITIES. This module treats the void as a first-class object: slice the part
densely along an axis, extract each station's enclosed void region (the outer
envelope minus the material — bores, chambers, annular rings), TRACK components
across stations by overlap, and emit per-cavity features: axial span, volume,
shape (circular / annular / profiled), centre, and radius profile. Circular and
annular cavities come with a ready-to-paste build123d cut. The atlas render
shows the internal silhouette (equivalent radius vs station) plus the slices
where the topology CHANGES — births, deaths, steps — which are exactly the
feature boundaries a modeller needs.

Robust to non-watertight meshes (sections only need local closure).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def _station_voids(mesh, axis, coord):
    """Enclosed void components at one station: fill the material region's outer
    rings to get the envelope, subtract the material. Returns a list of shapely
    polygons (each an enclosed cavity cross-section), or []."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from .featurecheck import _slice_region
    region = _slice_region(mesh, axis, coord)
    if region is None:
        return []
    try:
        envelope = unary_union([Polygon(g.exterior)
                                for g in getattr(region, "geoms", [region])
                                if hasattr(g, "exterior")])
        void = envelope.difference(region)
    except Exception:  # noqa: BLE001
        return []
    if void.is_empty:
        return []
    return [g for g in getattr(void, "geoms", [void])
            if hasattr(g, "exterior") and g.area > 0.15]


def analyze_voids(mesh, axis: int, *, t_range=None, n: int = 72) -> dict:
    """Dense void scan along `axis`: track cavity components across stations and
    dimension each as a feature. Returns {"features": [...], "stations": [...]}."""
    lo = float(mesh.bounds[0][axis]); hi = float(mesh.bounds[1][axis])
    if t_range is not None:
        lo, hi = max(lo, t_range[0]), min(hi, t_range[1])
    if hi - lo <= 1e-6:
        return {"features": [], "stations": []}
    coords = np.linspace(lo + 1e-3, hi - 1e-3, n)
    ds = (hi - lo) / n

    tracks: list[dict] = []          # live + finished component tracks
    live: list[dict] = []
    for c in coords:
        comps = _station_voids(mesh, axis, float(c))
        matched = [False] * len(comps)
        new_live = []
        for tr in live:
            best, best_a = None, 0.0
            for i, g in enumerate(comps):
                if matched[i]:
                    continue
                try:
                    a = tr["geom"].intersection(g).area
                except Exception:  # noqa: BLE001
                    a = 0.0
                if a > best_a:
                    best, best_a = i, a
            if best is not None and best_a > 0.2 * min(tr["geom"].area,
                                                       comps[best].area):
                g = comps[best]
                matched[best] = True
                tr["geom"] = g
                tr["stations"].append((float(c), g))
                new_live.append(tr)
            else:
                tracks.append(tr)     # died
        for i, g in enumerate(comps):
            if not matched[i]:
                tr = {"geom": g, "stations": [(float(c), g)]}
                new_live.append(tr)
        live = new_live
    tracks.extend(live)

    features = []
    for tr in tracks:
        # split at area STEPS (counterbore shoulders, orifice necks) so each
        # segment is ~constant-section — individually dimensionable and
        # buildable as one cut
        for sts in _split_steps(tr["stations"]):
            vol = sum(g.area for _, g in sts) * ds
            if vol < 5.0:
                continue             # dust
            features.append(_dimension_feature(sts, axis, ds, mesh))
    features.sort(key=lambda f: -f["volume_mm3"])
    return {"features": features, "axis": "xyz"[axis],
            "band_mm": [round(lo, 2), round(hi, 2)], "n_stations": n}


def _split_steps(sts, ratio: float = 1.35):
    """Split a tracked cavity at abrupt area changes between stations."""
    segs, cur = [], [sts[0]]
    for prev, nxt in zip(sts, sts[1:]):
        ra, rb = prev[1].area, nxt[1].area
        if max(ra, rb) / max(min(ra, rb), 1e-9) > ratio:
            segs.append(cur); cur = [nxt]
        else:
            cur.append(nxt)
    segs.append(cur)
    return [s for s in segs if len(s) >= 2]


def _circle_of(geom):
    """(cx, cy, r, resid) least-squares circle of a polygon's exterior."""
    from .draw import _circle_fit_2d
    pts = np.asarray(geom.exterior.coords)[:, :2]
    return _circle_fit_2d(pts)


def _dimension_feature(sts, axis, ds, mesh) -> dict:
    """Turn one tracked component into a dimensioned cavity feature."""
    names = "xyz"
    u, v = [i for i in (0, 1, 2) if i != axis]
    zs = [c for c, _ in sts]
    areas = np.array([g.area for _, g in sts])
    cents = np.array([[g.centroid.x, g.centroid.y] for _, g in sts])
    r_eq = np.sqrt(areas / np.pi)
    # shape at the reference (max-area) station
    ref = sts[int(np.argmax(areas))][1]
    shape, dims = "profiled", {}
    cfit = _circle_of(ref)
    islands = list(ref.interiors)
    # circle sanity: a lengthwise-cut bore (a long slot) "fits" a huge arc that
    # is not a circle. Require (a) near-square bbox, (b) fitted Ø ≈ bbox size,
    # (c) residual small vs the component's own equivalent radius.
    b = ref.bounds
    w, h = b[2] - b[0], b[3] - b[1]
    ref_span = max(w, h)
    aspect = ref_span / max(min(w, h), 1e-6)
    ref_req = float(np.sqrt(max(ref.area, 1e-9) / np.pi))
    if cfit is not None and (aspect > 1.7 or 2 * cfit[2] > 1.4 * ref_span
                             or cfit[3] > 0.1 * ref_req):
        cfit = None
    if cfit is not None and cfit[3] < 0.08 * max(cfit[2], 1e-6):
        if islands:
            from shapely.geometry import Polygon
            inner = max((Polygon(i) for i in islands), key=lambda p: p.area)
            icfit = _circle_of(inner)
            if icfit is not None and icfit[3] < 0.15 * max(icfit[2], 1e-6):
                shape = "annular"
                dims = {"outer_diameter_mm": round(2 * cfit[2], 2),
                        "inner_diameter_mm": round(2 * icfit[2], 2)}
            else:
                shape = "circular"
                dims = {"diameter_mm": round(2 * cfit[2], 2)}
        else:
            shape = "circular"
            dims = {"diameter_mm": round(2 * cfit[2], 2)}
    else:
        b = ref.bounds
        dims = {"bbox_uv_mm": [round(x, 1) for x in b]}
    # open at the part's axial faces, or sealed inside?
    lo_m = float(mesh.bounds[0][axis]); hi_m = float(mesh.bounds[1][axis])
    open_lo = zs[0] - lo_m < 2.5 * ds
    open_hi = hi_m - zs[-1] < 2.5 * ds
    drift = float(np.linalg.norm(cents[-1] - cents[0]))
    feat = {
        "axis": names[axis],
        "span_mm": [round(zs[0], 2), round(zs[-1], 2)],
        "length_mm": round(zs[-1] - zs[0], 2),
        "volume_mm3": round(float(areas.sum() * ds), 1),
        "shape": shape, **dims,
        "center_uv_mm": [round(float(cents[:, 0].mean()), 2),
                         round(float(cents[:, 1].mean()), 2)],
        "uv_axes": f"({names[u]},{names[v]})",
        "ends": ("through" if open_lo and open_hi else
                 "open one end" if open_lo or open_hi else "SEALED inside"),
        "r_eq_profile_mm": [round(float(r), 2)
                            for r in r_eq[::max(1, len(r_eq) // 8)]],
        "area_variation": round(float(areas.max() / max(areas.min(), 1e-6)), 2),
    }
    if drift > 1.0:
        feat["centroid_drift_mm"] = round(drift, 1)
        feat["note"] = "centroid drifts along the span — oblique or stepped"
    return feat


def cut_snippet(feat: dict) -> str | None:
    """Ready-to-paste build123d cut for a constant circular/annular cavity."""
    if feat["shape"] not in ("circular", "annular") or feat["area_variation"] > 1.3:
        return None
    plane = {"x": "YZ", "y": "XZ", "z": "XY"}[feat["axis"]]
    # build123d's Plane.XZ normal is -Y: negate offset + amount for the y axis
    sg = -1.0 if feat["axis"] == "y" else 1.0
    z0, z1 = feat["span_mm"]
    cu, cv = feat["center_uv_mm"]
    L = [f"    # cavity: {feat['shape']} void, {feat['ends']}, "
         f"axis {feat['axis']} {z0}..{z1}",
         f"    with BuildSketch(Plane.{plane}.offset({sg * z0})):",
         f"        with Locations(({cu}, {cv})):"]
    if feat["shape"] == "circular":
        L.append(f"            Circle({feat['diameter_mm'] / 2})")
    else:
        L.append(f"            Circle({feat['outer_diameter_mm'] / 2})")
        L.append(f"            Circle({feat['inner_diameter_mm'] / 2}, "
                 f"mode=Mode.SUBTRACT)")
    L.append(f"    extrude(amount={round(sg * (z1 - z0), 3)}, mode=Mode.SUBTRACT)")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# atlas render
# --------------------------------------------------------------------------- #
def render_void_atlas(mesh, axis: int, result: dict, path: str,
                      *, max_slices: int = 8) -> str | None:
    """Internal-structure atlas: the internal silhouette (each cavity's
    equivalent radius vs station — the 'drill profile' an engineer would draw)
    plus void-highlighted slices at the CHANGE points (component births/deaths/
    steps), where the feature boundaries are."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .featurecheck import _slice_region, _draw_region

    feats = result.get("features") or []
    if not feats:
        return None
    names = "xyz"
    # change stations: every feature's span ends + biggest area steps
    marks = set()
    for f in feats[:6]:
        z0, z1 = f["span_mm"]
        marks.add(round(z0 + 0.3, 2)); marks.add(round(z1 - 0.3, 2))
        marks.add(round(0.5 * (z0 + z1), 2))
    marks = sorted(marks)
    if len(marks) > max_slices:
        idx = np.linspace(0, len(marks) - 1, max_slices).astype(int)
        marks = [marks[i] for i in idx]

    rows = 1 + int(np.ceil(len(marks) / 4))
    fig = plt.figure(figsize=(13.5, 3.1 * rows))
    ax0 = fig.add_subplot(rows, 1, 1)
    for k, f in enumerate(feats[:6]):
        z0, z1 = f["span_mm"]
        zz = np.linspace(z0, z1, len(f["r_eq_profile_mm"]))
        lbl = (f"#{k+1} {f['shape']} "
               + (f"Ø{f.get('diameter_mm', f.get('outer_diameter_mm', '?'))}"
                  if f["shape"] != "profiled" else "profiled")
               + f" {f['ends']}")
        ax0.plot(zz, f["r_eq_profile_mm"], "-o", ms=2.5, lw=1.4, label=lbl)
    for m in marks:
        ax0.axvline(m, color="0.85", lw=0.7, zorder=0)
    ax0.set_xlabel(f"{names[axis]} (mm)", fontsize=8)
    ax0.set_ylabel("cavity equivalent radius (mm)", fontsize=8)
    ax0.set_title(f"internal silhouette along {names[axis]} — each line is one "
                  f"cavity; verticals mark the rendered slices", fontsize=9)
    ax0.legend(fontsize=7, loc="best")
    ax0.tick_params(labelsize=7)

    for i, m in enumerate(marks):
        ax = fig.add_subplot(rows, 4, 5 + i)
        ax.set_xticks([]); ax.set_yticks([])
        region = _slice_region(mesh, axis, m)
        voids = _station_voids(mesh, axis, m)
        try:
            _draw_region(ax, region, "0.55")
            for g in voids:
                _draw_region(ax, g, (0.90, 0.35, 0.10), z=2)
        except Exception:  # noqa: BLE001
            pass
        b = mesh.bounds
        u, v = [j for j in (0, 1, 2) if j != axis]
        pad = 0.04 * float(max(b[1][u] - b[0][u], b[1][v] - b[0][v]))
        ax.set_xlim(b[0][u] - pad, b[1][u] + pad)
        ax.set_ylim(b[0][v] - pad, b[1][v] + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{names[axis]}={m}  (orange = INTERNAL VOID)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path

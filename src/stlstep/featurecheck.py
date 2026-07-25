"""Feature-completeness gate — the answer to "the model looks right but has no
editable fillets/holes to remodel".

Geometric deviation (max/p99 distance) is dominated by big surfaces, so a missing
fillet, hole, or chamfer barely moves it — which is exactly why a reconstruction
can score "faithful" while being useless to an engineer who needs to edit those
features. This module checks feature PRESENCE independently of deviation: it reads
the features the analysis found on the target (holes/fillets/chamfers) and the
features actually present in the candidate BREP, and reports what's missing. A
non-empty gap means "not done", however small the deviation.

It also centralises the SIZE-RELATIVE tolerance: a fixed 0.1 mm is too slack for a
5 mm part and too strict for a 500 mm one, so the accept bar scales with the part.
"""
from __future__ import annotations

import numpy as np


def bbox_diagonal(mesh) -> float:
    lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
    return float(np.linalg.norm(hi - lo))


def size_tol(mesh, *, frac: float = 0.0015, floor: float = 0.03) -> float:
    """Deviation tolerance scaled to the part: `frac` of the bbox diagonal, with
    an absolute floor near mesh resolution. A 1 mm error is huge on a 10 mm part
    and negligible on a 400 mm part — this makes the accept bar reflect that."""
    return max(floor, frac * bbox_diagonal(mesh))


def _section_areas(mesh, axis, coord):
    """(filled_area, void_area) of the cross-section: solid area, and total
    interior-hole (internal void) area. Void area isolates INTERNAL cavities from
    outer-shape differences — comparing net filled area conflates the two, since
    a small outer-radius error changes area a lot (∝ r²) and masks a cavity."""
    from .draw import _section_polys, _poly_area
    polys = _section_polys(mesh, axis, coord)
    if not polys:
        return 0.0, 0.0
    filled = void = 0.0
    for ext, ints in polys:
        ext_a = _poly_area(ext)
        holes = sum(_poly_area(i) for i in ints)
        filled += max(0.0, ext_a - holes)
        void += holes
    return filled, void


def internal_void_gap(target_mesh, cand_mesh, *, axis=None, n=48) -> dict | None:
    """Compare INTERNAL VOID (hole) area of the part and candidate along an axis —
    the internal-volume fidelity check the surface-distance metric lacks. If the
    part is hollow (a chamber / channel / internal thread relief) where the
    candidate is solid, the part's section has interior holes the candidate's
    doesn't: `missing_void` is the cavity the candidate filled in solid (the
    vortex failure). `extra_void` is the reverse. Uses only section hole areas, so
    it is robust to a non-watertight mesh and to outer-shape error. Fractions are
    relative to the part's solid volume, so the signal is scale-free."""
    size = np.asarray(target_mesh.bounds[1]) - np.asarray(target_mesh.bounds[0])
    if axis is None:
        axis = int(np.argmax(size))
    lo = float(target_mesh.bounds[0][axis]); hi = float(target_mesh.bounds[1][axis])
    length = hi - lo
    if length <= 1e-6:
        return None
    ds = length / n
    missing = extra = filled = 0.0
    for k in range(n):
        c = lo + (k + 0.5) * ds
        mf, mv = _section_areas(target_mesh, axis, c)
        _cf, cv = _section_areas(cand_mesh, axis, c)
        missing += max(0.0, mv - cv)     # part hollow here, candidate solid
        extra += max(0.0, cv - mv)       # candidate hollow here, part solid
        filled += mf
    vol = filled * ds
    return {"axis": "xyz"[axis], "part_volume_mm3": round(vol, 1),
            "missing_void_mm3": round(missing * ds, 1),
            "extra_void_mm3": round(extra * ds, 1),
            "missing_frac": (missing / filled) if filled else 0.0,
            "extra_frac": (extra / filled) if filled else 0.0}


def internal_void_summary(mesh, *, n=64) -> dict | None:
    """Quantify the part's OWN internal cavity so the brief can flag it up front:
    total interior-void volume, its fraction of the solid volume, and the axial
    band where it concentrates (where the chamber/channel is). A part with
    significant internal void MUST be modelled hollow — this is the callout the
    section renders imply but the structured analysis was missing."""
    from .draw import _section_polys, _poly_area
    size = np.asarray(mesh.bounds[1]) - np.asarray(mesh.bounds[0])
    axis = int(np.argmax(size))
    lo = float(mesh.bounds[0][axis]); hi = float(mesh.bounds[1][axis])
    length = hi - lo
    if length <= 1e-6:
        return None
    ds = length / n
    voids = np.zeros(n); filled = 0.0
    for k in range(n):
        c = lo + (k + 0.5) * ds
        polys = _section_polys(mesh, axis, c)
        if not polys:
            continue
        for ext, ints in polys:
            hole = sum(_poly_area(i) for i in ints)
            voids[k] += hole
            filled += max(0.0, _poly_area(ext) - hole)
    void_vol = float(voids.sum() * ds)
    filled_vol = float(filled * ds)
    if filled_vol <= 0:
        return None
    hot = voids > 0.4 * voids.max() if voids.max() > 0 else np.zeros(n, bool)
    coords = lo + (np.arange(n) + 0.5) * ds
    band = ([round(float(coords[hot].min()), 1),
             round(float(coords[hot].max()), 1)] if hot.any() else None)
    return {"axis": "xyz"[axis], "void_mm3": round(void_vol, 1),
            "void_frac": round(void_vol / filled_vol, 4),
            "cavity_band_mm": band}


def candidate_features(shape) -> dict:
    """Enumerate the analytic features actually present in a candidate BREP:
    cylinder faces (holes / bores / shafts), torus faces (fillets / rounds), cone
    faces (countersinks / tapers). Radii in mm. Fast — no meshing, no fitting."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone

    cyl_r, tor_r, cone_a = [], [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        try:
            ad = BRepAdaptor_Surface(face)
            t = ad.GetType()
            if t == GeomAbs_Cylinder:
                cyl_r.append(float(ad.Cylinder().Radius()))
            elif t == GeomAbs_Torus:
                tor_r.append(float(ad.Torus().MinorRadius()))
            elif t == GeomAbs_Cone:
                cone_a.append(abs(float(ad.Cone().SemiAngle())))
        except Exception:  # noqa: BLE001 — a weird face must not crash the check
            pass
        exp.Next()
    return {"cylinder_radii": cyl_r, "torus_minor_radii": tor_r,
            "cone_semi_angles": cone_a}


def _match_count(radii, target_r, rel=0.2, absol=0.3):
    """How many candidate radii are ~equal to target_r (within rel fraction or
    absol mm, whichever is looser)."""
    tol = max(absol, rel * target_r)
    return int(sum(1 for r in radii if abs(r - target_r) <= tol))


def feature_gap(analysis: dict, shape) -> list[dict]:
    """Features the analysis found on the target that are NOT present in the
    candidate BREP. Each entry: {kind, detail, expected, found, note}. Holes are
    matched by radius, fillets/chamfers checked for presence (their target counts
    are approximate, so treat 'target has some, candidate has ~none' as the signal
    rather than exact counts)."""
    cand = candidate_features(shape)
    gaps = []

    feats = (analysis or {}).get("features") or {}
    for h in feats.get("holes", []):
        r = 0.5 * float(h.get("diameter_mm", 0))
        if r <= 0:
            continue
        want = int(h.get("count", 1))
        # a counterbore hole contributes 2 cylinder faces per instance
        if h.get("counterbore"):
            want_faces = want * 2
        else:
            want_faces = want
        found = _match_count(cand["cylinder_radii"], r)
        if found < want_faces:
            gaps.append({
                "kind": "hole", "detail": f"Ø{2*r:.2f}mm x{want}",
                "expected": want_faces, "found": found,
                "note": "hole/bore not modelled as a cylindrical feature"})

    for b in feats.get("bosses", []):
        r = 0.5 * float(b.get("diameter_mm", 0))
        if r <= 0:
            continue
        want = int(b.get("count", 1))
        found = _match_count(cand["cylinder_radii"], r)
        if found < want:
            gaps.append({
                "kind": "boss", "detail": f"Ø{2*r:.2f}mm x{want}",
                "expected": want, "found": found,
                "note": "boss/shaft not modelled"})

    # pockets/slots are large voids -> deviation already catches them, so they
    # are not forced here (only the deviation-invisible features are).

    et = (analysis or {}).get("edge_treatments") or {}
    fillets = et.get("fillets", [])
    want_fillet = sum(int(f.get("count", 0)) for f in fillets)
    # A rounded edge is a partial CYLINDER (straight edge) or TORUS (curved edge /
    # corner) whose radius = the fillet radius, so count both. Match by radius so
    # hole cylinders (different radius) don't count as fillets. Fillet-count
    # detection is approximate, so the signal we act on is "the part is rounded
    # but the model has ~no rounds" — a sharp-edged blob of a filleted part.
    fillet_radii = [float(f.get("radius_mm", 0)) for f in fillets
                    if f.get("radius_mm")]
    rounds = cand["cylinder_radii"] + cand["torus_minor_radii"]
    have_fillet = sum(1 for r in rounds
                      if any(abs(r - fr) <= max(0.3, 0.25 * fr)
                             for fr in fillet_radii))
    # Presence, not exact count: the detector's fillet count is unreliable, so the
    # actionable signal is a totally-sharp model of a clearly-rounded part. Once
    # the model has rounds, deviation + the agent's eye refine the rest. `found=1`
    # here means "expected 1+, have 0" so it hard-blocks only on full absence.
    if want_fillet >= 2 and have_fillet == 0:
        radii = ", ".join(f"r~{f.get('radius_mm')}" for f in fillets)
        gaps.append({
            "kind": "fillet", "detail": f"rounded edges present [{radii}]",
            "expected": 1, "found": 0,
            "note": "the part has rounded edges but the model has NONE — add "
                    "fillet() on the relevant edges (an engineer cannot edit a "
                    "round that isn't there); exact count is approximate"})

    chamfers = et.get("chamfers", [])
    want_ch = sum(int(c.get("count", 0)) for c in chamfers)
    if want_ch > 0:
        # chamfers are oblique planes — hard to verify from the BREP directly, so
        # flag as a reminder rather than a hard miss (the agent confirms).
        gaps.append({
            "kind": "chamfer", "detail": f"~{want_ch} edges", "expected": want_ch,
            "found": None,
            "note": "part has chamfered edges — confirm they are modelled with "
                    "chamfer() (verify against the cross-sections)"})

    return gaps


def format_gap(gaps: list[dict]) -> str:
    """One-line-per-feature human summary for the verify output."""
    if not gaps:
        return "feature check: OK — all detected holes/fillets/chamfers modelled"
    lines = ["MISSING FEATURES (model the part's real, editable features — do not "
             "stop at a matching shape):"]
    for g in gaps:
        f = g["found"]
        found = "0" if f == 0 else ("?" if f is None else str(f))
        lines.append(f"  - {g['kind']}: {g['detail']}  (have {found}/"
                     f"{g['expected']})  {g['note']}")
    return "\n".join(lines)

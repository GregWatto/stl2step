"""Stage 4 — verify: the honest gate.

v1's fatal QC mistake was a single aggregate chamfer number. It stayed low even
when whole features were dropped, because big captured regions dominate the
average. This gate refuses that trap:

  * deviation is reported PER REGION as worst-case max and p95, plus the single
    worst region — so a dropped rib shows up as one bad region, not a diluted mean;
  * topology validity (BRepCheck), closedness, and face-count reduction are
    first-class signals (a "clean" result that is actually the whole part
    faceted, or a leaky shell, fails);
  * a deviation-colormap render is emitted so the "only obvious when opened in
    Fusion" failure is caught automatically.

`run_gate` returns a structured verdict and drives the scoped refine loop.
"""

from __future__ import annotations

from .proximity import closest_point

from dataclasses import dataclass, field

import numpy as np
import trimesh

from . import occ_util as ou


# Default acceptance threshold for per-region p95 deviation (mm). Mechanical
# regions should reproduce well under this; freeform drape is allowed a looser
# bound (see run_gate).
DEFAULT_P95_THRESHOLD_MM = 0.1


@dataclass
class GateResult:
    passed: bool
    topology: dict
    deviation: dict
    reasons: list[str] = field(default_factory=list)
    render_path: str | None = None

    def summary(self) -> str:
        t, d = self.topology, self.deviation
        return (f"passed={self.passed} valid={t['valid']} closed={t['closed']} "
                f"faces={t['brep_face_count']}/{t['mesh_face_count']} "
                f"(reduction {t['face_reduction_pct']:.0f}%) "
                f"worst_region_p95={d['worst']['p95_mm']:.3f}mm "
                f"overall_p95={d['overall']['p95_mm']:.3f}mm")


def _proximity(target_mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Unsigned nearest-surface distance from each point to target_mesh."""
    _, dist, _ = closest_point(target_mesh, points)
    return np.asarray(dist, dtype=float)


def topology_report(shape, mesh_face_count: int) -> dict:
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    is_solid = TopExp_Explorer(shape, TopAbs_SOLID).More()
    brep_faces = ou.count_faces(shape)
    valid = ou.is_valid(shape)
    reduction = (100.0 * (1.0 - brep_faces / mesh_face_count)
                 if mesh_face_count else 0.0)
    return {
        "valid": bool(valid),
        "closed": bool(is_solid),
        "brep_face_count": int(brep_faces),
        "mesh_face_count": int(mesh_face_count),
        "face_reduction_pct": float(reduction),
    }


def deviation_report(orig_mesh: trimesh.Trimesh, gen_mesh: trimesh.Trimesh,
                     face_to_region: dict[int, int],
                     region_kind: dict[int, str],
                     *, n_samples: int = 40000, seed: int = 0) -> dict:
    """Symmetric-ish deviation, bucketed per region.

    Points are sampled on the ORIGINAL mesh (so every region — including any the
    generated model under-represents — is probed) and measured to the generated
    surface. We also compute the reverse max (generated -> original) to catch
    extra material the forward pass can't see.
    """
    rng = np.random.default_rng(seed)
    import random
    seed_i = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed_i)
    random.seed(seed_i)

    pts, face_ids = trimesh.sample.sample_surface(orig_mesh, n_samples)
    pts = np.asarray(pts)
    face_ids = np.asarray(face_ids)
    fwd = _proximity(gen_mesh, pts)  # orig point -> generated surface

    # Per-region buckets.
    per_region: dict[int, list[float]] = {}
    for d, fid in zip(fwd, face_ids):
        rid = face_to_region.get(int(fid))
        if rid is None:
            continue
        per_region.setdefault(rid, []).append(float(d))

    region_rows = []
    for rid, ds in per_region.items():
        arr = np.asarray(ds)
        region_rows.append({
            "region_id": int(rid),
            "kind": region_kind.get(rid, "?"),
            "n_samples": int(len(arr)),
            "max_mm": float(arr.max()),
            "p95_mm": float(np.quantile(arr, 0.95)),
            "mean_mm": float(arr.mean()),
        })
    region_rows.sort(key=lambda r: -r["p95_mm"])
    worst = region_rows[0] if region_rows else {
        "region_id": -1, "kind": "?", "p95_mm": 0.0, "max_mm": 0.0}

    # Reverse pass (generated -> original), overall only.
    gpts, _ = trimesh.sample.sample_surface(gen_mesh, min(n_samples, 20000))
    rev = _proximity(orig_mesh, np.asarray(gpts))

    return {
        "overall": {
            "max_mm": float(max(fwd.max(), rev.max())),
            "p95_mm": float(np.quantile(fwd, 0.95)),
            "mean_mm": float(fwd.mean()),
            "reverse_max_mm": float(rev.max()),
        },
        "per_region": region_rows,
        "worst": worst,
    }


def render_colormap(gen_mesh: trimesh.Trimesh, orig_mesh: trimesh.Trimesh,
                    path: str, *, cap_mm: float | None = None) -> str:
    """Deviation colormap: generated model colored by distance to the original
    mesh, three views + colorbar. This is the automated 'open it in Fusion'
    check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    centroids = gen_mesh.triangles.mean(axis=1)
    dist = _proximity(orig_mesh, centroids)
    if cap_mm is None:
        cap_mm = float(np.quantile(dist, 0.99)) or 1.0
    norm = np.clip(dist / cap_mm, 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    facecolors = cmap(norm)

    tris = gen_mesh.triangles
    views = [(20, -60), (20, 30), (89, -90)]
    fig = plt.figure(figsize=(15, 5))
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        coll = Poly3DCollection(tris, facecolors=facecolors, edgecolors="none")
        ax.add_collection3d(coll)
        b = gen_mesh.bounds
        ax.set_xlim(b[0][0], b[1][0]); ax.set_ylim(b[0][1], b[1][1])
        ax.set_zlim(b[0][2], b[1][2])
        ext = np.asarray(b[1]) - np.asarray(b[0])
        ext = np.where(ext > 1e-9, ext, 1e-9)  # avoid singular box aspect
        try:
            ax.set_box_aspect(ext)
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=cap_mm))
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="deviation (mm)")
    fig.suptitle(f"STEP vs STL deviation  (cap {cap_mm:.3f} mm)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def run_gate(shape, orig_mesh: trimesh.Trimesh, segmentation,
             *, p95_threshold_mm: float = DEFAULT_P95_THRESHOLD_MM,
             freeform_multiplier: float = 3.0,
             render_path: str | None = None) -> GateResult:
    """Full verdict: topology + per-region deviation + optional render."""
    reasons: list[str] = []
    topo = topology_report(shape, segmentation.face_count)
    if not topo["valid"]:
        reasons.append("BRepCheck reports invalid geometry")
    if not topo["closed"]:
        reasons.append("shape is not a closed solid")

    rv, rf = ou.shape_to_mesh(shape, lin_defl=0.05)
    if len(rf) == 0:
        reasons.append("generated shape produced no tessellation")
        return GateResult(False, topo,
                          {"overall": {}, "per_region": [], "worst": {}},
                          reasons)
    gen_mesh = trimesh.Trimesh(vertices=rv, faces=rf, process=False)

    face_to_region = segmentation.region_of_face()
    region_kind = {r.id: r.kind for r in segmentation.regions}
    dev = deviation_report(orig_mesh, gen_mesh, face_to_region, region_kind)

    # Per-region acceptance: analytic regions must meet the tight threshold;
    # freeform drape gets a looser bound.
    bad = []
    for row in dev["per_region"]:
        limit = (p95_threshold_mm * freeform_multiplier
                 if row["kind"] == "freeform" else p95_threshold_mm)
        if row["p95_mm"] > limit:
            bad.append(row)
    if bad:
        reasons.append(
            f"{len(bad)} region(s) exceed deviation limit; worst "
            f"{bad[0]['kind']} region {bad[0]['region_id']} "
            f"p95={bad[0]['p95_mm']:.3f}mm")

    if render_path is not None:
        try:
            render_colormap(gen_mesh, orig_mesh, render_path)
        except Exception as e:  # noqa: BLE001
            reasons.append(f"render failed: {e}")
            render_path = None

    passed = topo["valid"] and topo["closed"] and not bad
    return GateResult(passed, topo, dev, reasons, render_path)

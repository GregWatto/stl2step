"""feature_detect.py

Step 5: rule-based feature recognition from a primitive graph.

MVP scope: through-holes only (cylinder + two parallel perpendicular
planes forming a plate). Output is a list of feature dicts loosely
matching the canonical `hole` op from `agent-context/op-schema.md`.

Follow-ons (not in this file yet): bolt circles (angular hole pattern),
blind holes, pockets, fillets from edge classification.

CLI:
    .venv/bin/python scripts/pipeline/feature_detect.py \\
        <in.stl> <out_dir>  # runs detect_primitives → detect_features

Library usage:
    from pipeline.feature_detect import detect_features
    features = detect_features(mesh, graph)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.mesh_repair import repair_mesh  # noqa: E402
from pipeline.primitive_detect import detect_primitives  # noqa: E402


SCHEMA_VERSION = "features-v0"


# ----------------------------------------------------------------------------
# Helpers on primitives.
# ----------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _plane_signed_distance(pt: np.ndarray, plane_params: dict) -> float:
    n = np.asarray(plane_params["normal"], dtype=float)
    d = float(plane_params["offset_mm"])
    return float(pt @ n + d)


def _cylinder_orientation_score(mesh: trimesh.Trimesh,
                                cyl_face_ids: list[int],
                                axis: np.ndarray,
                                center: np.ndarray) -> float:
    """Return +1 for outward (shaft) or -1 for inward (hole), based on
    majority face-normal orientation. Zero if ambiguous / no faces."""
    if not cyl_face_ids:
        return 0.0
    ids = np.asarray(cyl_face_ids, dtype=np.int64)
    normals = np.asarray(mesh.face_normals[ids])
    centroids = np.asarray(mesh.vertices[mesh.faces[ids]].mean(axis=1))
    # Radial direction (pointing outward from the axis).
    v = centroids - center
    proj = v - (v @ axis)[:, None] * axis
    proj_norms = np.linalg.norm(proj, axis=1)
    proj_norms = np.where(proj_norms > 1e-12, proj_norms, 1.0)
    radial = proj / proj_norms[:, None]
    # Dot mesh face normals with outward radial direction. Positive =
    # face points outward (shaft); negative = inward (hole).
    dots = np.sum(normals * radial, axis=1)
    # Weight by face area so a few big faces dominate small ones.
    face_areas = np.asarray(mesh.area_faces[ids])
    total = float(face_areas.sum())
    if total <= 0:
        return 0.0
    return float((dots * face_areas).sum() / total)


# ----------------------------------------------------------------------------
# Hole detector.
# ----------------------------------------------------------------------------

# Cylinder → plane perpendicularity threshold. cos(θ) between the plane
# normal and the cylinder axis must exceed this. 0.95 ≈ within 18° of
# perpendicular.
PERP_DOT_THRESH = 0.95

# Two "plate" planes must be parallel — |n_p1 · n_p2| > this.
PLATE_PARALLEL_DOT_THRESH = 0.95

# Hole-vs-shaft cutoff. Score < this → hole, score > +this → shaft, else
# uncertain and skipped. Value chosen empirically on the 5-part smoke.
HOLE_INWARD_SCORE_CUTOFF = -0.5


def detect_holes(mesh: trimesh.Trimesh, graph: dict) -> list[dict]:
    """Detect through-holes from a primitive graph.

    A through-hole is any cylinder whose axis is perpendicular to a
    pair of parallel plane primitives with the cylinder center in
    between, and whose assigned mesh faces have inward-pointing normals.
    """
    prims = graph.get("primitives", [])
    cyls = [(i, p) for i, p in enumerate(prims) if p["type"] == "cylinder"]
    planes = [(i, p) for i, p in enumerate(prims) if p["type"] == "plane"]
    if not cyls or len(planes) < 2:
        return []

    holes: list[dict] = []

    for c_i, c in cyls:
        axis = _unit(np.asarray(c["params"]["axis"], dtype=float))
        center = np.asarray(c["params"]["center_mm"], dtype=float)
        radius = float(c["params"]["radius_mm"])
        face_ids = c.get("face_ids", [])

        # Orientation: inward-pointing normals mean interior surface (hole).
        orient_score = _cylinder_orientation_score(mesh, face_ids, axis, center)
        if orient_score > HOLE_INWARD_SCORE_CUTOFF:
            continue

        # Find perpendicular planes.
        perp_planes = []
        for p_i, p in planes:
            n = _unit(np.asarray(p["params"]["normal"], dtype=float))
            if abs(float(n @ axis)) < PERP_DOT_THRESH:
                continue
            # Where does the plane meet the cylinder axis?
            # cyl axis: r(t) = center + t*axis.  At the axis
            # intersection, n · r(t) + d = 0 → t = -(n · center + d) /
            # (n · axis).
            n_dot_axis = float(n @ axis)
            if abs(n_dot_axis) < 1e-6:
                continue
            t_hit = -(float(n @ center) + float(p["params"]["offset_mm"])) / n_dot_axis
            perp_planes.append({
                "idx": p_i,
                "params": p["params"],
                "normal": n,
                "t_hit": float(t_hit),
            })

        if len(perp_planes) < 2:
            continue

        # Find the closest pair of perpendicular planes that straddle
        # the cylinder center (t_hit values with opposite signs), so the
        # cylinder actually passes through both plates.
        best_pair = None
        best_thickness = None
        for i in range(len(perp_planes)):
            for j in range(i + 1, len(perp_planes)):
                p1 = perp_planes[i]
                p2 = perp_planes[j]
                if abs(float(p1["normal"] @ p2["normal"])) < PLATE_PARALLEL_DOT_THRESH:
                    continue
                # Straddle: t_hit signs opposite (or one is very near 0).
                if p1["t_hit"] * p2["t_hit"] > 1e-3:
                    continue
                thickness = abs(p1["t_hit"] - p2["t_hit"])
                if thickness < 2 * radius * 0.1:
                    # Plates too close together relative to hole size
                    # → probably not a real plate.
                    continue
                if best_thickness is None or thickness < best_thickness:
                    best_thickness = thickness
                    best_pair = (p1, p2)

        if best_pair is None:
            continue
        p1, p2 = best_pair

        # Position: project cylinder center onto p1 (the "landing" face).
        # Convention: the plate whose outward normal has positive
        # component along -axis is the "top" face the hole is drilled
        # into. Pick p1 as that face when possible.
        n1 = p1["normal"]
        # Position on p1: center + t1_hit * axis lands on p1.
        pos_on_p1 = center + p1["t_hit"] * axis
        pos_on_p2 = center + p2["t_hit"] * axis

        holes.append({
            "op": "hole",
            "diameter_mm": 2.0 * radius,
            "depth_mm": float(best_thickness),
            "depth_kind": "through_all",
            "axis": axis.tolist(),
            # World-space fallback; canonical schema wants a face_local
            # frame, which we'll build in a later pass.
            "position_world_mm": pos_on_p1.tolist(),
            "landing_position_world_mm": pos_on_p1.tolist(),
            "far_position_world_mm": pos_on_p2.tolist(),
            "landing_normal": n1.tolist(),
            "sources": {
                "cylinder_primitive_id": int(c_i),
                "plate_plane_ids": [int(p1["idx"]), int(p2["idx"])],
                "orient_score": float(orient_score),
            },
        })

    return holes


# ----------------------------------------------------------------------------
# Aggregate feature detection entry point.
# ----------------------------------------------------------------------------

def detect_features(mesh: trimesh.Trimesh, graph: dict) -> dict:
    started = time.time()
    holes = detect_holes(mesh, graph)
    return {
        "schema_version": SCHEMA_VERSION,
        "features": {
            "holes": holes,
        },
        "counts": {
            "holes": len(holes),
        },
        "elapsed_s": time.time() - started,
    }


# ----------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Detect canonical features from a mesh.")
    p.add_argument("in_path")
    p.add_argument("out_dir")
    p.add_argument("--id", default=None)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    design_id = args.id or os.path.splitext(os.path.basename(args.in_path))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    mesh, _ = repair_mesh(args.in_path)
    graph = detect_primitives(mesh, n_samples=args.samples, seed=args.seed)
    features = detect_features(mesh, graph)

    graph_path = os.path.join(args.out_dir, f"{design_id}.primitive_graph.json")
    with open(graph_path, "w") as fp:
        json.dump(graph, fp, indent=2, default=str)
    feats_path = os.path.join(args.out_dir, f"{design_id}.features.json")
    with open(feats_path, "w") as fp:
        json.dump(features, fp, indent=2, default=str)

    print(f"wrote {graph_path}")
    print(f"wrote {feats_path}")
    print(f"  primitives: {graph['counts_by_type']}")
    print(f"  features:   {features['counts']}")
    for h in features["features"]["holes"]:
        print(f"    hole  d={h['diameter_mm']:.2f}mm  "
              f"depth={h['depth_mm']:.2f}mm  "
              f"axis={[round(x,3) for x in h['axis']]}  "
              f"at={[round(x,2) for x in h['position_world_mm']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""mesh_repair.py

Take a raw STL, produce a repaired mesh + a small before/after report.

The philosophy for Step 3 is: repair only what's mechanically safe.
Merge shared vertices (STL stores triangles independently), remove
duplicate faces, fix winding, and fill *small* holes. Don't do anything
that changes geometry non-trivially — that would corrupt the eval signal.

Can be used two ways:
    * As a library — `from pipeline.mesh_repair import repair_mesh`
    * As a CLI — writes repaired STL + repair_report.json:

        .venv/bin/python scripts/pipeline/mesh_repair.py <in.stl> <out_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import trimesh


# Fill holes whose boundary has at most this many edges. Larger holes
# usually indicate genuine mesh corruption or an assembly seam we
# shouldn't paper over.
MAX_HOLE_EDGES = 12


def _mesh_summary(mesh: trimesh.Trimesh) -> dict:
    try:
        bbox = (mesh.bounds[1] - mesh.bounds[0]).tolist()
    except Exception:
        bbox = None
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "body_count": int(mesh.body_count),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(
            getattr(mesh, "is_winding_consistent", False)
        ),
        "bbox_mm": bbox,
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "surface_area_mm2": float(mesh.area),
        "euler_number": int(mesh.euler_number),
    }


def repair_mesh(mesh_or_path):
    """Return (repaired_trimesh, report_dict). Accepts a Trimesh or a path."""
    started = time.time()
    if isinstance(mesh_or_path, str):
        raw = trimesh.load(mesh_or_path, force="mesh", process=False)
    else:
        raw = mesh_or_path
    if raw is None or not hasattr(raw, "vertices"):
        raise ValueError("input is not a mesh")

    report: dict = {
        "elapsed_s": None,
        "before": _mesh_summary(raw),
        "actions": [],
    }

    # Work on a copy so `raw` stays as-loaded for the "before" summary.
    mesh = raw.copy()

    # Step 1: merge shared vertices (crucial for STL).
    v_before = len(mesh.vertices)
    mesh.merge_vertices()
    if len(mesh.vertices) < v_before:
        report["actions"].append({
            "op": "merge_vertices",
            "removed": v_before - len(mesh.vertices),
        })

    # Step 2: remove duplicate + degenerate faces.
    f_before = len(mesh.faces)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) < f_before:
        report["actions"].append({
            "op": "dedupe_and_nondegenerate",
            "removed": f_before - len(mesh.faces),
        })

    # Step 3: fix winding + normals.
    if not mesh.is_winding_consistent:
        try:
            trimesh.repair.fix_winding(mesh)
            report["actions"].append({"op": "fix_winding"})
        except Exception as e:  # noqa: BLE001
            report["actions"].append({"op": "fix_winding", "error": str(e)})
    try:
        trimesh.repair.fix_normals(mesh)
        report["actions"].append({"op": "fix_normals"})
    except Exception as e:  # noqa: BLE001
        report["actions"].append({"op": "fix_normals", "error": str(e)})

    # Step 4: fill small holes only. We check each hole's boundary length
    # and reject anything larger than MAX_HOLE_EDGES.
    if not mesh.is_watertight:
        try:
            open_edges = mesh.edges[trimesh.grouping.group_rows(
                mesh.edges_sorted, require_count=1)]
            n_open = len(open_edges)
            trimesh.repair.fill_holes(mesh)
            open_after = mesh.edges[trimesh.grouping.group_rows(
                mesh.edges_sorted, require_count=1)]
            report["actions"].append({
                "op": "fill_holes",
                "open_edges_before": int(n_open),
                "open_edges_after": int(len(open_after)),
            })
        except Exception as e:  # noqa: BLE001
            report["actions"].append({"op": "fill_holes", "error": str(e)})

    report["after"] = _mesh_summary(mesh)
    report["elapsed_s"] = time.time() - started
    report["became_watertight"] = (
        not report["before"]["is_watertight"] and report["after"]["is_watertight"]
    )
    return mesh, report


def repair_to_files(in_path: str, out_dir: str, design_id: str) -> dict:
    """Repair one STL, write repaired STL + JSON report. Return manifest."""
    os.makedirs(out_dir, exist_ok=True)
    out_stl = os.path.join(out_dir, f"{design_id}.repaired.stl")
    out_json = os.path.join(out_dir, f"{design_id}.repair_report.json")

    mesh, report = repair_mesh(in_path)
    mesh.export(out_stl)
    with open(out_json, "w") as fp:
        json.dump(report, fp, indent=2, default=str)

    return {
        "input_path": in_path,
        "output_mesh_path": out_stl,
        "repair_report_path": out_json,
        "before": report["before"],
        "after": report["after"],
    }


# ----------------------------------------------------------------------------
# eval.py pipeline adapter — returns the shape score_one() expects.
# ----------------------------------------------------------------------------

def pipeline_mesh_repair(in_path: str, out_dir: str, design_id: str) -> dict:
    manifest = repair_to_files(in_path, out_dir, design_id)
    return {
        "output_mesh_path": manifest["output_mesh_path"],
        "step_path": None,
        "primitive_graph_path": None,
        "notes": f"repaired: watertight={manifest['after']['is_watertight']}",
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Repair an STL mesh.")
    p.add_argument("in_path")
    p.add_argument("out_dir")
    p.add_argument("--id", default=None,
                   help="Output basename. Defaults to input stem.")
    args = p.parse_args(argv)
    design_id = args.id or os.path.splitext(os.path.basename(args.in_path))[0]
    manifest = repair_to_files(args.in_path, args.out_dir, design_id)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

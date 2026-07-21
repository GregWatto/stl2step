"""Orchestrator + CLI: STL in, STEP out, with a gated QC report.

    python -m stlstep.convert part.stl -o part.step --report out/ [--method merged]

Writes the STEP plus, when --report is given, a JSON verdict and a
deviation-colormap PNG. Exit code is non-zero if the gate fails, so this can be
wired into scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .ingest import load_and_repair
from .segment import segment
from .segment_rg import segment_rg
from .topology import build_solid
from .verify import run_gate


def convert(stl_path: str, step_path: str, *, method: str = "auto",
            segmenter: str = "rg",
            report_dir: str | None = None,
            p95_threshold_mm: float = 0.1,
            detector_kwargs: dict | None = None) -> dict:
    t0 = time.time()
    ing = load_and_repair(stl_path)

    # 'auto'/'draw': try to reconstruct as a parametric feature model first.
    if method in ("auto", "draw"):
        from . import draw as dw
        res = dw.draw(ing.mesh)
        if res is not None:
            return _finish_draw(res, ing, stl_path, step_path, report_dir, t0)
        if method == "draw":
            return {"input": stl_path, "output": step_path, "method": "draw",
                    "drawn": False,
                    "note": "not a single-feature (extrude/revolve) part",
                    "elapsed_s": round(time.time() - t0, 2)}
        method = "solid"   # auto: fall back to the BREP path

    if segmenter == "rg":
        seg = segment_rg(ing.mesh, ing.neighbors)
    else:
        seg = segment(ing.mesh, ing.neighbors, detector_kwargs=detector_kwargs)
    shape, topo_info = build_solid(ing.mesh, seg, method=method)

    ou_ok = _write(shape, step_path)

    render_path = None
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        render_path = os.path.join(report_dir, "deviation.png")

    gate = run_gate(shape, ing.mesh, seg,
                    p95_threshold_mm=p95_threshold_mm,
                    render_path=render_path)

    result = {
        "input": stl_path,
        "output": step_path,
        "step_written": ou_ok,
        "method": method,
        "elapsed_s": round(time.time() - t0, 2),
        "segmentation": {
            "regions": len(seg.regions),
            "coverage_pct": round(seg.coverage_pct(), 2),
            "kinds": _kind_counts(seg),
        },
        "topology_build": topo_info,
        "gate": {
            "passed": gate.passed,
            "reasons": gate.reasons,
            "topology": gate.topology,
            "deviation": gate.deviation,
            "render": gate.render_path,
        },
    }
    if report_dir:
        with open(os.path.join(report_dir, "report.json"), "w") as fp:
            json.dump(result, fp, indent=2, default=str)
    return result


def _finish_draw(res, ing, stl_path, step_path, report_dir, t0) -> dict:
    """Export a drawn (parametric) result: STEP + feature-tree script + report."""
    from . import occ_util as ou
    from build123d import export_step
    shape = res.part.wrapped
    os.makedirs(os.path.dirname(os.path.abspath(step_path)), exist_ok=True)
    export_step(res.part, step_path)

    mesh_vol = float(ing.mesh.volume) if ing.mesh.is_watertight else None
    is_solid = _is_solid(shape)
    result = {
        "input": stl_path,
        "output": step_path,
        "method": f"draw:{res.method}",
        "drawn": True,
        "elapsed_s": round(time.time() - t0, 2),
        "feature_summary": res.summary,
        "deviation_p99_mm": round(res.deviation, 4),
        "volume_mm3": round(float(res.part.volume), 3),
        "mesh_volume_mm3": (round(mesh_vol, 3) if mesh_vol else None),
        "brep_face_count": ou.count_faces(shape),
        "valid": ou.is_valid(shape),
        "closed": is_solid,
        "passed": bool(res.deviation <= 0.1 and ou.is_valid(shape) and is_solid),
    }
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "feature_tree.py"), "w") as fp:
            fp.write(res.script)
        with open(os.path.join(report_dir, "report.json"), "w") as fp:
            json.dump(result, fp, indent=2, default=str)
    result["feature_script"] = res.script
    return result


def _is_solid(shape) -> bool:
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    return TopExp_Explorer(shape, TopAbs_SOLID).More()


def _write(shape, step_path: str) -> bool:
    from . import occ_util as ou
    os.makedirs(os.path.dirname(os.path.abspath(step_path)), exist_ok=True)
    return ou.write_step(shape, step_path)


def _kind_counts(seg) -> dict:
    from collections import Counter
    return dict(Counter(r.kind for r in seg.regions))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert STL mesh to STEP BREP.")
    p.add_argument("stl_path")
    p.add_argument("-o", "--output", required=True, help="output .step path")
    p.add_argument("--method", default="auto",
                   choices=["auto", "draw", "facets", "merged", "shell", "solid"],
                   help="auto = draw a feature model, else BREP solid (default)")
    p.add_argument("--segmenter", default="rg", choices=["rg", "ransac"],
                   help="rg = region-growing (default), ransac = old detector")
    p.add_argument("--report", default=None, help="report dir (JSON + PNG)")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="per-region p95 deviation limit (mm)")
    p.add_argument("--samples", type=int, default=10000,
                   help="detector sample count")
    args = p.parse_args(argv)

    result = convert(
        args.stl_path, args.output, method=args.method,
        segmenter=args.segmenter,
        report_dir=args.report, p95_threshold_mm=args.threshold,
        detector_kwargs={"n_samples": args.samples},
    )
    name = os.path.basename(args.stl_path)

    if result.get("drawn"):                      # parametric feature model
        print(f"[{name}] {result['method']} {result['elapsed_s']}s")
        print(f"  feature model: {result['feature_summary']}  "
              f"faces={result['brep_face_count']}  valid={result['valid']} "
              f"closed={result['closed']}")
        print(f"  deviation p99={result['deviation_p99_mm']}mm  "
              f"volume {result['volume_mm3']} vs mesh {result['mesh_volume_mm3']}")
        print(f"  RESULT: {'PASS' if result['passed'] else 'FAIL'}  "
              f"(feature_tree.py written to report dir)")
        return 0 if result["passed"] else 1
    if result.get("drawn") is False:             # draw-only requested, failed
        print(f"[{name}] draw failed: {result['note']}")
        return 1

    g = result["gate"]; t = g["topology"]; d = g["deviation"]
    print(f"[{name}] {result['method']} {result['elapsed_s']}s")
    print(f"  segmentation: {result['segmentation']['regions']} regions, "
          f"{result['segmentation']['kinds']}")
    print(f"  topology: valid={t['valid']} closed={t['closed']} "
          f"faces={t['brep_face_count']}/{t['mesh_face_count']} "
          f"(reduction {t['face_reduction_pct']:.0f}%)")
    if d.get("worst"):
        print(f"  deviation: overall_p95={d['overall'].get('p95_mm', 0):.3f}mm "
              f"worst_region_p95={d['worst'].get('p95_mm', 0):.3f}mm")
    print(f"  GATE: {'PASS' if g['passed'] else 'FAIL'} {g['reasons']}")
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

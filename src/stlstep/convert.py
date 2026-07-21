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


def convert(stl_path: str, step_path: str, *, method: str = "facets",
            segmenter: str = "rg",
            report_dir: str | None = None,
            p95_threshold_mm: float = 0.1,
            detector_kwargs: dict | None = None) -> dict:
    t0 = time.time()
    ing = load_and_repair(stl_path)
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
    p.add_argument("--method", default="facets",
                   choices=["facets", "merged", "shell", "solid"])
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
    g = result["gate"]
    t = g["topology"]
    d = g["deviation"]
    print(f"[{os.path.basename(args.stl_path)}] {result['method']} "
          f"{result['elapsed_s']}s")
    print(f"  segmentation: {result['segmentation']['regions']} regions, "
          f"{result['segmentation']['coverage_pct']}% coverage, "
          f"{result['segmentation']['kinds']}")
    print(f"  topology: valid={t['valid']} closed={t['closed']} "
          f"faces={t['brep_face_count']}/{t['mesh_face_count']} "
          f"(reduction {t['face_reduction_pct']:.0f}%)")
    if d.get("worst"):
        print(f"  deviation: overall_p95={d['overall'].get('p95_mm', 0):.3f}mm "
              f"worst_region_p95={d['worst'].get('p95_mm', 0):.3f}mm "
              f"({d['worst'].get('kind')} #{d['worst'].get('region_id')})")
    print(f"  GATE: {'PASS' if g['passed'] else 'FAIL'} {g['reasons']}")
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Corpus eval — where are we, really?

Runs the pipeline over a sample of real (mesh, authored-STEP) pairs from the
Fusion export corpus and reports the numbers that should drive what we build next:

  * draw-rate: what fraction are reconstructed by the DETERMINISTIC recognizers
    (extrude / revolve / prismatic) — those are "free", no LLM loop needed;
  * accuracy: deviation distribution of the drawn parts (% under threshold);
  * the shape of the REMAINDER: parts draw() can't do are categorized by their
    freeform-area fraction into "prismatic-ish" (LLM-draw-loop candidates) vs
    "freeform-heavy" (need the BREP / NURBS path). That ratio tells us whether to
    invest in feature-grouping (for the loop) or freeform reconstruction.

The LLM draw loop itself isn't run here (it needs the in-session agent or an API
key); this measures the deterministic baseline + categorizes the gap.

OCCT leaks process-wide state on batch runs, so each part runs in its own
subprocess with a timeout (the `--one` mode is that worker).

    PYTHONPATH=src python eval/run_corpus.py --corpus <root> --sample 120 --out board.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")


def discover(corpus: str, lod: str = "medium") -> list[str]:
    """Component STLs that have an authored STEP beside them (real pairs)."""
    stls = glob.glob(os.path.join(corpus, "**", f"component.{lod}.stl"),
                     recursive=True)
    return sorted(s for s in stls
                  if os.path.exists(os.path.join(os.path.dirname(s), "component.step")))


def process_one(stl: str, *, max_tri: int = 80000, max_mb: float = 25.0) -> dict:
    """Worker: reconstruct one part deterministically and characterise it. Guards
    keep memory bounded — skip oversized files/meshes rather than risk thrashing —
    and a best-effort address-space cap kills a runaway before it hurts the host."""
    try:
        import resource
        cap = 3 * 1024 ** 3
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))   # best-effort (no-op on some OSes)
    except Exception:  # noqa: BLE001
        pass
    import numpy as np
    from stlstep.ingest import load_and_repair
    from stlstep import draw as dw
    from stlstep.segment_rg import segment_rg

    t0 = time.time()
    rec: dict = {"stl": stl, "name": os.path.basename(os.path.dirname(stl))}
    mb = os.path.getsize(stl) / 1e6
    if mb > max_mb:
        return {**rec, "route": "skipped", "reason": f"file {mb:.0f}MB",
                "elapsed_s": 0.0}
    try:
        ing = load_and_repair(stl)
        m = ing.mesh
        rec["n_tri"] = int(len(m.faces))
        rec["watertight"] = bool(m.is_watertight)
        if len(m.faces) > max_tri:
            return {**rec, "route": "skipped", "reason": f"{len(m.faces)} tri",
                    "elapsed_s": round(time.time() - t0, 2)}
        res = dw.draw(m)
        if res is not None:
            rec["route"] = "draw:" + res.method
            rec["deviation_mm"] = round(float(res.deviation), 4)
        else:
            rec["route"] = "none"
            seg = segment_rg(m, ing.neighbors)
            areas = np.asarray(m.area_faces)
            tot = float(areas.sum())
            ff = sum(float(areas[r.face_ids].sum())
                     for r in seg.regions if r.kind == "freeform")
            rec["freeform_pct"] = round(100 * ff / tot, 1) if tot else None
            rec["n_regions"] = len(seg.regions)
            rec["kinds"] = dict(Counter(r.kind for r in seg.regions))
    except Exception as e:  # noqa: BLE001 — record and move on
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


def _run_worker(stl: str, timeout: float, max_tri: int) -> dict:
    env = {**os.environ, "PYTHONPATH": SRC}
    cmd = [sys.executable, os.path.abspath(__file__), "--one", stl,
           "--max-tri", str(max_tri)]
    name = os.path.basename(os.path.dirname(stl))
    kw = dict(capture_output=True, text=True, timeout=timeout, env=env)
    try:
        try:
            out = subprocess.run(cmd, preexec_fn=lambda: os.nice(12), **kw)
        except (TypeError, ValueError):          # preexec_fn unsupported
            out = subprocess.run(cmd, **kw)
        for line in reversed(out.stdout.splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        return {"stl": stl, "name": name,
                "error": "no-output: " + (out.stderr.strip()[-160:] or "?")}
    except subprocess.TimeoutExpired:
        return {"stl": stl, "name": name, "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"stl": stl, "name": name, "error": f"driver: {e}"[:160]}


def _pctl(vals: list[float], p: float):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(p * len(s)))], 4)


def summarize(results: list[dict]) -> dict:
    errs = [r for r in results if r.get("error")]
    skipped = [r for r in results if r.get("route") == "skipped"]
    ok = [r for r in results if not r.get("error") and r.get("route") != "skipped"]
    drawn = [r for r in ok if r.get("route", "none") != "none"]
    none = [r for r in ok if r.get("route") == "none"]
    routes = Counter(r["route"] for r in drawn)
    devs = [r["deviation_mm"] for r in drawn if r.get("deviation_mm") is not None]
    ffs = [r["freeform_pct"] for r in none if r.get("freeform_pct") is not None]
    prismatic_none = sum(1 for f in ffs if f < 20)     # LLM-draw-loop candidates
    freeform_none = sum(1 for f in ffs if f >= 20)      # need BREP / NURBS
    return {
        "n": len(results), "ok": len(ok), "errors": len(errs),
        "skipped_too_large": len(skipped),
        "error_kinds": dict(Counter(r["error"].split(":")[0] for r in errs)),
        "draw_rate_pct": round(100 * len(drawn) / max(len(ok), 1), 1),
        "routes": dict(routes),
        "drawn_deviation_mm": {"median": _pctl(devs, 0.5), "p90": _pctl(devs, 0.9),
                               "pct_under_0.1mm": round(100 * sum(d <= 0.1 for d in devs)
                                                        / max(len(devs), 1), 1)},
        "remainder": {
            "n": len(none),
            "prismatic_ish_llm_candidates": prismatic_none,
            "freeform_heavy_need_brep": freeform_none,
            "median_freeform_pct": _pctl(ffs, 0.5),
        },
        "n_tri_median": _pctl([r["n_tri"] for r in ok if "n_tri" in r], 0.5),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Corpus eval over (mesh, STEP) pairs.")
    p.add_argument("--corpus", default=os.path.expanduser(
        "~/my-project-data/stl-step-data-v1"))
    p.add_argument("--sample", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2,
                   help="concurrent subprocesses — keep low (each loads OCP)")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--lod", default="medium", choices=["low", "medium", "high"],
                   help="mesh LOD to test (medium keeps memory bounded)")
    p.add_argument("--max-tri", type=int, default=80000,
                   help="skip meshes above this triangle count")
    p.add_argument("--out", default="eval/scoreboard.json")
    p.add_argument("--one", default=None, help="worker mode: process one STL")
    args = p.parse_args(argv)

    if args.one:
        print(json.dumps(process_one(args.one, max_tri=args.max_tri)))
        return 0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    stls = discover(args.corpus, args.lod)
    random.seed(args.seed); random.shuffle(stls)
    sample = stls[:args.sample]
    print(f"corpus: {len(stls)} paired parts ({args.lod} LOD); sampling "
          f"{len(sample)} (seed {args.seed}, {args.workers} workers, "
          f"max {args.max_tri} tri)")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_worker, s, args.timeout, args.max_tri): s
                for s in sample}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result(); results.append(r)
            tag = r.get("error") or r.get("route", "?")
            dev = f" {r.get('deviation_mm')}mm" if r.get("deviation_mm") else ""
            print(f"[{i + 1}/{len(sample)}] {tag}{dev}  {r.get('name', '')[:40]}")

    board = summarize(results)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump({"summary": board, "results": results}, fp, indent=2)

    print("\n===== SCOREBOARD =====")
    print(json.dumps(board, indent=2))
    print(f"\nfull results -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

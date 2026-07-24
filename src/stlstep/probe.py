"""Read-only mesh measurement CLI — the "measure, don't guess" toolbox.

The draw/compose loop constantly needs to extract *real* geometry from the mesh
to match a colormap-flagged region: the footprint of a feature at some height,
whether a gap is solid or void, the radius profile of a revolve, and so on. Doing
that with ad-hoc `python -c "import trimesh; ..."` one-liners is (a) error-prone
and (b) un-allowlistable — every one is a fresh arbitrary-code command that the
permission layer must gate, and any `#` comment inside the quoted script trips the
"newline + # can hide arguments" heuristic and forces a prompt regardless.

This module turns those measurements into a fixed, side-effect-free subcommand
(`python -m stlstep.probe ...`) that reads an STL and prints numbers. Nothing here
writes files, mutates state, or executes caller-supplied code, so it is safe to
allow-list once (`Bash(PYTHONPATH=src ./.venv/bin/python -m stlstep.probe:*)`) and
never see a prompt for measurement again.

Subcommands:
  bbox      overall bounds / size / centroid / volume / watertight
  sections  loops at one or more planes ⟂ an axis, with per-loop extents
  slice     material extent in a plane ⟂ an axis, optionally within a window
  profile   radius-vs-axial-position profile (for revolves / silhouettes)
  point     inside/outside solid test for one or more points (through vs blind)
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import trimesh

_AXES = {"x": 0, "y": 1, "z": 2}


def _load(stl_path: str) -> trimesh.Trimesh:
    """Load an STL as a single Trimesh (concatenating a Scene if need be).
    `process=True` only merges STL's duplicate per-triangle vertices — geometry is
    unchanged — so watertight detection and inside/outside tests are accurate."""
    m = trimesh.load(stl_path, process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise SystemExit(f"not a triangle mesh: {stl_path}")
    return m


def _axis(name: str) -> int:
    if name not in _AXES:
        raise SystemExit(f"--axis must be x/y/z, got {name!r}")
    return _AXES[name]


def _perp(ai: int) -> tuple[int, int]:
    return tuple(i for i in (0, 1, 2) if i != ai)  # type: ignore[return-value]


def _fmt(v) -> str:
    a = np.asarray(v, dtype=float).ravel()
    return "[" + ", ".join(f"{x:.3f}" for x in a) + "]"


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_bbox(m: trimesh.Trimesh, args) -> dict:
    lo, hi = np.asarray(m.bounds[0]), np.asarray(m.bounds[1])
    out = {
        "min_mm": lo.tolist(),
        "max_mm": hi.tolist(),
        "size_mm": (hi - lo).tolist(),
        "centroid_mm": np.asarray(m.centroid).tolist(),
        "volume_mm3": float(m.volume),
        "watertight": bool(m.is_watertight),
        "n_faces": int(len(m.faces)),
        "n_vertices": int(len(m.vertices)),
    }
    if not args.json:
        print(f"bbox min  {_fmt(out['min_mm'])}")
        print(f"     max  {_fmt(out['max_mm'])}")
        print(f"     size {_fmt(out['size_mm'])} mm")
        print(f"centroid  {_fmt(out['centroid_mm'])}")
        print(f"volume    {out['volume_mm3']:.3f} mm^3   "
              f"watertight={out['watertight']}")
        print(f"faces {out['n_faces']}  vertices {out['n_vertices']}")
    return out


def _section_loops(m, ai, coord):
    """Closed polylines where the plane {axis == coord} cuts the mesh."""
    origin = m.centroid.copy()
    origin[ai] = coord
    normal = np.zeros(3)
    normal[ai] = 1.0
    sec = m.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return []
    return list(sec.discrete)


def cmd_sections(m: trimesh.Trimesh, args) -> dict:
    ai = _axis(args.axis)
    lo, hi = float(m.bounds[0][ai]), float(m.bounds[1][ai])
    coords = [float(x) for x in args.at.split(",") if x.strip() != ""]
    if args.frac:
        coords = [lo + f * (hi - lo) for f in coords]
    u, v = _perp(ai)
    names = "xyz"
    result = {"axis": args.axis, "planes": []}
    for c in coords:
        loops = _section_loops(m, ai, c)
        plane = {"at_mm": c, "n_loops": len(loops), "loops": []}
        for lp in loops:
            lp = np.asarray(lp, dtype=float)
            uu, vv = lp[:, u], lp[:, v]
            plane["loops"].append({
                f"{names[u]}_min": float(uu.min()), f"{names[u]}_max": float(uu.max()),
                f"{names[v]}_min": float(vv.min()), f"{names[v]}_max": float(vv.max()),
                "centroid": [float(uu.mean()), float(vv.mean())],
            })
        result["planes"].append(plane)
    if not args.json:
        for pl in result["planes"]:
            print(f"{args.axis}={pl['at_mm']:.3f}: {pl['n_loops']} loop(s)")
            for lp in pl["loops"]:
                print(f"    {names[u]}[{lp[names[u]+'_min']:.2f},{lp[names[u]+'_max']:.2f}]"
                      f" {names[v]}[{lp[names[v]+'_min']:.2f},{lp[names[v]+'_max']:.2f}]"
                      f"  c={_fmt(lp['centroid'])}")
    return result


def cmd_slice(m: trimesh.Trimesh, args) -> dict:
    ai = _axis(args.axis)
    u, v = _perp(ai)
    names = "xyz"
    loops = _section_loops(m, ai, args.at)
    pts = np.vstack([np.asarray(lp, dtype=float) for lp in loops]) if loops \
        else np.empty((0, 3))
    if args.window:
        w = [float(x) for x in args.window.split(",")]
        if len(w) != 4:
            raise SystemExit("--window needs umin,umax,vmin,vmax (4 numbers)")
        umin, umax, vmin, vmax = w
        mask = ((pts[:, u] >= umin) & (pts[:, u] <= umax) &
                (pts[:, v] >= vmin) & (pts[:, v] <= vmax))
        pts = pts[mask]
    out = {"axis": args.axis, "at_mm": args.at, "has_material": bool(len(pts))}
    if len(pts):
        out.update({
            f"{names[u]}_min": float(pts[:, u].min()), f"{names[u]}_max": float(pts[:, u].max()),
            f"{names[v]}_min": float(pts[:, v].min()), f"{names[v]}_max": float(pts[:, v].max()),
        })
    if not args.json:
        if not len(pts):
            print(f"{args.axis}={args.at}: NO material"
                  + (" in window" if args.window else ""))
        else:
            print(f"{args.axis}={args.at}: material "
                  f"{names[u]}[{out[names[u]+'_min']:.2f},{out[names[u]+'_max']:.2f}] "
                  f"{names[v]}[{out[names[v]+'_min']:.2f},{out[names[v]+'_max']:.2f}]")
    return out


def cmd_profile(m: trimesh.Trimesh, args) -> dict:
    ai = _axis(args.axis)
    u, v = _perp(ai)
    lo, hi = float(m.bounds[0][ai]), float(m.bounds[1][ai])
    if args.center:
        cu, cv = (float(x) for x in args.center.split(","))
    else:
        c = m.centroid
        cu, cv = float(c[u]), float(c[v])
    n = max(2, args.n)
    stations = np.linspace(lo, hi, n)
    names = "xyz"
    rows = []
    for s in stations:
        loops = _section_loops(m, ai, float(s))
        if not loops:
            rows.append({"at_mm": float(s), "r_min": None, "r_max": None})
            continue
        pts = np.vstack([np.asarray(lp, dtype=float) for lp in loops])
        r = np.hypot(pts[:, u] - cu, pts[:, v] - cv)
        rows.append({"at_mm": float(s), "r_min": float(r.min()),
                     "r_max": float(r.max())})
    out = {"axis": args.axis, "center": [cu, cv], "stations": rows}
    if not args.json:
        print(f"radius profile along {args.axis} about "
              f"({names[u]}={cu:.2f},{names[v]}={cv:.2f}):")
        for row in rows:
            if row["r_max"] is None:
                print(f"  {args.axis}={row['at_mm']:7.2f}   (empty)")
            else:
                print(f"  {args.axis}={row['at_mm']:7.2f}   "
                      f"r_out={row['r_max']:6.2f}  r_in={row['r_min']:6.2f}")
    return out


def cmd_point(m: trimesh.Trimesh, args) -> dict:
    pts = []
    for tok in args.points:
        xyz = [float(x) for x in tok.split(",")]
        if len(xyz) != 3:
            raise SystemExit(f"point must be x,y,z — got {tok!r}")
        pts.append(xyz)
    arr = np.asarray(pts, dtype=float)
    inside = m.contains(arr)
    out = {"watertight": bool(m.is_watertight),
           "points": [{"xyz": p, "inside": bool(i)}
                      for p, i in zip(pts, inside)]}
    if not args.json:
        if not m.is_watertight:
            print("WARNING: mesh not watertight — inside/outside may be unreliable")
        for rec in out["points"]:
            print(f"  {_fmt(rec['xyz'])}  ->  "
                  f"{'INSIDE (solid)' if rec['inside'] else 'outside (void)'}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="stlstep.probe",
        description="Read-only mesh measurements (measure, don't guess).")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bbox", help="overall bounds/size/volume/watertight")
    b.add_argument("stl"); b.set_defaults(fn=cmd_bbox)

    s = sub.add_parser("sections", help="loops at planes ⟂ an axis, with extents")
    s.add_argument("stl"); s.add_argument("--axis", required=True)
    s.add_argument("--at", required=True,
                   help="comma list of coordinates (or fractions with --frac)")
    s.add_argument("--frac", action="store_true",
                   help="treat --at values as fractions 0..1 of the axis span")
    s.set_defaults(fn=cmd_sections)

    sl = sub.add_parser("slice", help="material extent in one plane ⟂ an axis")
    sl.add_argument("stl"); sl.add_argument("--axis", required=True)
    sl.add_argument("--at", required=True, type=float)
    sl.add_argument("--window", default=None,
                    help="umin,umax,vmin,vmax on the two perpendicular axes; "
                         "use = form for negatives (--window=-76,-54,0,20)")
    sl.set_defaults(fn=cmd_slice)

    pr = sub.add_parser("profile", help="radius vs axial position (revolves)")
    pr.add_argument("stl"); pr.add_argument("--axis", required=True)
    pr.add_argument("--n", type=int, default=40, help="number of stations")
    pr.add_argument("--center", default=None,
                    help="u,v center in the perpendicular plane (default centroid)")
    pr.set_defaults(fn=cmd_profile)

    pt = sub.add_parser("point", help="inside/outside solid test (through vs blind)")
    pt.add_argument("stl"); pt.add_argument("points", nargs="+",
                                            help="one or more x,y,z")
    pt.set_defaults(fn=cmd_point)

    args = p.parse_args(argv)
    m = _load(args.stl)
    out = args.fn(m, args)
    if args.json:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

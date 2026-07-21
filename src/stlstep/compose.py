"""The LLM composition path — "the LLM draws the part".

This is the img2threejs loop applied to CAD reconstruction. The deterministic
recognizers in `draw.py` handle parts that ARE a single extrude/revolve or a
prismatic stack. Everything else — cross-axis holes, fillets, parts that are an
unusual composition of features — is where judgment is needed, and that is what
an LLM is uniquely good at.

The division of labour (the project's core principle): the LLM never emits
geometry directly. It STUDIES the part (a structured, dimensioned analysis +
rendered views) and COMPOSES a build123d feature script — sketch / extrude /
revolve / holes / fillet operations. The kernel (build123d/OCCT) executes it
deterministically; the gate verifies it against the mesh and renders a deviation
colormap; the LLM reads that feedback and refines. Bounded iterations, then fall
back to BREP.

Pieces:
  analyze(mesh)          -> structured dimensioned feature report (the facts)
  render_views(mesh)     -> shaded multiview PNG (the picture)
  build_brief(mesh, dir) -> Brief: the drawing task package handed to the LLM
  run_candidate(script)  -> Candidate: exec + verify + render a proposed script
  compose_loop(mesh, composer) -> best Candidate (generate -> verify -> refine)

`Composer` is pluggable: an Anthropic-API backend for headless runs, or an
in-session driver where the operator's own model composes (see compose_demo.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .ingest import build_face_neighbors
from .segment_rg import segment_rg


# ----------------------------------------------------------------------------
# Analysis — the dimensioned feature report the LLM reasons over.
# ----------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 1e-12 else v.tolist()


def _r(x, n=3):
    """Round for display: scalars, lists, nested lists."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, n) for v in np.asarray(x).tolist()]
    return round(float(x), n)


def _cylinder_desc(mesh, region):
    """Axis, radius, axial extent, and hole-vs-boss orientation for a cylinder
    region — the facts an engineer reads off before drawing a hole/boss/shaft."""
    p = region.params
    axis = np.asarray(p["axis"], dtype=float); axis /= np.linalg.norm(axis)
    center = np.asarray(p["center_mm"], dtype=float)
    radius = float(p["radius_mm"])
    ids = np.asarray(region.face_ids, dtype=np.int64)
    tri = mesh.vertices[mesh.faces[ids]]
    pts = tri.reshape(-1, 3)
    t = (pts - center) @ axis
    lo, hi = float(t.min()), float(t.max())
    end_lo = (center + lo * axis).tolist()
    end_hi = (center + hi * axis).tolist()
    # orientation: mean face-normal dot outward radial (area-weighted)
    normals = np.asarray(mesh.face_normals[ids])
    cents = tri.mean(axis=1)
    rad = cents - center
    rad = rad - (rad @ axis)[:, None] * axis
    rn = np.linalg.norm(rad, axis=1)
    rad = rad / np.where(rn > 1e-9, rn, 1.0)[:, None]
    areas = np.asarray(mesh.area_faces[ids])
    score = float((np.sum(normals * rad, axis=1) * areas).sum() / max(areas.sum(), 1e-9))
    orient = "boss/shaft (convex)" if score > 0.3 else \
             "hole/bore (concave)" if score < -0.3 else "uncertain"
    return {
        "kind": "cylinder",
        "orientation": orient,
        "diameter_mm": _r(2 * radius),
        "radius_mm": _r(radius),
        "axis": _r(axis, 4),
        "length_mm": _r(hi - lo),
        "end_a_mm": _r(end_lo),
        "end_b_mm": _r(end_hi),
        "area_mm2": _r(float(areas.sum()), 1),
    }


def analyze(mesh: trimesh.Trimesh) -> dict:
    """Segment the mesh and describe it as a dimensioned list of surfaces plus
    what the deterministic recognizers make of it. This is the factual half of
    the brief the LLM draws from."""
    from . import draw as dw
    neighbors = build_face_neighbors(mesh)
    seg = segment_rg(mesh, neighbors)
    areas = np.asarray(mesh.area_faces)

    surfaces = []
    for region in sorted(seg.regions, key=lambda r: -areas[r.face_ids].sum()):
        p = region.params
        a = float(areas[region.face_ids].sum())
        if region.kind == "plane":
            n = _unit(p["normal"])
            cent = mesh.vertices[mesh.faces[region.face_ids]].reshape(-1, 3).mean(axis=0)
            surfaces.append({"kind": "plane", "area_mm2": _r(a, 1),
                             "normal": _r(n, 4), "point_mm": _r(cent)})
        elif region.kind == "cylinder":
            surfaces.append(_cylinder_desc(mesh, region))
        elif region.kind == "cone":
            surfaces.append({"kind": "cone", "area_mm2": _r(a, 1),
                             "axis": _r(_unit(p["axis"]), 4),
                             "apex_mm": _r(p["apex_mm"]),
                             "semi_angle_deg": _r(np.degrees(p["semi_angle_rad"])),
                             "ref_radius_mm": _r(p["ref_radius_mm"])})
        elif region.kind == "sphere":
            surfaces.append({"kind": "sphere", "area_mm2": _r(a, 1),
                             "center_mm": _r(p["center_mm"]),
                             "radius_mm": _r(p["radius_mm"])})
        elif region.kind == "torus":
            surfaces.append({"kind": "torus", "area_mm2": _r(a, 1),
                             "center_mm": _r(p["center_mm"]),
                             "axis": _r(_unit(p["axis"]), 4),
                             "major_radius_mm": _r(p["major_radius_mm"]),
                             "minor_radius_mm": _r(p["minor_radius_mm"])})
        else:
            bb = mesh.vertices[mesh.faces[region.face_ids]].reshape(-1, 3)
            surfaces.append({"kind": "freeform", "area_mm2": _r(a, 1),
                             "bbox_min_mm": _r(bb.min(axis=0)),
                             "bbox_max_mm": _r(bb.max(axis=0))})

    # what the deterministic recognizers see (hints, not commands)
    recog = {"extrude": False, "revolve": False, "prismatic": None}
    try:
        recog["extrude"] = dw.detect_extrusion(mesh) is not None
    except Exception:  # noqa: BLE001
        pass
    try:
        recog["revolve"] = dw.detect_revolve(mesh) is not None
    except Exception:  # noqa: BLE001
        pass
    try:
        pr = dw.detect_prismatic(mesh)
        if pr is not None:
            recog["prismatic"] = {"axis": "xyz"[pr.axis],
                                  "level_bounds": _r([lo for lo, _, _ in pr.levels]
                                                     + [pr.levels[-1][1]])}
    except Exception:  # noqa: BLE001
        pass

    lo, hi = mesh.bounds
    return {
        "units": "mm",
        "bounding_box": {"min": _r(lo), "max": _r(hi), "size": _r(hi - lo)},
        "volume_mm3": (_r(float(mesh.volume), 1) if mesh.is_watertight else None),
        "watertight": bool(mesh.is_watertight),
        "n_triangles": int(len(mesh.faces)),
        "surfaces": surfaces,
        "deterministic_recognizers": recog,
    }


# ----------------------------------------------------------------------------
# Rendering — the picture the LLM sees.
# ----------------------------------------------------------------------------

def render_views(mesh: trimesh.Trimesh, path: str, *, title: str = "",
                 color=None) -> str:
    """Shaded multiview render of a mesh (matplotlib, offscreen). Faces are
    lambert-shaded by normal so form reads clearly. `color` overrides the base
    RGB (e.g. per-face deviation colours)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = mesh.triangles
    fn = np.asarray(mesh.face_normals)
    light = np.array([0.4, 0.3, 0.85]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(np.abs(fn @ light), 0, 1)
    base = np.array([0.55, 0.62, 0.72]) if color is None else np.asarray(color)
    if base.ndim == 1:
        facecolors = np.clip(base[None, :] * shade[:, None], 0, 1)
    else:
        facecolors = np.clip(base * shade[:, None], 0, 1)

    views = [(22, -60), (22, 45), (89, -90), (5, 0)]
    labels = ["iso 1", "iso 2", "top (−Z)", "front (+X)"]
    fig = plt.figure(figsize=(16, 4.2))
    b = mesh.bounds
    ext = np.asarray(b[1]) - np.asarray(b[0]); ext = np.where(ext > 1e-9, ext, 1e-9)
    for i, ((elev, azim), lab) in enumerate(zip(views, labels)):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=facecolors,
                                             edgecolors="k", linewidths=0.05))
        ax.set_xlim(b[0][0], b[1][0]); ax.set_ylim(b[0][1], b[1][1])
        ax.set_zlim(b[0][2], b[1][2])
        try:
            ax.set_box_aspect(ext)
        except Exception:  # noqa: BLE001
            pass
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title(lab, fontsize=9)
        ax.tick_params(labelsize=6)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# The brief — the drawing task package handed to the composer.
# ----------------------------------------------------------------------------

@dataclass
class Brief:
    workdir: str
    analysis: dict
    views_png: str
    mesh: trimesh.Trimesh

    def prompt_text(self) -> str:
        """The textual half of the drawing task (analysis + instructions)."""
        return (DRAW_INSTRUCTIONS + "\n\n## Part analysis\n\n```json\n"
                + json.dumps(self.analysis, indent=2) + "\n```\n")


DRAW_INSTRUCTIONS = """\
You are reconstructing a mechanical part as clean, editable CAD. You are given a
dimensioned analysis of a triangle mesh and shaded rendered views of it. Write a
build123d Python script that draws the part as a feature model (sketches +
extrude/revolve/holes/fillets), matching the real geometry.

Rules:
- Output ONLY a Python code block: a build123d script. No prose outside it.
- Put the final solid in a BuildPart context named `part` (i.e. `with BuildPart()
  as part:`), exactly as build123d examples do. `from build123d import *` first.
- Work in millimetres and in the SAME world coordinates as the analysis (so the
  result overlays the mesh). Use the reported axes, radii, centres and extents.
- Prefer the simplest faithful feature tree. Model holes as cuts, bosses as adds,
  fillets/chamfers with `fillet`/`chamfer` on the relevant edges.
- Don't invent features that aren't supported by the analysis or the views.
- You will get back a deviation colormap and metrics; fix the largest errors and
  resubmit. Aim for max deviation under 0.1 mm.
"""


def build_brief(mesh: trimesh.Trimesh, workdir: str) -> Brief:
    os.makedirs(workdir, exist_ok=True)
    analysis = analyze(mesh)
    size = analysis["bounding_box"]["size"]
    views = os.path.join(workdir, "part_views.png")
    render_views(mesh, views, title=f"Target part  —  bbox {size} mm")
    with open(os.path.join(workdir, "analysis.json"), "w") as fp:
        json.dump(analysis, fp, indent=2)
    with open(os.path.join(workdir, "brief.md"), "w") as fp:
        fp.write(f"# Drawing brief\n\nViews: `part_views.png`\n\n"
                 + Brief(workdir, analysis, views, mesh).prompt_text())
    return Brief(workdir, analysis, views, mesh)


# ----------------------------------------------------------------------------
# Execute + verify a proposed script.
# ----------------------------------------------------------------------------

@dataclass
class Candidate:
    ok: bool
    script: str
    error: str | None = None
    part: object = None
    deviation_mm: float | None = None
    reverse_max_mm: float | None = None
    valid: bool | None = None
    closed: bool | None = None
    volume_mm3: float | None = None
    render_png: str | None = None
    colormap_png: str | None = None       # model coloured by dist to mesh (extra)
    colormap_fwd_png: str | None = None   # mesh coloured by dist to model (missing)
    metrics: dict = field(default_factory=dict)

    def feedback(self, mesh_volume: float | None) -> str:
        """Human/LLM-readable verdict used to drive the next refinement."""
        if not self.ok:
            return f"BUILD FAILED:\n{self.error}"
        vol = (f"{self.volume_mm3:.1f}" if self.volume_mm3 is not None else "?")
        mv = (f"{mesh_volume:.1f}" if mesh_volume else "?")
        L = [f"max deviation (mesh->model): {self.deviation_mm:.4f} mm",
             f"max reverse (model->mesh):   {self.reverse_max_mm:.4f} mm",
             f"volume: {vol} mm^3  (mesh {mv} mm^3)",
             f"valid solid: {self.valid}   closed: {self.closed}",
             "Two deviation colormaps are attached: 'missing' (the target mesh "
             "coloured by distance to your model — red = a feature you have NOT "
             "modelled, e.g. an omitted hole/pocket) and 'extra' (your model "
             "coloured by distance to the mesh — red = material you added that "
             "isn't there). Fix the largest red regions first."]
        return "\n".join(L)


def _extract_part(ns):
    """Pull a build123d Part out of an executed script namespace, tolerant of
    naming: a BuildPart named `part`, or any Part/Solid left in `part`/`result`."""
    from build123d import BuildPart, Part, Solid, Compound
    for key in ("part", "result", "p"):
        v = ns.get(key)
        if isinstance(v, BuildPart):
            return v.part
        if isinstance(v, (Part, Solid, Compound)):
            return v
    for v in ns.values():   # last resort: first BuildPart in the namespace
        if isinstance(v, BuildPart):
            return v.part
    return None


def exec_script(script: str):
    """Execute a build123d script and return its Part (raises on failure)."""
    import build123d
    ns = {"__name__": "_candidate"}
    exec(compile(script, "<candidate>", "exec"), {**vars(build123d), **ns}, ns)
    part = _extract_part(ns)
    if part is None:
        raise ValueError("script produced no build123d Part "
                         "(leave it in `with BuildPart() as part:`)")
    if getattr(part, "volume", 0) <= 0:
        raise ValueError("script produced an empty/zero-volume part")
    return part


def run_candidate(script: str, mesh: trimesh.Trimesh, workdir: str,
                  *, tag: str = "cand", dev_samples: int = 40000) -> Candidate:
    """Execute a proposed script, measure it against the mesh, and render both a
    shaded view and a deviation colormap. Never raises — a failure comes back as
    a Candidate with ok=False and the error text (that IS the refine signal)."""
    from . import occ_util as ou
    os.makedirs(workdir, exist_ok=True)
    try:
        part = exec_script(script)
    except Exception as e:  # noqa: BLE001
        import traceback
        return Candidate(ok=False, script=script,
                         error="".join(traceback.format_exception_only(type(e), e)).strip()
                               + "\n" + traceback.format_exc(limit=3))

    shape = part.wrapped
    rv, rf = ou.shape_to_mesh(shape, 0.05)
    gen = trimesh.Trimesh(vertices=rv, faces=rf, process=False)

    pts, _ = trimesh.sample.sample_surface(mesh, dev_samples)
    _, fwd, _ = trimesh.proximity.closest_point(gen, np.asarray(pts))
    gpts, _ = trimesh.sample.sample_surface(gen, min(dev_samples, 20000))
    _, rev, _ = trimesh.proximity.closest_point(mesh, np.asarray(gpts))

    cand = Candidate(
        ok=True, script=script, part=part,
        deviation_mm=float(np.quantile(fwd, 0.99)),
        reverse_max_mm=float(np.quantile(rev, 0.99)),
        valid=bool(ou.is_valid(shape)), closed=_is_solid(shape),
        volume_mm3=float(part.volume),
        metrics={"fwd_max": float(fwd.max()), "rev_max": float(rev.max()),
                 "brep_faces": ou.count_faces(shape)},
    )
    # renders: candidate shaded, + candidate coloured by deviation to the mesh
    cand.render_png = render_views(gen, os.path.join(workdir, f"{tag}_model.png"),
                                   title=f"Candidate model ({tag})")
    # 'extra': model coloured by distance to the mesh (added material shows).
    cand.colormap_png = _render_deviation(gen, mesh,
                                          os.path.join(workdir, f"{tag}_extra.png"))
    # 'missing': mesh coloured by distance to the model (omitted features show).
    cand.colormap_fwd_png = _render_deviation(mesh, gen,
                                              os.path.join(workdir, f"{tag}_missing.png"))
    return cand


def _render_deviation(gen, mesh, path):
    from .verify import render_colormap
    try:
        return render_colormap(gen, mesh, path)
    except Exception:  # noqa: BLE001
        return None


def _is_solid(shape) -> bool:
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    return TopExp_Explorer(shape, TopAbs_SOLID).More()


# ----------------------------------------------------------------------------
# The generate -> verify -> refine loop.
# ----------------------------------------------------------------------------

class Composer:
    """Interface for whatever writes the build123d script. `propose` returns a
    script given the brief and the history of prior (Candidate) attempts."""

    def propose(self, brief: Brief, history: list[Candidate]) -> str:
        raise NotImplementedError


def extract_code(text: str) -> str:
    """Pull the first ```python ...``` block out of an LLM reply (or return the
    whole thing if it's already bare code)."""
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _img_block(path: str) -> dict:
    import base64
    with open(path, "rb") as fp:
        data = base64.standard_b64encode(fp.read()).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data}}


class FileComposer(Composer):
    """A composer that replays a fixed list of scripts — for driving the loop
    from hand-written attempts (the in-session operator) or for deterministic
    tests of the harness without an API. Repeats the last script once exhausted."""

    def __init__(self, scripts: list[str]):
        self.scripts = list(scripts)
        self.i = 0

    def propose(self, brief: Brief, history: list[Candidate]) -> str:
        s = self.scripts[min(self.i, len(self.scripts) - 1)]
        self.i += 1
        return s


class AnthropicComposer(Composer):
    """The headless "LLM draws it" backend: hands the analysis + rendered views
    to Claude and asks for a build123d script, feeding each attempt's deviation
    colormap back as a multi-turn conversation so the model refines. Requires the
    `anthropic` SDK and an API key (ANTHROPIC_API_KEY or passed in)."""

    def __init__(self, *, model: str = "claude-opus-4-8", api_key: str | None = None,
                 max_tokens: int = 4096):
        import anthropic
        self.client = (anthropic.Anthropic(api_key=api_key) if api_key
                       else anthropic.Anthropic())
        self.model = model
        self.max_tokens = max_tokens

    def propose(self, brief: Brief, history: list[Candidate]) -> str:
        mesh_vol = float(brief.mesh.volume) if brief.mesh.is_watertight else None
        messages = [{"role": "user", "content": [
            {"type": "text", "text": brief.prompt_text()},
            _img_block(brief.views_png)]}]
        for cand in history:
            messages.append({"role": "assistant",
                             "content": "```python\n" + cand.script + "\n```"})
            fb = [{"type": "text", "text": cand.feedback(mesh_vol)
                   + "\n\nRevise the script to reduce the deviation."}]
            if cand.ok:
                if cand.colormap_fwd_png:
                    fb.append({"type": "text", "text": "'missing' colormap:"})
                    fb.append(_img_block(cand.colormap_fwd_png))
                if cand.colormap_png:
                    fb.append({"type": "text", "text": "'extra' colormap:"})
                    fb.append(_img_block(cand.colormap_png))
            messages.append({"role": "user", "content": fb})
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=DRAW_INSTRUCTIONS, messages=messages)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return extract_code(text)


@dataclass
class ComposeResult:
    best: Candidate | None
    attempts: list[Candidate]
    passed: bool


def compose_loop(mesh: trimesh.Trimesh, composer: Composer, workdir: str,
                 *, max_iters: int = 4, dev_tol: float = 0.1) -> ComposeResult:
    """Run the img2threejs loop: build a brief, ask the composer for a script,
    execute+verify+render it, feed the verdict back, and iterate until a valid
    closed solid meets `dev_tol` (or the iteration budget runs out)."""
    brief = build_brief(mesh, workdir)
    mesh_vol = float(mesh.volume) if mesh.is_watertight else None
    attempts: list[Candidate] = []
    best: Candidate | None = None
    for i in range(max_iters):
        script = composer.propose(brief, attempts)
        cand = run_candidate(script, mesh, workdir, tag=f"iter{i}")
        attempts.append(cand)
        with open(os.path.join(workdir, f"iter{i}.py"), "w") as fp:
            fp.write(script)
        if cand.ok:
            passing = (cand.valid and cand.closed and
                       cand.deviation_mm <= dev_tol and
                       cand.reverse_max_mm <= dev_tol)
            if best is None or (cand.ok and (not best.ok or
                    cand.deviation_mm < (best.deviation_mm or 1e9))):
                best = cand
            if passing:
                return ComposeResult(best=cand, attempts=attempts, passed=True)
    return ComposeResult(best=best, attempts=attempts, passed=False)


# ----------------------------------------------------------------------------
# CLI — the steps the `stl-draw` skill drives (with Claude as the composer).
# ----------------------------------------------------------------------------

def _surf_line(s: dict) -> str:
    k = s["kind"]
    if k == "plane":
        return f"plane   normal={s['normal']} at {s['point_mm']}  area={s['area_mm2']}"
    if k == "cylinder":
        return (f"cylinder {s['orientation']:22s} Ø{s['diameter_mm']} "
                f"axis={s['axis']} len={s['length_mm']} {s['end_a_mm']}->{s['end_b_mm']}")
    if k == "cone":
        return (f"cone    Ø_ref {2 * s['ref_radius_mm']:.3f} half={s['semi_angle_deg']}° "
                f"axis={s['axis']} apex={s['apex_mm']}")
    if k == "sphere":
        return f"sphere  r={s['radius_mm']} center={s['center_mm']}"
    if k == "torus":
        return (f"torus   R={s['major_radius_mm']} r={s['minor_radius_mm']} "
                f"axis={s['axis']} center={s['center_mm']}")
    return f"freeform (curved)  bbox {s['bbox_min_mm']}->{s['bbox_max_mm']}  area={s['area_mm2']}"


def _cli_brief(args) -> int:
    from .ingest import load_and_repair
    ing = load_and_repair(args.stl)
    brief = build_brief(ing.mesh, args.workdir)
    a = brief.analysis
    print(f"brief written to {args.workdir}/")
    print(f"  analysis : {args.workdir}/analysis.json")
    print(f"  views    : {brief.views_png}   <- Read this image")
    print(f"  bbox {a['bounding_box']['size']} mm  min {a['bounding_box']['min']}  "
          f"volume {a['volume_mm3']} mm^3  watertight={a['watertight']}")
    print(f"  deterministic recognizers: {a['deterministic_recognizers']}")
    print(f"  {len(a['surfaces'])} surfaces:")
    for s in a["surfaces"]:
        print("    " + _surf_line(s))
    return 0


def _cli_run(args) -> int:
    from .ingest import load_and_repair
    with open(args.script) as fp:
        script = fp.read()
    ing = load_and_repair(args.stl)
    cand = run_candidate(script, ing.mesh, args.workdir, tag=args.tag)
    mv = float(ing.mesh.volume) if ing.mesh.is_watertight else None
    print(cand.feedback(mv))
    passed = bool(cand.ok and cand.valid and cand.closed and
                  cand.deviation_mm <= args.tol and cand.reverse_max_mm <= args.tol)
    if cand.ok:
        print(f"colormaps: MISSING={cand.colormap_fwd_png}  "
              f"EXTRA={cand.colormap_png}   <- Read these images")
        print(f"model view: {cand.render_png}")
        with open(os.path.join(args.workdir, "metrics.json"), "w") as fp:
            json.dump({"deviation_mm": cand.deviation_mm,
                       "reverse_max_mm": cand.reverse_max_mm,
                       "volume_mm3": cand.volume_mm3, "valid": cand.valid,
                       "closed": cand.closed, "passed": passed,
                       "metrics": cand.metrics}, fp, indent=2)
        if args.out:
            from build123d import export_step
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                        exist_ok=True)
            export_step(cand.part, args.out)
            with open(os.path.join(args.workdir, "feature_tree.py"), "w") as fp:
                fp.write(script)
            print(f"exported {args.out}  +  {args.workdir}/feature_tree.py")
    print(f"STATUS: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="LLM draw-it harness: build a brief, run+verify a script.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("brief", help="analyze an STL and render views for drawing")
    b.add_argument("stl"); b.add_argument("workdir")
    b.set_defaults(fn=_cli_brief)

    r = sub.add_parser("run", help="execute + verify a proposed build123d script")
    r.add_argument("stl"); r.add_argument("workdir"); r.add_argument("script")
    r.add_argument("--tag", default="iter0")
    r.add_argument("--tol", type=float, default=0.1)
    r.add_argument("--out", default=None, help="export STEP here if it builds")
    r.set_defaults(fn=_cli_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())

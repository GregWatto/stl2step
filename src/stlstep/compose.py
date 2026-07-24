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


def detect_edge_treatments(mesh, seg, neighbors) -> dict:
    """Approximate fillet/chamfer detection from mesh curvature (the segmenter
    grows THROUGH tangent fillets, so they hide inside freeform regions — this
    recovers them independently). A HINT, not ground truth: fillet radii come out
    ~10% high and chamfer widths aren't measured, so the composer confirms exact
    values against the cross-sections. Fillets = connected bands of rounded convex
    face-pairs that fit a small cylinder tightly; chamfers = flat bands whose
    normal is oblique to every world axis (and which aren't fillet faces)."""
    from .segment_rg import _fit_cylinder, _patch_points
    from .ingest import connected_components
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
    fn = np.asarray(mesh.face_normals)
    try:
        ang = np.degrees(np.asarray(mesh.face_adjacency_angles))
        conv = np.asarray(mesh.face_adjacency_convex)
        pairs = np.asarray(mesh.face_adjacency)
    except Exception:  # noqa: BLE001 — non-manifold mesh: no adjacency angles
        return {"fillets": [], "chamfers": []}
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    analytic = {int(f) for r in seg.regions if r.kind != "plane"
                and r.kind != "freeform" for f in r.face_ids}

    rounded = (ang > 2) & (ang < 45) & conv
    fillet_faces = {int(f) for f in np.unique(pairs[rounded])}
    radii = []
    for band in connected_components([f for f in fillet_faces if f not in analytic],
                                     neighbors):
        if len(band) < 10:
            continue
        fit = _fit_cylinder(*_patch_points(V, F, fn, band))
        if fit is None:
            continue
        params, resid = fit
        r = float(params["radius_mm"])
        if 0.2 < r < 0.35 * diag and resid < 0.06 * r:   # tight resid = a real round
            radii.append(r)

    oblique = np.abs(fn).max(axis=1) < 0.94
    chamfer_bands = [b for b in connected_components(
        [int(f) for f in np.where(oblique)[0]
         if int(f) not in analytic and int(f) not in fillet_faces], neighbors)
        if len(b) >= 6]
    return {"fillets": _cluster_radii(radii),
            "chamfers": ([{"count": len(chamfer_bands)}] if chamfer_bands else [])}


def _cluster_radii(radii: list[float]) -> list[dict]:
    """Group nearby fillet radii (within 15%) -> [{radius_mm, count}]."""
    out = []
    for r in sorted(radii):
        if out and r <= out[-1]["_sum"] / out[-1]["count"] * 1.15:
            out[-1]["_sum"] += r; out[-1]["count"] += 1
        else:
            out.append({"_sum": r, "count": 1})
    return [{"radius_mm": _r(g["_sum"] / g["count"]), "count": g["count"]}
            for g in out]


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

    try:
        edge_treatments = detect_edge_treatments(mesh, seg, neighbors)
    except Exception:  # noqa: BLE001 — a hint; never fatal
        edge_treatments = {"fillets": [], "chamfers": []}

    lo, hi = mesh.bounds
    try:
        from .features import group_features
        features = group_features(mesh, seg, surfaces, (hi - lo).tolist())
    except Exception:  # noqa: BLE001 — grouping is a convenience; never fatal
        features = None

    return {
        "units": "mm",
        "bounding_box": {"min": _r(lo), "max": _r(hi), "size": _r(hi - lo)},
        "volume_mm3": (_r(float(mesh.volume), 1) if mesh.is_watertight else None),
        "watertight": bool(mesh.is_watertight),
        "n_triangles": int(len(mesh.faces)),
        "features": features,
        "edge_treatments": edge_treatments,
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


def dominant_axis(mesh: trimesh.Trimesh, analysis: dict) -> int:
    """The part's main axis (0/1/2): the world axis most cylinder area aligns
    with, else the prismatic-stack axis, else the longest bbox dimension. Used to
    orient the cross-sections."""
    score = [0.0, 0.0, 0.0]
    for s in analysis["surfaces"]:
        if s["kind"] == "cylinder":
            ax = np.abs(np.asarray(s["axis"], dtype=float))
            i = int(np.argmax(ax))
            if ax[i] > 0.9:
                score[i] += float(s.get("area_mm2", 0) or 0)
    if max(score) > 0:
        return int(np.argmax(score))
    pr = analysis.get("deterministic_recognizers", {}).get("prismatic")
    if pr:
        return "xyz".index(pr["axis"])
    return int(np.argmax(analysis["bounding_box"]["size"]))


def render_sections(mesh: trimesh.Trimesh, path: str, axis: int,
                    *, title: str = "") -> str:
    """Cross-section line drawings — the view that reveals INTERNAL geometry
    (cavity floors, counterbores, wall thicknesses) that shaded 3D renders hide.
    Two axial slices (planes containing `axis`, seen from the two perpendicular
    directions) + two transverse slices (perpendicular to `axis`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["X", "Y", "Z"]
    c = 0.5 * (np.asarray(mesh.bounds[0]) + np.asarray(mesh.bounds[1]))
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    others = [i for i in (0, 1, 2) if i != axis]

    panels = []
    # two axial slices: plane normal = each perpendicular axis, through centre
    for j in others:
        k = [i for i in others if i != j][0]        # in-plane horizontal axis
        origin = c.copy()
        panels.append(("axial", tuple(origin), j, k, axis,
                       f"axial slice (normal {names[j]}):  {names[k]} vs {names[axis]} (axis)"))
    # two transverse slices at 1/3 and 2/3 of the axis span
    for frac in (0.34, 0.66):
        origin = c.copy(); origin[axis] = lo + frac * (hi - lo)
        panels.append(("trans", tuple(origin), axis, others[0], others[1],
                       f"cross slice at {names[axis]}={origin[axis]:.1f}:  "
                       f"{names[others[0]]} vs {names[others[1]]}"))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, (kind, origin, jn, hax, vax, lab) in zip(axes, panels):
        normal = [0, 0, 0]; normal[jn] = 1
        sec = mesh.section(plane_origin=list(origin), plane_normal=normal)
        if sec is not None:
            for ent in sec.discrete:
                p = np.asarray(ent)
                ax.plot(p[:, hax], p[:, vax], "k-", lw=0.9)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.axhline(0, color="r", lw=0.4); ax.axvline(0, color="r", lw=0.4)
        ax.set_xlabel(names[hax]); ax.set_ylabel(names[vax])
        ax.set_title(lab, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(title or "Cross-sections (reveal internal geometry)", fontsize=11)
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
    sections_png: str | None = None

    def prompt_text(self) -> str:
        """The textual half of the drawing task (analysis + instructions)."""
        return (DRAW_INSTRUCTIONS + "\n\n## Part analysis\n\n```json\n"
                + json.dumps(self.analysis, indent=2) + "\n```\n")


DRAW_INSTRUCTIONS = """\
You are reconstructing a mechanical part as clean, editable CAD. You are given a
dimensioned analysis of a triangle mesh, shaded rendered views, and CROSS-SECTION
drawings. Write a build123d Python script that draws the part as a feature model
(sketches + extrude/revolve/holes/fillets), matching the real geometry.

Read the cross-sections carefully: they reveal INTERNAL geometry the shaded views
hide — cavity floor heights, counterbore steps, wall thicknesses, chamfers. Read
exact depths and radii off them. Treat analytic fits in the analysis as hints, not
truth: a large-radius sphere/torus is often a degenerate fit to a nearly-flat or
stepped region — trust the cross-section over it.

Start from `features` (the grouped view), not the raw `surfaces`: it gives named
holes/counterbores/bosses with through-vs-blind, depths, and PATTERNS (a hole with
count>1 and its positions_mm — model it with one `with Locations(*positions):`).
`counterbore` means a two-diameter hole. `pockets` gives recessed cavities (opening
normal, floor level, depth, size) — sketch on the opening face and extrude(-depth,
SUBTRACT), reading the exact outline off the sections. `low_confidence_fits` lists
analytic fits to IGNORE (a mis-fit sphere/torus). `features` is best-effort — it
may miss a pocket or a round/oblique one; fall back to `surfaces` + the sections
for anything it doesn't list.

Rules:
- Output ONLY a Python code block: a build123d script. No prose outside it.
- Put the final solid in a BuildPart context named `part` (i.e. `with BuildPart()
  as part:`), exactly as build123d examples do. `from build123d import *` first.
- Work in millimetres and in the SAME world coordinates as the analysis (so the
  result overlays the mesh). Use the reported axes, radii, centres and extents.
- Prefer the simplest faithful feature tree. Model holes as cuts, bosses as adds,
  fillets/chamfers with `fillet`/`chamfer` on the relevant edges.
- EDGE TREATMENTS: fillets (rounded edges) read as ARCS in the cross-sections;
  chamfers read as short DIAGONAL cuts. The analysis' `edge_treatments` gives
  approximate fillet radii and counts (often ~10% high) — read the exact radius
  off the section and let the fitter refine it. Apply them AFTER the base feature,
  selecting edges e.g. `fillet(part.edges().filter_by(Axis.Z), radius=r)` for
  vertical edges or `chamfer(part.faces().sort_by(Axis.Z)[-1].edges(), length=w)`
  for a top perimeter. Order matters (fillet, then chamfer the remaining edges).
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
    axis = dominant_axis(mesh, analysis)
    sections = os.path.join(workdir, "part_sections.png")
    try:
        render_sections(mesh, sections, axis,
                        title=f"Cross-sections (axis = {'XYZ'[axis]})")
    except Exception:  # noqa: BLE001 — sections are a bonus, never fatal
        sections = None
    with open(os.path.join(workdir, "analysis.json"), "w") as fp:
        json.dump(analysis, fp, indent=2)
    brief = Brief(workdir, analysis, views, mesh, sections_png=sections)
    with open(os.path.join(workdir, "brief.md"), "w") as fp:
        fp.write(f"# Drawing brief\n\nViews: `part_views.png`\n"
                 f"Cross-sections: `part_sections.png`\n\n" + brief.prompt_text())
    return brief


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
# Visual comparator — the FINAL gate / cross-model diagnostic.
#
# The deviation colormap stays the per-iteration judge (it pinpoints geometric
# error precisely). This is the human-equivalent check the colormap can't give:
# matched-camera renders of the target mesh next to the candidate, plus a
# deterministic silhouette overlay that names *where the shapes differ* in image
# space. It works even when the mesh is non-watertight (where the distance/
# contains passes are unreliable), and it is the artifact to eyeball when a model
# gets the STRUCTURE wrong (box-vs-hex, flipped, mis-proportioned) rather than the
# dimensions. Because both meshes are rendered from identical cameras/limits, the
# silhouettes are pixel-aligned and the XOR is meaningful.
# ----------------------------------------------------------------------------

# elev, azim, label — the same cameras render_views uses, so the comparison
# matches the shaded brief renders.
_CMP_VIEWS = [(22, -60, "iso 1"), (22, 45, "iso 2"),
              (89, -90, "top -Z"), (5, 0, "front +X")]


def _render_array(mesh, elev, azim, bounds, px=560):
    """Shaded render of one mesh to an (H,W,3) float array on a WHITE background,
    with a fixed camera and shared axis limits. White background makes the
    silhouette trivial to extract (any non-white pixel is the part), and sharing
    `bounds`/camera across meshes makes two such arrays pixel-comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = mesh.triangles
    fn = np.asarray(mesh.face_normals)
    light = np.array([0.4, 0.3, 0.85]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(np.abs(fn @ light), 0, 1)
    base = np.array([0.45, 0.55, 0.68])
    facecolors = np.clip(base[None, :] * shade[:, None], 0, 1)

    lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    ext = np.where(hi - lo > 1e-9, hi - lo, 1e-9)
    dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, facecolors=facecolors,
                                         edgecolors="none"))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(ext)
    except Exception:  # noqa: BLE001
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255.0
    plt.close(fig)
    return buf[:, :, :3]


def _silhouette(img):
    """Boolean mask of the part: any pixel that isn't (near-)white background."""
    return img.min(axis=2) < 0.95


def _erode(mask, iters=1):
    """Binary erosion (4-neighbour) in pure numpy — shrinks a mask by `iters`
    pixels. Used to strip the 1-2px anti-aliasing fringe from the silhouette-diff
    so only genuine (thick) discrepancies are coloured, not rasterisation noise."""
    m = mask
    for _ in range(iters):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e
    return m


def render_comparison(target: trimesh.Trimesh, cand: trimesh.Trimesh,
                      path: str, *, title: str = "") -> str:
    """Matched-camera triptych per view: TARGET | CANDIDATE | silhouette overlay.
    Overlay is deterministic: red where the target has material the candidate
    lacks (a MISSING feature), blue where the candidate has material the target
    lacks (EXTRA / wrong), grey where they agree. One PNG for the model to Read."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # union bounds so both meshes are framed identically
    lo = np.minimum(target.bounds[0], cand.bounds[0])
    hi = np.maximum(target.bounds[1], cand.bounds[1])
    pad = 0.03 * (np.asarray(hi) - np.asarray(lo))
    bounds = (np.asarray(lo) - pad, np.asarray(hi) + pad)

    n = len(_CMP_VIEWS)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.6 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["TARGET (mesh)", "CANDIDATE (your model)",
                  "overlay  (red=missing, blue=extra)"]
    for r, (elev, azim, lab) in enumerate(_CMP_VIEWS):
        timg = _render_array(target, elev, azim, bounds)
        cimg = _render_array(cand, elev, azim, bounds)
        tm, cm = _silhouette(timg), _silhouette(cimg)
        ov = np.ones(timg.shape)
        ov[tm | cm] = (0.72, 0.72, 0.72)            # either present -> agreement grey
        # erode the difference regions to drop the AA fringe (keep thick patches)
        ov[_erode(tm & ~cm)] = (0.85, 0.12, 0.12)   # target-only -> MISSING
        ov[_erode(cm & ~tm)] = (0.12, 0.35, 0.88)   # candidate-only -> EXTRA
        for c, im in enumerate((timg, cimg, ov)):
            ax = axes[r, c]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(lab, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=95, bbox_inches="tight")
    plt.close(fig)
    return path


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
        content = [{"type": "text", "text": brief.prompt_text()},
                   {"type": "text", "text": "Shaded views:"},
                   _img_block(brief.views_png)]
        if brief.sections_png:
            content.append({"type": "text", "text": "Cross-sections (internal "
                            "geometry — cavity floors, counterbores, wall depths):"})
            content.append(_img_block(brief.sections_png))
        messages = [{"role": "user", "content": content}]
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
    if brief.sections_png:
        print(f"  sections : {brief.sections_png}   <- Read this (INTERNAL geometry)")
    print(f"  bbox {a['bounding_box']['size']} mm  min {a['bounding_box']['min']}  "
          f"volume {a['volume_mm3']} mm^3  watertight={a['watertight']}")
    feats = a.get("features") or {}
    if feats.get("holes") or feats.get("bosses") or feats.get("pockets"):
        print("  features (grouped):")
        for h in feats.get("holes", []):
            cb = (f" +c'bore Ø{h['counterbore']['diameter_mm']}x"
                  f"{h['counterbore']['depth_mm']}" if h.get("counterbore") else "")
            thru = {True: "through", False: "blind", None: "?"}[h.get("through")]
            print(f"    hole Ø{h['diameter_mm']} {thru} depth {h['depth_mm']} "
                  f"x{h['count']} [{h['arrangement']}]{cb}")
        for b in feats.get("bosses", []):
            print(f"    boss Ø{b['diameter_mm']} h{b['depth_mm']} "
                  f"x{b['count']} [{b['arrangement']}]")
        for pk in feats.get("pockets", []):
            print(f"    pocket {pk['size_mm']} depth {pk['depth_mm']} "
                  f"floor@{pk['floor_level_mm']} normal {pk['opening_normal']}")
    if feats.get("low_confidence_fits"):
        print(f"  low-confidence fits (distrust): {feats['low_confidence_fits']}")
    et = a.get("edge_treatments", {})
    if et.get("fillets") or et.get("chamfers"):
        fil = ", ".join(f"r~{f['radius_mm']}mm x{f['count']}"
                        for f in et.get("fillets", []))
        chm = ", ".join(f"x{c['count']}" for c in et.get("chamfers", []))
        print(f"  edge treatments (approx — confirm vs sections): "
              f"fillets [{fil or 'none'}]  chamfers [{chm or 'none'}]")
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


def _cli_fit(args) -> int:
    from .ingest import load_and_repair
    from . import fit as ft
    with open(args.script) as fp:
        script = fp.read()
    ing = load_and_repair(args.stl)
    os.makedirs(args.workdir, exist_ok=True)
    res = ft.fit_parameters(script, ing.mesh, max_evals=args.max_evals)
    print(f"fitted {res.n_params} numeric params in {res.n_evals} evals")
    print(f"deviation before {res.dev_before:.3f} mm  ->  after "
          f"{res.dev_after:.4f} mm")
    fitted_path = os.path.join(args.workdir, "attempt_fitted.py")
    with open(fitted_path, "w") as fp:
        fp.write(res.script)
    print(f"fitted script: {fitted_path}")
    # verify + render + export the fitted version through the normal path
    args.script = fitted_path
    args.tag = args.tag + "_fit"
    return _cli_run(args)


def _cli_compare(args) -> int:
    from .ingest import load_and_repair
    from . import occ_util as ou
    with open(args.script) as fp:
        script = fp.read()
    ing = load_and_repair(args.stl)
    os.makedirs(args.workdir, exist_ok=True)
    try:
        part = exec_script(script)
    except Exception as e:  # noqa: BLE001
        import traceback
        print("CANDIDATE FAILED TO BUILD — nothing to compare:")
        print("".join(traceback.format_exception_only(type(e), e)).strip())
        print(traceback.format_exc(limit=3))
        return 1
    rv, rf = ou.shape_to_mesh(part.wrapped, 0.05)
    cand = trimesh.Trimesh(vertices=rv, faces=rf, process=False)
    out = os.path.join(args.workdir, f"{args.tag}_compare.png")
    render_comparison(ing.mesh, cand, out,
                      title=f"target vs candidate ({args.tag})")
    print(f"comparison: {out}   <- Read this image")
    print("Then LIST what differs between TARGET and CANDIDATE, per view. In the "
          "overlay: RED = material in the part you did NOT model (a missing "
          "feature); BLUE = material you added that isn't in the part. Grey = "
          "agreement. Name the shape difference (e.g. 'foot is a hex prism, "
          "modelled as a box'), then fix the largest discrepancy and re-run.")
    return 0


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

    f = sub.add_parser("fit", help="snap a script's dimensions to the mesh, then "
                                   "verify (LLM gets structure, math gets mm)")
    f.add_argument("stl"); f.add_argument("workdir"); f.add_argument("script")
    f.add_argument("--tag", default="iter0")
    f.add_argument("--tol", type=float, default=0.1)
    f.add_argument("--max-evals", type=int, default=400)
    f.add_argument("--out", default=None, help="export STEP here if it builds")
    f.set_defaults(fn=_cli_fit)

    c = sub.add_parser("compare", help="matched-view render of target vs candidate "
                                       "+ silhouette diff (final gate / diagnostic)")
    c.add_argument("stl"); c.add_argument("workdir"); c.add_argument("script")
    c.add_argument("--tag", default="final")
    c.set_defaults(fn=_cli_compare)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())

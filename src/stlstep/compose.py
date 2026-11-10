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

from .proximity import closest_point

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


def _cylinder_support(mesh, face_ids, params):
    """Independent normal evidence for a positional cylinder fit.

    Polygon corners can lie on a circle even when the intervening faces are
    flat facets. A cylinder's face normals must also point radially.
    """
    axis = np.array(params["axis"], dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    center = np.asarray(params["center_mm"], dtype=float)
    radial = np.asarray(mesh.triangles_center[face_ids]) - center
    radial -= (radial @ axis)[:, None] * axis
    radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
    agreement = np.abs(np.sum(radial * mesh.face_normals[face_ids], axis=1))
    # Area weighting prevents tessellation density from creating confidence.
    weights = np.asarray(mesh.area_faces[face_ids])
    supported = float(weights[agreement >= np.cos(np.radians(10))].sum()
                      / max(float(weights.sum()), 1e-12))
    u = np.cross(axis, [1., 0, 0] if abs(axis[0]) < .9 else [0., 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.sort(np.arctan2(radial @ v, radial @ u))
    span = 360 - float(np.degrees(np.diff(np.r_[angles, angles[0] + 2*np.pi]).max()))
    return {"normal_support_fraction": round(supported, 4),
            "angular_span_deg": round(span, 2),
            "confident": supported >= .95}


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
        "fit_support": _cylinder_support(mesh, ids, p),
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
    except Exception as exc:
        raise RuntimeError("cannot analyze edge adjacency") from exc
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    analytic = {int(f) for r in seg.regions if r.kind != "plane"
                and r.kind != "freeform" for f in r.face_ids}

    rounded = (ang > 2) & (ang < 45) & conv
    fillet_faces = {int(f) for f in np.unique(pairs[rounded])}
    radii, uncertain = [], []
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
            support = _cylinder_support(mesh, band, params)
            if support["confident"] and support["angular_span_deg"] >= 30:
                radii.append(r)
            else:
                uncertain.append({"radius_mm": _r(r), **support,
                                  "note": "insufficient cylindrical normal/turn evidence; inspect sections"})

    oblique = np.abs(fn).max(axis=1) < 0.94
    chamfer_bands = [b for b in connected_components(
        [int(f) for f in np.where(oblique)[0]
         if int(f) not in analytic and int(f) not in fillet_faces], neighbors)
        if len(b) >= 6]
    # locate each chamfer band (not just count it) so the composer knows WHERE to
    # add a chamfer, not only how many exist. center = band vertex centroid; width
    # = the strip's cross dimension (middle PCA extent — length is largest, ~0 is
    # thickness); normal = mean oblique face normal (the cut plane's facing).
    chamfers = []
    for band in chamfer_bands:
        vids = np.unique(F[np.asarray(band, int)].ravel())
        pts = V[vids]
        c = pts.mean(axis=0)
        try:
            ext = np.sort(np.linalg.svd(pts - c, compute_uv=False))
            # singular values ~ sqrt(N)*extent; normalise to a length scale
            scale = np.sqrt(max(len(pts), 1))
            width = float(ext[1] / scale) if len(ext) >= 2 else 0.0
        except Exception:  # noqa: BLE001
            width = 0.0
        nrm = fn[np.asarray(band, int)].mean(axis=0)
        nrm = nrm / (np.linalg.norm(nrm) or 1.0)
        chamfers.append({"center_mm": [round(float(x), 2) for x in c],
                         "width_mm": round(width, 2),
                         "normal": [round(float(x), 3) for x in nrm],
                         "faces": int(len(band))})
    return {"fillets": _cluster_radii(radii),
            "low_confidence_fillets": uncertain,
            "chamfers": ([{"count": len(chamfer_bands),
                           "bands": chamfers}] if chamfer_bands else [])}


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

    edge_treatments = detect_edge_treatments(mesh, seg, neighbors)

    lo, hi = mesh.bounds
    from .features import group_features
    features = group_features(mesh, seg, surfaces, (hi - lo).tolist())

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
                  *, tag: str = "cand", dev_samples: int = 40000,
                  exclude_bands: list | None = None) -> Candidate:
    """Execute a proposed script, measure it against the mesh, and render both a
    shaded view and a deviation colormap. Never raises — a failure comes back as
    a Candidate with ok=False and the error text (that IS the refine signal).

    `exclude_bands`: cylindrical shells of IDENTIFIED threads — a nominal
    hole/stud legitimately deviates from the mesh helix there, so those sample
    points are excluded from the deviation quantiles (counts reported)."""
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

    excl_fwd = excl_rev = 0
    pts, _ = trimesh.sample.sample_surface(mesh, dev_samples)
    pts = np.asarray(pts)
    gpts, _ = trimesh.sample.sample_surface(gen, min(dev_samples, 20000))
    gpts = np.asarray(gpts)
    if exclude_bands:
        from . import featurecheck as fc
        keep = fc.mask_points_in_bands(pts, exclude_bands)
        excl_fwd = int((~keep).sum()); pts = pts[keep]
        keep = fc.mask_points_in_bands(gpts, exclude_bands)
        excl_rev = int((~keep).sum()); gpts = gpts[keep]
    _, fwd, _ = closest_point(gen, pts)
    _, rev, _ = closest_point(mesh, gpts)

    cand = Candidate(
        ok=True, script=script, part=part,
        deviation_mm=float(np.quantile(fwd, 0.99)),
        reverse_max_mm=float(np.quantile(rev, 0.99)),
        valid=bool(ou.is_valid(shape)), closed=_is_solid(shape),
        volume_mm3=float(part.volume),
        metrics={"fwd_max": float(fwd.max()), "rev_max": float(rev.max()),
                 "brep_faces": ou.count_faces(shape),
                 "thread_band_excluded": [excl_fwd, excl_rev]},
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


def _base_solid_along(base_mesh, p, ax) -> bool:
    """True if the base is (mostly) SOLID along the bore axis at p — the bore is
    MISSING from the base and worth carving. False if already void there (the
    silhouette has it -> re-carving would double-cut) or if we can't confirm (the
    bore centre falls outside the base footprint -> a straight cut isn't trusted).
    This is the guard that keeps recovered open-bores from over-carving."""
    if base_mesh is None:
        return True
    axv = np.asarray(ax, float); axv = axv / (np.linalg.norm(axv) or 1.0)
    lo_b, hi_b = base_mesh.bounds
    reach = float(np.linalg.norm(hi_b - lo_b))
    ts = np.linspace(-reach, reach, 60)
    pts = np.asarray(p, float)[None, :] + ts[:, None] * axv[None, :]
    inside = (pts >= lo_b - 1e-6).all(1) & (pts <= hi_b + 1e-6).all(1)
    if inside.sum() < 4:
        return False
    try:
        return float(base_mesh.contains(pts[inside]).mean()) > 0.4
    except Exception:  # noqa: BLE001 — can't confirm -> don't carve
        return False


def _scaffold_ok(script: str, ref_faces: int) -> bool:
    """Exec a scaffold and check it's a CLEAN candidate: watertight (tested the
    same way verify/diff does — STEP round-trip + fixed-tolerance tessellation, so
    band-interface slivers are caught) AND parsimonious (B-Rep face count not a
    'triangle soup', i.e. brep_faces <= 0.25*ref_faces, the diff's soup threshold).
    Used to accept the stepped base only when it beats those bars; else the caller
    falls back to the single-station base. False on any failure."""
    import tempfile
    from build123d import export_step, import_step, export_stl
    try:
        part = exec_script(script)
        st = tempfile.NamedTemporaryFile(suffix=".step", delete=False); st.close()
        export_step(part, st.name)
        reimported = import_step(st.name)
        sl = tempfile.NamedTemporaryFile(suffix=".stl", delete=False); sl.close()
        export_stl(reimported, sl.name, tolerance=0.05, angular_tolerance=0.2)
        if not trimesh.load(sl.name, force="mesh").is_watertight:
            return False
        n_faces = len(reimported.faces())
        return n_faces <= 0.25 * max(1, ref_faces)
    except Exception:  # noqa: BLE001 — any failure -> don't trust the stepped base
        return False


def build_scaffold(mesh: trimesh.Trimesh, analysis: dict) -> tuple[str, dict]:
    """Deterministic feature -> build123d scaffold: the reliable, fully-specified
    part of the model, emitted as a clean editable feature tree so the LLM starts
    from geometry-true structure instead of hand-transcribing (and mis-placing) it.

    * BASE  = the profile2d station with the LARGEST cross-section (outer envelope
      before top-face pockets/chamfers shrink it), extruded along the prismatic axis.
    * HOLES = every detected hole as an oriented Cylinder SUBTRACT (all axes, exact
      position + axis + Ø + through/depth) — the error-prone part hand-modelling
      gets wrong. Recovered open-bores (rod channels) are auto-cut ONLY when their
      axis ∥ the extrude axis AND the base is solid there (else -> LLM note).
    * BOSSES as ADDs; POCKETS as oriented Box SUBTRACTs where a centre is known.
    Fillets/chamfers and under-placed features are left as explicit TODO comments —
    the LLM's remaining job, spelled out. Returns ``(script, stats)``."""
    from build123d import export_stl
    from . import draw
    bb = analysis["bounding_box"]
    lo = np.asarray(bb["min"], float); hi = np.asarray(bb["max"], float)
    diag = float(np.linalg.norm(hi - lo))
    axis = (analysis["deterministic_recognizers"].get("prismatic") or {}).get(
        "axis", "z")
    ai = {"x": 0, "y": 1, "z": 2}[axis]
    span = hi[ai] - lo[ai]
    feats = analysis.get("features", {})

    # 1. base outer form = max cross-section station
    def _snip_volume(snip):
        if not snip:
            return 0.0
        try:
            return abs(float(exec_script(snip).volume))
        except Exception:  # noqa: BLE001 — an uncompilable base loses the scan
            return 0.0

    # single-station base: the station with the LARGEST cross-section (outer
    # envelope before top-face pockets/chamfers shrink it)
    best_v, base = -1.0, ""
    for frac in (0.12, 0.25, 0.4, 0.55, 0.7, 0.85):
        snip, _ = draw.profile2d_snippet(mesh, ai, float(lo[ai] + frac * span))
        v = _snip_volume(snip)
        if v > best_v:
            best_v, base = v, snip

    if not base:
        raise ValueError("profile2d produced no base snippet for any station")
    single_base = base

    def _emit_features(base_snip, stepped):
        """Emit the full scaffold (base + gated features) for a given base."""
        base_mesh = None
        try:
            import tempfile
            tf = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
            tf.close()
            export_stl(exec_script(base_snip), tf.name)
            base_mesh = trimesh.load(tf.name)
        except Exception:  # noqa: BLE001 — solidity guard is best-effort
            base_mesh = None
        extrude_axis = np.zeros(3); extrude_axis[ai] = 1.0
        base_kind = "stepped" if stepped else "single-station"
        lines = [base_snip, "",
                 f"    # === auto-emitted features ({base_kind} base) ===",
                 "    # holes: exact position + axis + diameter (all axes)"]
        nh = nct = nb = npk = 0; skipped = []
        for h in feats.get("holes", []):
            dia = float(h["diameter_mm"]); ax = h["axis"]
            L = diag * 1.2 if h.get("through") else float(h.get("depth_mm") or diag)
            cut_positions = []
            for p in h["positions_mm"]:
                if h.get("source") == "open-bore":
                    # open-bore: straight-cut only PARALLEL bores the base lacks
                    # (avoids straight-cutting a curved duct)
                    parallel = abs(float(np.dot(np.asarray(ax, float),
                                                extrude_axis))) > 0.9
                    if not (parallel and _base_solid_along(base_mesh, p, ax)):
                        why = ("axis ⟂ extrude — base can't hold it"
                               if not parallel else "base already has it")
                        skipped.append(
                            f"    #   OPEN-BORE Ø{dia:.2f} at "
                            f"{tuple(round(x, 2) for x in p)} axis {ax} — {why}; "
                            f"LLM to verify/place")
                        continue
                elif stepped and not _base_solid_along(base_mesh, p, ax):
                    # a stepped base already carries every section-visible feature;
                    # carve only holes the bands genuinely miss (e.g. ⟂ holes).
                    # Re-cutting a section-captured hole double-removes / conflicts.
                    continue
                lines.append(
                    f"    with Locations(Plane(origin=({p[0]:.4f}, {p[1]:.4f}, "
                    f"{p[2]:.4f}), z_dir=({ax[0]:.4f}, {ax[1]:.4f}, "
                    f"{ax[2]:.4f}))):")
                lines.append(f"        Cylinder(radius={dia / 2:.4f}, "
                             f"height={L:.4f}, mode=Mode.SUBTRACT)")
                nh += 1
                cut_positions.append(p)
            # counterbore / countersink at the mouths — a sub-band face feature the
            # stepped base never captures, so emit at EVERY instance even when the
            # through-cut was gated out. Skip open-bores (channels, no screw mouth).
            treat_positions = (h["positions_mm"]
                               if h.get("source") != "open-bore" else cut_positions)
            axv = np.asarray(ax, float); ai_h = int(np.argmax(np.abs(axv)))
            into = np.sign(axv[ai_h]) if axv[ai_h] else 1.0
            for t in h.get("end_treatments", []):
                major = t["major_dia_mm"]; dep = t["depth_mm"]
                zd = [0.0, 0.0, 0.0]
                zd[ai_h] = float(into) if t["end"] == "lo" else -float(into)
                for p in treat_positions:
                    o = list(p); o[ai_h] = t["face_mm"]
                    lines.append(
                        f"    with Locations(Plane(origin=({o[0]:.4f}, {o[1]:.4f}, "
                        f"{o[2]:.4f}), z_dir=({zd[0]:.1f}, {zd[1]:.1f}, "
                        f"{zd[2]:.1f}))):")
                    if t["kind"] == "counterbore":
                        lines.append(
                            f"        Cylinder(radius={major / 2:.4f}, "
                            f"height={dep:.4f}, align=(Align.CENTER, Align.CENTER, "
                            f"Align.MIN), mode=Mode.SUBTRACT)  # counterbore")
                    else:
                        # countersink: cone WIDE at the face tapering to a near-
                        # point INTO the hole void. Stopping at the hole radius
                        # leaves a zero-thickness tangent sliver (non-watertight);
                        # running the cone transversally through the wall to a point
                        # keeps the solid watertight.
                        slope = max((major - dia) / 2 / max(dep, 0.05), 0.1)
                        csk_h = (major / 2) / slope
                        lines.append(
                            f"        Cone(bottom_radius={major / 2:.4f}, "
                            f"top_radius=0.01, height={csk_h:.4f}, "
                            f"align=(Align.CENTER, Align.CENTER, Align.MIN), "
                            f"mode=Mode.SUBTRACT)  # countersink (~45deg)")
                    nct += 1
        if skipped:
            lines.append("    # recovered open-bores NOT auto-cut (LLM to place):")
            lines += skipped

        for b in feats.get("bosses", []):
            dia = float(b.get("diameter_mm", 0)); ax = b.get("axis", [0, 0, 1])
            L = float(b.get("height_mm") or b.get("depth_mm") or 1)
            for p in b.get("positions_mm", []):
                # stepped base already carries section-visible bosses
                if stepped and _base_solid_along(base_mesh, p, ax):
                    continue
                lines.append(
                    f"    with Locations(Plane(origin=({p[0]:.4f}, {p[1]:.4f}, "
                    f"{p[2]:.4f}), z_dir=({ax[0]:.4f}, {ax[1]:.4f}, "
                    f"{ax[2]:.4f}))):")
                lines.append(f"        Cylinder(radius={dia / 2:.4f}, "
                             f"height={L:.4f}, mode=Mode.ADD)")
                nb += 1

        if feats.get("pockets"):
            lines.append("    # pockets: centre + size + depth on the opening face")
        for p in feats.get("pockets", []):
            c = p.get("center_mm")
            if not c:
                continue
            n = p["opening_normal"]; w, ht = p["size_mm"]
            depth = float(p["depth_mm"])
            # stepped base already carries a section-visible pocket
            if stepped and not _base_solid_along(base_mesh, c, n):
                continue
            ax = p.get("plane_axes")
            xdir = (f", x_dir=({ax[0][0]:.4f}, {ax[0][1]:.4f}, {ax[0][2]:.4f})"
                    if ax else "")
            lines.append(
                f"    with Locations(Plane(origin=({c[0]:.4f}, {c[1]:.4f}, "
                f"{c[2]:.4f}){xdir}, z_dir=({n[0]:.4f}, {n[1]:.4f}, "
                f"{n[2]:.4f}))):")
            lines.append(f"        Box({w:.4f}, {ht:.4f}, {depth + 2:.4f}, "
                         f"mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, "
                         f"Align.MIN))")
            npk += 1

        # what the analysis still can't place precisely — the LLM's remaining job
        todo = []
        for p in feats.get("pockets", []):
            if not p.get("center_mm"):
                todo.append(f"#   POCKET {p['size_mm']} depth {p['depth_mm']} "
                            f"normal {p['opening_normal']} — no centre; from a "
                            f"section")
        for note in feats.get("notes", []):
            if "SADDLE" in note:
                todo.append("#   " + note)
        et = analysis.get("edge_treatments", {})
        for f in et.get("fillets", []):
            todo.append(f"#   FILLET r={f.get('radius_mm')} x{f.get('count')} — "
                        f"locations not detected; add on the relevant edges")
        for c in et.get("chamfers", []):
            bands = c.get("bands") or []
            if bands:
                todo.append(f"#   CHAMFER x{c.get('count')} — locations below; add "
                            f"chamfer() on the edge nearest each centre:")
                for bd in bands:
                    todo.append(f"#     at {tuple(bd['center_mm'])} ~{bd['width_mm']}"
                                f"mm wide, cut-face normal {tuple(bd['normal'])}")
            else:
                todo.append(f"#   CHAMFER x{c.get('count')} — locations not detected; "
                            f"add on the relevant edges")
        if todo:
            lines.append("")
            lines.append("    # === TODO (underspecified — LLM to place, measure "
                         "with probe) ===")
            lines += ["    " + t for t in todo]

        stats = {"holes": nh, "counterbores": nct, "bosses": nb, "pockets": npk,
                 "skipped_bores": len(skipped), "todos": len(todo),
                 "base": base_kind}
        return "\n".join(lines) + "\n", stats

    # Prefer a STEPPED base (fits each level band's own section — far better on
    # parts whose section varies along the axis), but keep it ONLY if the full
    # scaffold stays watertight: feature cuts across band interfaces can leave
    # slivers, and a non-watertight scaffold is invalid. Otherwise fall back to
    # the single-station base (never a regression vs. the old behaviour).
    #
    # Try coalesce tolerances TIGHT-FIRST: a low tolerance keeps each band's own
    # section (most accurate), so accept the first that clears the gate. Escalating
    # to a looser tolerance MERGES near-identical bands, which removes the band
    # interfaces where feature cuts sliver (restoring watertightness) and trims the
    # face count that trips the parsimony bar — unlocking a stepped base on parts a
    # single tolerance couldn't (e.g. x-carriage: 7 bands non-watertight -> 4 bands
    # watertight). The gate is the accuracy safety net: an over-merged base that no
    # longer matches the part fails watertight/parsimony and is rejected.
    level_bounds = (analysis["deterministic_recognizers"].get("prismatic")
                    or {}).get("level_bounds")
    if level_bounds and len(level_bounds) >= 3:
        for ctol in (0.02, 0.15, 0.35):
            step_snip, _ = draw.profile2d_stepped(
                mesh, ai, level_bounds, coalesce_tol=ctol)
            step_v = _snip_volume(step_snip)
            if not (step_snip and step_v > 0
                    and (best_v <= 0 or step_v < 3.0 * best_v)):
                continue
            script, stats = _emit_features(step_snip, True)
            if _scaffold_ok(script, len(mesh.faces)):
                return script, stats
    return _emit_features(single_base, False)


def _cli_emit(args) -> int:
    from .ingest import load_and_repair
    ing = load_and_repair(args.stl)
    apath = os.path.join(args.workdir, "analysis.json")
    if os.path.exists(apath):
        with open(apath) as fp:
            analysis = json.load(fp)
    else:
        os.makedirs(args.workdir, exist_ok=True)
        analysis = analyze(ing.mesh)
        with open(apath, "w") as fp:
            json.dump(analysis, fp, indent=2)
    script, stats = build_scaffold(ing.mesh, analysis)
    out = args.out or os.path.join(args.workdir, "scaffold.py")
    with open(out, "w") as fp:
        fp.write(script)
    print(f"scaffold written to {out}")
    print(f"  auto-emitted: {stats['holes']} hole-cut(s), "
          f"{stats['counterbores']} counterbore/countersink(s), {stats['bosses']} "
          f"boss(es), {stats['pockets']} pocket(s)")
    if stats["skipped_bores"]:
        print(f"  {stats['skipped_bores']} recovered open-bore(s) left as notes "
              f"(axis ⟂ extrude / base already has it)")
    print(f"  {stats['todos']} feature(s) left as TODO for the LLM "
          f"(fillets/chamfers/under-placed)")
    print("  -> verify with:  compose run "
          f"{args.stl} {args.workdir} {out}")
    return 0


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
    if not a["watertight"]:
        print("  WARNING: mesh is NOT watertight — volume/contains checks are "
              "unreliable; trust the cross-sections for internal geometry.")
    from . import featurecheck as fc
    voidsum = fc.internal_void_summary(ing.mesh)
    if voidsum and voidsum["void_frac"] > 0.02:
        print(f"  INTERNAL CAVITY: {voidsum['void_mm3']} mm^3 of internal void "
              f"({100*voidsum['void_frac']:.1f}% of solid), concentrated along "
              f"{voidsum['axis']} in {voidsum['cavity_band_mm']} mm. The part is "
              f"HOLLOW inside here — model each cavity below as a CUT; a solid "
              f"body will fail the internal-void check.")
        # dimension the cavities on all three axes — the internal structure the
        # composer must rebuild, as named/numbered features, not just pictures
        try:
            from . import voids as vd
            best_axis, best_vol, tables = 0, -1.0, []
            for ax in (0, 1, 2):
                res = vd.analyze_voids(ing.mesh, ax, n=48)
                vol = sum(f["volume_mm3"] for f in res["features"])
                tables.append(res)
                if vol > best_vol:
                    best_axis, best_vol = ax, vol
            print("  internal cavity features (probe voids for snippets/detail):")
            shown = 0
            for res in tables:
                for f in res["features"]:
                    if shown >= 10:
                        break
                    d = f.get("diameter_mm") or f.get("outer_diameter_mm")
                    dd = f"Ø{d}" if d else f"bbox {f.get('bbox_uv_mm')}"
                    print(f"    {f['shape']:9} {dd:>13}  {f['axis']} "
                          f"{f['span_mm'][0]}..{f['span_mm'][1]}  vol "
                          f"{f['volume_mm3']}mm^3  centre {f['center_uv_mm']} "
                          f"{f['uv_axes']}  {f['ends']}")
                    shown += 1
            atlas = vd.render_void_atlas(
                ing.mesh, best_axis, tables[best_axis],
                os.path.join(args.workdir, "void_atlas.png"))
            if atlas:
                print(f"  void atlas : {atlas}   <- Read this image (internal "
                      f"silhouette + change-point slices)")
        except Exception:  # noqa: BLE001 — cavity table is best-effort
            pass
    feats = a.get("features") or {}
    if feats.get("holes") or feats.get("bosses") or feats.get("pockets"):
        print("  features (grouped):")
        for h in feats.get("holes", []):
            cb = (f" +c'bore Ø{h['counterbore']['diameter_mm']}x"
                  f"{h['counterbore']['depth_mm']}" if h.get("counterbore") else "")
            thru = {True: "through", False: "blind", None: "?"}[h.get("through")]
            td = h.get("threaded")
            ts = (f"  THREADED {td.get('spec') or ('pitch %.2f' % td['pitch_mm'])}"
                  f" (model at minor Ø{td['minor_diameter_mm']}, record spec)"
                  if td else "")
            print(f"    hole Ø{h['diameter_mm']} {thru} depth {h['depth_mm']} "
                  f"x{h['count']} [{h['arrangement']}]{cb}{ts}")
        for b in feats.get("bosses", []):
            td = b.get("threaded")
            ts = (f"  THREADED {td.get('spec') or ('pitch %.2f' % td['pitch_mm'])}"
                  f" (model at major Ø{td['major_diameter_mm']}, record spec)"
                  if td else "")
            print(f"    boss Ø{b['diameter_mm']} h{b['depth_mm']} "
                  f"x{b['count']} [{b['arrangement']}]{ts}")
        for tc in feats.get("thread_candidates", []):
            print(f"    possible THREAD near {tc['center_mm']} "
                  f"(extent {tc['extent_mm']}, {tc['fragments']} helical "
                  f"fragments) — identify with `probe thread` before modelling")
        for pk in feats.get("pockets", []):
            print(f"    pocket {pk['size_mm']} depth {pk['depth_mm']} "
                  f"floor@{pk['floor_level_mm']} normal {pk['opening_normal']}")
    if feats.get("low_confidence_fits"):
        print(f"  low-confidence fits (distrust): {feats['low_confidence_fits']}")
    et = a.get("edge_treatments", {})
    if et.get("low_confidence_fillets"):
        print(f"  low-confidence fillet hints (confirm vs sections): "
              f"{et['low_confidence_fillets']}")
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


def _parse_thread_band(spec: str) -> dict:
    """--thread-band 'axis,cu,cv,t0,t1,r0,r1' -> band dict. cu,cv are the axis
    position in the perpendicular plane (in x,y,z order minus the axis)."""
    parts = spec.split(",")
    ai = {"x": 0, "y": 1, "z": 2}[parts[0].strip()]
    cu, cv, t0, t1, r0, r1 = (float(x) for x in parts[1:7])
    u, v = [i for i in (0, 1, 2) if i != ai]
    pt = [0.0, 0.0, 0.0]
    pt[u], pt[v] = cu, cv
    d = [0.0, 0.0, 0.0]; d[ai] = 1.0
    return {"point": pt, "axis": d, "t": [t0, t1], "r": [r0, r1],
            "label": "declared"}


def _cli_run(args) -> int:
    from .ingest import load_and_repair
    from . import featurecheck as fc
    with open(args.script) as fp:
        script = fp.read()
    ing = load_and_repair(args.stl)

    # thread bands: auto (identified in the analysis) + declared (--thread-band
    # after a `probe thread` identification). A nominal hole/stud legitimately
    # deviates from the mesh helix inside these; exclude, don't punish.
    analysis = None
    validation_errors = []
    apath = os.path.join(args.workdir, "analysis.json")
    if os.path.exists(apath):
        try:
            with open(apath) as fp:
                analysis = json.load(fp)
        except Exception as exc:  # report invalid analysis; never skip its gate
            validation_errors.append(f"analysis: {type(exc).__name__}: {exc}")
    if not isinstance(analysis, dict) or not isinstance(analysis.get("features"), dict):
        validation_errors.append("analysis: missing feature analysis; run `brief` first")
        analysis = None
    bands = fc.thread_bands(analysis) if analysis else []
    for spec in (args.thread_band or []):
        try:
            bands.append(_parse_thread_band(spec))
        except Exception:  # noqa: BLE001
            raise SystemExit(f"bad --thread-band {spec!r} "
                             f"(want axis,cu,cv,t0,t1,r0,r1)")

    cand = run_candidate(script, ing.mesh, args.workdir, tag=args.tag,
                         exclude_bands=bands or None)
    mv = float(ing.mesh.volume) if ing.mesh.is_watertight else None
    print(cand.feedback(mv))
    if bands and cand.ok:
        ex = cand.metrics.get("thread_band_excluded", [0, 0])
        print(f"thread bands: {len(bands)} recognized "
              f"({', '.join(b['label'] for b in bands)}) — excluded "
              f"{ex[0]}+{ex[1]} deviation samples (nominal thread "
              f"representation is correct there)")

    # size-relative tolerance: a fixed mm bar is wrong across scales.
    tol = args.tol if args.tol is not None else fc.size_tol(ing.mesh)
    if args.tol is None:
        print(f"tolerance: {tol:.3f} mm  (max of a 0.10 mm floor and 0.15% of the "
              f"{fc.bbox_diagonal(ing.mesh):.1f} mm bbox diagonal)")

    # feature-completeness: deviation can't see small features, so check that the
    # part's holes/fillets/chamfers are actually modelled as editable ops.
    gaps, hard_gap = [], False
    if cand.ok:
        if analysis is not None:
            try:
                gaps = fc.feature_gap(analysis, cand.part.wrapped)
                if not isinstance(gaps, list):
                    raise ValueError("feature check returned no result list")
                if bands:
                    # helix flanks masquerade as fillet bands in the detector;
                    # with threads recognized, fillet demands become advisory
                    for g in gaps:
                        if g["kind"] == "fillet":
                            g["found"] = None
                            g["note"] += (" (thread present — helix flanks can "
                                          "read as fillets; confirm visually)")
                print(fc.format_gap(gaps))
                hard_gap = any(g["found"] is not None and g["found"] < g["expected"]
                               for g in gaps)
            except Exception as exc:  # validation failures must be visible
                validation_errors.append(f"feature check: {type(exc).__name__}: {exc}")
        else:
            print("feature check: unavailable (run `brief` first)")

    # slice-occupancy: the volumetric-truth check. Surface distance measures how
    # close the skins are; this slices BOTH solids along all three axes and
    # intersects the filled regions, so it directly measures whether the same 3D
    # space is solid — catching filled-in internal cavities, dropped pockets, and
    # oversize/undersize material that surface metrics under-report. Robust to a
    # non-watertight mesh, and it localises every divergence to a slice + region.
    occ = None
    occ_gap = False
    if cand.ok:
        try:
            cg = fc.CADSections(cand.part)
            occ = fc.slice_occupancy(ing.mesh, cg, exclude_bands=bands or None)
            axes = occ.get("per_axis", []) if isinstance(occ, dict) else []
            if ({a["axis"] for a in axes} != set("xyz") or
                    any(a.get("mean_iou") is None or not a.get("coords") for a in axes)):
                raise ValueError("occupancy did not measure all three axes")
            if not all(np.isfinite(occ[k]) for k in
                       ("mean_iou", "missing_frac", "extra_frac", "void_missing_frac")):
                raise ValueError("occupancy returned non-finite metrics")
            if not isinstance(occ.get("loop_mismatches"), list):
                raise ValueError("opening check returned no mismatch list")
            if occ and occ.get("mean_iou") is not None:
                print(f"slice occupancy (solid-vs-void, 3 axes): mean IoU "
                      f"{occ['mean_iou']}  |  MISSING material {occ['missing_mm3']}"
                      f"mm^3 ({100*occ['missing_frac']:.1f}%)  EXTRA material "
                      f"{occ['extra_mm3']}mm^3 ({100*occ['extra_frac']:.1f}%)")
                if occ["void_missing_frac"] > 0.02:
                    print(f"  internal cavity filled solid: "
                          f"{occ['void_missing_mm3']}mm^3 "
                          f"({100*occ['void_missing_frac']:.1f}%) — the part is "
                          f"HOLLOW where your model is SOLID. Section the cavity "
                          f"band and cut it (Mode.SUBTRACT).")
                if occ.get("loop_mismatches"):
                    print("  OPENING MISMATCHES (scale-free per-loop check — "
                          "every hole/opening must match SHAPE, however small):")
                    for g in occ["loop_mismatches"]:
                        got = (f" modelled as {g['cand_shape']}"
                               if g.get("cand_shape") else "")
                        print(f"    {g['kind']}: {g.get('mesh_shape', '?')} at "
                              f"{g['axis']}~{g['stations'][0]} centre "
                              f"{g['center_uv']}{got}  (IoU {g['iou']}, "
                              f"{len(g['stations'])} station(s))")
                if occ["worst_slices"]:
                    print("  worst slices (fix these first — probe the section "
                          "at each coordinate):")
                    for w in occ["worst_slices"][:5]:
                        loc = (f"  region {w['region_uv_bounds']}"
                               if w.get("region_uv_bounds") else "")
                        print(f"    {w['axis']}={w['at_mm']}  IoU {w['iou']}  "
                              f"missing {w['missing_mm2']}mm^2  extra "
                              f"{w['extra_mm2']}mm^2{loc}")
                spng = fc.render_slice_report(
                    ing.mesh, cg, occ,
                    os.path.join(args.workdir, f"{args.tag}_slices.png"))
                if spng:
                    print(f"slice report: {spng}   <- Read this image (grey=match, "
                          f"red=missing, blue=extra, per worst slice)")
                occ_gap = (occ["missing_frac"] > 0.03 or occ["extra_frac"] > 0.03
                           or bool(occ.get("loop_mismatches")))
        except Exception as exc:  # validation failures must be visible
            validation_errors.append(f"occupancy check: {type(exc).__name__}: {exc}")

    for error in validation_errors:
        print(f"VALIDATION ERROR: {error}")
    passed = bool(cand.ok and cand.valid and cand.closed and
                  cand.deviation_mm <= tol and cand.reverse_max_mm <= tol and
                  not hard_gap and not occ_gap and not validation_errors)
    if cand.ok:
        print(f"colormaps: MISSING={cand.colormap_fwd_png}  "
              f"EXTRA={cand.colormap_png}   <- Read these images")
        print(f"model view: {cand.render_png}")
        with open(os.path.join(args.workdir, "metrics.json"), "w") as fp:
            json.dump({"deviation_mm": cand.deviation_mm,
                       "reverse_max_mm": cand.reverse_max_mm,
                       "volume_mm3": cand.volume_mm3, "valid": cand.valid,
                       "closed": cand.closed, "passed": passed,
                       "tolerance_mm": round(tol, 4),
                       "feature_gaps": gaps,
                       "validation_errors": validation_errors,
                       "slice_occupancy": ({k: v for k, v in occ.items()
                                            if k != "per_axis"} if occ else None),
                       "metrics": cand.metrics}, fp, indent=2)
        if args.out:
            from build123d import export_step
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                        exist_ok=True)
            export_step(cand.part, args.out)
            with open(os.path.join(args.workdir, "feature_tree.py"), "w") as fp:
                fp.write(script)
            print(f"exported {args.out}  +  {args.workdir}/feature_tree.py")
    if not passed and cand.ok:
        why = list(validation_errors)
        if cand.deviation_mm > tol or cand.reverse_max_mm > tol:
            why.append(f"deviation {max(cand.deviation_mm, cand.reverse_max_mm):.3f}"
                       f" > {tol:.3f} mm")
        if hard_gap:
            why.append("missing editable features (above)")
        if occ_gap:
            parts_ = []
            if occ["missing_frac"] > 0.03 or occ["extra_frac"] > 0.03:
                parts_.append(f"missing {100*occ['missing_frac']:.1f}% / extra "
                              f"{100*occ['extra_frac']:.1f}% of volume")
            if occ.get("loop_mismatches"):
                parts_.append(f"{len(occ['loop_mismatches'])} opening shape "
                              f"mismatch(es) — see OPENING MISMATCHES above")
            why.append("occupancy: " + "; ".join(parts_))
        print(f"STATUS: FAIL  ({'; '.join(why)})" if why else "STATUS: FAIL")
    else:
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


def _cli_slices(args) -> int:
    """Targeted occupancy re-analysis: densely slice a REQUESTED band and diff
    target vs candidate slice-by-slice, rendering every slice. The zoom lens for
    a region the uniform verify pass flagged (or stepped over)."""
    from .ingest import load_and_repair
    from . import featurecheck as fc
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
        return 1
    cg = fc.CADSections(part)

    ai = {"x": 0, "y": 1, "z": 2}.get(args.axis)
    if ai is None:
        raise SystemExit(f"--axis must be x/y/z, got {args.axis!r}")
    r = [float(x) for x in args.range.split(",")]
    if len(r) != 2:
        raise SystemExit("--range needs lo,hi (use --range=-76,-55 for negatives)")
    recs = fc.zoom_occupancy(ing.mesh, cg, ai, r[0], r[1], n=args.n)
    for rec in recs:
        loc = (f"  region {rec['region_uv_bounds']}"
               if rec.get("region_uv_bounds") else "")
        print(f"  {rec['axis']}={rec['at_mm']}  IoU {rec['iou']}  "
              f"target {rec['target_mm2']}mm^2 cand {rec['cand_mm2']}mm^2  "
              f"missing {rec['missing_mm2']}  extra {rec['extra_mm2']}{loc}")
    out_png = os.path.join(args.workdir, f"{args.tag}_slices_zoom.png")
    coords = [rec["at_mm"] for rec in recs]
    fc.render_slice_strip(ing.mesh, ai, coords, out_png, cand_mesh=cg)
    print(f"zoom report: {out_png}   <- Read this image (grey=match, "
          f"red=missing, blue=extra)")
    return 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="LLM draw-it harness: build a brief, run+verify a script.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("brief", help="analyze an STL and render views for drawing")
    b.add_argument("stl"); b.add_argument("workdir")
    b.set_defaults(fn=_cli_brief)

    e = sub.add_parser("emit", help="seed a build123d scaffold from detected "
                                    "features (profile2d base + holes/pockets + "
                                    "TODO notes) for the LLM to finish")
    e.add_argument("stl"); e.add_argument("workdir")
    e.add_argument("--out", default=None,
                   help="write the scaffold here (default: WORKDIR/scaffold.py)")
    e.set_defaults(fn=_cli_emit)

    r = sub.add_parser("run", help="execute + verify a proposed build123d script")
    r.add_argument("stl"); r.add_argument("workdir"); r.add_argument("script")
    r.add_argument("--tag", default="iter0")
    r.add_argument("--thread-band", action="append", default=None,
                   help="declare an identified thread band to exclude from "
                        "deviation: axis,cu,cv,t0,t1,r0,r1 (repeatable; use "
                        "after `probe thread` confirms a spec)")
    r.add_argument("--tol", type=float, default=None,
                   help="deviation tolerance mm (default: max of 0.10mm and "
                        "0.15%% of bbox diagonal)")
    r.add_argument("--out", default=None, help="export STEP here if it builds")
    r.set_defaults(fn=_cli_run)

    f = sub.add_parser("fit", help="snap a script's dimensions to the mesh, then "
                                   "verify (LLM gets structure, math gets mm)")
    f.add_argument("stl"); f.add_argument("workdir"); f.add_argument("script")
    f.add_argument("--tag", default="iter0")
    f.add_argument("--tol", type=float, default=None)
    f.add_argument("--max-evals", type=int, default=400)
    f.add_argument("--out", default=None, help="export STEP here if it builds")
    f.add_argument("--thread-band", action="append", default=None,
                   help="same as run --thread-band (fit re-verifies via run)")
    f.set_defaults(fn=_cli_fit)

    c = sub.add_parser("compare", help="matched-view render of target vs candidate "
                                       "+ silhouette diff (final gate / diagnostic)")
    c.add_argument("stl"); c.add_argument("workdir"); c.add_argument("script")
    c.add_argument("--tag", default="final")
    c.set_defaults(fn=_cli_compare)

    z = sub.add_parser("slices", help="ZOOM: densely slice a chosen band and "
                                      "diff target vs candidate per slice")
    z.add_argument("stl"); z.add_argument("workdir"); z.add_argument("script")
    z.add_argument("--axis", required=True, help="x, y, or z")
    z.add_argument("--range", required=True,
                   help="lo,hi band (use --range=-76,-55 for negatives)")
    z.add_argument("--n", type=int, default=12, help="slices across the band")
    z.add_argument("--tag", default="zoom")
    z.set_defaults(fn=_cli_slices)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())

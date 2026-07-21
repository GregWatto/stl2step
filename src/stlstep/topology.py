"""Stage 3 — topology: build a watertight BREP solid from the segmentation.

Two methods, selected by `method=`:

  * "facets"  — baseline. One planar BRep face per mesh triangle, sewn into a
    solid. Dimensionally exact (it *is* the mesh) and always watertight when the
    mesh is, but with zero face reduction — the honest starting point that the
    gate reports as "not editable yet". This exercises the whole harness.

  * "merged"  — the real target (built out in later stages): one analytic/NURBS
    face per region, meeting neighbours along real surface-surface intersection
    curves. Falls back to per-region faceting for any region whose clean face
    cannot yet be constructed, so the pipeline never regresses to a crash.

Returning per-region provenance lets the gate attribute deviation to regions.
"""

from __future__ import annotations

import numpy as np

from . import occ_util as ou
from . import surfaces as su


def build_solid(mesh, segmentation, *, method: str = "facets"):
    """Return (shape, info). `info` records how each region was realized.

    Methods:
      facets  faceted watertight solid (exact, 0% reduction) — the baseline.
      merged  watertight planar-merge: coplanar faces collapsed via OCCT
              UnifySameDomain; curved regions stay faceted. Valid + watertight.
      shell   per-region analytic faces (real cylinders/planes/tori) as a
              non-watertight shell — the most editable surfaces, but not a solid.
              Watertight analytic stitching needs surface-surface intersection
              curves (see docs/topology-experiments.md); this is the reachable
              editable output short of that.
    """
    if method == "facets":
        return _build_facets(mesh)
    if method == "merged":
        return _build_merged(mesh, segmentation)
    if method == "shell":
        return _build_shell(mesh, segmentation)
    if method == "solid":
        return _build_solid_analytic(mesh, segmentation)
    raise ValueError(f"unknown topology method: {method}")


def _build_facets(mesh):
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    faces = []
    n_fail = 0
    for tri in F:
        f = ou.triangle_face(V[tri[0]], V[tri[1]], V[tri[2]])
        if f is None:
            n_fail += 1
            continue
        faces.append(f)
    shell = ou.sew_faces(faces)
    solid = ou.make_solid(shell)
    info = {"method": "facets", "n_triangle_faces": len(faces),
            "n_failed": n_fail}
    return solid, info


def _nurbs_face(mesh, region, nb):
    """Fit a B-spline surface to a region and merge its triangles onto it. One
    smooth face instead of many facets; guarded, so a bad drape falls back."""
    surf = nb.fit_bspline_surface(mesh, region)
    if surf is None:
        return None
    return merge_region_on_surface(mesh, region, surf, 0.0, 0.0)


def _facet_region(mesh, face_ids):
    """Faceted BRep faces for one region (fallback when a clean merged face
    can't be built). Keeps the region correct, just not reduced."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    out = []
    for f in face_ids:
        tri = F[f]
        face = ou.triangle_face(V[tri[0]], V[tri[1]], V[tri[2]])
        if face is not None:
            out.append(face)
    return out


def _region_faces_on_surface(mesh, region):
    """Build a region's triangles as faces on its shared analytic surface.
    Returns (faces, n_fail). Empty faces -> caller facets the region instead."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    surf = ou.surface_from_region(region.kind, region.params)
    if surf is None:
        return [], len(region.face_ids)
    sa = ou.surface_analyzer(surf)
    u_p, v_p = ou.surface_periods(region.kind)
    faces = []
    n_fail = 0
    for f in region.face_ids:
        tri = F[f]
        face = ou.triangle_face_on_surface(
            surf, sa, [V[tri[0]], V[tri[1]], V[tri[2]]], u_p, v_p)
        if face is None:
            n_fail += 1
        else:
            faces.append(face)
    return faces, n_fail


def merge_analytic_region(mesh, region):
    """Collapse one analytic region's triangles into its analytic face(s)."""
    surf = ou.surface_from_region(region.kind, region.params)
    if surf is None:
        return None
    u_p, v_p = ou.surface_periods(region.kind)
    return merge_region_on_surface(mesh, region, surf, u_p, v_p)


def merge_region_on_surface(mesh, region, surf, u_p, v_p):
    """Build each region triangle on the shared `surf` (arc edges in uv), sew,
    and UnifySameDomain into a single face. Works for any surface — analytic or
    a fitted B-spline (NURBS drape). Returns the merged shape, or None if too
    few triangles projected or the result fails the region-match guard."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    sa = ou.surface_analyzer(surf)
    tri_faces = []
    for f in region.face_ids:
        tri = F[f]
        fc = ou.triangle_face_on_surface(
            surf, sa, [V[tri[0]], V[tri[1]], V[tri[2]]], u_p, v_p)
        if fc is not None:
            tri_faces.append(fc)
    if len(tri_faces) < 0.5 * len(region.face_ids):
        return None
    sewn = ou.sew_faces(tri_faces, tol=1e-3)
    merged = ou.unify_same_domain(sewn, tol=1e-2)
    return merged if face_matches_region(mesh, region, merged) else None


def face_matches_region(mesh, region, shape) -> bool:
    """Accept an analytic face/shape only if it faithfully represents the
    region: area matches (a collapsed/degenerate merge fails), geometry is
    valid (bad seam/pcurve fails), and it stays close to the region mesh in both
    directions (a mis-centred torus that passes through the vertices but bulges
    elsewhere fails). Guards both the polyline merge and the clean-edge build."""
    mesh_area = float(np.asarray(mesh.area_faces)[np.asarray(region.face_ids)].sum())
    if mesh_area <= 0 or abs(ou.shape_area(shape) - mesh_area) > 0.15 * mesh_area:
        return False
    if not ou.is_valid(shape):
        return False
    if _region_deviation(mesh, region, shape) > _dev_limit(mesh):
        return False
    return True


def _region_deviation(mesh, region, merged_shape) -> float:
    """Bidirectional deviation between the region's mesh patch and the merged
    analytic surface. One direction (mesh->surface) misses a surface that passes
    through the vertices but bulges elsewhere (a mis-centred torus); the reverse
    (surface->mesh) catches it. Returns the larger p98."""
    import trimesh
    rv, rf = ou.shape_to_mesh(merged_shape, lin_defl=0.1)
    if len(rf) == 0:
        return float("inf")
    gen = trimesh.Trimesh(vertices=rv, faces=rf, process=False)

    F = np.asarray(mesh.faces)
    reg = mesh.submesh([np.asarray(region.face_ids)], append=True)

    # mesh vertices -> generated surface
    _, d1, _ = trimesh.proximity.closest_point(gen, reg.vertices)
    # generated-surface sample points -> mesh patch
    gpts, _ = trimesh.sample.sample_surface(gen, min(4000, 4 * len(gen.faces) + 10))
    _, d2, _ = trimesh.proximity.closest_point(reg, np.asarray(gpts))
    return max(float(np.quantile(d1, 0.98)), float(np.quantile(d2, 0.98)))


def _dev_limit(mesh) -> float:
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    return max(0.15, diag * 1e-3)


def _build_shell(mesh, segmentation):
    """Assemble a non-watertight analytic shell: each analytic region merged to
    its analytic face(s), freeform regions kept faceted. Real editable surfaces;
    free edges remain between regions (needs intersection curves to close)."""
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    stats = {"analytic_regions_merged": 0, "faceted_regions": 0, "by_kind": {}}
    for r in segmentation.regions:
        merged = merge_analytic_region(mesh, r) if r.is_analytic else None
        if merged is not None:
            builder.Add(comp, merged)
            stats["analytic_regions_merged"] += 1
            stats["by_kind"][r.kind] = stats["by_kind"].get(r.kind, 0) + 1
        else:
            for fc in _facet_region(mesh, r.face_ids):
                builder.Add(comp, fc)
            stats["faceted_regions"] += 1

    info = {"method": "shell", "brep_face_count": ou.count_faces(comp), **stats}
    return comp, info


def _build_solid_analytic(mesh, segmentation, *, sew_tol: float = 0.1):
    """Analytic watertight solid with clean edges.

    For each analytic region, try in order:
      1. clean_face   — analytic surface + clean intersection-curve edges
                        (circles/lines; curved faces via the uv-bounded seam
                        constructor). Small files, dimensionable edges.
      2. merge_region — analytic surface with polyline boundary (robust fallback
                        when the clean build fails on a region).
      3. facet        — keep the region's triangles (freeform, or both above
                        failed) so the solid never gets a hole.
    Then sew + heal + solidify. The clean analytic edges coincide exactly
    between neighbours, so tolerant sewing closes the body."""
    from . import edges as ed
    from . import nurbs as nb
    ctx = ed.build_context(mesh, segmentation)
    faces = []
    stats = {"clean_faces": 0, "merged_faces": 0, "nurbs_faces": 0,
             "freeform_faceted": 0, "by_kind": {}}
    for r in segmentation.regions:
        # 1. clean analytic face (analytic surface + intersection-curve edges)
        face = ed.clean_face(ctx, r) if r.is_analytic else None
        if face is not None and face_matches_region(mesh, r, face):
            faces.append(face)
            stats["clean_faces"] += 1
            stats["by_kind"][r.kind] = stats["by_kind"].get(r.kind, 0) + 1
            continue
        # 2. analytic surface with polyline boundary
        merged = merge_analytic_region(mesh, r) if r.is_analytic else None
        if merged is not None:
            faces.append(merged)
            stats["merged_faces"] += 1
            continue
        # 3. NURBS drape — smooth B-spline over any region (freeform, or an
        #    analytic fit that was rejected) so it doesn't stay faceted
        draped = _nurbs_face(mesh, r, nb)
        if draped is not None:
            faces.append(draped)
            stats["nurbs_faces"] += 1
            continue
        # 4. last resort: keep the triangles, so the solid never gets a hole
        faces.extend(_facet_region(mesh, r.face_ids))
        stats["freeform_faceted"] += 1

    solid = ou.sew_solidify(faces, tol=sew_tol)
    info = {"method": "solid", "brep_face_count": ou.count_faces(solid),
            "volume_mm3": round(ou.solid_volume(solid), 3),
            "edge_stats": ctx.stats, **stats}
    return solid, info


def _build_merged(mesh, segmentation):
    """Watertight planar-merge.

    Start from the faceted solid (already watertight — it *is* the mesh) and let
    OCCT's UnifySameDomain merge coplanar faces while preserving topology, then
    ShapeFix. This robustly collapses every planar region to a single face and
    stays valid + watertight + dimensionally exact.

    Curved regions remain faceted here: merging them into single analytic
    cylinder/torus faces requires shared-edge (intersection-curve) topology,
    which the per-region on-surface proof (see docs) shows works within a region
    but cannot yet be stitched watertight across regions. That is the next
    frontier; this method is the robust floor that never regresses.
    """
    solid, facet_info = _build_facets(mesh)
    n0 = ou.count_faces(solid)
    merged = ou.unify_same_domain(solid, tol=1e-3)
    healed = ou.heal(merged, tol=1e-3)
    # UnifySameDomain preserves the single closed shell, so re-solidify safely.
    out = ou.make_solid(healed)
    info = {
        "method": "merged(planar)",
        "faces_before": n0,
        "faces_after": ou.count_faces(out),
        "n_regions": len(segmentation.regions),
    }
    return out, info

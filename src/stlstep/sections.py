"""Planar material regions from oriented mesh intersections at branch vertices.

Tangent bores create degree-four contour nodes. Treating the linework as simple
nested loops can invert holes. Polygonized cells plus signed winding retain the
actual material without filling missing mesh faces or inventing closing edges.
"""
from __future__ import annotations
import numpy as np
import trimesh


class IncompleteSectionError(ValueError):
    """The measured intersection cannot establish a closed material region."""


def material_region(mesh, axis, coordinate):
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union
    if not mesh.is_winding_consistent:
        raise IncompleteSectionError('source mesh winding is inconsistent')
    uv = [a for a in range(3) if a != axis]
    # +UV orientation: XZ's normal is -Y, unlike XY and YZ.
    normal = np.cross(np.eye(3)[uv[0]], np.eye(3)[uv[1]])
    origin = np.zeros(3); origin[axis] = coordinate
    segments, face_ids = trimesh.intersections.mesh_plane(
        mesh, normal, origin, return_faces=True)
    if len(segments) == 0:
        return None
    directions = np.cross(normal, mesh.face_normals[face_ids])
    reverse = np.sum((segments[:,1]-segments[:,0])*directions, axis=1) < 0
    segments[reverse] = segments[reverse, ::-1]
    points = segments[:,:,uv]
    # Numerical noding only: <=1e-9 of diagonal, far below fit/gate tolerances.
    grid = max(1e-9, 1e-9*float(np.linalg.norm(mesh.extents)))
    anchor = np.asarray(mesh.bounds[0])[uv]
    quantized = np.rint((points-anchor)/grid).astype(np.int64)
    keep = np.any(quantized[:,0] != quantized[:,1], axis=1)
    quantized = quantized[keep]
    if len(quantized) == 0:
        return None
    _, inverse = np.unique(quantized.reshape(-1,2), axis=0, return_inverse=True)
    incoming = np.bincount(inverse[1::2], minlength=int(inverse.max())+1)
    outgoing = np.bincount(inverse[::2], minlength=int(inverse.max())+1)
    if np.any(incoming != outgoing):
        raise IncompleteSectionError(
            f'open intersection boundary at {"xyz"[axis]}={coordinate:g}')
    points = quantized*grid+anchor
    cells = list(polygonize(unary_union([LineString(line) for line in points])))
    if not cells:
        raise IncompleteSectionError('closed intersection produced no planar cells')
    filled = []
    for cell in cells:
        if cell.area <= grid*grid:
            continue
        point = np.asarray(cell.representative_point().coords[0])
        a, b = points[:,0]-point, points[:,1]-point
        angles = np.arctan2(a[:,0]*b[:,1]-a[:,1]*b[:,0], np.sum(a*b,axis=1))
        winding = float(angles.sum()/(2*np.pi))
        if abs(winding-round(winding)) > 1e-5:
            raise IncompleteSectionError('ambiguous material winding in section')
        if round(winding) > 0:
            filled.append(cell)
    if not filled:
        raise IncompleteSectionError('intersection has no positive material winding')
    region = unary_union(filled)
    if region.is_empty or not region.is_valid:
        raise IncompleteSectionError('intersection produced an invalid material region')
    return region


def section_polygons(mesh, axis, coordinate):
    region = material_region(mesh, axis, coordinate)
    if region is None:
        return None
    return [(np.asarray(p.exterior.coords), [np.asarray(h.coords) for h in p.interiors])
            for p in getattr(region, 'geoms', [region])]

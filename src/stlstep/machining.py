"""Conservative block + cylindrical-feature reconstruction from measured surfaces.

This is a proposal generator. It neither infers dimensions from fixture names nor
changes validation: draw() measures the proposal, and compose runs the full gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MachiningSpec:
    lower: np.ndarray
    upper: np.ndarray
    holes: list[dict]
    bosses: list[dict]
    pockets: list[dict] = field(default_factory=list)


def _body_bounds(mesh):
    """Find six dominant outward-facing planar support levels.

    A boss's small end cap must not replace the block's larger shoulder plane.
    Coordinates are area-weighted; coplanar fragments separated by holes count
    together. This currently recognizes axis-aligned stock, in any translation.
    """
    areas = np.asarray(mesh.area_faces)
    centers = np.asarray(mesh.triangles_center)
    normals = np.asarray(mesh.face_normals)
    tolerance = max(1e-6, 1e-6*float(np.linalg.norm(mesh.extents)))
    lower, upper = np.zeros(3), np.zeros(3)
    supports = []
    for axis in range(3):
        for sign, bounds in [(-1, lower), (1, upper)]:
            selected = np.flatnonzero(sign*normals[:, axis] > 1-1e-6)
            if not len(selected):
                return None
            keys = np.round(centers[selected, axis]/tolerance).astype(np.int64)
            groups = []
            for key in np.unique(keys):
                ids = selected[keys == key]
                area = float(areas[ids].sum())
                level = float(np.average(centers[ids, axis], weights=areas[ids]))
                groups.append((area, level))
            projected = float(np.prod(np.delete(mesh.extents, axis)))
            supported = [(area, level) for area, level in groups if area >= .2*projected]
            if not supported:
                return None
            area, level = max(supported, key=lambda item: sign*item[1])
            bounds[axis] = level
            supports.append((axis, area))
    size = upper-lower
    if np.any(size <= tolerance):
        return None
    for axis, area in supports:
        face_area = float(np.prod(np.delete(size, axis)))
        if area < .2*face_area:
            return None
    return lower, upper


def _supported_feature(feature):
    # Recovered section/open-bore hints and multi-stage features need further
    # measurements. Do not silently omit or guess them in this narrow generator.
    if (feature.get('source') or feature.get('threaded')
            or feature.get('end_treatments')):
        return False
    cb = feature.get('counterbore')
    if cb and not (cb.get('end') in ('low', 'high') and cb.get('stage_count') == 2 and cb.get('contiguous') is True
                   and float(cb.get('diameter_mm', 0)) > float(feature.get('diameter_mm', 0))
                   and 0 < float(cb.get('depth_mm', 0)) < float(feature.get('depth_mm', 0))):
        return False
    axis = np.asarray(feature.get('axis', []), dtype=float)
    positions = feature.get('positions_mm', [])
    return (axis.shape == (3,) and np.all(np.isfinite(axis))
            and np.linalg.norm(axis) > 0
            and float(feature.get('diameter_mm', 0)) > 0
            and float(feature.get('depth_mm', 0)) > 0
            and int(feature.get('count', 1)) > 0
            and len(positions) == int(feature.get('count', 1))
            and all(np.asarray(p).shape == (3,) and np.all(np.isfinite(p)) for p in positions))


def detect_machining(mesh):
    """A measured rectangular body with confident simple holes and bosses."""
    if len(mesh.faces) > 100000 or not mesh.is_winding_consistent:
        return None
    bounds = _body_bounds(mesh)
    if bounds is None:
        return None
    from .ingest import build_face_neighbors
    from .segment_rg import segment_rg
    from .compose import _cylinder_desc
    from .features import group_features
    segmentation = segment_rg(mesh, build_face_neighbors(mesh))
    surfaces = [_cylinder_desc(mesh, region) for region in segmentation.regions
                if region.kind == 'cylinder']
    features = group_features(mesh, segmentation, surfaces, mesh.extents)
    holes, bosses = features.get('holes', []), features.get('bosses', [])
    pockets = features.get('pockets', [])
    if (not holes or not all(_supported_pocket(p) for p in pockets)
            or sum(f.get('count', 1) for f in holes+bosses) > 64
            or not all(_supported_feature(f) for f in holes+bosses)):
        return None
    return MachiningSpec(*bounds, holes, bosses, pockets)


def _supported_pocket(p):
    try:
        n = np.asarray(p['opening_normal'], float)
        axes = np.asarray(p['plane_axes'], float)
        center = np.asarray(p['center_mm'], float)
        size = np.asarray(p['size_mm'], float)
        basis = np.vstack([axes, n])
        return (p.get('profile_kind') == 'rectangle' and basis.shape == (3,3)
                and np.all(np.isfinite(basis)) and np.allclose(basis@basis.T, np.eye(3))
                and np.allclose(np.max(abs(basis),axis=1), 1)
                and np.linalg.det(basis) > .99 and center.shape == (3,)
                and np.all(np.isfinite(center)) and size.shape == (2,)
                and np.all(np.isfinite(size)) and np.all(size > 0)
                and np.isfinite(p['depth_mm']) and p['depth_mm'] > 0)
    except (KeyError, TypeError, ValueError):
        return False


def _vector(values):
    return '(' + ', '.join(format(float(v), '.12g') for v in values) + ')'


def _boss_attachment(start, axis, depth, lower, upper):
    """Reconcile feature-table rounding with the measured supporting plane.

    Grouped positions are rounded to 0.01 mm. An intended touching boss can
    otherwise become a disconnected solid. For an axial boss, preserve its high
    endpoint and move the start only within that recorded position resolution.
    """
    dominant = int(np.argmax(np.abs(axis)))
    if abs(axis[dominant]) < 1-1e-6:
        return start, depth
    nearest = min((lower[dominant], upper[dominant]),
                  key=lambda value: abs(value-start[dominant]))
    if abs(nearest-start[dominant]) > .01:
        return start, depth
    shift = (nearest-start[dominant])/axis[dominant]
    if depth-shift <= 0:
        return start, depth
    return start+shift*axis, depth-shift


def machining_script(spec):
    """Emit actual stock/add/cut operations with endpoint-anchored cylinders.

    Feature positions are HIGH axial endpoints, not cylinder midpoints. Explicit
    MIN alignment prevents moving a boss above its shoulder or shortening a bore.
    """
    lower, upper = np.asarray(spec.lower), np.asarray(spec.upper)
    lines = ['from build123d import *', '',
             '# Measured stock, cylindrical additions, then bores on their own axes.',
             'with BuildPart() as part:',
             f'    with Locations({_vector(lower)}):',
             '        Box(' + ', '.join(format(float(v), '.12g') for v in upper-lower)
             + ', align=(Align.MIN, Align.MIN, Align.MIN))']
    for kind, features, mode in [('boss', spec.bosses, 'ADD'), ('bore', spec.holes, 'SUBTRACT')]:
        for feature in features:
            axis = np.array(feature['axis'], dtype=float, copy=True)
            axis /= np.linalg.norm(axis)
            depth = float(feature['depth_mm'])
            radius = .5*float(feature['diameter_mm'])
            for high in feature['positions_mm']:
                start = np.asarray(high, dtype=float) - depth*axis
                height = depth
                if kind == 'boss':
                    start, height = _boss_attachment(start, axis, depth, lower, upper)
                lines += [f'    # {kind}: high endpoint {_vector(high)}',
                          f'    with Locations(Plane(origin={_vector(start)}, z_dir={_vector(axis)})):',
                          f'        Cylinder({radius:.12g}, {height:.12g}, '
                          f'align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.{mode})']
                cb = feature.get('counterbore')
                if cb:
                    cbdepth = float(cb['depth_mm'])
                    cbstart = np.asarray(high)-cbdepth*axis if cb['end']=='high' else start
                    lines += [f'    with Locations(Plane(origin={_vector(cbstart)}, z_dir={_vector(axis)})):',
                              f"        Cylinder({.5*cb['diameter_mm']:.12g}, {cbdepth:.12g}, "
                              'align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)']
    for pocket in spec.pockets:
        if not _supported_pocket(pocket):
            raise ValueError('unsupported pocket profile or frame')
        width, height = pocket['size_mm']
        lines += [f"    with Locations(Plane(origin={_vector(pocket['center_mm'])}, "
                  f"x_dir={_vector(pocket['plane_axes'][0])}, z_dir={_vector(pocket['opening_normal'])})):",
                  f"        Box({width:.12g}, {height:.12g}, {pocket['depth_mm']:.12g}, "
                  'align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)']
    return '\n'.join(lines) + '\n'


def draw_machining(spec):
    from .compose import exec_script
    script = machining_script(spec)
    part = exec_script(script)
    if not part.is_valid or len(part.solids()) != 1:
        raise ValueError('machining proposal must form one valid connected solid')
    return part, script, {'body_planes': 6,
                          'holes': sum(h.get('count', 1) for h in spec.holes),
                          'bosses': sum(b.get('count', 1) for b in spec.bosses),
                          'pockets': len(spec.pockets)}

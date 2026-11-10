"""Measured rounded rectangular extrusions with one uniform 45-degree chamfer."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class EdgeBlock:
    lower: np.ndarray
    upper: np.ndarray
    axis: int
    radius: float
    chamfer: float
    chamfer_end: str
    holes: list  # [(world transverse centre, radius), ...], all through-bores


def _section(mesh, axis, coordinate):
    from shapely.geometry import Polygon
    from .draw import _section_polys
    polygons = _section_polys(mesh, axis, coordinate)
    if not polygons or len(polygons) != 1:
        return None
    outer, holes = polygons[0]
    outline = Polygon(outer)
    if not outline.is_valid or outline.is_empty:
        return None
    return outline, outer, holes


def _circles(holes, tolerance):
    from .draw import fit_loop
    result = []
    for points in holes:
        segments = fit_loop(points)
        if not segments or len(segments) != 1 or segments[0].kind != 'circle':
            return None
        circle = segments[0]
        errors = np.abs(np.linalg.norm(points-np.asarray(circle.center), axis=1)-circle.radius)
        if np.max(errors) > tolerance:
            return None
        result.append((np.asarray(circle.center), float(circle.radius)))
    return sorted(result, key=lambda item: tuple(item[0]))


def _same_holes(reference, candidate, tolerance):
    if candidate is None or len(candidate) != len(reference):
        return False
    unused = list(range(len(candidate)))
    for center, radius in reference:
        index = min(unused, key=lambda i: np.linalg.norm(candidate[i][0]-center))
        other_center, other_radius = candidate[index]
        if np.linalg.norm(center-other_center) > tolerance or abs(radius-other_radius) > tolerance:
            return False
        unused.remove(index)
    return True


def detect_edgeblock(mesh):
    """Require a rounded-rectangle section, constant bores, and linear end inset.

    The narrow scope is deliberate: variable radii, tapered bores, two treated
    ends, nonuniform chamfers, and continuously drafted walls are not guessed.
    """
    from shapely.geometry import box
    from .draw import fit_loop
    if not mesh.is_watertight or len(mesh.faces) > 100000:
        return None
    lower, upper = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
    tolerance = max(.02, 1e-4*float(np.linalg.norm(mesh.extents)))
    for axis in (2, 1, 0):
        uv = [i for i in range(3) if i != axis]
        height = upper[axis]-lower[axis]
        if height <= 10*tolerance:
            continue
        reference = _section(mesh, axis, .5*(lower[axis]+upper[axis]))
        if reference is None:
            continue
        outline, points, holes = reference
        expected_bounds = np.r_[lower[uv], upper[uv]]
        if np.max(np.abs(np.asarray(outline.bounds)-expected_bounds)) > tolerance:
            continue
        segments = fit_loop(points)
        arcs = [s for s in (segments or []) if s.kind == 'arc']
        if len(arcs) < 4:
            continue
        radius = float(np.median([s.radius for s in arcs]))
        width, depth = upper[uv]-lower[uv]
        if radius <= 2*tolerance or 2*radius >= min(width, depth)-2*tolerance:
            continue
        center = .5*(lower[uv]+upper[uv])
        corner_centers = [center+np.array([sx*(width/2-radius), sy*(depth/2-radius)])
                          for sx in (-1,1) for sy in (-1,1)]
        if any(not any(abs(a.radius-radius) <= tolerance and
                       np.linalg.norm(np.asarray(a.center)-c) <= tolerance
                       for a in arcs) for c in corner_centers):
            continue
        rounded = box(*(np.r_[lower[uv]+radius, upper[uv]-radius])).buffer(radius, quad_segs=64)
        if rounded.boundary.hausdorff_distance(outline.exterior) > tolerance:
            continue
        bores = _circles(holes, tolerance)
        if bores is None:
            continue
        epsilon = max(1e-4, 1e-5*height)
        caps = [_section(mesh, axis, lower[axis]+epsilon),
                _section(mesh, axis, upper[axis]-epsilon)]
        if any(cap is None for cap in caps):
            continue
        insets = []
        for cap in caps:
            b = np.asarray(cap[0].bounds)
            sides = np.r_[b[:2]-lower[uv], upper[uv]-b[2:]]
            if np.min(sides) < -tolerance or np.ptp(sides) > tolerance:
                break
            insets.append(float(np.mean(sides)))
        if len(insets) != 2:
            continue
        treated = [i for i, inset in enumerate(insets) if inset > 3*tolerance]
        if len(treated) != 1:
            continue
        end = treated[0]
        chamfer = float(insets[end]+epsilon)
        if insets[1-end] > tolerance or chamfer >= min(.45*height, radius-tolerance):
            continue
        # Nine interior stations and the near-cap sections must match the same
        # constant body + linear inward offset; a rounded lip is not a chamfer.
        coordinates = [lower[axis]+height*f for f in np.linspace(.05,.95,9)]
        coordinates += [lower[axis]+epsilon, upper[axis]-epsilon]
        # Sample the inferred bevel itself even when it lies entirely between
        # the global stations. Unequal chamfer legs must not be assumed 45°.
        for fraction in (.1, .5, .9, 1.1):
            distance = chamfer*fraction
            coordinates.append(upper[axis]-distance if end else lower[axis]+distance)
        matched = True
        for coordinate in coordinates:
            section = _section(mesh, axis, coordinate)
            if section is None:
                matched = False; break
            distance_to_end = upper[axis]-coordinate if end else coordinate-lower[axis]
            inset = max(0., chamfer-distance_to_end)
            expected = rounded.buffer(-inset, quad_segs=64) if inset else rounded
            if (expected.is_empty or
                    expected.boundary.hausdorff_distance(section[0].exterior) > tolerance or
                    not _same_holes(bores, _circles(section[2], tolerance), tolerance)):
                matched = False; break
        if matched:
            return EdgeBlock(lower.copy(), upper.copy(), axis, radius, chamfer,
                             'high' if end else 'low', bores)
    return None


def edgeblock_script(spec):
    from .machining import _vector
    lower, upper = np.asarray(spec.lower), np.asarray(spec.upper)
    direction = np.eye(3)[spec.axis]
    axis = f'Axis((0, 0, 0), {_vector(direction)})'
    index = -1 if spec.chamfer_end == 'high' else 0
    lines = ['from build123d import *', '',
             '# Measured stock, longitudinal rounds, one end chamfer, through-bores.',
             'with BuildPart() as part:',
             f'    with Locations({_vector(lower)}):',
             '        Box(' + ', '.join(format(float(x), '.12g') for x in upper-lower)
             + ', align=(Align.MIN, Align.MIN, Align.MIN))',
             f'    fillet(part.edges().filter_by({axis}), radius={spec.radius:.12g})',
             f'    end_face = part.faces().sort_by({axis})[{index}]',
             f'    chamfer(end_face.outer_wire().edges(), length={spec.chamfer:.12g})']
    uv = [i for i in range(3) if i != spec.axis]
    for center, radius in spec.holes:
        start = lower.copy(); start[uv] = center
        lines += [f'    with Locations(Plane(origin={_vector(start)}, z_dir={_vector(direction)})):',
                  f'        Cylinder({radius:.12g}, {upper[spec.axis]-lower[spec.axis]:.12g}, '
                  'align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)']
    return '\n'.join(lines)+'\n'


def draw_edgeblock(spec):
    from .compose import exec_script
    script = edgeblock_script(spec)
    part = exec_script(script)
    if not part.is_valid or len(part.solids()) != 1:
        raise ValueError('edge-treated proposal must form one valid connected solid')
    return part, script, {'vertical_fillets':4, 'fillet_radius_mm':spec.radius,
                          'chamfer_mm':spec.chamfer, 'chamfer_end':spec.chamfer_end,
                          'holes':len(spec.holes), 'axis':spec.axis}

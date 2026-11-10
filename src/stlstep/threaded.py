"""Measured single-start ISO internal threads in plain cylindrical rings.

Identification uses the same signature routine as ``probe thread``, then checks
whole-wall helical phase and triangular flank evidence. Nominal CAD and its
precise measured relief band travel together; no fixture names are consulted.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ThreadedRing:
    axis: int
    low_point: tuple
    height: float
    outer_radius: float
    minor_diameter: float
    evidence: dict

    def features(self):
        direction = np.eye(3)[self.axis]
        high = np.asarray(self.low_point) + self.height*direction
        common = {'axis': direction.tolist(), 'depth_mm': self.height,
                  'through': True, 'count': 1, 'positions_mm': [high.tolist()],
                  'arrangement': 'single'}
        return {'holes': [{**common, 'diameter_mm': self.minor_diameter,
                           'threaded': self.evidence}],
                'bosses': [{**common, 'diameter_mm': 2*self.outer_radius}],
                'pockets': [], 'thread_candidates': [], 'low_confidence_fits': [],
                'notes': ['Confirmed single-start internal thread; nominal minor-diameter '
                          'bore. Reapply the recorded thread specification in downstream CAD.']}

    def bands(self):
        from .featurecheck import thread_bands
        return thread_bands({'features': self.features()})


def _helical_phase(theta, height, radius, pitch_hint):
    """A true thread advances one radial cycle per revolution and axial pitch.

    The fundamental locates pitch/hand; triangular flank agreement then rejects
    sinusoidal grooves, knurling, annular grooves and incoherent periodic noise.
    """
    from scipy.optimize import minimize_scalar
    whole_theta, whole_height, whole_radius = theta, height, radius
    ids = np.linspace(0, len(radius)-1, min(len(radius), 4096)).astype(int)
    theta, height, radius = theta[ids], height[ids], radius[ids]
    variance = float(np.sum((radius-radius.mean())**2))
    if variance <= 0:
        return None

    def fit(pitch, hand):
        phase = theta + hand*2*np.pi*height/pitch
        design = np.column_stack([np.ones(len(radius)), np.cos(phase), np.sin(phase)])
        coef = np.linalg.lstsq(design, radius, rcond=None)[0]
        r2 = 1-float(np.sum((radius-design@coef)**2))/variance
        return r2, coef, phase

    grid = np.linspace(.88*pitch_hint, 1.12*pitch_hint, 65)
    proposals = [(fit(p, hand)[0], p, hand) for hand in (-1, 1) for p in grid]
    _, pitch, hand = max(proposals)
    step = grid[1]-grid[0]
    result = minimize_scalar(lambda p: -fit(p, hand)[0],
                             bounds=(pitch-step, pitch+step), method='bounded',
                             options={'xatol': 1e-6*pitch_hint})
    pitch = float(result.x)
    r2, coef, phase = fit(pitch, hand)
    if r2 < .95:
        return None
    offset = np.arctan2(coef[2], coef[1])
    triangle = 2/np.pi*np.arcsin(np.clip(np.cos(phase-offset), -1, 1))
    design = np.column_stack([np.ones(len(radius)), triangle])
    coef = np.linalg.lstsq(design, radius, rcond=None)[0]
    triangle_r2 = 1-float(np.sum((radius-design@coef)**2))/variance
    if triangle_r2 < .995 or coef[1] <= 0:
        return None
    # Fit efficiently on a bounded subset, but validate every measured vertex.
    phase = whole_theta + hand*2*np.pi*whole_height/pitch
    triangle = 2/np.pi*np.arcsin(np.clip(np.cos(phase-offset), -1, 1))
    residual = whole_radius-(coef[0]+coef[1]*triangle)
    whole_variance = float(np.sum((whole_radius-whole_radius.mean())**2))
    whole_r2 = 1-float(residual@residual)/whole_variance
    if whole_r2 < .995:
        return None
    return {'full_wall_r2': whole_r2, 'measured_pitch_mm': pitch, 'hand': 'right' if hand == -1 else 'left',
            'fundamental_r2': r2, 'triangular_flank_r2': triangle_r2}


def detect_threaded_ring(mesh):
    if (len(mesh.faces) > 200000 or not mesh.is_watertight
            or not mesh.is_winding_consistent or mesh.body_count != 1):
        return None
    vertices = np.asarray(mesh.vertices)
    diag = float(np.linalg.norm(mesh.extents))
    tol = max(1e-5, 1e-6*diag)
    for axis in range(3):
        direction = np.eye(3)[axis]
        uv = [a for a in range(3) if a != axis]
        centre = mesh.bounds[:, uv].mean(0)
        radial = np.linalg.norm(vertices[:, uv]-centre, axis=1)
        outer_radius = float(radial.max())
        outer = abs(radial-outer_radius) <= tol
        inner = radial < .9*outer_radius
        if not np.all(outer | inner) or inner.sum() < 256 or outer.sum() < 64:
            continue
        lo, hi = (float(v) for v in mesh.bounds[:, axis])
        height = hi-lo
        if height <= 0:
            continue
        t = vertices[:, axis]
        faces = mesh.faces
        caps = ((np.max(abs(t[faces]-lo), axis=1) <= tol)
                | (np.max(abs(t[faces]-hi), axis=1) <= tol))
        outside, inside = np.all(outer[faces], axis=1), np.all(inner[faces], axis=1)
        if (not np.all(caps | outside | inside) or not np.any(outside & ~caps)
                or not np.any(inside & ~caps)):
            continue
        # A smooth circular stock wall and two flat annular caps are required.
        normals = mesh.face_normals
        stock_ids = np.flatnonzero(outside & ~caps)
        stock_radial = mesh.triangles_center[stock_ids][:, uv]-centre
        stock_radial /= np.linalg.norm(stock_radial, axis=1)[:, None]
        if (np.max(abs(normals[stock_ids, axis])) > 1e-5
                or np.min(np.sum(normals[stock_ids][:, uv]*stock_radial, axis=1)) < .995
                or np.any(abs(normals[caps, axis]) < 1-1e-5)):
            continue
        wall_ids = np.flatnonzero(inside & ~caps)
        wall_radial = mesh.triangles_center[wall_ids][:, uv]-centre
        if np.any(np.sum(normals[wall_ids][:, uv]*wall_radial, axis=1) >= 0):
            continue
        point = np.zeros(3); point[uv] = centre; point[axis] = lo
        r = radial[inner]; axial = t[inner]-lo
        r0, r1 = float(r.min()), float(r.max())
        if r1-r0 < .1 or abs(axial.min()) > tol or abs(axial.max()-height) > tol:
            continue
        from .features import thread_signature
        signature = thread_signature(mesh, direction, point,
                                     t_range=(0, height), r_hint=.5*(r0+r1), internal=True)
        if (signature is None or not signature.get('spec')
                or not str(signature['spec']).startswith('M')
                or abs(signature['taper_deg']) > .1):
            continue
        e1 = np.eye(3)[uv[0]]; e2 = np.cross(direction, e1)
        rel = vertices[inner]-point
        theta = np.arctan2(rel@e2, rel@e1)
        angles = np.sort(np.unique(np.round(theta, 6)))
        gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
        levels = np.unique(np.round(axial, 6))
        if (gaps.max() > 2*np.pi/32 or len(levels) < 24
                or np.diff(levels).max() > signature['pitch_mm']/8
                or height < 3*signature['pitch_mm']):
            continue
        # Entry recesses, partial threads and tapered envelopes must not share
        # a whole-height exclusion. Every axial quarter needs the same relief.
        envelopes = [(r[(axial>=height*k/4)&(axial<=height*(k+1)/4)].min(),
                      r[(axial>=height*k/4)&(axial<=height*(k+1)/4)].max()) for k in range(4)]
        if np.max(np.ptp(envelopes, axis=0)) > .01:
            continue
        phase = _helical_phase(theta, axial, r, signature['pitch_mm'])
        if phase is None:
            continue
        major, pitch = signature['spec_major_mm'], signature['spec_pitch_mm']
        minor = round(major-1.0825*pitch, 3)
        if (abs(phase['measured_pitch_mm']-pitch) > .02*pitch
                or abs(2*r1-major) > max(.03,.02*pitch)
                or abs(2*r0-minor) > max(.03,.03*pitch)):
            continue
        # 0.05 mm is compose's existing tessellation chord allowance. The
        # measured shell, not a broad feature-size padding, defines this band.
        band = {'r': [min(r0, minor/2)-.05, max(r1, major/2)+.05],
                't': [-height, 0.0]}
        if band['r'][0] <= 0 or outer_radius-band['r'][1] < .1*outer_radius:
            continue
        evidence = {**signature, **phase, 'spec': f'M{major:g}x{pitch:g}',
                    'nominal_minor_diameter_mm': minor,
                    'measured_relief_radii_mm': [r0,r1],
                    'validation_band': band, 'source': 'probe-signature+whole-wall-phase',
                    'representation': 'nominal_internal_minor', 'tessellation_allowance_mm': .05}
        return ThreadedRing(axis, tuple(float(x) for x in point), height,
                            outer_radius, minor, evidence)
    return None


def draw_threaded_ring(spec):
    from .compose import exec_script
    axis = tuple(float(v) for v in np.eye(3)[spec.axis])
    script = ('from build123d import *\n\n'
              f"# {spec.evidence['spec']} {spec.evidence['hand']}-hand internal thread.\n"
              '# Nominal tap-drill representation; reapply Thread in downstream CAD.\n'
              f"# Measured validation band: {spec.bands()!r}\n"
              'with BuildPart() as part:\n'
              f'    with Locations(Plane(origin={spec.low_point!r}, z_dir={axis!r})):\n'
              f'        Cylinder({spec.outer_radius!r}, {spec.height!r}, '
              'align=(Align.CENTER, Align.CENTER, Align.MIN))\n'
              f'        Cylinder({spec.minor_diameter/2!r}, {spec.height!r}, '
              'align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)\n')
    part = exec_script(script)
    if not part.is_valid or len(part.solids()) != 1:
        raise ValueError('nominal thread ring must form one valid connected solid')
    return part, script, {'representation': 'nominal_thread',
                          'thread_evidence': spec.evidence,
                          'validation_bands': spec.bands()}

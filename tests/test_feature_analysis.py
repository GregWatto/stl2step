"""Keep genuine CAD features while rejecting positional-only surface fits."""
import unittest
from pathlib import Path
import numpy as np
import trimesh
from stlstep.compose import _cylinder_support, detect_edge_treatments, exec_script
from stlstep.draw import (Seg, Lofted, fit_loop, _constant_loft_hole,
                          _build_lofted, _emit_lofted_script)
from stlstep.featurecheck import candidate_features
from stlstep.ingest import load_and_repair, build_face_neighbors
from stlstep.segment_rg import segment_rg


class CylinderEvidenceTests(unittest.TestCase):
    def test_circle_vertices_do_not_make_flat_facets_a_cylinder(self):
        for sections, expected in [(4, False), (64, True)]:
            with self.subTest(sections=sections):
                mesh = trimesh.creation.cylinder(radius=5, height=10, sections=sections)
                walls = np.flatnonzero(abs(mesh.face_normals[:, 2]) < .1)
                evidence = _cylinder_support(mesh, walls, {
                    'axis': [0, 0, 1], 'center_mm': [0, 0, 0]})
                self.assertEqual(evidence['confident'], expected)

    def test_true_fillets_still_detected(self):
        mesh = load_and_repair(str(Path(__file__).parent / 'fixtures/filletblock.stl')).mesh
        neighbors = build_face_neighbors(mesh)
        result = detect_edge_treatments(mesh, segment_rg(mesh, neighbors), neighbors)
        self.assertGreaterEqual(sum(f['count'] for f in result['fillets']), 4)


class LoftBoreTests(unittest.TestCase):
    def fitted(self, radii=(3., 3.000001), centers=((0, 0), (0, 0))):
        outer = fit_loop(np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]]))
        circle = Seg('circle', (0, 0), (0, 0), center=(0, 0), radius=8)
        holes = [(z, [Seg('circle', (0, 0), (0, 0), center=c, radius=r)])
                 for z, r, c in zip([0., 20.], radii, centers)]
        return ([(0., outer), (20., [circle])], holes)

    def test_constant_bore_stays_analytic_in_kernel_and_emitted_script(self):
        fitted = self.fitted()
        spec = Lofted(axis=2, stations=[])
        actual = _build_lofted(spec, fitted, [0, 1])
        script = _emit_lofted_script(spec, fitted, [0, 1])
        rebuilt = exec_script(script)
        for part in (actual, rebuilt):
            self.assertTrue(any(abs(r - 3) < 1e-4 for r in
                                candidate_features(part.wrapped)['cylinder_radii']))
        self.assertAlmostEqual(actual.volume, rebuilt.volume, places=4)

    def test_tapered_or_offset_bore_is_not_made_constant(self):
        for radii, centers in [((3., 3.1), ((0, 0), (0, 0))),
                               ((3., 3.), ((0, 0), (.1, 0)))]:
            fitted = self.fitted(radii, centers)
            self.assertFalse(_constant_loft_hole(fitted[1], [0, 1]))
            script = _emit_lofted_script(Lofted(axis=2, stations=[]), fitted, [0, 1])
            self.assertIn('loft(mode=Mode.SUBTRACT)', script)


class PrismaticScriptTests(unittest.TestCase):
    def test_adjacent_slabs_do_not_gain_rounding_gaps(self):
        from stlstep.draw import Prismatic, _build_prismatic, _emit_prismatic_script
        profile = fit_loop(np.array([[0, 0], [20, 0], [20, 10], [0, 10]]))
        join = 20.4665938
        stop = 106.9665938
        fitted = [(0., join, [(profile, [])]), (join, stop, [(profile, [])])]
        for axis in (0, 1, 2):
            with self.subTest(axis=axis):
                spec = Prismatic(axis=axis, levels=[])
                built = _build_prismatic(spec, fitted)
                rebuilt = exec_script(_emit_prismatic_script(spec, fitted))
                self.assertEqual(len(rebuilt.solids()), 1)
                self.assertAlmostEqual(rebuilt.volume, built.volume, places=6)


class ProfileFitTests(unittest.TestCase):
    def test_fine_quarter_circle_does_not_collapse_to_chord(self):
        from stlstep.draw import _curvature_fit
        for count in (24, 180, 720):
            angles = np.linspace(0, np.pi/2, count)
            points = 5*np.column_stack((np.cos(angles), np.sin(angles)))
            segments = _curvature_fit(points, 4, .02, .06, True)
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].kind, 'arc')
            self.assertAlmostEqual(segments[0].radius, 5, places=6)

    def test_polyline_fallback_preserves_fine_curve(self):
        from stlstep.draw import _curvature_fit
        angles = np.linspace(0, np.pi/2, 180)
        points = 5*np.column_stack((np.cos(angles), np.sin(angles)))
        segments = _curvature_fit(points, 4, .02, .06, False)
        self.assertGreater(len(segments), 4)
        self.assertTrue(all(s.kind == 'line' for s in segments))


class ProfileEvidenceTests(unittest.TestCase):
    def test_near_axial_lines_snap_but_real_taper_stays(self):
        from stlstep.draw import _align_profile_lines
        points = [(5, 0), (5.002, 4), (7, 4), (7, 0)]
        segments = [Seg('line', points[i], points[(i+1)%4]) for i in range(4)]
        aligned = _align_profile_lines(segments)
        self.assertEqual(aligned[0].p0[0], aligned[0].p1[0])
        self.assertTrue(all(aligned[i].p1 == aligned[(i+1)%4].p0 for i in range(4)))
        segments[0].p1 = (5.2, 4)
        segments[1].p0 = (5.2, 4)
        aligned = _align_profile_lines(segments)
        self.assertNotEqual(aligned[0].p0[0], aligned[0].p1[0])

    def test_circle_boundary_rejects_circular_fit_of_square_corners(self):
        from stlstep.features import _circle_boundary_supported
        square = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])
        self.assertFalse(_circle_boundary_supported(square, 0, 0, np.sqrt(2)))
        angles = np.linspace(0,2*np.pi,180)
        circle = 5*np.column_stack((np.cos(angles), np.sin(angles)))
        self.assertTrue(_circle_boundary_supported(circle, 0, 0, 5))


if __name__ == '__main__':
    unittest.main()

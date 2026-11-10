"""Machining proposals must generalize beyond fixture dimensions/orientation."""
from pathlib import Path
import unittest
import numpy as np
import trimesh
from stlstep.ingest import load_and_repair
from stlstep.machining import (MachiningSpec, detect_machining, draw_machining,
                               _body_bounds, _supported_feature)


class MachiningTests(unittest.TestCase):
    def test_translated_scaled_axis_permutation_is_detected(self):
        for translation in ([11,-8,17], [11.126,-8.2345,17.3456]):
            with self.subTest(translation=translation):
                mesh = load_and_repair(str(Path(__file__).parent/'fixtures/crossblock.stl')).mesh
                matrix = np.array([[0,0,1], [1,0,0], [0,1,0]], dtype=float)
                mesh.vertices = 1.4*(mesh.vertices @ matrix.T) + translation
                spec = detect_machining(mesh)
                self.assertIsNotNone(spec)
                np.testing.assert_allclose(spec.lower, np.array(translation)+[0,-35,-28], atol=1e-5)
                np.testing.assert_allclose(spec.upper, np.array(translation)+[28,35,28], atol=1e-5)
                part, script, stats = draw_machining(spec)
                self.assertTrue(part.is_valid)
                self.assertEqual(len(part.solids()), 1)
                self.assertEqual(stats['holes'], 5)
                self.assertEqual(stats['bosses'], 1)
                self.assertLess(abs(part.volume-mesh.volume)/mesh.volume, .002)
                self.assertNotIn('crossblock', script)

    def test_cylinder_high_endpoints_are_not_treated_as_centers(self):
        lower, upper = np.array([10.,-15.,3.]), np.array([50.,15.,23.])
        size = upper-lower
        for axis in range(3):
            with self.subTest(boss_axis=axis):
                direction = np.eye(3)[axis]
                boss_end = .5*(lower+upper)
                boss_end[axis] = upper[axis]+7
                hole_axis = (axis+1)%3
                hole_end = .5*(lower+upper)
                hole_end[hole_axis] = upper[hole_axis]
                hole_end[(axis+2)%3] = lower[(axis+2)%3]+3
                boss = {'axis':direction, 'depth_mm':7, 'diameter_mm':8,
                        'positions_mm':[boss_end.tolist()], 'count':1}
                hole = {'axis':np.eye(3)[hole_axis], 'depth_mm':size[hole_axis],
                        'diameter_mm':4, 'positions_mm':[hole_end.tolist()], 'count':1}
                part, _, _ = draw_machining(MachiningSpec(lower, upper, [hole], [boss]))
                expected = np.prod(size)+np.pi*16*7-np.pi*4*size[hole_axis]
                self.assertAlmostEqual(part.volume, expected, places=5)
                self.assertEqual(len(part.solids()), 1)
                np.testing.assert_allclose(tuple(part.bounding_box().min), lower, atol=1e-6)
                expected_upper = upper.copy(); expected_upper[axis] += 7
                np.testing.assert_allclose(tuple(part.bounding_box().max), expected_upper, atol=1e-6)

    def test_nonplanar_stock_is_not_guessed_as_a_box(self):
        self.assertIsNone(_body_bounds(trimesh.creation.icosphere(subdivisions=2)))

    def test_unsupported_stages_require_further_measurement(self):
        simple = {'axis':[0,0,1], 'diameter_mm':4, 'depth_mm':10,
                  'positions_mm':[[0,0,10]], 'count':1}
        self.assertTrue(_supported_feature(simple))
        for field, value in [('counterbore', {'diameter_mm':8}),
                             ('threaded', {'spec':'M4'}), ('source','open-bore'),
                             ('end_treatments',[{'kind':'countersink'}])]:
            self.assertFalse(_supported_feature({**simple, field:value}))

    def test_disconnected_boss_is_not_accepted(self):
        boss = {'axis':[0,0,1], 'depth_mm':5, 'diameter_mm':4,
                'positions_mm':[[0,0,30]], 'count':1}
        spec = MachiningSpec(np.array([-10.,-10.,0.]), np.array([10.,10.,10.]), [], [boss])
        with self.assertRaisesRegex(ValueError, 'connected solid'):
            draw_machining(spec)


class PocketCounterboreTests(unittest.TestCase):
    def test_counterbore_end_is_preserved_on_all_axes(self):
        from stlstep.featurecheck import CADSections
        from shapely.geometry import Point
        for axis in range(3):
            for end in ('low','high'):
                lower = np.array([11.,21.,31.]); upper = lower+20
                high = (lower+upper)/2; high[axis] = upper[axis]
                hole = {'axis':np.eye(3)[axis], 'depth_mm':20, 'diameter_mm':4,
                        'positions_mm':[high.tolist()], 'count':1,
                        'counterbore':{'diameter_mm':8,'depth_mm':3,'end':end,'stage_count':2,'contiguous':True}}
                self.assertTrue(_supported_feature(hole))
                part, _, _ = draw_machining(MachiningSpec(lower,upper,[hole],[]))
                sections = CADSections(part)
                uv = [i for i in range(3) if i != axis]
                point = high[uv].copy(); point[0] += 3
                for side, t in [('low',lower[axis]+1), ('high',upper[axis]-1)]:
                    self.assertEqual(sections.section_region(axis,t).contains(Point(*point)), side != end)

    def test_pocket_floor_is_anchored_and_profile_must_be_supported(self):
        from stlstep.machining import _supported_pocket
        pocket = {'opening_normal':[0,0,-1], 'plane_axes':[[1,0,0],[0,-1,0]],
                  'center_mm':[0,0,3], 'size_mm':[8,6], 'depth_mm':3,'profile_kind':'rectangle'}
        self.assertTrue(_supported_pocket(pocket))
        part, _, _ = draw_machining(MachiningSpec(np.array([-10.,-10.,0]),np.array([10.,10.,10]),[],[],[pocket]))
        self.assertAlmostEqual(part.volume, 4000-8*6*3, places=5)
        self.assertFalse(_supported_pocket({**pocket,'profile_kind':'profiled'}))

    def test_manifold_recognition_generalizes_to_scaled_permuted_stock(self):
        mesh = load_and_repair(str(Path(__file__).parent/'fixtures/manifold.stl')).mesh
        matrix = np.array([[0,0,1],[1,0,0],[0,1,0]],float)
        mesh.vertices = 1.3*(mesh.vertices@matrix.T)+[7.123,-11.456,19.789]
        spec = detect_machining(mesh)
        self.assertIsNotNone(spec)
        self.assertEqual(len(spec.pockets), 1)
        self.assertEqual(sum(h['count'] for h in spec.holes), 5)
        np.testing.assert_allclose(spec.upper-spec.lower,[36.4,78,52],atol=1e-5)
        part, _, _ = draw_machining(spec)
        self.assertEqual(len(part.solids()),1)
        self.assertTrue(part.is_valid)

    def test_ambiguous_or_disconnected_counterbore_is_rejected(self):
        from stlstep.features import _collapse_coaxial, _patternize
        mesh = trimesh.creation.box(extents=[20,20,20])
        axis = np.array([0.,0.,1.])
        def wall(dia, lo, hi):
            return {'dia':dia,'ea':axis*lo,'eb':axis*hi}
        valid = _collapse_coaxial([wall(4,-10,7),wall(8,7,10)],axis,mesh)
        valid.update(count=1,positions_mm=[[0,0,10]])
        self.assertTrue(_supported_feature(valid))
        for walls in ([wall(4,-10,5),wall(8,7,10)],
                      [wall(4,-10,10),wall(8,-10,-7),wall(8,7,10)],
                      [wall(4,-10,10),wall(8,-3,3)]):
            feature = _collapse_coaxial(walls,axis,mesh)
            feature.update(count=1,positions_mm=[[0,0,10]])
            self.assertFalse(_supported_feature(feature))
        low = _collapse_coaxial([wall(8,-10,-7),wall(4,-7,10)],axis,mesh)
        self.assertEqual(len(_patternize([valid,low])),2)

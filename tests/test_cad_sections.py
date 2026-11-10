import unittest
import numpy as np
from build123d import BuildPart, Box, Cylinder, Locations, Mode
from stlstep.featurecheck import CADSections, feature_gap, slice_occupancy
from stlstep.occ_util import shape_to_mesh
import trimesh


class CADSectionTests(unittest.TestCase):
    def test_direct_section_preserves_hole_and_expected_area(self):
        with BuildPart() as target:
            Box(20, 20, 10)
            Cylinder(2, 12, mode=Mode.SUBTRACT)
        region = CADSections(target.part).section_region(2, 0)
        self.assertAlmostEqual(region.area, 400-4*np.pi, delta=.2)
        self.assertEqual(len(region.interiors), 1)

    def test_wrong_hole_still_fails_with_direct_cad_sections(self):
        with BuildPart() as target:
            Box(20, 20, 10)
            Cylinder(2, 12, mode=Mode.SUBTRACT)
        with BuildPart() as wrong:
            Box(20, 20, 10)
            Box(4, 4, 12, mode=Mode.SUBTRACT)
        v, f = shape_to_mesh(target.part.wrapped, .01)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        result = slice_occupancy(mesh, CADSections(wrong.part), n=8)
        self.assertTrue(result['loop_mismatches'])


class CounterboreTests(unittest.TestCase):
    def model(self, counterbores):
        with BuildPart() as part:
            Box(30, 30, 10)
            with Locations((-7,-7), (-7,7), (7,-7), (7,7)):
                Cylinder(1.5,12,mode=Mode.SUBTRACT)
            if counterbores:
                with Locations((-7,-7,4), (-7,7,4), (7,-7,4), (7,7,4)):
                    Cylinder(3,2,mode=Mode.SUBTRACT)
        return part.part

    def analysis(self):
        return {'features': {'holes': [{'diameter_mm':3, 'count':4,
                                       'counterbore':{'diameter_mm':6}}]}}

    def test_both_diameters_checked_without_doubling_same_radius_count(self):
        self.assertEqual(feature_gap(self.analysis(), self.model(True).wrapped), [])

    def test_missing_counterbore_still_fails(self):
        gaps = feature_gap(self.analysis(), self.model(False).wrapped)
        self.assertTrue(any('counterbore' in g['detail'] for g in gaps))

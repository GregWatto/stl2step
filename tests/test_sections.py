"""Oriented sections preserve tangent voids without closing genuine gaps."""
from pathlib import Path
import unittest
import numpy as np
import trimesh
from stlstep.ingest import load_and_repair
from stlstep.sections import material_region, IncompleteSectionError


class OrientedSectionTests(unittest.TestCase):
    def test_box_orientation_on_every_axis_after_translation(self):
        mesh = trimesh.creation.box(extents=[6,8,10])
        mesh.apply_translation([13,-29,101])
        for axis, area in enumerate([80,60,48]):
            region = material_region(mesh, axis, mesh.centroid[axis])
            self.assertAlmostEqual(region.area, area, places=5)

    def test_tangent_counterbores_do_not_fill_manifold_pocket(self):
        mesh = load_and_repair(str(Path(__file__).parent/'fixtures/manifold.stl')).mesh
        self.assertFalse(mesh.is_watertight)
        for z in (24.2,26,27.9):
            region = material_region(mesh, 2, z)
            self.assertAlmostEqual(region.area, 1037.416, delta=.01)
            from shapely.geometry import Point
            self.assertFalse(region.contains(Point(0,0)))
            self.assertFalse(region.contains(Point(24,15)))

    def test_open_boundary_is_rejected(self):
        mesh = trimesh.creation.box()
        # Remove both triangles of one side, leaving an actual gap in this cut.
        mesh.update_faces(mesh.face_normals[:,0] < .9)
        with self.assertRaisesRegex(IncompleteSectionError, 'open intersection'):
            material_region(mesh, 2, 0)

    def test_inconsistent_winding_is_rejected(self):
        mesh = trimesh.creation.box()
        mesh.faces[0] = mesh.faces[0,::-1]
        with self.assertRaisesRegex(IncompleteSectionError, 'winding'):
            material_region(mesh, 2, 0)

    def test_plane_outside_mesh_is_empty(self):
        self.assertIsNone(material_region(trimesh.creation.box(), 0, 10))

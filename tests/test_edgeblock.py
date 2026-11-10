"""Measured edge treatments must survive poses and reject unsupported geometry."""
from pathlib import Path
import unittest
import numpy as np
import trimesh
from build123d import (BuildPart, BuildSketch, Plane, Axis, Box, Circle,
                      RectangleRounded, Mode, Align, fillet, chamfer, loft)
from stlstep.edgeblock import EdgeBlock, detect_edgeblock, draw_edgeblock
from stlstep.featurecheck import candidate_features
from stlstep.ingest import load_and_repair
from stlstep.occ_util import shape_to_mesh


def mesh_of(part):
    vertices, faces = shape_to_mesh(part.wrapped, .005, .1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if not mesh.is_watertight:
        raise AssertionError('test input must be a closed mesh')
    return mesh


class EdgeBlockTests(unittest.TestCase):
    def test_fixture_and_transformed_low_end_chamfer(self):
        rotations = [np.eye(3), np.array([[0,0,-1],[1,0,0],[0,-1,0]])]
        for index, rotation in enumerate(rotations):
            with self.subTest(pose=index):
                mesh = load_and_repair(str(Path(__file__).parent/'fixtures/filletblock.stl')).mesh
                scale = 1 if index == 0 else 1.25
                mesh.vertices = scale*(mesh.vertices@rotation.T)+[11.126,-8.2345,17.3456]
                spec = detect_edgeblock(mesh)
                self.assertIsNotNone(spec)
                self.assertEqual(spec.axis, 2 if index == 0 else 0)
                self.assertEqual(spec.chamfer_end, 'high' if index == 0 else 'low')
                self.assertAlmostEqual(spec.radius, 6*scale, delta=.02)
                self.assertAlmostEqual(spec.chamfer, 3*scale, delta=.02)
                part, script, _ = draw_edgeblock(spec)
                self.assertTrue(part.is_valid)
                self.assertEqual(len(part.solids()), 1)
                self.assertLess(abs(part.volume-mesh.volume)/mesh.volume,.002)
                self.assertIn('fillet(',script)
                self.assertIn('chamfer(',script)
                self.assertNotIn('filletblock',script)

    def test_analytic_rounds_and_bores_on_all_axes_and_ends(self):
        lower = np.array([11.126,-8.2345,17.3456])
        upper = lower + [40,32,24]
        for axis in range(3):
            for end in ('low','high'):
                with self.subTest(axis=axis,end=end):
                    uv = [a for a in range(3) if a != axis]
                    center = .5*(lower[uv]+upper[uv])+[.75,-.5]
                    spec = EdgeBlock(lower,upper,axis,4,1.75,end,[(center,2)])
                    part, _, _ = draw_edgeblock(spec)
                    features = candidate_features(part.wrapped)
                    self.assertEqual(sum(abs(r-4)<1e-6 for r in features['cylinder_radii']),4)
                    self.assertEqual(sum(abs(r-2)<1e-6 for r in features['cylinder_radii']),1)
                    self.assertGreaterEqual(len(features['cone_semi_angles']),4)
                    np.testing.assert_allclose(tuple(part.bounding_box().min),lower,atol=1e-6)
                    np.testing.assert_allclose(tuple(part.bounding_box().max),upper,atol=1e-6)

    def test_continuous_draft_is_not_mistaken_for_end_chamfer(self):
        with BuildPart() as part:
            with BuildSketch(Plane.XY):
                RectangleRounded(40,32,4)
            with BuildSketch(Plane.XY.offset(24)):
                RectangleRounded(36,28,2)
            loft()
        self.assertIsNone(detect_edgeblock(mesh_of(part.part)))

    def test_tapered_bore_is_not_made_cylindrical(self):
        with BuildPart() as part:
            Box(40,32,24,align=(Align.CENTER,Align.CENTER,Align.MIN))
            fillet(part.edges().filter_by(Axis.Z),radius=4)
            chamfer(part.faces().sort_by(Axis.Z)[-1].outer_wire().edges(),length=1.75)
            with BuildSketch(Plane.XY):
                Circle(2)
            with BuildSketch(Plane.XY.offset(24)):
                Circle(3)
            loft(mode=Mode.SUBTRACT)
        self.assertIsNone(detect_edgeblock(mesh_of(part.part)))

    def test_two_treated_ends_are_not_guessed_as_one(self):
        with BuildPart() as part:
            Box(40,32,24,align=(Align.CENTER,Align.CENTER,Align.MIN))
            fillet(part.edges().filter_by(Axis.Z),radius=4)
            faces = part.faces().sort_by(Axis.Z)
            chamfer(list(faces[0].outer_wire().edges())+list(faces[-1].outer_wire().edges()),length=1.5)
        self.assertIsNone(detect_edgeblock(mesh_of(part.part)))

    def test_thin_nonuniform_bevel_is_not_assumed_to_be_45_degrees(self):
        with BuildPart() as part:
            Box(40,32,24,align=(Align.CENTER,Align.CENTER,Align.MIN))
            fillet(part.edges().filter_by(Axis.Z),radius=4)
            chamfer(part.faces().sort_by(Axis.Z)[-1].outer_wire().edges(),
                    length=.25,length2=.5)
        self.assertIsNone(detect_edgeblock(mesh_of(part.part)))

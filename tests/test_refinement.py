"""Refinement tools need measured stationary voids and circular tapers."""
import unittest
import numpy as np
import trimesh
from build123d import BuildPart, Box, Cylinder, Cone, Locations, Plane, Mode, Align
from stlstep.occ_util import shape_to_mesh
from stlstep.compose import exec_script
from stlstep.draw import Prismatic, _section_polys
from stlstep.refinement import residual_cut_script, taper_script


def mesh_of(part):
    v,f=shape_to_mesh(part.wrapped,.01)
    return trimesh.Trimesh(vertices=v,faces=f,process=True)


class ResidualCutTests(unittest.TestCase):
    def test_stationary_blind_void_is_recovered_as_an_analytic_cut(self):
        with BuildPart() as target:
            Box(20,20,20)
            with Locations(Plane(origin=(5,0,0),z_dir=(1,0,0))):
                Cylinder(2,3,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)
        script='from build123d import *\nwith BuildPart() as part:\n    Box(20,20,20)\n'
        base=exec_script(script)
        result,count=residual_cut_script(mesh_of(target.part),base,script)
        self.assertGreater(count,0)
        actual=exec_script(result)
        self.assertAlmostEqual(actual.volume,target.part.volume,delta=.2)
        self.assertIn('Circle(',result)

    def test_varying_void_is_not_cut_as_a_constant_profile(self):
        with BuildPart() as target:
            Box(20,20,20)
            with Locations(Plane(origin=(5,0,0),z_dir=(1,0,0))):
                Cone(2,3,3,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)
        script='from build123d import *\nwith BuildPart() as part:\n    Box(20,20,20)\n'
        result,count=residual_cut_script(mesh_of(target.part),exec_script(script),script)
        self.assertIsNone(result)
        self.assertEqual(count,0)


class TaperTests(unittest.TestCase):
    def test_tapered_ring_preserves_hole_on_every_axis(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                origin=np.array([11.,-7.,23.])
                with BuildPart() as target:
                    with Locations(Plane(origin=tuple(origin),z_dir=tuple(np.eye(3)[axis]))):
                        Cone(9,7,4,align=(Align.CENTER,Align.CENTER,Align.MIN))
                        Cylinder(2,4,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)
                mesh=mesh_of(target.part)
                lo,hi=origin[axis],origin[axis]+4
                spec=Prismatic(axis=axis,levels=[(lo,hi,_section_polys(mesh,axis,(lo+hi)/2))])
                script=taper_script(spec,mesh)
                self.assertIsNotNone(script)
                actual=exec_script(script)
                self.assertTrue(actual.is_valid)
                self.assertAlmostEqual(actual.volume,target.part.volume,delta=2)
                self.assertIn('loft(ruled=True)',script)

    def test_varying_bore_is_not_replaced_with_a_constant_hole(self):
        with BuildPart() as target:
            Cone(9,7,4,align=(Align.CENTER,Align.CENTER,Align.MIN))
            Cone(2,3,4,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)
        mesh=mesh_of(target.part)
        spec=Prismatic(axis=2,levels=[(0.,4.,_section_polys(mesh,2,2.))])
        self.assertIsNone(taper_script(spec,mesh))


class ThinResidualTests(unittest.TestCase):
    def test_thin_measured_band_is_not_lost_to_part_size_cutoff(self):
        with BuildPart() as target:
            Box(20,20,20)
            with Locations(Plane(origin=(5,0,0),z_dir=(1,0,0))):
                Cylinder(2,.1,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)
        script='from build123d import *\nwith BuildPart() as part:\n    Box(20,20,20)\n'
        result,count=residual_cut_script(mesh_of(target.part),exec_script(script),script)
        self.assertGreater(count,0)
        self.assertAlmostEqual(exec_script(result).volume,target.part.volume,delta=.01)

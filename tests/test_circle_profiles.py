import unittest
import numpy as np
from stlstep.draw import fit_loop


class CircleBoundaryTests(unittest.TestCase):
    def test_regular_polygons_keep_their_flat_sides(self):
        for count in (4,8,12):
            a=np.linspace(0,2*np.pi,count,endpoint=False)
            points=8*np.column_stack((np.cos(a),np.sin(a)))
            self.assertFalse(any(s.kind=='circle' for s in fit_loop(points)))

    def test_dense_circular_mesh_boundary_stays_analytic(self):
        a=np.linspace(0,2*np.pi,128,endpoint=False)
        points=[13,-7]+8*np.column_stack((np.cos(a),np.sin(a)))
        fit=fit_loop(points)
        self.assertEqual(len(fit),1)
        self.assertEqual(fit[0].kind,'circle')
        self.assertAlmostEqual(fit[0].radius,8,places=6)


class MeasuredBoreTests(unittest.TestCase):
    def test_vertex_evidence_resolves_section_chord_noise(self):
        import trimesh
        from stlstep.draw import Lofted, Seg, _measured_loft_bore, _constant_loft_hole
        mesh=trimesh.creation.cylinder(radius=6,height=20,sections=96)
        holes=[(t,[Seg('circle',(6,0),(6,0),center=(0,0),radius=r)])
               for t,r in zip([-10,0,10],[6,5.999,6])]
        actual=_measured_loft_bore(Lofted(axis=2,stations=[]),holes,mesh)
        self.assertTrue(_constant_loft_hole(actual,[0,1,2]))
        self.assertAlmostEqual(actual[0][1][0].radius,6,places=6)

    def test_real_small_taper_is_not_snapped_to_a_cylinder(self):
        import trimesh
        from stlstep.draw import Lofted, Seg, _measured_loft_bore, _constant_loft_hole
        # Even a 0.001 mm taper exceeds the unchanged 1e-4 wall-evidence bound.
        mesh=trimesh.creation.cylinder(radius=6,height=20,sections=96)
        mesh.vertices[mesh.vertices[:,2]>0,:2]*=6.001/6
        holes=[(t,[Seg('circle',(6,0),(6,0),center=(0,0),radius=r)])
               for t,r in zip([-10,0,10],[6,6.0005,6.001])]
        actual=_measured_loft_bore(Lofted(axis=2,stations=[]),holes,mesh)
        self.assertFalse(_constant_loft_hole(actual,[0,1,2]))

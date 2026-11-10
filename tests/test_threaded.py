"""Identify complete helical threads without masking unrelated geometry."""
from pathlib import Path
import unittest
import numpy as np
from build123d import BuildPart, Cylinder, Locations, Align, Mode, Plane
from stlstep.ingest import load_and_repair
from stlstep.threaded import detect_threaded_ring, draw_threaded_ring
from stlstep.featurecheck import feature_gap, mask_points_in_bands, thread_bands


class ThreadIdentificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = load_and_repair(str(Path(__file__).parent/'fixtures/threadring.stl')).mesh

    def modified_wall(self, function):
        mesh = self.mesh.copy()
        vertices = mesh.vertices.copy()
        radial = np.linalg.norm(vertices[:,:2],axis=1)
        inner = radial<6
        theta = np.arctan2(vertices[inner,1],vertices[inner,0])
        radius = function(theta,vertices[inner,2],radial[inner])
        vertices[inner,:2] *= (radius/radial[inner])[:,None]
        mesh.vertices = vertices
        return mesh

    def test_fixture_records_pitch_hand_standard_and_precise_band(self):
        spec = detect_threaded_ring(self.mesh)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.evidence['spec'],'M10x1.5')
        self.assertEqual(spec.evidence['hand'],'right')
        self.assertAlmostEqual(spec.evidence['measured_pitch_mm'],1.5,places=3)
        self.assertGreater(spec.evidence['triangular_flank_r2'],.999)
        np.testing.assert_allclose(spec.bands()[0]['r'],[4.138,5.05],atol=1e-5)
        self.assertEqual(spec.bands()[0]['t'],[-12,0])
        part, script, _ = draw_threaded_ring(spec)
        self.assertTrue(part.is_valid)
        self.assertEqual(feature_gap({'features':spec.features()},part.wrapped),[])
        self.assertIn('M10x1.5',script)
        self.assertIn('Nominal',script)
        self.assertNotIn('threadring.stl',script)

    def test_translations_and_proper_axis_rotations_keep_right_hand(self):
        rotations = [np.eye(3),np.array([[0,0,1],[1,0,0],[0,1,0]]),
                     np.array([[0,1,0],[0,0,1],[1,0,0]]),np.diag([-1,1,-1])]
        for rotation in rotations:
            with self.subTest(rotation=rotation.tolist()):
                mesh=self.mesh.copy()
                transform=np.eye(4); transform[:3,:3]=rotation
                transform[:3,3]=[17.123,-9.456,28.789]
                mesh.apply_transform(transform)
                spec=detect_threaded_ring(mesh)
                self.assertIsNotNone(spec)
                self.assertEqual(spec.evidence['hand'],'right')
                part, _, _=draw_threaded_ring(spec)
                self.assertEqual(feature_gap({'features':spec.features()},part.wrapped),[])

    def test_left_hand_is_measured_after_reflection(self):
        mesh=self.mesh.copy()
        transform=np.diag([-1.,1.,1.,1.]); mesh.apply_transform(transform)
        spec=detect_threaded_ring(mesh)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.evidence['hand'],'left')

    def test_another_standard_size_is_identified_from_geometry(self):
        # M8x1, with a different stock diameter and axial span.
        def wall(theta,z,r):
            phase=(theta/(2*np.pi)-.75*z)%1
            return 4-(4-(8-1.0825)/2)*2*abs(phase-.5)
        mesh=self.modified_wall(wall)
        vertices=mesh.vertices.copy()
        outer=np.linalg.norm(vertices[:,:2],axis=1)>6
        vertices[outer,:2]*=.8
        vertices[:,2]*=.75
        mesh.vertices=vertices
        spec=detect_threaded_ring(mesh)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.evidence['spec'],'M8x1')
        self.assertAlmostEqual(spec.outer_radius,6.4,places=5)

    def test_annular_grooves_are_not_threads(self):
        mesh=self.modified_wall(lambda th,z,r: 5-.812*2*abs((z/1.5)%1-.5))
        self.assertIsNone(detect_threaded_ring(mesh))

    def test_multi_start_and_sinusoidal_relief_are_not_this_thread_model(self):
        functions=[lambda th,z,r: 5-.812*2*abs((2*th/(2*np.pi)-z/1.5)%1-.5),
                   lambda th,z,r: 4.594+.406*np.cos(th-2*np.pi*z/1.5)]
        for fn in functions:
            self.assertIsNone(detect_threaded_ring(self.modified_wall(fn)))

    def test_plain_partial_tapered_and_unknown_thread_dimensions_reject(self):
        functions=[lambda th,z,r: np.full_like(r,4.188),
                   lambda th,z,r: np.where(z>8,4.188,r),
                   lambda th,z,r: r+.1*z/12,
                   lambda th,z,r: r+.2]
        for fn in functions:
            self.assertIsNone(detect_threaded_ring(self.modified_wall(fn)))

    def test_damaged_stock_and_open_source_reject(self):
        mesh=self.mesh.copy(); vertices=mesh.vertices.copy()
        index=np.argmax(np.linalg.norm(vertices[:,:2],axis=1))
        vertices[index,:2]*=.8; mesh.vertices=vertices
        self.assertIsNone(detect_threaded_ring(mesh))
        mesh=self.mesh.copy(); mesh.update_faces(np.arange(len(mesh.faces))!=0)
        self.assertIsNone(detect_threaded_ring(mesh))

    def test_band_does_not_extend_beyond_measured_thread_or_into_stock(self):
        band=detect_threaded_ring(self.mesh).bands()
        points=np.array([[4.6,0,6],[7,0,6],[0,0,6],[4.6,0,-.1],[4.6,0,12.1]])
        np.testing.assert_array_equal(mask_points_in_bands(points,band),[False,True,True,True,True])

    def test_invalid_measured_band_fails_instead_of_widening(self):
        features=detect_threaded_ring(self.mesh).features()
        features['holes'][0]['threaded']['validation_band']['r']=[np.nan,8]
        with self.assertRaises(ValueError):
            thread_bands({'features':features})

    def test_mask_cannot_hide_wrong_nominal_bore_geometry(self):
        features=detect_threaded_ring(self.mesh).features()
        radius=features['holes'][0]['diameter_mm']/2
        for bore,offset,height in [(4.6,0,12),(radius,.02,12),(radius,0,10)]:
            with self.subTest(bore=bore,offset=offset,height=height):
                with BuildPart() as part:
                    Cylinder(8,12,align=(Align.CENTER,Align.CENTER,Align.MIN))
                    with Locations((offset,0,0)):
                        Cylinder(bore,height,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)
                gaps=feature_gap({'features':features},part.part.wrapped)
                self.assertTrue(any(g['kind']=='nominal_thread' for g in gaps))

    def test_relief_mask_does_not_hide_wrong_outer_stock(self):
        from stlstep.draw import _part_deviation
        spec=detect_threaded_ring(self.mesh)
        with BuildPart() as wrong:
            Cylinder(8.4,12,align=(Align.CENTER,Align.CENTER,Align.MIN))
            Cylinder(spec.minor_diameter/2,12,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)
        self.assertGreater(_part_deviation(wrong.part,self.mesh,exclude_bands=spec.bands()),.2)


    def test_mask_cannot_hide_a_trimmed_hole_in_nominal_cylinder_wall(self):
        spec=detect_threaded_ring(self.mesh)
        with BuildPart() as wrong:
            Cylinder(8,12,align=(Align.CENTER,Align.CENTER,Align.MIN))
            Cylinder(spec.minor_diameter/2,12,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)
            with Locations(Plane(origin=(4,0,6),z_dir=(1,0,0))):
                Cylinder(.3,.8,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)
        gaps=feature_gap({'features':spec.features()},wrong.part.wrapped)
        self.assertTrue(any(g['kind']=='nominal_thread' for g in gaps))


    def test_phase_fit_subset_cannot_hide_damage_in_remaining_vertices(self):
        from stlstep.threaded import _helical_phase
        theta=np.tile(np.linspace(-np.pi,np.pi,96,endpoint=False),160)
        z=np.repeat(np.linspace(0,12,160),96)
        radius=5-.812*2*abs((theta/(2*np.pi)-z/1.5)%1-.5)
        self.assertIsNotNone(_helical_phase(theta,z,radius,1.452))
        # Preserve a clean fitting subset but corrupt the rest of the measured wall.
        sampled=np.linspace(0,len(radius)-1,4096).astype(int)
        unsampled=np.ones(len(radius),dtype=bool); unsampled[sampled]=False
        radius[unsampled]=9.188-radius[unsampled]
        self.assertIsNone(_helical_phase(theta,z,radius,1.452))

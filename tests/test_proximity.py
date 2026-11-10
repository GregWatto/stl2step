import unittest
import numpy as np
import trimesh
from stlstep.proximity import closest_point


class BatchedProximityTests(unittest.TestCase):
    def test_matches_exact_unbatched_queries(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        points = np.random.default_rng(123).normal(size=(133, 3))
        expected = trimesh.proximity.closest_point(mesh, points)
        for batch_size in [1, 17, 512]:
            actual = closest_point(mesh, points, batch_size=batch_size)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_empty_and_invalid_inputs(self):
        mesh = trimesh.creation.box()
        nearest, distances, faces = closest_point(mesh, [])
        self.assertEqual(nearest.shape, (0, 3))
        self.assertEqual(len(distances), 0)
        self.assertEqual(len(faces), 0)
        with self.assertRaises(ValueError):
            closest_point(mesh, [[0, 0, 0]], batch_size=0)

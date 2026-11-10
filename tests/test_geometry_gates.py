"""Known wrong solids must fail the existing occupancy/opening geometry gates."""
from pathlib import Path
import unittest
import trimesh
from build123d import BuildPart, Box, Cylinder, Mode
from stlstep import featurecheck, occ_util
from stlstep.ingest import load_and_repair


def mesh(part):
    vertices, faces = occ_util.shape_to_mesh(part.wrapped, 0.05)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class GeometricGateTests(unittest.TestCase):
    def test_filling_open_channel_is_rejected(self):
        target = load_and_repair(str(Path(__file__).parent / 'fixtures/channel.stl')).mesh
        filled = trimesh.creation.box(extents=[100, 40, 30])
        filled.apply_translation([50, 0, 15])
        occupancy = featurecheck.slice_occupancy(target, filled)
        self.assertGreater(occupancy['extra_frac'], 0.03)

    def test_square_substitute_for_round_hole_is_rejected(self):
        with BuildPart() as target:
            Box(20, 20, 10)
            Cylinder(2, 12, mode=Mode.SUBTRACT)
        with BuildPart() as wrong:
            Box(20, 20, 10)
            Box(4, 4, 12, mode=Mode.SUBTRACT)
        occupancy = featurecheck.slice_occupancy(mesh(target.part), mesh(wrong.part))
        self.assertTrue(occupancy['loop_mismatches'])
        # The overall material difference is tiny; the per-opening gate matters.
        self.assertLess(occupancy['missing_frac'], 0.03)
        self.assertLess(occupancy['extra_frac'], 0.03)


if __name__ == '__main__':
    unittest.main()

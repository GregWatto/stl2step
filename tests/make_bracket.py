"""Generate a controlled 2.5D prismatic test part.

A plate + cylindrical boss + blind rectangular pocket + four through-holes.
All features are axis-aligned prisms along Z, so the part is exactly a
prismatic stack — the ground truth for the prismatic-stack recognizer.

Known dimensions (mm):
  base plate : 80 (X) x 50 (Y) x 6 (Z), Z in [0, 6]
  boss       : cylinder r=8, Z in [6, 12], centered at (+25, 0)
  pocket     : 24 x 24 blind pocket, floor at Z=2, open at Z=6, at (-20, 0)
  holes      : 4 x r=3 through-holes at (+/-32, +/-19), Z in [0, 6]
"""
from build123d import (BuildPart, BuildSketch, BuildLine, Plane, Locations,
                        Box, Cylinder, Rectangle, Circle, Align, Mode, extrude,
                        export_stl)

with BuildPart() as p:
    # base plate, bottom face on Z=0
    with BuildSketch(Plane.XY):
        Rectangle(80, 50)
    extrude(amount=6)
    # cylindrical boss on top, +25 in X
    with BuildSketch(Plane.XY.offset(6)):
        with Locations((25, 0)):
            Circle(8)
    extrude(amount=6)
    # blind pocket: cut 24x24 from the top down to Z=2 (depth 4), at -20 in X
    with BuildSketch(Plane.XY.offset(6)):
        with Locations((-20, 0)):
            Rectangle(24, 24)
    extrude(amount=-4, mode=Mode.SUBTRACT)
    # four through holes
    with BuildSketch(Plane.XY):
        with Locations((32, 19), (-32, 19), (32, -19), (-32, -19)):
            Circle(3)
    extrude(amount=6, mode=Mode.SUBTRACT)

out = "/Users/nickp/Documents/Projects/stl-step-v2/tests/fixtures/bracket.stl"
export_stl(p.part, out, tolerance=0.01, angular_tolerance=0.1)
print("wrote", out)
print("volume mm^3:", round(p.part.volume, 3))
print("bounds:", p.part.bounding_box())

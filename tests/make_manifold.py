"""Generate a moderately-complex, cleanly-composable test part — a manifold block.

Multi-axis features that defeat the deterministic recognizers (a cross-bore along
X breaks the Z-constant-section prismatic test; counterbores are two-diameter),
but all prismatic/cylindrical — the LLM draw path's sweet spot. Expected to reach
a clean pass via compose + fit.

Known dimensions (mm):
  body        : 60 (X) x 40 (Y) x 28 (Z), Z in [0, 28]
  top pocket  : 44 x 24 blind pocket, floor at Z=18 (10 deep)
  cross-bore  : Ø14 through along X, at (y=0, z=9)
  4 mounting  : Ø6 through-holes along Z at (+/-24, +/-15), each with a
                Ø10 counterbore 4 deep from the top face
"""
from build123d import (BuildPart, BuildSketch, Plane, Locations, Box, Rectangle,
                        Circle, Align, Mode, extrude, export_stl)

with BuildPart() as p:
    Box(60, 40, 28, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # blind pocket in the top, floor at Z=18
    with BuildSketch(Plane.XY.offset(28)):
        Rectangle(44, 24)
    extrude(amount=-10, mode=Mode.SUBTRACT)
    # cross-bore Ø14 through along X, at (y=0, z=9)
    with BuildSketch(Plane.YZ):        # normal +X; sketch coords (Y, Z)
        with Locations((0, 9)):
            Circle(7)
    extrude(amount=32, both=True, mode=Mode.SUBTRACT)
    # 4 mounting holes Ø6 through Z
    with BuildSketch(Plane.XY):
        with Locations((24, 15), (-24, 15), (24, -15), (-24, -15)):
            Circle(3)
    extrude(amount=28, mode=Mode.SUBTRACT)
    # Ø10 counterbores, 4 deep from the top
    with BuildSketch(Plane.XY.offset(28)):
        with Locations((24, 15), (-24, 15), (24, -15), (-24, -15)):
            Circle(5)
    extrude(amount=-4, mode=Mode.SUBTRACT)

out = "/Users/nickp/Documents/Projects/stl-step-v2/tests/fixtures/manifold.stl"
export_stl(p.part, out, tolerance=0.008, angular_tolerance=0.1)
print("wrote", out)
print("volume mm^3:", round(p.part.volume, 3))
print("bounds:", p.part.bounding_box())

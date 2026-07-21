"""Generate a part that defeats every deterministic recognizer, to exercise the
LLM composition path.

Not a single extrusion (a cross hole runs perpendicular to the layering axis),
not a revolve (not rotationally symmetric), not a prismatic stack (the cross hole
makes the Z cross-section non-constant). But it is trivially composable from CAD
features — exactly what the LLM should draw.

Known dimensions (mm):
  base block : 50 (X) x 40 (Y) x 20 (Z), Z in [0, 20], centered in XY
  boss       : cylinder r=9 on top, Z in [20, 30], at (0, 0)
  cross hole : Ø12 through the block along Y, axis at (x=0, z=10)
  corner holes: 4 x Ø5 through the block along Z at (+/-19, +/-14)
"""
from build123d import (BuildPart, BuildSketch, Plane, Locations, Box, Cylinder,
                        Circle, Align, Mode, Axis, extrude, export_stl)

with BuildPart() as p:
    Box(50, 40, 20, align=(Align.CENTER, Align.CENTER, Align.MIN))   # Z in [0,20]
    # boss on top
    with BuildSketch(Plane.XY.offset(20)):
        Circle(9)
    extrude(amount=10)
    # cross hole through the block along Y, centered, at z=10
    with BuildSketch(Plane.XZ):        # XZ plane's normal is Y
        with Locations((0, 10)):       # (x, z) in the sketch
            Circle(6)
    extrude(amount=30, both=True, mode=Mode.SUBTRACT)
    # four corner through-holes along Z
    with BuildSketch(Plane.XY):
        with Locations((19, 14), (-19, 14), (19, -14), (-19, -14)):
            Circle(2.5)
    extrude(amount=20, mode=Mode.SUBTRACT)

out = "/Users/nickp/Documents/Projects/stl-step-v2/tests/fixtures/crossblock.stl"
export_stl(p.part, out, tolerance=0.008, angular_tolerance=0.1)
print("wrote", out)
print("volume mm^3:", round(p.part.volume, 3))
print("bounds:", p.part.bounding_box())

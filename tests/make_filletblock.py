"""Generate a part with fillets AND chamfers, to validate edge-treatment support.

A rounded-corner block with a chamfered top and a central bore. The edge
treatments are the point: 4 vertical fillets (r=6) and a top-perimeter chamfer
(3mm). Used to check that the LLM can compose fillet()/chamfer() and that the
parameter fitter can snap their radii.

Known dimensions (mm):
  block   : 50 (X) x 40 (Y) x 20 (Z), Z in [0, 20]
  fillets : 4 vertical edges, radius 6
  chamfer : top perimeter, 3 mm
  bore    : Ø12 through Z, centered
"""
from build123d import (BuildPart, BuildSketch, Plane, Box, Circle, Align, Axis,
                        Mode, fillet, chamfer, extrude, export_stl)

with BuildPart() as p:
    Box(50, 40, 20, align=(Align.CENTER, Align.CENTER, Align.MIN))
    fillet(p.edges().filter_by(Axis.Z), radius=6)                 # 4 vertical edges
    chamfer(p.faces().sort_by(Axis.Z)[-1].edges(), length=3)      # top perimeter
    with BuildSketch(Plane.XY):
        Circle(6)
    extrude(amount=20, mode=Mode.SUBTRACT)

out = "/Users/nickp/Documents/Projects/stl-step-v2/tests/fixtures/filletblock.stl"
export_stl(p.part, out, tolerance=0.005, angular_tolerance=0.08)
print("wrote", out)
print("volume mm^3:", round(p.part.volume, 3))
print("bounds:", p.part.bounding_box())

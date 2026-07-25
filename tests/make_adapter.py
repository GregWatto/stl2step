"""Generate a square-to-round transition adapter — a pure LOFT part.

The cross-section changes continuously along Z: a 50x50 square at the base
morphs to a Ø30 circle at the top. No single extrude/revolve/prismatic-stack can
reproduce it — the section is neither constant (extrude) nor rotationally
symmetric (revolve) nor piecewise-constant (prismatic). It is the canonical test
for multi-section / loft reconstruction. A Ø12 through-bore keeps a hole in play.

Known dimensions (mm):
  base square : 50 x 50 at Z=0
  top circle  : Ø30 at Z=60
  height      : 60 (Z in [0, 60])
  bore        : Ø12 through along Z, centred
"""
from build123d import (BuildPart, BuildSketch, Plane, Locations, Rectangle,
                       Circle, Align, Mode, loft, extrude, export_stl)

with BuildPart() as p:
    with BuildSketch(Plane.XY):
        Rectangle(50, 50)
    with BuildSketch(Plane.XY.offset(60)):
        Circle(15)
    loft()
    # Ø12 through-bore along Z
    with BuildSketch(Plane.XY.offset(-1)):
        Circle(6)
    extrude(amount=62, mode=Mode.SUBTRACT)

out = "/Users/nickp/Documents/Projects/stl-step-v2/tests/fixtures/adapter.stl"
export_stl(p.part, out)
print(f"wrote {out}  volume={p.part.volume:.1f} mm^3")

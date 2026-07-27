"""Generate an ASYMMETRIC y-axis prismatic part — the Plane.XZ sign sentinel.

build123d's Plane.XZ has normal (0,-1,0): offset(+d) lands at world y=-d and a
positive extrude goes toward -Y. Every emitter must negate offsets/amounts for
the y axis (draw._PSIGN). The channel fixture never caught a violation because
it is symmetric about y=0 — a mirrored rebuild matches it exactly. This part is
asymmetric (y in [2,14], boss off-centre), so any sign regression mirrors it to
y in [-14,-2] and fails loudly instead of silently passing.

Known: base slab 40x30, y[2,6]; boss 10x8 at (x,z)=(25,18), y[6,14].
"""
from build123d import *

with BuildPart() as p:
    with BuildSketch(Plane.XZ.offset(-2)):     # sketch lands at world y=+2
        Rectangle(40, 30, align=(Align.MIN, Align.MIN))
    extrude(amount=-4)                          # -amount -> +Y: y 2..6
    with BuildSketch(Plane.XZ.offset(-6)):
        with Locations((25, 18)):
            Rectangle(10, 8)
    extrude(amount=-8)                          # y 6..14

out = "/Users/nickp/Documents/Projects/stl-step-v2/tests/fixtures/ytower.stl"
export_stl(p.part, out)
print(f"wrote {out}  bounds={p.part.bounding_box()}")

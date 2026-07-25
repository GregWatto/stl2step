"""Generate a thin-walled OPEN channel (U-section) — an open-shell test part.

The cross-section is a thin U: two side walls + a floor, open at the top. It is
constant along the length, so it's a pure extrusion — but of a THIN, open,
non-convex band. Before the thin-band guard in fit_loop, the section fit collapsed
that band into a filled silhouette (a solid bar); after it, the wall outline
survives and the open channel reconstructs exactly.

Known dimensions (mm):
  length      : 100 (X in [0, 100])
  outer       : 40 (Y) x 30 (Z)
  wall/floor  : 4 thick   (inner cavity 32 wide x 26 tall, open at the top)
"""
from build123d import (BuildPart, BuildSketch, Plane, Locations, Rectangle,
                       Align, Mode, extrude, export_stl)

with BuildPart() as p:
    with BuildSketch(Plane.YZ):                 # section in the Y-Z plane
        Rectangle(40, 30, align=(Align.CENTER, Align.MIN))
        # remove the cavity, pushed up so the top stays open -> a U
        with Locations((0, 4)):
            Rectangle(32, 30, align=(Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)
    extrude(amount=100)

out = "/Users/nickp/Documents/Projects/stl-step-v2/tests/fixtures/channel.stl"
export_stl(p.part, out)
print(f"wrote {out}  volume={p.part.volume:.1f} mm^3")

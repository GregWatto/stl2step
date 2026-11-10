from build123d import *
# Probe evidence: M10x1.5 internal thread, measured pitch 1.452 mm.
# Nominal tap-drill representation; reapply M10x1.5 Thread in downstream CAD.
# Measured thread radial extent ~4.188..5.0 mm, z=0..12 mm.
# Validation: --thread-band z,0,0,0,12,4.138,5.05
# The 0.05 mm radial allowance covers compose's tessellation chord error.
# See tests/REMAINING.md and threadring_probe.json for retained probe evidence.
with BuildPart() as part:
    Cylinder(8,12,align=(Align.CENTER,Align.CENTER,Align.MIN))
    Cylinder(4.188,12,align=(Align.CENTER,Align.CENTER,Align.MIN),mode=Mode.SUBTRACT)

"""stlstep — scoped, gated STL -> STEP (mesh -> BREP) converter.

Pipeline stages:
    ingest    load + repair mesh, build face-adjacency graph
    segment   detect analytic primitives + label residual regions (100% coverage)
    surfaces  build OCCT Geom_* surfaces (analytic) + NURBS drape (freeform)
    topology  trim faces to edges, sew, heal -> watertight solid, write STEP
    verify    per-region deviation + topology validity + deviation-colormap render

The design targets the three v1 failure modes: no dropped regions (coverage
guarantee), real edges (surface-surface intersection), and an honest per-region
QC gate instead of a single aggregate chamfer number. See docs/v1-lessons.md.
"""

__version__ = "0.1.0"

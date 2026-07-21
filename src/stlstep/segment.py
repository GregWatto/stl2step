"""Stage 1 — segment: partition *every* face into a coherent region.

This is where v2 diverges from v1. v1 detected primitives and then dropped or
mangled whatever was left over, so missed features became slits/pie-wedges. Here
the detector's output is only a starting point: we then

  1. split each analytic primitive's faces into edge-connected patches (two
     coplanar-but-separate faces must become two regions, not one), and
  2. cluster the leftover (unassigned) faces into connected freeform patches.

Result: a list of Regions whose face sets partition the whole mesh (100%
coverage by construction). Analytic regions carry primitive params; freeform
regions are draped with NURBS downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import trimesh

from ._detector import detect_primitives
from .ingest import connected_components


@dataclass
class Region:
    id: int
    kind: str                 # "plane" | "cylinder" | "sphere" | "torus" | "freeform"
    face_ids: list[int]
    params: dict = field(default_factory=dict)   # primitive params; empty for freeform
    residual_mm_mean: float | None = None
    residual_mm_p95: float | None = None

    @property
    def is_analytic(self) -> bool:
        return self.kind != "freeform"


@dataclass
class Segmentation:
    regions: list[Region]
    graph: dict               # raw detector output, kept for reporting/debug
    face_count: int

    def coverage_pct(self) -> float:
        covered = sum(len(r.face_ids) for r in self.regions)
        return 100.0 * covered / self.face_count if self.face_count else 0.0

    def region_of_face(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for r in self.regions:
            for f in r.face_ids:
                out[f] = r.id
        return out


def segment(mesh: trimesh.Trimesh, neighbors: list[list[int]],
            *, detector_kwargs: dict | None = None) -> Segmentation:
    """Detect primitives, then split into fully-covering connected regions."""
    graph = detect_primitives(mesh, **(detector_kwargs or {}))
    total_faces = int(len(mesh.faces))

    regions: list[Region] = []
    claimed: set[int] = set()
    next_id = 0

    # Analytic regions, split into connected patches per primitive.
    for prim in graph["primitives"]:
        faces = [int(f) for f in prim["face_ids"]]
        if not faces:
            continue
        for comp in connected_components(faces, neighbors):
            regions.append(Region(
                id=next_id,
                kind=prim["type"],
                face_ids=comp,
                params=prim["params"],
                residual_mm_mean=prim.get("residual_mm_mean"),
                residual_mm_p95=prim.get("residual_mm_p95"),
            ))
            next_id += 1
            claimed.update(comp)

    # Everything not claimed by a primitive -> freeform regions. This is the
    # coverage guarantee: leftover faces are clustered, never discarded.
    leftover = [f for f in range(total_faces) if f not in claimed]
    for comp in connected_components(leftover, neighbors):
        regions.append(Region(
            id=next_id, kind="freeform", face_ids=comp,
        ))
        next_id += 1

    return Segmentation(regions=regions, graph=graph, face_count=total_faces)

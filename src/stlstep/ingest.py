"""Stage 0 — ingest: load, repair, and expose the mesh face-adjacency graph.

Repair logic is lifted verbatim from v1 (`_mesh_repair.repair_mesh`) — it was
never the problem. What we add here is the connectivity scaffolding the rest of
the pipeline needs: a face-adjacency graph and connected-component labelling
over arbitrary face subsets, used by `segment` to cluster the residual
(non-primitive) faces into freeform regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ._mesh_repair import repair_mesh


@dataclass
class Ingested:
    mesh: trimesh.Trimesh
    report: dict
    # neighbors[f] = list of face ids sharing an edge with face f
    neighbors: list[list[int]]


def load_and_repair(stl_path: str) -> Ingested:
    """Load an STL, run the v1 repair, and precompute face adjacency."""
    mesh, report = repair_mesh(stl_path)
    neighbors = build_face_neighbors(mesh)
    return Ingested(mesh=mesh, report=report, neighbors=neighbors)


def build_face_neighbors(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Adjacency list over faces (edge-sharing). Mirrors the structure the v1
    grower built internally, exposed here for reuse by segmentation."""
    n_faces = int(len(mesh.faces))
    neighbors: list[list[int]] = [[] for _ in range(n_faces)]
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    for a, b in adjacency:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
    return neighbors


def connected_components(face_ids, neighbors: list[list[int]]) -> list[list[int]]:
    """Partition `face_ids` into edge-connected components, restricting the
    walk to the subset itself. Used to split leftover faces into coherent
    freeform patches (one NURBS surface per patch)."""
    subset = set(int(f) for f in face_ids)
    seen: set[int] = set()
    out: list[list[int]] = []
    for start in subset:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[int] = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for nb in neighbors[f]:
                if nb in subset and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        out.append(sorted(comp))
    return out

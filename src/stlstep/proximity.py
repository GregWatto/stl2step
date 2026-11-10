"""Exact trimesh proximity queries with bounded intermediate query batches."""
from __future__ import annotations
import numpy as np
import trimesh


def closest_point(mesh, points, *, batch_size=512):
    """Same points and exact query as trimesh; only allocation size is bounded.

    Large planar faces can create huge flattened candidate-triangle arrays for
    a 20,000-point query. Splitting the query preserves all samples and gates.
    """
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        points = points.reshape((0, 3))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    nearest = np.empty_like(points)
    distances = np.empty(len(points), dtype=float)
    faces = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), batch_size):
        stop = start + batch_size
        nearest[start:stop], distances[start:stop], faces[start:stop] = (
            trimesh.proximity.closest_point(mesh, points[start:stop]))
    return nearest, distances, faces

"""Low-level OpenCASCADE (OCP) helpers.

Kept deliberately small and dependency-light so the geometry stages read
cleanly. Idioms for gp_Ax3 construction, analytic surface builders, sewing,
STEP I/O, solid validity, and re-tessellation are consolidated here. The
analytic-surface builders mirror v1's `_surface_from_primitive` but return bare
`Geom_Surface` handles so the topology stage owns face construction.
"""

from __future__ import annotations

import numpy as np

from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Dir, gp_Ax2, gp_Ax3, gp_Vec
from OCP.Geom import (
    Geom_Plane,
    Geom_CylindricalSurface,
    Geom_ConicalSurface,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
)
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeSolid,
)
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.GCE2d import GCE2d_MakeSegment
from OCP.BRepLib import BRepLib
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Shell, TopoDS_Solid
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone


# ----------------------------------------------------------------------------
# Geometry primitives.
# ----------------------------------------------------------------------------

def _pnt(xyz) -> gp_Pnt:
    return gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _dir(xyz) -> gp_Dir:
    v = np.asarray(xyz, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        v = np.array([0.0, 0.0, 1.0])
    else:
        v = v / n
    return gp_Dir(float(v[0]), float(v[1]), float(v[2]))


def ax3_from_axis(origin, axis) -> gp_Ax3:
    """Right-handed frame with Z = axis through `origin`. OCCT picks a valid X."""
    return gp_Ax3(_pnt(origin), _dir(axis))


def surface_from_region(kind: str, params: dict):
    """Build the unbounded Geom_Surface for an analytic region, or None."""
    if kind == "plane":
        n = params["normal"]
        # A point on the plane: -offset * normal (offset = d/|n|, plane n.x+d=0).
        pt = (-float(params["offset_mm"]) * np.asarray(n, dtype=float)).tolist()
        return Geom_Plane(_pnt(pt), _dir(n))
    if kind == "cylinder":
        ax3 = ax3_from_axis(params["center_mm"], params["axis"])
        return Geom_CylindricalSurface(ax3, float(params["radius_mm"]))
    if kind == "cone":
        ax3 = ax3_from_axis(params["ref_point_mm"], params["axis"])
        return Geom_ConicalSurface(ax3, float(params["semi_angle_rad"]),
                                   float(params["ref_radius_mm"]))
    if kind == "sphere":
        ax3 = ax3_from_axis(params["center_mm"], [0.0, 0.0, 1.0])
        return Geom_SphericalSurface(ax3, float(params["radius_mm"]))
    if kind == "torus":
        ax3 = ax3_from_axis(params["center_mm"], params["axis"])
        return Geom_ToroidalSurface(ax3, float(params["major_radius_mm"]),
                                    float(params["minor_radius_mm"]))
    return None


# ----------------------------------------------------------------------------
# Faces from mesh triangles (baseline faceted topology).
# ----------------------------------------------------------------------------

def triangle_face(p0, p1, p2):
    """A planar BRep face for one mesh triangle. Returns TopoDS_Face or None."""
    poly = BRepBuilderAPI_MakePolygon(_pnt(p0), _pnt(p1), _pnt(p2), True)
    if not poly.IsDone():
        return None
    mk = BRepBuilderAPI_MakeFace(poly.Wire(), True)
    if not mk.IsDone():
        return None
    return mk.Face()


# Half-period unwrap so a triangle straddling a surface's parametric seam
# (u wraps 0..2pi on cylinders/spheres/tori) stays contiguous in uv.
_PI = np.pi


def surface_periods(kind: str):
    """(u_period, v_period); 0.0 = not periodic on that axis."""
    if kind == "cylinder":
        return 2 * _PI, 0.0
    if kind == "cone":
        return 2 * _PI, 0.0
    if kind == "sphere":
        return 2 * _PI, 0.0
    if kind == "torus":
        return 2 * _PI, 2 * _PI
    return 0.0, 0.0


def triangle_face_on_surface(surface, sa: "ShapeAnalysis_Surface", pts3,
                             u_period: float, v_period: float,
                             proj_tol: float = 1e-2):
    """Build a BRep face for one mesh triangle whose underlying surface is the
    shared analytic `surface`. Vertices are projected to (u,v); seam-straddling
    triangles are unwrapped. All triangles of a region share `surface`, so
    UnifySameDomain later merges them into a single analytic face."""
    uvs = []
    for p in pts3:
        q = sa.ValueOfUV(gp_Pnt(float(p[0]), float(p[1]), float(p[2])),
                         proj_tol)
        uvs.append([q.X(), q.Y()])
    u0, v0 = uvs[0]
    for k in (1, 2):
        if u_period:
            while uvs[k][0] - u0 > u_period / 2: uvs[k][0] -= u_period
            while uvs[k][0] - u0 < -u_period / 2: uvs[k][0] += u_period
        if v_period:
            while uvs[k][1] - v0 > v_period / 2: uvs[k][1] -= v_period
            while uvs[k][1] - v0 < -v_period / 2: uvs[k][1] += v_period
    p2 = [gp_Pnt2d(u, v) for u, v in uvs]
    try:
        edges = []
        for i in range(3):
            seg = GCE2d_MakeSegment(p2[i], p2[(i + 1) % 3])
            if not seg.IsDone():
                return None
            e = BRepBuilderAPI_MakeEdge(seg.Value(), surface)
            if not e.IsDone():
                return None
            edges.append(e.Edge())
        w = BRepBuilderAPI_MakeWire(edges[0], edges[1], edges[2])
        if not w.IsDone():
            return None
        mk = BRepBuilderAPI_MakeFace(surface, w.Wire(), True)
        if not mk.IsDone():
            return None
        face = mk.Face()
        BRepLib.BuildCurves3d_s(face)
        return face
    except Exception:  # noqa: BLE001
        return None


def surface_analyzer(surface) -> "ShapeAnalysis_Surface":
    return ShapeAnalysis_Surface(surface)


# ----------------------------------------------------------------------------
# Sewing + solid.
# ----------------------------------------------------------------------------

def sew_faces(faces, tol: float = 1e-3) -> TopoDS_Shape:
    """Sew a list of faces into a shell/compound."""
    sew = BRepBuilderAPI_Sewing(tol, True, True, True, False)
    for f in faces:
        if f is not None:
            sew.Add(f)
    sew.Perform()
    return sew.SewedShape()


def make_solid(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Wrap the first shell of `shape` into a solid, if closed, and orient it so
    the enclosed volume is positive (outward normals). Returns the solid when
    successful, else the input shape unchanged."""
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    if not exp.More():
        return shape
    shell = TopoDS.Shell_s(exp.Current())
    mk = BRepBuilderAPI_MakeSolid(shell)
    if not mk.IsDone():
        return shape
    solid = mk.Solid()
    if solid_volume(solid) < 0:  # inward-facing -> flip to conventional outward
        solid = TopoDS.Solid_s(solid.Reversed())
    return solid


def solid_volume(shape: TopoDS_Shape) -> float:
    """Signed enclosed volume (mm^3). Negative if faces are inward-oriented."""
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def shape_area(shape: TopoDS_Shape) -> float:
    """Total surface area (mm^2). Used to sanity-check a merged region against
    its source mesh — a collapsed/degenerate merge has a wildly wrong area."""
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    return float(props.Mass())


def sew_solidify(faces, tol: float = 0.05, heal_tol: float = 0.1) -> TopoDS_Shape:
    """Sew faces, heal, solidify, then heal again at the solid level and orient
    outward. The second heal fixes assembly-level invalidity (shared-edge
    tolerances) that only appears once the shell is closed into a solid. The
    workhorse for the analytic `solid` method."""
    sewn = sew_faces(faces, tol=tol)
    healed = heal(sewn, tol=heal_tol)
    solid = make_solid(healed)
    if not is_valid(solid):
        # Assembly-level invalidity (shared-edge tolerances) only appears once
        # the shell is closed. Heal, then re-wrap the shell into a fresh
        # (oriented) solid so we keep a valid *solid*, not a healed shell.
        solid = make_solid(heal(solid, tol=1e-3))
    return solid


# ----------------------------------------------------------------------------
# Validity + I/O.
# ----------------------------------------------------------------------------

def unify_same_domain(shape: TopoDS_Shape, tol: float = 0.1) -> TopoDS_Shape:
    """Merge faces sharing a surface (same-domain) into single faces. This is
    what turns a region's per-triangle same-surface faces into one analytic
    face, and does the topological merge robustly (OCCT owns it)."""
    up = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    up.SetLinearTolerance(tol)
    up.SetAngularTolerance(tol)
    up.Build()
    return up.Shape()


def heal(shape: TopoDS_Shape, tol: float = 1e-2) -> TopoDS_Shape:
    """ShapeFix pass to repair tolerances/orientation after merging."""
    fix = ShapeFix_Shape(shape)
    fix.SetPrecision(tol)
    fix.SetMaxTolerance(max(tol, 0.1))
    fix.Perform()
    return fix.Shape()


def is_valid(shape: TopoDS_Shape) -> bool:
    return bool(BRepCheck_Analyzer(shape).IsValid())


def count_faces(shape: TopoDS_Shape) -> int:
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        n += 1
        exp.Next()
    return n


def write_step(shape: TopoDS_Shape, path: str) -> bool:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    return writer.Write(path) == IFSelect_RetDone


def shape_to_mesh(shape: TopoDS_Shape, lin_defl: float = 0.1,
                  ang_defl: float = 0.5):
    """Re-tessellate a shape to (vertices Nx3, faces Mx3). Used by the gate to
    measure deviation between the generated STEP and the original mesh."""
    BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, True)
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        exp.Next()
        if tri is None:
            continue
        trsf = loc.Transformation()
        base = len(verts)
        nb_nodes = tri.NbNodes()
        for i in range(1, nb_nodes + 1):
            p = tri.Node(i).Transformed(trsf)
            verts.append([p.X(), p.Y(), p.Z()])
        reversed_face = face.Orientation().value == 1  # TopAbs_REVERSED
        for i in range(1, tri.NbTriangles() + 1):
            t = tri.Triangle(i)
            a, b, c = t.Value(1), t.Value(2), t.Value(3)
            if reversed_face:
                b, c = c, b
            faces.append([base + a - 1, base + b - 1, base + c - 1])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)

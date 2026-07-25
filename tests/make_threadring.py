"""Generate a ring with a KNOWN internal thread — ground truth for the thread
detector. M10x1.5 internal thread (ISO 261): major D=10, pitch P=1.5, internal
minor D1 = D - 1.0825*P = 8.376. The bore wall radius oscillates helically
between r_min=D1/2 and r_max=D/2 with a triangular flank profile. Outer wall is
a plain Ø16 cylinder; annular caps close the solid.

Known ground truth: internal thread M10x1.5, right-hand, straight (no taper),
bore axis Z, z in [0, 12].
"""
import numpy as np
import trimesh

P = 1.5                    # pitch, mm
R_MAJ = 5.0                # major radius (thread root in the wall)
R_MIN = 8.376 / 2          # internal minor radius (crest, smallest bore)
R_OUT = 8.0                # plain outer wall
H = 12.0                   # ring height
NZ, NT = 160, 96           # grid resolution


def r_inner(theta, z):
    """Triangular thread flank: crest (r_min) -> root (r_max) -> crest."""
    phase = (theta / (2 * np.pi) - z / P) % 1.0
    tri = 2 * np.abs(phase - 0.5)          # 1 at crest, 0 at root
    return R_MAJ - (R_MAJ - R_MIN) * tri


zs = np.linspace(0, H, NZ)
ts = np.linspace(-np.pi, np.pi, NT, endpoint=False)
TT, ZZ = np.meshgrid(ts, zs)

def grid_verts(rfun):
    R = rfun(TT, ZZ)
    return np.stack([R * np.cos(TT), R * np.sin(TT), ZZ], axis=-1)

inner = grid_verts(r_inner)
outer = grid_verts(lambda t, z: np.full_like(t, R_OUT))

verts, faces = [], []

def add_tube(P3, flip=False):
    base = len(verts)
    verts.extend(P3.reshape(-1, 3))
    for i in range(NZ - 1):
        for j in range(NT):
            a = base + i * NT + j
            b = base + i * NT + (j + 1) % NT
            c = base + (i + 1) * NT + j
            d = base + (i + 1) * NT + (j + 1) % NT
            if flip:
                faces.extend([[a, c, b], [b, c, d]])
            else:
                faces.extend([[a, b, c], [b, d, c]])

add_tube(inner, flip=True)      # bore wall faces inward
add_tube(outer, flip=False)

# annular caps (bottom z=0, top z=H) between inner ring and outer ring
for zi, flip in ((0, True), (NZ - 1, False)):
    ib = 0 if zi == 0 else (NZ - 1) * NT               # inner ring offset
    ob = NZ * NT + (0 if zi == 0 else (NZ - 1) * NT)   # outer ring offset
    for j in range(NT):
        a, b = ib + j, ib + (j + 1) % NT
        c, d = ob + j, ob + (j + 1) % NT
        if flip:
            faces.extend([[a, b, c], [b, d, c]])
        else:
            faces.extend([[a, c, b], [b, c, d]])

mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces))
mesh.merge_vertices()
out = "/Users/nickp/Documents/Projects/stl-step-v2/tests/fixtures/threadring.stl"
mesh.export(out)
print(f"wrote {out}  watertight={mesh.is_watertight}  faces={len(mesh.faces)}")

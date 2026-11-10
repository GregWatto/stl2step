"""Conservative refinements of a measured slab proposal.

Circular outer tapers are measured at multiple stations. Local residual cuts
must be constant between measured planar supports. All results are proposals;
full geometric validation remains mandatory.
"""
from __future__ import annotations
import numpy as np


def taper_script(pr, mesh):
    from .draw import (_fit_prismatic, _emit_prismatic_script, _section_polys,
                       fit_loop, _emit_loop_lines, _PLANE, _PSIGN)
    from shapely.geometry import Polygon
    fitted = _fit_prismatic(pr, arcs=True)
    if not fitted:
        return None
    lines = ['from build123d import *', '', 'with BuildPart() as part:']
    changes = 0
    sg, plane = _PSIGN[pr.axis], _PLANE[pr.axis]
    tol = max(.02, 1e-5*float(np.linalg.norm(mesh.extents)))
    for lo, hi, level in fitted:
        lo,hi=float(lo),float(hi)
        replacement = []
        if len(level) == 1 and len(level[0][0]) == 1 and level[0][0][0].kind == 'circle':
            # Mesh ring boundaries also mark cone/cylinder joins without flat caps.
            from .draw import _cluster_coords
            raw = mesh.vertices[:,pr.axis]
            bounds = _cluster_coords(raw[(raw > lo+1e-6)&(raw < hi-1e-6)],
                                     1e-6, np.ones(np.count_nonzero((raw > lo+1e-6)&(raw < hi-1e-6))),
                                     min_weight_frac=0)
            bounds = [lo]+bounds+[hi]
            if len(bounds) <= 34:
                for a,b in zip(bounds,bounds[1:]):
                    a,b=float(a),float(b)
                    sections = [_section_polys(mesh,pr.axis,a+f*(b-a)) for f in (.15,.35,.5,.65,.85)]
                    if any(not s or len(s)!=1 for s in sections):
                        replacement=[]; break
                    circles = [fit_loop(s[0][0]) for s in sections]
                    if any(not c or len(c)!=1 or c[0].kind!='circle' for c in circles):
                        replacement=[]; break
                    centres = np.array([c[0].center for c in circles])
                    if np.max(np.linalg.norm(centres-centres.mean(0),axis=1)) > tol:
                        replacement=[]; break
                    radii = np.array([c[0].radius for c in circles])
                    slope,intercept = np.polyfit([.15,.35,.5,.65,.85],radii,1)
                    if np.max(abs(radii-(intercept+slope*np.array([.15,.35,.5,.65,.85])))) > tol:
                        replacement=[]; break
                    # Interior profiles must remain stationary; never loft across
                    # a changing hole, pocket, or topology transition.
                    inners = [[Polygon(h) for h in s[0][1]] for s in sections]
                    from shapely.ops import unary_union
                    holes = [unary_union(h) for h in inners]
                    if any(h.symmetric_difference(holes[2]).area > tol*max(holes[2].length,1)
                           for h in holes):
                        replacement=[]; break
                    hs = [fit_loop(h) for h in sections[2][0][1]]
                    if any(not h for h in hs):
                        replacement=[]; break
                    replacement.append((a,b,tuple(float(x) for x in centres.mean(0)),float(intercept),float(intercept+slope),hs))
        if replacement and any(abs(r1-r0)>tol for _,_,_,r0,r1,_ in replacement):
            changes += 1
            for a,b,centre,r0,r1,holes in replacement:
                if min(r0,r1)<=0:
                    return None
                lines += [f'    # measured circular envelope: {a:.9g} .. {b:.9g}',
                          f'    with BuildSketch(Plane.{plane}.offset({sg*a!r})):',
                          f'        with Locations(({centre[0]!r}, {centre[1]!r})):',
                          f'            Circle({r0!r})',
                          f'    with BuildSketch(Plane.{plane}.offset({sg*b!r})):',
                          f'        with Locations(({centre[0]!r}, {centre[1]!r})):',
                          f'            Circle({r1!r})',
                          '    loft(ruled=True)']
                if holes:
                    lines += [f'    with BuildSketch(Plane.{plane}.offset({sg*a!r})):']
                    for h in holes:
                        lines += _emit_loop_lines(h,'',8)
                    lines += [f'    extrude(amount={sg*(b-a)!r}, mode=Mode.SUBTRACT)']
        else:
            # Keep the exact original slab endpoints and profiles.
            body = _emit_prismatic_script(pr,[(lo,hi,level)]).split('with BuildPart() as part:\n',1)[1]
            lines += body.splitlines()
    return '\n'.join(lines)+'\n' if changes else None


def residual_cut_script(mesh, part, script):
    """Subtract a stationary excess profile between measured planar supports.

    A residual is measured independently at nine interior stations, including
    near each end. Missing material is never filled by this operation.
    """
    from .featurecheck import CADSections, _slice_region
    from .draw import _cluster_coords, fit_loop, _emit_loop_lines, _PLANE, _PSIGN
    from shapely.geometry import Polygon
    diag = float(np.linalg.norm(mesh.extents))
    tol = max(.02,1e-5*diag)
    cad = CADSections(part)
    lines=[]; cuts=0
    for axis in range(3):
        ids = np.flatnonzero(abs(mesh.face_normals[:,axis])>1-1e-6)
        if len(ids)==0:
            continue
        levels = _cluster_coords(mesh.triangles_center[ids,axis],max(1e-6,1e-6*diag),
                                 mesh.area_faces[ids],min_weight_frac=0)
        if len(levels)>16:
            continue
        for a,b in zip(levels,levels[1:]):
            a,b=float(a),float(b)
            if b-a < 2*tol:
                continue
            measurements=[]
            for f in (.01,.05,.15,.3,.5,.7,.85,.95,.99):
                t=a+f*(b-a)
                target=_slice_region(mesh,axis,t); candidate=cad.section_region(axis,t)
                if target is None or candidate is None:
                    measurements=[]; break
                extra=candidate.difference(target)
                measurements.append((target,list(getattr(extra,'geoms',[extra]))))
            if not measurements:
                continue
            for region in measurements[4][1]:
                if region.is_empty or region.geom_type!='Polygon' or region.interiors or region.area < .001*diag**2:
                    continue
                stable=True
                for target,extras in measurements:
                    matches=[r for r in extras if r.geom_type=='Polygon' and r.intersection(region).area>.5*region.area]
                    if not matches or min(r.symmetric_difference(region).area for r in matches)>tol*region.length:
                        stable=False; break
                if not stable:
                    continue
                profile=fit_loop(np.asarray(region.exterior.coords))
                if not profile:
                    continue
                # Fitted arcs may add material to a cut: validate that footprint
                # against the independently measured profile in every station.
                from .compose import exec_script
                sg,plane=_PSIGN[axis],_PLANE[axis]
                cutlines=[f'    # stationary residual cut: {"xyz"[axis]}={a:.9g}..{b:.9g}',
                          f'    with BuildSketch(Plane.{plane}.offset({sg*a!r})):']
                cutlines+=_emit_loop_lines(profile,'',8)
                cutlines+=[f'    extrude(amount={sg*(b-a)!r}, mode=Mode.SUBTRACT)']
                # Verify the actual fitted tool rather than only its source loop.
                toolbody=['from build123d import *','with BuildPart() as part:']+cutlines[1:-1]+[f'    extrude(amount={sg*(b-a)!r})']
                tool=CADSections(exec_script('\n'.join(toolbody)))
                for f,(target,_) in zip((.01,.05,.15,.3,.5,.7,.85,.95,.99),measurements):
                    toolregion=tool.section_region(axis,a+f*(b-a))
                    if toolregion is None or target.intersection(toolregion).area > tol*region.length:
                        stable=False; break
                if stable:
                    lines+=cutlines; cuts+=1
                if cuts>=8:
                    return script+'\n'+'\n'.join(lines)+'\n',cuts
    return (script+'\n'+'\n'.join(lines)+'\n',cuts) if cuts else (None,0)


def refine_prismatic(pr, mesh):
    from .compose import exec_script
    from .draw import _emit_prismatic_script, _fit_prismatic
    if len(mesh.faces)>10000 or not mesh.is_watertight:
        return None
    script=taper_script(pr,mesh) or _emit_prismatic_script(pr,_fit_prismatic(pr,arcs=True))
    part=exec_script(script)
    refined,cuts=residual_cut_script(mesh,part,script)
    if refined:
        script=refined; part=exec_script(script)
    if not part.is_valid or len(part.solids())!=1:
        return None
    return part,script,{'levels':len(pr.levels),'stationary_cuts':cuts}

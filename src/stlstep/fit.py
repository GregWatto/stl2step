"""Numerical parameter fitting — snap a drawn script's dimensions to the mesh.

The LLM (or a recognizer) proposes the feature STRUCTURE — which operations, how
they compose. It is good at that and bad at the last 0.1 mm: it guessed the
diff_case cavity floor at y=7.0 when the truth was 6.8. This module closes that
gap deterministically, the way CADFit does: take the numeric literals out of the
build123d script, treat them as free parameters, and search for the values that
minimise deviation to the mesh. "The LLM gets the topology; the math gets the
millimetres."

It is composer-agnostic: it operates on the emitted script's numbers by source
span (so formatting and comments survive), rebuilding + measuring per trial with a
cheap derivative-free pattern search.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class Param:
    start: int          # absolute char offset of the literal in the source
    end: int
    value: float
    is_int: bool


def numeric_params(script: str) -> list[Param]:
    """Every numeric literal in the script, by source span. Booleans are skipped
    (they're `Constant` too). Negative numbers parse as USub(Constant), so the
    span covers the magnitude — the sign stays fixed, the magnitude is fitted."""
    tree = ast.parse(script)
    starts = _line_starts(script)
    out: list[Param] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool) \
                and isinstance(node.value, (int, float)):
            a = starts[node.lineno - 1] + node.col_offset
            b = starts[node.end_lineno - 1] + node.end_col_offset
            out.append(Param(a, b, float(node.value), isinstance(node.value, int)))
    out.sort(key=lambda p: p.start)
    return out


def render(script: str, params: list[Param], values, *, fmt="{:.10g}") -> str:
    """Substitute `values` into the script at each param's span (back-to-front so
    earlier offsets stay valid)."""
    s = script
    for p, v in sorted(zip(params, values), key=lambda pv: -pv[0].start):
        s = s[:p.start] + fmt.format(float(v)) + s[p.end:]
    return s


def _line_starts(s: str) -> list[int]:
    starts, off = [0], 0
    for line in s.splitlines(keepends=True):
        off += len(line)
        starts.append(off)
    return starts


# ----------------------------------------------------------------------------
# Objective + search.
# ----------------------------------------------------------------------------

def _build(script: str):
    import build123d
    from .compose import _extract_part
    ns = {}
    exec(compile(script, "<fit>", "exec"), {**vars(build123d)}, ns)
    return _extract_part(ns)


def _deviation(part, mesh, pts_on_mesh):
    """Symmetric p99 deviation: max of (mesh->model) and (model->mesh). The
    objective MUST be deterministic across trials or the pattern search chases
    sampling noise and stalls — so the forward pass reuses the fixed `pts_on_mesh`
    and the reverse pass uses the model's tessellation VERTICES (deterministic per
    part), not a fresh random surface sample."""
    from . import occ_util as ou
    if part is None or getattr(part, "volume", 0) <= 0:
        return 1e6
    rv, rf = ou.shape_to_mesh(part.wrapped, 0.03)   # fine enough to resolve <0.1mm
    if len(rf) == 0:
        return 1e6
    gen = trimesh.Trimesh(vertices=rv, faces=rf, process=False)
    _, fwd, _ = trimesh.proximity.closest_point(gen, pts_on_mesh)
    _, rev, _ = trimesh.proximity.closest_point(mesh, np.asarray(rv))
    return float(max(np.quantile(fwd, 0.99), np.quantile(rev, 0.99)))


@dataclass
class FitResult:
    script: str
    dev_before: float
    dev_after: float
    n_params: int
    n_evals: int


def fit_parameters(script: str, mesh: trimesh.Trimesh, *, samples: int = 2500,
                   max_evals: int = 400, seed: int = 0,
                   revert_tol: float = 0.02) -> FitResult:
    """Snap the script's numeric literals to the mesh, trusting the LLM's values
    unless the mesh clearly says otherwise.

    Anti-drift is essential: when the remaining error is STRUCTURAL (a missing
    feature), the deviation is flat w.r.t. most parameters, so the optimiser
    wanders those flat directions and corrupts already-correct dimensions (it will
    even prefer a globally-wrong model whose scalar deviation is marginally lower —
    on diff_case it shrank a correct radius to spread a uniform 0.6mm error). The
    guard is a REVERT-CLEANUP pass after optimisation: snap each parameter back to
    its initial value `x0` unless moving it genuinely helped (true deviation stays
    within `revert_tol`), largest drift first (that's where corruption hides).
    A never-regress guard then returns the original script if nothing helped — which
    is itself the signal that the STRUCTURE, not the numbers, is wrong. (An
    in-objective Tikhonov penalty was tried and rejected: it degraded Powell's
    convergence on the clean cases.)"""
    params = numeric_params(script)
    np.random.seed(seed)                        # fix the one-time forward sample
    pts, _ = trimesh.sample.sample_surface(mesh, samples)
    pts = np.asarray(pts)

    n_evals = [0]

    def dev(vals) -> float:
        n_evals[0] += 1
        try:
            return _deviation(_build(render(script, params, vals)), mesh, pts)
        except Exception:  # noqa: BLE001 — a bad parameter set just scores inf
            return 1e6

    x0 = np.array([p.value for p in params], dtype=float)
    dev0 = dev(x0)
    if not len(x0):
        return FitResult(script, dev0, dev0, 0, n_evals[0])

    x, _ = _powell(dev, x0, max_evals)

    # Revert-cleanup on true deviation: undo any drift that didn't earn its keep.
    best = dev(x)
    for i in np.argsort(-np.abs(x - x0)):
        if x[i] == x0[i]:
            continue
        trial = x.copy(); trial[i] = x0[i]
        f = dev(trial)
        if f <= best + revert_tol:
            x, best = trial, f

    if best >= dev0:                            # nothing helped -> structure is wrong
        x, best = x0.copy(), dev0
    fitted = render(script, params, x, fmt="{:.6g}")
    return FitResult(fitted, dev0, best, len(params), n_evals[0])


def _powell(objective, x0, max_evals):
    """Derivative-free minimisation. Powell handles coupled parameters (a hole
    radius trading against a plate size) far better than coordinate descent; the
    numpy pattern-search is a fallback if SciPy is unavailable."""
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, method="Powell",
                       options={"maxfev": max_evals, "xtol": 1e-4, "ftol": 1e-4})
        return np.asarray(res.x, dtype=float), float(res.fun)
    except Exception:  # noqa: BLE001
        pass
    x = x0.copy(); best = objective(x)
    step = np.maximum(np.abs(x) * 0.10, 0.10)
    evals = 1
    while evals < max_evals and step.max() > 1e-3:
        moved = False
        for i in range(len(x)):
            for s in (step[i], -step[i]):
                t = x.copy(); t[i] += s
                f = objective(t); evals += 1
                if f < best - 1e-6:
                    x, best, moved = t, f, True
                    step[i] *= 1.3
                    break
            else:
                step[i] *= 0.5
        if not moved and step.max() <= 1e-3:
            break
    return x, best

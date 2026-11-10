#!/usr/bin/env python3
"""Reproduce upstream reconstruction and require execution of every compose gate.

No reconstruction algorithms or tolerances are changed. Workers observe gate
calls so exceptions suppressed by upstream cannot turn into a pipeline PASS.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe_gate(name, fn, receipts):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        record = {'name': name, 'completed': False}
        receipts.append(record)
        try:
            result = fn(*args, **kwargs)
            if name == 'feature_gap' and not isinstance(result, list):
                raise ValueError('feature check returned no list')
            if name == 'slice_occupancy':
                axes = result.get('per_axis', []) if isinstance(result, dict) else []
                if ({a['axis'] for a in axes} != set('xyz')
                        or any(a.get('mean_iou') is None or not a.get('coords') for a in axes)):
                    raise ValueError('occupancy did not measure all three axes')
                record['axes'] = [a['axis'] for a in axes]
            record['completed'] = True
            return result
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise
    return wrapped


def worker(argv):
    """One fresh CAD process per stage; observes, but never bypasses, gates."""
    sys.path.insert(0, str(ROOT / 'src'))
    seed = int(os.environ.get('STLSTEP_TEST_SEED', '0'))
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    # trimesh>=4 uses a fresh default_rng when sample_surface has no seed.
    # Supply a reproducible sequence only in this test process; same sampler,
    # counts and acceptance thresholds, with a distinct stream for each call.
    import trimesh
    sampler = trimesh.sample.sample_surface
    streams = np.random.default_rng(seed)

    @functools.wraps(sampler)
    def sample_surface(*args, **kwargs):
        if len(args) < 4 and kwargs.get('seed') is None:
            kwargs['seed'] = int(streams.integers(0, 2**32))
        return sampler(*args, **kwargs)

    trimesh.sample.sample_surface = sample_surface
    if argv[0] == 'convert':
        from stlstep.convert import main
        return main(argv[1:])
    from stlstep import compose, featurecheck
    if argv[0] != 'run':
        return compose.main(argv)
    receipts = []
    names = ('feature_gap', 'slice_occupancy', '_match_rings')
    originals = {name: getattr(featurecheck, name) for name in names}
    for name in names:
        setattr(featurecheck, name, observe_gate(name, originals[name], receipts))
    receipt_path = Path(argv[2]) / 'gate-execution.json'
    try:
        code = compose.main(argv)
    finally:
        for name, original in originals.items():
            setattr(featurecheck, name, original)
        trimesh.sample.sample_surface = sampler
        write_json(receipt_path, receipts)
    missing = set(names) - {r['name'] for r in receipts if r['completed']}
    errors = [r for r in receipts if not r['completed']]
    if missing or errors:
        print(f'INCOMPLETE VALIDATION: missing={sorted(missing)}, errors={errors}')
        return 2
    return code


def assess_metrics(metrics):
    """Check serialized upstream results independently; reject non-finite data."""
    failures = []
    for key in ('deviation_mm', 'reverse_max_mm', 'tolerance_mm', 'volume_mm3'):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f'missing/non-finite {key}')
    if failures:
        return failures
    if metrics['tolerance_mm'] <= 0 or metrics['volume_mm3'] <= 0:
        failures.append('nonpositive tolerance or volume')
    # tolerance_mm is rounded by upstream; use the upstream verdict for its
    # exact boundary decision rather than comparing against a rounded threshold.
    if not all(metrics.get(k) is True for k in ('passed', 'valid', 'closed')):
        failures.append('upstream validity/closure/deviation/feature verdict failed')
    gaps = metrics.get('feature_gaps')
    if not isinstance(gaps, list):
        failures.append('feature gaps missing')
    elif any(g['found'] is not None and g['found'] < g['expected'] for g in gaps):
        failures.append('missing editable features')
    occ = metrics.get('slice_occupancy')
    if not isinstance(occ, dict) or occ.get('mean_iou') is None:
        failures.append('occupancy missing')
    else:
        for key in ('missing_frac', 'extra_frac', 'mean_iou', 'void_missing_frac'):
            value = occ.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(f'missing/non-finite occupancy {key}')
        if any(occ.get(k, float('inf')) > 0.03 for k in ('missing_frac', 'extra_frac')):
            failures.append('occupancy exceeds upstream 3% limit')
        if not isinstance(occ.get('loop_mismatches'), list) or occ['loop_mismatches']:
            failures.append('opening shape check failed or missing')
    return failures


def run_case(stl, script, output, timeout, bands, seed):
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'MPLBACKEND': 'Agg',
           'PYTHONHASHSEED': str(seed), 'STLSTEP_TEST_SEED': str(seed),
           'XDG_CACHE_HOME': str(output / 'cache'),
           'MPLCONFIGDIR': str(output / 'cache/matplotlib'),
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    record = {'input': str(stl), 'input_sha256': sha256(stl), 'seed': seed,
              'stages': [], 'failures': [], 'geometry_pass': False,
              'visual_review': 'pending'}

    def stage(name, args, allowed=(0,)):
        command = [sys.executable, str(Path(__file__).resolve()), '_worker', *args]
        entry = {'stage': name, 'command': command}
        start = time.monotonic()
        with (output / f'{name}.log').open('w') as log:
            try:
                p = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout)
                entry['exit_code'] = p.returncode
            except subprocess.TimeoutExpired:
                entry['error'] = f'timeout after {timeout}s'
        entry['elapsed_s'] = round(time.monotonic() - start, 3)
        record['stages'].append(entry)
        if entry.get('exit_code') not in allowed:
            record['failures'].append(f'{name}: {entry.get("error", entry.get("exit_code"))}')
            return False
        return True

    if script:
        candidate = output / 'candidate.py'
        shutil.copyfile(script, candidate)
        record['script_source'] = str(script)
    else:
        stage('convert', ['convert', str(stl), '-o', str(output / 'proposal.step'),
                         '--method', 'draw', '--report', str(output)], allowed=(0, 1))
        candidate = output / 'feature_tree.py'
    if not candidate.exists():
        record['failures'].append('no editable candidate; recognizer miss/error (see convert.log)')
    else:
        # Keep an immutable input copy; compose writes feature_tree.py on export.
        if candidate.name != 'candidate.py':
            shutil.copyfile(candidate, output / 'candidate.py')
            candidate = output / 'candidate.py'
        record['script_sha256'] = sha256(candidate)
        if stage('brief', ['brief', str(stl), str(output)]):
            args = ['run', str(stl), str(output), str(candidate), '--tag', 'checked',
                    '--out', str(output / 'part.step')]
            for band in bands:
                args.extend(['--thread-band', band])
            stage('gates', args)
            metrics_path = output / 'metrics.json'
            if metrics_path.exists():
                record['failures'].extend(assess_metrics(json.loads(metrics_path.read_text())))
            else:
                record['failures'].append('missing metrics.json')
            stage('compare', ['compare', str(stl), str(output), str(candidate), '--tag', 'final'])
            for artifact in ('part.step', 'feature_tree.py', 'analysis.json',
                             'checked_missing.png', 'checked_extra.png',
                             'checked_slices.png', 'final_compare.png', 'gate-execution.json'):
                if not (output / artifact).is_file() or not (output / artifact).stat().st_size:
                    record['failures'].append(f'missing artifact {artifact}')
    record['geometry_pass'] = not record['failures']
    write_json(output / 'result.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--fixture', action='append', help='fixture stem; repeatable')
    inputs.add_argument('--all-fixtures', action='store_true')
    inputs.add_argument('--stl', type=Path)
    parser.add_argument('--script', type=Path, help='Codex-authored build123d candidate (with --stl)')
    parser.add_argument('--out', type=Path, default=ROOT / 'out' / time.strftime('pipeline-%Y%m%d-%H%M%S'))
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--thread-band', action='append', default=[])
    args = parser.parse_args(argv)
    if args.script and not args.stl:
        parser.error('--script requires --stl')
    stls = ([args.stl.resolve()] if args.stl else
            sorted((ROOT / 'tests/fixtures').glob('*.stl')) if args.all_fixtures else
            [ROOT / 'tests/fixtures' / f'{name}.stl' for name in args.fixture])
    if not stls or any(not p.is_file() for p in stls):
        parser.error('no matching STL fixtures, or an input does not exist')
    if args.script and not args.script.is_file():
        parser.error('candidate script does not exist')
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    (output / 'dependencies.log').write_text(check.stdout + check.stderr)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (output / 'requirements-resolved.txt').write_text(freeze.stdout)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    provenance = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'src').rglob('*.py'))}
    summary = {'python': sys.version, 'platform': platform.platform(), 'revision': revision,
               'source_sha256': provenance, 'pipeline_sha256': sha256(__file__),
               'dependencies_pass': check.returncode == 0, 'cases': []}
    if check.returncode == 0:
        for index, stl in enumerate(stls):
            result = run_case(stl, args.script.resolve() if args.script else None,
                              output / f'{index:02d}-{stl.stem}', args.timeout, args.thread_band, args.seed)
            summary['cases'].append(result)
            print(f'{stl.name}: {"PASS" if result["geometry_pass"] else "FAIL"} {result["failures"]}', flush=True)
            write_json(output / 'summary.json', summary)
    summary['passed'] = bool(summary['dependencies_pass'] and summary['cases'] and
                             all(r['geometry_pass'] for r in summary['cases']))
    write_json(output / 'summary.json', summary)
    print(f'Results: {output}; visual review remains pending')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_worker':
        raise SystemExit(worker(sys.argv[2:]))
    raise SystemExit(main())

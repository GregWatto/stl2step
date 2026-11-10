"""The public compose run CLI must fail explicitly when validation is incomplete."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import trimesh
from stlstep import compose


class ComposeValidationTests(unittest.TestCase):
    def run_cli(self, *, analysis=True, feature_error=False, occupancy_error=False,
                incomplete=False, nonfinite=False, missing_openings=False):
        occupancy = {'per_axis': [{'axis': a, 'mean_iou': 1, 'coords': [0]} for a in 'xyz'],
                     'mean_iou': 1, 'missing_mm3': 0, 'extra_mm3': 0,
                     'missing_frac': 0, 'extra_frac': 0, 'void_missing_frac': 0,
                     'void_missing_mm3': 0, 'loop_mismatches': [], 'worst_slices': []}
        if incomplete:
            occupancy['per_axis'].pop()
        if nonfinite:
            occupancy['mean_iou'] = float('nan')
        if missing_openings:
            del occupancy['loop_mismatches']
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root / 'attempt.py').write_text('# candidate mocked at the kernel boundary\n')
            if analysis:
                (root / 'analysis.json').write_text(json.dumps({'features': {'holes': [], 'bosses': []}}))
            candidate = compose.Candidate(ok=True, script='', part=SimpleNamespace(wrapped=object()),
                                          valid=True, closed=True, deviation_mm=0.,
                                          reverse_max_mm=0., volume_mm3=1.)
            stack.enter_context(patch('stlstep.ingest.load_and_repair', return_value=SimpleNamespace(
                mesh=trimesh.creation.box())))
            stack.enter_context(patch.object(compose, 'run_candidate', return_value=candidate))
            stack.enter_context(patch('stlstep.featurecheck.CADSections', return_value=trimesh.creation.box()))
            stack.enter_context(patch('stlstep.featurecheck.feature_gap', return_value=[],
                                     side_effect=RuntimeError('feature kernel broke') if feature_error else None))
            stack.enter_context(patch('stlstep.featurecheck.slice_occupancy', return_value=occupancy,
                                     side_effect=RuntimeError('occupancy kernel broke') if occupancy_error else None))
            stack.enter_context(patch('stlstep.featurecheck.render_slice_report', return_value=None))
            stack.enter_context(patch('stlstep.occ_util.shape_to_mesh', return_value=(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])))
            output = StringIO()
            with redirect_stdout(output):
                code = compose.main(['run', 'unused.stl', directory, str(root / 'attempt.py')])
            metrics = json.loads((root / 'metrics.json').read_text())
        return code, metrics, output.getvalue()

    def test_complete_checks_pass(self):
        code, metrics, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertTrue(metrics['passed'])
        self.assertEqual(metrics['validation_errors'], [])

    def test_missing_analysis_fails(self):
        code, metrics, output = self.run_cli(analysis=False)
        self.assertEqual(code, 1)
        self.assertFalse(metrics['passed'])
        self.assertIn('run `brief` first', output)

    def test_feature_and_occupancy_exceptions_are_reported(self):
        for option, message in [('feature_error', 'feature kernel broke'),
                                ('occupancy_error', 'occupancy kernel broke')]:
            with self.subTest(option=option):
                code, metrics, output = self.run_cli(**{option: True})
                self.assertEqual(code, 1)
                self.assertFalse(metrics['passed'])
                self.assertTrue(any(message in e for e in metrics['validation_errors']))
                self.assertIn('VALIDATION ERROR:', output)

    def test_incomplete_or_nonfinite_occupancy_fails(self):
        for option in ['incomplete', 'nonfinite', 'missing_openings']:
            with self.subTest(option=option):
                code, metrics, _ = self.run_cli(**{option: True})
                self.assertEqual(code, 1)
                self.assertTrue(metrics['validation_errors'])


if __name__ == '__main__':
    unittest.main()

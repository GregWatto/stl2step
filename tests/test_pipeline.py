"""Regression contracts for the validation boundary, not reconstruction internals."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).resolve().parents[1] / 'tools/reconstruction_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class GateExecutionTests(unittest.TestCase):
    def test_swallowed_feature_exception_cannot_pass_worker(self):
        from stlstep import compose, featurecheck
        with tempfile.TemporaryDirectory() as directory:
            def upstream(_):
                try:
                    featurecheck.feature_gap({}, None)
                except RuntimeError:
                    pass  # simulate a regression that suppresses a gate error
                return 0
            with patch.object(featurecheck, 'feature_gap', side_effect=RuntimeError('injected kernel failure')):
                with patch.object(compose, 'main', side_effect=upstream):
                    self.assertEqual(pipeline.worker(['run', 'unused.stl', directory, 'unused.py']), 2)
            receipts = json.loads((Path(directory) / 'gate-execution.json').read_text())
            self.assertIn('injected kernel failure', receipts[0]['error'])

    def test_missing_occupancy_axis_is_incomplete(self):
        receipts = []
        fn = pipeline.observe_gate('slice_occupancy', lambda: {
            'per_axis': [{'axis': 'x', 'coords': [1], 'mean_iou': 1.0}]}, receipts)
        with self.assertRaisesRegex(ValueError, 'all three axes'):
            fn()
        self.assertFalse(receipts[0]['completed'])

    def test_opening_check_exception_is_recorded_even_if_caught(self):
        receipts = []
        def broken():
            raise ValueError('invalid opening polygon')
        fn = pipeline.observe_gate('_match_rings', broken, receipts)
        try:
            fn()
        except ValueError:
            pass
        self.assertFalse(receipts[0]['completed'])
        self.assertIn('invalid opening polygon', receipts[0]['error'])

    def test_skipped_checks_cannot_pass_worker(self):
        from stlstep import compose
        with tempfile.TemporaryDirectory() as directory, patch.object(compose, 'main', return_value=0):
            self.assertEqual(pipeline.worker(['run', 'unused.stl', directory, 'unused.py']), 2)


class MetricContractTests(unittest.TestCase):
    def metrics(self):
        return {'deviation_mm': 0, 'reverse_max_mm': 0, 'tolerance_mm': 0.1,
                'volume_mm3': 100, 'valid': True, 'closed': True, 'passed': True,
                'feature_gaps': [], 'slice_occupancy': {
                    'mean_iou': 1, 'missing_frac': 0, 'extra_frac': 0,
                    'void_missing_frac': 0, 'loop_mismatches': []}}

    def test_complete_pass(self):
        self.assertEqual(pipeline.assess_metrics(self.metrics()), [])

    def test_filled_cavity_cannot_pass(self):
        metrics = self.metrics()
        metrics['slice_occupancy']['extra_frac'] = 0.2
        self.assertTrue(pipeline.assess_metrics(metrics))

    def test_small_wrong_opening_cannot_pass(self):
        metrics = self.metrics()
        metrics['slice_occupancy']['loop_mismatches'] = [{'kind': 'shape mismatch'}]
        self.assertTrue(pipeline.assess_metrics(metrics))

    def test_nonfinite_deviation_cannot_pass(self):
        metrics = self.metrics()
        metrics['deviation_mm'] = float('nan')
        self.assertTrue(pipeline.assess_metrics(metrics))

    def test_missing_feature_cannot_pass(self):
        metrics = self.metrics()
        metrics['feature_gaps'] = [{'expected': 1, 'found': 0}]
        self.assertTrue(pipeline.assess_metrics(metrics))


if __name__ == '__main__':
    unittest.main()

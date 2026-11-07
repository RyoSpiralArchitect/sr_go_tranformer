import json
import math
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / 'validation' / 'performance'


def receipt(name):
    if Path(name).name != name:
        raise ValueError('receipt must be a basename')
    return json.loads((ROOT / name).read_text())


class FP32OutputsReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = receipt('fp32-outputs-selection-20261107.json')

    def check_row(self, row, ratios_key, count, expected):
        ratios = row[ratios_key]
        for values in (row['before_ns'], row['after_ns'], ratios):
            self.assertEqual(len(values), count)
            self.assertTrue(all(math.isfinite(x) and x > 0 for x in values))
        for old, new, ratio in zip(row['before_ns'], row['after_ns'], ratios):
            self.assertTrue(math.isclose(new / old, ratio, rel_tol=1e-12))
        median = statistics.median(ratios)
        self.assertTrue(math.isclose(median, row['median_ratio'], rel_tol=1e-12))
        self.assertLessEqual(abs(median - expected), 0.000050001)

    def test_kernel(self):
        group = self.index['kernel']
        hashes = None
        self.assertEqual(len(group['receipts']), 3)
        for run, name in enumerate(group['receipts']):
            data = receipt(name)
            self.assertEqual(data['Protocol'], group['protocol'])
            self.assertEqual(data['Hashes']['monolith.go'], self.index['baseline_runtime_sha256'])
            if hashes is None:
                hashes = data['Hashes']
            self.assertEqual(data['Hashes'], hashes)
            self.assertEqual(len(data['Results']), group['conditions'])
            expected = group['approximate_process_medians']
            self.assertEqual({row['name'] for row in data['Results']}, set(expected))
            for row in data['Results']:
                self.check_row(row, 'paired_ratios', group['trials'], expected[row['name']][run])
                self.assertEqual(row['before_allocs'], 0)
                self.assertEqual(row['after_allocs'], 0)

    def test_generation_and_controls(self):
        group = self.index['generation']
        build = receipt(group['build_receipt'])
        control = receipt(group['control_build_receipt'])
        self.assertEqual(build['baseline'], self.index['baseline'])
        self.assertEqual(build['candidate_runtime_sha256'], self.index['candidate_runtime_sha256'])
        self.assertEqual(control['candidate_runtime_sha256'], self.index['baseline_runtime_sha256'])
        self.assertEqual(build['driver_sha256'], control['driver_sha256'])
        self.assertEqual(build['go_version'], control['go_version'])
        for names, expected, is_control in (
            (group['receipts'], group['approximate_process_medians'], False),
            (group['control_receipts'], group['approximate_control_medians'], True),
        ):
            self.assertEqual(len(names), 3)
            for run, name in enumerate(names):
                rows = receipt(name)
                self.assertEqual(len(rows), 5)
                self.assertEqual({f"prefix{r['prefix']}/outputs{r['outputs']}" for r in rows}, set(expected))
                for row in rows:
                    key = f"prefix{row['prefix']}/outputs{row['outputs']}"
                    self.check_row(row, 'ratios', group['trials'], expected[key][run])
                    if not is_control and key == 'prefix33/outputs128':
                        elapsed = group['outputs128_process_elapsed_medians_ms']
                        self.assertAlmostEqual(statistics.median(row['before_ns']) / 1e6, elapsed['before'][run])
                        self.assertAlmostEqual(statistics.median(row['after_ns']) / 1e6, elapsed['after'][run])


if __name__ == '__main__':
    unittest.main()

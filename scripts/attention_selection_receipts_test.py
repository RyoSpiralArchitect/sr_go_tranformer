import json
import math
import statistics
import unittest
from pathlib import Path


PERFORMANCE = Path(__file__).resolve().parents[1] / 'validation' / 'performance'


def receipt(name):
    if Path(name).name != name:
        raise ValueError('receipt must be a basename')
    return json.loads((PERFORMANCE / name).read_text())


class AttentionSelectionReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = receipt('attention-candidate-selection-20261106.json')

    def check_ratios(self, first, second, ratios):
        self.assertEqual(len(first), len(second))
        self.assertEqual(len(first), len(ratios))
        for a, b, ratio in zip(first, second, ratios):
            self.assertTrue(all(math.isfinite(x) and x > 0 for x in (a, b, ratio)))
            self.assertTrue(math.isclose(b / a, ratio, rel_tol=1e-12, abs_tol=1e-12))

    def test_controlled_and_callpath_receipts(self):
        self.assertEqual(len(self.index['groups']), 3)
        for group in self.index['groups']:
            self.assertEqual(len(group['receipts']), 3)
            first_hashes = None
            for run, name in enumerate(group['receipts']):
                with self.subTest(receipt=name):
                    data = receipt(name)
                    self.assertEqual(data['Protocol'], group['protocol'])
                    self.assertEqual(data['TrialsPerKind'], group['trials'])
                    self.assertEqual(data['CallsPerBlock'], group['calls'])
                    self.assertEqual((data['Workers'], data['Procs']), (4, 4))
                    self.assertEqual(data['Hashes']['monolith.go'], self.index['runtime_sha256'])
                    for digest in data['Hashes'].values():
                        self.assertRegex(digest, r'^[0-9a-f]{64}$')
                    if first_hashes is None:
                        first_hashes = data['Hashes']
                    self.assertEqual(data['Hashes'], first_hashes)
                    self.assertEqual(data.get('Allocations', {}), group.get('allocations', {}))
                    summaries = group['approximate_medians']
                    self.assertEqual({x['Kind'] for x in data['Samples']}, set(summaries))
                    self.assertEqual(len(data['Samples']), group['trials'] * len(summaries))
                    for kind, medians in summaries.items():
                        self.assertEqual(len(medians), 3)
                        samples = [x for x in data['Samples'] if x['Kind'] == kind]
                        self.assertEqual(len(samples), group['trials'])
                        self.assertEqual({x['Trial'] for x in samples}, set(range(group['trials'])))
                        for sample in samples:
                            self.assertEqual(sample['FirstVariant'], sample['Trial'] % 2)
                        self.check_ratios([x['NSPerCall'][0] for x in samples],
                                          [x['NSPerCall'][1] for x in samples],
                                          [x['Ratio'] for x in samples])
                        self.assertLessEqual(abs(statistics.median(x['Ratio'] for x in samples)
                                                 - medians[run]), 0.000050001)

    def test_generation_receipts(self):
        group = self.index['generation']
        build = receipt(group['build_receipt'])
        self.assertEqual(build['baseline'], group['baseline'])
        self.assertEqual(build['candidate_runtime_sha256'], self.index['runtime_sha256'])
        self.assertEqual(len(group['receipts']), 3)
        expected = group['approximate_medians']
        for run, name in enumerate(group['receipts']):
            with self.subTest(receipt=name):
                rows = receipt(name)
                self.assertEqual(len(rows), len(expected))
                self.assertEqual({f"{x['prefix']}/{x['outputs']}" for x in rows}, set(expected))
                for row in rows:
                    self.assertEqual(len(row['ratios']), group['trials'])
                    self.check_ratios(row['before_ns'], row['after_ns'], row['ratios'])
                    median = statistics.median(row['ratios'])
                    self.assertTrue(math.isclose(row['median_ratio'], median, rel_tol=1e-12))
                    key = f"{row['prefix']}/{row['outputs']}"
                    self.assertEqual(len(expected[key]), 3)
                    self.assertLessEqual(abs(median - expected[key][run]), 0.000500001)


if __name__ == '__main__':
    unittest.main()

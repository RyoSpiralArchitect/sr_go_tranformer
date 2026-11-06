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


class TokenizerPiecesReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = receipt('tokenizer-pieces-packing-20261106.json')

    def test_standalone_allocation_reduction(self):
        benchmark = self.index['microbenchmark']
        self.assertEqual(benchmark['repetitions'], 3)
        self.assertEqual(benchmark['iterations'], 1000)
        self.assertEqual(len(benchmark['fixtures']), 3)
        for fixture in benchmark['fixtures']:
            for field in ('before_allocs', 'after_allocs', 'before_bytes', 'after_bytes',
                          'before_ns', 'after_ns'):
                self.assertEqual(len(fixture[field]), 3)
                self.assertTrue(all(math.isfinite(x) and x > 0 for x in fixture[field]))
            for old, new in zip(fixture['before_allocs'], fixture['after_allocs']):
                self.assertEqual(old - new, 255)
            for old, new in zip(fixture['before_bytes'], fixture['after_bytes']):
                self.assertLessEqual(abs(old - new), 1)
        self.assertEqual(benchmark['fixtures'][0]['after_allocs'], [2, 2, 2])

    def test_generation_pairs(self):
        group = self.index['generation']
        build = receipt(group['build_receipt'])
        self.assertEqual(build['baseline'], self.index['baseline'])
        self.assertEqual(build['candidate_runtime_sha256'], self.index['candidate_runtime_sha256'])
        self.assertEqual(len(group['receipts']), 3)
        expected = group['approximate_medians']
        for run, name in enumerate(group['receipts']):
            with self.subTest(receipt=name):
                rows = receipt(name)
                self.assertEqual(len(rows), len(expected))
                self.assertEqual({f"{x['prefix']}/{x['outputs']}" for x in rows}, set(expected))
                for row in rows:
                    for field in ('before_ns', 'after_ns', 'ratios'):
                        self.assertEqual(len(row[field]), group['trials'])
                        self.assertTrue(all(math.isfinite(x) and x > 0 for x in row[field]))
                    for old, new, ratio in zip(row['before_ns'], row['after_ns'], row['ratios']):
                        self.assertTrue(math.isclose(new / old, ratio, rel_tol=1e-12))
                    median = statistics.median(row['ratios'])
                    self.assertTrue(math.isclose(median, row['median_ratio'], rel_tol=1e-12))
                    key = f"{row['prefix']}/{row['outputs']}"
                    self.assertEqual(len(expected[key]), 3)
                    self.assertLessEqual(abs(median - expected[key][run]), 0.000500001)


class ByteDecodeReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = receipt('byte-decode-fastpath-20261106.json')

    def test_micro_allocations_and_unfavorable_case(self):
        cases = self.index['selected_microbenchmark']['cases']
        self.assertEqual(len(cases), 3)
        for case in cases:
            for field in ('before_ns', 'after_ns', 'before_bytes', 'after_bytes',
                          'before_allocs', 'after_allocs'):
                self.assertEqual(len(case[field]), 3)
                self.assertTrue(all(math.isfinite(x) and x >= 0 for x in case[field]))
        self.assertEqual(cases[0]['before_allocs'], [8, 8, 8])
        self.assertEqual(cases[0]['after_allocs'], [1, 1, 1])
        self.assertEqual(cases[1]['after_allocs'], [2, 2, 2])
        self.assertEqual(cases[2]['after_allocs'], [0, 0, 0])
        self.assertGreater(statistics.median(cases[2]['after_ns']),
                           statistics.median(cases[2]['before_ns']))

    def test_decode_only_generation(self):
        group = self.index['generation']
        self.assertEqual(len(group['receipts']), 3)
        hashes = None
        for run, name in enumerate(group['receipts']):
            with self.subTest(receipt=name):
                data = receipt(name)
                self.assertEqual(data['Protocol'], group['protocol'])
                self.assertEqual(data['Trials'], group['trials'])
                self.assertEqual(data['Procs'], 4)
                self.assertEqual(data['Hashes']['monolith.go'], self.index['candidate_runtime_sha256'])
                for digest in data['Hashes'].values():
                    self.assertRegex(digest, r'^[0-9a-f]{64}$')
                if hashes is None:
                    hashes = data['Hashes']
                self.assertEqual(data['Hashes'], hashes)
                rows = data['Results']
                expected = group['approximate_medians']
                self.assertEqual(len(rows), 10)
                self.assertEqual({x['Name'] for x in rows}, set(expected))
                for row in rows:
                    for field in ('BeforeNS', 'AfterNS', 'Ratios'):
                        self.assertEqual(len(row[field]), group['trials'])
                        self.assertTrue(all(math.isfinite(x) and x > 0 for x in row[field]))
                    for old, new, ratio in zip(row['BeforeNS'], row['AfterNS'], row['Ratios']):
                        self.assertTrue(math.isclose(new / old, ratio, rel_tol=1e-12))
                    median = statistics.median(row['Ratios'])
                    self.assertTrue(math.isclose(median, row['MedianRatio'], rel_tol=1e-12))
                    self.assertLessEqual(abs(median - expected[row['Name']][run]), 0.000050001)


if __name__ == '__main__':
    unittest.main()

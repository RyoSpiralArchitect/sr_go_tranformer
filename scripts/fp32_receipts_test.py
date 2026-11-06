import hashlib
import json
import math
import statistics
import unittest
from pathlib import Path


class FP32ReceiptTests(unittest.TestCase):
    def test_frozen_receipt_links_and_fingerprints(self):
        root = Path(__file__).resolve().parents[1] / 'validation/performance'
        index = json.loads((root / 'fp32-dispatch-pointer-20261106.json').read_text())
        for protocol in ('paired_protocol', 'generation_protocol'):
            for name, digest in index[protocol]['raw_receipts'].items():
                with self.subTest(name=name):
                    self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
        build = json.loads((root / index['generation_protocol']['build_receipt']).read_text())
        self.assertEqual(build['binary_sha256'], index['generation_protocol']['binary_sha256'])
        self.assertEqual(build['candidate_runtime_sha256'], index['candidate_runtime_sha256'])
        self.assertEqual(build['baseline'], index['baseline_commit'])
        self.assertTrue((root / index['generation_protocol']['separate_process_comparison']).is_file())

    def test_paired_medians_retain_all_trials(self):
        root = Path(__file__).resolve().parents[1] / 'validation/performance'
        for prefix, size, ratio_key in (('fp32-paired-pointer', 15, 'paired_ratios'),
                                        ('fp32-generation-paired', 5, 'ratios')):
            for run in (1, 2, 3):
                rows = json.loads((root / f'{prefix}-run{run}.json').read_text())
                self.assertEqual(len(rows), size)
                identities = [row.get('name', (row.get('prefix'), row.get('outputs'))) for row in rows]
                self.assertEqual(len(set(identities)), size)
                for row in rows:
                    with self.subTest(prefix=prefix, run=run, case=row.get('name', row.get('prefix'))):
                        for key in ('before_ns', 'after_ns', ratio_key):
                            self.assertEqual(len(row[key]), 24)
                            self.assertTrue(all(math.isfinite(v) and v > 0 for v in row[key]))
                        for before, after, ratio in zip(row['before_ns'], row['after_ns'], row[ratio_key]):
                            self.assertAlmostEqual(ratio, after / before, places=12)
                        self.assertAlmostEqual(row['median_ratio'], statistics.median(row[ratio_key]), places=12)


if __name__ == '__main__':
    unittest.main()

# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
import json
import unittest
from pathlib import Path

from assess_q4_prefill import assess


class Q4GateTests(unittest.TestCase):
    # Reuse actual receipt shapes and numerical signatures. Only its fixture
    # identities and threshold are adapted; this is not a new performance run.
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / 'validation/performance'
        self.plan = json.loads((root/'mistral-q4-prefill-protocol-20261101.json').read_text())
        self.p = json.loads((root/'mistral-selective-head-20261030.json').read_text())
        self.d = json.loads((root/'mistral7b-q4mixed-q8-pair-20261029.json').read_text())
        self.p['protocol']['baseline'] = self.d['baseline_commit'] = self.plan['baseline']
        hashes = self.plan['comparison_input_sha256']
        for name in ('before','after'):
            base = {n:hashes[n] for n in ('go.mod','go.sum')}
            base['monolith.go'] = self.plan['runtime_sha256'][name]
            self.p['builds'][name]['inputs_sha256'] = {**base, 'prompt_real_test.go':hashes['prompt_prefill_perf_test.go']}
            self.d['builds'][name]['inputs_sha256'] = {**base, 'inference_perf_test.go':hashes['inference_perf_test.go']}
        self.x = dict(passed=True, sources=self.plan['exact_source_sha256'], tests=self.plan['exact_tests'],
                      modules={n:hashes[n] for n in ('go.mod','go.sum')}, log_sha256='1'*64)

    def check(self):
        return assess(self.plan,self.p,self.d,self.x)

    def test_failed_timing_is_a_rejection(self):
        result = self.check()
        self.assertFalse(result['cases']['long']['passed'])
        self.assertFalse(result['accepted'])

    def test_baseline_against_itself_and_missing_tests_fail_closed(self):
        self.p['builds']['after']['inputs_sha256']['monolith.go'] = self.plan['runtime_sha256']['before']
        with self.assertRaisesRegex(ValueError,'prompt build'):
            self.check()
        self.x['tests'] = []
        with self.assertRaisesRegex(ValueError,'exactness'):
            self.check()

    def test_all_variants_with_the_same_empty_digest_or_bad_durations_are_invalid(self):
        for r in self.d['reports']:
            for c in r['report']['cases']:
                for t in c['trials']:
                    t['kv_sha256'] = ''
        with self.assertRaisesRegex(ValueError,'bit digests'):
            self.check()

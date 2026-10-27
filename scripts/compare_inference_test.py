# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
import json
import copy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from compare_inference import isolated_go_environment, numerical_signature, process_schedule, validate_case_order


class ComparisonProtocolTests(unittest.TestCase):
    def test_balanced_schedule_crosses_variant_and_case_position(self):
        schedule = process_schedule('balanced')
        self.assertEqual([variant for variant, _ in schedule], ['before', 'after', 'after', 'before'])
        for variant in ('before', 'after'):
            self.assertEqual(sorted(order for v, order in schedule if v == variant), [[1, 2], [2, 1]])
        self.assertTrue(all(order == [1, 2] for _, order in process_schedule('fixed')))

    def test_order_validation_rejects_missing_duplicate_or_misreported_cases(self):
        report = dict(case_order=[2, 1], cases=[dict(sessions=2), dict(sessions=1)])
        validate_case_order(report, [2, 1])
        for wrong in ([1, 2], [2], [2, 2]):
            with self.assertRaises(RuntimeError):
                validate_case_order(report, wrong)
        report['cases'].reverse()
        with self.assertRaises(RuntimeError):
            validate_case_order(report, [2, 1])

    def test_numerical_comparison_ignores_order_and_time_but_not_values(self):
        report = dict(plan={'weight_bytes': 123}, threads=4, kernel_workers=4, context=128,
                      prefill_tokens_per_session=8, decode_steps=7, generated_ids_per_session=8,
                      cases=[dict(sessions=n, prompt_ids=[[n]], trials=[dict(
                          logits_sha256='logits'+str(n), kv_sha256='kv'+str(n),
                          token_ids=[[n+1]], workspace_bytes=n*100, decode_seconds=1.0)]) for n in (1, 2)])
        reversed_report = copy.deepcopy(report)
        reversed_report['cases'].reverse()
        reversed_report['cases'][0]['trials'][0]['decode_seconds'] = 99.0
        self.assertEqual(numerical_signature(report), numerical_signature(reversed_report))
        for field, bad in (('logits_sha256', 'bad'), ('kv_sha256', 'bad'),
                           ('token_ids', [[99]]), ('workspace_bytes', 999)):
            changed = copy.deepcopy(reversed_report)
            changed['cases'][0]['trials'][0][field] = bad
            self.assertNotEqual(numerical_signature(report), numerical_signature(changed))


class BuildIsolationTests(unittest.TestCase):
    def test_saved_and_process_go_flags_are_disabled(self):
        go = shutil.which('go')
        if not go:
            fallback = Path(__file__).resolve().parents[1]/'.tools/go/bin/go'
            if fallback.is_file():
                go = str(fallback)
        if not go:
            self.skipTest('Go toolchain unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'go.env'
            config.write_text('GOFLAGS=-gcflags=all=-N\nGOTOOLCHAIN=auto\n')
            # Neither a saved -N nor a process-level -race may silently enter
            # a benchmark binary that claims the ordinary compiler settings.
            with patch.dict(os.environ, {'GOENV': str(config), 'GOFLAGS': '-race',
                                         'GOWORK': '/nonexistent/workspace', 'GOTOOLCHAIN': 'auto'}):
                env = isolated_go_environment()
            got = json.loads(subprocess.check_output(
                [go, 'env', '-json', 'GOENV', 'GOFLAGS', 'GOWORK', 'GOTOOLCHAIN', 'GOPROXY'],
                env=env, text=True, timeout=10))
            self.assertEqual(env['GOENV'], 'off')
            self.assertEqual(got, dict(GOENV='', GOFLAGS='', GOWORK='off',
                                       GOTOOLCHAIN='local', GOPROXY='off'))
            self.assertEqual(config.read_text(), 'GOFLAGS=-gcflags=all=-N\nGOTOOLCHAIN=auto\n')


if __name__ == '__main__':
    unittest.main()

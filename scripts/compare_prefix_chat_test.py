# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
import unittest

from compare_prefix_chat import check_report, fixed_control


class ChatComparisonTests(unittest.TestCase):
    def test_control_replaces_only_the_pinned_capture_function(self):
        def src(guard):
            return b'prefix\nfunc (p *prefixSnapshot) capture() {\n' + guard + b'\n}\nsuffix'
        old, grow = src(b'old'), src(b'grow')
        production = grow + b' independently optimized runtime'
        self.assertEqual(fixed_control(production, old, grow), old + b' independently optimized runtime')
        with self.assertRaises(ValueError):
            fixed_control(src(b'changed'), old, grow)

    def test_claimed_growth_requires_matching_prompt_ids(self):
        # Five monotonically growing prompts; the unforwarded output token has
        # no KV position. Divergence at the first ID invalidates the claimed hit.
        report = dict(mode='growing', threads=4, plan={'weight_format':'q4-mixed'},
                      questions=['q']*5, sampling={}, turns=[])
        for i in range(5):
            n = 32*(i+1) + 1
            ids = list(range(n))
            reused = 32*i
            report['turns'].append(dict(prompt_ids=ids, computed_input_tokens=n-reused,
                reused_input_tokens=reused, retained_prefix_bytes=(i+1)*(8 << 20),
                allocated_kv_bytes=(i+2)*(8 << 20), first_token_seconds=1, completion_seconds=1.5,
                completion=dict(prompt_tokens=n, completion_tokens=1, token_ids=[123])))
        check_report(report, 'growing')
        report['turns'][2]['prompt_ids'][0] = 456
        with self.assertRaisesRegex(ValueError, 'prefix hit'):
            check_report(report, 'growing')

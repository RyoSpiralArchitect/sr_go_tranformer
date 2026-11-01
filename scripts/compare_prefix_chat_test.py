# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
import unittest

from compare_prefix_chat import check_report, fixed_control


class ChatComparisonTests(unittest.TestCase):
    def test_control_replaces_only_the_pinned_capture_function(self):
        def src(guard):
            return b"prefix\nfunc (p *prefixSnapshot) capture() {\n" + guard + b"\n}\nsuffix"

        old, grow = src(b"old"), src(b"grow")
        production = grow + b" independently optimized runtime"
        self.assertEqual(
            fixed_control(production, old, grow), old + b" independently optimized runtime"
        )
        with self.assertRaises(ValueError):
            fixed_control(src(b"changed"), old, grow)

    def test_claimed_growth_requires_matching_prompt_ids(self):
        # Five monotonically growing prompts; the unforwarded output token has
        # no KV position. Divergence at the first ID invalidates the claimed hit.
        report = dict(
            mode="growing",
            build_policy="growing",
            threads=4,
            plan={"weight_format": "q4-mixed"},
            questions=["q"] * 5,
            sampling={},
            turns=[],
        )
        for i in range(5):
            n = 32 * (i + 1) + 1
            ids = list(range(n))
            reused = 32 * i
            report["turns"].append(
                dict(
                    prompt_ids=ids,
                    computed_input_tokens=n - reused,
                    reused_input_tokens=reused,
                    retained_prefix_bytes=(i + 1) * (8 << 20),
                    allocated_kv_bytes=(i + 2) * (8 << 20),
                    first_token_seconds=1,
                    completion_seconds=1.5,
                    stream_ids=[123],
                    stream_text="cat",
                    completion=dict(prompt_tokens=n, completion_tokens=1, token_ids=[123], text="cat"),
                )
            )
        check_report(report, "growing")
        report["turns"][2]["prompt_ids"][0] = 456
        with self.assertRaisesRegex(ValueError, "prefix hit"):
            check_report(report, "growing")

    def test_unforwarded_token_cannot_complete_a_page_and_fixed_hits_do_not_extend(self):
        report = dict(
            mode="fixed",
            build_policy="fixed",
            threads=4,
            plan={"weight_format": "q4-mixed"},
            questions=["q"] * 5,
            sampling={},
            turns=[],
        )
        for n, reused, pages in [(31, 0, 0), (33, 0, 1), (65, 32, 1), (97, 32, 1), (129, 32, 1)]:
            report["turns"].append(
                dict(
                    prompt_ids=list(range(n)),
                    computed_input_tokens=n - reused,
                    reused_input_tokens=reused,
                    retained_prefix_bytes=pages * (8 << 20),
                    allocated_kv_bytes=(n // 32 + 2) * (8 << 20),
                    first_token_seconds=0.1,
                    completion_seconds=0.2,
                    stream_ids=[4],
                    stream_text="x",
                    completion=dict(token_ids=[4], prompt_tokens=n, completion_tokens=1, text="x"),
                )
            )
        check_report(report, "fixed")
        report["turns"][0]["retained_prefix_bytes"] = 8 << 20
        with self.assertRaisesRegex(ValueError, "snapshot payload"):
            check_report(report, "fixed")

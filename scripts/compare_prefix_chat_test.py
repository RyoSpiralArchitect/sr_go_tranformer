# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
import unittest

from compare_prefix_chat import check_report, fixed_control, tail_result


class ChatComparisonTests(unittest.TestCase):
    def test_control_replaces_only_the_pinned_capture_guard(self):
        def src(guard):
            return b"prefix\nfunc (p *prefixSnapshot) capture() {\n" + guard + b"\n\t}\n\tstore()\n}\nsuffix"

        old, grow = src(b"old"), src(b"grow")
        production = grow.replace(b"store()", b"newStore()")
        self.assertEqual(
            fixed_control(production, old, grow), old.replace(b"store()", b"newStore()")
        )
        with self.assertRaises(ValueError):
            fixed_control(src(b"changed"), old, grow)
        with self.assertRaises(ValueError):
            fixed_control(production, old.replace(b"store()", b"different()"), grow)

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

    def test_partial_hits_stop_at_divergence_and_the_last_forwarded_position(self):
        report = dict(mode="tail", build_policy="growing", tail_enabled=True, threads=4,
                      plan={"weight_format": "q4-mixed"}, questions=["q"] * 5, sampling={}, turns=[])
        prompts = [list(range(n)) for n in (31, 32, 63, 64, 70)]
        prompts[3][35] = 200  # Diverge within the second page twice.
        for ids, reused, pages in zip(prompts, (0, 31, 32, 35, 35), (1, 1, 2, 2, 3)):
            report["turns"].append(dict(
                prompt_ids=ids, computed_input_tokens=len(ids) - reused, reused_input_tokens=reused,
                retained_prefix_bytes=pages << 23, allocated_kv_bytes=(pages + 2) << 23,
                first_token_seconds=1, completion_seconds=2, stream_ids=[123], stream_text="cat",
                completion=dict(token_ids=[123], text="cat", completion_tokens=1, prompt_tokens=len(ids))))
        check_report(report, "tail")
        report["turns"][3].update(reused_input_tokens=36, computed_input_tokens=28)
        with self.assertRaisesRegex(ValueError, "prefix hit"):
            check_report(report, "tail")

    def test_tail_decision_excludes_cold_turn_and_uses_process_totals(self):
        reports = []
        for mode, tts in [("growing", [1, 1, 2, 3, 4]), ("tail", [200, 1, 2, 2, 4]),
                          ("tail", [200, 1, 2, 3, 3]), ("growing", [1, 1, 2, 3, 4])]:
            reports.append(dict(mode=mode, report=dict(turns=[{"first_token_seconds": x} for x in tts])))
        rows = {m: [dict(computed_input_tokens=n, reused_input_tokens=100-n, allocated_kv_bytes=b)]
                for m, n, b in [("growing", 80, 56 << 20), ("tail", 60, 64 << 20)]}
        result = tail_result(reports, rows)
        self.assertEqual(result["checks"], dict(computed=True, kv=True, warm_ttft=True))
        reports[1]["report"]["turns"][2]["first_token_seconds"] += 5
        rows["tail"][0]["allocated_kv_bytes"] += 1
        rows["tail"][0]["computed_input_tokens"] += 1
        self.assertEqual(tail_result(reports, rows)["checks"], dict(computed=False, kv=False, warm_ttft=False))

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

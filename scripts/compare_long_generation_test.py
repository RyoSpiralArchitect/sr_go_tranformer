# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
import copy
import unittest

from compare_long_generation import signature, validate_report


def fixture():
    report = dict(plan={'weight_format':'q4-mixed'}, threads=4, kernel_workers=4,
                  context=1024, prefill_chunk=16, case_order=['short','long'], cases=[])
    for name, prefix, outputs, count in [('short',8,8,3), ('long',256,256,1)]:
        trial = dict(token_ids=[1]*outputs, decode_step_seconds=[.1]*(outputs-1),
                     logits_sha256='logits', kv_sha256='kv', kv_position=prefix+outputs-1,
                     kv_bytes=2048, workspace_bytes=4096, score_bytes=1024,
                     score_reallocations=0, decode_seconds=1.)
        report['cases'].append(dict(name=name, prompt_ids=[1]*prefix, output_ids=outputs,
                                    trials=[copy.deepcopy(trial) for _ in range(count)]))
    return report


class LongGenerationContractTests(unittest.TestCase):
    def test_only_score_scratch_and_measurements_may_differ(self):
        before=fixture();after=copy.deepcopy(before)
        after['cases'].reverse()
        for c in after['cases']:
            for t in c['trials']:
                t['score_bytes']+=512;t['workspace_bytes']+=512
                t['decode_seconds']=99.;t['score_reallocations']=1
        self.assertEqual(signature(before),signature(after))
        for key,bad in [('logits_sha256','bad'),('kv_sha256','bad'),('token_ids',[3]),
                        ('kv_bytes',99),('kv_position',99),('workspace_bytes',999)]:
            wrong=copy.deepcopy(after);wrong['cases'][0]['trials'][0][key]=bad
            self.assertNotEqual(signature(before),signature(wrong))

    def test_cannot_pass_shortened_long_decode_or_wrong_order(self):
        good=fixture();validate_report(good,['short','long'],'after')
        for field in ('token_ids','decode_step_seconds'):
            bad=copy.deepcopy(good);bad['cases'][1]['trials'][0][field].pop()
            with self.assertRaises(RuntimeError):validate_report(bad,['short','long'],'after')
        with self.assertRaises(RuntimeError):validate_report(good,['long','short'],'after')
        bad=copy.deepcopy(good);bad['cases'][1]['trials']=[]
        with self.assertRaises(RuntimeError):validate_report(bad,['short','long'],'after')

    def test_candidate_scratch_and_growth_are_bounded(self):
        for key,bad_value in [('score_bytes',4*1024*4+4),('score_reallocations',9)]:
            report=fixture();report['cases'][1]['trials'][0][key]=bad_value
            with self.assertRaises(RuntimeError):validate_report(report,['short','long'],'after')
        for key,value in [('context',2048),('plan',{'weight_format':'q8'})]:
            report=fixture();report[key]=value
            with self.assertRaises(RuntimeError):validate_report(report,['short','long'],'after')


if __name__=='__main__':unittest.main()

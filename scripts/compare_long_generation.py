#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Bounded offline comparison of a continuous long decode and short control.

Uses the same harness against both sources in ABBA order, with case order
reversed in the second pair. No model download. The long case continues at EOS;
this is a fixed-step core-forward probe, not HTTP throughput or language quality.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess

from compare_inference import isolated_go_environment
from quant_quality import run_measured, sha


def signature(report):
    # Score scratch deliberately changes layout. Require the exact same model,
    # prompt, output/KV bits, KV allocation and all other workspace capacity.
    return dict(plan=report['plan'], threads=report['threads'],
                kernel_workers=report['kernel_workers'], context=report['context'],
                prefill_chunk=report['prefill_chunk'], cases=[dict(
                    name=c['name'], prompt_ids=c['prompt_ids'], output_ids=c['output_ids'],
                    trials=[dict(**{k:t[k] for k in ('token_ids','logits_sha256','kv_sha256','kv_position','kv_bytes')},
                                 non_score_workspace_bytes=t['workspace_bytes']-t['score_bytes'])
                            for t in c['trials']])
                    for c in sorted(report['cases'],key=lambda c:c['name'])])


def validate_report(report, order, variant):
    if report['case_order'] != order or [c['name'] for c in report['cases']] != order:
        raise RuntimeError('Unexpected case order')
    if report['context'] != 1024 or report['plan']['weight_format'] != 'q4-mixed':
        raise RuntimeError('Unexpected model settings')
    for c in report['cases']:
        prefix, outputs, trials = (8,8,3) if c['name']=='short' else (256,256,1)
        if len(c['prompt_ids']) != prefix or c['output_ids'] != outputs or len(c['trials']) != trials:
            raise RuntimeError('Unexpected workload')
        for t in c['trials']:
            if len(t['token_ids']) != outputs or len(t['decode_step_seconds']) != outputs-1 or t['kv_position'] != prefix+outputs-1:
                raise RuntimeError('Incomplete trajectory')
            if variant=='after' and (t['score_bytes']>4*1024*4 or t['score_reallocations']>(outputs-1+31)//32):
                raise RuntimeError('Score scratch or growth exceeds frozen bound')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--id',required=True)
    parser.add_argument('--baseline-ref',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    model,output=args.model.resolve(),args.output_dir.resolve()
    baseline=subprocess.check_output(['git','rev-parse','--verify','--end-of-options',args.baseline_ref+'^{commit}'],cwd=root,text=True).strip()
    catalog=root/'validation/hf/model-profiles.json'
    source=next(x for x in json.loads(catalog.read_text())['models'] if x['id']==args.id)
    for name,expected in source['files'].items():
        if sha(model/name)!=expected: raise RuntimeError('Pinned source mismatch: '+name)
    output.mkdir(parents=True,exist_ok=False)
    inputs=['monolith.go','go.mod','go.sum','inference_perf_test.go','long_generation_perf_test.go',
            'scripts/compare_long_generation.py','scripts/compare_inference.py','scripts/quant_quality.py',
            'scripts/go.sh','validation/quality/corpus.json','validation/hf/model-profiles.json']
    hashes={name:sha(root/name) for name in inputs}
    env=isolated_go_environment()
    env.update(MONOLITH_PERF_MODEL=str(model),MONOLITH_PERF_CORPUS=str(root/'validation/quality/corpus.json'))
    goroot=subprocess.check_output([str(root/'scripts/go.sh'),'env','GOROOT'],cwd=root,env=env,text=True).strip()
    go=str(Path(goroot)/'bin/go')
    effective=json.loads(subprocess.check_output([go,'env','-json','GOENV','GOFLAGS','GOWORK','GOTOOLCHAIN','GOEXPERIMENT','GOPROXY','GOSUMDB','GOOS','GOARCH'],cwd=root,env=env,text=True))
    expected=dict(env,GOENV='')
    if any(effective[k]!=expected[k] for k in effective if k not in ('GOOS','GOARCH')):
        raise RuntimeError('Unexpected effective Go environment')
    builds={}
    for variant in ('before','after'):
        build=output/variant;build.mkdir()
        for name in ('monolith.go','go.mod','go.sum'):
            data=subprocess.check_output(['git','show',baseline+':'+name],cwd=root) if variant=='before' else (root/name).read_bytes()
            (build/name).write_bytes(data)
        for name in ('inference_perf_test.go','long_generation_perf_test.go'):
            (build/name).write_bytes((root/name).read_bytes())
        binary=build/'probe.test'
        with (build/'build.log').open('w') as log:
            subprocess.run([go,'test','-c','-trimpath','-o',str(binary),'.'],cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        builds[variant]=dict(binary_sha256=sha(binary),inputs_sha256={p.name:sha(p) for p in build.iterdir() if p.suffix=='.go' or p.name in ('go.mod','go.sum')})
    reports=[];reference=None
    for index,variant in enumerate(('before','after','after','before')):
        order=['short','long'] if index<2 else ['long','short']
        result=output/f'{index}-{variant}.json'
        env.update(MONOLITH_PERF_OUTPUT=str(result),MONOLITH_LONG_CASE_ORDER=','.join(order))
        with (output/f'{index}-{variant}.log').open('w') as log:
            resources=run_measured([str(output/variant/'probe.test'),'-test.run=^TestHFLongGenerationPerformance$','-test.v','-test.timeout=19m'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        report=json.loads(result.read_text());validate_report(report,order,variant)
        current=signature(report)
        if reference is None: reference=current
        elif reference!=current: raise RuntimeError('Changed numerical signature, KV allocation or non-score workspace')
        reports.append(dict(variant=variant,process_index=index,report=report,process_resources=resources))
        print(index,variant,'complete; identical numerical signature',flush=True)
    if hashes!={name:sha(root/name) for name in inputs}: raise RuntimeError('Source changed during run')
    for name,expected in source['files'].items():
        if sha(model/name)!=expected: raise RuntimeError('Model changed during run: '+name)
    for variant,manifest in builds.items():
        if sha(output/variant/'probe.test')!=manifest['binary_sha256'] or {n:sha(output/variant/n) for n in manifest['inputs_sha256']}!=manifest['inputs_sha256']:
            raise RuntimeError('Build changed during run')
    receipt=dict(format='monolith-long-generation-comparison-v1',baseline_commit=baseline,
                 model=source['repo'],revision=source['revision'],files=source['files'],
                 host=platform.system()+'/'+platform.machine(),go_version=subprocess.check_output([go,'version'],env=env,text=True).strip(),
                 hf_hub_offline=True,effective_go_env=effective,builds=builds,source_inputs_sha256=hashes,
                 numerical_identity=True,non_score_workspace_identity=True,reports=reports,
                 note='ABBA, reversed case order in second pair. Long: 256-token prefix then 256 output IDs (255 decode calls), one trajectory per process; short: 8-prefix/8-ID, warmup plus three trials. Core intervals only, fixed-step continues at EOS. Score scratch may change; all other workspace and KV allocations must match.')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__': main()

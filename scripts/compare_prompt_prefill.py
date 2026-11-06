#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
import argparse,json,subprocess,sys,statistics,platform
from pathlib import Path
from quant_quality import git_source, source_path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--model',type=Path,required=True);p.add_argument('--harness',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',default='68ee4d8406b907c6bf6c77fdd364146da4aac7cf');p.add_argument('--long-ratio-max',type=float,default=.95);a=p.parse_args()
if not 0 < a.long_ratio_max <= 2:p.error('--long-ratio-max must be in (0, 2]')
root=a.root.resolve();a.harness=a.harness or source_path(root, 'prompt_prefill_perf_test.go');out=a.output.resolve();out.mkdir(exist_ok=False,parents=True)
sys.path.insert(0,str(root/'scripts'))
from quant_quality import sha,run_measured
from compare_inference import isolated_go_environment, write_parity_receipt
env=isolated_go_environment()
goroot=subprocess.check_output([str(root/'scripts/go.sh'),'env','GOROOT'],cwd=root,env=env,text=True).strip()
go=str(Path(goroot)/'bin/go');baseline=subprocess.check_output(['git','rev-parse','--verify',a.baseline+'^{commit}'],cwd=root,text=True).strip()
protocol={'format':'monolith-prompt-first-token-protocol-v1','baseline':baseline,'order':['before','after','after','before'],'case_orders':['short,long','short,long','long,short','long,short'],'model':'mistral-7b-v03','context':1024,'weight_format':'q4-mixed','threads':4,'long_trials_per_process':1,'short_trials_per_process':3,'warmups':0,'gate':{'long_median_ratio_max':0.95,'short_median_ratio_max':1.05,'exact_completion':True},'scope':'Generate to first output token, excludes load, includes prompt encoding and sampling. Two long trials per variant, no confidence intervals. Retain all attempts.'}
protocol['gate']['long_median_ratio_max']=a.long_ratio_max
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
source=next(x for x in json.loads((root/'validation/hf/model-profiles.json').read_text())['models'] if x['id']=='mistral-7b-v03')
def verify_model():
 for n,h in source['files'].items():
  if sha(a.model/n)!=h:raise RuntimeError('model mismatch '+n)
verify_model();builds={};inputs={n:sha(source_path(root, n)) for n in ['monolith.go','go.mod','go.sum']}
for variant in ('before','after'):
 d=out/variant;d.mkdir()
 for n in inputs:(d/n).write_bytes(git_source(root, baseline, n) if variant=='before' else source_path(root, n).read_bytes())
 (d/'prompt_real_test.go').write_bytes(a.harness.read_bytes());binary=d/'probe.test'
 with (d/'build.log').open('w') as log:subprocess.run([go,'test','-c','-trimpath','-o',str(binary),'.'],cwd=d,env=env,stdout=log,stderr=log,check=True)
 builds[variant]={'binary_sha256':sha(binary),'inputs_sha256':{n:sha(d/n) for n in list(inputs)+['prompt_real_test.go']}}
reports=[];reference=None;matched=True
for i,variant in enumerate(protocol['order']):
 result=out/f'{i}-{variant}.json';env.update(MONOLITH_STACK_MODEL=str(a.model),MONOLITH_STACK_OUTPUT=str(result),MONOLITH_STACK_ORDER=protocol['case_orders'][i])
 with (out/f'{i}-{variant}.log').open('w') as log:resources=run_measured([str(out/variant/'probe.test'),'-test.run=^TestHFPromptStackProbe$','-test.v','-test.timeout=13m'],timeout=800,cwd=root,env=env,stdout=log,stderr=log)
 r=json.loads(result.read_text());assert [x['name'] for x in r['cases']]==protocol['case_orders'][i].split(',')
 sig={x['name']:[t['completion'] for t in x['trials']] for x in r['cases']}
 assert all(len(sig[n])==(3 if n=='short' else 1) for n in sig)
 if reference is None:reference=sig
 same=sig==reference;matched=matched and same
 reports.append({'variant':variant,'report':r,'resources':resources});print(i,variant,'complete; completion '+('matches' if same else 'differs'),flush=True)
verify_model();assert inputs=={n:sha(source_path(root, n)) for n in inputs}
for v,b in builds.items():
 assert sha(out/v/'probe.test')==b['binary_sha256']
 assert b['inputs_sha256']=={n:sha(out/v/n) for n in b['inputs_sha256']}
summary={}
for case in ('short','long'):
 d={v:statistics.median(t['seconds'] for r in reports if r['variant']==v for c in r['report']['cases'] if c['name']==case for t in c['trials']) for v in ('before','after')};d['ratio']=d['after']/d['before'];summary[case]=d
checks={'identity':matched,'long':summary['long']['ratio']<=protocol['gate']['long_median_ratio_max'],'short':summary['short']['ratio']<=protocol['gate']['short_median_ratio_max']}
r={'format':'monolith-prompt-first-token-result-v1','protocol':protocol,'host':platform.system()+'/'+platform.machine(),'go_version':subprocess.check_output([go,'version'],env=env,text=True).strip(),'model':source['repo'],'revision':source['revision'],'files':source['files'],'builds':builds,'source_sha256':inputs,'reports':reports,'summary':summary,'checks':checks,'accepted':all(checks.values())}
write_parity_receipt(out/'receipt.json', r, matched);print(json.dumps({'summary':summary,'checks':checks}),flush=True)

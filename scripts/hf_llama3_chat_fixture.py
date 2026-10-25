#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Pinned Llama 3.2 Instruct text-only chat oracle; never executes model code."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--real', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = next(x for x in json.loads((root/'validation/hf/model-profiles.json').read_text())['models'] if x['id'] == 'llama32-1b-instruct')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        if hashlib.sha256((a.source/name).read_bytes()).hexdigest() != source['files'][name]:
            raise RuntimeError('Pinned source mismatch: '+name)
    spec = importlib.util.spec_from_file_location('reference',Path(__file__).with_name('hf_tokenizer_reference.py'))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    if reference.tokenizers.__version__ != '0.22.2':
        raise RuntimeError('Require tokenizers 0.22.2')
    folder, output = a.source, a.output
    if not a.real:
        folder,output=a.output,a.output/'reference.json'
        folder.mkdir(parents=True,exist_ok=True)
        base=root/'testdata/hf-llama3-tokenizer'
        tok=json.loads((base/'tokenizer.json').read_text())
        cfg=json.loads((a.source/'tokenizer_config.json').read_text())
        cfg['model_max_length']=512
        cfg['added_tokens_decoder']={str(t['id']):{k:v for k,v in t.items() if k!='id'} for t in tok['added_tokens']}
        ids={x['content']:x['id'] for x in tok['added_tokens']}
        h=json.loads((base/'config.json').read_text())
        h['eos_token_id']=[ids[s] for s in ('<|end_of_text|>','<|eom_id|>','<|eot_id|>')]
        for name,data in (('config.json',h),('tokenizer.json',tok),('tokenizer_config.json',cfg)):
            (folder/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    extra_chats=[
        [dict(role='system',content=' \x1cBe brief.\x1f\n'),dict(role='user',content='\t 猫とGo \u0085')],
        [dict(role='user',content=''),dict(role='assistant',content='  '),dict(role='user',content='Again.')],
        [dict(role='system',content='Only a system message.')],
        [dict(role='assistant',content='Hello'),dict(role='user',content='Hi'),dict(role='system',content='A later system message.')],
    ]
    reference.oracle(folder,output,add_special_tokens=True,chat_date='25 Oct 2026',extra_chats=extra_chats)


if __name__ == '__main__':
    main()

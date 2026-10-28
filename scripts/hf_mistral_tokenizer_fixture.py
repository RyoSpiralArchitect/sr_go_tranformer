#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Generate a seeded Mistral v3 tokenizer fixture or validate pinned real metadata.

Synthetic vocabulary is trained on authored text; the pinned Apache-2.0 chat
and pipeline metadata are retained. --real also checks default AutoTokenizer
(SentencePiece conversion) against the serialized tokenizers.json pipeline.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import tokenizers
import transformers
from tokenizers import AddedToken, Tokenizer, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--real', action='store_true')
    a = p.parse_args()
    if tokenizers.__version__ != '0.22.2' or transformers.__version__ != '4.57.1':
        raise RuntimeError('Require tokenizers 0.22.2 / transformers 4.57.1')
    root = Path(__file__).resolve().parents[1]
    profile = next(x for x in json.loads((root/'validation/hf/model-profiles.json').read_text())['models'] if x['id'] == 'mistral-7b-v03')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        if hashlib.sha256((a.source/name).read_bytes()).hexdigest() != profile['files'][name]:
            raise RuntimeError('Pinned source mismatch: ' + name)
    spec = importlib.util.spec_from_file_location('reference', Path(__file__).with_name('hf_tokenizer_reference.py'))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    cfg = json.loads((a.source/'tokenizer_config.json').read_text())
    raw = json.loads((a.source/'tokenizer.json').read_text())
    extra = ['<s>hello', '[INST]hello', 'a[INST]hello', '<s> hello', '[INST] hello',
             'hi</s>there', '▁hello', 'a▁b', '🌀\U000e0001', '\ttext', '<unk>',
             '<s><s>', '[control_768]a', 'a'*1024, ' '*64]
    chats = [[dict(role='system', content=' sys '), dict(role='user', content=' first '),
              dict(role='assistant', content=' \x1c ok \x1f '), dict(role='user', content=' next ')],
             [dict(role='system', content='system only')],
             [dict(role='user', content='A'), dict(role='assistant', content='B')],
             [dict(role='user', content='[INST]literal<s> 🐈‍⬛')]]
    if a.real:
        if hashlib.sha256((a.source/"tokenizer.model").read_bytes()).hexdigest() != profile["reference_files"]["tokenizer.model"]:
            raise RuntimeError("Pinned SentencePiece model mismatch")
        folder, output = a.source, a.output
        tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        backend = Tokenizer.from_file(str(folder/'tokenizer.json'))
        for text in reference.texts()+extra:
            if tok.encode(text) != backend.encode(text).ids:
                raise RuntimeError('Default AutoTokenizer/serialized pipeline mismatch: '+repr(text))
    else:
        added, post = raw['added_tokens'], raw['post_processor']
        raw['model'].update(vocab={}, merges=[])
        raw['added_tokens'], raw['post_processor'] = [], None
        backend = Tokenizer.from_str(json.dumps(raw))
        backend.train_from_iterator(reference.texts(), trainers.BpeTrainer(
            vocab_size=1300, min_frequency=1, initial_alphabet=list('abcdefghijklmnopqrstuvwxyz▁'),
            special_tokens=[AddedToken(x['content'], special=True, normalized=False) for x in added],
            show_progress=False))
        raw = json.loads(backend.to_str())
        for b in range(256):
            raw['model']['vocab'][f'<0x{b:02X}>'] = len(raw['model']['vocab'])
        raw['post_processor'] = post
        cfg['added_tokens_decoder'] = {str(x['id']): {k:v for k,v in x.items() if k != 'id'} for x in raw['added_tokens']}
        h = json.loads((a.source/'config.json').read_text())
        h.update(vocab_size=len(raw['model']['vocab']), hidden_size=24, intermediate_size=40,
                 num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
                 max_position_embeddings=512)
        folder, output = a.output, a.output/'reference.json'
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in (('tokenizer.json',raw),('tokenizer_config.json',cfg),('config.json',h)):
            (folder/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        # No synthetic SentencePiece model is fabricated. Use the serialized
        # pipeline directly, and separately check the official conversion in --real.
        tok = PreTrainedTokenizerFast(tokenizer_file=str(folder/'tokenizer.json'),
                                     bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                     chat_template=cfg['chat_template'], clean_up_tokenization_spaces=False)
    reference.oracle(folder, output, add_special_tokens=True, extra_texts=extra, extra_chats=chats, tokenizer=tok)
    report = json.loads(output.read_text())
    vocab = json.loads((folder/'tokenizer.json').read_text())['model']['vocab']
    sequences = [[65,255],[240,159],[65,240,159],[240,159,144,136],[255,65],[32],[32,32],[]]
    decode = [[vocab[f'<0x{b:02X}>'] for b in bs] for bs in sequences]
    decode += [[vocab['<0xF0>'],vocab['<0x9F>'],vocab['a'],vocab['<0x90>'],vocab['<0x88>']],
               [vocab['<0xF0>'],vocab['<0x9F>'],1,vocab['<0x90>'],vocab['<0x88>']],
               [1,vocab['▁'],vocab['▁'],vocab['a']]]
    report['decode_cases'] += [dict(ids=ids,decoded=tok.decode(ids,skip_special_tokens=True)) for ids in decode]
    report['transformers'] = transformers.__version__
    report['default_auto_tokenizer_checked'] = a.real
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()

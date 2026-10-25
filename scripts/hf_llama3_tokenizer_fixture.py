#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Generate a small Llama 3 tokenizer oracle, or check a pinned real tokenizer.

The synthetic vocabulary is trained on authored text, not copied model weights.
Only the exact pinned tokenizer pipeline/metadata is reused. No repository code
is executed. Run in the isolated hf_reference_requirements environment.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Tokenizer, trainers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--real', action='store_true', help='write an oracle for the unchanged source tokenizer')
    a = p.parse_args()
    if tokenizers.__version__ != '0.22.2':
        raise RuntimeError('Require tokenizers 0.22.2')
    root = Path(__file__).resolve().parents[1]
    source = next(x for x in json.loads((root/'validation/hf/model-profiles.json').read_text())['models'] if x['id'] == 'llama32-3b')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        if hashlib.sha256((a.source/name).read_bytes()).hexdigest() != source['files'][name]:
            raise RuntimeError('Pinned source mismatch: ' + name)
    spec = importlib.util.spec_from_file_location('reference', Path(__file__).with_name('hf_tokenizer_reference.py'))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    extra = ['<|begin_of_text|>Hello', '<|eot_id|><|end_of_text|>',
             '1234567 １２３４５６７ ١٢٣٤٥٦٧ Ⅳ½²Ⅵ', 'inference',
             "Hello . I 'm here , are n't you ?", 'e\u0301 café',
             "'S 'T 'RE 'VE 'M 'LL 'D 'ſ 'K"]
    if a.real:
        reference.oracle(a.source, a.output, add_special_tokens=True, extra_texts=extra)
        return
    raw = json.loads((a.source/'tokenizer.json').read_text())
    cfg = json.loads((a.source/'tokenizer_config.json').read_text())
    added = raw['added_tokens']
    post = raw['post_processor']
    raw['model'].update(vocab={}, merges=[])
    raw['added_tokens'] = []
    raw['post_processor'] = None
    tok = Tokenizer.from_str(json.dumps(raw))
    tok.train_from_iterator(reference.texts(), trainers.BpeTrainer(
        vocab_size=384, min_frequency=1,
        initial_alphabet=sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()), show_progress=False))
    raw = json.loads(tok.to_str())
    # An unreachable whole word proves ignore_merges is active, not just normal BPE.
    raw['model']['vocab']['inference'] = len(raw['model']['vocab'])
    tok = Tokenizer.from_str(json.dumps(raw))
    tok.add_special_tokens([AddedToken(x['content'], special=True, normalized=False) for x in added])
    raw = json.loads(tok.to_str())
    bos = tok.token_to_id('<|begin_of_text|>')
    post['processors'][1]['special_tokens']['<|begin_of_text|>']['ids'] = [bos]
    raw['post_processor'] = post
    cfg['added_tokens_decoder'] = {str(t['id']): {k:v for k,v in t.items() if k != 'id'} for t in raw['added_tokens']}
    cfg['model_max_length'] = 512
    h = json.loads((a.source/'config.json').read_text())
    h.update(vocab_size=tok.get_vocab_size(), hidden_size=24, intermediate_size=40,
             num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
             head_dim=4, max_position_embeddings=512, bos_token_id=bos,
             eos_token_id=tok.token_to_id('<|end_of_text|>'))
    h['rope_scaling']['original_max_position_embeddings'] = 128
    a.output.mkdir(parents=True, exist_ok=True)
    for name, data in (('tokenizer.json',raw), ('tokenizer_config.json',cfg), ('config.json',h)):
        (a.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    reference.oracle(a.output, a.output/'reference.json', add_special_tokens=True, extra_texts=extra)


if __name__ == '__main__':
    main()

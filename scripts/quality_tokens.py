#!/usr/bin/env python3
# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned external token IDs, or decode a completed Go quality receipt.

Experimental token-ID inference only. This does not add a Go text tokenizer.
Use the isolated reference environment; no model repository code is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import tokenizers


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, help='decode existing Go completions instead of preparing input IDs')
    a = p.parse_args()
    if tokenizers.__version__ != '0.22.2':
        raise RuntimeError('Reference requires tokenizers 0.22.2')
    root = Path(__file__).resolve().parents[1]
    source = next(x for x in json.loads((root/'validation/hf/model-profiles.json').read_text())['models'] if x['id'] == a.id)
    tokenizer_path = a.model/'tokenizer.json'
    if sha(tokenizer_path) != source['files']['tokenizer.json']:
        raise RuntimeError('Pinned tokenizer mismatch')
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    corpus = root/'validation/quality/corpus.json'
    data = json.loads(corpus.read_text())
    if a.receipt:
        r = json.loads(a.receipt.read_text())
        if r['model'] != source['repo'] or r['revision'] != source['revision'] or r['files']['tokenizer.json'] != source['files']['tokenizer.json']:
            raise RuntimeError('Receipt model/tokenizer mismatch')
        result = dict(format='monolith-quality-external-text-v1', receipt_sha256=sha(a.receipt),
            tokenizer_sha256=sha(tokenizer_path), tokenizers_version=tokenizers.__version__,
            decode='backend Tokenizer.decode(skip_special_tokens=True); no Transformers whitespace cleanup',
            completions={fmt: [dict(token_ids=c['token_ids'], text=tokenizer.decode(c['token_ids'], skip_special_tokens=True))
                               for c in report['completions']] for fmt, report in r['reports'].items()})
    else:
        result = dict(format='monolith-quality-tokens-v1', source=source['repo']+'@'+source['revision'],
            corpus_sha256=sha(corpus), tokenizer_sha256=sha(tokenizer_path), tokenizers_version=tokenizers.__version__,
            document_ids=[tokenizer.encode(x['text'], add_special_tokens=True).ids for x in data['evaluation_documents']],
            prompt_ids=[tokenizer.encode(x, add_special_tokens=True).ids for x in data['generation_prompts']])
    # Refuse to replace an input or receipt accidentally.
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()

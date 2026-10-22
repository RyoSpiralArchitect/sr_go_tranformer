#!/usr/bin/env python3
"""Generate offline tokenizer/chat oracles with tokenizers 0.22.2."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import tokenizers
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def texts():
    out = ["", "Hello, world!", "  hello   world  ", "a \n b", "\t\tword\n\n",
           "can't won't I'm I'll she'd they're we've I'M WE'RE",
           "Goの並行処理🐈‍⬛🌀", "café e\u0301 Ελληνικά العربية",
           " 12345.67 １２３ ٣٤ Ⅳ ½ ²", "a\u00a0\u2009b\u2028c",
           "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
           "<|im_start|><|im_start|><|im_end|>", "x\x00\x04\x06\x13y",
           "\U00040000\U00080000 \U0010ffff", "👩🏽‍💻", " \r\n\t   a   ",
           "a"*257, "    ", "<|im_start|>x<|im_end|>"*80]
    rng = random.Random(135)
    alphabet = ["a", "Z", "café", "猫", "🌀", "\u0301", " ", "\n", "\t",
                "123", "Ⅳ", "½", "١٢", "?!", "'re", "'s", "\u00a0",
                "<|im_start|>", "<|im_end|>", " e", "\U00040000", "\x04"]
    out += ["".join(rng.choices(alphabet, k=rng.randrange(1, 30))) for _ in range(128)]
    out += ["e\u0301 A\u030a \u212b \u1100\u1161", "I'M WE'RE She's 'RE 'ſ",
            "!hello 👋word \tword\nword", " \n  \r\n\t text",
            "123456 １２３ ١٢٣", "<tool_call>{\"x\":1}</tool_call>",
            "a" + "\u0301"*25, "a\u034f\u0301", " \n\n", "\r\n   ",
            "!? \r\nx", "\u2028\u2029word"]
    return out


def fixture(folder):
    folder.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Digits(individual_digits=True),
         pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tok.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(vocab_size=384, min_frequency=1,
                  initial_alphabet=[c for c in pre_tokenizers.ByteLevel.alphabet() if c != "Ą"],
                  special_tokens=[AddedToken(s, special=True, normalized=False) for s in special],
                  show_progress=False)
    # Exercise the target's null-UNK behavior with one absent byte alphabet entry.
    tok.train_from_iterator([s.replace("\x04", "") for s in texts()], trainer)
    tok.save(str(folder / "tokenizer.json"))
    raw = json.loads(tok.to_str())
    cfg = dict(add_prefix_space=False, added_tokens_decoder={
        str(t["id"]): {k: v for k, v in t.items() if k != "id"} for t in raw["added_tokens"]},
        additional_special_tokens=special[1:], bos_token=special[1], eos_token=special[2],
        pad_token=special[2], unk_token=special[0], chat_template=TEMPLATE,
        clean_up_tokenization_spaces=False, model_max_length=8192,
        tokenizer_class="GPT2Tokenizer", vocab_size=tok.get_vocab_size())
    (folder / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def oracle(folder, output):
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    cases = []
    for text in texts():
        ids = tok.encode(text, add_special_tokens=False)
        cases.append(dict(text=text, ids=ids, decoded=tok.decode(ids, skip_special_tokens=True)))
    chats = []
    for messages in [
        [dict(role="user", content="Hello! 🐈‍⬛")],
        [dict(role="system", content="Answer briefly."), dict(role="user", content="2+2?")],
        [dict(role="user", content="Hi"), dict(role="assistant", content="Hello."),
         dict(role="user", content="Let's write Go.")]]:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        chats.append(dict(messages=messages, prompt=text, ids=ids))
    # Exercise lossy UTF-8 decoding independently of encoded text.
    raw = json.loads((folder / "tokenizer.json").read_text())
    v = raw["model"]["vocab"]
    decode_cases = []
    candidates = [["ð", "Ł"], ["â", "Ģ", "x"], ["Ģ", "Ģ"], ["í", "ł", "Ģ"],
                  ["ð", "Ł", "Ĳ", "Ī"], ["<|im_end|>"]]
    for pieces in candidates:
        if all(s in v for s in pieces):
            ids = [v[s] for s in pieces]
            decode_cases.append(dict(ids=ids, decoded=tok.decode(ids, skip_special_tokens=True)))
    report = dict(tokenizers=tokenizers.__version__, cases=cases, chats=chats,
                  decode_cases=decode_cases, files={
                      name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                      for name in ("tokenizer.json", "tokenizer_config.json")})
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if tokenizers.__version__ != "0.22.2":
        raise RuntimeError("Require tokenizers 0.22.2")
    if a.fixture:
        fixture(a.fixture)
        oracle(a.fixture, a.fixture / "reference.json")
    elif a.model and a.output:
        oracle(a.model, a.output)
    else:
        p.error("provide --fixture, or --model and --output")

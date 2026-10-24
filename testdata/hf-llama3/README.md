# Synthetic Llama 3 RoPE fixtures

Seed 132, randomly initialized tiny Llama; no pretrained weights or user text.
Generated with `scripts/hf_reference.py --fixtures testdata/hf-llama3 --llama3`
and the pinned `scripts/hf_reference_requirements.txt` environment.

The 16-dimensional attention heads exercise high, medium and low frequency
scaling. The 37 input IDs cross the original context (32) and a KV page.
Each source dtype is loaded as FP32 in Transformers CPU eager attention.
Reference JSON records every logit and K/V row and the Safetensors hash.
See `LLAMA3_INFERENCE.md` for the supported configuration boundary.

`frequencies.json` comes from the pinned Transformers RoPE helper without
allocating model weights. It covers the 3B configuration, ordinary fractional
thresholds and the narrow-band rounding regression.

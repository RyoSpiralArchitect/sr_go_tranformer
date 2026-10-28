# Synthetic Mistral fixtures

Generated locally by `scripts/hf_mistral_fixtures.py` with seed 217 using
Transformers 4.57.1, Torch 2.8.0 CPU eager attention and Safetensors 0.6.2.
No pretrained model weights are included. The tiny full-attention Mistral has
2 layers, dimension 24, 6 query / 2 KV heads, vocabulary 258 and an untied head.
F32/F16/BF16 source tensors are loaded as FP32 for numerical references.
`reference.json` binds each weight file by SHA-256 and includes logits and all
K/V rows for 37 deterministic input IDs. Main tests cover chunking, page
boundaries, ragged batches, worker counts and low-bit reconstruction.

# Authored held-out fixture

These short English passages were authored for this repository and are released under its Apache-2.0 license. They were not downloaded from a corpus. Each line is a separate document; no line is repeated within or across the files. Validation passages were written separately and are never fed to the training stream or used to fit a tokenizer.

This is a small, deliberately related-domain fixture for reproducible training/evaluation and performance regression checks. It is not a representative language benchmark, evidence of general LLM capability, or a statistically independent sample of natural writing. Repeated epochs reuse the training documents. Once inspected, validation is a development holdout, not a blind final test. Keep these files fixed across comparisons.

# LM_Dec_Base_NoPE

NoPE - Base family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked (3 seed-repeat runs, see results)**
- Failure class: Note: full memory-update solution in 1 of 2 long runs; LM-02 unstable at 1B tokens

## Provenance

- Source repository: (new) lucidrains/x-transformers 2.31.7
- Upstream / license: lucidrains/x-transformers (MIT), pip 2.31.7

## Differences from the source ([Unified] protocol)

- same decoder as xt_alibi; no position information (causal mask only)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 19.16 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.523 -> 0.186 nats, pass
- LM-02 enwik8 (200M tokens): valid 2.159 / test 2.147 bits per byte
- LM-02 enwik8 (1B tokens): valid 3.608 / test 3.584 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 360 MB, inference 132 MB
- Peak GPU memory at length 8192 (batch 1): train 1307 MB, inference 426 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.26 (keys removed 0.24); 4 pairs @ 2048: 0.23 (keys removed 0.30); 16 pairs @ 512: 0.06 (keys removed 0.07); 16 pairs @ 2048: 0.07 (keys removed 0.06)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.25 (keys removed 0.25); 4 pairs @ 2048: 0.26 (keys removed 0.26); 4 pairs @ 8192: 0.25 (keys removed 0.23); 16 pairs @ 512: 0.07 (keys removed 0.07); 16 pairs @ 2048: 0.06 (keys removed 0.06); 16 pairs @ 8192: 0.07 (keys removed 0.06)
- LM-06 memory update `lm06`: @512: latest 0.49, unchanged 0.03, stale 0.02; @2048: latest 0.46, unchanged 0.04, stale 0.02
- LM-06 memory update `lm06long`: @512: latest 1.00, unchanged 0.95, stale 0.00; @2048: latest 1.00, unchanged 0.96, stale 0.00; @8192: latest 1.00, unchanged 0.95, stale 0.00
- LM-06 memory update `lm06s2`: @512: latest 0.48, unchanged 0.03, stale 0.02; @2048: latest 0.49, unchanged 0.03, stale 0.01
- LM-06 memory update `lm06s3`: @512: latest 0.47, unchanged 0.02, stale 0.02; @2048: latest 0.48, unchanged 0.03, stale 0.02
- LM-06 memory update `lm06longs2`: @512: latest 0.81, unchanged 0.13, stale 0.01; @2048: latest 0.79, unchanged 0.12, stale 0.02; @8192: latest 0.71, unchanged 0.09, stale 0.03

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy x-transformers==2.31.7
python run.py check xt_nope
python run.py lm02 xt_nope      # enwik8 is downloaded to data/ on first use
```

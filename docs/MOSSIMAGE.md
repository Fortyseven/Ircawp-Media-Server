# MOSS-Image-SFT (`mossimage`)

[MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) (OpenMOSS,
Apache-2.0, revision `daf706d`) is a ~5B text-to-image DiT with a Qwen3.5-9B text
encoder and the FLUX.2 VAE. It ships as custom Diffusers code
(`pipeline_moss_image.py`), not a built-in Diffusers class. **Text-to-image only**:
reference images are rejected with an error, and remastering changes nothing.

## Local Models

The transformer must be a local GGUF file; URLs and missing paths fail before
loading. The default is `models/moss_image_sft_Q4_K_S.gguf`. Retain
`models/moss_image_sft_F16.gguf` as the unquantized source.

```bash
.venv/bin/python scripts/convert_mossimage_to_gguf.py \
    --repo OpenMOSS-Team/MOSS-Image-SFT \
    --out models/moss_image_sft_F16.gguf

/home/fortyseven/dev/_outside/llama.cpp/build/bin/llama-quantize \
    models/moss_image_sft_F16.gguf models/moss_image_sft_Q4_K_S.gguf Q4_K_S
```

The converter keeps the 463 official tensor names, writes matrices as F16 (and
fails on F16 overflow) and vectors as F32. It uses the `qwen2` metadata
workaround for llama-quantize only; MOSS dimensions (3072 / 42 layers) are baked
into its metadata. Do not reuse it for other architectures.

| File | Size | Tensor types |
| --- | --- | --- |
| F16 | 10,165,416,352 B | 269 F16, 194 F32 |
| Q4_K_S | 2,875,929,088 B | 263 Q4_K, 5 Q5_K, 1 Q5_0 fallback (`video_embedder`, 128 columns), 194 F32 |

Both files match all 463 official tensor names and shapes.

## Configuration

```yaml
backends:
    mossimage:
        max_output_size: 1024
        steps: 50
        gguf: "models/moss_image_sft_Q4_K_S.gguf"
        base_model: "OpenMOSS-Team/MOSS-Image-SFT"
        text_encoder_4bit: true
```

Select `model: "mossimage"` per request. The default backend is unchanged.
`base_model` may be a repo id (only non-transformer files are downloaded: about
19 GB text encoder plus VAE, tokenizer, scheduler and `pipeline_moss_image.py`) or
a local directory containing them, for offline use. The pipeline code is
imported from that directory, so review it when updating the revision.

## Sampling

- Upstream defaults are used: flow-matching Euler, shift 5, 50 steps, CFG 3, empty
  negative prompt, and negative-branch skipping of block 22 (two transformer
  forwards per step). CFG 1 runs one forward.
- The schedule snaps to the training grid and may run fewer steps than
  requested; progress and PNG metadata report the actual count.
- Python-side sizes must be multiples of 16 (the server already rounds).
- The custom pipeline has no step callback, so cancellation and progress hook
  `scheduler.step`; cancellation takes effect between steps.
- The 24 linear-attention layers of the text encoder use Transformers' slow
  PyTorch fallback because `flash-linear-attention` and `causal-conv1d` are not
  installed. Prompt encoding is a small part of total time.
- The model's GGUF weights are dequantized per layer on the fly (compute dtype
  bf16). MOSS casts activations to `weight.dtype`, which would be uint8 for
  packed GGUF weights, so the loader wraps those parameters to report bf16.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests
```

Tests are mocked or use tiny fixtures (including a real Q8_0 GGUF through the
loader); they do not prove image quality.

### Local GPU Smoke Test (2026-10-16)

RTX 4090, Q4_K_S transformer, 4-bit NF4 text encoder, model CPU offload:

| Check | Result |
| --- | --- |
| Automated tests | 59 passed |
| Backend load (cached files) | about 25 seconds |
| 1024 x 1024, 50 steps, CFG 3 | 63 s total (58 s sampling), 6.3 GB peak allocated |
| `/images/generations` 768 x 512, 30 steps, CFG 3.5 | 33 s; correct size and metadata |
| `/images/edits` | HTTP 500 with the text-to-image-only error |
| Image inspection | Coherent scene; handwritten "Buy milk and eggs!" and "DAILY FOX" rendered correctly |

Not verified: offline operation, Q4_K_S versus F16 or BF16 quality comparison,
`flash-linear-attention` speedup, layout (JSON) prompts, concurrent use with other
loaded backends. The first run downloaded ~27 GB in roughly seven minutes; that
is not a load benchmark.

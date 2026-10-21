# Fibo 2.0 (Bria)

Backends: `fibo2` (Dev: text-to-image + edit) and `fibo2editturbo` (edit-only).
Code: [fibo2.py](../app/backends/fibo2.py), [fibo2editturbo.py](../app/backends/fibo2editturbo.py).
Tests: [test_fibo2.py](../tests/test_fibo2.py).

## Weights

- Official diffusers checkpoints `briaai/Fibo-2.0-dev` and
  `briaai/Fibo-2.0-edit-turbo` (about 25.5 GB each: 8B DiT 16.3 GB, Qwen3-VL-4B
  text encoder 8.9 GB, Flux.2 VAE). Both repos are gated: accept the terms on
  Hugging Face **per repo** while logged in as the token's user.
- License: CC BY-NC 4.0 (non-commercial). Commercial use needs a Bria agreement.
- The `Comfy-Org/Fibo-2.0` repo was deliberately not used: it is a ComfyUI
  repack (fused qkv/to_kv, w1/w3 fused and zero-padded to 9728, custom int8
  "convrot" quant) that Diffusers cannot load.
- Needs a Diffusers build containing `BriaFibo2Pipeline` (merged to main
  2026-10-20; the lock was upgraded to 0.42.0.dev0).

## Quantization

User chose bf16 checkpoint + on-load quantization instead of GGUF.
`transformer_4bit` (default true) quantizes the transformer to bitsandbytes NF4;
`text_encoder_4bit` (default false) is opt-in because the conditioning stage is
the most quantization-sensitive. Weights are not stand-alone files in
`models/`; they live in the Hugging Face cache (set `base_model` to a local
directory for offline use).

## Sampling

| Backend | Steps | CFG | Notes |
| --- | --- | --- | --- |
| `fibo2` | 30 (pipeline default; model card suggests 50) | 5.0 | t2i uses the card's negative caption |
| `fibo2editturbo` | 4 (overrides normalized) | 1.0 (overrides normalized) | text-to-image requests raise `ValueError` |

Edits accept 1-5 images (`<image_1>`, `<image_2>` in the instruction). Masked
edits are supported by the pipeline but not exposed (no mask API field).
`max_output_size` is applied to the longest side like other backends; the model
was trained near 1 MP. No per-request scheduler (the pipeline sets its own
resolution-shifted flow schedule). `quality=high` (remaster) is ignored.

## Measured (2026-10-20, RTX 4090, otherwise idle, 21.7 GiB free)

- Load 6 s (warm cache), Dev, transformer NF4, text encoder bf16, offload on.
- 1024x1024, 30 steps, CFG 5: 55 s, peak 8.4 GiB allocated.
- 1024x1024 edit, 30 steps, CFG 5: 112 s, peak 10.0 GiB. Colour edit
  ("make the fox black") kept the scene intact.
- Edit Turbo (own gated repo): 1024x1024 edit, 4 steps: 16 s, peak 10.0 GiB;
  same colour edit as above, clean result. Text-to-image is rejected.
- Peak figures cover allocated torch memory only, not CPU-offload staging.

## Prompts (TODO)

Fibo is trained on structured JSON captions (Bria's VGL). Currently a plain
prompt is passed as a `short_description` (or as the edit instruction), which
works but under-uses the model. Planned: add a Fibo system prompt to
`app/prompt_prompts.py` so the existing `/prompt-rewrite` LLM flow expands
prompts to structured JSON captions (the backend stays prompt-in/image-out).
Bria's `bria-vgl` helper needs a Gemini API key and is intentionally not used.
Text-rendering captions (`text_render`) can use the `fonts` pipeline argument
(Aileron/DejaVu from the repo's `fonts/` folder); the backend does not pass
fonts yet.

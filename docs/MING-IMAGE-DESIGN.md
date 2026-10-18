# Ming-Image-0.1-Design (`ming_design`)

[inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)
(MIT) is a text-rich design/UI/poster model with RGBA (transparent-background)
output. Upstream code: [inclusionAI/Ming-Image](https://github.com/inclusionAI/Ming-Image)
(commit `3515cd7`). Supports text-to-image and single-reference **image editing**
(upstream's `image-edit` task); see Image Editing below. Files:

- [app/backends/ming_design.py](../app/backends/ming_design.py): the backend.
- [app/backends/ming_loader.py](../app/backends/ming_loader.py): loaders that
  run the whole model from community GGUFs on a 24 GB card.
- [tests/test_ming_design.py](../tests/test_ming_design.py),
  [tests/test_ming_loader.py](../tests/test_ming_loader.py): GPU-free tests for
  the backend contract, import isolation, GGUF reading, name mapping, fused
  experts, seeded sampling, size planning, reference preparation and the edit
  path.
- [scripts/ming_spike.py](../scripts/ming_spike.py): standalone encode, free,
  denoise, save. Usage: `MING_SIZE=1024 MING_OUT=/tmp/x.png .venv/bin/python scripts/ming_spike.py "<prompt>"`.
  [scripts/ming_edit_spike.py](../scripts/ming_edit_spike.py) does the same for
  edits: `.venv/bin/python scripts/ming_edit_spike.py in.png "<instruction>" out.png`.

## Configuration

```yaml
backends:
    ming_design:
        max_output_size: 1024
        steps: 12
        encoder_gguf: "models/ming-image/Text Encoders (UPDATED)/Ming-Image-0.1-Ling-Mini-2.0-Q4_K_M.gguf"
        dit_gguf: "models/ming-image/Ming-Image-0.1-Design-Q4_K_M.gguf"
        base_dir: "models/ming-image/official"   # configs + vae/diffusion_pytorch_model.safetensors
        # code_dir: "vendor/ming-image"          # default; created by `just ming-setup`
```

Select `model: "ming_design"` per request; the default backend is unchanged. All
paths must be local (URLs are rejected, missing files fail before any GPU
loading); relative paths resolve against the server's working directory. The
entry is in `config.example.yml` and the active `config.yml`.

Behavior:

- Defaults: 12 steps, `guidance_scale` 1.0 (advertised as an experimental
  control; upstream recommends 1.0 and uses zero negative conditioning). A
  request `steps` overrides the backend `steps`.
- Sizes: the long side is `max_output_size` (default 1024, aspect 3:2 gives
  1024x688) or the explicit `size`. Like upstream, the model runs at the nearest
  1024 or 2048 aspect bucket (`plan_size`) and the result is resized to the
  requested size. Ratios outside 1:4 to 4:1 are rejected.
- Transparency: begin the prompt with one of the ten upstream phrases, for
  example `RGBA, 4-channel, transparent background.`. The PNG keeps its alpha
  channel; the backend never converts to RGB.
- Memory: the DiT and VAE stay on the GPU (5.8 GiB). The encoder (LLM, vision
  tower, connector, about 11 GiB) lives in CPU RAM and is moved to the GPU only
  for the prefill, then back, and cached CUDA memory is released after every
  request. Peak at 1024 px is about 17 GiB (text-to-image 16.9, edit 17.1);
  idle after a request about 6 GiB reserved. `/backends/unload` (`dispose()`)
  frees everything.
- Editing: pass exactly one input image to `/images/edits` (more are rejected).
  Without a `size` the output keeps the input's aspect at the nearest 1024-bucket
  size (a 1280x720 input gives 1280x720). With a `size` the input is
  center-cropped to that aspect, edited, and resized to the size. `max_output_size`
  is ignored for edits. PNG metadata gets `task: image-edit`.
- The seed comes from a per-request `torch.Generator`; global RNG state is not
  touched. Cancellation and progress use the pipeline's step callback.
- 2048 px peaks at about 20.7 GiB and needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  in the server's environment (set before CUDA initializes), and no other loaded
  backend.

## Upstream Code

The backend imports upstream's flat modules from a **pinned clone** in
`vendor/ming-image` (git-ignored), created by `just ming-setup` at commit
`3515cd7b1e3049fc517eed7d35020251cd06519a`. Override the location with
`MING_CODE_DIR` (spike) or the `code_dir` argument. The directory is not in the
`push-to-wheeljack` rsync excludes, so it deploys without its `.git`; the recipe
leaves such a copy alone. `Justfile` itself is git-ignored in this repo.

## Architecture

The "6B" figure is the DiT only. The official checkpoint is 52.9 GB bf16.

| Stage | Component | Official size |
| --- | --- | --- |
| 0 | Qwen2.5-VL vision tower + 20-layer `BailingMoeV2` LLM (256 experts, top-8, hidden 2048, MultiRouter, 3D RoPE), `mllm/` | 34 GB |
| 0 | 256 learned query tokens, projections (`mlp/`); hidden states from LLM layers 5/12/20 | 0.1 GB |
| 1 | Qwen2-style connector (28 layers, bidirectional), `connector/` | 6.2 GB |
| 1 | 30-layer Z-Image-style `DiffusionTransformer` (dim 3840), `transformer/` | 12.3 GB |
| 1 | 4-channel `AutoencoderKLQwenImage` VAE (RGBA), `vae/` | 0.25 GB |

Text-to-image flow: tokenize (`<role>` chat template) with 256 query-token
placeholders appended, prefill the LLM once (no KV cache, no `lm_head`), take the
final-layer states of the query tokens through `proj_in`, the connector and
`proj_out` (DiT context), and the text-token states of layers 5/12/20 through
`proj_directvlm` (second context). Sampling: flow-matching Euler, 12 steps,
CFG 1.0 (negative conditioning is zeros, so no second pass), 1024 or 2048 px
buckets. Upstream validates only an 80 GB GPU in BF16.

Prompts: best results come from a Figma-style JSON caption produced by an
external rewriter (system prompt in the upstream README). Transparent output
needs one of ten fixed phrases at the start of the prompt (upstream README). The
server-side `/prompt-rewrite` path ([app/prompt_prompts.py](../app/prompt_prompts.py))
is the place for the rewriter, not the backend.

## Local Files

Community GGUFs from
[realrebelai/Ming-Image_GGUFs](https://huggingface.co/realrebelai/Ming-Image_GGUFs)
(MIT, unofficial, made for ComfyUI-GGUF) plus small official files, all under
`models/ming-image/` (git-ignored):

| Path | Size | Contents |
| --- | --- | --- |
| `Ming-Image-0.1-Design-Q4_K_M.gguf` | 5.87 GB | DiT: 309 BF16 / 120 Q8_0 / 30 Q6_K / 60 Q4_K |
| `Text Encoders (UPDATED)/Ming-Image-0.1-Ling-Mini-2.0-Q4_K_M.gguf` | 11.5 GB | ViT, LLM, connector, query tokens, projections |
| `official/` | 0.3 GB | Official configs, tokenizer, scheduler config, and `vae/diffusion_pytorch_model.safetensors` (diffusers layout) |
| `ming_image_vae_bf16.safetensors` | 0.25 GB | Community VAE in ComfyUI key layout. **Not used**; it does not load into the diffusers class |

Other DiT quants: Q2_K 4.71, Q3_K_M 5.13, Q5_K_M 6.16, Q6_K 6.76, Q8_0 7.33 GB.
The encoder also has Q2_K (7.95 GB). No BF16 master is published, so the guide's
"retain the unquantized GGUF" rule cannot be met; keep the Q8_0 DiT as the
highest-fidelity artifact if quality needs it. The `official/` directory was
fetched with `snapshot_download(..., allow_patterns=["*.json","*.txt","*.model","LICENSE"])`
plus the VAE weights.

GGUF header findings (checked without downloading the full files):

- **DiT:** all 519 tensor names and logical shapes match the official
  `transformer/` safetensors. Architecture metadata says `lumina2` (transport
  only). The HQ mix keeps refiners, embedders, final layer and adaLN in BF16 and
  protects attention and `feed_forward.w2`.
- **Encoder bundle:** 1037 tensors, renamed from upstream (`thinker.*`,
  `connector.*`, `vision.*`). Routed experts are **fused 3D Q4_K banks** per layer
  (`experts.gate_up_proj` 256x1024x2048, `experts.down_proj` 256x2048x512).
  `audio_gate` routers are dropped, only the `16x16` query-token set is kept, and
  `mlp/` tensors that were F32 are BF16. Some connector MLP weights are Q4_K.
- **Verified against official weights** (single tensors by HTTP range): fused
  `gate_up_proj` is `[gate; up]` along the output dimension; expert 3 of layer 1
  has cosine 0.9974 for gate, up and down, attention `query_key_value` 0.9976,
  and BF16 tensors (shared expert) are bit-identical.

## Loader Notes

[app/backends/ming_loader.py](../app/backends/ming_loader.py) works with the
installed torch 2.14, transformers 5.17 and diffusers 0.41.dev (upstream pins
2.4 / 4.57.1 / 0.36.0). The compatibility shims live in the `ming_imports()`
context manager, which also puts the clone on `sys.path`. On exit it restores
the patches, `sys.path`, and removes the flat upstream modules (`diffusion`,
`modeling_utils`, ...) from `sys.modules`, so other backends in the process never
see them (tests cover this, including a pre-existing entry and an exception
inside the block). Objects built inside keep working because upstream does no
lazy imports on these paths. Shims:

- `transformers.utils.import_utils.is_torch_fx_available` was removed; define it.
- `ROPE_INIT_FUNCTIONS["default"]` was removed; register an equivalent
  (theta 600000, 64 rotary dims).
- Upstream's processor and tokenizer class fail under Transformers 5. For
  text-only requests the processor reduces to a chat template string plus plain
  tokenization, so the loader builds that string and uses the `tokenizers`
  library directly. The EOS token is `<|role_end|>` from `special_tokens_map.json`.
- Upstream's per-expert `nn.ModuleList` is replaced by a fused `ExpertBank`
  that dequantizes one expert at a time (diffusers' `dequantize_gguf_tensor`).
  Other Q4_K linears use `PackedLinear`; the token embedding is dequantized once.
- `accelerate.set_module_tensor_to_device` casts to the meta parameter's dtype
  (fp32) unless `dtype=` is passed; that silently doubled small-weight memory and
  mixed dtypes.
- `lm_head` is not loaded. The vision tower and `linear_proj` are loaded (the
  tower is BF16, `linear_proj` is dequantized at load) and only run for edits.
- Edit preprocessing reuses upstream's image fetcher, `BailingMM2ImageProcessor`
  (images are resized to about 672x672 pixels for the vision tower) and
  `transform_reference_images`; all of them work on Transformers 5. Only
  upstream's processor class does not, so the chat template is still built by
  hand.
- DiT: diffusers' GGUF reader wraps BF16 GGUF tensors as packed bytes, which
  would corrupt norm weights (not tested), so BF16/F32 tensors are loaded as
  plain tensors and only Q8_0/Q6_K/Q4_K become `GGUFParameter`.
- `ImageGenerator.__init__` insists on the safetensors DiT, so the generator is
  assembled without it (`ImageGenerator.__new__` plus the upstream pipeline).
  Its `sample()` seeds the global RNG with `torch.manual_seed`, so
  `sample_image()` calls the pipeline directly with a per-request generator
  (same-seed output is pixel-identical to the upstream path).

## Spike Results (2026-10-17, RTX 4090, 22.4 GiB free, nothing else on the GPU)

| Stage | Time | VRAM |
| --- | --- | --- |
| Encoder load (page-cached) | 6.9 s | 9.7 GiB allocated |
| Encode, 33 to 124 text tokens (plus 256 query tokens) | 2.8 s | 9.8 GiB peak |
| DiT + VAE load | 3.4 to 5.6 s | 5.8 GiB allocated after the encoder is freed |
| 1024x1024, 12 steps | 9.0 s (1.39 it/s) | 9.9 GiB peak |
| 2048x2048, 12 steps | 49.3 s | 20.7 GiB peak; needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, OOM without it |

Output checks (seed 42, Q4_K_M, CFG 1.0):

- "A red circle on a white background": clean flat circle on white.
- The upstream README landing-page prompt at 1024 px: correct layout; every
  quoted string rendered correctly ("Flow", "Product / Pricing / Docs", "Focus
  without the noise", "Start free"); unspecified body copy was garbled.
  At 2048 px the headline and button text were crisp; the nav gained an invented
  extra link ("Proding").
- "RGBA, 4-channel, transparent background. A glossy red apple with a green
  leaf": RGBA PNG with alpha 0 to 255, 57% of pixels transparent.

Not verified: comparison with the official bf16 pipeline (needs the 52.9 GB
download and offloading), quality versus Q8_0 or Q2_K, JSON-caption prompts,
concurrent use with other loaded backends, `llama-server` coexisting.

## Backend Verification (2026-10-17, same host)

GPU-free tests: 158 pass across the suite (`tests/test_ming_design.py` has 51,
`tests/test_ming_loader.py` 34). Real checks:

| Check | Result |
| --- | --- |
| Backend construction (encoder to CPU RAM, DiT/VAE to GPU) | 25.4 s, 5.8 GiB VRAM |
| 1024x1024, 12 steps, seed 42 | 17.2 s total (about 8.5 s sampling, the rest is moving the encoder to the GPU, encoding and moving it back); 15.6 GiB peak, 5.8 GiB after |
| Same seed vs the spike image | pixel-identical (max difference 0) |
| Default size (1024x688, resized from the bucket) | 16.8 s, RGBA, progress reached 12/12 |
| Cancel after step 3 of 12 | `GenerationCancelled` after 10.5 s; the encoder was back on the CPU |
| `dispose()` | GPU allocation fell to 0.01 GiB |
| HTTP `POST /images/generations`, `model: ming_design`, `size: 1280x720`, seed 11 | 200 in 30.8 s including the first load; RGBA PNG with metadata; headline "JAZZ NIGHT", "Live at The Blue Door" and "Friday 8 PM" all rendered correctly |
| HTTP `POST /images/edits` with two images | 500 "ming_design edits take exactly one reference image, got 2" |
| `GET /backends` | lists `ming_design` with `cfg`, `cfg_experimental` and `seed` |

## Image Editing

The Design checkpoint is a `generation_edit` profile in upstream's
`inference_profile.py` (zero-masked padding, 4-channel argmax VAE, single frame),
which accepts the `text-to-image` and `image-edit` tasks; `image-edit` requires a
reference image and runs at the 1024 bucket. The vLLM-Omni recipe documents
editing with this same checkpoint, and the community encoder GGUF was re-uploaded
because an earlier version "was stripped of editing components". The upstream
model card and README present Design mainly as text-to-image and give no editing
examples.

How it works here: the reference image goes through the vision tower and
`linear_proj` (L2-normalized embeddings that replace the image-patch tokens in
the prompt, ahead of the 256 query tokens), and, as 4-channel RGBA latents from
the VAE, into the DiT as `ref_hidden_states`, so each step handles about twice
the image tokens.

Results (seed fixed, Q4_K_M, 12 steps, CFG 1.0; input is the generated 1280x720
"JAZZ NIGHT" poster, spike and HTTP runs gave pixel-identical output):

- "Change the background color to deep red and make the saxophone silver": clean
  and exact; the headline, subtitle and footer text were preserved.
- "Change the headline text to 'BLUES NIGHT' ...": the headline changed
  correctly, but the small subtitle was corrupted in both runs I inspected (seed
  42 at native size, and a 1024x1024 crop): duplicated words, e.g. "Live Live at
  The The Blue Blue Door Door", worse in the crop. Edits that regenerate text
  can damage neighbouring small text.
- "Change the apple to green and add a small white price tag reading '$2'" on the
  transparent apple image: worked.

Timing on the 4090: encode about 3 s, 12 sampling steps 16 to 20 s at 1280x720
(1.4 to 1.6 s per step), about 27 s per request including the encoder swap.
Not tested: images with unusual aspect ratios, very small inputs (the output is
still at the 1024 bucket, so they are upscaled), multiple references (rejected).

## Remaining Work

- Not verified: comparison with the official bf16 pipeline (needs the 52.9 GB
  download and offloading), quality versus Q8_0 or Q2_K, JSON-caption prompts,
  coexistence with other loaded backends or `llama-server` (the encoder swap
  needs about 17 GiB free), cancellation through the HTTP `request_id`, 2048 px
  through the backend (only the spike), and how the frontend displays
  transparent PNGs.
- About 8 s of each request is moving the encoder over PCIe from pageable
  memory; pinning it would cut that, at the cost of about 11 GiB locked RAM.
- Edits: no `max_output_size` handling and no multi-reference support (upstream
  and vLLM-Omni also take one).
- Optional: add the Ming rewriter system prompt to the prompt-rewrite path.
- Upstream code is a pinned clone (decided); revisit vendoring a trimmed copy
  only if the shims become a maintenance burden.

## Risks

- Q4_K_M quality against the official model is unmeasured beyond per-tensor
  cosine (about 0.997).
- 2048 px peaks at 20.7 GiB; it will not fit beside another loaded backend.
- Upstream code runs on transformers 5 only through the shims above; a
  Transformers upgrade can break them.

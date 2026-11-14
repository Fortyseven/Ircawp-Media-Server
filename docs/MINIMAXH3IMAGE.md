# MiniMax-H3-Image (`minimaxh3image`)

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) (checkpoint
revision `00dd081ebdd33315adf2c4d84f64aa327ddf3039`, released 2026-11-03) is a
33.1B-parameter single-stream diffusion transformer, distilled to **six
sampling steps with no guidance**, that generates natively at 4-20 megapixels.
It handles text-to-image and multi-reference editing (up to 16 references).
This backend runs it on a 24 GB card with a 4-bit transformer and a 4-bit text
encoder, for text-to-image and for reference-image editing. **Editing has unit
tests only; it has not been run on the real model yet** (see [Editing](#editing)).

## What is in the checkpoint, and what this backend runs

| Part                                                   | Official                                       | This backend                                                                 |
| ------------------------------------------------------ | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| Transformer (`MiniMaxH3ImageTransformer2DModel`)       | 66.3 GB BF16, 50 blocks, hidden 5376, 56 heads | `models/minimax_h3_image_Q4_K_S.gguf`, 12.05 GB                              |
| Text encoder (Qwen3-VL-32B, reads `hidden_states[50]`) | 66.7 GB BF16, 64 layers                        | `models/minimax-h3-image/text_encoder`, bitsandbytes NF4, 51 layers, 17.4 GB |
| VAE (24-ch, 16x, ViT decoder)                          | 2.34 GB FP32                                   | copied unchanged, `models/minimax-h3-image/vae`                              |
| Tokenizer, processor, scheduler, `pdd_config.json`     | small                                          | copied unchanged                                                             |

Total on disk for the local model: **31.9 GB** (12.05 GB GGUF + 19.8 GB component
folder), plus the 40.5 GB retained BF16 GGUF. The official downloads in the Hugging
Face cache (117 GB, plus ~7 GB of stale `.incomplete` blobs from an aborted
download of the unneeded `transformer_diffusers/`) are not needed at run time and
can be deleted to reclaim space; they are only the source for re-converting.

Sampling: the DiT is distilled on a fixed grid `sigmas = [1.0, 0.9675, 0.8875,
0.75, 0.5328, 0.2320, 0]` with **one output head per step** (the "PDD" heads,
`pdd_fused_heads.safetensors`), guidance baked in. There is no `steps`,
`guidance_scale`, `negative_prompt` or scheduler choice; the backend ignores
overrides of these and reports six steps in progress and PNG metadata. The
pipeline is Diffusers' `MiniMaxH3ImageModularPipeline` (modular only; there is
no `DiffusionPipeline`).

## Files

```
models/
  minimax_h3_image_BF16.gguf        40.5 GB retained unquantized source (see below)
  minimax_h3_image_Q4_K_S.gguf      12.05 GB transformer used at run time
  minimax-h3-image/                  components, no network needed at run time
    transformer/config.json
    text_encoder/                    NF4, 51 layers, vision tower kept in bf16
    vae/  tokenizer/  processor/  scheduler/  pdd_config.json  LICENSE
```

All binaries live under the git-ignored `models/`. The deployment recipe
(`just push-to-wheeljack`) excludes `models/`; copy these files separately.

## Reproducing the files

Hardware used: RTX 4090 24 GB, 125 GiB RAM. Disk: the official download is
66 GB (transformer) + 57 GB (the 12 text-encoder shards that are needed), the
BF16 GGUF is 40 GB and conversion needs the same again as scratch.
**Set `TMPDIR` to a large volume**; `/` had only ~97 GB free.

```bash
REV=00dd081ebdd33315adf2c4d84f64aa327ddf3039
# 1. Official weights. transformer/ is the original layout; the repo also carries a
#    66 GB transformer_diffusers/ copy that is NOT needed. Text encoder: only shards
#    1-11 and 14 hold layers 0-50, the embeddings and the vision tower.
hf download MiniMaxAI/MiniMax-H3-Image --exclude "text_encoder/*" --exclude "assets/*" \
    --exclude "examples/*" --exclude "transformer_diffusers/*" --exclude "vae/*"
hf download MiniMaxAI/MiniMax-H3-Image $(for i in 01 02 03 04 05 06 07 08 09 10 11 14; do \
    echo --include text_encoder/model-000$i-of-00014.safetensors; done) \
    --include "text_encoder/*.json" --include "text_encoder/*.txt"
SNAP=~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/$REV

# 2. Retained unquantized GGUF, then the 4-bit file derived from it.
TMPDIR=/path/with/space .venv/bin/python scripts/convert_minimaxh3image_to_gguf.py \
    convert --repo $SNAP --revision $REV --out models/minimax_h3_image_BF16.gguf
.venv/bin/python scripts/convert_minimaxh3image_to_gguf.py quantize   # Q4_K_S

# 3. Component folder: copies the small files and builds the NF4 text encoder.
.venv/bin/python scripts/prepare_minimaxh3image_components.py prepare --repo $SNAP

# 4. Checks (see Verification).
.venv/bin/python scripts/verify_minimaxh3image_gguf.py models/minimax_h3_image_BF16.gguf
.venv/bin/python scripts/prepare_minimaxh3image_components.py verify --repo $SNAP
```

The `quantize` subcommand runs this `llama-quantize` (from
`/home/fortyseven/dev/_outside/llama.cpp`; override with `--llama-quantize`):

```
llama-quantize --tensor-type proj_in=f32 --tensor-type context_embedder=f32 \
    --tensor-type time_embedder=f32 --tensor-type norm_out=f32 \
    --tensor-type pdd_proj_out=f32 --tensor-type token_refiner=q8_0 \
    models/minimax_h3_image_BF16.gguf models/minimax_h3_image_Q4_K_S.gguf Q4_K_S
```

## The AdaLN table (the one non-obvious departure from the official weights)

Every transformer block has an `adaln_proj` linear (2688 -> 96768). It is
**13B of the 33B parameters**, and because 2688 is not a multiple of 256 it
cannot be stored as Q4_K (llama.cpp would fall back to a larger type). Its input
is only the timestep embedding, and the checkpoint only ever runs at seven
timesteps: `1 - sigma_i` for the six steps (0, 0.0325, 0.1125, 0.25, 0.4672,
0.7680) and the 0.999 level reserved for edit references. So the converter
evaluates `adaln_proj` once at those seven timesteps with the official bf16
weights and stores the result as `transformer_blocks.N.adaln_proj.table`
`(7, 12 * 5376)`, dropping the audio-modality third that the image model never
reads. The backend swaps each block's `adaln_proj` for `TabulatedAdaLN`, which
looks rows up by timestep.

- This is exact for the released schedule (see Verification), and removes ~8 GB
  that a 4-bit `adaln_proj` would have needed. It is **unusable for any other
  schedule**: a timestep off the grid raises `ValueError`. The official
  safetensors remain the source for a different schedule.
- The community Comfy-Org "pruned" checkpoint (and the GGUFs derived from it) does
  something related with a low-rank basis and different tensor layout.
  Diffusers' converter rejects it explicitly; it was not used.

## Tensor types in the Q4_K_S file

595 tensors, 12,053,168,288 bytes (4.79 bits/weight overall):

| Tensors                                                                                                                         | Type |
| ------------------------------------------------------------------------------------------------------------------------------- | ---- |
| 300 block linears (to_q/k/v, to_out, ff in/out; 50 blocks x 6)                                                                  | Q4_K |
| 12 token-refiner linears (text conditioning path)                                                                               | Q8_0 |
| patch input, context embedder, time embedder, norm-out linear, 6 PDD heads, norms, biases, 50 AdaLN tables, timestep grid (283) | F32  |

The BF16 file has the same 595 tensors with the 312 matrices in BF16 and the 283
small tensors in F32. Matrices that must stay high precision are F32 in both.
The `qwen2` architecture key and its dimensions inside the GGUFs are spoofed only
so llama-quantize accepts them (as for the Qwen and MOSS converters); the
backend loads tensors by name and ignores them.

## Configuration

```yaml
backends:
    minimaxh3image:
        max_output_size: 2048
        gguf: "models/minimax_h3_image_Q4_K_S.gguf"
        base_model: "models/minimax-h3-image"
        # Park every component on the CPU after each request, freeing the GPU
        # for other users (default false: the last-used component stays resident).
        release_gpu_after_request: false
```

Select `model: "minimaxh3image"` in an API request. The top-level default
`backend:` is unchanged. Both paths must be local; URLs and missing paths fail
before any GPU loading, and a missing component folder names what is absent.
Relative paths resolve against the server's working directory; the built-in
defaults are absolute and project-relative. `GET /backends` advertises `seed`
only (no steps, CFG, negative prompt or scheduler controls).

Behavior notes:

- `max_output_size` caps the long side when no explicit `size` is given and
  defaults to 2048; sizes derived from it snap to multiples of 32.
- The model generates on at least a **4 MP canvas** (2048x2048 area) and resizes
  back: a 1024x1024 request costs the same as 2048x2048. `size` below that is
  therefore a quality/speed trade only in the final resize.
- A cancellation request is honoured between transformer steps; progress is
  `step/6`. `GET /images/progress/{id}` also carries a `phase` (`encoding prompt`,
  `denoising`, `decoding`) that the UI shows next to the percentage; it is absent
  for backends that don't set it and cleared when the request ends.
- `release_gpu_after_request: true` moves every component off the GPU when a
  request ends (success, error or cancellation), at ~0.3 s cost, so something
  else (e.g. `llama-server`) can use the card. With it off, ~13-18 GB stays
  allocated while idle. It does not lower the peak _during_ a request.
- Prompts are used verbatim (no chat template), truncated at 4096 tokens. The
  model wants long, concrete English descriptions. The card's prompt expander
  (MiniMax-H3-Image-PE) is a separate model and is not integrated; prompt
  rewriting stays with the client.

## Offload and latency

The text encoder (17 GB), transformer (11 GB) and VAE cannot share a 24 GB card,
so Diffusers' `ComponentsManager` swaps them per request. Its stock offload copies
each evicted component back to the CPU, and a GPU-to-CPU copy of that size costs
4-5 s (it allocates pageable memory), against 1-2 s for CPU-to-GPU. Measured for a
1024x1024 request (canvas 2048x2048):

|                                                           | copy-back offload | `CpuMirror` |
| --------------------------------------------------------- | ----------------- | ----------- |
| Whole request                                             | 26.9 s            | 15.6 s      |
| Step 0 (evict text encoder, load transformer, first step) | 9.4 s             | 3.1 s       |
| Steps 1-5 and VAE decode                                  | 1.8 s each        | 1.8 s each  |
| After the VAE decode (offload, PNG save)                  | 4.7-7 s           | 0.4 s       |

`CpuMirror` (now the shared `app/backends/cpu_mirror.py`, also used by the other
backends; `fast_offload: false` disables it here too) exploits that the weights never
change: the CPU tensors a component had while parked are kept as masters, and
"offloading" puts them back, which drops the GPU copy with no transfer. It also
restores bitsandbytes' `quant_state` tensors, which `Module.to()` moves in place.
It is installed on the manager's hooks once all components are loaded.

- **Verified bit-exact on the real model**: the 1024x1024 image is identical
  (max pixel difference 0) to the one made with the stock copy-back offload, over
  repeated request cycles and after a full release. A bitsandbytes upgrade that
  changes where `QuantState` keeps its tensors would be caught by repeating that
  same-seed comparison; the unit tests only use a stand-in `QuantState`.
- No extra CPU memory by construction: the masters are the same tensors that
  were already held to park each component. (Server RSS was measured at ~18 GB
  loaded before `CpuMirror` was added and not re-measured since.)

## Editing

Reference-image editing runs through the pipeline's `image_conditioned`
workflow. It is selected by passing `media` to `execute()` (the server's
`POST /images/edits` does this), and the same checkpoint, GGUF and text encoder
are used. **Status: implemented and unit-tested with a mocked pipeline; not yet
run on the real model or GPU.** What that leaves unverified is listed below.

How a request is handled:

- **References**: up to 16 images (the checkpoint's limit; more raise a clear
  error before any GPU work). They are opened upright (EXIF orientation applied)
  and converted to RGB, which **drops any alpha channel** (the pipeline does the
  same). Each side must be at least 32 px; unreadable files raise a
  `ValueError` naming the file.
- **Prompt**: refer to them in order as `<Picture 1>`, `<Picture 2>` ...; the
  prompt is used verbatim. MiniMax suggests naming what to preserve, then what to
  change, then describing the target image in full, e.g. "Keep everything in
  `<Picture 1>` as it is, but change the blue net to a red one."
- **Output size**: an explicit `size` (width/height) wins. Otherwise the output
  follows the _first_ reference's aspect ratio with its long edge at
  `max_output_size`, rounded to 16 px. The server sets `max_output_size` to the
  input's own long edge, so an edit returns at the input's size by default (the
  pipeline's own default would be 2048x2048). The same 4 MP minimum canvas as
  text-to-image applies: a 1024 px edit is generated on a 2048x2048-area canvas
  and resized back.
- **Reference resolution**: the server first shrinks inputs to
  `backends.minimaxh3image.max_input_edge` (default 1440). The pipeline then
  resizes each reference for the VAE so its short edge is within 1024-2048, and
  for the vision encoder to exactly 1024, so **small references are upscaled**
  and every reference costs at least ~1024 packed-sequence rows plus ~1024
  vision tokens. Expect time and VRAM to grow with the number of references.
- **Metadata**: the PNG records `task=image-edit` and the reference count.
  Progress phase is `encoding references` instead of `encoding prompt`.
- Everything else (fixed six steps, no `steps`/CFG/negative prompt, seed,
  cancellation, `fast_offload`, `release_gpu_after_request`) is as for
  text-to-image.

How it is built on what was prepared earlier: the text encoder keeps the
Qwen3-VL **vision tower in bf16** (~1 GB) that the edit presentation needs, the
GGUF holds the AdaLN modulation for the 0.999 reference timestep, and the
loader's `TabulatedAdaLN` selects rows for a forward with two distinct timesteps
(unit-tested against a dense model at tiny scale).

### Why editing is slower than text-to-image

Every reference adds rows to the packed sequence the transformer attends over,
on all six steps: ~1000 latent rows for its VAE copy, plus ~1000 vision tokens in
the text-encoder prompt (a small reference is upscaled to 1024 px first, so it
costs the same). For a 1024x1024 input and output the sequence is ~6,200 rows
against ~4,150 for text-to-image (+50%); for a 1440x1080 input about +70%.
Linear cost grows with rows and attention with rows squared. A crude step-time
curve fitted to two of the text-to-image runs predicts about +1 s per step, so
**+5-6 s over six steps** (estimate, not a measurement), plus the vision tower,
the longer text pass and the reference VAE encode. A user-reported edit was
4-5 s slower than text-to-image, consistent with that. It is inherent to how the
model conditions on references (the reference rows cannot be cached: attention
is bidirectional and they change with the generated rows at every layer).

```
[minimaxh3image] edit (1 reference) 1024x1024, 6195 rows: setup 4.1s | steps 3.2 2.8 2.8 2.8 2.8 2.8 (sum 17.2s) | decode+save 2.0s | total 23.3s
```

(illustrative numbers; `setup` is everything before step 0: text/vision encode,
reference VAE encode and the model swaps). Levers that would cut the cost, **not
implemented because each trades quality or fidelity and none is measured**:

- Generate on a smaller canvas: the pipeline's `min_generation_area` (default
  2048x2048) makes any request under 4 MP generate on a 4 MP canvas and resize
  back. Lowering it shrinks the generated rows most (a 1 MP canvas has a quarter
  of them), but the card says output quality is best from 4 MP up.
- Lower the references' minimum short edge (`reference_min_short_edge`, default 1024) so small references are not upscaled. This changes what the model sees
  relative to how it was trained.
- Fewer or smaller references.

### Not yet verified (needs the real model)

- That an edit actually works: instruction following, and that unchanged scene
  detail is preserved. This is the first thing to check by eye.
- Peak VRAM and time with 1, a few, and many references, and with references
  larger than ~1440 px. The 16-reference limit is the model's, not a measured
  limit of this card; the text-to-image peak at 16 MP was already within ~1.5 GB
  of the card's capacity.
- The 4-bit text encoder's _vision_ path (the tower is bf16, but the language
  layers that consume its features are NF4), and the `deepstack` feature
  injection at layers 8/16/24 inside the truncated 51-layer stack.
- Whether alpha-dropping (transparent PNGs become whatever RGB sits under the
  alpha) is acceptable for your inputs.

### Trying it

With the server running and the GPU free (unload other backends first):

```python
# python3 edit_test.py input.png [second.png ...]   (stdlib only)
import base64, json, sys, time, urllib.request

images = [{"image_url": "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()}
          for p in sys.argv[1:]]
body = {"model": "minimaxh3image", "seed": 7, "images": images,
        "prompt": "Keep everything in <Picture 1> as it is, but change the blue net to a red one."}
t = time.time()
req = urllib.request.Request("http://127.0.0.1:8100/images/edits", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
data = json.load(urllib.request.urlopen(req, timeout=900))["data"][0]["b64_json"]
open("edited.png", "wb").write(base64.b64decode(data))
print(f"done in {time.time() - t:.1f}s -> edited.png")
```

Watch `nvidia-smi` for the peak while it runs, and try two or three references
(`images` is a list; refer to them as `<Picture 1>`, `<Picture 2>` ...). Please
record the time, peak VRAM and whether the result is what you asked for.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests
```

`tests/test_minimaxh3image.py` (59 tests) covers: converter name mapping, QKV
de-interleave, SwiGLU gate/value swap, PDD heads, dtypes, tiny end-to-end
safetensors-to-GGUF conversion loaded into the real
`MiniMaxH3ImageTransformer2DModel` class; the AdaLN table equal to the real
`MiniMaxH3ImageAdaLayerNormModulation` module; the GGUF loader reproducing a
dense transformer's outputs (including a quantized-linear path and a two-timestep
call); backend path validation, request sizing, seed, metadata, cancellation,
progress and phases, editing (reference forwarding, EXIF orientation, output sizing,
the 16-reference and 32 px limits, unreadable files), the release option, `CpuMirror` (including restoring
in-place-mutated quantization state), no state leaking between requests, disposal
and the advertised capabilities. These are mocked or tiny; they do not establish image quality.

Real artifact checks (2026-11-11):

| Check                                                                | Result                                                                                                                                             |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| BF16 GGUF vs official `transformer_diffusers` (HTTP range reads)     | 595 tensor names/shapes match; 64 spot-checked tensors (blocks 0 and 49, refiner, embedders, norm-out, PDD head 5) bit-identical (max abs error 0) |
| AdaLN tables, blocks 0 and 49, vs the real module at the 7 timesteps | max abs error 0                                                                                                                                    |
| Q4_K_S tensor types via `GGUFReader`                                 | 300 Q4_K, 12 Q8_0, 283 F32                                                                                                                         |
| 4-bit text encoder vs bf16 reference, `hidden_states[50]`, 3 prompts | cosine mean 0.9961-0.9970, min 0.988-0.994, relative error 1.3-1.4%                                                                                |
| Transformer alone on the GPU (synthetic input, 4096 image tokens)    | 11.12 GiB resident, 12.13 GiB peak, ~2 s/step                                                                                                      |

### Local GPU smoke test (2026-11-11)

RTX 4090 24 GB (about 1.5 GB used by the desktop, no other compute process,
`llama-server` idle), 125 GiB RAM, Q4_K_S GGUF + NF4 text encoder, Diffusers
`875373a`. Everything loaded from `models/`; the server runs used
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and every `*_PROXY` pointed at a dead
port, so any network access would have failed. Times are one whole `execute()`:
text encoding, offload swaps, six steps and VAE decode. They are warm (files in
the page cache) and were each measured once or twice, not benchmarked. The
first table is **with `CpuMirror`** (see above); the figures in parentheses are
the earlier ones with the stock copy-back offload.

| Request                                                                                                   | Canvas    | Time            | Whole-GPU peak (`nvidia-smi`) |
| --------------------------------------------------------------------------------------------------------- | --------- | --------------- | ----------------------------- |
| 1024x1024 (seed 1)                                                                                        | 2048x2048 | 15.6 s (26.9 s) | -                             |
| 2048x2048                                                                                                 | 2048x2048 | 17.4 s (23.5 s) | 18.8 GB (18.7)                |
| 4096x4096 (the card's 1:1 recommendation, 16 MP)                                                          | 4096x4096 | 61.1 s (71.7 s) | 22.1 GB (20.9)                |
| 5472x3072 (the card's 16:9 recommendation, 16.8 MP)                                                       | 5472x3072 | 62.5 s (72.1 s) | 19.4 GB (20.9)                |
| 1024x1024 through `/images/generations`, warm                                                             | 2048x2048 | 15.8 s          | -                             |
| 1536x1024 through `/images/generations`, first request after server start, incl. model load (earlier run) | 2496x1664 | 50.7 s          | -                             |

The 4096x4096 peak is higher than before because the manager now keeps the
transformer resident during the VAE decode when it fits; that leaves **about 1.5
GB of headroom at 16 MP**, so close other GPU users before large sizes.

- **Backend load** (from local files, no network): 18-32 s depending on page-cache
  warmth. Idle GPU afterwards: 13-18 GB with the default, ~2 GB with
  `release_gpu_after_request: true`. Server RSS was ~18 GB
  while loaded and ~7 GB after `/backends/unload`, and the GPU returned to ~2 GB
  (measured before `CpuMirror`; unload was not re-run since).
- **The 17.4 GB text encoder and the 11 GB transformer are never resident
  together**; the peak is one of them plus activations (the transformer plus the
  VAE at the largest sizes). At 16 MP there is only ~1.5-3 GB of headroom, so
  **a loaded `llama-server` (~22 GB) will make large sizes fail**; unload it
  first. Sizes above ~17 MP (the card allows up to 20) were not tried.
- **Same seed reproduces bit-identically** (two 1024x1024 runs differ by 0).
- **Images were inspected**: faithful to long prompts at all sizes tried; the 1:1
  crop of the 4096x4096 shows clean burlap weave, crust and wood grain with no
  tiling seams or decode artifacts. Non-square sizes return the requested size.
- **API**: `GET /backends` lists `minimaxh3image` with `{seed: true, steps: false,
cfg: false, negative_prompt: false, schedulers: []}` and the default backend is
  unchanged. `POST /images/generations` with `model: "minimaxh3image"` works, and
  `steps: 30` is ignored (PNG metadata says 6). `/images/progress` reports `1/6`
  mid-run, `/images/cancellations` aborts it (HTTP 409 "Generation cancelled")
  and, with `release_gpu_after_request: true`, the GPU is released afterwards. `/images/edits` returned a "does not
  support reference images" error as HTTP 500 at the time of this test, before
  editing was added.
- Not tested: the frontend UI, running next to other loaded backends, long-term
  stability, concurrent requests.

## Limitations

- Image quality of the 4-bit stack has **not** been compared with the bf16 model:
  the bf16 pipeline does not fit this machine, so there is no same-seed image
  comparison. The only quantitative evidence is the text-encoder similarity above
  and the bit-exact transformer-side checks (the 4-bit transformer's own error is
  unmeasured). Outputs look excellent by eye, which is not a measurement.
- The Q4_K_S file uses llama.cpp's default K-quant for every block linear; no
  per-layer sensitivity study was done. Q5_K_M/Q6_K of the same BF16 file would
  fit easily (the transformer is only 11 GB) if quality ever disappoints.
- Editing is untested on the real model (see [Editing](#editing)).
- Prompt expansion (MiniMax-H3-Image-PE) is not integrated.
- Diffusers is now pinned to commit `875373ac7f5856071e345a3e394c74efccc6b9e6`
  (the merge of upstream PR #15230 that added MiniMax-H3-Image, 2026-11-10); the
  previous pin (`5378870`) predates it. The unit tests of the other backends pass
  as before (4 pre-existing `qwenimage21turbo` failures are unchanged), but the
  other backends were **not** re-run on the GPU after the bump.
- `uv lock` with the installed uv also rewrote unrelated environment markers in
  `uv.lock` (lockfile revision 3 -> 5); the package set only changed for diffusers.

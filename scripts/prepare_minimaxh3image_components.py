"""Assemble the standalone, offline component folder for the minimaxh3image backend.

  prepare  copy the small official components and build the 4-bit text encoder
  verify   compare the 4-bit text encoder's conditioning with a bf16 reference

The output folder (default ``models/minimax-h3-image``) holds everything the
backend needs besides the transformer GGUF:

  vae/            official Diffusers-format image VAE (float32, ~2.3 GB)
  transformer/    config.json only (the weights are the GGUF)
  tokenizer/ processor/ scheduler/   official, tiny
  pdd_config.json LICENSE            sampling grid and upstream license
  text_encoder/   Qwen3-VL-32B, bitsandbytes NF4, truncated to 51 layers

The transformer reads ``hidden_states[50]`` of the encoder, so layers 52-64 are
dropped.  51 layers are kept rather than 50 because the Diffusers pipeline
refuses a stack of exactly 50: the last hidden state of a truncated stack is
post-norm, which is not the conditioning the transformer was trained against.
The vision tower stays in bfloat16 (it is only used for reference-image edits,
which this backend does not expose yet, and it is ~1 GB).
"""

import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

REPO = "MiniMaxAI/MiniMax-H3-Image"
KEEP_LAYERS = 51  # hidden_states[50] must not be the final (post-norm) state
CONDITIONING_LAYER = 50
SMALL_FILES = {
    "vae_diffusers": "vae",
    "transformer_diffusers": "transformer",  # config.json only, see copy_dir()
    "tokenizer": "tokenizer",
    "processor": "processor",
    "scheduler": "scheduler",
}


def snapshot(repo_or_dir):
    if Path(repo_or_dir).is_dir():
        return Path(repo_or_dir)
    from huggingface_hub import snapshot_download

    patterns = [f"{d}/*" for d in SMALL_FILES if d != "transformer_diffusers"]
    patterns += ["transformer_diffusers/config.json", "pdd_config.json", "LICENSE",
                 "text_encoder/*.json", "text_encoder/*.txt"]
    return Path(snapshot_download(repo_or_dir, allow_patterns=patterns))


def copy_dir(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if src.name == "transformer_diffusers" and item.suffix != ".json":
            continue  # the weights come from the GGUF
        if item.name.endswith(".index.json"):
            continue
        shutil.copyfile(item.resolve(), dst / item.name)  # resolve HF cache symlinks


def filtered_text_encoder_source(text_encoder_dir, workdir, keep_layers=KEEP_LAYERS):
    """A folder of symlinks + an index limited to the shards/keys actually present and needed."""
    import re

    src = Path(text_encoder_dir)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = {}
    for key, shard in index["weight_map"].items():
        match = re.search(r"language_model\.layers\.(\d+)\.", key)
        if match and int(match.group(1)) >= keep_layers:
            continue
        if not (src / shard).exists():
            raise FileNotFoundError(f"{shard} is needed for {key} but is not downloaded")
        weight_map[key] = shard
    out = Path(workdir)
    for item in src.iterdir():
        if item.name != "model.safetensors.index.json" and not item.name.endswith(".safetensors"):
            (out / item.name).symlink_to(item.resolve())
    for shard in set(weight_map.values()):
        (out / shard).symlink_to((src / shard).resolve())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": weight_map})
    )
    return out


def build_text_encoder(text_encoder_dir, out_dir):
    import torch
    from transformers import AutoConfig, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    with tempfile.TemporaryDirectory() as work:
        source = filtered_text_encoder_source(text_encoder_dir, work)
        config = AutoConfig.from_pretrained(source)
        config.text_config.num_hidden_layers = KEEP_LAYERS
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            source, config=config, quantization_config=quantization,
            dtype=torch.bfloat16, device_map={"": 0},
        )
        model.save_pretrained(out_dir, safe_serialization=True)
        # The tokenizer-side files travel with the checkpoint folder.
        for name in ("chat_template.json",):
            if (Path(text_encoder_dir) / name).exists():
                shutil.copyfile((Path(text_encoder_dir) / name).resolve(), Path(out_dir) / name)
    size = sum(f.stat().st_size for f in Path(out_dir).glob("*.safetensors"))
    print(f"Wrote 4-bit text encoder to {out_dir} ({size} bytes)")


def prepare(snapshot_dir, out):
    out = Path(out)
    snap = snapshot(snapshot_dir)
    for source, target in SMALL_FILES.items():
        copy_dir(snap / source, out / target)
    for name in ("pdd_config.json", "LICENSE"):
        shutil.copyfile((snap / name).resolve(), out / name)
    build_text_encoder(snap / "text_encoder", out / "text_encoder")


def _encode(model, tokenizer, prompt, device):
    import torch

    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        outputs = model.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=torch.zeros_like(ids),
            use_cache=False,
            output_hidden_states=True,
        )
    return outputs.hidden_states[CONDITIONING_LAYER].float().cpu()[0]


PROMPTS = [
    "A close-up photograph of a weathered fisherman mending a blue net on a wooden dock at sunrise, "
    "shallow depth of field, warm light, visible rope fibres.",
    "A neon sign reading 'OPEN LATE' above a rainy Tokyo alley, reflections on wet asphalt, cinematic.",
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table.",
]


def verify(snapshot_dir, out):
    """Cosine similarity of hidden_states[50] (4-bit vs bf16) on a few prompts."""
    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen3VLForConditionalGeneration

    out = Path(out)
    snap = snapshot(snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(out / "tokenizer")

    quantized = Qwen3VLForConditionalGeneration.from_pretrained(
        out / "text_encoder", dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    q = [_encode(quantized, tokenizer, p, "cuda") for p in PROMPTS]
    del quantized
    gc.collect()
    torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory() as work:
        source = filtered_text_encoder_source(snap / "text_encoder", work)
        config = AutoConfig.from_pretrained(source)
        config.text_config.num_hidden_layers = KEEP_LAYERS
        reference = Qwen3VLForConditionalGeneration.from_pretrained(
            source, config=config, dtype=torch.bfloat16, device_map="auto",
            max_memory={0: "10GiB", "cpu": "90GiB"},
        ).eval()
        r = [_encode(reference, tokenizer, p, "cuda") for p in PROMPTS]
    for prompt, a, b in zip(PROMPTS, q, r):
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        rel = (a - b).norm() / b.norm()
        print(
            f"tokens={len(a):3d} cos mean={cosine.mean():.5f} min={cosine.min():.5f} "
            f"rel_err={rel:.4f}  {prompt[:50]!r}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument("--repo", default=REPO, help="HF repo id or local snapshot directory")
    parser.add_argument("--out", type=Path, default=Path("models/minimax-h3-image"))
    args = parser.parse_args(argv)
    (prepare if args.command == "prepare" else verify)(args.repo, args.out)


if __name__ == "__main__":
    main()

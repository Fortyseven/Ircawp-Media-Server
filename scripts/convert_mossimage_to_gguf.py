"""Convert MOSS-Image transformer tensors to an unquantized GGUF without renaming them.

Matrices (2-D+) are written as F16, vectors (norms, biases, RoPE bands) as F32.
The qwen2 architecture key is spoofed only so llama-quantize accepts the file;
it does not describe the diffusion model. The backend loads tensors by name and
ignores the language-model metadata. Dimensions below are MOSS-Image's own
(dim 3072, 42 layers); do not reuse them for another architecture.
"""

import argparse
import json
from pathlib import Path

import gguf
from huggingface_hub import hf_hub_download
import safetensors
import torch

DIM = 3072
NUM_LAYERS = 42
INDEX_NAME = "transformer/diffusion_pytorch_model.safetensors.index.json"


def _resolve(repo, filename):
    local = Path(repo)
    if local.is_dir():
        return str(local / filename)
    return hf_hub_download(repo_id=repo, filename=filename)


def convert(repo, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(_resolve(repo, INDEX_NAME), encoding="utf-8") as index_file:
        index = json.load(index_file)

    model_name = Path(repo).name if Path(repo).is_dir() else repo.rsplit("/", 1)[-1]
    writer = gguf.GGUFWriter(str(out), "qwen2", use_temp_file=True)
    tensor_count = 0
    try:
        writer.add_name(f"{model_name} F16")
        writer.add_source_repo_url(f"https://huggingface.co/{repo}")
        writer.add_description(
            f"{model_name} transformer tensors converted to F16/F32; qwen2 "
            "metadata is spoofed only for llama-quantize compatibility."
        )
        writer.add_context_length(4096)
        writer.add_embedding_length(DIM)
        writer.add_block_count(NUM_LAYERS)
        writer.add_layer_norm_rms_eps(1e-6)

        for shard_name in dict.fromkeys(index["weight_map"].values()):
            shard_path = _resolve(repo, f"transformer/{shard_name}")
            with safetensors.safe_open(
                shard_path, framework="pt", device="cpu"
            ) as shard:
                for name in shard.keys():
                    source = shard.get_tensor(name)
                    if source.ndim >= 2:
                        tensor = source.to(dtype=torch.float16)
                        raw_dtype = gguf.GGMLQuantizationType.F16
                        # BF16 has a wider range than F16; overflow would corrupt weights.
                        if not torch.isfinite(tensor).all():
                            raise ValueError(f"{name} overflows F16")
                    else:
                        tensor = source.to(dtype=torch.float32)
                        raw_dtype = gguf.GGMLQuantizationType.F32
                    writer.add_tensor(
                        name,
                        tensor.numpy(),
                        raw_shape=tensor.shape,
                        raw_dtype=raw_dtype,
                    )
                    tensor_count += 1
                    del source, tensor

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    print(f"Wrote {tensor_count} tensors to {out} ({out.stat().st_size} bytes)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="OpenMOSS-Team/MOSS-Image-SFT")
    parser.add_argument(
        "--out", type=Path, default=Path("models/moss_image_sft_F16.gguf")
    )
    args = parser.parse_args(argv)
    convert(args.repo, args.out)


if __name__ == "__main__":
    main()

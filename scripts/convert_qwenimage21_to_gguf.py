"""Convert Qwen-Image Diffusers tensors to F16 GGUF without renaming them.

The qwen2 architecture and dimensions are spoofed only for llama-quantize
compatibility, not a native description of the diffusion model. Diffusers
consumers load tensors by name and ignore these language-model metadata values.
"""

import argparse
import json
from pathlib import Path

import gguf
from huggingface_hub import hf_hub_download
import safetensors
import torch


def convert(repo, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    index_path = hf_hub_download(
        repo_id=repo,
        filename="transformer/diffusion_pytorch_model.safetensors.index.json",
    )
    with open(index_path, encoding="utf-8") as index_file:
        index = json.load(index_file)

    model_name = repo.rsplit("/", 1)[-1]
    writer = gguf.GGUFWriter(str(out), "qwen2", use_temp_file=True)
    tensor_count = 0
    try:
        writer.add_name(f"{model_name} F16")
        writer.add_source_repo_url(f"https://huggingface.co/{repo}")
        writer.add_description(
            f"{model_name} Diffusers tensors converted to F16; qwen2 metadata "
            "is spoofed only for llama-quantize compatibility."
        )
        writer.add_context_length(4096)
        writer.add_embedding_length(4096)
        writer.add_block_count(32)
        writer.add_layer_norm_rms_eps(1e-6)

        for shard_name in dict.fromkeys(index["weight_map"].values()):
            shard_path = hf_hub_download(
                repo_id=repo,
                filename=f"transformer/{shard_name}",
            )
            with safetensors.safe_open(
                shard_path,
                framework="pt",
                device="cpu",
            ) as shard:
                for name in shard.keys():
                    tensor = shard.get_tensor(name).to(dtype=torch.float16)
                    array = tensor.numpy()
                    writer.add_tensor(
                        name,
                        array,
                        raw_shape=tensor.shape,
                        raw_dtype=gguf.GGMLQuantizationType.F16,
                    )
                    tensor_count += 1
                    del array, tensor

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    print(f"Wrote {tensor_count} tensors to {out} ({out.stat().st_size} bytes)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen-Image-2.1-Turbo")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/qwen_image_2.1_turbo_F16.gguf"),
    )
    args = parser.parse_args(argv)
    convert(args.repo, args.out)


if __name__ == "__main__":
    main()

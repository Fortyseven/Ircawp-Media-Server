"""MOSS-Image-SFT 8-step data-space DMD backend (lodestones/MOSS-Image-SFT-8-steps-dmd).

Reuses the base `mossimage` loading (local Q4_K_S GGUF of the SFT transformer,
4-bit text encoder, FLUX.2 VAE) and applies the distillation adapters unmerged:
a rank-128 LoRA, fp32 time-embedding modules and the data-space sampler state.
The distillation delta is below bf16 resolution, so it cannot be merged into a
quantized checkpoint. Text-to-image only; no CFG.
"""

import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from .mossimage import mossimage


ADAPTER_REPO = "lodestones/MOSS-Image-SFT-8-steps-dmd"
ADAPTER_FILES = (
    "lora.safetensors",
    "time_embedding.safetensors",
    "dataspace.safetensors",
)
LORA_TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
DEFAULT_STEPS = 8


class _LoRALinear(nn.Module):
    """A (possibly quantized) linear plus an unmerged fp32 low-rank branch."""

    def __init__(self, base, down, up, scale):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(down, requires_grad=False)
        self.up = nn.Parameter(up, requires_grad=False)
        self.scale = scale

    # MOSS casts activations to the wrapped layer's `weight.dtype`.
    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        delta = F.linear(F.linear(x.to(self.down.dtype), self.down), self.up)
        return out + (delta * self.scale).to(out.dtype)


def _fp32_linear(weight, bias):
    layer = nn.Linear(weight.shape[1], weight.shape[0])
    layer.weight = nn.Parameter(weight.float(), requires_grad=False)
    layer.bias = nn.Parameter(bias.float(), requires_grad=False)
    return layer


def _apply_adapters(transformer, lora, time_modules):
    """Wrap every targeted layer with its LoRA branch and install fp32 time modules."""
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

    expected_lora = {
        f"diffusion_model.layers.{index}.proj.{name}.{suffix}"
        for index in range(len(transformer.layers))
        for name in LORA_TARGETS
        for suffix in ("lora_down.weight", "lora_up.weight", "alpha")
    }
    if set(lora) != expected_lora:
        missing = sorted(expected_lora - set(lora))
        unexpected = sorted(set(lora) - expected_lora)
        raise ValueError(
            f"LoRA does not match the transformer: missing {missing[:5]}, "
            f"unexpected {unexpected[:5]}"
        )

    for index, layer in enumerate(transformer.layers):
        for name in LORA_TARGETS:
            prefix = f"diffusion_model.layers.{index}.proj.{name}"
            down = lora[f"{prefix}.lora_down.weight"]
            scale = float(lora[f"{prefix}.alpha"]) / down.shape[0]
            layer.proj[name] = _LoRALinear(
                layer.proj[name], down, lora[f"{prefix}.lora_up.weight"], scale
            )

    expected = {
        "time_embedding.0.weight",
        "time_embedding.0.bias",
        "time_embedding.2.weight",
        "time_embedding.2.bias",
        "shared_block_modulation.bias",
        "video_head_modulation.bias",
    }
    if set(time_modules) != expected:
        raise ValueError("time_embedding.safetensors has unexpected tensors")

    for position in (0, 2):
        transformer.time_embedding[position] = _fp32_linear(
            time_modules[f"time_embedding.{position}.weight"],
            time_modules[f"time_embedding.{position}.bias"],
        )
    # The t=1 time vector is folded into these biases, so they must stay fp32.
    for name in ("shared_block_modulation", "video_head_modulation"):
        old = getattr(transformer, name)
        setattr(
            transformer,
            name,
            _fp32_linear(
                dequantize_gguf_tensor(old.weight), time_modules[f"{name}.bias"]
            ),
        )
    return transformer


class mossimage8(mossimage):
    model_id = "mossimage8"

    def __init__(self, backend_config: dict = {}):
        adapter = os.fspath(backend_config.get("adapter", ADAPTER_REPO))
        if os.path.isdir(adapter):
            paths = {name: os.path.join(adapter, name) for name in ADAPTER_FILES}
            for path in paths.values():
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
        else:
            paths = {name: hf_hub_download(adapter, name) for name in ADAPTER_FILES}
        self._adapter_paths = paths
        super().__init__(backend_config)

    def _prepare_transformer(self, transformer):
        state = load_file(self._adapter_paths["dataspace.safetensors"])
        self.mix = state["latent_mix.Q"]
        self.step_codes = state["step_codes"]
        return _apply_adapters(
            transformer,
            load_file(self._adapter_paths["lora.safetensors"]),
            load_file(self._adapter_paths["time_embedding.safetensors"]),
        )

    def _decode(self, latents):
        pipe = self.pipe
        z = latents.to(dtype=pipe.vae.dtype)
        bn = pipe.vae.bn
        eps = pipe.vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(
            1, -1, 1, 1
        ) + bn.running_mean.to(z).view(1, -1, 1, 1)
        b, c, h, w = z.shape
        z = (
            z.reshape(b, c // 4, 2, 2, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(b, c // 4, 2 * h, 2 * w)
        )
        image = pipe.vae.decode(z, return_dict=False)[0].clamp(-1, 1)
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    @torch.no_grad()
    def execute(
        self,
        prompt: str,
        config: dict = {},
        batch_id=None,
        media=[],
    ) -> tuple[str, str]:
        output_file, seed, width, height = self._plan_request(config, batch_id, media)
        final_prompt = prompt.strip()
        # Each prefix of the 8 learned step codes is a trained sampler.
        steps = min(config.get("steps") or DEFAULT_STEPS, len(self.step_codes))

        pipe = self.pipe
        observe = self.cancellation_callback(config, steps)
        self.raise_if_cancelled(config)
        embeds, mask = pipe.encode_prompt(final_prompt)
        device = pipe._execution_device
        noise = torch.randn(
            (1, pipe.transformer.config.latent_channels, height // 16, width // 16),
            generator=torch.Generator("cpu").manual_seed(seed),
            dtype=torch.float32,
        ).to(device)
        mix = self.mix.to(device)
        x0 = torch.zeros_like(noise)
        for step in range(steps):
            self.raise_if_cancelled(config)
            x = noise + torch.einsum("oc,bchw->bohw", mix, x0)
            code = self.step_codes[step].to(device)
            velocity = pipe.transformer(x, code, embeds, mask).sample.float()
            x0 = noise - velocity
            observe(pipe, step, code, {})
        self.raise_if_cancelled(config)
        output_image = self._decode(x0)
        pipe.maybe_free_model_hooks()

        self._save_image_with_metadata(
            output_image,
            output_file,
            final_prompt,
            seed=seed,
            model=self.model_id,
            inference_steps=steps,
            width=width,
            height=height,
        )
        self.last_imagegen_prompt = final_prompt
        return output_file, final_prompt

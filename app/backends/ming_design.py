"""Ming-Image-0.1-Design backend for media-server.

Text-to-image, or single-reference image editing when one media path is given.
Uses community GGUFs for the Ling-Mini text encoder (staged through the GPU
only while encoding) and the DiT, plus the official VAE and config from
`base_dir`. Model plumbing lives in ming_loader.py.

Prompt refinement is handled by the client (ircawp bot).
This backend receives a final, ready-to-use prompt.
"""

from .MediaBackend import MediaBackend
from .ming_loader import (
    load_encoder,
    load_generator,
    plan_size,
    prepare_reference,
    sample_image,
)

import os
import random

import torch
from PIL import Image


MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "ming-image",
)
DEFAULT_ENCODER_GGUF = os.path.join(
    MODELS_DIR, "Text Encoders (UPDATED)", "Ming-Image-0.1-Ling-Mini-2.0-Q4_K_M.gguf"
)
DEFAULT_DIT_GGUF = os.path.join(MODELS_DIR, "Ming-Image-0.1-Design-Q4_K_M.gguf")
DEFAULT_BASE_DIR = os.path.join(MODELS_DIR, "official")
VAE_FILE = os.path.join("vae", "diffusion_pytorch_model.safetensors")
DEFAULT_ASPECT = 1.5
DEFAULT_MAX_OUTPUT_SIZE = 1024
DEFAULT_STEPS = 12
DEFAULT_GUIDANCE = 1.0


def _local_path(value, label: str) -> str:
    path = os.fspath(value)
    if "://" in path or path.lower().startswith(("hf.co/", "huggingface.co/")):
        raise ValueError(f"ming_design requires a local {label}, not a remote URL")
    return os.path.abspath(path)


def _release_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ming_design(MediaBackend):
    model_id = "ming_design"

    def __init__(self, backend_config: dict = {}):
        backend_config = dict(backend_config)
        super().__init__(backend_config)

        encoder_gguf = _local_path(
            backend_config.get("encoder_gguf", DEFAULT_ENCODER_GGUF), "encoder GGUF"
        )
        dit_gguf = _local_path(
            backend_config.get("dit_gguf", DEFAULT_DIT_GGUF), "DiT GGUF"
        )
        base_dir = _local_path(
            backend_config.get("base_dir", DEFAULT_BASE_DIR), "base directory"
        )
        vae_path = os.path.join(base_dir, VAE_FILE)
        for path in (encoder_gguf, dit_gguf, vae_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

        self.code_dir = backend_config.get("code_dir")
        self.encoder = load_encoder(
            encoder_gguf, base_dir, code_dir=self.code_dir, device="cpu"
        )
        self.generator = load_generator(
            dit_gguf, vae_path, base_dir, code_dir=self.code_dir, device="cuda"
        )

    @staticmethod
    def _parse_aspect(config: dict) -> float:
        aspect = config.get("aspect", DEFAULT_ASPECT)
        if isinstance(aspect, str):
            if ":" in aspect:
                parts = aspect.split(":")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    return float(parts[0]) / float(parts[1])
            try:
                return float(aspect)
            except ValueError:
                return DEFAULT_ASPECT
        return float(aspect) if isinstance(aspect, (int, float)) else DEFAULT_ASPECT

    def _plan_request(self, config: dict, batch_id, media):
        """Return (output_file, seed, width, height, reference); size is None for edits without both."""
        reference = None
        if len(media) > 1:
            raise ValueError(
                f"{self.model_id} edits take exactly one reference image, got {len(media)}"
            )
        if media:
            reference = os.fspath(media[0])
            if not os.path.isfile(reference):
                raise FileNotFoundError(reference)

        if batch_id is not None:
            output_file = f"/tmp/ircawp_generated/{self.model_id}.{batch_id}.png"
        else:
            output_file = config.get(
                "output_file", f"/tmp/ircawp_generated/{self.model_id}.png"
            )

        seed = config.get("seed")
        if seed is None:
            # stdlib RNG keeps global torch state untouched.
            seed = random.randrange(1000000)

        if "width" in config and "height" in config:
            width, height = config["width"], config["height"]
        elif reference:
            width = height = None
        else:
            max_output_size = config.get(
                "max_output_size",
                self.backend_config.get("max_output_size", DEFAULT_MAX_OUTPUT_SIZE),
            )
            aspect = self._parse_aspect(config)
            if aspect >= 1.0:
                width, height = max_output_size, int(max_output_size / aspect)
            else:
                width, height = int(max_output_size * aspect), max_output_size
            width = round(width / 16) * 16
            height = round(height / 16) * 16
        return output_file, seed, width, height, reference

    def execute(
        self,
        prompt: str,
        config: dict = {},
        batch_id=None,
        media=[],
    ) -> tuple[str, str]:
        self.raise_if_cancelled(config)
        output_file, seed, width, height, reference = self._plan_request(
            config, batch_id, media
        )

        final_prompt = prompt.strip()
        steps = config.get("steps") or self.backend_config.get("steps") or DEFAULT_STEPS
        guidance_scale = config.get("guidance_scale", DEFAULT_GUIDANCE)

        encode_kwargs = {}
        sample_kwargs = {}
        metadata = {}
        if reference:
            with Image.open(reference) as opened:
                opened.load()
                plan = prepare_reference(
                    opened, width=width, height=height, code_dir=self.code_dir
                )
            encode_kwargs = {"image": plan.image}
            sample_kwargs = {"ref_x": plan.ref_x}
            metadata = {"task": "image-edit"}
            internal_w, internal_h = plan.width, plan.height
            if width is None:
                width, height = plan.width, plan.height
        else:
            internal_w, internal_h = plan_size(width, height, code_dir=self.code_dir)

        self.encoder.to("cuda")
        try:
            context, directvlm = self.encoder.encode(final_prompt, **encode_kwargs)
        finally:
            self.encoder.to("cpu")
            _release_cuda_cache()

        self.raise_if_cancelled(config)
        try:
            image = sample_image(
                self.generator,
                context,
                directvlm,
                steps=steps,
                cfg=guidance_scale,
                seed=seed,
                height=internal_h,
                width=internal_w,
                callback_on_step_end=self.cancellation_callback(config, steps),
                **sample_kwargs,
            )
        finally:
            # An idle server would otherwise keep the sampling peak reserved.
            _release_cuda_cache()
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)

        self._save_image_with_metadata(
            image,
            output_file,
            final_prompt,
            seed=seed,
            model=self.model_id,
            inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            **metadata,
        )

        self.last_imagegen_prompt = final_prompt
        return output_file, final_prompt

    def dispose(self) -> None:
        self.encoder = None
        self.generator = None
        super().dispose()

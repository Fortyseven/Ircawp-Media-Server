"""Local GGUF FLUX.3 Klein distilled backend."""

import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from types import MethodType

import torch
from diffusers import (
    Flux3DistilledModularPipeline,
    Flux3Transformer2DModel,
    GGUFQuantizationConfig,
)
from diffusers.utils import load_image

from .MediaBackend import MediaBackend


BASE_MODEL = "black-forest-labs/flux-3-image-klein"
DEFAULT_GGUF = str(
    Path(__file__).resolve().parents[2] / "models" / "flux3_image_klein_sd_Q5_K_M.gguf"
)
DEFAULT_FILENAME = "/tmp/ircawp_generated/flux3klein.png"
DEFAULT_STEPS = 16
DEFAULT_MAX_OUTPUT_SIZE = 1024


class _Flux3ProgressBar:
    def __init__(self, backend, config: dict, total: int):
        self.backend = backend
        self.config = config
        self.total = total
        self.progress_state = config.get("progress_state")

    def __enter__(self):
        if self.progress_state is not None:
            self.progress_state["step"] = 0
            self.progress_state["total_steps"] = self.total
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, count=1):
        self.backend.raise_if_cancelled(self.config)
        if self.progress_state is not None:
            self.progress_state["step"] = min(
                self.total, self.progress_state.get("step", 0) + count
            )


class flux3klein(MediaBackend):
    def __init__(self, backend_config: dict = {}):
        backend_config = dict(backend_config)
        gguf_path = os.fspath(backend_config.get("gguf", DEFAULT_GGUF))
        if "://" in gguf_path or gguf_path.lower().startswith(
            ("hf.co/", "huggingface.co/")
        ):
            raise ValueError(
                "FLUX.3 Klein requires a local GGUF file, not a remote URL"
            )
        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(gguf_path)
        backend_config["gguf"] = os.path.abspath(gguf_path)
        backend_config.setdefault("base_model", BASE_MODEL)
        super().__init__(backend_config)
        self._inference_lock = Lock()

        transformer = Flux3Transformer2DModel.from_single_file(
            self.backend_config["gguf"],
            config=self.backend_config["base_model"],
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        )
        self.pipe = Flux3DistilledModularPipeline.from_pretrained(
            self.backend_config["base_model"],
            transformer=transformer,
        )
        self.pipe.load_components(dtype=torch.bfloat16)
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _parse_aspect(config: dict) -> float:
        aspect = config.get("aspect", 1.0)
        if isinstance(aspect, str):
            if ":" in aspect:
                numerator, denominator, *rest = aspect.split(":")
                if not rest:
                    try:
                        return float(numerator) / float(denominator)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
            try:
                return float(aspect)
            except ValueError:
                pass
        return float(aspect) if isinstance(aspect, (int, float)) else 1.0

    @contextmanager
    def _report_denoising_progress(self, config: dict):
        originals = []

        def install(block):
            sub_blocks = getattr(block, "sub_blocks", None)
            if isinstance(sub_blocks, Mapping):
                for sub_block in sub_blocks.values():
                    install(sub_block)

            if not (
                block.__class__.__module__.endswith("modular_pipelines.flux3.denoise")
                and hasattr(block, "progress_bar")
            ):
                return
            original = block.progress_bar

            def progress_bar(instance, iterable=None, total=None):
                if total is None:
                    return original(iterable=iterable, total=total)
                return _Flux3ProgressBar(self, config, total)

            originals.append((block, original))
            block.progress_bar = MethodType(progress_bar, block)

        blocks = getattr(self.pipe, "_blocks", None)
        if blocks is not None:
            install(blocks)
        try:
            yield
        finally:
            for block, original in originals:
                block.progress_bar = original

    def execute(
        self,
        prompt: str,
        config: dict = {},
        batch_id=None,
        media=[],
    ) -> tuple[str, str]:
        config = dict(config)
        output_file = (
            f"/tmp/ircawp_generated/flux3klein.{batch_id}.png"
            if batch_id is not None
            else config.get("output_file", DEFAULT_FILENAME)
        )
        seed = config.get("seed")
        if seed is None:
            seed = torch.randint(0, 1_000_000, (1,)).item()
        images = [load_image(image_path) for image_path in media]
        max_output_size = config.get("max_output_size", DEFAULT_MAX_OUTPUT_SIZE)
        if "width" in config and "height" in config:
            width, height = config["width"], config["height"]
        elif images:
            source_width, source_height = images[0].size
            scale = max_output_size / max(source_width, source_height)
            width = round(source_width * scale / 16) * 16
            height = round(source_height * scale / 16) * 16
        else:
            aspect = self._parse_aspect(config)
            if aspect >= 1.0:
                width = max_output_size
                height = round(max_output_size / aspect / 16) * 16
            else:
                width = round(max_output_size * aspect / 16) * 16
                height = max_output_size
        final_prompt = prompt.strip()
        self.raise_if_cancelled(config)
        progress_state = config.get("progress_state")
        if progress_state is not None:
            progress_state["step"] = 0
            progress_state["total_steps"] = DEFAULT_STEPS

        with self._inference_lock, self._report_denoising_progress(config):
            output_image = self.pipe(
                prompt=final_prompt,
                width=width,
                height=height,
                num_inference_steps=DEFAULT_STEPS,
                generator=torch.Generator("cpu").manual_seed(seed),
                image=images if images else None,
                output="images",
            )[0]
        self.raise_if_cancelled(config)
        if progress_state is not None:
            progress_state["step"] = DEFAULT_STEPS
        self._save_image_with_metadata(
            output_image,
            output_file,
            final_prompt,
            seed=seed,
            model="flux3klein",
            inference_steps=DEFAULT_STEPS,
            width=width,
            height=height,
        )
        self.last_imagegen_prompt = final_prompt
        return output_file, final_prompt

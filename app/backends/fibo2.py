"""Fibo 2.0 (Bria) backend: text-to-image and multi-image editing.

Loads the official gated briaai diffusers checkpoint and quantizes the
transformer to 4-bit (bitsandbytes NF4) on load. See docs/FIBO2.md.

Prompt refinement is handled by the client. Plain-text prompts reach the model
as a short description (or, when editing, as the edit instruction).
"""

import os

import torch
from diffusers import BriaFibo2Pipeline
from diffusers.utils import load_image

from .MediaBackend import MediaBackend


BASE_MODEL = "briaai/Fibo-2.0-dev"
DEFAULT_ASPECT = 1.5
DEFAULT_MAX_OUTPUT_SIZE = 1024
DEFAULT_STEPS = 30
DEFAULT_CFG = 5.0
MAX_EDIT_IMAGES = 5

# Negative caption from the Fibo 2.0 model card for text-to-image; edits use the
# pipeline's null caption instead.
NEGATIVE_T2I = (
    '{"short_description":"","objects":[{"description":"the main subject",'
    '"texture":"smooth, featureless, plastic-like, airbrushed, no fine surface detail",'
    '"appearance_details":"soft blurred edges, simplified shapes, no fine detail",'
    '"skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],'
    '"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, '
    'blurred and featureless, no visible texture","lighting":{},"aesthetics":{"artifacts":'
    '["oversaturated colours","neon colour cast","overexposed highlights","lens flare",'
    '"slight blur","motion blur","haze"]},"photographic_characteristics":'
    '{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame","focus":'
    '"soft, slightly out of focus, lacking fine texture"},"style_medium":"","context":"",'
    '"artistic_style":""}'
)


class fibo2(MediaBackend):
    model_id = "fibo2"
    base_model = BASE_MODEL
    default_steps = DEFAULT_STEPS
    default_cfg = DEFAULT_CFG
    edit_only = False
    # Flow-matching with a resolution-shifted schedule inside the pipeline.
    scheduler_family = None

    def __init__(self, backend_config: dict = {}):
        super().__init__(dict(backend_config))

        base_model = self.backend_config.get("base_model", self.base_model)
        transformer_4bit = self.backend_config.get("transformer_4bit", True)
        text_encoder_4bit = self.backend_config.get("text_encoder_4bit", False)

        pipe_kwargs = {"torch_dtype": torch.bfloat16}
        quant_mapping = {}
        if transformer_4bit:
            from diffusers import BitsAndBytesConfig as DiffusersBnbConfig

            quant_mapping["transformer"] = DiffusersBnbConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if text_encoder_4bit:
            from transformers.utils.quantization_config import BitsAndBytesConfig

            quant_mapping["text_encoder"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if quant_mapping:
            from diffusers.quantizers import PipelineQuantizationConfig

            pipe_kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_mapping=quant_mapping
            )

        self.pipe = BriaFibo2Pipeline.from_pretrained(base_model, **pipe_kwargs)
        self.pipe.enable_model_cpu_offload()

    def _resolve_steps(self, config: dict) -> int:
        return int(config.get("steps") or self.default_steps)

    def _resolve_cfg(self, config: dict) -> float:
        cfg = config.get("guidance_scale")
        return float(cfg) if cfg is not None else self.default_cfg

    @staticmethod
    def _parse_aspect(aspect) -> float:
        if isinstance(aspect, str):
            if ":" in aspect:
                parts = aspect.split(":")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    return float(parts[0]) / float(parts[1])
            try:
                return float(aspect)
            except ValueError:
                return DEFAULT_ASPECT
        if isinstance(aspect, (int, float)) and aspect > 0:
            return float(aspect)
        return DEFAULT_ASPECT

    @staticmethod
    def _fit_longest_side(aspect: float, max_output_size: int) -> tuple[int, int]:
        if aspect >= 1.0:
            width, height = max_output_size, max_output_size / aspect
        else:
            width, height = max_output_size * aspect, max_output_size
        return max(16, round(width / 16) * 16), max(16, round(height / 16) * 16)

    def execute(
        self,
        prompt: str,
        config: dict = {},
        batch_id=None,
        media=[],
    ) -> tuple[str, str]:
        media = list(media)
        if self.edit_only and not media:
            raise ValueError(
                f"{self.model_id} only edits images; provide an input image"
            )
        if len(media) > MAX_EDIT_IMAGES:
            raise ValueError(
                f"{self.model_id} accepts at most {MAX_EDIT_IMAGES} input images"
            )

        if batch_id is not None:
            output_file = f"/tmp/ircawp_generated/{self.model_id}.{batch_id}.png"
        else:
            output_file = config.get(
                "output_file", f"/tmp/ircawp_generated/{self.model_id}.png"
            )
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        seed = config.get("seed")
        if seed is None:
            seed = torch.randint(0, 1000000, (1,)).item()
        steps = self._resolve_steps(config)
        guidance_scale = self._resolve_cfg(config)
        max_output_size = config.get("max_output_size", DEFAULT_MAX_OUTPUT_SIZE)

        media_pil = [load_image(path) for path in media]

        if "width" in config and "height" in config:
            width, height = config["width"], config["height"]
        elif media_pil:
            first = media_pil[0]
            width, height = self._fit_longest_side(
                first.width / first.height, max_output_size
            )
        else:
            width, height = self._fit_longest_side(
                self._parse_aspect(config.get("aspect", DEFAULT_ASPECT)),
                max_output_size,
            )

        final_prompt = prompt.strip()

        pipeline_kwargs = {
            "prompt": final_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": torch.Generator("cpu").manual_seed(seed),
            "callback_on_step_end": self.cancellation_callback(config, steps),
        }
        if media_pil:
            pipeline_kwargs["image"] = media_pil
        elif guidance_scale > 1.0:
            pipeline_kwargs["negative_prompt"] = NEGATIVE_T2I

        output_image = self.pipe(**pipeline_kwargs).images[0]
        self.raise_if_cancelled(config)

        self._save_image_with_metadata(
            output_image,
            output_file,
            final_prompt,
            seed=seed,
            model=self.model_id,
            inference_steps=steps,
            guidance_scale=guidance_scale,
            width=output_image.width,
            height=output_image.height,
            aspect_ratio=str(output_image.width / output_image.height),
        )

        self.last_imagegen_prompt = final_prompt
        return output_file, final_prompt

"""MOSS-Image-SFT backend for media-server.

Text-to-image only. Uses a local GGUF 4-bit quantized transformer (made with
scripts/convert_mossimage_to_gguf.py + llama-quantize) with the Qwen3.5 text
encoder (4-bit bitsandbytes), FLUX.2 VAE, tokenizer and scheduler from the
OpenMOSS-Team/MOSS-Image-SFT repo. The pipeline and transformer classes ship
as custom code (pipeline_moss_image.py) inside that repo.

Prompt refinement is handled by the client (ircawp bot).
This backend receives a final, ready-to-use prompt.
"""

from .MediaBackend import MediaBackend

import importlib.util
import os
import sys

import torch
from huggingface_hub import snapshot_download


BASE_MODEL = "OpenMOSS-Team/MOSS-Image-SFT"
DEFAULT_GGUF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "moss_image_sft_Q4_K_S.gguf",
)
PIPELINE_FILE = "pipeline_moss_image.py"
DEFAULT_ASPECT = 1.5
DEFAULT_MAX_OUTPUT_SIZE = 1024
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 3.0


def _load_pipeline_module(base_dir: str):
    """Import the custom pipeline/transformer code shipped with the checkpoint."""
    path = os.path.join(base_dir, PIPELINE_FILE)
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_moss_image"] = module
    spec.loader.exec_module(module)
    return module


def _make_dtype_reporting_parameter(compute_dtype):
    from diffusers.quantizers.gguf.utils import GGUFParameter

    class DtypeReportingGGUFParameter(GGUFParameter):
        # MOSS casts activations to `weight.dtype`, but packed GGUF bytes report uint8.
        @property
        def dtype(self):
            return compute_dtype

    return DtypeReportingGGUFParameter


def _load_gguf_transformer(module, gguf_path, base_dir, compute_dtype=torch.bfloat16):
    """Build the MOSS transformer on the meta device and fill it from a GGUF file.

    The custom class is not in diffusers' single-file registry, so this repeats
    the registry-free steps of FromOriginalModelMixin.from_single_file.
    """
    from accelerate import init_empty_weights
    from diffusers.loaders.single_file_utils import load_single_file_checkpoint
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    from diffusers.quantizers import DiffusersAutoQuantizer
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter
    from diffusers.quantizers.quantization_config import GGUFQuantizationConfig

    transformer_class = module.MOSSImageTransformer
    config = transformer_class.load_config(base_dir, subfolder="transformer")
    checkpoint = load_single_file_checkpoint(gguf_path)

    with init_empty_weights():
        model = transformer_class.from_config(config)

    expected = set(model.state_dict())
    missing = sorted(expected - set(checkpoint))
    unexpected = sorted(set(checkpoint) - expected)
    if missing or unexpected:
        raise ValueError(
            f"GGUF does not match the transformer: missing {missing[:5]}, "
            f"unexpected {unexpected[:5]}"
        )

    quantizer = DiffusersAutoQuantizer.from_config(
        GGUFQuantizationConfig(compute_dtype=compute_dtype)
    )
    quantizer.validate_environment()
    quantizer.preprocess_model(
        model=model, device_map=None, state_dict=checkpoint, keep_in_fp32_modules=[]
    )
    load_model_dict_into_meta(
        model,
        checkpoint,
        dtype=compute_dtype,
        device_map={"": torch.device("cpu")},
        hf_quantizer=quantizer,
        keep_in_fp32_modules=[],
        unexpected_keys=[],
    )
    quantizer.postprocess_model(model)
    model.hf_quantizer = quantizer

    parameter_class = _make_dtype_reporting_parameter(compute_dtype)
    for submodule in model.modules():
        if isinstance(submodule, GGUFLinear) and isinstance(
            submodule.weight, GGUFParameter
        ):
            weight = submodule.weight
            submodule.weight = parameter_class(
                weight.as_tensor(), requires_grad=False, quant_type=weight.quant_type
            )
    return model.eval()


class mossimage(MediaBackend):
    model_id = "mossimage"

    def __init__(self, backend_config: dict = {}):
        backend_config = dict(backend_config)
        super().__init__(backend_config)

        gguf_path = os.fspath(backend_config.get("gguf", DEFAULT_GGUF))
        if "://" in gguf_path or gguf_path.lower().startswith(
            ("hf.co/", "huggingface.co/")
        ):
            raise ValueError("mossimage requires a local GGUF file, not a remote URL")
        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(gguf_path)
        gguf_path = os.path.abspath(gguf_path)

        text_encoder_4bit = backend_config.get("text_encoder_4bit", True)
        base_model = backend_config.get("base_model", BASE_MODEL)
        if os.path.isdir(base_model):
            base_dir = os.fspath(base_model)
        else:
            # The transformer comes from the local GGUF, so skip its weights.
            base_dir = snapshot_download(
                base_model, ignore_patterns=["transformer/*.safetensors", "assets/*"]
            )

        from diffusers import AutoencoderKLFlux2
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils.quantization_config import BitsAndBytesConfig

        module = _load_pipeline_module(base_dir)
        transformer = _load_gguf_transformer(module, gguf_path, base_dir)

        encoder_kwargs = {"dtype": torch.bfloat16}
        if text_encoder_4bit:
            encoder_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        text_encoder = AutoModel.from_pretrained(
            os.path.join(base_dir, "text_encoder"), **encoder_kwargs
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
        # MOSS decodes latents in float32.
        vae = AutoencoderKLFlux2.from_pretrained(
            base_dir, subfolder="vae", torch_dtype=torch.float32
        ).eval()
        scheduler = module.MOSSImageScheduler.from_pretrained(
            base_dir, subfolder="scheduler"
        )

        self.pipe = module.MOSSImagePipeline(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )
        self.pipe.enable_model_cpu_offload()

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

    def execute(
        self,
        prompt: str,
        config: dict = {},
        batch_id=None,
        media=[],
    ) -> tuple[str, str]:
        if media:
            raise ValueError(
                "mossimage is text-to-image only; image editing is not supported"
            )

        if batch_id is not None:
            output_file = f"/tmp/ircawp_generated/{self.model_id}.{batch_id}.png"
        else:
            output_file = config.get(
                "output_file", f"/tmp/ircawp_generated/{self.model_id}.png"
            )

        seed = config.get("seed")
        if seed is None:
            seed = torch.randint(0, 1000000, (1,)).item()

        if "width" in config and "height" in config:
            width, height = config["width"], config["height"]
        else:
            max_output_size = config.get("max_output_size", DEFAULT_MAX_OUTPUT_SIZE)
            aspect = self._parse_aspect(config)
            if aspect >= 1.0:
                width, height = max_output_size, int(max_output_size / aspect)
            else:
                width, height = int(max_output_size * aspect), max_output_size
            width = round(width / 16) * 16
            height = round(height / 16) * 16

        final_prompt = prompt.strip()
        requested_steps = config.get("steps") or DEFAULT_STEPS
        guidance_scale = config.get("guidance_scale", DEFAULT_GUIDANCE)

        # The schedule snaps to the training grid and may run fewer steps than requested.
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(requested_steps)
        steps = len(scheduler.timesteps)

        observe = self.cancellation_callback(config, steps)
        original_step = scheduler.step
        step_index = 0

        # The custom pipeline has no step callback; the scheduler steps once per step.
        def observed_step(*args, **kwargs):
            nonlocal step_index
            self.raise_if_cancelled(config)
            result = original_step(*args, **kwargs)
            observe(self.pipe, step_index, args[1] if len(args) > 1 else None, {})
            step_index += 1
            return result

        self.raise_if_cancelled(config)
        scheduler.step = observed_step
        try:
            output_image = self.pipe(
                prompt=final_prompt,
                width=width,
                height=height,
                num_inference_steps=requested_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
        finally:
            scheduler.__dict__.pop("step", None)

        self._save_image_with_metadata(
            output_image,
            output_file,
            final_prompt,
            seed=seed,
            model=self.model_id,
            inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )

        self.last_imagegen_prompt = final_prompt
        return output_file, final_prompt

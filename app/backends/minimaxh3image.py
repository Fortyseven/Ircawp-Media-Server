"""MiniMax-H3-Image backend for media-server (text-to-image).

A 4-bit GGUF transformer (made with scripts/convert_minimaxh3image_to_gguf.py +
llama-quantize) runs inside Diffusers' ``MiniMaxH3ImageModularPipeline`` with a
pre-quantized NF4 Qwen3-VL text encoder, the official image VAE, tokenizer,
processor and scheduler, all read from a local folder built by
scripts/prepare_minimaxh3image_components.py.  Nothing is downloaded at run
time.

The checkpoint is distilled to a fixed six-step grid with one output head per
step and no guidance, so ``steps``, guidance and negative prompts do not apply.
Reference-image editing is supported by the pipeline but not exposed here yet.

The GGUF does not contain the per-block ``adaln_proj`` weights.  Their output
only depends on the timestep, and the checkpoint only runs at seven timesteps,
so the converter stored the exact outputs as ``adaln_proj.table``; see
``TabulatedAdaLN``.

Prompt refinement is handled by the client (ircawp bot).
This backend receives a final, ready-to-use prompt.
"""

from .MediaBackend import GenerationCancelled, MediaBackend  # noqa: F401

import os

import torch
from torch import nn


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_GGUF = os.path.join(PROJECT_ROOT, "models", "minimax_h3_image_Q4_K_S.gguf")
DEFAULT_BASE_MODEL = os.path.join(PROJECT_ROOT, "models", "minimax-h3-image")
DEFAULT_ASPECT = 1.0
DEFAULT_MAX_OUTPUT_SIZE = 2048
NUM_STEPS = 6
REQUIRED_SUBFOLDERS = ("vae", "text_encoder", "tokenizer", "processor", "scheduler")
TABLE_SUFFIX = ".adaln_proj.table"
TIMESTEP_TOLERANCE = 1e-5


class TabulatedAdaLN(nn.Module):
    """Replaces a block's ``adaln_proj`` with its precomputed per-timestep output.

    ``table`` is (num_timesteps, 2 * 6 * hidden): the shift/scale/gate rows of the
    image and text modalities for each timestep the checkpoint runs at.  The audio
    modality (never used by the image model) is dropped and returned as zeros so
    the block's ``timestep_index * 3 + modality`` indexing is unchanged.
    """

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("table", table.to(torch.bfloat16), persistent=False)
        self.rows = None

    def forward(self, temb: torch.Tensor):
        if self.rows is None:
            raise RuntimeError("TabulatedAdaLN used before the timestep rows were selected")
        rows = self.table.index_select(0, self.rows.to(self.table.device))
        count, width = rows.shape[0], rows.shape[1] // 2
        rows = rows.view(count, 2, width)
        rows = torch.cat([rows, rows.new_zeros(count, 1, width)], dim=1)
        return rows.reshape(count * 3, width).chunk(6, dim=-1)


def _select_timestep_rows(modules, grid):
    """Forward pre-hook for the transformer: point every table at this call's timesteps."""

    def hook(_module, args, kwargs):
        timestep = kwargs["timestep"] if "timestep" in kwargs else args[2]
        values = timestep.detach().float().cpu().reshape(-1)
        distance = (values[:, None] - grid[None, :]).abs()
        nearest = distance.argmin(dim=1)
        if (distance.gather(1, nearest[:, None]) > TIMESTEP_TOLERANCE).any():
            raise ValueError(
                f"timesteps {values.tolist()} are not on the checkpoint's schedule "
                f"{grid.tolist()}; the GGUF only holds modulation for that schedule"
            )
        for module in modules:
            module.rows = nearest
        return None

    return hook


def _load_gguf_transformer(gguf_path, base_dir=None, compute_dtype=torch.bfloat16, config=None):
    """Build MiniMaxH3ImageTransformer2DModel on the meta device and fill it from a GGUF.

    Repeats the registry-free steps of FromOriginalModelMixin.from_single_file
    (the same approach as the mossimage backend).  The GGUF carries Diffusers
    tensor names, so no renaming is needed.
    """
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3ImageTransformer2DModel
    from diffusers.loaders.single_file_utils import load_single_file_checkpoint
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    from diffusers.quantizers import DiffusersAutoQuantizer
    from diffusers.quantizers.quantization_config import GGUFQuantizationConfig

    checkpoint = load_single_file_checkpoint(gguf_path)
    tables = {
        name[: -len(TABLE_SUFFIX)]: checkpoint.pop(name)
        for name in [n for n in checkpoint if n.endswith(TABLE_SUFFIX)]
    }
    grid = checkpoint.pop("adaln_timesteps", None)
    if not tables or grid is None:
        raise ValueError("GGUF has no adaln_proj tables / adaln_timesteps; was it made by the converter?")

    with init_empty_weights():
        model = MiniMaxH3ImageTransformer2DModel.from_config(config or {})

    modules = []
    for name, table in tables.items():
        block = model.get_submodule(name)
        block.adaln_proj = TabulatedAdaLN(torch.as_tensor(table))
        modules.append(block.adaln_proj)
    if len(modules) != len(model.transformer_blocks):
        raise ValueError(f"expected {len(model.transformer_blocks)} adaln tables, found {len(modules)}")

    expected = set(model.state_dict())
    missing = sorted(expected - set(checkpoint))
    unexpected = sorted(set(checkpoint) - expected)
    if missing or unexpected:
        raise ValueError(
            f"GGUF does not match the transformer: missing {missing[:5]}, unexpected {unexpected[:5]}"
        )

    quantizer = DiffusersAutoQuantizer.from_config(GGUFQuantizationConfig(compute_dtype=compute_dtype))
    quantizer.validate_environment()
    quantizer.preprocess_model(
        model=model,
        device_map=None,
        state_dict=checkpoint,
        keep_in_fp32_modules=model._keep_in_fp32_modules,
    )
    load_model_dict_into_meta(
        model,
        checkpoint,
        dtype=compute_dtype,
        device_map={"": torch.device("cpu")},
        hf_quantizer=quantizer,
        keep_in_fp32_modules=model._keep_in_fp32_modules,
        unexpected_keys=[],
    )
    quantizer.postprocess_model(model)
    model.hf_quantizer = quantizer
    model.register_forward_pre_hook(
        _select_timestep_rows(modules, torch.as_tensor(grid).float().cpu()), with_kwargs=True
    )
    return model.eval()


class CpuMirror:
    """Offload that frees the GPU copy instead of copying it back to the CPU.

    The weights never change, so the CPU tensors a module had while parked are
    kept as masters.  ``module.to("cuda")`` makes GPU copies as usual;
    ``offload()`` puts the masters back, which drops the GPU copies without a
    device-to-host transfer (those cost 4-5 s per large component into pageable
    memory; moving to the GPU takes 1-2 s).

    ``Module.to`` replaces parameter data and, for bitsandbytes 4-bit weights,
    also moves the ``quant_state`` tensors *in place*, so those are snapshotted
    and restored too.  The module must be on the CPU when this is created.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self._params = []
        self._buffers = []
        self._states = []
        seen = set()
        for owner in module.modules():
            for key, param in owner._parameters.items():
                if param is None:
                    continue
                self._require_cpu(param, key)
                self._params.append((owner, key, param, param.data))
                self._capture_state(getattr(param, "quant_state", None), seen)
            for key, buffer in owner._buffers.items():
                if buffer is None:
                    continue
                self._require_cpu(buffer, key)
                self._buffers.append((owner, key, buffer))

    @staticmethod
    def _require_cpu(tensor, key):
        if tensor.device.type != "cpu":
            raise RuntimeError(f"CpuMirror needs the module on the CPU, but {key} is on {tensor.device}")

    def _capture_state(self, obj, seen):
        # Only bitsandbytes' own state objects; their tensor attributes are what `.to` mutates.
        if obj is None or id(obj) in seen or not type(obj).__module__.startswith("bitsandbytes"):
            return
        seen.add(id(obj))
        for attr, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                self._states.append((obj, attr, value))
            else:
                self._capture_state(value, seen)

    def offload(self) -> None:
        for owner, key, param, data in self._params:
            owner._parameters[key] = param
            param.data = data
        for owner, key, buffer in self._buffers:
            owner._buffers[key] = buffer
        for obj, attr, tensor in self._states:
            setattr(obj, attr, tensor)


def _is_remote(path: str) -> bool:
    return "://" in path or path.lower().startswith(("hf.co/", "huggingface.co/"))


def _validate_base_model(base_model: str) -> str:
    if _is_remote(base_model):
        raise ValueError("minimaxh3image requires a local component folder, not a remote URL")
    if not os.path.isdir(base_model):
        raise FileNotFoundError(base_model)
    required = [os.path.join(base_model, name) for name in REQUIRED_SUBFOLDERS]
    required += [
        os.path.join(base_model, "transformer", "config.json"),
        os.path.join(base_model, "pdd_config.json"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "minimaxh3image component folder is incomplete (run "
            f"scripts/prepare_minimaxh3image_components.py): missing {missing}"
        )
    return os.path.abspath(base_model)


class minimaxh3image(MediaBackend):
    model_id = "minimaxh3image"
    # Distilled six-step grid with its own sampler: no per-request scheduler.
    scheduler_family = None

    def __init__(self, backend_config: dict = {}):
        backend_config = dict(backend_config)
        super().__init__(backend_config)

        gguf_path = os.fspath(backend_config.get("gguf", DEFAULT_GGUF))
        if _is_remote(gguf_path):
            raise ValueError("minimaxh3image requires a local GGUF file, not a remote URL")
        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(gguf_path)
        gguf_path = os.path.abspath(gguf_path)
        base_dir = _validate_base_model(os.fspath(backend_config.get("base_model", DEFAULT_BASE_MODEL)))

        # Keep the last-used component on the GPU between requests (default),
        # or move everything off the GPU after each one.
        self.release_gpu_after_request = bool(backend_config.get("release_gpu_after_request", False))
        self._active_config = None
        self.components_manager = None
        self.pipe = self._build_pipeline(gguf_path, base_dir)

    # -- loading -----------------------------------------------------------

    def _build_pipeline(self, gguf_path: str, base_dir: str):
        import json

        from diffusers import (
            AutoencoderKLMiniMaxH3Image,
            ComponentsManager,
            MiniMaxH3ImageModularPipeline,
            MiniMaxH3Scheduler,
        )
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        with open(os.path.join(base_dir, "transformer", "config.json"), encoding="utf-8") as handle:
            transformer_config = {k: v for k, v in json.load(handle).items() if not k.startswith("_")}
        with open(os.path.join(base_dir, "pdd_config.json"), encoding="utf-8") as handle:
            pdd = json.load(handle)
        sigmas = pdd["sigmas"]
        if len(sigmas) != NUM_STEPS + 1:
            raise ValueError(f"pdd_config.json has {len(sigmas) - 1} steps, expected {NUM_STEPS}")

        transformer = _load_gguf_transformer(gguf_path, config=transformer_config)
        transformer.register_forward_pre_hook(self._before_step, with_kwargs=True)
        transformer.register_forward_hook(self._after_step, with_kwargs=True)

        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(base_dir, "text_encoder"), dtype=torch.bfloat16, device_map={"": 0}
        ).eval()
        # Only text_encoder.model is called; the vocabulary projection would waste ~1.5 GB.
        text_encoder.lm_head = nn.Identity()

        self.components_manager = ComponentsManager()
        self.components_manager.enable_auto_cpu_offload(device="cuda")
        pipe = MiniMaxH3ImageModularPipeline(components_manager=self.components_manager)
        pipe.update_components(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=Qwen2TokenizerFast.from_pretrained(os.path.join(base_dir, "tokenizer")),
            processor=Qwen3VLProcessor.from_pretrained(os.path.join(base_dir, "processor")),
            vae=AutoencoderKLMiniMaxH3Image.from_pretrained(
                os.path.join(base_dir, "vae"), torch_dtype=torch.float32
            ).eval(),
            scheduler=MiniMaxH3Scheduler.from_pretrained(os.path.join(base_dir, "scheduler")),
            sample_sigmas=sigmas,
        )
        self._install_cpu_mirrors()
        return pipe

    def _install_cpu_mirrors(self) -> None:
        """Swap the manager's copy-back offload for CpuMirror (all components are parked on the CPU now)."""
        for user_hook in self.components_manager.model_hooks or []:
            mirror = CpuMirror(user_hook.model)
            user_hook.hook.init_hook = lambda module, mirror=mirror: mirror.offload() or module

    # -- per-step progress and cancellation ---------------------------------

    def _set_phase(self, config, phase) -> None:
        progress_state = config.get("progress_state") if config else None
        if progress_state is not None:
            if phase is None:
                progress_state.pop("phase", None)
            else:
                progress_state["phase"] = phase

    def _before_step(self, _module, args, kwargs):
        config = self._active_config
        if config is not None:
            self.raise_if_cancelled(config)
            if kwargs["pdd_step_index"] == 0:
                self._set_phase(config, "denoising")
        return None

    def _after_step(self, _module, args, kwargs, output):
        config = self._active_config
        progress_state = config.get("progress_state") if config else None
        if progress_state is not None:
            progress_state["step"] = kwargs["pdd_step_index"] + 1
            progress_state["total_steps"] = NUM_STEPS
            if kwargs["pdd_step_index"] + 1 == NUM_STEPS:
                self._set_phase(config, "decoding")
        return None

    def _release_gpu(self) -> None:
        """Park every managed component on the CPU (opt-in: ``release_gpu_after_request``).

        The auto-offload manager only evicts a component when another needs the
        room, so the last one used (up to ~17 GB) stays resident while the
        server idles. Releasing frees it for other GPU users, e.g. a
        llama-server; with CpuMirror it costs no device-to-host copy.
        """
        manager = self.components_manager
        for hook in (getattr(manager, "model_hooks", None) or []):
            hook.offload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- request planning ---------------------------------------------------

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

    def _plan_request(self, config: dict, batch_id, media) -> tuple[str, int, int, int]:
        """Return (output_file, seed, width, height) for a text-to-image request."""
        if media:
            raise ValueError(
                f"{self.model_id} does not support reference images yet; "
                "image editing is not exposed by this backend"
            )

        if batch_id is not None:
            output_file = f"/tmp/ircawp_generated/{self.model_id}.{batch_id}.png"
        else:
            output_file = config.get("output_file", f"/tmp/ircawp_generated/{self.model_id}.png")

        seed = config.get("seed")
        if seed is None:
            seed = torch.randint(0, 1000000, (1,)).item()

        if "width" in config and "height" in config:
            width, height = config["width"], config["height"]
        else:
            max_output_size = config.get("max_output_size", DEFAULT_MAX_OUTPUT_SIZE)
            aspect = self._parse_aspect(config)
            if aspect >= 1.0:
                width, height = max_output_size, max_output_size / aspect
            else:
                width, height = max_output_size * aspect, max_output_size
            # The pipeline generates on a 32-pixel grid; matching it avoids a resample.
            width = max(32, round(width / 32) * 32)
            height = max(32, round(height / 32) * 32)
        return output_file, seed, width, height

    # -- generation ---------------------------------------------------------

    def execute(self, prompt: str, config: dict = {}, batch_id=None, media=[]) -> tuple[str, str]:
        output_file, seed, width, height = self._plan_request(config, batch_id, media)
        final_prompt = prompt.strip()
        if not final_prompt:
            raise ValueError("prompt must not be empty")

        # The schedule is fixed: `steps`, guidance and negative prompts do not apply.
        self._active_config = config
        try:
            self.raise_if_cancelled(config)
            self._set_phase(config, "encoding prompt")
            images = self.pipe(
                prompt=final_prompt,
                height=height,
                width=width,
                generator=torch.Generator("cpu").manual_seed(seed),
                output="images",
            )
            self.raise_if_cancelled(config)
        finally:
            self._set_phase(config, None)
            self._active_config = None
            if self.release_gpu_after_request:
                self._release_gpu()

        output_image = images[0]
        self._save_image_with_metadata(
            output_image,
            output_file,
            final_prompt,
            seed=seed,
            model=self.model_id,
            inference_steps=NUM_STEPS,
            width=output_image.width,
            height=output_image.height,
        )
        self.last_imagegen_prompt = final_prompt
        return output_file, final_prompt

    def dispose(self) -> None:
        self.components_manager = None
        self._active_config = None
        super().dispose()

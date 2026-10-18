"""Loaders for Ming-Image-0.1-Design from community GGUFs (realrebelai/Ming-Image_GGUFs).

The upstream modeling code (github.com/inclusionAI/Ming-Image, flat modules) is
imported from a pinned local clone (`just ming-setup`), only inside
`ming_imports()`, which undoes its compatibility patches afterwards. The GGUF
encoder bundle stores the routed experts as fused 3D banks and renames tensors,
so the LLM is rebuilt here with a fused, on-the-fly-dequantizing expert module.
Text-to-image and single-reference editing; the audio router and `lm_head` are
not loaded.
"""

import contextlib
import math
import os
import sys
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CODE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "vendor",
    "ming-image",
)


@contextlib.contextmanager
def ming_imports(code_dir=None):
    """Import upstream's flat modules under newer Transformers, then undo every patch.

    Patches and `sys.path`/`sys.modules` entries exist only inside the block, so
    other backends in the same process never see them. Objects built inside keep
    working; upstream does no lazy imports on the paths used here.
    """
    code_dir = os.path.abspath(code_dir or DEFAULT_CODE_DIR)
    if not os.path.isdir(code_dir):
        raise FileNotFoundError(
            f"Ming-Image code not found at {code_dir}; run `just ming-setup`"
        )

    import transformers.utils.import_utils as import_utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    modules_before = set(sys.modules)
    # Both entries were removed in Transformers 5 but upstream still looks them up.
    patch_fx = not hasattr(import_utils, "is_torch_fx_available")
    patch_rope = "default" not in ROPE_INIT_FUNCTIONS
    sys.path.insert(0, code_dir)
    if patch_fx:
        import_utils.is_torch_fx_available = lambda: False
    if patch_rope:
        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
    try:
        yield code_dir
    finally:
        sys.path.remove(code_dir)
        if patch_fx:
            del import_utils.is_torch_fx_available
        if patch_rope:
            ROPE_INIT_FUNCTIONS.pop("default", None)
        for name in set(sys.modules) - modules_before:
            module_file = getattr(sys.modules[name], "__file__", None) or ""
            if module_file.startswith(code_dir + os.sep):
                del sys.modules[name]


def _default_rope_init(config, device=None, seq_len=None):
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        theta = config.rope_parameters["rope_theta"]
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    steps = torch.arange(0, dim, 2, dtype=torch.int64).to(
        device=device, dtype=torch.float
    )
    return 1.0 / (theta ** (steps / dim)), 1.0


def _dequant(packed: torch.Tensor, quant_type, dtype: torch.dtype) -> torch.Tensor:
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor

    return dequantize_gguf_tensor(GGUFParameter(packed, quant_type=quant_type)).to(
        dtype
    )


def _dequant_rows(packed, quant_type, dtype, rows_per_chunk=16384):
    parts = [
        _dequant(packed[i : i + rows_per_chunk], quant_type, dtype)
        for i in range(0, packed.shape[0], rows_per_chunk)
    ]
    return torch.cat(parts, dim=0)


def read_gguf_tensor(tensor):
    """Return (torch tensor, quant type). Packed types stay uint8 bytes."""
    import gguf
    import numpy as np

    quant_type = tensor.tensor_type
    data = torch.from_numpy(np.array(tensor.data))
    shape = tuple(int(x) for x in reversed(tensor.shape))
    if quant_type == gguf.GGMLQuantizationType.BF16:
        return data.view(torch.bfloat16).reshape(shape), quant_type
    if quant_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return data.reshape(shape), quant_type
    return data, quant_type


def is_packed(quant_type) -> bool:
    import gguf

    return quant_type not in (
        gguf.GGMLQuantizationType.BF16,
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    )


class PackedLinear(nn.Module):
    """Linear layer whose weight stays GGUF-packed and is dequantized per call."""

    def __init__(self, packed, quant_type):
        super().__init__()
        self.register_buffer("packed", packed, persistent=False)
        self.quant_type = quant_type

    def forward(self, x):
        return F.linear(x, _dequant(self.packed, self.quant_type, x.dtype))


class ExpertBank(nn.Module):
    """Fused routed experts: gate_up (E, 2*I, H) and down (E, H, I), GGUF-packed."""

    def __init__(self):
        super().__init__()
        self.register_buffer("gate_up", None, persistent=False)
        self.register_buffer("down", None, persistent=False)
        self.gate_up_type = None
        self.down_type = None

    def set(self, which: str, packed, quant_type):
        self.register_buffer(which, packed, persistent=False)
        setattr(self, f"{which}_type", quant_type)

    def forward_expert(self, index: int, x: torch.Tensor) -> torch.Tensor:
        gate_up = _dequant(self.gate_up[index], self.gate_up_type, x.dtype)
        gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
        down = _dequant(self.down[index], self.down_type, x.dtype)
        return F.linear(F.silu(gate) * up, down)


def _patch_moe(moe_module):
    """Swap upstream's per-expert ModuleList for a fused bank."""
    block = moe_module.BailingMoeV2SparseMoeBlock

    def setup_experts(self):
        self.experts = ExpertBank()

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        counts = topk_ids.new_zeros((topk_ids.shape[0], self.config.num_experts))
        counts.scatter_(1, topk_ids, 1)
        tokens_per_expert = counts.sum(dim=0).cpu().tolist()
        order = topk_ids.view(-1).argsort()
        sorted_tokens = x[order // topk_ids.shape[1]]
        outputs = []
        start = 0
        for index, n in enumerate(tokens_per_expert):
            if n == 0:
                continue
            outputs.append(
                self.experts.forward_expert(index, sorted_tokens[start : start + n])
            )
            start += n
        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
        restored = torch.empty_like(outs)
        restored[order] = outs
        return (
            restored.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(restored.dtype)
        )

    block._setup_experts = setup_experts
    block.moe_infer = moe_infer


def _llm_target(name: str):
    """Map a GGUF `thinker.*` tensor name to a BailingMoeV2Model parameter path."""
    name = name.removeprefix("thinker.")
    if name == "embed_tokens.weight":
        return "word_embeddings.weight"
    return name.replace(".mlp.gate.proj.weight", ".mlp.gate.weight").replace(
        ".mlp.image_gate.proj.weight", ".mlp.image_gate.weight"
    )


def _set(root: nn.Module, path: str, value, device):
    from accelerate.utils import set_module_tensor_to_device

    # Without an explicit dtype accelerate casts to the meta parameter's fp32.
    set_module_tensor_to_device(root, path, device, value=value, dtype=value.dtype)


def _replace(root: nn.Module, module_path: str, new_module: nn.Module):
    parent = root
    *parents, leaf = module_path.split(".")
    for part in parents:
        parent = getattr(parent, part)
    setattr(parent, leaf, new_module)


class MingEncoder:
    """Prefill-only conditioning stage: prompt -> (DiT context, direct-VLM context)."""

    def __init__(
        self,
        llm,
        connector,
        proj_in,
        proj_out,
        proj_directvlm,
        host,
        tokenizer,
        eos_token,
        template,
        mlp_config,
        vision=None,
        linear_proj=None,
        image_tools=None,
    ):
        self.llm = llm
        self.connector = connector
        self.proj_in = proj_in
        self.proj_out = proj_out
        self.proj_directvlm = proj_directvlm
        self.host = host
        self.tokenizer = tokenizer
        self.eos_token = eos_token
        self.template = template
        self.mlp_config = mlp_config
        self.vision = vision
        self.linear_proj = linear_proj
        self.image_tools = image_tools

    def to(self, device):
        modules = (
            self.llm,
            self.connector,
            self.proj_in,
            self.proj_out,
            self.proj_directvlm,
        )
        for module in modules + tuple(
            m for m in (self.vision, self.linear_proj) if m is not None
        ):
            module.to(device)
        # Not a submodule of llm, so llm.to() does not reach it.
        self.host.query_tokens_dict.to(device)
        return self

    def _tokenize(self, prompt: str, device, image_tokens: int = 0):
        # Same template as upstream's processor.apply_chat_template for a single turn.
        system, user, assistant = self.template
        if image_tokens:
            start, patch, end = self.image_tools["tokens"]
            user = user + start + patch * image_tokens + end + "\n"
        text = system + self.eos_token + user + prompt + self.eos_token + assistant
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        return input_ids, torch.ones_like(input_ids)

    def _image_features(self, image, device):
        """Vision-tower embeddings of the reference image, as upstream's extract_image_feature."""
        if self.vision is None:
            raise RuntimeError("this encoder was loaded without the vision tower")
        tools = self.image_tools
        batch = tools["image_processor"](
            images=tools["fetch_image"]({"type": "image", "image": image}),
            return_tensors="pt",
        )
        grid_thw = batch["image_grid_thw"].to(device)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            embeds = self.vision(batch["pixel_values"].to(device), grid_thw=grid_thw)
        embeds = self.linear_proj(embeds.to(self.linear_proj[0].weight.dtype))
        return F.normalize(embeds, dim=-1), grid_thw

    @torch.no_grad()
    def encode(self, prompt: str, image=None):
        host, llm = self.host, self.llm
        device = next(llm.parameters()).device
        image_embeds = image_grid_thw = None
        image_tokens = 0
        if image is not None:
            image_embeds, image_grid_thw = self._image_features(image, device)
            image_tokens = int(image_grid_thw.prod(dim=1).sum().item()) // 4
        input_ids, attention_mask = self._tokenize(prompt, device, image_tokens)
        patch_id = llm.config.image_patch_token

        input_ids, labels, attention_mask, gen_mask = (
            host.append_input_ids_with_multiscale_learnable_tokens(
                input_ids,
                attention_mask,
                host.img_gen_scales,
                patch_id + 1,
                patch_id + 2,
                patch_id,
            )
        )
        grid_thw, query_embeds = host.appand_learnable_tokens(
            input_ids, gen_mask, image_embeds, image_grid_thw, patch_id
        )
        embeddings, image_mask, _ = llm.prompt_wrap_navit(
            input_ids=input_ids.clip(0, llm.word_embeddings.weight.shape[0] - 1),
            config=llm.config,
            query_embeds_image=query_embeds.to(llm.word_embeddings.weight.dtype),
        )
        outputs = llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=embeddings,
            image_grid_thw=grid_thw,
            use_cache=False,
            image_mask=image_mask,
            audio_mask=None,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]

        # Direct-VLM context: text-token states from selected layers.
        text_mask = torch.lt(labels, 0).int().to(attention_mask.dtype) * attention_mask
        limit = text_mask.cumsum(-1).argmax(-1).max().item() + 1
        text_mask = text_mask[:, :limit]
        directvlm = torch.cat(
            [
                outputs.hidden_states[i][:, :limit, :] * text_mask.unsqueeze(-1)
                for i in self.mlp_config["selected_hidden_states_layers"]
            ],
            dim=-1,
        )
        directvlm = self.proj_directvlm(directvlm)

        # Context: states of the learned query tokens, through the connector.
        gen_select = gen_mask.unsqueeze(-1).expand_as(hidden).bool()
        gen_hidden = torch.masked_select(hidden, gen_select).view(
            hidden.shape[0], -1, hidden.shape[-1]
        )
        scale = host.img_gen_scales[-1]
        gen_hidden = gen_hidden[:, -scale * scale :, :]
        seq = self.proj_in(gen_hidden)
        mask = torch.zeros(
            seq.shape[0],
            1,
            seq.shape[1],
            seq.shape[1],
            dtype=seq.dtype,
            device=seq.device,
        )
        context = self.connector(
            inputs_embeds=seq, attention_mask=mask
        ).last_hidden_state
        context = self.proj_out(context)
        if self.mlp_config.get("connector_norm", False):
            context = F.normalize(context, dim=-1)
        return context, directvlm


def load_encoder(
    gguf_path: str,
    official_dir: str,
    code_dir=None,
    device="cuda",
    dtype=torch.bfloat16,
):
    """Build the conditioning stage on `device` from the community encoder GGUF."""
    with ming_imports(code_dir):
        return _build_encoder(gguf_path, official_dir, device, dtype)


def _build_encoder(gguf_path: str, official_dir: str, device, dtype):
    import json

    import gguf
    from accelerate import init_empty_weights
    from diffusers.models.normalization import RMSNorm
    from transformers import Qwen2Config, Qwen2Model

    import modeling_bailing_moe_v2 as moe
    import processing_bailingmm2 as template
    from configuration_bailingmm2 import BailingMM2Config
    from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

    _patch_moe(moe)
    device = torch.device(device)

    mllm_dir = os.path.join(official_dir, "mllm")
    config = BailingMM2Config.from_dict(BailingMM2Config.get_config_dict(mllm_dir)[0])
    llm_config = config.llm_config
    llm_config._attn_implementation = "eager"
    with open(os.path.join(official_dir, "mlp", "config.json")) as handle:
        mlp_config = json.load(handle)
    scales = mlp_config["img_gen_scales"]
    hidden = llm_config.hidden_size

    with init_empty_weights():
        llm = moe.BailingMoeV2Model(llm_config)
        connector_config = Qwen2Config.from_pretrained(
            os.path.join(official_dir, "connector")
        )
        connector_config._attn_implementation = "sdpa"
        connector = Qwen2Model(connector_config)
    llm.rotary_emb = moe.BailingMoeV2RotaryEmbedding3D(config=llm_config, device=device)
    for layer in connector.layers:
        layer.self_attn.is_causal = False

    proj_in = nn.Linear(hidden, connector_config.hidden_size)
    proj_out = nn.Linear(
        connector_config.hidden_size, mlp_config["diffusion_c_input_dim"]
    )
    directvlm_dim = hidden * len(mlp_config["selected_hidden_states_layers"])
    proj_directvlm = nn.Sequential(
        RMSNorm(directvlm_dim, eps=1e-5),
        nn.Linear(directvlm_dim, mlp_config["diffusion_inner_dim"]),
    )
    small = {
        "proj_in": proj_in,
        "proj_out": proj_out,
        "proj_directvlm.0": proj_directvlm[0],
        "proj_directvlm.1": proj_directvlm[1],
    }
    query_tokens = nn.ParameterDict()

    from image_processing_bailingmm2 import BailingMM2ImageProcessor
    from bailingmm_utils import fetch_image
    from qwen2_5_vit import Qwen2_5_VisionTransformer

    vision_config = config.vision_config
    vision_config._attn_implementation = "sdpa"
    with init_empty_weights():
        vision = Qwen2_5_VisionTransformer(vision_config)
    linear_proj = nn.Sequential(
        nn.Linear(vision.image_emb_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
    )
    small.update({"linear_proj.0": linear_proj[0], "linear_proj.2": linear_proj[2]})

    reader = gguf.GGUFReader(gguf_path)
    loaded_llm = set()
    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith(("thinker.lm_head", "tokenizer_json")):
            continue
        value, quant_type = read_gguf_tensor(tensor)

        if name.startswith("vision."):
            _set(vision, name.removeprefix("vision."), value.to(dtype), device)
        elif name.startswith("thinker."):
            path = _llm_target(name)
            if ".mlp.experts." in path:
                layer_path, kind = path.split(".mlp.experts.")
                bank = llm.get_submodule(f"{layer_path}.mlp.experts")
                bank.set(
                    "gate_up" if kind.startswith("gate_up") else "down",
                    value.to(device),
                    quant_type,
                )
            elif is_packed(quant_type) and name.endswith("embed_tokens.weight"):
                _set(
                    llm,
                    path,
                    _dequant_rows(value.to(device), quant_type, dtype),
                    device,
                )
            elif is_packed(quant_type):
                _replace(
                    llm,
                    path.removesuffix(".weight"),
                    PackedLinear(value.to(device), quant_type),
                )
            else:
                keep = (
                    torch.float32
                    if value.dtype == torch.float32 and "expert_bias" in path
                    else dtype
                )
                _set(llm, path, value.to(keep), device)
            loaded_llm.add(path)
        elif name.startswith("connector."):
            path = name.removeprefix("connector.")
            if is_packed(quant_type):
                _replace(
                    connector,
                    path.removesuffix(".weight"),
                    PackedLinear(value.to(device), quant_type),
                )
            else:
                _set(connector, path, value.to(dtype), device)
        elif name == "query_tokens":
            query_tokens[f"{scales[-1]}x{scales[-1]}"] = nn.Parameter(
                value.to(device, dtype), requires_grad=False
            )
        else:
            module_name, _, param = name.rpartition(".")
            if is_packed(quant_type):
                # Small projections (linear_proj) are cheap to keep dequantized.
                value = _dequant_rows(value.to(device), quant_type, dtype)
            _set(small[module_name], param, value.to(dtype), device)

    # The audio router is dropped from the GGUF and never selected for text-to-image.
    for index, layer in enumerate(llm.layers):
        gate = getattr(layer.mlp, "audio_gate", None)
        if gate is not None:
            for param_name, param in list(gate.named_parameters()):
                shape = tuple(param.shape)
                _set(gate, param_name, torch.zeros(shape, dtype=dtype), device)

    # Tied embeddings are unused: the connector receives inputs_embeds.
    connector.embed_tokens = nn.Embedding(1, connector_config.hidden_size).to(
        device=device, dtype=dtype
    )
    connector.to(device)
    for module in (proj_in, proj_out, proj_directvlm, linear_proj, vision):
        module.to(device=device, dtype=dtype)

    host = BailingMM2NativeForConditionalGeneration(config, empty_load=True)
    host.img_gen_scales = scales
    host.query_tokens_dict = query_tokens
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(os.path.join(mllm_dir, "tokenizer.json"))
    with open(os.path.join(mllm_dir, "special_tokens_map.json")) as handle:
        eos_token = json.load(handle)["eos_token"]["content"]
    return MingEncoder(
        llm.eval(),
        connector.eval(),
        proj_in,
        proj_out,
        proj_directvlm,
        host,
        tokenizer,
        eos_token,
        (
            template.SYSTEM_PROMPT_LINGV2_FLASH_NOTHINK,
            template.USER_PREFIX,
            template.ASSISTANT_PREFIX,
        ),
        mlp_config,
        vision=vision.eval(),
        linear_proj=linear_proj.eval(),
        image_tools={
            "fetch_image": fetch_image,
            "image_processor": BailingMM2ImageProcessor.from_pretrained(mllm_dir),
            "tokens": (
                template.DEFAULT_IM_START_TOKEN,
                template.DEFAULT_IMAGE_PATCH_TOKEN,
                template.DEFAULT_IM_END_TOKEN,
            ),
        },
    )


@torch.no_grad()
def sample_image(
    generator,
    context,
    directvlm,
    *,
    steps,
    cfg,
    seed,
    height,
    width,
    callback_on_step_end=None,
    ref_x=None,
):
    """Upstream `ImageGenerator.sample` with a per-request RNG and a step callback."""
    sampling = generator.inference_profile.resolve_sampling_parameters(
        steps=steps, cfg=cfg
    )
    cast = dict(device=generator.device, dtype=generator.train_model.dtype)
    embeds = list(context.to(**cast).unbind(dim=0))
    embeds_2 = list(directvlm.to(**cast).unbind(dim=0))
    return generator.pipelines(
        prompt_embeds=embeds,
        negative_prompt_embeds=[t * 0 for t in embeds],
        prompt_embeds_2=embeds_2,
        negative_prompt_embeds_2=[t * 0 for t in embeds_2],
        guidance_scale=sampling.cfg,
        generator=torch.Generator("cpu").manual_seed(seed),
        num_inference_steps=sampling.steps,
        height=height,
        width=width,
        max_sequence_length=512,
        device=generator.device,
        ref_hidden_states=ref_x,
        sample_mode=generator.vae_sample_mode,
        num_frames_per_prompt=1,
        callback_on_step_end=callback_on_step_end,
    ).images[0]


def _check_size(width, height):
    for value in (width, height):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"width and height must be positive integers, got {width}x{height}"
            )
    if not 0.25 <= width / height <= 4.0:
        raise ValueError(
            f"aspect ratio must be within 1:4 and 4:1, got {width}x{height}"
        )


def plan_size(width, height, code_dir=None):
    """Return the (width, height) upstream generates internally for a requested size."""
    _check_size(width, height)
    side = round(math.sqrt(width * height))
    bucket = min((1024, 2048), key=lambda b: abs(b - side))
    with ming_imports(code_dir):
        from bailingmm_utils import process_ratio

        closest, _ = process_ratio(ori_h=height, ori_w=width, highres=bucket)
    return closest[1], closest[0]


class ReferencePlan(NamedTuple):
    ref_x: torch.Tensor
    width: int
    height: int
    image: object


def prepare_reference(image, width=None, height=None, code_dir=None) -> ReferencePlan:
    """Crop (when a size is requested) and encode a reference image as RGBA latents."""
    if (width is None) != (height is None):
        raise ValueError("width and height must be given together")
    if width is not None:
        _check_size(width, height)
    _check_size(image.width, image.height)
    with ming_imports(code_dir):
        from processing_bailingmm2 import crop_to_aspect_max, transform_reference_images

        image = image.convert("RGBA")
        if width is not None:
            image = crop_to_aspect_max(image, width / height)
        # 1024 is the only edit bucket upstream uses.
        ref_x, _, _ = transform_reference_images([image], None, 1024, 4)
    return ReferencePlan(ref_x, ref_x.shape[-1], ref_x.shape[-2], image)


def _load_dit(gguf_path: str, official_dir: str, device, dtype):
    """Build the DiT on the meta device and fill it from the community GGUF."""
    import gguf
    from accelerate import init_empty_weights
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    from diffusers.quantizers import DiffusersAutoQuantizer
    from diffusers.quantizers.gguf.utils import GGUFParameter
    from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
    from diffusion.transformer import DiffusionTransformer

    config = DiffusionTransformer.load_config(official_dir, subfolder="transformer")
    with init_empty_weights():
        model = DiffusionTransformer.from_config(config)

    checkpoint = {}
    for tensor in gguf.GGUFReader(gguf_path).tensors:
        value, quant_type = read_gguf_tensor(tensor)
        checkpoint[tensor.name] = (
            GGUFParameter(value, quant_type=quant_type)
            if is_packed(quant_type)
            else value
        )
    expected = set(model.state_dict())
    if expected != set(checkpoint):
        raise ValueError(
            f"GGUF does not match the DiT: missing {sorted(expected - set(checkpoint))[:5]}, "
            f"unexpected {sorted(set(checkpoint) - expected)[:5]}"
        )

    quantizer = DiffusersAutoQuantizer.from_config(
        GGUFQuantizationConfig(compute_dtype=dtype)
    )
    quantizer.validate_environment()
    quantizer.preprocess_model(
        model=model, device_map=None, state_dict=checkpoint, keep_in_fp32_modules=[]
    )
    load_model_dict_into_meta(
        model,
        checkpoint,
        dtype=dtype,
        device_map={"": torch.device(device)},
        hf_quantizer=quantizer,
        keep_in_fp32_modules=[],
        unexpected_keys=[],
    )
    quantizer.postprocess_model(model)
    model.hf_quantizer = quantizer
    return model.eval()


def load_generator(
    dit_gguf: str,
    vae_path: str,
    official_dir: str,
    code_dir=None,
    device="cuda",
    dtype=torch.bfloat16,
):
    """Assemble upstream's ImageGenerator around the GGUF DiT and the VAE file."""
    with ming_imports(code_dir):
        return _build_generator(dit_gguf, vae_path, official_dir, device, dtype)


def _build_generator(dit_gguf: str, vae_path: str, official_dir: str, device, dtype):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file

    from diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    from diffusion.generator import ConditionedTransformer, ImageGenerator
    from diffusion.pipeline import ImageGenerationPipeline
    from inference_profile import load_checkpoint_capabilities

    profile = load_checkpoint_capabilities(official_dir)
    vae_config = AutoencoderKLQwenImage.load_config(official_dir, subfolder="vae")
    vae = AutoencoderKLQwenImage.from_config(vae_config)
    vae.load_state_dict(load_file(vae_path), strict=True)
    vae = vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae.input_channels = 4

    transformer = _load_dit(dit_gguf, official_dir, device, dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        official_dir, subfolder="scheduler"
    )
    scheduler.config["use_dynamic_shifting"] = True

    # Skip ImageGenerator.__init__, which insists on loading the safetensors DiT.
    generator = ImageGenerator.__new__(ImageGenerator)
    torch.nn.Module.__init__(generator)
    generator.inference_profile = profile
    generator.vae = vae
    generator.vae_sample_mode = "argmax"
    generator.train_model = ConditionedTransformer(
        transformer,
        vision_dim=2560,
        use_identity_mlp=True,
        text_encoder_norm=False,
    )
    generator.noise_scheduler = scheduler
    generator.pipelines = ImageGenerationPipeline(
        vae=vae,
        transformer=generator.train_model,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
    )
    return generator

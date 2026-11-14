"""GPU-free contract for the MiniMax-H3-Image backend, its GGUF loader and converter."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gguf
import numpy as np
from safetensors.torch import save_file
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER = PROJECT_ROOT / "scripts" / "convert_minimaxh3image_to_gguf.py"

# A tiny geometry with the same structure as the real checkpoint.
HEADS, HEAD_DIM, HIDDEN, FFN = 2, 4, 6, 10
TIME_EMBED, TIME_HIDDEN, FREQ, TEXT_DIM = 8, 12, 16, 5
BLOCKS, REFINER = 2, 1
SIGMAS = [1.0, 0.75, 0.5, 0.0]  # three steps
PATCH_DIM = 24 * 4


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_minimaxh3image", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_constants(module):
    return patch.multiple(
        module,
        HIDDEN=HIDDEN,
        HEADS=HEADS,
        HEAD_DIM=HEAD_DIM,
        FFN=FFN,
        NUM_LAYERS=BLOCKS,
        FREQ_DIM=FREQ,
        TIME_EMBED_DIM=TIME_EMBED,
    )


def bf16(*shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(torch.bfloat16)


def f32(*shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def official_checkpoint():
    """Tensors named and shaped like the official transformer shards."""
    inner = HEADS * HEAD_DIM
    seed = iter(range(1000))
    t = {
        "video_patch_proj.weight": f32(HIDDEN, PATCH_DIM, seed=next(seed)),
        "video_patch_proj.bias": f32(HIDDEN, seed=next(seed)),
        "condition_proj.weight": bf16(HIDDEN, TEXT_DIM, seed=next(seed)),
        "condition_proj.bias": bf16(HIDDEN, seed=next(seed)),
        "time_embedder.proj_in.weight": f32(TIME_HIDDEN, FREQ, seed=next(seed)),
        "time_embedder.proj_in.bias": f32(TIME_HIDDEN, seed=next(seed)),
        "time_embedder.proj_out.weight": f32(TIME_EMBED, TIME_HIDDEN, seed=next(seed)),
        "time_embedder.proj_out.bias": f32(TIME_EMBED, seed=next(seed)),
        "token_refiner.final_norm.weight": bf16(HIDDEN, seed=next(seed)),
        "final_layer.norm.weight": bf16(HIDDEN, seed=next(seed)),
        "final_layer.adaln_proj.linear.weight": bf16(2 * HIDDEN, TIME_EMBED, seed=next(seed)),
        "final_layer.adaln_proj.linear.bias": bf16(2 * HIDDEN, seed=next(seed)),
        "final_layer.video_out.weight": f32(PATCH_DIM, HIDDEN, seed=next(seed)),
        "final_layer.video_out.bias": f32(PATCH_DIM, seed=next(seed)),
        "rope.inv_freq": f32(16, seed=next(seed)),
    }
    for prefix, count, adaln in (("blocks", BLOCKS, True), ("token_refiner.blocks", REFINER, False)):
        for i in range(count):
            p = f"{prefix}.{i}"
            t[f"{p}.norm1.weight"] = bf16(HIDDEN, seed=next(seed))
            t[f"{p}.norm2.weight"] = bf16(HIDDEN, seed=next(seed))
            t[f"{p}.attn.qkv_proj.weight"] = bf16(3 * inner, HIDDEN, seed=next(seed))
            t[f"{p}.attn.q_norm.weight"] = bf16(HEAD_DIM, seed=next(seed))
            t[f"{p}.attn.k_norm.weight"] = bf16(HEAD_DIM, seed=next(seed))
            t[f"{p}.attn.out_proj.weight"] = bf16(HIDDEN, inner, seed=next(seed))
            t[f"{p}.mlp.fc1.weight"] = bf16(2 * FFN, HIDDEN, seed=next(seed))
            t[f"{p}.mlp.fc2.weight"] = bf16(HIDDEN, FFN, seed=next(seed))
            if adaln:
                t[f"{p}.adaln_proj.linear.weight"] = bf16(18 * HIDDEN, TIME_EMBED, seed=next(seed))
                t[f"{p}.adaln_proj.linear.bias"] = bf16(18 * HIDDEN, seed=next(seed))
    return t


def write_repo(root):
    """Write the tensors as two shards plus index, PDD heads and sampling config."""
    tensors = official_checkpoint()
    names = sorted(tensors)
    halves = [names[: len(names) // 2], names[len(names) // 2 :]]
    (root / "transformer").mkdir(parents=True)
    weight_map = {}
    for number, half in enumerate(halves, 1):
        shard = f"model-0000{number}-of-00002.safetensors"
        save_file({n: tensors[n] for n in half}, str(root / "transformer" / shard))
        weight_map.update({n: shard for n in half})
    (root / "transformer" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    heads = {
        "video_out.weight": f32(len(SIGMAS) - 1, PATCH_DIM, HIDDEN, seed=900),
        "video_out.bias": f32(len(SIGMAS) - 1, PATCH_DIM, seed=901),
    }
    save_file(heads, str(root / "pdd_fused_heads.safetensors"))
    (root / "pdd_config.json").write_text(
        json.dumps({"num_inference_steps": len(SIGMAS) - 1, "sigmas": SIGMAS})
    )
    return tensors, heads


def read_gguf(path):
    reader = gguf.GGUFReader(str(path))
    return {t.name: t for t in reader.tensors}, reader


def as_torch(tensor):
    shape = tuple(int(x) for x in reversed(tensor.shape))
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        return torch.from_numpy(tensor.data.copy().view(np.int16)).view(torch.bfloat16).reshape(shape)
    return torch.from_numpy(tensor.data.copy()).reshape(shape)


class ConverterHelperTests(unittest.TestCase):
    def setUp(self):
        self.converter = load_converter()

    def test_interleaved_qkv_becomes_stacked_q_k_v(self):
        rows = HEADS * 3 * HEAD_DIM
        weight = torch.arange(rows, dtype=torch.float32)[:, None].repeat(1, 3)
        stacked = self.converter.reorder_interleaved_qkv(weight, HEADS, HEAD_DIM)
        # Source rows are [h0 q, h0 k, h0 v, h1 q, h1 k, h1 v] in blocks of HEAD_DIM.
        block = lambda index: torch.arange(index * HEAD_DIM, (index + 1) * HEAD_DIM)
        expected = torch.cat(
            [block(0), block(3), block(1), block(4), block(2), block(5)]
        ).float()
        self.assertTrue(torch.equal(stacked[:, 0], expected))

    def test_swiglu_halves_swap_from_gate_value_to_value_gate(self):
        fc1 = torch.arange(8, dtype=torch.float32)[:, None]
        swapped = self.converter.swap_gate_value(fc1)
        self.assertEqual(swapped[:, 0].tolist(), [4, 5, 6, 7, 0, 1, 2, 3])

    def test_official_names_map_to_diffusers_names(self):
        rename = self.converter.rename
        self.assertEqual(rename("blocks.3.attn.out_proj.weight"), "transformer_blocks.3.attn.to_out.0.weight")
        self.assertEqual(rename("blocks.3.attn.q_norm.weight"), "transformer_blocks.3.attn.norm_q.weight")
        self.assertEqual(rename("blocks.3.mlp.fc2.weight"), "transformer_blocks.3.ff.net.2.weight")
        self.assertEqual(
            rename("token_refiner.blocks.1.attn.k_norm.weight"),
            "token_refiner.refiner_blocks.1.attn.norm_k.weight",
        )
        self.assertEqual(rename("video_patch_proj.weight"), "proj_in.weight")
        self.assertEqual(rename("condition_proj.bias"), "context_embedder.bias")
        self.assertEqual(rename("final_layer.adaln_proj.linear.weight"), "norm_out.linear.weight")
        self.assertEqual(rename("time_embedder.proj_out.bias"), "time_embedder.linear_2.bias")

    def test_replaced_and_recomputed_tensors_are_dropped(self):
        self.assertIsNone(self.converter.rename("rope.inv_freq"))
        self.assertIsNone(self.converter.rename("final_layer.video_out.weight"))
        self.assertIsNone(self.converter.rename("final_layer.video_out.bias"))

    def test_adaln_timesteps_are_one_minus_sigma_then_reference_level(self):
        steps = self.converter.adaln_timesteps(SIGMAS)
        self.assertEqual(steps.dtype, torch.float32)
        expected = [0.0, 0.25, 0.5, self.converter.REFERENCE_TIMESTEP]
        self.assertEqual([round(float(x), 6) for x in steps], expected)

    def test_target_dtype_keeps_matrices_bf16_and_everything_small_float32(self):
        target = self.converter.target_dtype
        self.assertEqual(target("x.weight", bf16(4, 4, seed=1))[1], gguf.GGMLQuantizationType.BF16)
        self.assertEqual(target("x.bias", bf16(4, seed=1))[1], gguf.GGMLQuantizationType.F32)
        self.assertEqual(target("proj_in.weight", f32(4, 4, seed=1))[1], gguf.GGMLQuantizationType.F32)
        self.assertEqual(target("context_embedder.weight", bf16(4, 4, seed=1))[1], gguf.GGMLQuantizationType.F32)
        self.assertEqual(target("norm_out.linear.weight", bf16(4, 4, seed=1))[1], gguf.GGMLQuantizationType.F32)

    def test_quantize_command_protects_sensitive_tensors(self):
        rules = self.converter.QUANTIZE_TENSOR_TYPES
        for name in ("proj_in", "context_embedder", "time_embedder", "norm_out", "pdd_proj_out"):
            self.assertIn(f"{name}=f32", rules)
        self.assertIn("token_refiner=q8_0", rules)


class ConverterTests(unittest.TestCase):
    def convert(self, directory):
        converter = load_converter()
        root = Path(directory) / "repo"
        tensors, heads = write_repo(root)
        out = Path(directory) / "out.gguf"
        with tiny_constants(converter):
            converter.convert(str(root), out, revision="abc123")
        return converter, tensors, heads, out

    def test_every_expected_tensor_is_written_with_diffusers_names(self):
        with TemporaryDirectory() as directory:
            _, _, _, out = self.convert(directory)
            written, reader = read_gguf(out)
            names = set(written)
            for n in range(BLOCKS):
                p = f"transformer_blocks.{n}"
                for suffix in (
                    "adaln_proj.table", "attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
                    "attn.to_out.0.weight", "attn.norm_q.weight", "attn.norm_k.weight",
                    "ff.net.0.proj.weight", "ff.net.2.weight", "norm1.weight", "norm2.weight",
                ):
                    self.assertIn(f"{p}.{suffix}", names)
                self.assertNotIn(f"{p}.adaln_proj.linear.weight", names)
                self.assertNotIn(f"{p}.attn.qkv_proj.weight", names)
            self.assertIn("token_refiner.refiner_blocks.0.attn.to_q.weight", names)
            self.assertNotIn("token_refiner.refiner_blocks.0.adaln_proj.table", names)
            for step in range(len(SIGMAS) - 1):
                self.assertIn(f"pdd_proj_out.{step}.weight", names)
                self.assertIn(f"pdd_proj_out.{step}.bias", names)
            self.assertIn("adaln_timesteps", names)
            self.assertNotIn("rope.inv_freq", names)
            self.assertNotIn("proj_out.weight", names)
            self.assertEqual(bytes(reader.fields["general.architecture"].parts[-1]), b"qwen2")
            self.assertEqual(bytes(reader.fields["minimax_h3_image.source_revision"].parts[-1]), b"abc123")

    def test_qkv_is_deinterleaved_and_split_and_fc1_halves_swapped(self):
        with TemporaryDirectory() as directory:
            converter, tensors, _, out = self.convert(directory)
            written, _ = read_gguf(out)
            source = tensors["blocks.1.attn.qkv_proj.weight"]
            stacked = converter.reorder_interleaved_qkv(source, HEADS, HEAD_DIM)
            q, k, v = stacked.split(HEADS * HEAD_DIM, dim=0)
            for name, expected in (("to_q", q), ("to_k", k), ("to_v", v)):
                got = as_torch(written[f"transformer_blocks.1.attn.{name}.weight"])
                self.assertTrue(torch.equal(got, expected), name)
            gate, value = tensors["blocks.0.mlp.fc1.weight"].chunk(2, dim=0)
            got = as_torch(written["transformer_blocks.0.ff.net.0.proj.weight"])
            self.assertTrue(torch.equal(got, torch.cat([value, gate], dim=0)))

    def test_dtypes_bf16_matrices_float32_small_tensors(self):
        with TemporaryDirectory() as directory:
            _, _, _, out = self.convert(directory)
            written, _ = read_gguf(out)
            kinds = {n: t.tensor_type for n, t in written.items()}
            bf, f = gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32
            self.assertEqual(kinds["transformer_blocks.0.attn.to_q.weight"], bf)
            self.assertEqual(kinds["transformer_blocks.0.ff.net.2.weight"], bf)
            self.assertEqual(kinds["transformer_blocks.0.norm1.weight"], f)
            self.assertEqual(kinds["transformer_blocks.0.adaln_proj.table"], f)
            self.assertEqual(kinds["proj_in.weight"], f)
            self.assertEqual(kinds["context_embedder.weight"], f)
            self.assertEqual(kinds["pdd_proj_out.0.weight"], f)
            self.assertEqual(kinds["time_embedder.linear_1.weight"], f)

    def test_pdd_heads_are_split_per_step_with_bit_identical_values(self):
        with TemporaryDirectory() as directory:
            _, _, heads, out = self.convert(directory)
            written, _ = read_gguf(out)
            for step in range(len(SIGMAS) - 1):
                self.assertTrue(
                    torch.equal(as_torch(written[f"pdd_proj_out.{step}.weight"]), heads["video_out.weight"][step])
                )
                self.assertTrue(
                    torch.equal(as_torch(written[f"pdd_proj_out.{step}.bias"]), heads["video_out.bias"][step])
                )

    def test_adaln_table_equals_the_real_modulation_module(self):
        from diffusers.models.embeddings import TimestepEmbedding, Timesteps
        from diffusers.models.transformers.transformer_minimax_h3_image import (
            MiniMaxH3ImageAdaLayerNormModulation,
        )

        with TemporaryDirectory() as directory:
            converter, tensors, _, out = self.convert(directory)
            written, _ = read_gguf(out)
            timesteps = as_torch(written["adaln_timesteps"])
            embedder = TimestepEmbedding(in_channels=FREQ, time_embed_dim=TIME_HIDDEN, out_dim=TIME_EMBED)
            embedder.load_state_dict(
                {
                    "linear_1.weight": tensors["time_embedder.proj_in.weight"],
                    "linear_1.bias": tensors["time_embedder.proj_in.bias"],
                    "linear_2.weight": tensors["time_embedder.proj_out.weight"],
                    "linear_2.bias": tensors["time_embedder.proj_out.bias"],
                }
            )
            projection = Timesteps(num_channels=FREQ, flip_sin_to_cos=True, downscale_freq_shift=0)
            with torch.no_grad():
                temb = embedder(projection(timesteps))
                for n in range(BLOCKS):
                    module = MiniMaxH3ImageAdaLayerNormModulation(TIME_EMBED, HIDDEN).to(torch.bfloat16)
                    module.linear.weight.data = tensors[f"blocks.{n}.adaln_proj.linear.weight"]
                    module.linear.bias.data = tensors[f"blocks.{n}.adaln_proj.linear.bias"]
                    parts = module(temb)  # six (T * 3, HIDDEN), rows t * 3 + modality
                    reference = torch.stack([p.view(len(timesteps), 3, HIDDEN) for p in parts], dim=2)
                    table = as_torch(written[f"transformer_blocks.{n}.adaln_proj.table"])
                    table = table.view(len(timesteps), 2, 6, HIDDEN)
                    self.assertTrue(torch.equal(table, reference[:, :2].float()))

    def test_converted_names_and_shapes_load_into_the_real_transformer_class(self):
        from accelerate import init_empty_weights
        from diffusers import MiniMaxH3ImageTransformer2DModel

        with TemporaryDirectory() as directory:
            _, _, _, out = self.convert(directory)
            written, _ = read_gguf(out)
        config = dict(
            num_attention_heads=HEADS, attention_head_dim=HEAD_DIM, hidden_size=HIDDEN, num_layers=BLOCKS,
            num_refiner_layers=REFINER, ffn_dim=FFN, text_dim=TEXT_DIM, freq_dim=FREQ,
            time_embed_hidden_dim=TIME_HIDDEN, time_embed_dim=TIME_EMBED, rope_freq_dim=1,
            num_pdd_heads=len(SIGMAS) - 1,
        )
        with init_empty_weights():
            model = MiniMaxH3ImageTransformer2DModel(**config)
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items() if ".adaln_proj.linear." not in k}
        got = {
            n: tuple(int(x) for x in reversed(t.shape))
            for n, t in written.items()
            if not n.endswith(".adaln_proj.table") and n != "adaln_timesteps"
        }
        self.assertEqual(got, expected)

    def test_bad_pdd_config_is_rejected(self):
        converter = load_converter()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            write_repo(root)
            (root / "pdd_config.json").write_text(json.dumps({"num_inference_steps": 3, "sigmas": [0.9, 0.5, 0.0]}))
            with tiny_constants(converter), self.assertRaises(ValueError):
                converter.convert(str(root), Path(directory) / "out.gguf")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Backend and loader
# ---------------------------------------------------------------------------

from threading import Event  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import Mock  # noqa: E402

from PIL import Image  # noqa: E402

from app.backends.MediaBackend import GenerationCancelled, MediaBackend  # noqa: E402
from app.backends import minimaxh3image as backend_module  # noqa: E402

STEPS = 6


def make_component_folder(root, drop=None):
    base = Path(root) / "components"
    for name in backend_module.REQUIRED_SUBFOLDERS:
        (base / name).mkdir(parents=True)
    (base / "transformer").mkdir(exist_ok=True)
    (base / "transformer" / "config.json").write_text("{}")
    (base / "pdd_config.json").write_text(json.dumps({"num_inference_steps": 6, "sigmas": [1, 0.9, 0.8, 0.6, 0.4, 0.2, 0]}))
    if drop:
        target = base / drop
        if target.is_dir():
            for child in target.iterdir():
                child.unlink()
            target.rmdir()
        else:
            target.unlink()
    return base


class BackendTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.gguf = self.directory / "transformer.gguf"
        self.gguf.write_bytes(b"mocked transformer")
        self.base = make_component_folder(self.directory)
        self.output = self.directory / "result.png"
        self.pipeline = Mock(side_effect=self.render)
        self.builder = self.enterContext(
            patch.object(backend_module.minimaxh3image, "_build_pipeline", return_value=self.pipeline)
        )
        self.backend_instance = None

    def render(self, **kwargs):
        # Emulates the six transformer calls the real pipeline makes.
        for step in range(STEPS):
            self.backend_instance._before_step(None, (), {"pdd_step_index": step})
            self.backend_instance._after_step(None, (), {"pdd_step_index": step}, None)
        return [Image.new("RGB", (kwargs["width"], kwargs["height"]), "teal")]

    def backend(self, **overrides):
        config = {"gguf": str(self.gguf), "base_model": str(self.base), **overrides}
        self.backend_instance = backend_module.minimaxh3image(config)
        return self.backend_instance

    def request(self, **overrides):
        return {"output_file": str(self.output), "width": 96, "height": 64, "seed": 123456, **overrides}

    def test_exports_media_backend_and_is_found_by_the_dynamic_loader(self):
        self.assertTrue(issubclass(backend_module.minimaxh3image, MediaBackend))
        self.assertEqual(backend_module.minimaxh3image.model_id, "minimaxh3image")
        self.assertIsNone(backend_module.minimaxh3image.scheduler_family)

    def test_default_paths_are_absolute_and_project_local(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(Path(backend_module.DEFAULT_GGUF), root / "models" / "minimax_h3_image_Q4_K_S.gguf")
        self.assertEqual(Path(backend_module.DEFAULT_BASE_MODEL), root / "models" / "minimax-h3-image")
        self.assertTrue(Path(backend_module.DEFAULT_GGUF).is_absolute())

    def test_missing_gguf_or_directory_fails_before_loading(self):
        for bad in (str(self.directory / "missing.gguf"), str(self.directory)):
            with self.assertRaises(FileNotFoundError):
                backend_module.minimaxh3image({"gguf": bad, "base_model": str(self.base)})
        self.builder.assert_not_called()

    def test_remote_gguf_or_component_folder_is_rejected_before_loading(self):
        for gguf_path in ("https://example.com/m.gguf", "hf.co/org/repo:Q4_K_S", "huggingface.co/org/repo"):
            with self.assertRaisesRegex(ValueError, "local"):
                backend_module.minimaxh3image({"gguf": gguf_path, "base_model": str(self.base)})
        with self.assertRaisesRegex(ValueError, "local"):
            backend_module.minimaxh3image(
                {"gguf": str(self.gguf), "base_model": "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image"}
            )
        # A bare repo id is a missing path (never silently treated as a remote repository).
        with self.assertRaises(FileNotFoundError):
            backend_module.minimaxh3image({"gguf": str(self.gguf), "base_model": "MiniMaxAI/MiniMax-H3-Image"})
        self.builder.assert_not_called()

    def test_incomplete_component_folder_names_what_is_missing(self):
        for missing in ("vae", "text_encoder", "tokenizer", "processor", "scheduler", "pdd_config.json"):
            with TemporaryDirectory() as directory:
                base = make_component_folder(directory, drop=missing)
                with self.assertRaisesRegex(FileNotFoundError, missing):
                    backend_module.minimaxh3image({"gguf": str(self.gguf), "base_model": str(base)})
        self.builder.assert_not_called()

    def test_generation_forwards_prompt_size_and_seeded_cpu_generator(self):
        backend = self.backend()
        progress = {}
        config = self.request(progress_state=progress)
        original = dict(config)
        path, prompt = backend.execute("  a lighthouse at dusk  ", config)
        self.assertEqual(config, original)
        self.assertEqual((path, prompt), (str(self.output), "a lighthouse at dusk"))
        self.assertEqual(backend.last_imagegen_prompt, prompt)
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["prompt"], prompt)
        self.assertEqual((kwargs["width"], kwargs["height"]), (96, 64))
        self.assertEqual(kwargs["output"], "images")
        self.assertEqual(kwargs["generator"].device.type, "cpu")
        self.assertEqual(kwargs["generator"].initial_seed(), 123456)
        for unsupported in ("num_inference_steps", "guidance_scale", "negative_prompt"):
            self.assertNotIn(unsupported, kwargs)
        self.assertEqual(progress, {"step": STEPS, "total_steps": STEPS})

    def test_png_metadata_describes_the_actual_fixed_schedule(self):
        backend = self.backend()
        backend.execute("a fox", self.request(steps=30, guidance_scale=7.0, negative_prompt="blurry"))
        with Image.open(self.output) as saved:
            self.assertEqual(saved.size, (96, 64))
            self.assertEqual(saved.info["model"], "minimaxh3image")
            self.assertEqual(saved.info["inference_steps"], "6")
            self.assertEqual(saved.info["seed"], "123456")
            self.assertEqual((saved.info["width"], saved.info["height"]), ("96", "64"))
            self.assertEqual(saved.info["prompt"], "a fox")
            self.assertNotIn("guidance_scale", saved.info)

    def test_batch_id_names_the_output_file(self):
        backend = self.backend()
        path, _ = backend.execute("a fox", {"width": 64, "height": 64, "seed": 1}, batch_id="b7")
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        self.assertEqual(path, "/tmp/ircawp_generated/minimaxh3image.b7.png")

    def test_missing_seed_is_generated_and_recorded(self):
        backend = self.backend()
        config = self.request()
        config.pop("seed")
        backend.execute("a fox", config)
        seed = self.pipeline.call_args.kwargs["generator"].initial_seed()
        with Image.open(self.output) as saved:
            self.assertEqual(saved.info["seed"], str(seed))

    def test_dimensions_derive_from_aspect_and_max_output_size_on_the_32_grid(self):
        backend = self.backend()
        for config, expected in (
            ({"max_output_size": 1024}, (1024, 1024)),
            ({"max_output_size": 2048, "aspect": "16:9"}, (2048, 1152)),
            ({"max_output_size": 2048, "aspect": "9:16"}, (1152, 2048)),
            ({"max_output_size": 1000, "aspect": 1.5}, (992, 672)),
            ({}, (2048, 2048)),
        ):
            with self.subTest(config=config):
                backend.execute("a fox", {"output_file": str(self.output), "seed": 1, **config})
                kwargs = self.pipeline.call_args.kwargs
                self.assertEqual((kwargs["width"], kwargs["height"]), expected)
                self.assertEqual(kwargs["width"] % 32, 0)
                self.assertEqual(kwargs["height"] % 32, 0)

    def test_explicit_dimensions_take_precedence_over_aspect_and_max_size(self):
        backend = self.backend()
        backend.execute("a fox", self.request(aspect="16:9", max_output_size=4096, width=1360, height=768))
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (1360, 768))

    # -- editing -----------------------------------------------------------

    def reference(self, name="ref.png", size=(640, 480), color="red", mode="RGB"):
        path = self.directory / name
        Image.new(mode, size, color).save(path)
        return str(path)

    def test_text_to_image_passes_no_image_argument(self):
        self.backend().execute("a fox", self.request())
        self.assertNotIn("image", self.pipeline.call_args.kwargs)
        with Image.open(self.output) as saved:
            self.assertNotIn("task", saved.info)
            self.assertNotIn("references", saved.info)

    def test_references_are_forwarded_in_order_as_rgb_images(self):
        first = self.reference("a.png", (640, 480), "red")
        second = self.reference("b.png", (300, 400), (0, 128, 255, 90), mode="RGBA")
        backend = self.backend()
        media = [first, second]
        backend.execute("Swap <Picture 1> and <Picture 2>", self.request(), media=media)
        images = self.pipeline.call_args.kwargs["image"]
        self.assertEqual([image.size for image in images], [(640, 480), (300, 400)])
        self.assertTrue(all(image.mode == "RGB" for image in images))
        self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(media, [first, second])  # the caller's list is not mutated
        # The prompt, with its <Picture N> references, is passed verbatim.
        self.assertEqual(self.pipeline.call_args.kwargs["prompt"], "Swap <Picture 1> and <Picture 2>")

    def test_edit_metadata_records_the_task_and_reference_count(self):
        backend = self.backend()
        backend.execute("edit", self.request(), media=[self.reference("a.png"), self.reference("b.png")])
        with Image.open(self.output) as saved:
            self.assertEqual(saved.info["task"], "image-edit")
            self.assertEqual(saved.info["references"], "2")
            self.assertEqual(saved.info["inference_steps"], "6")

    def test_exif_orientation_is_applied_before_sizing(self):
        path = self.directory / "rotated.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6  # stored 320x160, displayed rotated to 160x320
        Image.new("RGB", (320, 160), "blue").save(path, exif=exif)
        backend = self.backend()
        config = {"output_file": str(self.output), "seed": 1, "max_output_size": 320}
        backend.execute("edit", config, media=[str(path)])
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["image"][0].size, (160, 320))
        # Portrait, not the stored landscape (which would give 320x160).
        self.assertEqual((kwargs["width"], kwargs["height"]), (160, 320))

    def test_edit_output_follows_the_first_reference_aspect_at_max_output_size(self):
        backend = self.backend()
        for size, max_edge, expected in (
            ((640, 480), 320, (320, 240)),
            ((480, 640), 320, (240, 320)),
            ((512, 512), 1024, (1024, 1024)),
            ((300, 200), 960, (960, 640)),
        ):
            with self.subTest(size=size, max_edge=max_edge):
                config = {"output_file": str(self.output), "seed": 1, "max_output_size": max_edge}
                backend.execute("edit", config, media=[self.reference("r.png", size)])
                kwargs = self.pipeline.call_args.kwargs
                self.assertEqual((kwargs["width"], kwargs["height"]), expected)

    def test_only_the_first_reference_decides_the_output_aspect(self):
        backend = self.backend()
        config = {"output_file": str(self.output), "seed": 1, "max_output_size": 320}
        backend.execute(
            "edit", config, media=[self.reference("a.png", (640, 480)), self.reference("b.png", (100, 400))]
        )
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (320, 240))

    def test_edit_defaults_to_the_models_2048_long_edge(self):
        backend = self.backend()
        backend.execute("edit", {"output_file": str(self.output), "seed": 1}, media=[self.reference(size=(600, 300))])
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (2048, 1024))

    def test_explicit_size_beats_the_reference_aspect(self):
        backend = self.backend()
        backend.execute("edit", self.request(width=1360, height=768), media=[self.reference(size=(300, 900))])
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (1360, 768))

    def test_up_to_sixteen_references_are_accepted_and_more_are_rejected_before_generating(self):
        backend = self.backend()
        sixteen = [self.reference(f"r{i}.png", (64, 64)) for i in range(16)]
        backend.execute("edit", self.request(), media=sixteen)
        self.assertEqual(len(self.pipeline.call_args.kwargs["image"]), 16)
        self.pipeline.reset_mock()
        with self.assertRaisesRegex(ValueError, "at most 16"):
            backend.execute("edit", self.request(), media=sixteen + [self.reference("extra.png", (64, 64))])
        self.pipeline.assert_not_called()

    def test_too_small_reference_is_rejected_by_name_before_generating(self):
        tiny = self.reference("tiny.png", (31, 200))
        with self.assertRaisesRegex(ValueError, r"tiny\.png.*at least 32"):
            self.backend().execute("edit", self.request(), media=[self.reference("ok.png"), tiny])
        self.pipeline.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unreadable_or_missing_reference_is_a_clear_error_not_a_crash(self):
        broken = self.directory / "broken.png"
        broken.write_bytes(b"not an image")
        backend = self.backend()
        for path in (str(broken), str(self.directory / "missing.png")):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "cannot read reference image"):
                    backend.execute("edit", self.request(), media=[path])
        self.pipeline.assert_not_called()

    def test_edit_reports_the_reference_phase_and_text_to_image_does_not(self):
        backend = self.backend()
        phases = []

        def record(**kwargs):
            phases.append(progress.get("phase"))
            return [Image.new("RGB", (kwargs["width"], kwargs["height"]))]

        self.pipeline.side_effect = record
        progress = {}
        backend.execute("edit", self.request(progress_state=progress), media=[self.reference()])
        backend.execute("a fox", self.request(progress_state=progress))
        self.assertEqual(phases, ["encoding references", "encoding prompt"])

    def test_cancellation_still_aborts_an_edit_between_steps(self):
        backend = self.backend()
        cancellation = Event()

        def cancel_after_first_step(**kwargs):
            backend._before_step(None, (), {"pdd_step_index": 0})
            backend._after_step(None, (), {"pdd_step_index": 0}, None)
            cancellation.set()
            backend._before_step(None, (), {"pdd_step_index": 1})
            self.fail("cancellation did not abort the edit")

        self.pipeline.side_effect = cancel_after_first_step
        with self.assertRaises(GenerationCancelled):
            backend.execute("edit", self.request(cancellation_event=cancellation), media=[self.reference()])
        self.assertFalse(self.output.exists())
        self.assertIsNone(backend._active_config)

    def test_empty_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prompt"):
            self.backend().execute("   ", self.request())
        self.pipeline.assert_not_called()

    def test_cancellation_between_steps_aborts_before_png_is_saved(self):
        backend = self.backend()
        cancellation = Event()
        progress = {}

        def cancel_after_second_step(**kwargs):
            for step in range(STEPS):
                backend._before_step(None, (), {"pdd_step_index": step})
                backend._after_step(None, (), {"pdd_step_index": step}, None)
                if step == 1:
                    cancellation.set()
            self.fail("cancellation did not abort generation")

        self.pipeline.side_effect = cancel_after_second_step
        with self.assertRaises(GenerationCancelled):
            backend.execute("cancel me", self.request(cancellation_event=cancellation, progress_state=progress))
        self.assertEqual(progress, {"step": 2, "total_steps": STEPS})
        self.assertFalse(self.output.exists())
        self.assertIsNone(backend._active_config)

    def test_cancellation_before_start_never_calls_the_pipeline(self):
        cancellation = Event()
        cancellation.set()
        with self.assertRaises(GenerationCancelled):
            self.backend().execute("cancel me", self.request(cancellation_event=cancellation))
        self.pipeline.assert_not_called()

    def test_request_state_does_not_leak_into_the_next_request(self):
        backend = self.backend()
        first = {}
        backend.execute("one", self.request(progress_state=first))
        self.assertIsNone(backend._active_config)
        self.pipeline.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            backend.execute("two", self.request())
        self.assertIsNone(backend._active_config)
        second = {}
        self.pipeline.side_effect = self.render
        backend.execute("three", self.request(progress_state=second))
        self.assertEqual(second, {"step": STEPS, "total_steps": STEPS})

    def hooks_that_record(self, backend):
        offloaded = []
        backend.components_manager = SimpleNamespace(
            model_hooks=[SimpleNamespace(offload=lambda i=i: offloaded.append(i)) for i in range(3)]
        )
        return offloaded

    def test_gpu_is_kept_between_requests_by_default(self):
        backend = self.backend()
        self.assertFalse(backend.release_gpu_after_request)
        offloaded = self.hooks_that_record(backend)
        backend.execute("a fox", self.request())
        self.assertEqual(offloaded, [])

    def test_release_gpu_after_request_offloads_after_success_and_after_failure(self):
        backend = self.backend(release_gpu_after_request=True)
        self.assertTrue(backend.release_gpu_after_request)
        offloaded = self.hooks_that_record(backend)
        backend.execute("a fox", self.request())
        self.assertEqual(offloaded, [0, 1, 2])
        offloaded.clear()
        self.pipeline.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            backend.execute("a fox", self.request())
        self.assertEqual(offloaded, [0, 1, 2])

    def test_progress_reports_phases_in_order_and_clears_them_afterwards(self):
        backend = self.backend()
        progress = {}
        seen = []

        def record_phases(**kwargs):
            seen.append(progress.get("phase"))
            for step in range(STEPS):
                backend._before_step(None, (), {"pdd_step_index": step})
                seen.append((progress.get("phase"), progress.get("step")))
                backend._after_step(None, (), {"pdd_step_index": step}, None)
            seen.append((progress.get("phase"), progress.get("step")))
            return [Image.new("RGB", (kwargs["width"], kwargs["height"]))]

        self.pipeline.side_effect = record_phases
        backend.execute("a fox", self.request(progress_state=progress))
        self.assertEqual(seen[0], "encoding prompt")
        self.assertEqual([phase for phase, _ in seen[1:-1]], ["denoising"] * STEPS)
        self.assertEqual(seen[-1], ("decoding", STEPS))
        self.assertNotIn("phase", progress)
        self.assertEqual(progress, {"step": STEPS, "total_steps": STEPS})

    def test_phase_is_cleared_when_generation_fails_or_is_cancelled(self):
        backend = self.backend()
        progress = {}
        self.pipeline.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            backend.execute("a fox", self.request(progress_state=progress))
        self.assertNotIn("phase", progress)

    # -- timing log --------------------------------------------------------

    def run_and_capture(self, backend, *args, **kwargs):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = backend.execute(*args, **kwargs)
        return result, buffer.getvalue()


    def test_dispose_releases_pipeline_and_component_manager(self):
        backend = self.backend()
        backend.components_manager = object()
        backend.dispose()
        self.assertIsNone(backend.pipe)
        self.assertIsNone(backend.components_manager)

    def test_capabilities_hide_steps_and_guidance_and_offer_no_scheduler(self):
        from app import main

        caps = main._backend_capabilities("minimaxh3image")
        self.assertTrue(caps["seed"])
        self.assertFalse(caps["steps"])
        self.assertFalse(caps["cfg"])
        self.assertFalse(caps["negative_prompt"])
        self.assertEqual(caps["schedulers"], [])


class MinimaxMirrorWiringTests(unittest.TestCase):
    def test_manager_hooks_are_rewired_to_free_instead_of_copy_back(self):
        module = torch.nn.Sequential(torch.nn.Linear(4, 3))
        calls = []
        hook = SimpleNamespace(init_hook=lambda m: calls.append("copy-back"))
        user_hook = SimpleNamespace(model=module, hook=hook)
        backend = object.__new__(backend_module.minimaxh3image)
        backend.components_manager = SimpleNamespace(model_hooks=[user_hook])
        backend._install_cpu_mirrors()
        weight = module[0].weight.data
        module.to(torch.float64)
        hook.init_hook(module)
        self.assertEqual(calls, [])
        self.assertEqual(module[0].weight.data.data_ptr(), weight.data_ptr())

    def test_fast_offload_can_be_switched_off(self):
        with TemporaryDirectory() as directory:
            gguf_path = Path(directory) / "t.gguf"
            gguf_path.write_bytes(b"x")
            base = make_component_folder(directory)
            with patch.object(backend_module.minimaxh3image, "_build_pipeline", return_value=object()):
                on = backend_module.minimaxh3image({"gguf": str(gguf_path), "base_model": str(base)})
                off = backend_module.minimaxh3image(
                    {"gguf": str(gguf_path), "base_model": str(base), "fast_offload": False}
                )
        self.assertTrue(on.fast_offload)
        self.assertFalse(off.fast_offload)


class TabulatedAdaLNTests(unittest.TestCase):
    def test_returns_six_chunks_with_zeroed_audio_rows(self):
        hidden = 3
        table = torch.arange(2 * 2 * 6 * hidden, dtype=torch.float32).view(2, 2 * 6 * hidden)
        module = backend_module.TabulatedAdaLN(table)
        module.rows = torch.tensor([1, 0])
        chunks = module(torch.zeros(2, 4))
        self.assertEqual(len(chunks), 6)
        for chunk in chunks:
            self.assertEqual(tuple(chunk.shape), (2 * 3, hidden))  # (timesteps * 3 modalities, hidden)
            self.assertTrue(torch.equal(chunk.view(2, 3, hidden)[:, 2], torch.zeros(2, hidden, dtype=chunk.dtype)))
        # timestep 0 of this call is table row 1; its image modality is the first half of that row.
        expected = table[1].view(2, 6, hidden)[0].to(torch.bfloat16)
        got = torch.stack([c.view(2, 3, hidden)[0, 0] for c in chunks])
        self.assertTrue(torch.equal(got, expected))

    def test_use_before_row_selection_is_an_error(self):
        module = backend_module.TabulatedAdaLN(torch.zeros(1, 12))
        with self.assertRaises(RuntimeError):
            module(torch.zeros(1, 4))

    def test_hook_rejects_timesteps_off_the_checkpoint_schedule(self):
        module = backend_module.TabulatedAdaLN(torch.zeros(2, 12))
        hook = backend_module._select_timestep_rows([module], torch.tensor([0.0, 0.5]))
        hook(None, (), {"timestep": torch.tensor([0.5, 0.0])})
        self.assertEqual(module.rows.tolist(), [1, 0])
        with self.assertRaisesRegex(ValueError, "schedule"):
            hook(None, (), {"timestep": torch.tensor([0.25])})


# Tiny multiples of 32 so Q8_0 could be used too; the loader is size-agnostic.
LOAD = dict(
    num_attention_heads=2, attention_head_dim=32, hidden_size=64, num_layers=2, num_refiner_layers=1,
    ffn_dim=64, text_dim=32, freq_dim=32, time_embed_hidden_dim=64, time_embed_dim=32,
    rope_freq_dim=1, num_pdd_heads=3,
)


def tiny_dense_model():
    from diffusers import MiniMaxH3ImageTransformer2DModel

    torch.manual_seed(0)
    model = MiniMaxH3ImageTransformer2DModel(**LOAD).to(torch.bfloat16)
    for name in ("proj_in", "time_embedder"):
        getattr(model, name).to(torch.float32)
    for head in model.pdd_proj_out:
        head.to(torch.float32)
    for tensor in model.parameters():  # a non-trivial signal everywhere
        tensor.data.mul_(4.0) if tensor.ndim == 1 else None
    return model.eval()


def write_loader_gguf(path, model, timesteps, quantize_ff=False):
    """Write a GGUF the way the converter does, from a dense model's own state dict."""
    from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageAdaLayerNormModulation  # noqa: F401

    writer = gguf.GGUFWriter(str(path), "qwen2")
    writer.add_block_count(LOAD["num_layers"])
    writer.add_embedding_length(LOAD["hidden_size"])

    def add(name, tensor, qtype=None):
        if qtype is None:
            tensor = tensor.float().contiguous()
            writer.add_tensor(name, tensor.numpy(), raw_shape=tensor.shape, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            data = gguf.quants.quantize(tensor.float().numpy(), qtype)
            writer.add_tensor(name, data, raw_dtype=qtype)  # quantized: byte shape comes from data

    stamp = torch.tensor(timesteps, dtype=torch.float32)
    with torch.no_grad():
        temb = model.time_embedder(model.time_proj(stamp).to(torch.float32))
    for name, tensor in model.state_dict().items():
        if ".adaln_proj.linear." in name:
            continue
        qtype = gguf.GGMLQuantizationType.Q8_0 if quantize_ff and name.endswith("ff.net.2.weight") else None
        add(name, tensor, qtype)
    for index, block in enumerate(model.transformer_blocks):
        with torch.no_grad():
            out = torch.nn.functional.linear(
                torch.nn.functional.silu(temb).to(torch.bfloat16), block.adaln_proj.linear.weight,
                block.adaln_proj.linear.bias,
            )
        out = out.view(len(timesteps), 3, -1)[:, :2].reshape(len(timesteps), -1)
        add(f"transformer_blocks.{index}.adaln_proj.table", out)
    add("adaln_timesteps", stamp)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def transformer_inputs(timestep):
    n_text, side = 5, 3
    n_image = side * side
    sequence = n_text + n_image
    generator = torch.Generator().manual_seed(1)
    return dict(
        hidden_states=torch.randn(1, n_image, 96, generator=generator),
        encoder_hidden_states=torch.randn(1, n_text, LOAD["text_dim"], generator=generator),
        timestep=torch.tensor([timestep]),
        timestep_indices=torch.zeros(sequence, dtype=torch.long),
        token_tags=torch.cat([torch.ones(n_text), torch.zeros(n_image)]).long(),
        position_ids=torch.stack(
            [torch.zeros(sequence), torch.arange(sequence) // side, torch.arange(sequence) % side], 1
        ).long(),
        image_indices=torch.arange(n_text, sequence),
        text_indices=torch.arange(n_text),
    )


class GGUFTransformerLoaderTests(unittest.TestCase):
    TIMESTEPS = [0.0, 0.25, 0.5, 0.999]

    def test_loaded_transformer_reproduces_the_dense_model_with_tabulated_adaln(self):
        dense = tiny_dense_model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            write_loader_gguf(path, dense, self.TIMESTEPS)
            loaded = backend_module._load_gguf_transformer(str(path), config=LOAD)
        for step, timestep in enumerate(self.TIMESTEPS[:3]):
            inputs = transformer_inputs(timestep)
            with torch.no_grad():
                expected = dense(**inputs, pdd_step_index=step, return_dict=False)[0]
                got = loaded(**inputs, pdd_step_index=step, return_dict=False)[0]
            self.assertEqual(got.shape, expected.shape)
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)

    def test_edit_style_call_with_two_distinct_timesteps_selects_each_row(self):
        dense = tiny_dense_model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            write_loader_gguf(path, dense, self.TIMESTEPS)
            loaded = backend_module._load_gguf_transformer(str(path), config=LOAD)
        inputs = transformer_inputs(0.25)
        # Reference rows at 0.999 (index 1 in the sorted unique pair), the rest at 0.25.
        inputs["timestep"] = torch.tensor([0.25, 0.999])
        indices = torch.zeros(len(inputs["token_tags"]), dtype=torch.long)
        indices[-3:] = 1
        inputs["timestep_indices"] = indices
        with torch.no_grad():
            expected = dense(**inputs, pdd_step_index=1, return_dict=False)[0]
            got = loaded(**inputs, pdd_step_index=1, return_dict=False)[0]
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)

    def test_quantized_linear_weights_use_the_gguf_path(self):
        from diffusers.quantizers.gguf.utils import GGUFLinear

        dense = tiny_dense_model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            write_loader_gguf(path, dense, self.TIMESTEPS, quantize_ff=True)
            loaded = backend_module._load_gguf_transformer(str(path), config=LOAD)
        self.assertIsInstance(loaded.transformer_blocks[0].ff.net[2], GGUFLinear)
        inputs = transformer_inputs(0.0)
        with torch.no_grad():
            expected = dense(**inputs, pdd_step_index=0, return_dict=False)[0]
            got = loaded(**inputs, pdd_step_index=0, return_dict=False)[0]
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(got, expected, rtol=1e-1, atol=1e-1)

    def test_fp32_modules_stay_float32_and_the_rest_runs_in_bf16(self):
        dense = tiny_dense_model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            write_loader_gguf(path, dense, self.TIMESTEPS)
            loaded = backend_module._load_gguf_transformer(str(path), config=LOAD)
        self.assertEqual(loaded.proj_in.weight.dtype, torch.float32)
        self.assertEqual(loaded.time_embedder.linear_1.weight.dtype, torch.float32)
        self.assertEqual(loaded.pdd_proj_out[0].weight.dtype, torch.float32)
        self.assertEqual(loaded.context_embedder.weight.dtype, torch.bfloat16)
        self.assertEqual(loaded.transformer_blocks[0].attn.to_q.weight.dtype, torch.bfloat16)

    def test_gguf_without_adaln_tables_is_rejected(self):
        dense = tiny_dense_model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            writer = gguf.GGUFWriter(str(path), "qwen2")
            tensor = torch.zeros(4, 4)
            writer.add_tensor("proj_in.weight", tensor.numpy(), raw_shape=tensor.shape, raw_dtype=gguf.GGMLQuantizationType.F32)
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
            with self.assertRaisesRegex(ValueError, "adaln"):
                backend_module._load_gguf_transformer(str(path), config=LOAD)
        del dense

    def test_gguf_missing_a_tensor_is_rejected(self):
        dense = tiny_dense_model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            write_loader_gguf(path, dense, self.TIMESTEPS)
            reader = gguf.GGUFReader(str(path))
            names = [t.name for t in reader.tensors if t.name != "context_embedder.weight"]
            broken = Path(directory) / "broken.gguf"
            writer = gguf.GGUFWriter(str(broken), "qwen2")
            for t in reader.tensors:
                if t.name in names:
                    shape = tuple(int(x) for x in reversed(t.shape))
                    writer.add_tensor(t.name, t.data.copy(), raw_shape=shape, raw_dtype=t.tensor_type)
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
            with self.assertRaisesRegex(ValueError, "context_embedder"):
                backend_module._load_gguf_transformer(str(broken), config=LOAD)

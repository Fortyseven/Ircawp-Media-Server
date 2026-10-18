"""GPU-free checks for the Ming-Image loader: import isolation, GGUF reading, fused experts."""

import os
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import gguf
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from gguf import GGMLQuantizationType as QuantType
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
import transformers.utils.import_utils as import_utils

from app.backends import ming_loader
from app.backends.ming_loader import (
    ExpertBank,
    _llm_target,
    _patch_moe,
    ming_imports,
    read_gguf_tensor,
)


def write_fake_upstream(root: str) -> None:
    with open(os.path.join(root, "ming_fake_flat.py"), "w") as handle:
        handle.write(
            "from transformers.utils.import_utils import is_torch_fx_available\n"
            "from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS\n"
            "assert 'default' in ROPE_INIT_FUNCTIONS\n"
            "VALUE = is_torch_fx_available()\n"
        )
    package = os.path.join(root, "ming_fake_pkg")
    os.makedirs(package)
    open(os.path.join(package, "__init__.py"), "w").close()
    with open(os.path.join(package, "sub.py"), "w") as handle:
        handle.write("VALUE = 1\n")


class ImportIsolationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        write_fake_upstream(self.root)
        self.path_before = list(sys.path)

    def assert_clean(self):
        self.assertEqual(sys.path, self.path_before)
        self.assertNotIn("ming_fake_flat", sys.modules)
        self.assertNotIn("ming_fake_pkg", sys.modules)
        self.assertNotIn("ming_fake_pkg.sub", sys.modules)

    def test_shims_exist_only_inside_the_block(self):
        had_fx = hasattr(import_utils, "is_torch_fx_available")
        had_rope = "default" in ROPE_INIT_FUNCTIONS
        with ming_imports(self.root):
            import ming_fake_flat
            import ming_fake_pkg.sub

            self.assertFalse(ming_fake_flat.VALUE and not had_fx)
            self.assertEqual(ming_fake_pkg.sub.VALUE, 1)
        self.assertEqual(hasattr(import_utils, "is_torch_fx_available"), had_fx)
        self.assertEqual("default" in ROPE_INIT_FUNCTIONS, had_rope)
        self.assert_clean()

    def test_existing_entries_are_left_alone(self):
        sentinel = object()
        with unittest.mock.patch.dict(ROPE_INIT_FUNCTIONS, {"default": sentinel}):
            with ming_imports(self.root):
                self.assertIs(ROPE_INIT_FUNCTIONS["default"], sentinel)
            self.assertIs(ROPE_INIT_FUNCTIONS["default"], sentinel)

    def test_cleanup_runs_when_the_block_raises(self):
        with self.assertRaises(RuntimeError):
            with ming_imports(self.root):
                import ming_fake_flat  # noqa: F401

                raise RuntimeError("boom")
        self.assert_clean()

    def test_missing_directory_points_at_the_setup_recipe(self):
        with self.assertRaisesRegex(FileNotFoundError, "just ming-setup"):
            with ming_imports(os.path.join(self.root, "absent")):
                pass


def pack_q8_0(array: np.ndarray) -> np.ndarray:
    return gguf.quants.quantize(array.astype(np.float32), QuantType.Q8_0)


class GgufReadingTest(unittest.TestCase):
    def test_bf16_f32_and_packed_tensors_round_trip(self):
        rng = np.random.default_rng(0)
        bf16 = torch.tensor(rng.standard_normal((4, 8)), dtype=torch.bfloat16)
        f32 = rng.standard_normal((3, 5)).astype(np.float32)
        packed = pack_q8_0(rng.standard_normal((2, 64)))
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.gguf")
            writer = gguf.GGUFWriter(path, "llama")
            writer.add_tensor(
                "bf", bf16.view(torch.uint8).numpy(), raw_dtype=QuantType.BF16
            )
            writer.add_tensor("f32", f32)
            writer.add_tensor("q8", packed, raw_dtype=QuantType.Q8_0)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            tensors = {
                t.name: read_gguf_tensor(t) for t in gguf.GGUFReader(path).tensors
            }

        value, quant_type = tensors["bf"]
        self.assertEqual((value.dtype, tuple(value.shape)), (torch.bfloat16, (4, 8)))
        self.assertTrue(torch.equal(value, bf16))
        self.assertEqual(quant_type, QuantType.BF16)
        value, _ = tensors["f32"]
        self.assertTrue(np.array_equal(value.numpy(), f32))
        value, quant_type = tensors["q8"]
        self.assertEqual((value.dtype, quant_type), (torch.uint8, QuantType.Q8_0))
        self.assertEqual(tuple(value.shape), packed.shape)


class LlmNameMappingTest(unittest.TestCase):
    def test_renames_follow_the_upstream_module_names(self):
        cases = {
            "thinker.embed_tokens.weight": "word_embeddings.weight",
            "thinker.norm.weight": "norm.weight",
            "thinker.layers.3.mlp.gate.proj.weight": "layers.3.mlp.gate.weight",
            "thinker.layers.3.mlp.image_gate.proj.weight": "layers.3.mlp.image_gate.weight",
            "thinker.layers.3.mlp.gate.expert_bias": "layers.3.mlp.gate.expert_bias",
            "thinker.layers.0.mlp.gate_proj.weight": "layers.0.mlp.gate_proj.weight",
            "thinker.layers.2.attention.dense.weight": "layers.2.attention.dense.weight",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(_llm_target(source), expected)


HIDDEN, INTER, EXPERTS = 64, 32, 4
# Diffusers decodes Q8_0 in half precision, so it differs slightly from gguf-py;
# swapping gate and up would be off by orders of magnitude more.
ATOL, RTOL = 1e-3, 3e-2


def make_bank(seed: int = 0):
    rng = np.random.default_rng(seed)
    gate_up = rng.standard_normal((EXPERTS, 2 * INTER, HIDDEN)).astype(np.float32) * 0.1
    down = rng.standard_normal((EXPERTS, HIDDEN, INTER)).astype(np.float32) * 0.1
    gate_up_q, down_q = pack_q8_0(gate_up), pack_q8_0(down)
    bank = ExpertBank()
    bank.set("gate_up", torch.from_numpy(gate_up_q), QuantType.Q8_0)
    bank.set("down", torch.from_numpy(down_q), QuantType.Q8_0)
    # Reference weights are what the packed bytes decode to.
    return (
        bank,
        torch.from_numpy(gguf.quants.dequantize(gate_up_q, QuantType.Q8_0)),
        torch.from_numpy(gguf.quants.dequantize(down_q, QuantType.Q8_0)),
    )


def reference_expert(gate_up, down, index, x):
    gate, up = gate_up[index, :INTER], gate_up[index, INTER:]
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down[index].T


class FusedExpertsTest(unittest.TestCase):
    def test_gate_is_the_first_half_of_the_fused_projection(self):
        bank, gate_up, down = make_bank()
        x = torch.randn(5, HIDDEN)
        for index in range(EXPERTS):
            torch.testing.assert_close(
                bank.forward_expert(index, x),
                reference_expert(gate_up, down, index, x),
                atol=ATOL,
                rtol=RTOL,
            )

    def test_routed_output_matches_a_per_token_reference(self):
        bank, gate_up, down = make_bank(seed=1)
        block_class = type("Block", (), {})
        _patch_moe(SimpleNamespace(BailingMoeV2SparseMoeBlock=block_class))

        tokens, top_k = 7, 2
        x = torch.randn(tokens, HIDDEN)
        topk_ids = torch.stack([torch.randperm(EXPERTS)[:top_k] for _ in range(tokens)])
        topk_weight = torch.rand(tokens, top_k)
        block = SimpleNamespace(
            config=SimpleNamespace(num_experts=EXPERTS), experts=bank
        )
        actual = block_class.moe_infer(block, x, topk_ids, topk_weight)

        expected = torch.stack(
            [
                sum(
                    topk_weight[t, k]
                    * reference_expert(
                        gate_up, down, int(topk_ids[t, k]), x[t : t + 1]
                    )[0]
                    for k in range(top_k)
                )
                for t in range(tokens)
            ]
        )
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)

    def test_unrouted_experts_are_never_dequantized(self):
        bank, _, _ = make_bank()
        block_class = type("Block", (), {})
        _patch_moe(SimpleNamespace(BailingMoeV2SparseMoeBlock=block_class))
        seen = []
        original = bank.forward_expert
        bank.forward_expert = lambda index, x: (seen.append(index), original(index, x))[
            1
        ]
        block = SimpleNamespace(
            config=SimpleNamespace(num_experts=EXPERTS), experts=bank
        )
        block_class.moe_infer(
            block,
            torch.randn(3, HIDDEN),
            torch.tensor([[1], [1], [3]]),
            torch.ones(3, 1),
        )
        self.assertEqual(sorted(seen), [1, 3])


class ExpertBankDeviceTest(unittest.TestCase):
    def test_banks_are_unset_until_set_is_called(self):
        bank = ExpertBank()
        self.assertIsNone(bank.gate_up)
        self.assertIsNone(bank.down)

    def test_packed_banks_follow_to_device_without_entering_the_state_dict(self):
        bank, _, _ = make_bank()
        self.assertEqual(bank.state_dict(), {})
        bank.to("meta")
        self.assertEqual(bank.gate_up.device.type, "meta")
        self.assertEqual(bank.down.device.type, "meta")
        self.assertEqual(bank.state_dict(), {})


def make_stub_encoder():
    modules = [nn.Linear(2, 2) for _ in range(5)]
    host = SimpleNamespace(
        query_tokens_dict=nn.ParameterDict({"16x16": nn.Parameter(torch.zeros(2, 2))})
    )
    encoder = ming_loader.MingEncoder(
        *modules, host, object(), "<eos>", ("s", "u", "a"), {}
    )
    return encoder, modules, host


class EncoderDeviceTest(unittest.TestCase):
    def test_to_moves_every_module_and_the_host_query_tokens(self):
        encoder, modules, host = make_stub_encoder()
        result = encoder.to("meta")
        self.assertIs(result, encoder)
        for module in modules:
            for param in module.parameters():
                self.assertEqual(param.device.type, "meta")
        params = list(host.query_tokens_dict.parameters())
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].device.type, "meta")

    def test_to_also_moves_the_vision_tower_and_projection_when_present(self):
        modules = [nn.Linear(2, 2) for _ in range(5)]
        host = SimpleNamespace(
            query_tokens_dict=nn.ParameterDict(
                {"16x16": nn.Parameter(torch.zeros(2, 2))}
            )
        )
        vision = nn.Linear(2, 2)
        linear_proj = nn.Sequential(nn.Linear(2, 2), nn.GELU(), nn.Linear(2, 2))
        encoder = ming_loader.MingEncoder(
            *modules,
            host,
            object(),
            "<eos>",
            ("s", "u", "a"),
            {},
            vision=vision,
            linear_proj=linear_proj,
        )
        self.assertIs(encoder.to("meta"), encoder)
        for module in (vision, linear_proj):
            for param in module.parameters():
                self.assertEqual(param.device.type, "meta")

    def test_to_works_without_vision_modules(self):
        encoder, _, _ = make_stub_encoder()
        self.assertIsNone(encoder.vision)
        self.assertIsNone(encoder.linear_proj)
        encoder.to("meta")


class FakeTokenizer:
    def __init__(self):
        self.texts = []

    def encode(self, text, add_special_tokens=False):
        self.add_special_tokens = add_special_tokens
        self.texts.append(text)
        return SimpleNamespace(ids=[1, 2, 3])


class EncoderTokenizeTest(unittest.TestCase):
    def make_encoder(self):
        tokenizer = FakeTokenizer()
        encoder = ming_loader.MingEncoder(
            *[nn.Linear(2, 2) for _ in range(5)],
            SimpleNamespace(),
            tokenizer,
            "<E>",
            ("SYS", "<U>", "<A>"),
            {},
            image_tools={"tokens": ("<image>", "<imagePatch>", "</image>")},
        )
        return encoder, tokenizer

    def test_text_only_prompt_uses_the_plain_template(self):
        encoder, tokenizer = self.make_encoder()
        input_ids, attention_mask = encoder._tokenize("prompt", torch.device("cpu"))
        self.assertEqual(tokenizer.texts, ["SYS<E><U>prompt<E><A>"])
        self.assertFalse(tokenizer.add_special_tokens)
        self.assertEqual(tuple(input_ids.shape), (1, 3))
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertTrue(torch.equal(attention_mask, torch.ones_like(input_ids)))

    def test_image_tokens_are_inserted_after_the_user_prefix(self):
        encoder, tokenizer = self.make_encoder()
        input_ids, attention_mask = encoder._tokenize(
            "prompt", torch.device("cpu"), image_tokens=3
        )
        self.assertEqual(
            tokenizer.texts,
            [
                "SYS<E><U><image><imagePatch><imagePatch><imagePatch></image>\nprompt<E><A>"
            ],
        )
        self.assertEqual(tuple(input_ids.shape), (1, 3))
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertTrue(torch.equal(attention_mask, torch.ones_like(input_ids)))


class FakeVision:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def __call__(self, pixel_values, **kwargs):
        self.calls.append((pixel_values, kwargs))
        return torch.linspace(-1, 1, self.rows * 8).reshape(self.rows, 8)


class EncoderImageFeaturesTest(unittest.TestCase):
    def make_encoder(self, vision):
        self.fetched = []
        self.processed = []
        marker = object()
        self.marker = marker
        pixel_values = torch.ones(24, 8)
        grid_thw = torch.tensor([[1, 4, 6]])

        def fetch_image(arg):
            self.fetched.append(arg)
            return marker

        def image_processor(**kwargs):
            self.processed.append(kwargs)
            return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}

        linear_proj = (
            nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 4))
            if vision is not None
            else None
        )
        encoder = ming_loader.MingEncoder(
            *[nn.Linear(2, 2) for _ in range(5)],
            SimpleNamespace(),
            object(),
            "<E>",
            ("s", "u", "a"),
            {},
            vision=vision,
            linear_proj=linear_proj,
            image_tools={
                "fetch_image": fetch_image,
                "image_processor": image_processor,
                "tokens": ("<image>", "<imagePatch>", "</image>"),
            },
        )
        return encoder, grid_thw

    def test_returns_normalized_embeddings_and_the_grid(self):
        vision = FakeVision(24)
        encoder, grid_thw = self.make_encoder(vision)
        image = Image.new("RGB", (32, 48))

        embeds, grid = encoder._image_features(image, torch.device("cpu"))

        self.assertEqual(self.fetched, [{"type": "image", "image": image}])
        self.assertIs(self.fetched[0]["image"], image)
        self.assertEqual(len(self.processed), 1)
        self.assertIs(self.processed[0]["images"], self.marker)
        self.assertEqual(self.processed[0]["return_tensors"], "pt")
        self.assertEqual(tuple(embeds.shape), (24, 4))
        torch.testing.assert_close(
            embeds.norm(dim=-1), torch.ones(24), atol=1e-5, rtol=1e-5
        )
        self.assertTrue(torch.equal(grid, grid_thw))
        self.assertEqual(len(vision.calls), 1)
        self.assertTrue(torch.equal(vision.calls[0][1]["grid_thw"], grid_thw))

    def test_encoder_without_vision_refuses_images(self):
        encoder, _ = self.make_encoder(None)
        with self.assertRaisesRegex(RuntimeError, "vision"):
            encoder._image_features(Image.new("RGB", (8, 8)), torch.device("cpu"))


class FakeProfile:
    def __init__(self, max_steps=None):
        self.max_steps = max_steps
        self.calls = []

    def resolve_sampling_parameters(self, steps, cfg):
        self.calls.append((steps, cfg))
        resolved_steps = min(steps, self.max_steps) if self.max_steps else steps
        return SimpleNamespace(steps=resolved_steps, cfg=cfg + 1.0)


def make_fake_generator(max_steps=None):
    image = object()
    calls = []

    def pipelines(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=[image])

    generator = SimpleNamespace(
        inference_profile=FakeProfile(max_steps),
        train_model=SimpleNamespace(dtype=torch.float32),
        device=torch.device("cpu"),
        vae_sample_mode="argmax",
        pipelines=pipelines,
    )
    return generator, calls, image


class SampleImageTest(unittest.TestCase):
    def run_sample(self, generator, seed=1234, steps=20, cfg=4.0, callback=None):
        # linspace, not randn: drawing random numbers would itself move the global RNG.
        context = (
            torch.linspace(-1, 1, 256 * 2560).reshape(1, 256, 2560).to(torch.bfloat16)
        )
        directvlm = (
            torch.linspace(-1, 1, 33 * 3840).reshape(1, 33, 3840).to(torch.bfloat16)
        )
        image = ming_loader.sample_image(
            generator,
            context,
            directvlm,
            steps=steps,
            cfg=cfg,
            seed=seed,
            height=1024,
            width=768,
            callback_on_step_end=callback,
        )
        return image, context, directvlm

    def test_pipeline_receives_resolved_parameters_and_zero_negatives(self):
        generator, calls, sentinel = make_fake_generator(max_steps=8)
        callback = lambda *args, **kwargs: None
        image, context, directvlm = self.run_sample(
            generator, steps=20, cfg=4.0, callback=callback
        )

        self.assertIs(image, sentinel)
        self.assertEqual(generator.inference_profile.calls, [(20, 4.0)])
        self.assertEqual(len(calls), 1)
        kwargs = calls[0]
        self.assertEqual(kwargs["num_inference_steps"], 8)
        self.assertEqual(kwargs["guidance_scale"], 5.0)
        self.assertEqual((kwargs["height"], kwargs["width"]), (1024, 768))
        self.assertEqual(kwargs["max_sequence_length"], 512)
        self.assertEqual(kwargs["device"], generator.device)
        self.assertIsNone(kwargs["ref_hidden_states"])
        self.assertEqual(kwargs["sample_mode"], "argmax")
        self.assertEqual(kwargs["num_frames_per_prompt"], 1)
        self.assertIs(kwargs["callback_on_step_end"], callback)

        for key, negative_key, source in (
            ("prompt_embeds", "negative_prompt_embeds", context),
            ("prompt_embeds_2", "negative_prompt_embeds_2", directvlm),
        ):
            embeds, negatives = kwargs[key], kwargs[negative_key]
            self.assertIsInstance(embeds, list)
            self.assertEqual(len(embeds), 1)
            self.assertEqual(embeds[0].dtype, torch.float32)
            torch.testing.assert_close(embeds[0], source[0].to(torch.float32))
            self.assertEqual(len(negatives), 1)
            self.assertEqual(negatives[0].shape, source[0].shape)
            self.assertEqual(negatives[0].dtype, torch.float32)
            self.assertEqual(negatives[0].abs().sum().item(), 0.0)

    def test_reference_latents_are_forwarded_by_identity(self):
        generator, calls, _ = make_fake_generator()
        ref_x = torch.zeros(1, 4, 16, 16)
        ming_loader.sample_image(
            generator,
            torch.zeros(1, 4, 8),
            torch.zeros(1, 4, 8),
            steps=4,
            cfg=1.0,
            seed=1,
            height=256,
            width=256,
            ref_x=ref_x,
        )
        self.assertIs(calls[0]["ref_hidden_states"], ref_x)

    def test_generator_is_seeded_per_request(self):
        generator, calls, _ = make_fake_generator()
        self.run_sample(generator, seed=987654321)
        torch_generator = calls[0]["generator"]
        self.assertIsInstance(torch_generator, torch.Generator)
        self.assertEqual(torch_generator.initial_seed(), 987654321)

    def test_global_rng_is_left_untouched(self):
        generator, _, _ = make_fake_generator()
        torch.manual_seed(5)
        before = torch.get_rng_state()
        self.run_sample(generator, seed=42)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class PlanSizeErrorTest(unittest.TestCase):
    def test_invalid_sizes_are_rejected_before_any_upstream_import(self):
        for width, height in [(4000, 100), (100, 4000), (0, 10), (-5, 10), (10.5, 10)]:
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    ming_loader.plan_size(
                        width, height, code_dir="/nonexistent/ming-image"
                    )


@unittest.skipUnless(
    os.path.isdir(ming_loader.DEFAULT_CODE_DIR),
    "Ming-Image clone missing; run `just ming-setup`",
)
class PlanSizeUpstreamTest(unittest.TestCase):
    def test_square_buckets_are_kept(self):
        self.assertEqual(ming_loader.plan_size(1024, 1024), (1024, 1024))
        self.assertEqual(ming_loader.plan_size(2048, 2048), (2048, 2048))

    def test_other_ratios_snap_to_16_and_keep_the_bucket_area(self):
        for width, height in [(1024, 672), (1920, 1080)]:
            with self.subTest(width=width, height=height):
                internal_width, internal_height = ming_loader.plan_size(width, height)
                self.assertEqual(internal_width % 16, 0)
                self.assertEqual(internal_height % 16, 0)
                self.assertAlmostEqual(
                    internal_width * internal_height / 1024**2, 1.0, delta=0.2
                )

    def test_upstream_modules_do_not_leak(self):
        ming_loader.plan_size(1024, 672)
        self.assertNotIn("bailingmm_utils", sys.modules)


class PrepareReferenceErrorTest(unittest.TestCase):
    def prepare(self, size=(100, 100), **kwargs):
        return ming_loader.prepare_reference(
            Image.new("RGB", size), code_dir="/nonexistent/ming-image", **kwargs
        )

    def test_exactly_one_dimension_is_rejected(self):
        for kwargs in ({"width": 512}, {"height": 512}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    self.prepare(**kwargs)

    def test_invalid_dimensions_are_rejected_before_any_upstream_import(self):
        for width, height in [
            (0, 10),
            (-5, 10),
            (10.5, 10),
            (10, 10.0),
            (True, 10),
            (10, True),
        ]:
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    self.prepare(width=width, height=height)

    def test_extreme_requested_ratios_are_rejected(self):
        for width, height in [(4000, 100), (100, 4000)]:
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    self.prepare(width=width, height=height)

    def test_extreme_source_ratios_are_rejected(self):
        for size in [(4000, 100), (100, 4000)]:
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    self.prepare(size=size)
                with self.assertRaises(ValueError):
                    self.prepare(size=size, width=512, height=512)


@unittest.skipUnless(
    os.path.isdir(ming_loader.DEFAULT_CODE_DIR),
    "Ming-Image clone missing; run `just ming-setup`",
)
class PrepareReferenceUpstreamTest(unittest.TestCase):
    @staticmethod
    def gradient(width, height, mode="RGB"):
        x = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(height, axis=0)
        y = np.linspace(0, 255, height, dtype=np.uint8)[:, None].repeat(width, axis=1)
        return Image.fromarray(np.stack([x, y, 255 - x], axis=-1), "RGB").convert(mode)

    def test_unsized_reference_keeps_its_size_and_becomes_rgba_latents(self):
        plan = ming_loader.prepare_reference(self.gradient(1280, 720))
        self.assertEqual(plan.image.mode, "RGBA")
        self.assertEqual(plan.image.size, (1280, 720))
        self.assertEqual(tuple(plan.ref_x.shape), (1, 4, plan.height, plan.width))
        self.assertGreaterEqual(plan.ref_x.min().item(), -1.0)
        self.assertLessEqual(plan.ref_x.max().item(), 1.0)
        self.assertEqual(plan.width % 16, 0)
        self.assertEqual(plan.height % 16, 0)
        self.assertGreater(plan.width * plan.height, 0.7 * 1024 * 1024)
        self.assertLess(plan.width * plan.height, 1.3 * 1024 * 1024)

    def test_requested_size_center_crops_to_its_aspect_ratio(self):
        plan = ming_loader.prepare_reference(
            self.gradient(1000, 1000), width=512, height=1024
        )
        self.assertEqual(plan.image.size, (500, 1000))
        self.assertAlmostEqual(plan.width / plan.height, 0.5, delta=0.05)
        self.assertEqual(
            (plan.width, plan.height), (plan.ref_x.shape[-1], plan.ref_x.shape[-2])
        )

    def test_alpha_channel_survives_into_the_fourth_latent_channel(self):
        source = self.gradient(256, 256, "RGBA")
        alpha = np.zeros((256, 256), dtype=np.uint8)
        alpha[:, 128:] = 255
        source.putalpha(Image.fromarray(alpha))
        plan = ming_loader.prepare_reference(source)
        self.assertLess(plan.ref_x[:, 3].min().item(), 0)
        self.assertGreater(plan.ref_x[:, 3].max().item(), 0)

    def test_upstream_modules_do_not_leak(self):
        ming_loader.prepare_reference(self.gradient(640, 480))
        self.assertNotIn("bailingmm_utils", sys.modules)
        self.assertNotIn("processing_bailingmm2", sys.modules)


if __name__ == "__main__":
    unittest.main()

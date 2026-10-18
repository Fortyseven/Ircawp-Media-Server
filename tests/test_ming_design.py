"""GPU-free, weight-free contract for the Ming-Image-0.1-Design backend."""

import copy
import importlib
import inspect
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch
import yaml

from app.backends.MediaBackend import GenerationCancelled, MediaBackend
from app.backends.ming_loader import ReferencePlan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "models" / "ming-image"
VAE_RELATIVE = os.path.join("vae", "diffusion_pytorch_model.safetensors")


class FakeEncoder:
    def __init__(self, calls):
        self.calls = calls
        self.error = None
        self.context = object()
        self.directvlm = object()

    def to(self, device):
        self.calls.append(f"to:{device}")
        return self

    def encode(self, prompt, image=None):
        self.calls.append("encode")
        self.prompts = getattr(self, "prompts", []) + [prompt]
        self.images = getattr(self, "images", []) + [image]
        if self.error is not None:
            raise self.error
        return self.context, self.directvlm


class MingDesignBackendTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.encoder_gguf = self.directory / "encoder.gguf"
        self.encoder_gguf.write_bytes(b"fake encoder")
        self.dit_gguf = self.directory / "dit.gguf"
        self.dit_gguf.write_bytes(b"fake dit")
        self.base = self.directory / "official"
        (self.base / "vae").mkdir(parents=True)
        self.vae = self.base / VAE_RELATIVE
        self.vae.write_bytes(b"fake vae")
        self.output = self.directory / "result.png"

        self.calls = []
        self.encoder = FakeEncoder(self.calls)
        self.generator = object()
        self.sample_kwargs = None
        self.sample_args = None
        self.on_sample = None
        self.ref_x = torch.zeros(1, 4, 16, 32)
        self.plan_image = Image.new("RGBA", (32, 16), (0, 255, 0, 255))
        self.plan = ReferencePlan(self.ref_x, 32, 16, self.plan_image)
        self.reference = self.directory / "reference.png"
        Image.new("RGB", (64, 48), (10, 20, 30)).save(self.reference)

        self.load_encoder = self.enterContext(
            patch("app.backends.ming_design.load_encoder", return_value=self.encoder)
        )
        self.load_generator = self.enterContext(
            patch(
                "app.backends.ming_design.load_generator",
                return_value=self.generator,
            )
        )
        self.sample_image = self.enterContext(
            patch("app.backends.ming_design.sample_image", side_effect=self.fake_sample)
        )
        self.plan_size = self.enterContext(
            patch(
                "app.backends.ming_design.plan_size",
                side_effect=lambda width, height, code_dir=None: (width, height),
            )
        )
        self.had_prepare_reference = hasattr(self.module(), "prepare_reference")
        # create=True keeps unrelated tests runnable; the import itself is checked separately.
        self.prepare_reference = self.enterContext(
            patch(
                "app.backends.ming_design.prepare_reference",
                side_effect=lambda image,
                width=None,
                height=None,
                code_dir=None: self.plan,
                create=True,
            )
        )

    def fake_sample(
        self,
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
        self.calls.append("sample")
        self.sample_args = (generator, context, directvlm)
        self.sample_kwargs = {
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "height": height,
            "width": width,
            "callback": callback_on_step_end,
            "ref_x": ref_x,
        }
        if self.on_sample is not None:
            self.on_sample(callback_on_step_end)
        return Image.new("RGBA", (width, height), (255, 0, 0, 128))

    def module(self):
        return importlib.import_module("app.backends.ming_design")

    def backend_config(self, **overrides):
        return {
            "encoder_gguf": str(self.encoder_gguf),
            "dit_gguf": str(self.dit_gguf),
            "base_dir": str(self.base),
            **overrides,
        }

    def backend(self, **overrides):
        backend = self.module().ming_design(self.backend_config(**overrides))
        self.calls.clear()
        return backend

    def request(self, **overrides):
        return {
            "output_file": str(self.output),
            "width": 96,
            "height": 64,
            "seed": 123456,
            **overrides,
        }

    def edit_request(self, **overrides):
        return {"output_file": str(self.output), "seed": 123456, **overrides}

    def edit(self, backend, prompt="make it blue", **overrides):
        return backend.execute(
            prompt, self.edit_request(**overrides), media=[str(self.reference)]
        )

    def assert_no_loading(self):
        self.load_encoder.assert_not_called()
        self.load_generator.assert_not_called()

    def assert_no_model_calls(self):
        self.assertEqual(self.calls, [])
        self.sample_image.assert_not_called()
        self.prepare_reference.assert_not_called()

    def saved(self):
        with Image.open(self.output) as image:
            image.load()
            return image.copy(), dict(image.info)

    # 1. Constructor config keys and defaults

    def test_backend_class_and_identity(self):
        cls = self.module().ming_design
        self.assertTrue(issubclass(cls, MediaBackend))
        self.assertEqual(cls.model_id, "ming_design")

    def test_default_paths_are_absolute_and_project_relative(self):
        module = self.module()
        expected = {
            "DEFAULT_ENCODER_GGUF": MODELS
            / "Text Encoders (UPDATED)"
            / "Ming-Image-0.1-Ling-Mini-2.0-Q4_K_M.gguf",
            "DEFAULT_DIT_GGUF": MODELS / "Ming-Image-0.1-Design-Q4_K_M.gguf",
            "DEFAULT_BASE_DIR": MODELS / "official",
        }
        for name, path in expected.items():
            with self.subTest(name=name):
                value = getattr(module, name)
                self.assertTrue(os.path.isabs(value))
                self.assertEqual(Path(value), path)

    # 2. Validation before loading

    def test_urls_are_rejected_before_loading(self):
        cls = self.module().ming_design
        remotes = (
            "https://huggingface.co/example/model/resolve/main/model.gguf",
            "hf.co/example/model/model.gguf",
            "huggingface.co/example/model/model.gguf",
            "hf://example/model",
        )
        for key in ("encoder_gguf", "dit_gguf", "base_dir"):
            for remote in remotes:
                with self.subTest(key=key, remote=remote):
                    with self.assertRaises(ValueError):
                        cls(self.backend_config(**{key: remote}))
                    self.assert_no_loading()

    def test_missing_files_are_rejected_before_loading(self):
        cls = self.module().ming_design
        missing_vae_base = self.directory / "no_vae"
        missing_vae_base.mkdir()
        cases = {
            "encoder": {"encoder_gguf": str(self.directory / "missing_enc.gguf")},
            "dit": {"dit_gguf": str(self.directory / "missing_dit.gguf")},
            "vae": {"base_dir": str(missing_vae_base)},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(FileNotFoundError):
                    cls(self.backend_config(**overrides))
                self.assert_no_loading()

    def test_caller_config_is_not_mutated(self):
        config = self.backend_config(steps=9, code_dir=str(self.directory))
        snapshot = copy.deepcopy(config)
        self.module().ming_design(config)
        self.assertEqual(config, snapshot)

        bad = self.backend_config(dit_gguf=str(self.directory / "missing.gguf"))
        bad_snapshot = copy.deepcopy(bad)
        with self.assertRaises(FileNotFoundError):
            self.module().ming_design(bad)
        self.assertEqual(bad, bad_snapshot)

    # 3. Loading

    def test_constructor_loads_encoder_on_cpu_and_generator_on_cuda(self):
        backend = self.module().ming_design(self.backend_config())
        self.load_encoder.assert_called_once_with(
            str(self.encoder_gguf), str(self.base), code_dir=None, device="cpu"
        )
        self.load_generator.assert_called_once_with(
            str(self.dit_gguf),
            str(self.vae),
            str(self.base),
            code_dir=None,
            device="cuda",
        )
        self.assertIs(backend.encoder, self.encoder)
        self.assertIs(backend.generator, self.generator)

    def test_code_dir_is_forwarded_to_both_loaders(self):
        code_dir = str(self.directory / "code")
        self.module().ming_design(self.backend_config(code_dir=code_dir))
        self.assertEqual(self.load_encoder.call_args.kwargs["code_dir"], code_dir)
        self.assertEqual(self.load_generator.call_args.kwargs["code_dir"], code_dir)

    def test_relative_paths_are_resolved_to_absolute(self):
        relative = lambda path: os.path.relpath(path, os.getcwd())
        self.module().ming_design(
            {
                "encoder_gguf": relative(self.encoder_gguf),
                "dit_gguf": relative(self.dit_gguf),
                "base_dir": relative(self.base),
            }
        )
        encoder_args = self.load_encoder.call_args.args
        generator_args = self.load_generator.call_args.args
        self.assertEqual(encoder_args, (str(self.encoder_gguf), str(self.base)))
        self.assertEqual(
            generator_args, (str(self.dit_gguf), str(self.vae), str(self.base))
        )

    # 4. Reference image validation

    def test_prepare_reference_is_imported_into_backend_namespace(self):
        self.assertTrue(self.had_prepare_reference)

    def test_more_than_one_reference_is_rejected(self):
        second = self.directory / "second.png"
        Image.new("RGB", (8, 8)).save(second)
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, r"(?i)one reference image"):
            backend.execute(
                "a cat",
                self.request(),
                media=[str(self.reference), str(second)],
            )
        self.assert_no_model_calls()
        self.assertFalse(self.output.exists())

    def test_missing_or_non_file_reference_is_rejected(self):
        backend = self.backend()
        for path in (self.directory / "nope.png", self.directory):
            with self.subTest(path=path):
                with self.assertRaises(FileNotFoundError):
                    backend.execute("a cat", self.request(), media=[str(path)])
                self.assert_no_model_calls()
                self.assertFalse(self.output.exists())

    def test_text_to_image_does_not_use_reference_path(self):
        backend = self.backend()
        backend.execute("a cat", self.request(), media=[])
        self.prepare_reference.assert_not_called()
        self.assertEqual(self.encoder.images, [None])
        self.assertIsNone(self.sample_kwargs["ref_x"])
        self.plan_size.assert_called_once()

    # 4b. Image edit

    def test_edit_prepares_reference_from_opened_file(self):
        backend = self.backend(code_dir="/some/code")
        self.edit(backend)
        self.prepare_reference.assert_called_once()
        call = self.prepare_reference.call_args
        self.assertEqual(len(call.args), 1)
        self.assertEqual(call.args[0].size, (64, 48))
        self.assertEqual(
            call.kwargs, {"width": None, "height": None, "code_dir": "/some/code"}
        )

    def test_edit_forwards_requested_size_only_when_both_given(self):
        cases = (
            ({"width": 96, "height": 64}, (96, 64)),
            ({"width": 96}, (None, None)),
            ({"height": 64}, (None, None)),
            ({}, (None, None)),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                self.prepare_reference.reset_mock()
                backend = self.backend()
                self.edit(backend, **overrides)
                kwargs = self.prepare_reference.call_args.kwargs
                self.assertEqual((kwargs["width"], kwargs["height"]), expected)

    def test_edit_encodes_stripped_prompt_with_prepared_image(self):
        backend = self.backend()
        self.edit(backend, prompt="  make it blue \n")
        self.assertEqual(self.encoder.prompts, ["make it blue"])
        self.assertEqual(len(self.encoder.images), 1)
        self.assertIs(self.encoder.images[0], self.plan_image)

    def test_edit_samples_with_reference_latents_and_planned_size(self):
        backend = self.backend()
        self.edit(backend, seed=99, steps=7, guidance_scale=1.5)
        self.plan_size.assert_not_called()
        self.assertIs(self.sample_kwargs["ref_x"], self.ref_x)
        self.assertEqual(self.sample_kwargs["width"], 32)
        self.assertEqual(self.sample_kwargs["height"], 16)
        self.assertEqual(self.sample_kwargs["steps"], 7)
        self.assertEqual(self.sample_kwargs["cfg"], 1.5)
        self.assertEqual(self.sample_kwargs["seed"], 99)
        self.assertTrue(callable(self.sample_kwargs["callback"]))
        self.assertEqual(
            self.sample_args,
            (self.generator, self.encoder.context, self.encoder.directvlm),
        )
        self.assertEqual(self.sample_image.call_args.args[0], self.generator)

    def test_edit_output_defaults_to_planned_size(self):
        backend = self.backend()
        self.edit(backend)
        image, info = self.saved()
        self.assertEqual(image.size, (32, 16))
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0, 128))
        self.assertEqual((int(info["width"]), int(info["height"])), (32, 16))

    def test_edit_output_is_resized_to_requested_size(self):
        backend = self.backend()
        self.edit(backend, width=96, height=64)
        image, info = self.saved()
        self.assertEqual(image.size, (96, 64))
        self.assertEqual(image.mode, "RGBA")
        self.assertAlmostEqual(image.getpixel((10, 10))[3], 128, delta=2)
        self.assertEqual((int(info["width"]), int(info["height"])), (96, 64))

    def test_edit_metadata_marks_task_and_records_parameters(self):
        backend = self.backend()
        self.edit(backend, prompt="  make it blue ", seed=4242)
        info = self.saved()[1]
        self.assertEqual(info["task"], "image-edit")
        self.assertEqual(info["prompt"], "make it blue")
        self.assertEqual(info["seed"], "4242")
        self.assertEqual(info["model"], "ming_design")
        self.assertEqual(int(info["inference_steps"]), 12)
        self.assertEqual(float(info["guidance_scale"]), 1.0)

    def test_edit_returns_path_and_stripped_prompt(self):
        backend = self.backend()
        result = self.edit(backend, prompt="  make it blue ")
        self.assertEqual(result, (str(self.output), "make it blue"))
        self.assertEqual(backend.last_imagegen_prompt, "make it blue")

    def test_edit_uses_batch_id_output_path(self):
        backend = self.backend()
        with patch.object(backend, "_save_image_with_metadata"):
            path, _ = backend.execute(
                "x", self.edit_request(), batch_id=5, media=[str(self.reference)]
            )
        self.assertEqual(path, "/tmp/ircawp_generated/ming_design.5.png")

    def test_edit_default_steps_and_cfg(self):
        backend = self.backend()
        self.edit(backend)
        self.assertEqual(self.sample_kwargs["steps"], 12)
        self.assertEqual(self.sample_kwargs["cfg"], 1.0)

    def test_edit_encoder_is_staged_through_gpu_before_sampling(self):
        backend = self.backend()
        self.edit(backend)
        self.assertEqual(self.calls, ["to:cuda", "encode", "to:cpu", "sample"])

    def test_edit_encoder_returns_to_cpu_when_encode_fails(self):
        self.encoder.error = RuntimeError("encode boom")
        backend = self.backend()
        with self.assertRaisesRegex(RuntimeError, "encode boom"):
            self.edit(backend)
        self.assertEqual(self.calls, ["to:cuda", "encode", "to:cpu"])
        self.sample_image.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_edit_cancelled_before_start_skips_open_prepare_and_encode(self):
        backend = self.backend()
        event = Event()
        event.set()
        with patch("PIL.Image.open", wraps=Image.open) as opener:
            with self.assertRaises(GenerationCancelled):
                self.edit(backend, cancellation_event=event)
        opener.assert_not_called()
        self.assert_no_model_calls()
        self.assertFalse(self.output.exists())

    def test_edit_cancelled_during_sampling_propagates_and_writes_nothing(self):
        backend = self.backend()
        event = Event()

        def cancel_then_step(callback):
            event.set()
            callback(None, 0, None, {"latents": 1})

        self.on_sample = cancel_then_step
        with self.assertRaises(GenerationCancelled):
            self.edit(backend, cancellation_event=event)
        self.assertFalse(self.output.exists())

    def test_edit_step_callback_reports_progress(self):
        backend = self.backend()
        progress = {}

        def step(callback):
            callback(None, 0, None, {"latents": 1})

        self.on_sample = step
        self.edit(backend, steps=5, progress_state=progress)
        self.assertEqual(progress["step"], 1)
        self.assertEqual(progress["total_steps"], 5)

    def test_default_media_list_is_not_mutated(self):
        backend = self.backend()
        default = inspect.signature(backend.execute).parameters["media"].default
        backend.execute("a cat", self.request())
        self.assertEqual(default, [])
        self.assertEqual(
            inspect.signature(backend.execute).parameters["media"].default, []
        )

    # 5. Output path and return value

    def test_output_path_and_return_value(self):
        backend = self.backend()
        cases = (
            ({"seed": 1}, 3, "/tmp/ircawp_generated/ming_design.3.png"),
            ({"seed": 1, "output_file": str(self.output)}, None, str(self.output)),
            ({"seed": 1}, None, "/tmp/ircawp_generated/ming_design.png"),
        )
        for extra, batch_id, expected in cases:
            with self.subTest(expected=expected):
                config = {"width": 96, "height": 64, **extra}
                with patch.object(backend, "_save_image_with_metadata") as save:
                    result = backend.execute("  a cat  ", config, batch_id=batch_id)
                self.assertEqual(result, (expected, "a cat"))
                self.assertEqual(save.call_args.args[1], expected)
                self.assertEqual(backend.last_imagegen_prompt, "a cat")

    def test_batch_id_wins_over_output_file(self):
        backend = self.backend()
        with patch.object(backend, "_save_image_with_metadata"):
            path, _ = backend.execute("x", self.request(), batch_id=3)
        self.assertEqual(path, "/tmp/ircawp_generated/ming_design.3.png")

    def test_prompt_is_stripped_before_encoding_and_saved(self):
        backend = self.backend()
        backend.execute("  a cat \n", self.request())
        self.assertEqual(self.encoder.prompts, ["a cat"])
        self.assertEqual(self.saved()[1]["prompt"], "a cat")

    # 6. Seed

    def test_given_seed_reaches_sampler_and_metadata(self):
        backend = self.backend()
        backend.execute("a cat", self.request(seed=424242))
        self.assertEqual(self.sample_kwargs["seed"], 424242)
        self.assertEqual(self.saved()[1]["seed"], "424242")

    def test_random_seed_is_recorded_consistently(self):
        backend = self.backend()
        config = self.request()
        del config["seed"]
        backend.execute("a cat", config)
        seed = self.sample_kwargs["seed"]
        self.assertIsInstance(seed, int)
        self.assertTrue(0 <= seed < 1_000_000)
        self.assertEqual(self.saved()[1]["seed"], str(seed))

    def test_given_seed_leaves_global_torch_rng_untouched(self):
        backend = self.backend()
        before = torch.get_rng_state()
        backend.execute("a cat", self.request(seed=7))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    # 7. Sampling parameters

    def test_default_steps_and_cfg(self):
        backend = self.backend()
        backend.execute("a cat", self.request())
        self.assertEqual(self.sample_kwargs["steps"], 12)
        self.assertEqual(self.sample_kwargs["cfg"], 1.0)
        info = self.saved()[1]
        self.assertEqual(int(info["inference_steps"]), 12)
        self.assertEqual(float(info["guidance_scale"]), 1.0)

    def test_steps_precedence_request_then_backend_then_default(self):
        cases = (
            ({"steps": 20}, {"steps": 30}, 30),
            ({"steps": 20}, {}, 20),
            ({}, {}, 12),
        )
        for backend_overrides, request_overrides, expected in cases:
            with self.subTest(backend=backend_overrides, request=request_overrides):
                backend = self.backend(**backend_overrides)
                backend.execute("a cat", self.request(**request_overrides))
                self.assertEqual(self.sample_kwargs["steps"], expected)
                self.assertEqual(int(self.saved()[1]["inference_steps"]), expected)

    def test_guidance_scale_overrides_cfg(self):
        backend = self.backend()
        backend.execute("a cat", self.request(guidance_scale=2.5))
        self.assertEqual(self.sample_kwargs["cfg"], 2.5)
        self.assertEqual(float(self.saved()[1]["guidance_scale"]), 2.5)

    # 8. Sizes

    def test_explicit_size_is_used_as_requested(self):
        backend = self.backend()
        backend.execute("a cat", self.request(width=144, height=96))
        self.assertEqual(self.plan_size.call_args.args[:2], (144, 96))
        image, info = self.saved()
        self.assertEqual(image.size, (144, 96))
        self.assertEqual((int(info["width"]), int(info["height"])), (144, 96))

    def test_derived_size_defaults_to_1024_by_688(self):
        backend = self.backend()
        config = {"output_file": str(self.output), "seed": 1}
        backend.execute("a cat", config)
        self.assertEqual(self.plan_size.call_args.args[:2], (1024, 688))
        self.assertEqual(self.saved()[0].size, (1024, 688))

    def test_derived_size_from_aspect_and_max_output_size(self):
        cases = (
            ({"aspect": "3:2"}, {}, (1024, 688)),
            ({"aspect": 1.0}, {}, (1024, 1024)),
            ({"aspect": "16:9"}, {}, (1024, 576)),
            ({"aspect": 0.5}, {}, (512, 1024)),
            ({}, {"max_output_size": 768}, (768, 512)),
            ({"max_output_size": 512}, {"max_output_size": 768}, (512, 336)),
        )
        for request_overrides, backend_overrides, expected in cases:
            with self.subTest(request=request_overrides, backend=backend_overrides):
                backend = self.backend(**backend_overrides)
                config = {
                    "output_file": str(self.output),
                    "seed": 1,
                    **request_overrides,
                }
                with patch.object(backend, "_save_image_with_metadata"):
                    backend.execute("a cat", config)
                self.assertEqual(self.plan_size.call_args.args[:2], expected)

    def test_sampler_gets_internal_size_and_output_is_resized(self):
        self.plan_size.side_effect = lambda width, height, code_dir=None: (112, 80)
        backend = self.backend()
        backend.execute("a cat", self.request(width=96, height=64))
        self.assertEqual(self.plan_size.call_args.args[:2], (96, 64))
        self.assertEqual(self.sample_kwargs["width"], 112)
        self.assertEqual(self.sample_kwargs["height"], 80)
        image, info = self.saved()
        self.assertEqual(image.size, (96, 64))
        self.assertEqual(image.mode, "RGBA")
        self.assertAlmostEqual(image.getpixel((10, 10))[3], 128, delta=2)
        self.assertEqual((int(info["width"]), int(info["height"])), (96, 64))

    def test_png_keeps_rgba_and_metadata(self):
        backend = self.backend()
        backend.execute("a cat", self.request())
        image, info = self.saved()
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0, 128))
        self.assertEqual(info["model"], "ming_design")
        self.assertEqual((int(info["width"]), int(info["height"])), (96, 64))

    # 9. Staged memory use

    def test_encoder_is_staged_through_gpu_before_sampling(self):
        backend = self.backend()
        backend.execute("a cat", self.request())
        self.assertEqual(self.calls, ["to:cuda", "encode", "to:cpu", "sample"])
        self.assertEqual(
            self.sample_args,
            (self.generator, self.encoder.context, self.encoder.directvlm),
        )

    def test_encoder_returns_to_cpu_when_encode_fails(self):
        self.encoder.error = RuntimeError("encode boom")
        backend = self.backend()
        with self.assertRaisesRegex(RuntimeError, "encode boom"):
            backend.execute("a cat", self.request())
        self.assertEqual(self.calls, ["to:cuda", "encode", "to:cpu"])
        self.sample_image.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cuda_cache_is_released_after_sampling_even_when_it_fails(self):
        def raise_runtime(callback):
            raise RuntimeError("sample boom")

        def raise_cancelled(callback):
            raise GenerationCancelled()

        cases = (
            ("text-to-image", False, None, None),
            ("edit", True, None, None),
            ("text-to-image runtime error", False, raise_runtime, RuntimeError),
            ("edit runtime error", True, raise_runtime, RuntimeError),
            ("text-to-image cancelled", False, raise_cancelled, GenerationCancelled),
            ("edit cancelled", True, raise_cancelled, GenerationCancelled),
        )
        for name, is_edit, on_sample, error in cases:
            with self.subTest(case=name):
                self.on_sample = on_sample
                empty_cache = Mock(side_effect=lambda: self.calls.append("empty_cache"))
                with (
                    patch(
                        "app.backends.ming_design.torch.cuda.is_available",
                        return_value=True,
                    ),
                    patch(
                        "app.backends.ming_design.torch.cuda.empty_cache", empty_cache
                    ),
                ):
                    backend = self.backend()
                    run = (
                        (lambda: self.edit(backend))
                        if is_edit
                        else (lambda: backend.execute("a cat", self.request()))
                    )
                    if error is None:
                        run()
                    else:
                        with self.assertRaises(error):
                            run()
                self.assertIn("sample", self.calls)
                after_sample = self.calls[self.calls.index("sample") + 1 :]
                self.assertIn("empty_cache", after_sample)

    # 10. Cancellation

    def test_cancelled_before_start_skips_all_model_work(self):
        backend = self.backend()
        event = Event()
        event.set()
        with self.assertRaises(GenerationCancelled):
            backend.execute("a cat", self.request(cancellation_event=event))
        self.assert_no_model_calls()
        self.assertFalse(self.output.exists())

    def test_cancelled_during_sampling_propagates_and_writes_nothing(self):
        backend = self.backend()
        event = Event()

        def cancel_then_step(callback):
            event.set()
            callback(None, 0, None, {"latents": 1})

        self.on_sample = cancel_then_step
        with self.assertRaises(GenerationCancelled):
            backend.execute("a cat", self.request(cancellation_event=event))
        self.assertFalse(self.output.exists())

    def test_step_callback_reports_progress(self):
        backend = self.backend()
        progress = {}
        returned = []

        def step(callback):
            returned.append(callback(None, 0, None, {"latents": 1}))

        self.on_sample = step
        backend.execute("a cat", self.request(steps=5, progress_state=progress))
        self.assertEqual(progress["step"], 1)
        self.assertEqual(progress["total_steps"], 5)
        self.assertEqual(returned, [{"latents": 1}])

    # 11. Disposal

    def test_dispose_releases_models_and_is_idempotent(self):
        backend = self.backend()
        backend.pipe = object()
        backend.dispose()
        self.assertIsNone(backend.encoder)
        self.assertIsNone(backend.generator)
        self.assertIsNone(backend.pipe)
        backend.dispose()
        self.assertIsNone(backend.encoder)
        self.assertIsNone(backend.generator)


# 12. Registration


class MingDesignRegistrationTests(unittest.TestCase):
    def test_capabilities_are_advertised(self):
        from app.main import BACKEND_CAPABILITIES

        self.assertEqual(
            BACKEND_CAPABILITIES.get("ming_design"),
            {"cfg": "guidance_scale", "cfg_experimental": True, "seed": True},
        )

    def test_example_config_has_ming_design_entry(self):
        config = yaml.safe_load((PROJECT_ROOT / "config.example.yml").read_text())
        entry = config["backends"].get("ming_design")
        self.assertIsNotNone(entry)
        self.assertEqual(
            set(entry),
            {"max_output_size", "steps", "encoder_gguf", "dit_gguf", "base_dir"},
        )
        self.assertEqual(entry["max_output_size"], 1024)
        self.assertEqual(entry["steps"], 12)
        for key in ("encoder_gguf", "dit_gguf", "base_dir"):
            with self.subTest(key=key):
                self.assertIsInstance(entry[key], str)
                self.assertTrue(entry[key].startswith("models/ming-image/"))


if __name__ == "__main__":
    unittest.main()

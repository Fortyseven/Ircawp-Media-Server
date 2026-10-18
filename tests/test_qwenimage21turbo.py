"""GPU-free integration contract for the Qwen-Image-2.1 Turbo backend."""

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app.backends.MediaBackend import GenerationCancelled
from app.backends.qwenimage21 import qwenimage21


TURBO_MODEL = "Qwen/Qwen-Image-2.1-Turbo"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TURBO_GGUF = PROJECT_ROOT / "models" / "qwen_image_2.1_turbo_Q4_K_S.gguf"
SAMPLE_SIGMAS = [
    1.0,
    0.978453,
    0.95418,
    0.926626,
    0.89508,
    0.845148,
    0.704534,
    0.414568,
]


class QwenImage21TurboIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.gguf = self.directory / "transformer.gguf"
        self.gguf.write_bytes(b"mocked transformer loader fixture")
        self.output = self.directory / "result.png"
        self.pipeline = Mock()
        # Deliberately no sigmas on the constructed pipeline's config.
        self.pipeline.config = {}
        self.pipeline.side_effect = self.render
        self.decode_calls = []
        self.enterContext(
            patch.object(qwenimage21, "_decode_latents", autospec=True)
        ).side_effect = lambda _self, *args: self.decode(*args)
        self.transformer_loader = self.enterContext(
            patch("diffusers.QwenImage21Transformer2DModel.from_single_file")
        )
        self.pipeline_loader = self.enterContext(
            patch(
                "diffusers.QwenImage21Pipeline.from_pretrained",
                return_value=self.pipeline,
            )
        )
        self.config_loader = self.enterContext(
            patch(
                "diffusers.QwenImage21Pipeline.load_config",
                return_value={"sample_sigmas": list(SAMPLE_SIGMAS)},
            )
        )

    def render(self, **kwargs):
        callback = kwargs["callback_on_step_end"]
        for step_index in range(kwargs["num_inference_steps"]):
            callback_kwargs = {"latents": object()}
            self.assertIs(
                callback(self.pipeline, step_index, step_index, callback_kwargs),
                callback_kwargs,
            )
        return SimpleNamespace(
            images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "teal")]
        )

    def decode(self, latents, width, height):
        self.decode_calls.append((width, height))
        return latents[0]

    def turbo_class(self):
        module = importlib.import_module("app.backends.qwenimage21turbo")
        return getattr(module, "qwenimage21turbo")

    def turbo(self, **overrides):
        return self.turbo_class()({"gguf": str(self.gguf), **overrides})

    def request(self, **overrides):
        return {
            "output_file": str(self.output),
            "width": 96,
            "height": 64,
            "seed": 123456,
            **overrides,
        }

    def assert_no_loading(self):
        self.transformer_loader.assert_not_called()
        self.pipeline_loader.assert_not_called()
        self.config_loader.assert_not_called()

    def assert_turbo_result(self, backend, config, media=None, sigmas=SAMPLE_SIGMAS):
        original_config = dict(config)
        output_path, prompt = backend.execute(
            "  a precise illustration  ", config, media=media or []
        )
        self.assertEqual(config, original_config)
        self.assertEqual(output_path, str(self.output))
        self.assertEqual(backend.last_imagegen_prompt, prompt.strip())
        kwargs = self.pipeline.call_args.kwargs
        steps = config.get("steps") or 16
        if steps != len(sigmas):
            sigmas = np.interp(
                np.linspace(0, 1, steps), np.linspace(0, 1, len(sigmas)), sigmas
            ).tolist()
        self.assertEqual(list(kwargs["sigmas"]), sigmas)
        self.assertEqual(len(kwargs["sigmas"]), steps)
        self.assertEqual(kwargs["num_inference_steps"], steps)
        self.assertIs(kwargs["use_kv_cache"], True)
        self.assertEqual(kwargs["true_cfg_scale"], config.get("true_cfg_scale", 1.0))
        self.assertEqual((kwargs["width"], kwargs["height"]), (96, 64))
        self.assertEqual(kwargs["generator"].initial_seed(), config["seed"])
        self.assertEqual(kwargs["output_type"], "latent")
        self.assertEqual(self.decode_calls[-1], (96, 64))
        with Image.open(output_path) as saved:
            self.assertEqual(saved.size, (96, 64))
            self.assertEqual(saved.info["model"], "qwenimage21turbo")
            self.assertEqual(saved.info["inference_steps"], str(steps))
            self.assertEqual(saved.info["seed"], str(config["seed"]))
            self.assertEqual(saved.info["width"], "96")
            self.assertEqual(saved.info["height"], "64")
            self.assertEqual(saved.info["prompt"], prompt)
        return kwargs, prompt

    def test_dynamic_import_exports_backend_subclass(self):
        self.assertTrue(issubclass(self.turbo_class(), qwenimage21))

    def test_default_transformer_is_absolute_project_local_file(self):
        backend_class = self.turbo_class()
        self.assertTrue(DEFAULT_TURBO_GGUF.is_absolute())
        if DEFAULT_TURBO_GGUF.is_file():
            backend_class({})
            loaded_path = Path(self.transformer_loader.call_args.args[0])
            self.assertTrue(loaded_path.is_absolute())
            self.assertEqual(loaded_path, DEFAULT_TURBO_GGUF)
        else:
            with self.assertRaises(FileNotFoundError) as failure:
                backend_class({})
            self.assertIn(str(DEFAULT_TURBO_GGUF), str(failure.exception))
            self.assert_no_loading()

    def test_missing_override_fails_before_any_model_or_config_loading(self):
        backend_class = self.turbo_class()
        missing = str(self.directory / "missing.gguf")
        with self.assertRaises(FileNotFoundError) as failure:
            backend_class({"gguf": missing})
        self.assertIn(missing, str(failure.exception))
        self.assert_no_loading()

    def test_directory_is_not_a_transformer_file(self):
        backend_class = self.turbo_class()
        with self.assertRaises((FileNotFoundError, ValueError)):
            backend_class({"gguf": str(self.directory)})
        self.assert_no_loading()

    def test_remote_transformers_are_rejected_before_loading(self):
        backend_class = self.turbo_class()
        for remote in (
            "https://huggingface.co/example/model/resolve/main/model.gguf",
            "http://example.com/model.gguf",
            "hf.co/example/model/model.gguf",
            "huggingface.co/example/model/model.gguf",
            "hf://example/model/model.gguf",
        ):
            with self.subTest(remote=remote):
                with self.assertRaisesRegex(ValueError, r"(?i)local|remote|url"):
                    backend_class({"gguf": remote})
                self.assert_no_loading()

    def test_components_and_schedule_use_turbo_repo_and_inherited_4bit_encoder(self):
        self.turbo()
        self.assertEqual(self.transformer_loader.call_args.args[0], str(self.gguf))
        self.assertEqual(
            self.transformer_loader.call_args.kwargs["config"], TURBO_MODEL
        )
        self.assertEqual(
            self.transformer_loader.call_args.kwargs["subfolder"], "transformer"
        )
        self.assertEqual(self.pipeline_loader.call_args.args[0], TURBO_MODEL)
        self.assertIs(
            self.pipeline_loader.call_args.kwargs["transformer"],
            self.transformer_loader.return_value,
        )
        quantization = self.pipeline_loader.call_args.kwargs["quantization_config"]
        encoder_config = quantization.quant_mapping["text_encoder"]
        self.assertTrue(encoder_config.load_in_4bit)
        self.assertTrue(self.config_loader.called)
        self.assertEqual(self.config_loader.call_args.args[0], TURBO_MODEL)
        self.pipeline.enable_model_cpu_offload.assert_called_once_with()

    def test_local_base_model_override_supplies_components_and_schedule(self):
        base_model = self.directory / "local-base-model"
        base_model.mkdir()
        self.turbo(base_model=str(base_model))
        self.assertEqual(
            self.transformer_loader.call_args.kwargs["config"], str(base_model)
        )
        self.assertEqual(self.pipeline_loader.call_args.args[0], str(base_model))
        self.assertEqual(self.config_loader.call_args.args[0], str(base_model))

    def test_loaded_schedule_is_retained_and_reaches_actual_pipeline_call(self):
        local_sigmas = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2]
        self.config_loader.return_value = {"sample_sigmas": list(local_sigmas)}
        backend = self.turbo()
        self.config_loader.return_value = {}
        self.config_loader.reset_mock()
        self.assert_turbo_result(backend, self.request(), sigmas=local_sigmas)
        self.config_loader.assert_not_called()

    def test_generation_saves_default_step_metadata_and_reports_default_steps(self):
        progress = {}
        kwargs, prompt = self.assert_turbo_result(
            self.turbo(), self.request(progress_state=progress)
        )
        self.assertIsNone(kwargs["image"])
        self.assertEqual(prompt, "a precise illustration")
        self.assertEqual(progress, {"step": 16, "total_steps": 16})

    def test_caller_steps_resample_schedule_and_remaster_adds_no_steps(self):
        backend = self.turbo()
        for overrides, expected_steps in (
            ({"steps": 40}, 40),
            ({"steps": 8}, 8),
            ({"steps": 1}, 1),
            ({"remaster": True}, 16),
            ({"steps": 12, "remaster": True}, 12),
        ):
            with self.subTest(overrides=overrides):
                progress = {}
                _, prompt = self.assert_turbo_result(
                    backend, self.request(progress_state=progress, **overrides)
                )
                self.assertEqual(
                    progress, {"step": expected_steps, "total_steps": expected_steps}
                )
                if overrides.get("remaster"):
                    self.assertIn("Enhance this image", prompt)

    def test_explicit_cfg_overrides_turbo_default(self):
        kwargs, _ = self.assert_turbo_result(
            self.turbo(), self.request(true_cfg_scale=3.5)
        )
        self.assertEqual(kwargs["true_cfg_scale"], 3.5)
        self.assertEqual(kwargs["negative_prompt"], "")

    def test_multiple_edit_images_are_loaded_by_real_pil_and_keep_explicit_size(self):
        media = []
        for index, (size, color) in enumerate((((32, 48), "red"), ((48, 32), "blue"))):
            path = self.directory / f"input-{index}.png"
            Image.new("RGB", size, color).save(path)
            media.append(str(path))
        kwargs, _ = self.assert_turbo_result(
            self.turbo(), self.request(steps=40, remaster=True), media=media
        )
        self.assertEqual(len(kwargs["image"]), 2)
        for loaded, size, pixel in zip(
            kwargs["image"], ((32, 48), (48, 32)), ((255, 0, 0), (0, 0, 255))
        ):
            self.assertIsInstance(loaded, Image.Image)
            self.assertEqual(loaded.size, size)
            self.assertEqual(loaded.getpixel((0, 0)), pixel)

    def test_cancellation_stops_generation_before_png_is_saved(self):
        backend = self.turbo()
        cancellation = Event()
        progress = {}

        def cancel_during_render(**kwargs):
            callback = kwargs["callback_on_step_end"]
            callback(self.pipeline, 0, 0, {})
            cancellation.set()
            callback(self.pipeline, 1, 1, {})
            self.fail("Cancellation callback did not abort generation")

        self.pipeline.side_effect = cancel_during_render
        with self.assertRaises(GenerationCancelled):
            backend.execute(
                "cancel me",
                self.request(
                    steps=40,
                    remaster=True,
                    cancellation_event=cancellation,
                    progress_state=progress,
                ),
            )
        self.assertEqual(progress, {"step": 1, "total_steps": 40})
        self.assertFalse(self.output.exists())

    def test_missing_or_invalid_schedule_fails_clearly_without_sampling(self):
        backend_class = self.turbo_class()
        invalid_configs = (
            {},
            {"sample_sigmas": None},
            {"sample_sigmas": []},
            {"sample_sigmas": SAMPLE_SIGMAS[:-1]},
            {"sample_sigmas": SAMPLE_SIGMAS + [0.1]},
            {"sample_sigmas": "12345678"},
            {"sample_sigmas": ["1.0", *SAMPLE_SIGMAS[1:]]},
            {"sample_sigmas": [True, *SAMPLE_SIGMAS[1:]]},
            {"sample_sigmas": [1.01, *SAMPLE_SIGMAS[1:]]},
            {"sample_sigmas": [*SAMPLE_SIGMAS[:-1], 0.0]},
            {"sample_sigmas": [*SAMPLE_SIGMAS[:-1], -0.1]},
            {"sample_sigmas": [1.0, 1.0, *SAMPLE_SIGMAS[2:]]},
            {"sample_sigmas": list(reversed(SAMPLE_SIGMAS))},
            {"sample_sigmas": [float("nan"), *SAMPLE_SIGMAS[1:]]},
            {"sample_sigmas": [float("inf"), *SAMPLE_SIGMAS[1:]]},
            {"sample_sigmas": [*SAMPLE_SIGMAS[:-1], float("-inf")]},
        )
        for model_config in invalid_configs:
            with self.subTest(model_config=model_config):
                self.config_loader.return_value = model_config
                with self.assertRaisesRegex(ValueError, r"(?i)sigma|schedule"):
                    backend = backend_class({"gguf": str(self.gguf)})
                    backend.execute("must not sample", self.request())
                self.pipeline.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_base_backend_keeps_default_and_remaster_steps_and_model_metadata(self):
        backend = qwenimage21({"gguf": str(self.gguf)})
        self.assertEqual(self.pipeline_loader.call_args.args[0], "Qwen/Qwen-Image-2.1")
        for remaster, expected_steps in ((False, 40), (True, 44)):
            with self.subTest(remaster=remaster):
                progress = {}
                backend.execute(
                    "base image",
                    self.request(remaster=remaster, progress_state=progress),
                )
                kwargs = self.pipeline.call_args.kwargs
                self.assertEqual(kwargs["num_inference_steps"], expected_steps)
                self.assertNotIn("sigmas", kwargs)
                self.assertNotIn("use_kv_cache", kwargs)
                self.assertNotIn("true_cfg_scale", kwargs)
                self.assertEqual(
                    progress, {"step": expected_steps, "total_steps": expected_steps}
                )
                with Image.open(self.output) as saved:
                    self.assertEqual(saved.info["model"], "qwenimage21")
                    self.assertEqual(saved.info["inference_steps"], str(expected_steps))
        self.config_loader.assert_not_called()

    def test_base_missing_transformer_keeps_existing_remote_loading_behavior(self):
        missing = str(self.directory / "missing-base.gguf")
        self.assertFalse(Path(missing).exists())
        qwenimage21({"gguf": missing})
        self.assertTrue(
            self.transformer_loader.call_args.args[0].startswith(
                "https://huggingface.co/"
            )
        )


if __name__ == "__main__":
    unittest.main()

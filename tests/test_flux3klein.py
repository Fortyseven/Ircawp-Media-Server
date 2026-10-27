"""GPU-free integration contract for the local GGUF FLUX.3 Klein backend."""

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app.backends.MediaBackend import GenerationCancelled, MediaBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GGUF = PROJECT_ROOT / "models" / "flux3_image_klein_sd_Q5_K_M.gguf"
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.3-Klein"


class Flux3KleinIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.gguf = self.directory / "transformer.gguf"
        self.gguf.write_bytes(b"mocked FLUX.3 transformer")
        self.output = self.directory / "result.png"
        self.pipeline = Mock()
        self.pipeline.side_effect = self.render
        self.transformer_loader = self.enterContext(
            patch("diffusers.Flux3Transformer2DModel.from_single_file")
        )
        self.pipeline_loader = self.enterContext(
            patch(
                "diffusers.modular_pipelines.flux3.modular_blocks_flux3_distilled"
                ".Flux3DistilledAutoBlocks.init_pipeline",
                return_value=self.pipeline,
            )
        )

    def render(self, **kwargs):
        self.assertEqual(kwargs["output"], "images")
        return [Image.new("RGB", (kwargs["width"], kwargs["height"]), "teal")]

    def backend_class(self):
        module = importlib.import_module("app.backends.flux3klein")
        return getattr(module, "flux3klein")

    def backend(self, **overrides):
        return self.backend_class()({"gguf": str(self.gguf), **overrides})

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

    def test_dynamic_import_exports_media_backend_subclass(self):
        self.assertTrue(issubclass(self.backend_class(), MediaBackend))

    def test_default_gguf_is_project_local_and_missing_default_stops_before_loading(
        self,
    ):
        backend_class = self.backend_class()
        self.assertTrue(DEFAULT_GGUF.is_absolute())
        if DEFAULT_GGUF.is_file():
            backend_class({})
            loaded_path = Path(self.transformer_loader.call_args.args[0])
            self.assertEqual(loaded_path, DEFAULT_GGUF)
        else:
            with self.assertRaises(FileNotFoundError) as failure:
                backend_class({})
            self.assertIn(str(DEFAULT_GGUF), str(failure.exception))
            self.assert_no_loading()

    def test_remote_missing_and_directory_ggufs_fail_before_expensive_loading(self):
        backend_class = self.backend_class()
        invalid_paths = (
            "https://huggingface.co/example/model/resolve/main/model.gguf",
            "http://example.com/model.gguf",
            "hf.co/example/model/model.gguf",
            "huggingface.co/example/model/model.gguf",
            "hf://example/model/model.gguf",
            str(self.directory / "missing.gguf"),
            str(self.directory),
        )
        for gguf in invalid_paths:
            with self.subTest(gguf=gguf):
                with self.assertRaises((FileNotFoundError, ValueError)):
                    backend_class({"gguf": gguf})
                self.assert_no_loading()

    def test_local_gguf_and_configurable_base_model_supply_distilled_pipeline_components(
        self,
    ):
        base_model = self.directory / "local-flux3-base"
        base_model.mkdir()
        constructor_config = {"gguf": str(self.gguf), "base_model": str(base_model)}
        original_config = dict(constructor_config)

        self.backend_class()(constructor_config)

        self.assertEqual(constructor_config, original_config)
        self.assertEqual(self.transformer_loader.call_args.args[0], str(self.gguf))
        self.assertEqual(
            self.transformer_loader.call_args.kwargs["config"], str(base_model)
        )
        self.assertEqual(
            self.transformer_loader.call_args.kwargs["subfolder"], "transformer"
        )
        self.assertEqual(self.pipeline_loader.call_args.args[0], str(base_model))
        self.pipeline.update_components.assert_called_once_with(
            transformer=self.transformer_loader.return_value
        )
        self.pipeline.load_components.assert_called_once_with(dtype=unittest.mock.ANY)

    def test_generation_uses_requested_steps_without_cfg_and_preserves_request_config(
        self,
    ):
        backend = self.backend()
        config = self.request(steps=2, guidance_scale=9.0)
        original_config = dict(config)
        output_path, prompt = backend.execute("  a precise illustration  ", config)

        self.assertEqual(config, original_config)
        self.assertEqual(output_path, str(self.output))
        self.assertEqual(prompt, "a precise illustration")
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["num_inference_steps"], 2)
        self.assertNotIn("guidance_scale", kwargs)
        self.assertEqual((kwargs["width"], kwargs["height"]), (96, 64))
        self.assertEqual(kwargs["generator"].device.type, "cpu")
        self.assertEqual(kwargs["generator"].initial_seed(), 123456)
        self.assertIsNone(kwargs["image"])
        with Image.open(output_path) as saved:
            self.assertEqual(saved.size, (96, 64))
            self.assertEqual(saved.info["model"], "flux3klein")
            self.assertEqual(saved.info["seed"], "123456")
            self.assertEqual(saved.info["inference_steps"], "2")
            self.assertEqual(saved.info["width"], "96")
            self.assertEqual(saved.info["height"], "64")
            self.assertEqual(saved.info["prompt"], prompt)

    def test_aspect_ratio_sets_dimensions_without_explicit_size(self):
        backend = self.backend()
        backend.execute(
            "wide scene",
            {
                "output_file": str(self.output),
                "max_output_size": 1024,
                "aspect": "16:9",
                "seed": 123456,
            },
        )
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (1024, 576))

    def test_progress_bar_reports_each_denoising_step_and_cancellation(self):
        module = importlib.import_module("app.backends.flux3klein")
        progress = {}
        cancellation = Event()
        bar = module._Flux3ProgressBar(
            backend=self.backend(),
            config={"progress_state": progress, "cancellation_event": cancellation},
            total=16,
        )

        with bar:
            bar.update()
            bar.update(3)
        self.assertEqual(progress, {"step": 4, "total_steps": 16})

        cancellation.set()
        with self.assertRaises(GenerationCancelled):
            bar.update()

    def test_progress_hook_skips_denoise_leaves_without_progress_bar(self):
        module = importlib.import_module("app.backends.flux3klein")

        class Loop:
            sub_blocks = {}

            def progress_bar(self, iterable=None, total=None):
                return object()

        class DenoiseLeaf:
            sub_blocks = {}

        Loop.__module__ = "diffusers.modular_pipelines.flux3.denoise"
        DenoiseLeaf.__module__ = "diffusers.modular_pipelines.flux3.denoise"
        loop = Loop()
        leaf = DenoiseLeaf()
        root = SimpleNamespace(sub_blocks={"loop": loop, "leaf": leaf})
        backend = self.backend()
        backend.pipe._blocks = root

        with backend._report_denoising_progress({"progress_state": {}}):
            self.assertIsInstance(loop.progress_bar(total=16), module._Flux3ProgressBar)
            self.assertFalse(hasattr(leaf, "progress_bar"))

    def test_multiple_edit_images_are_supported(self):
        media = []
        for index, (size, color) in enumerate((((32, 48), "red"), ((48, 32), "blue"))):
            path = self.directory / f"input-{index}.png"
            Image.new("RGB", size, color).save(path)
            media.append(str(path))

        progress = {}
        self.backend().execute(
            "edit both",
            self.request(progress_state=progress),
            media=media,
        )

        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (96, 64))
        self.assertEqual(len(kwargs["image"]), 2)
        self.assertEqual(
            [image.size for image in kwargs["image"]], [(32, 48), (48, 32)]
        )
        self.assertEqual(progress, {"step": 16, "total_steps": 16})

    def test_edit_without_explicit_size_matches_first_source_aspect(self):
        source = self.directory / "portrait.png"
        Image.new("RGB", (32, 48), "red").save(source)

        self.backend().execute(
            "match source",
            {
                "output_file": str(self.output),
                "max_output_size": 1024,
                "seed": 123456,
            },
            media=[str(source)],
        )

        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (688, 1024))

    def test_pre_cancelled_request_does_not_start_generation(self):
        cancellation = Event()
        cancellation.set()
        with self.assertRaises(GenerationCancelled):
            self.backend().execute(
                "cancel me", self.request(cancellation_event=cancellation)
            )
        self.pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()

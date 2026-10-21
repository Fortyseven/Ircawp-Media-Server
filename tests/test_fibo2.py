"""GPU-free integration contract for the Fibo 2.0 backends."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app import main
from app.backends.MediaBackend import GenerationCancelled
from app.backends.fibo2 import NEGATIVE_T2I, fibo2
from app.backends.fibo2editturbo import fibo2editturbo


class Fibo2Base(unittest.TestCase):
    backend_class = fibo2

    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.output = self.directory / "result.png"
        self.pipeline = Mock()
        self.pipeline.side_effect = self.render
        self.loader = self.enterContext(
            patch(
                "diffusers.BriaFibo2Pipeline.from_pretrained",
                return_value=self.pipeline,
            )
        )

    def render(self, **kwargs):
        callback = kwargs["callback_on_step_end"]
        for step_index in range(kwargs["num_inference_steps"]):
            callback(self.pipeline, step_index, step_index, {"latents": object()})
        return SimpleNamespace(
            images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "teal")]
        )

    def backend(self, **overrides):
        return self.backend_class(overrides)

    def request(self, **overrides):
        return {
            "output_file": str(self.output),
            "width": 96,
            "height": 64,
            "seed": 123,
            **overrides,
        }

    def source(self, name="source.png", size=(200, 100)):
        path = self.directory / name
        Image.new("RGB", size, "orange").save(path)
        return str(path)


class Fibo2Tests(Fibo2Base):
    def test_loads_official_repo_quantized_with_offload(self):
        self.backend()
        args, kwargs = self.loader.call_args
        self.assertEqual(args, ("briaai/Fibo-2.0-dev",))
        quant = kwargs["quantization_config"].quant_mapping
        self.assertEqual(set(quant), {"transformer"})
        self.assertTrue(quant["transformer"].load_in_4bit)
        self.pipeline.enable_model_cpu_offload.assert_called_once_with()

    def test_text_encoder_quantization_is_opt_in(self):
        self.backend(text_encoder_4bit=True)
        quant = self.loader.call_args.kwargs["quantization_config"].quant_mapping
        self.assertEqual(set(quant), {"transformer", "text_encoder"})

    def test_quantization_can_be_disabled(self):
        self.backend(transformer_4bit=False)
        self.assertNotIn("quantization_config", self.loader.call_args.kwargs)

    def test_base_model_is_configurable(self):
        self.backend(base_model="/some/local/dir")
        self.assertEqual(self.loader.call_args.args, ("/some/local/dir",))

    def test_constructor_does_not_mutate_config(self):
        config = {"steps": 5}
        fibo2(config)
        self.assertEqual(config, {"steps": 5})

    def test_text_to_image_forwards_sampling_arguments(self):
        backend = self.backend()
        config = self.request()
        original = dict(config)
        path, prompt = backend.execute("  a lighthouse at dusk  ", config)
        self.assertEqual(config, original)
        self.assertEqual((path, prompt), (str(self.output), "a lighthouse at dusk"))
        self.assertEqual(backend.last_imagegen_prompt, "a lighthouse at dusk")
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["prompt"], "a lighthouse at dusk")
        self.assertEqual((kwargs["width"], kwargs["height"]), (96, 64))
        self.assertEqual(kwargs["num_inference_steps"], 30)
        self.assertEqual(kwargs["guidance_scale"], 5.0)
        self.assertEqual(kwargs["negative_prompt"], NEGATIVE_T2I)
        self.assertEqual(kwargs["generator"].initial_seed(), 123)
        self.assertNotIn("image", kwargs)
        with Image.open(path) as saved:
            self.assertEqual(saved.size, (96, 64))
            self.assertEqual(saved.info["model"], "fibo2")
            self.assertEqual(saved.info["seed"], "123")
            self.assertEqual(saved.info["inference_steps"], "30")
            self.assertEqual(saved.info["guidance_scale"], "5.0")
            self.assertEqual(saved.info["prompt"], "a lighthouse at dusk")

    def test_step_and_guidance_overrides(self):
        backend = self.backend()
        backend.execute("x", self.request(steps=12, guidance_scale=3.5))
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["num_inference_steps"], 12)
        self.assertEqual(kwargs["guidance_scale"], 3.5)

    def test_guidance_one_omits_negative_prompt(self):
        self.backend().execute("x", self.request(guidance_scale=1.0))
        self.assertNotIn("negative_prompt", self.pipeline.call_args.kwargs)

    def test_dimensions_follow_aspect_and_max_output_size(self):
        config = {"output_file": str(self.output), "seed": 1}
        backend = self.backend()
        backend.execute("x", {**config, "aspect": "3:2", "max_output_size": 1024})
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (1024, 688))
        backend.execute("x", {**config, "aspect": "9:16", "max_output_size": 512})
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (288, 512))

    def test_missing_seed_is_generated_and_recorded(self):
        config = self.request()
        del config["seed"]
        path, _ = self.backend().execute("x", config)
        seed = self.pipeline.call_args.kwargs["generator"].initial_seed()
        with Image.open(path) as saved:
            self.assertEqual(saved.info["seed"], str(seed))

    def test_edit_passes_all_images_and_skips_negative_prompt(self):
        media = [self.source("a.png"), self.source("b.png")]
        config = {"output_file": str(self.output), "seed": 7, "max_output_size": 512}
        self.backend().execute("dress <image_1> in <image_2>", config, media=media)
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(len(kwargs["image"]), 2)
        self.assertTrue(all(isinstance(i, Image.Image) for i in kwargs["image"]))
        self.assertEqual((kwargs["width"], kwargs["height"]), (512, 256))
        self.assertNotIn("negative_prompt", kwargs)
        self.assertNotIn("mask", kwargs)

    def test_edit_honors_explicit_size(self):
        self.backend().execute("x", self.request(), media=[self.source()])
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (96, 64))

    def test_too_many_images_rejected_before_inference(self):
        media = [self.source(f"{i}.png") for i in range(6)]
        with self.assertRaises(ValueError):
            self.backend().execute("x", self.request(), media=media)
        self.pipeline.assert_not_called()

    def test_media_default_is_not_mutated(self):
        backend = self.backend()
        backend.execute("x", self.request())
        backend.execute("x", self.request(), media=[self.source()])
        backend.execute("x", self.request())
        self.assertNotIn("image", self.pipeline.call_args.kwargs)

    def test_cancellation_propagates_and_writes_nothing(self):
        event = Event()
        event.set()
        with self.assertRaises(GenerationCancelled):
            self.backend().execute("x", self.request(cancellation_event=event))
        self.assertFalse(self.output.exists())

    def test_progress_reports_actual_steps(self):
        progress = {}
        self.backend().execute("x", self.request(steps=6, progress_state=progress))
        self.assertEqual(progress, {"step": 6, "total_steps": 6})

    def test_batch_id_names_output_file(self):
        path, _ = self.backend().execute(
            "x", {"width": 32, "height": 32, "seed": 1}, batch_id="b1"
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        self.assertEqual(path, "/tmp/ircawp_generated/fibo2.b1.png")

    def test_dispose_releases_pipeline(self):
        backend = self.backend()
        backend.dispose()
        self.assertIsNone(backend.pipe)


class Fibo2EditTurboTests(Fibo2Base):
    backend_class = fibo2editturbo

    def test_subclasses_fibo2_and_loads_turbo_repo(self):
        self.assertTrue(issubclass(fibo2editturbo, fibo2))
        self.backend()
        self.assertEqual(self.loader.call_args.args, ("briaai/Fibo-2.0-edit-turbo",))

    def test_rejects_text_to_image(self):
        with self.assertRaises(ValueError):
            self.backend().execute("x", self.request())
        self.pipeline.assert_not_called()

    def test_edit_uses_four_steps_without_guidance(self):
        self.backend().execute("make it blue", self.request(), media=[self.source()])
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["num_inference_steps"], 4)
        self.assertEqual(kwargs["guidance_scale"], 1.0)
        self.assertNotIn("negative_prompt", kwargs)
        self.assertEqual(len(kwargs["image"]), 1)

    def test_step_and_guidance_overrides_are_normalized(self):
        path, _ = self.backend().execute(
            "x",
            self.request(steps=30, guidance_scale=5.0),
            media=[self.source()],
        )
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["num_inference_steps"], 4)
        self.assertEqual(kwargs["guidance_scale"], 1.0)
        with Image.open(path) as saved:
            self.assertEqual(saved.info["model"], "fibo2editturbo")
            self.assertEqual(saved.info["inference_steps"], "4")


class Fibo2DiscoveryTests(unittest.TestCase):
    def test_capabilities(self):
        self.assertEqual(
            main.BACKEND_CAPABILITIES["fibo2"],
            {"cfg": "guidance_scale", "seed": True},
        )
        self.assertEqual(
            main.BACKEND_CAPABILITIES["fibo2editturbo"], {"seed": True, "steps": False}
        )

    def test_steps_capability_defaults_on_and_is_hidden_for_turbo(self):
        self.assertTrue(main._backend_capabilities("fibo2")["steps"])
        self.assertTrue(main._backend_capabilities("qwenimage21")["steps"])
        self.assertTrue(main._backend_capabilities("unknown")["steps"])
        self.assertFalse(main._backend_capabilities("fibo2editturbo")["steps"])

    def test_no_scheduler_family(self):
        self.assertIsNone(fibo2.scheduler_family)
        self.assertIsNone(fibo2editturbo.scheduler_family)


if __name__ == "__main__":
    unittest.main()

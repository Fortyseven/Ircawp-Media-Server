"""GPU-free integration contract for the MOSS-Image backend and its GGUF converter."""

import importlib
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import gguf
import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch
from torch import nn

from app.backends.MediaBackend import GenerationCancelled, MediaBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GGUF = PROJECT_ROOT / "models" / "moss_image_sft_Q4_K_S.gguf"
CONVERTER = PROJECT_ROOT / "scripts" / "convert_mossimage_to_gguf.py"
BASE_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
# The schedule snaps sigmas to the training grid and can return fewer steps.
ACTUAL_STEP_LIMIT = 20


class FakeScheduler:
    def __init__(self):
        self.timesteps = []

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = list(range(min(num_inference_steps, ACTUAL_STEP_LIMIT)))

    def step(self, model_output, timestep, sample, return_dict=False):
        return (sample,)


class MossImageBackendTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.gguf = self.directory / "transformer.gguf"
        self.gguf.write_bytes(b"mocked transformer loader fixture")
        self.base = self.directory / "base"
        self.base.mkdir()
        self.output = self.directory / "result.png"

        self.scheduler = FakeScheduler()
        self.pipeline = Mock()
        self.pipeline.scheduler = self.scheduler
        self.pipeline.side_effect = self.render
        self.module = SimpleNamespace(
            MOSSImagePipeline=Mock(return_value=self.pipeline),
            MOSSImageScheduler=Mock(),
            MOSSImageTransformer=Mock(),
        )
        self.module.MOSSImageScheduler.from_pretrained.return_value = self.scheduler
        self.module_loader = self.enterContext(
            patch(
                "app.backends.mossimage._load_pipeline_module", return_value=self.module
            )
        )
        self.transformer_loader = self.enterContext(
            patch("app.backends.mossimage._load_gguf_transformer")
        )
        self.encoder_loader = self.enterContext(
            patch("transformers.AutoModel.from_pretrained")
        )
        self.tokenizer_loader = self.enterContext(
            patch("transformers.AutoTokenizer.from_pretrained")
        )
        self.vae_loader = self.enterContext(
            patch("diffusers.AutoencoderKLFlux2.from_pretrained")
        )
        self.snapshot = self.enterContext(
            patch(
                "app.backends.mossimage.snapshot_download", return_value=str(self.base)
            )
        )

    def render(self, **kwargs):
        self.scheduler.set_timesteps(kwargs["num_inference_steps"])
        for timestep in self.scheduler.timesteps:
            self.scheduler.step(None, timestep, None, return_dict=False)
        return SimpleNamespace(
            images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "teal")]
        )

    def backend_class(self):
        module = importlib.import_module("app.backends.mossimage")
        return getattr(module, "mossimage")

    def backend(self, **overrides):
        config = {"gguf": str(self.gguf), "base_model": str(self.base), **overrides}
        return self.backend_class()(config)

    def request(self, **overrides):
        return {
            "output_file": str(self.output),
            "width": 96,
            "height": 64,
            "seed": 123456,
            **overrides,
        }

    def assert_no_loading(self):
        self.module_loader.assert_not_called()
        self.transformer_loader.assert_not_called()
        self.encoder_loader.assert_not_called()
        self.vae_loader.assert_not_called()

    def test_dynamic_import_exports_media_backend(self):
        self.assertTrue(issubclass(self.backend_class(), MediaBackend))

    def test_default_transformer_is_absolute_project_local_file(self):
        backend_class = self.backend_class()
        self.assertTrue(DEFAULT_GGUF.is_absolute())
        if DEFAULT_GGUF.is_file():
            backend_class({"base_model": str(self.base)})
            self.assertEqual(
                Path(self.transformer_loader.call_args.args[1]), DEFAULT_GGUF
            )
        else:
            with self.assertRaises(FileNotFoundError) as failure:
                backend_class({"base_model": str(self.base)})
            self.assertIn(str(DEFAULT_GGUF), str(failure.exception))
            self.assert_no_loading()

    def test_missing_or_directory_transformer_fails_before_loading(self):
        backend_class = self.backend_class()
        missing = str(self.directory / "missing.gguf")
        with self.assertRaises(FileNotFoundError) as failure:
            backend_class({"gguf": missing})
        self.assertIn(missing, str(failure.exception))
        with self.assertRaises((FileNotFoundError, ValueError)):
            backend_class({"gguf": str(self.directory)})
        self.assert_no_loading()

    def test_remote_transformers_are_rejected_before_loading(self):
        backend_class = self.backend_class()
        for remote in (
            "https://huggingface.co/example/model/resolve/main/model.gguf",
            "hf.co/example/model/model.gguf",
            "huggingface.co/example/model/model.gguf",
            "hf://example/model/model.gguf",
        ):
            with self.subTest(remote=remote):
                with self.assertRaisesRegex(ValueError, r"(?i)local|remote|url"):
                    backend_class({"gguf": remote})
                self.assert_no_loading()
                self.snapshot.assert_not_called()

    def test_components_load_from_base_model_with_gguf_transformer(self):
        self.backend()
        self.module_loader.assert_called_once_with(str(self.base))
        self.assertEqual(
            self.transformer_loader.call_args.args[:3],
            (self.module, str(self.gguf), str(self.base)),
        )
        encoder_kwargs = self.encoder_loader.call_args.kwargs
        self.assertEqual(
            self.encoder_loader.call_args.args[0], str(self.base / "text_encoder")
        )
        self.assertTrue(encoder_kwargs["quantization_config"].load_in_4bit)
        self.assertEqual(encoder_kwargs["dtype"], torch.bfloat16)
        self.assertEqual(
            self.tokenizer_loader.call_args.args[0], str(self.base / "tokenizer")
        )
        self.assertEqual(self.vae_loader.call_args.args[0], str(self.base))
        self.assertEqual(self.vae_loader.call_args.kwargs["subfolder"], "vae")
        self.assertEqual(self.vae_loader.call_args.kwargs["torch_dtype"], torch.float32)
        self.module.MOSSImageScheduler.from_pretrained.assert_called_once_with(
            str(self.base), subfolder="scheduler"
        )
        pipeline_kwargs = self.module.MOSSImagePipeline.call_args.kwargs
        self.assertIs(
            pipeline_kwargs["transformer"], self.transformer_loader.return_value
        )
        self.assertIs(pipeline_kwargs["scheduler"], self.scheduler)
        self.assertIs(
            pipeline_kwargs["text_encoder"],
            self.encoder_loader.return_value.eval.return_value,
        )
        self.assertIs(pipeline_kwargs["tokenizer"], self.tokenizer_loader.return_value)
        self.assertIs(
            pipeline_kwargs["vae"], self.vae_loader.return_value.eval.return_value
        )
        self.pipeline.enable_model_cpu_offload.assert_called_once_with()
        self.snapshot.assert_not_called()

    def test_text_encoder_quantization_can_be_disabled(self):
        self.backend(text_encoder_4bit=False)
        self.assertNotIn("quantization_config", self.encoder_loader.call_args.kwargs)

    def test_repo_id_base_model_downloads_only_non_transformer_weights(self):
        self.backend_class()({"gguf": str(self.gguf)})
        self.assertEqual(self.snapshot.call_args.args[0], BASE_REPO)
        ignored = self.snapshot.call_args.kwargs["ignore_patterns"]
        self.assertIn("transformer/*.safetensors", ignored)
        self.module_loader.assert_called_once_with(str(self.base))

    def test_generation_forwards_defaults_and_saves_metadata(self):
        backend = self.backend()
        progress = {}
        config = self.request(progress_state=progress)
        original = dict(config)
        output_path, prompt = backend.execute("  a precise illustration  ", config)
        self.assertEqual(config, original)
        self.assertEqual(output_path, str(self.output))
        self.assertEqual(prompt, "a precise illustration")
        self.assertEqual(backend.last_imagegen_prompt, prompt)
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["prompt"], prompt)
        self.assertEqual((kwargs["width"], kwargs["height"]), (96, 64))
        self.assertEqual(kwargs["num_inference_steps"], 50)
        self.assertEqual(kwargs["guidance_scale"], 3.0)
        self.assertEqual(kwargs["generator"].initial_seed(), 123456)
        self.assertNotIn("negative_prompt", kwargs)
        self.assertNotIn("uncond_skip_layers", kwargs)
        self.assertEqual(
            progress, {"step": ACTUAL_STEP_LIMIT, "total_steps": ACTUAL_STEP_LIMIT}
        )
        with Image.open(output_path) as saved:
            self.assertEqual(saved.size, (96, 64))
            self.assertEqual(saved.info["model"], "mossimage")
            self.assertEqual(saved.info["inference_steps"], str(ACTUAL_STEP_LIMIT))
            self.assertEqual(saved.info["seed"], "123456")
            self.assertEqual(saved.info["guidance_scale"], "3.0")
            self.assertEqual(saved.info["width"], "96")
            self.assertEqual(saved.info["height"], "64")
            self.assertEqual(saved.info["prompt"], prompt)

    def test_steps_and_guidance_overrides_reach_pipeline(self):
        backend = self.backend()
        backend.execute("configured", self.request(steps=12))
        self.assertEqual(self.pipeline.call_args.kwargs["num_inference_steps"], 12)
        backend.execute("override", self.request(steps=7, guidance_scale=1.5))
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["num_inference_steps"], 7)
        self.assertEqual(kwargs["guidance_scale"], 1.5)
        with Image.open(self.output) as saved:
            self.assertEqual(saved.info["inference_steps"], "7")
            self.assertEqual(saved.info["guidance_scale"], "1.5")

    def test_remaster_does_not_change_prompt_or_steps(self):
        self.backend().execute("plain", self.request(remaster=True))
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(kwargs["prompt"], "plain")
        self.assertEqual(kwargs["num_inference_steps"], 50)

    def test_missing_seed_is_generated_and_recorded(self):
        config = self.request()
        del config["seed"]
        self.backend().execute("random", config)
        seed = self.pipeline.call_args.kwargs["generator"].initial_seed()
        with Image.open(self.output) as saved:
            self.assertEqual(saved.info["seed"], str(seed))

    def test_dimensions_derive_from_aspect_and_max_output_size(self):
        backend = self.backend()
        for config, expected in (
            ({}, (1024, 688)),
            ({"aspect": "16:9"}, (1024, 576)),
            ({"aspect": "9:16", "max_output_size": 512}, (288, 512)),
        ):
            with self.subTest(config=config):
                backend.execute(
                    "sized", {"output_file": str(self.output), "seed": 1, **config}
                )
                kwargs = self.pipeline.call_args.kwargs
                self.assertEqual((kwargs["width"], kwargs["height"]), expected)

    def test_reference_images_are_rejected_without_generating(self):
        source = self.directory / "input.png"
        Image.new("RGB", (32, 32), "red").save(source)
        with self.assertRaisesRegex(ValueError, r"(?i)edit|reference|image"):
            self.backend().execute("edit", self.request(), media=[str(source)])
        self.pipeline.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cancellation_stops_generation_before_png_is_saved(self):
        backend = self.backend()
        cancellation = Event()
        progress = {}

        def cancel_after_first_step(**kwargs):
            self.scheduler.set_timesteps(kwargs["num_inference_steps"])
            self.scheduler.step(None, 0, None)
            cancellation.set()
            self.scheduler.step(None, 1, None)
            self.fail("Cancellation did not abort generation")

        self.pipeline.side_effect = cancel_after_first_step
        with self.assertRaises(GenerationCancelled):
            backend.execute(
                "cancel me",
                self.request(cancellation_event=cancellation, progress_state=progress),
            )
        self.assertEqual(progress["step"], 1)
        self.assertEqual(progress["total_steps"], ACTUAL_STEP_LIMIT)
        self.assertFalse(self.output.exists())

    def test_scheduler_is_restored_after_success_and_failure(self):
        backend = self.backend()
        backend.execute("ok", self.request())
        self.assertNotIn("step", vars(self.scheduler))
        self.pipeline.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            backend.execute("fails", self.request())
        self.assertNotIn("step", vars(self.scheduler))

    def test_dispose_releases_pipeline(self):
        backend = self.backend()
        backend.dispose()
        self.assertIsNone(backend.pipe)


class TinyTransformer(ModelMixin, ConfigMixin):
    """Mimics MOSSImageTransformer's reliance on `weight.dtype` for activation casts."""

    config_name = "config.json"

    @register_to_config
    def __init__(self, dim=64):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.proj = nn.ModuleDict({"q": nn.Linear(dim, dim, bias=False)})

    def forward(self, x):
        dtype = self.proj["q"].weight.dtype
        return self.proj["q"](self.embed(x).to(dtype))


class GGUFTransformerLoaderTests(unittest.TestCase):
    def test_quantized_weights_load_and_keep_activations_floating_point(self):
        from app.backends.mossimage import _load_gguf_transformer

        with TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "transformer").mkdir()
            (base / "transformer" / "config.json").write_text(
                json.dumps({"_class_name": "TinyTransformer", "dim": 64})
            )
            generator = torch.Generator().manual_seed(0)
            q_weight = torch.randn(64, 64, generator=generator).numpy()
            embed_weight = torch.randn(64, 64, generator=generator).numpy()
            embed_bias = torch.randn(64, generator=generator).numpy()
            path = base / "tiny.gguf"
            writer = gguf.GGUFWriter(str(path), "qwen2")
            quantized = gguf.quants.quantize(q_weight, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(
                "proj.q.weight",
                quantized,
                raw_dtype=gguf.GGMLQuantizationType.Q8_0,
            )
            writer.add_tensor("embed.weight", embed_weight.astype(np.float16))
            writer.add_tensor("embed.bias", embed_bias)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            module = SimpleNamespace(MOSSImageTransformer=TinyTransformer)
            model = _load_gguf_transformer(module, str(path), str(base))

        self.assertEqual(model.proj["q"].weight.dtype, torch.bfloat16)
        x = torch.randn(2, 64, generator=generator)
        output = model(x.to(torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)
        dequantized = torch.from_numpy(
            gguf.quants.dequantize(quantized, gguf.GGMLQuantizationType.Q8_0)
        )
        hidden = x.to(torch.bfloat16).float() @ torch.from_numpy(
            embed_weight.astype(np.float16)
        ).float().T + torch.from_numpy(embed_bias)
        expected = hidden @ dequantized.T
        torch.testing.assert_close(output.float(), expected, atol=0.5, rtol=0.1)

    def test_checkpoint_missing_tensors_is_rejected(self):
        from app.backends.mossimage import _load_gguf_transformer

        with TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "transformer").mkdir()
            (base / "transformer" / "config.json").write_text(
                json.dumps({"_class_name": "TinyTransformer", "dim": 64})
            )
            path = base / "partial.gguf"
            writer = gguf.GGUFWriter(str(path), "qwen2")
            writer.add_tensor("embed.bias", np.zeros(64, dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            module = SimpleNamespace(MOSSImageTransformer=TinyTransformer)
            with self.assertRaisesRegex(ValueError, r"(?i)missing|unexpected"):
                _load_gguf_transformer(module, str(path), str(base))


class ConverterTests(unittest.TestCase):
    def test_matrices_become_f16_vectors_f32_and_names_and_shapes_are_kept(self):
        converter = runpy.run_path(str(CONVERTER))
        tensors = {
            "layers.0.proj.q.weight": torch.randn(4, 8).to(torch.bfloat16),
            "layers.0.norm1.weight": torch.randn(8).to(torch.bfloat16),
            "rope.bands": torch.linspace(1, 0.1, 16),
        }
        with TemporaryDirectory() as directory:
            repo = Path(directory) / "MOSS-Image-Test"
            (repo / "transformer").mkdir(parents=True)
            shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
            save_file(
                tensors, str(repo / "transformer" / shard), metadata={"format": "pt"}
            )
            (
                repo / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
            ).write_text(json.dumps({"weight_map": {name: shard for name in tensors}}))
            out = Path(directory) / "out.gguf"
            converter["convert"](str(repo), out)
            tensors_out = {t.name: t for t in gguf.GGUFReader(str(out)).tensors}

        self.assertEqual(set(tensors_out), set(tensors))
        kinds = {name: t.tensor_type for name, t in tensors_out.items()}
        self.assertEqual(kinds["layers.0.proj.q.weight"], gguf.GGMLQuantizationType.F16)
        self.assertEqual(kinds["layers.0.norm1.weight"], gguf.GGMLQuantizationType.F32)
        self.assertEqual(kinds["rope.bands"], gguf.GGMLQuantizationType.F32)
        for name, source in tensors.items():
            self.assertEqual(
                tuple(reversed(tensors_out[name].shape)), tuple(source.shape)
            )
        np.testing.assert_allclose(
            np.asarray(tensors_out["layers.0.proj.q.weight"].data).reshape(4, 8),
            tensors["layers.0.proj.q.weight"].to(torch.float16).numpy(),
        )

    def test_f16_overflow_is_rejected(self):
        converter = runpy.run_path(str(CONVERTER))
        tensors = {"layers.0.proj.q.weight": torch.full((2, 2), 1e6).to(torch.bfloat16)}
        with TemporaryDirectory() as directory:
            repo = Path(directory) / "MOSS-Image-Test"
            (repo / "transformer").mkdir(parents=True)
            shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
            save_file(
                tensors, str(repo / "transformer" / shard), metadata={"format": "pt"}
            )
            (
                repo / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
            ).write_text(json.dumps({"weight_map": {name: shard for name in tensors}}))
            with self.assertRaisesRegex(ValueError, r"(?i)overflow"):
                converter["convert"](str(repo), Path(directory) / "out.gguf")


if __name__ == "__main__":
    unittest.main()

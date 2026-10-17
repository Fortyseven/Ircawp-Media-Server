"""GPU-free integration contract for the 8-step MOSS-Image DMD backend."""

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from safetensors.torch import save_file
import torch
from torch import nn

from app.backends.MediaBackend import GenerationCancelled
from app.backends.mossimage import mossimage

CHANNELS = 4
CODES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


def tiny_transformer(layers=2, dim=6):
    model = nn.Module()
    model.layers = nn.ModuleList()
    for _ in range(layers):
        block = nn.Module()
        block.proj = nn.ModuleDict(
            {
                name: nn.Linear(dim, dim, bias=False)
                for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
            }
        )
        model.layers.append(block)
    model.time_embedding = nn.Sequential(
        nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim)
    )
    model.shared_block_modulation = nn.Linear(dim, 6 * dim)
    model.video_head_modulation = nn.Linear(dim, 2 * dim)
    return model


def tiny_adapters(layers=2, dim=6, rank=2, alpha=4.0):
    generator = torch.Generator().manual_seed(0)
    lora = {}
    for index in range(layers):
        for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            prefix = f"diffusion_model.layers.{index}.proj.{name}"
            lora[f"{prefix}.lora_down.weight"] = torch.randn(
                rank, dim, generator=generator
            )
            lora[f"{prefix}.lora_up.weight"] = torch.randn(
                dim, rank, generator=generator
            )
            lora[f"{prefix}.alpha"] = torch.tensor(alpha)
    time_modules = {
        "time_embedding.0.weight": torch.randn(dim, 4, generator=generator),
        "time_embedding.0.bias": torch.randn(dim, generator=generator),
        "time_embedding.2.weight": torch.randn(dim, dim, generator=generator),
        "time_embedding.2.bias": torch.randn(dim, generator=generator),
        "shared_block_modulation.bias": torch.randn(6 * dim, generator=generator),
        "video_head_modulation.bias": torch.randn(2 * dim, generator=generator),
    }
    return lora, time_modules


class AdapterApplicationTests(unittest.TestCase):
    def setUp(self):
        from app.backends.mossimage8 import _apply_adapters

        self.apply = _apply_adapters

    def test_lora_branch_is_unmerged_and_scaled_by_alpha_over_rank(self):
        model = tiny_transformer()
        lora, time_modules = tiny_adapters(rank=2, alpha=4.0)
        base_layer = model.layers[1].proj["ffn_down"]
        x = torch.randn(3, 6)
        base_output = base_layer(x)
        self.apply(model, lora, time_modules)
        wrapped = model.layers[1].proj["ffn_down"]
        prefix = "diffusion_model.layers.1.proj.ffn_down"
        delta = (x @ lora[f"{prefix}.lora_down.weight"].T) @ lora[
            f"{prefix}.lora_up.weight"
        ].T
        torch.testing.assert_close(wrapped(x), base_output + 2.0 * delta)
        self.assertIs(wrapped.weight, base_layer.weight)
        self.assertTrue(torch.equal(base_layer(x), base_output))

    def test_lora_is_applied_to_every_targeted_layer(self):
        model = tiny_transformer(layers=3)
        self.apply(model, *tiny_adapters(layers=3))
        for layer in model.layers:
            self.assertEqual(len(layer.proj), 6)
            for projection in layer.proj.values():
                self.assertTrue(hasattr(projection, "down"))

    def test_time_modules_are_replaced_with_fp32_values(self):
        model = tiny_transformer()
        for module in (model.time_embedding[0], model.shared_block_modulation):
            module.to(torch.bfloat16)
        lora, time_modules = tiny_adapters()
        self.apply(model, lora, time_modules)
        for position in (0, 2):
            layer = model.time_embedding[position]
            self.assertEqual(layer.weight.dtype, torch.float32)
            torch.testing.assert_close(
                layer.weight, time_modules[f"time_embedding.{position}.weight"]
            )
            torch.testing.assert_close(
                layer.bias, time_modules[f"time_embedding.{position}.bias"]
            )
        for name in ("shared_block_modulation", "video_head_modulation"):
            layer = getattr(model, name)
            self.assertEqual(layer.bias.dtype, torch.float32)
            torch.testing.assert_close(layer.bias, time_modules[f"{name}.bias"])

    def test_mismatched_lora_or_time_modules_are_rejected(self):
        lora, time_modules = tiny_adapters()
        missing = dict(lora)
        missing.popitem()
        with self.assertRaisesRegex(ValueError, r"(?i)missing"):
            self.apply(tiny_transformer(), missing, time_modules)
        extra = {**lora, "diffusion_model.layers.9.proj.q.alpha": torch.tensor(1.0)}
        with self.assertRaisesRegex(ValueError, r"(?i)unexpected"):
            self.apply(tiny_transformer(), extra, time_modules)
        with self.assertRaisesRegex(ValueError, r"(?i)time_embedding"):
            self.apply(
                tiny_transformer(), lora, {"time_embedding.0.weight": torch.zeros(1)}
            )


class MossImage8BackendTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.gguf = self.directory / "transformer.gguf"
        self.gguf.write_bytes(b"mocked transformer loader fixture")
        self.base = self.directory / "base"
        self.base.mkdir()
        self.adapter = self.directory / "adapter"
        self.adapter.mkdir()
        self.mix = torch.eye(CHANNELS) * 0.5 + 0.1
        save_file(
            {"latent_mix.Q": self.mix, "step_codes": torch.tensor(CODES)},
            str(self.adapter / "dataspace.safetensors"),
        )
        save_file({"x": torch.zeros(1)}, str(self.adapter / "lora.safetensors"))
        save_file(
            {"x": torch.zeros(1)}, str(self.adapter / "time_embedding.safetensors")
        )
        self.output = self.directory / "result.png"

        self.calls = []
        self.pipeline = Mock()
        self.pipeline.encode_prompt.return_value = (
            torch.zeros(1, 3, 8),
            torch.ones(1, 3, dtype=torch.bool),
        )
        self.pipeline._execution_device = torch.device("cpu")
        self.pipeline.transformer = Mock(side_effect=self.forward)
        self.pipeline.transformer.config.latent_channels = CHANNELS
        self.module = SimpleNamespace(
            MOSSImagePipeline=Mock(return_value=self.pipeline),
            MOSSImageScheduler=Mock(),
            MOSSImageTransformer=Mock(),
        )
        self.patches = {
            "loader": self.enterContext(
                patch(
                    "app.backends.mossimage._load_pipeline_module",
                    return_value=self.module,
                )
            ),
            "gguf": self.enterContext(
                patch("app.backends.mossimage._load_gguf_transformer")
            ),
            "adapters": self.enterContext(
                patch(
                    "app.backends.mossimage8._apply_adapters",
                    side_effect=lambda transformer, lora, time: transformer,
                )
            ),
            "download": self.enterContext(
                patch(
                    "app.backends.mossimage8.hf_hub_download",
                    side_effect=lambda repo, name: str(self.adapter / name),
                )
            ),
        }
        self.enterContext(patch("transformers.AutoModel.from_pretrained"))
        self.enterContext(patch("transformers.AutoTokenizer.from_pretrained"))
        self.enterContext(patch("diffusers.AutoencoderKLFlux2.from_pretrained"))

    def forward(self, x, code, embeds, mask):
        self.calls.append((x.clone(), code.clone()))
        return SimpleNamespace(sample=x * 0.25 + 1.0)

    def backend_class(self):
        return importlib.import_module("app.backends.mossimage8").mossimage8

    def backend(self, **overrides):
        config = {
            "gguf": str(self.gguf),
            "base_model": str(self.base),
            "adapter": str(self.adapter),
            **overrides,
        }
        backend = self.backend_class()(config)
        backend._decode = Mock(
            side_effect=lambda latents: Image.new(
                "RGB", (latents.shape[3] * 16, latents.shape[2] * 16), "teal"
            )
        )
        return backend

    def request(self, **overrides):
        return {
            "output_file": str(self.output),
            "width": 96,
            "height": 64,
            "seed": 123,
            **overrides,
        }

    def test_extends_base_backend(self):
        self.assertTrue(issubclass(self.backend_class(), mossimage))

    def test_adapters_are_applied_to_the_loaded_transformer(self):
        self.backend()
        adapters = self.patches["adapters"]
        self.assertIs(adapters.call_args.args[0], self.patches["gguf"].return_value)
        self.assertIs(
            self.module.MOSSImagePipeline.call_args.kwargs["transformer"],
            self.patches["gguf"].return_value,
        )

    def test_repo_adapter_downloads_the_three_files(self):
        self.backend_class()({"gguf": str(self.gguf), "base_model": str(self.base)})
        downloaded = {call.args[1] for call in self.patches["download"].call_args_list}
        self.assertEqual(
            downloaded,
            {"lora.safetensors", "time_embedding.safetensors", "dataspace.safetensors"},
        )
        self.assertEqual(
            self.patches["download"].call_args.args[0],
            "lodestones/MOSS-Image-SFT-8-steps-dmd",
        )

    def test_missing_local_adapter_file_fails_before_loading(self):
        (self.adapter / "lora.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            self.backend()
        self.patches["loader"].assert_not_called()
        self.patches["gguf"].assert_not_called()

    def test_missing_gguf_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            self.backend_class()(
                {
                    "gguf": str(self.directory / "missing.gguf"),
                    "adapter": str(self.adapter),
                }
            )

    def test_sampler_follows_data_space_recurrence_without_cfg(self):
        backend = self.backend()
        progress = {}
        output_path, prompt = backend.execute(
            "  a cat  ", self.request(progress_state=progress)
        )
        self.assertEqual((output_path, prompt), (str(self.output), "a cat"))
        self.pipeline.encode_prompt.assert_called_once_with("a cat")
        self.assertEqual(len(self.calls), 8)
        noise = torch.randn(
            (1, CHANNELS, 4, 6), generator=torch.Generator("cpu").manual_seed(123)
        )
        x0 = torch.zeros_like(noise)
        for step, (x, code) in enumerate(self.calls):
            torch.testing.assert_close(
                x, noise + torch.einsum("oc,bchw->bohw", self.mix, x0)
            )
            self.assertAlmostEqual(code.item(), CODES[step], places=6)
            x0 = noise - (x * 0.25 + 1.0)
        torch.testing.assert_close(backend._decode.call_args.args[0], x0)
        self.assertEqual(progress, {"step": 8, "total_steps": 8})

    def test_steps_use_a_prefix_of_the_codes_and_are_clamped_to_eight(self):
        backend = self.backend()
        for requested, actual in ((3, 3), (1, 1), (None, 8), (30, 8)):
            with self.subTest(requested=requested):
                self.calls.clear()
                backend.execute("p", self.request(steps=requested))
                self.assertEqual(
                    [round(c.item(), 6) for _, c in self.calls], CODES[:actual]
                )
                with Image.open(self.output) as saved:
                    self.assertEqual(saved.info["inference_steps"], str(actual))

    def test_metadata_and_size(self):
        backend = self.backend()
        backend.execute("meta", self.request(guidance_scale=9.0))
        with Image.open(self.output) as saved:
            self.assertEqual(saved.size, (96, 64))
            self.assertEqual(saved.info["model"], "mossimage8")
            self.assertEqual(saved.info["seed"], "123")
            self.assertNotIn("guidance_scale", saved.info)
        self.assertEqual(backend.last_imagegen_prompt, "meta")

    def test_reference_images_are_rejected(self):
        source = self.directory / "input.png"
        Image.new("RGB", (16, 16)).save(source)
        with self.assertRaisesRegex(ValueError, r"(?i)edit"):
            self.backend().execute("edit", self.request(), media=[str(source)])
        self.assertEqual(self.calls, [])
        self.assertFalse(self.output.exists())

    def test_cancellation_stops_between_steps_before_png_is_saved(self):
        backend = self.backend()
        cancellation = Event()
        progress = {}

        def cancel_after_first(x, code, embeds, mask):
            self.calls.append((x, code))
            if len(self.calls) == 2:
                cancellation.set()
            return SimpleNamespace(sample=x)

        self.pipeline.transformer.side_effect = cancel_after_first
        with self.assertRaises(GenerationCancelled):
            backend.execute(
                "cancel",
                self.request(cancellation_event=cancellation, progress_state=progress),
            )
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(progress, {"step": 1, "total_steps": 8})
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

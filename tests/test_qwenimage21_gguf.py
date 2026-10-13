"""CPU-only integration contract for official Qwen safetensors to F16 GGUF."""

from contextlib import chdir
import importlib.util
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from gguf import GGMLQuantizationType, GGUFReader
import numpy as np
import safetensors
from safetensors.torch import save_file
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER = PROJECT_ROOT / "scripts" / "convert_qwenimage21_to_gguf.py"
TURBO_REPO = "Qwen/Qwen-Image-2.1-Turbo"
BASE_REPO = "Qwen/Qwen-Image-2.1"
INDEX_NAME = "transformer/diffusion_pytorch_model.safetensors.index.json"


class QwenImage21GGUFIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.tensors = {
            "transformer_blocks.0.attn.to_q.weight": torch.tensor(
                [[0.1, -0.2, 1.5], [2.75, -4.125, 0.0]],
                dtype=torch.bfloat16,
                device="cpu",
            ),
            "norm_out.linear.weight": torch.tensor(
                [[0.333333, -0.123456], [3.141592, 0.000123], [12.3456, -7.89]],
                dtype=torch.float32,
                device="cpu",
            ),
            "transformer_blocks.0.attn.to_k.weight": torch.tensor(
                [[0.25, -0.5, 1.0, 2.0]],
                dtype=torch.float16,
                device="cpu",
            ),
        }
        shard_names = (
            "diffusion_pytorch_model-00001-of-00002.safetensors",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
        )
        tensor_names = list(self.tensors)
        weight_map = {
            tensor_names[0]: shard_names[0],
            tensor_names[1]: shard_names[0],
            tensor_names[2]: shard_names[1],
        }
        self.downloads = {}
        for shard_name in shard_names:
            shard_path = self.directory / shard_name
            save_file(
                {
                    name: self.tensors[name]
                    for name, shard in weight_map.items()
                    if shard == shard_name
                },
                str(shard_path),
                metadata={"format": "pt"},
            )
            self.downloads[f"transformer/{shard_name}"] = shard_path
        index_path = self.directory / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum(
                            tensor.numel() * tensor.element_size()
                            for tensor in self.tensors.values()
                        ),
                    },
                    "weight_map": weight_map,
                }
            ),
            encoding="utf-8",
        )
        self.downloads[INDEX_NAME] = index_path
        self.expected_repo = TURBO_REPO
        self.download = self.enterContext(
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=self.download_fixture,
            )
        )
        self.safe_open = self.enterContext(
            patch(
                "safetensors.safe_open",
                wraps=safetensors.safe_open,
            )
        )

    def download_fixture(self, repo_id, filename, *, subfolder=None, **kwargs):
        self.assertEqual(repo_id, self.expected_repo)
        remote_path = f"{subfolder}/{filename}" if subfolder else filename
        self.assertIn(remote_path, self.downloads)
        return str(self.downloads[remote_path])

    def load_converter(self):
        specification = importlib.util.spec_from_file_location(
            "qwenimage21_gguf_converter",
            CONVERTER,
        )
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    def assert_conversion(self, output, repo):
        self.assertTrue(output.is_file())
        reader = GGUFReader(str(output))
        actual_tensors = {tensor.name: tensor for tensor in reader.tensors}
        self.assertEqual(len(reader.tensors), 3)
        self.assertEqual(set(actual_tensors), set(self.tensors))
        for name, original in self.tensors.items():
            with self.subTest(tensor=name):
                actual = actual_tensors[name]
                self.assertEqual(actual.tensor_type, GGMLQuantizationType.F16)
                self.assertEqual(tuple(actual.shape), tuple(reversed(original.shape)))
                self.assertEqual(actual.data.shape, tuple(original.shape))
                self.assertEqual(actual.data.dtype, np.dtype("float16"))
                expected = original.to(torch.float16).numpy()
                np.testing.assert_array_equal(actual.data, expected)

        def metadata(key):
            field = reader.get_field(key)
            self.assertIsNotNone(field, f"Missing GGUF metadata: {key}")
            return field.contents()

        self.assertEqual(
            metadata("general.source.repo_url"), f"https://huggingface.co/{repo}"
        )
        self.assertIn(repo.rsplit("/", 1)[-1], metadata("general.name"))
        if repo == BASE_REPO:
            self.assertNotIn("Turbo", metadata("general.name"))
        self.assertEqual(metadata("general.architecture"), "qwen2")
        self.assertEqual(metadata("qwen2.context_length"), 4096)
        self.assertEqual(metadata("qwen2.embedding_length"), 4096)
        self.assertEqual(metadata("qwen2.block_count"), 32)
        self.assertAlmostEqual(
            metadata("qwen2.attention.layer_norm_rms_epsilon"),
            1e-6,
            delta=1e-12,
        )
        self.assertEqual(self.download.call_count, 3)
        opened_paths = {
            Path(call.args[0] if call.args else call.kwargs["filename"])
            for call in self.safe_open.call_args_list
        }
        self.assertTrue(
            {
                path
                for remote, path in self.downloads.items()
                if remote.endswith(".safetensors")
            }
            <= opened_paths
        )

    def test_turbo_conversion_preserves_mixed_dtype_tensors_as_f16(self):
        converter = self.load_converter()
        output = self.directory / "nested" / "models" / "turbo_F16.gguf"
        converter.convert(TURBO_REPO, output)
        self.assert_conversion(output, TURBO_REPO)

    def test_base_repo_conversion_uses_base_provenance_and_name(self):
        self.expected_repo = BASE_REPO
        converter = self.load_converter()
        output = self.directory / "base" / "model_F16.gguf"
        converter.convert(BASE_REPO, output)
        self.assert_conversion(output, BASE_REPO)

    def test_cli_defaults_to_official_turbo_and_project_models_output(self):
        with chdir(self.directory), patch("sys.argv", [str(CONVERTER)]):
            runpy.run_path(str(CONVERTER), run_name="__main__")
        self.assert_conversion(
            self.directory / "models" / "qwen_image_2.1_turbo_F16.gguf",
            TURBO_REPO,
        )

    def test_cli_accepts_base_repo_and_nested_output_override(self):
        self.expected_repo = BASE_REPO
        output = self.directory / "custom" / "nested" / "base_F16.gguf"
        with patch(
            "sys.argv",
            [
                str(CONVERTER),
                "--repo",
                BASE_REPO,
                "--out",
                str(output),
            ],
        ):
            runpy.run_path(str(CONVERTER), run_name="__main__")
        self.assert_conversion(output, BASE_REPO)


if __name__ == "__main__":
    unittest.main()

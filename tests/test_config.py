"""Configuration requirements for selecting the default backend."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from app.main import load_config


class BackendConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.config_path = Path(temporary_directory.name) / "config.yml"

    def test_missing_config_requires_explicit_backend(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            load_config(str(self.config_path))

    def test_missing_backend_is_rejected(self):
        for config in ({}, {"backends": {"flux2klein": {}}}):
            with self.subTest(config=config):
                self.config_path.write_text(yaml.safe_dump(config))
                with self.assertRaisesRegex(ValueError, "backend"):
                    load_config(str(self.config_path))

    def test_invalid_backend_is_rejected(self):
        for backend in (None, "", "   ", 0, False, [], {}):
            with self.subTest(backend=backend):
                self.config_path.write_text(yaml.safe_dump({"backend": backend}))
                with self.assertRaisesRegex(ValueError, "backend"):
                    load_config(str(self.config_path))

    def test_explicit_backend_is_preserved(self):
        self.config_path.write_text("backend: qwenimage21turbo\n")
        self.assertEqual(
            load_config(str(self.config_path))["backend"], "qwenimage21turbo"
        )

    def test_empty_environment_backend_is_rejected(self):
        self.config_path.write_text("backend: ${MEDIA_SERVER_TEST_BACKEND}\n")
        with patch.dict("os.environ", {"MEDIA_SERVER_TEST_BACKEND": ""}):
            with self.assertRaisesRegex(ValueError, "backend"):
                load_config(str(self.config_path))

    def test_environment_backend_is_expanded(self):
        self.config_path.write_text("backend: ${MEDIA_SERVER_TEST_BACKEND}\n")
        with patch.dict(
            "os.environ", {"MEDIA_SERVER_TEST_BACKEND": "qwenimage21turbo"}
        ):
            self.assertEqual(
                load_config(str(self.config_path))["backend"], "qwenimage21turbo"
            )

"""Image-edit request handling tests."""

import base64
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main


class RecordingBackend:
    def __init__(self):
        self.media_sizes = []

    def execute(self, *, prompt, config, media):
        self.media_sizes = []
        for path in media:
            with Image.open(path) as image:
                self.media_sizes.append(image.size)
        Path(config["output_file"]).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16)).save(config["output_file"])
        return config["output_file"]


class ImageEditsInputResizeTests(unittest.TestCase):
    def _image_data_url(self, size):
        image_bytes = BytesIO()
        Image.new("RGB", size).save(image_bytes, format="PNG")
        encoded = base64.b64encode(image_bytes.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _post_edit(self, client, image_url, **extra):
        return client.post(
            "/images/edits",
            json={
                "prompt": "make it brighter",
                "model": "test-backend",
                "images": [{"image_url": image_url}],
                **extra,
            },
        )

    def test_edits_resizes_input_to_configured_backend_maximum_edge(self):
        backend = RecordingBackend()
        with (
            patch.object(
                main,
                "CONFIG",
                {"backends": {"test-backend": {"max_input_edge": 1000}}},
            ),
            patch.object(main, "get_backend", return_value=backend),
            TestClient(main.app) as client,
        ):
            response = self._post_edit(client, self._image_data_url((2000, 1000)))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(backend.media_sizes, [(1000, 500)])

    def test_edits_preserves_input_dimensions_when_resize_is_disabled(self):
        backend = RecordingBackend()
        with (
            patch.object(
                main,
                "CONFIG",
                {"backends": {"test-backend": {"max_input_edge": 1000}}},
            ),
            patch.object(main, "get_backend", return_value=backend),
            TestClient(main.app) as client,
        ):
            response = self._post_edit(
                client,
                self._image_data_url((2000, 1000)),
                resize_input_images=False,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(backend.media_sizes, [(2000, 1000)])

"""Local GGUF Qwen-Image-2.1 Turbo backend with its eight-step schedule."""

import math
import os
from numbers import Real
from pathlib import Path

import numpy as np
from diffusers import QwenImage21Pipeline

from .qwenimage21 import qwenimage21


BASE_MODEL = "Qwen/Qwen-Image-2.1-Turbo"
DEFAULT_GGUF = str(
    Path(__file__).resolve().parents[2] / "models" / "qwen_image_2.1_turbo_Q4_K_S.gguf"
)
DEFAULT_STEPS = 16


class qwenimage21turbo(qwenimage21):
    model_id = "qwenimage21turbo"
    remaster_extra_steps = 0

    def __init__(self, backend_config: dict = {}):
        backend_config = dict(backend_config)
        gguf_path = os.fspath(backend_config.get("gguf", DEFAULT_GGUF))
        if "://" in gguf_path or gguf_path.lower().startswith(
            ("hf.co/", "huggingface.co/")
        ):
            raise ValueError("Turbo requires a local GGUF file, not a remote URL")
        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(gguf_path)
        backend_config["gguf"] = os.path.abspath(gguf_path)
        backend_config.setdefault("base_model", BASE_MODEL)

        model_config = QwenImage21Pipeline.load_config(backend_config["base_model"])
        sigmas = model_config.get("sample_sigmas")
        if (
            not isinstance(sigmas, (list, tuple))
            or len(sigmas) != 8
            or any(
                isinstance(sigma, bool)
                or not isinstance(sigma, Real)
                or not math.isfinite(sigma)
                or not 0 < sigma <= 1
                for sigma in sigmas
            )
            or any(
                current <= following for current, following in zip(sigmas, sigmas[1:])
            )
        ):
            raise ValueError(
                "sample_sigmas must contain exactly eight finite numeric, positive, "
                "strictly decreasing values no greater than one"
            )
        self.sample_sigmas = list(sigmas)
        super().__init__(backend_config)

    def _sampling_kwargs(self, config: dict) -> dict:
        # Explicit sigmas override num_inference_steps, so resample the
        # distilled schedule to the requested step count.
        steps = config.get("steps") or len(self.sample_sigmas)
        if steps == len(self.sample_sigmas):
            sigmas = self.sample_sigmas
        else:
            sigmas = np.interp(
                np.linspace(0, 1, steps),
                np.linspace(0, 1, len(self.sample_sigmas)),
                self.sample_sigmas,
            ).tolist()
        return {
            "sigmas": sigmas,
            "use_kv_cache": True,
            "true_cfg_scale": config.get("true_cfg_scale", 1.0),
        }

    def execute(
        self,
        prompt: str,
        config: dict = {},
        batch_id=None,
        media=[],
    ) -> tuple[str, str]:
        config = dict(config)
        config["steps"] = config.get("steps") or DEFAULT_STEPS
        return super().execute(prompt, config, batch_id=batch_id, media=media)

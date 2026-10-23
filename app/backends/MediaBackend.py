"""Base class for media (image generation) backends.

This is the media-server side — strictly prompt-in → image-out.
No LLM calls, no prompt refinement.
"""

from PIL import Image, PngImagePlugin


class GenerationCancelled(Exception):
    """Raised when a generation backend observes a cancellation request."""


class MediaBackend:
    # Family key into schedulers.FAMILIES; None means no per-request scheduler.
    scheduler_family: str | None = None
    _default_scheduler = None

    def __init__(self, backend_config: dict = {}):
        self.backend_config = backend_config
        self.last_imagegen_prompt = None

    def _apply_scheduler(self, config: dict, seed: int | None = None) -> str | None:
        """Install the scheduler requested in config["scheduler"] on self.pipe.

        Returns the name applied, or None when the pipeline's own default is used.
        """
        name = config.get("scheduler")
        if not name or name == "default":
            if self._default_scheduler is not None:
                self.pipe.scheduler = self._default_scheduler
            return None
        if self.scheduler_family is None:
            return None

        from .schedulers import build_scheduler

        if self._default_scheduler is None:
            self._default_scheduler = self.pipe.scheduler
        self.pipe.scheduler = build_scheduler(
            self.scheduler_family, name, self._default_scheduler.config
        )
        if self.scheduler_family == "flow" and seed is not None:
            import torch

            # Flow pipelines don't pass a generator to scheduler.step, so its
            # re-noising draws from the global RNG.
            torch.manual_seed(seed)
        return name

    def execute(
        self, prompt: str, config: dict = {}, batch_id=None, media=[]
    ) -> str | tuple[str, str]:
        """Execute image generation.

        Returns:
            str: path to generated image, or
            tuple[str, str]: (path, final_prompt)
        """
        return ""

    def dispose(self) -> None:
        """Release heavy model resources held by this backend.

        Called by the server's /backends/unload endpoint. Drops references
        to loaded pipelines/models so they can be garbage-collected, then
        frees the CUDA cache if a GPU is in use.
        """
        import gc

        for attr in ("pipe", "pipe2", "model", "vae", "text_encoder", "transformer"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def raise_if_cancelled(config: dict) -> None:
        """Abort cooperative backend work when the request has been cancelled."""
        cancellation_event = config.get("cancellation_event")
        if cancellation_event is not None and cancellation_event.is_set():
            raise GenerationCancelled()

    def cancellation_callback(self, config: dict, total_steps: int | None = None):
        """Return a diffusers-compatible callback that observes cancellation and reports progress."""
        progress_state = config.get("progress_state")

        def callback(pipe, step_index, timestep, callback_kwargs):
            self.raise_if_cancelled(config)
            if progress_state is not None:
                progress_state["step"] = step_index + 1
                if total_steps is not None:
                    progress_state["total_steps"] = total_steps
            return callback_kwargs

        return callback

    @staticmethod
    def _negative_prompt_kwargs(config: dict) -> dict:
        """Pipeline kwargs for the request's negative prompt (empty when unset)."""
        negative_prompt = config.get("negative_prompt")
        return {"negative_prompt": negative_prompt} if negative_prompt else {}

    def _save_image_with_metadata(
        self,
        image: Image.Image,
        output_path: str,
        prompt: str,
        **metadata_kwargs,
    ) -> None:
        """Save PIL Image to PNG with embedded text metadata."""
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("prompt", str(prompt))
        for key, value in metadata_kwargs.items():
            if value is not None:
                pnginfo.add_text(key, str(value))
        image.save(output_path, pnginfo=pnginfo)

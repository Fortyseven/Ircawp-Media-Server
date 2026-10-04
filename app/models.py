"""Pydantic models for the OpenAI-compatible images API."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field


InferenceSteps = Annotated[int, Field(ge=1, le=100)]


# ── Request Models ──────────────────────────────────────────────


class ImageRef(BaseModel):
    """A reference to an input image (base64 data URL)."""

    image_url: Optional[str] = None

    def __bool__(self):
        return self.image_url is not None


class ImageGenerationRequest(BaseModel):
    """Request body for POST /images/generations."""

    request_id: Optional[str] = Field(
        None, description="Optional identifier used to cancel this request."
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32000,
        description="Text description of the desired image(s).",
    )
    model: Optional[str] = Field(
        None,
        description="Backend/model to use (e.g. 'flux2klein'). Defaults to server default.",
    )
    steps: Optional[InferenceSteps] = Field(
        None, description="Number of inference steps (1-100)."
    )
    n: int = Field(1, ge=1, le=4, description="Number of images to generate (1-4).")
    size: Optional[str] = Field(
        None, description="Image size as 'WIDTHxHEIGHT', e.g. '1024x1024'."
    )
    output_size: Optional[int] = Field(
        None,
        ge=256,
        le=4096,
        description="Maximum output edge in pixels, preserving the selected aspect ratio.",
    )
    true_cfg_scale: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Optional classifier-free guidance scale, forwarded to the "
            "backend's native guidance parameter (model default applies when "
            "omitted). Experimental on CFG-free distillations (hyper_sdxl, "
            "zimageturbo)."
        ),
    )
    seed: Optional[int] = Field(
        None,
        ge=0,
        le=4294967295,
        description="Optional deterministic generation seed (all backends).",
    )
    quality: Optional[Literal["standard", "hd", "low", "medium", "high", "auto"]] = (
        Field(
            None, description="Image quality. Maps to remaster flag for our backends."
        )
    )
    lora: Optional[str] = Field(
        None,
        description="Optional LoRA adapter name for backends that support LoRA "
        "(e.g. qwenimage21). 'none' disables the backend's default LoRA.",
    )
    lora_scale: Optional[float] = Field(
        None,
        ge=0.0,
        description="Optional LoRA scale; overrides the adapter's configured scale.",
    )
    response_format: Optional[Literal["url", "b64_json"]] = Field(
        None, description="Response format. We always return b64_json."
    )
    user: Optional[str] = Field(None, description="End-user identifier (ignored).")
    verbose: Optional[bool] = Field(
        False,
        description="If true, include the full prompt in server logs. Default: false (privacy-first).",
    )


class ImageEditRequest(BaseModel):
    """Request body for POST /images/edits."""

    request_id: Optional[str] = Field(
        None, description="Optional identifier used to cancel this request."
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32000,
        description="Text description of the desired edit.",
    )
    images: list[ImageRef] = Field(
        ..., min_length=1, description="Input image(s) to edit."
    )
    model: Optional[str] = Field(None, description="Backend/model to use.")
    steps: Optional[InferenceSteps] = Field(
        None, description="Number of inference steps (1-100)."
    )
    n: int = Field(
        1, ge=1, le=4, description="Number of edited images to generate (1-4)."
    )
    size: Optional[str] = Field(
        None, description="Output image size as 'WIDTHxHEIGHT'."
    )
    output_size: Optional[int] = Field(
        None,
        ge=256,
        le=4096,
        description=(
            "Maximum output edge in pixels, preserving the source or selected "
            "aspect ratio. Defaults to the input image's max edge when omitted."
        ),
    )
    true_cfg_scale: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Optional classifier-free guidance scale, forwarded to the "
            "backend's native guidance parameter (model default applies when "
            "omitted). Experimental on CFG-free distillations (hyper_sdxl, "
            "zimageturbo)."
        ),
    )
    seed: Optional[int] = Field(
        None,
        ge=0,
        le=4294967295,
        description="Optional deterministic generation seed (all backends).",
    )
    quality: Optional[Literal["standard", "hd", "low", "medium", "high", "auto"]] = (
        Field(None, description="Image quality.")
    )
    lora: Optional[str] = Field(
        None,
        description="Optional LoRA adapter name for backends that support LoRA "
        "(e.g. qwenimage21). 'none' disables the backend's default LoRA.",
    )
    lora_scale: Optional[float] = Field(
        None,
        ge=0.0,
        description="Optional LoRA scale; overrides the adapter's configured scale.",
    )
    input_fidelity: Optional[Literal["high", "low"]] = Field(
        None, description="Fidelity to original input."
    )
    mask: Optional[ImageRef] = Field(None, description="Mask image for inpainting.")
    user: Optional[str] = Field(None, description="End-user identifier (ignored).")
    verbose: Optional[bool] = Field(
        False,
        description="If true, include the full prompt in server logs. Default: false (privacy-first).",
    )


class PromptRewriteRequest(BaseModel):
    """Request body for POST /prompt/rewrite.

    The server proxies this to the configured LLM endpoint (config.yml
    `llm` section) and returns the LLM's completion text.
    """

    mode: Literal["generate", "edit", "describe"] = Field(
        "generate",
        description=(
            "generate: rewrite a text-to-image prompt · edit: rewrite an "
            "image-edit instruction (requires images) · describe: write a "
            "prompt from the attached images (no user text)."
        ),
    )
    prompt: Optional[str] = Field(
        None, max_length=32000, description="User prompt text."
    )
    images: list[ImageRef] = Field(
        default_factory=list, description="Input image(s) (base64 data URLs)."
    )


# ── Response Models ─────────────────────────────────────────────


class Image(BaseModel):
    """A single generated/edited image in the response."""

    b64_json: Optional[str] = Field(None, description="Base64-encoded image data.")
    url: Optional[str] = Field(
        None, description="URL of the generated image (not used)."
    )
    revised_prompt: Optional[str] = Field(
        None, description="Revised prompt used for generation."
    )


class ImagesResponse(BaseModel):
    """Response body for image generation/edit endpoints."""

    created: int = Field(
        ..., description="Unix timestamp (seconds) when the response was created."
    )
    data: list[Image] = Field(
        default_factory=list, description="List of generated images."
    )


# ── Size Parsing ────────────────────────────────────────────────


def parse_size(size_str: Optional[str]) -> tuple[int, int] | None:
    """Parse a 'WIDTHxHEIGHT' string into (width, height) ints.

    Returns None if size_str is None or invalid.
    """
    if not size_str:
        return None

    parts = size_str.split("x")
    if len(parts) != 2:
        return None

    try:
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            return None
        return width, height
    except ValueError:
        return None


def ensure_divisible_by_16(width: int, height: int) -> tuple[int, int]:
    """Round dimensions to nearest multiple of 16 (required by diffusion models)."""
    return round(width / 16) * 16, round(height / 16) * 16

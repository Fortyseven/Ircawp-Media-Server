"""
ircawp media-server: image generation HTTP service.

OpenAI-compatible API:
  POST /images/generations  — text-to-image
  POST /images/edits        — image editing
  POST /prompt/rewrite      — LLM proxy (prompt rewrite / describe images)

LLM inference (prompt rewriting, describe-from-images) is proxied through
the server to the endpoint configured in config.yml (`llm` section), so the
browser never holds LLM credentials or talks to the LLM directly.
"""

from __future__ import annotations

import asyncio
import base64
import httpx
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rich.console import Console

from app.backends.MediaBackend import GenerationCancelled
from app.models import (
    Image,
    ImageEditRequest,
    ImageGenerationRequest,
    ImagesResponse,
    PromptRewriteRequest,
    ensure_divisible_by_16,
    parse_size,
)

console = Console()


def _expand_env(value):
    """Expand `${VAR}` / `${VAR:-default}` references in a config string.

    A `${...}` with no matching close brace, or a missing variable without a
    default, expands to the empty string. Unrelated text passes through.
    """
    if not isinstance(value, str):
        return value

    out = []
    i = 0
    while True:
        start = value.find("${", i)
        if start == -1:
            break
        out.append(value[i:start])
        end = value.find("}", start + 2)
        if end == -1:
            out.append(value[start:])
            return "".join(out)
        name, sep, default = value[start + 2 : end].partition(":-")
        env_val = os.environ.get(name.strip())
        out.append(env_val if env_val else (default if sep else ""))
        i = end + 1
    out.append(value[i:])
    return "".join(out)


def _expand_env_recursive(node):
    if isinstance(node, dict):
        return {k: _expand_env_recursive(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env_recursive(v) for v in node]
    return _expand_env(node)


def load_config(path: str = "config.yml") -> dict:
    config_path = Path(__file__).parent.parent / path
    if not config_path.is_file():
        return {}
    with open(config_path) as f:
        return _expand_env_recursive(yaml.safe_load(f) or {})


CONFIG = load_config()
SERVER_CONFIG = CONFIG.get("server", {})
DEFAULT_BACKEND = CONFIG.get("backend", "flux2klein")

def _llm_config(config: dict) -> dict:
    """LLM proxy settings with environment-variable fallbacks.

    endpoint / api_key fall back to OPENAI_API_BASE and OPENAI_API_KEY when
    unset (or set to an empty placeholder) in config.yml, so a deployment
    with only those env vars needs no `llm` section at all.
    """
    llm = dict(config.get("llm", {}) or {})
    if not (llm.get("endpoint") or "").strip():
        llm["endpoint"] = (os.environ.get("OPENAI_API_BASE") or "").strip()
    if not (llm.get("api_key") or "").strip():
        llm["api_key"] = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return llm


# LLM proxy settings — all LLM inference (prompt rewriting, prompt-from-images)
# is routed through the backend to this OpenAI-compatible chat endpoint, so the
# browser never talks to the LLM directly (no client-side TLS issues).
LLM_CONFIG = _llm_config(CONFIG)

# Temporary directory for generated images (cleaned up on shutdown)
_TEMP_DIR = Path(tempfile.mkdtemp())


def _new_temp_file(suffix=".png") -> Path:
    """Create a unique temp file path for a generated image."""
    return _TEMP_DIR / f"img_{time.time_ns()}{suffix}"


# Cache of backend instances (keeps models in memory across requests)
_backend_cache = {}
_active_cancellations: dict[str, Event] = {}
_active_progress: dict[str, dict] = {}


def _register_cancellation(request_id: str | None) -> Event | None:
    if request_id is None:
        return None
    cancellation_event = Event()
    _active_cancellations[request_id] = cancellation_event
    return cancellation_event


def _unregister_cancellation(
    request_id: str | None, cancellation_event: Event | None
) -> None:
    if (
        request_id is not None
        and cancellation_event is not None
        and _active_cancellations.get(request_id) is cancellation_event
    ):
        del _active_cancellations[request_id]


def _register_progress(request_id: str | None, n: int) -> dict | None:
    if request_id is None:
        return None
    progress_state = {"n": n, "image_index": 0, "step": 0, "total_steps": 0}
    _active_progress[request_id] = progress_state
    return progress_state


def _unregister_progress(request_id: str | None) -> None:
    if request_id is not None:
        _active_progress.pop(request_id, None)


def get_backend(backend_id: str):
    """Lazy-load a backend by ID, caching the instance so the model stays in memory."""
    if backend_id in _backend_cache:
        return _backend_cache[backend_id]

    import importlib

    try:
        module = importlib.import_module(f"app.backends.{backend_id}")
    except ModuleNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Backend '{backend_id}' not found: {e}",
        )

    backend_class = getattr(module, backend_id, None)
    if backend_class is None:
        raise HTTPException(
            status_code=400,
            detail=f"Backend module '{backend_id}' has no '{backend_id}' class",
        )

    backend_config = CONFIG.get("backends", {}).get(backend_id, {})
    instance = backend_class(backend_config)
    _backend_cache[backend_id] = instance
    return instance


def _image_to_response(image_path: str, final_prompt: str | None = None) -> Image:
    """Convert a generated image file to an Image response object."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    img = Image(
        b64_json=image_b64,
    )
    if final_prompt:
        img.revised_prompt = final_prompt
    return img


# Per-backend capabilities — the single source of truth for which knobs each
# model honors, advertised to clients via /backends. `cfg` is the backend
# config key the request's true_cfg_scale is forwarded as (absent: the model
# has no guidance knob); `cfg_experimental` marks CFG-free distillations whose
# pipeline accepts guidance but was never trained with it; `seed` honors a
# deterministic generation seed.
BACKEND_CAPABILITIES = {
    "qwenimage21": {"cfg": "true_cfg_scale", "seed": True},
    "flux2klein": {"cfg": "guidance_scale", "seed": True},
    "sd15": {"cfg": "guidance_scale", "seed": True},
    "sdxs": {"cfg": "guidance_scale", "seed": True},
    "hyper_sdxl": {"cfg": "guidance_scale", "cfg_experimental": True, "seed": True},
    "zimageturbo": {"cfg": "guidance_scale", "cfg_experimental": True, "seed": True},
}


def _backend_capabilities(backend_id: str) -> dict:
    """Client-facing capability flags for one backend (see BACKEND_CAPABILITIES)."""
    caps = BACKEND_CAPABILITIES.get(backend_id, {})
    return {
        "cfg": "cfg" in caps,
        "cfg_experimental": caps.get("cfg_experimental", False),
        "seed": caps.get("seed", False),
    }


def _request_extra(req) -> dict:
    """Optional per-request params (steps, lora, lora_scale) for backend config."""
    extra: dict = {}
    if req.steps is not None:
        extra["steps"] = req.steps
    if req.lora is not None:
        extra["lora"] = req.lora
    if req.lora_scale is not None:
        extra["lora_scale"] = req.lora_scale
    return extra


def _build_backend_config(
    *,
    backend_id: str,
    size: Optional[str],
    output_size: Optional[int],
    true_cfg_scale: Optional[float],
    seed: Optional[int],
    quality: Optional[str],
    batch_id=None,
    output_file: Optional[str] = None,
    extra: dict | None = None,
) -> dict:
    """Build the config dict passed to backend.execute() from request params.

    Per-backend settings from config.yml (backends.<id>) form the base;
    request-derived values (size, quality, output_file) override them.
    """
    config = dict(CONFIG.get("backends", {}).get(backend_id, {}))
    # `lora` in the base config holds adapter *definitions* (loaded by the
    # backend at startup); per-request `lora` carries an adapter *name*.
    config.pop("lora", None)

    if extra:
        config.update(extra)

    if output_file:
        config["output_file"] = output_file

    if output_size is not None:
        config["max_output_size"] = output_size

    caps = BACKEND_CAPABILITIES.get(backend_id, {})
    if true_cfg_scale is not None and caps.get("cfg"):
        config[caps["cfg"]] = true_cfg_scale
    if seed is not None and caps.get("seed"):
        config["seed"] = seed

    # Parse size into width/height if provided
    parsed = parse_size(size)
    if parsed:
        width, height = ensure_divisible_by_16(*parsed)
        config["width"] = width
        config["height"] = height
    elif size:
        # Invalid size string — let the backend use its default
        pass

    # Map quality to remaster flag
    if quality == "high":
        config["remaster"] = True
    elif quality == "low":
        config["remaster"] = False
    # "standard", "medium", "auto", "hd" — don't set remaster

    if batch_id is not None:
        config["batch_id"] = batch_id

    return config


def _input_max_edge(media_paths: list[str]) -> Optional[int]:
    """Default max output edge from the first readable input image.

    Rounded to the nearest multiple of 16 and clamped to the API's allowed
    [256, 4096] range. Returns None if no dimensions can be read, in which
    case the backend's own default applies.
    """
    from PIL import Image

    for path in media_paths:
        try:
            with Image.open(path) as img:
                longest = max(img.width, img.height)
        except Exception:
            continue
        edge = round(longest / 16) * 16
        return min(4096, max(256, edge))
    return None


def _decode_image_url(image_url: str) -> bytes:
    """Decode a data URL or return raw bytes for a base64 string."""
    if image_url.startswith("data:"):
        # data:image/png;base64,...
        header, b64_data = image_url.split(",", 1)
        return base64.b64decode(b64_data)
    elif image_url.startswith("http://") or image_url.startswith("https://"):
        raise HTTPException(
            status_code=400,
            detail="External URLs are not supported for image input. Use base64 data URLs.",
        )
    else:
        # Assume it's a raw base64 string
        return base64.b64decode(image_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ssl_cert = SERVER_CONFIG.get("ssl_cert")
    ssl_key = SERVER_CONFIG.get("ssl_key")
    scheme = "https" if (ssl_cert and ssl_key) else "http"

    console.log("[green]Media server starting")
    console.log(f"  backend: {DEFAULT_BACKEND}")
    console.log(f"  temp_dir: {_TEMP_DIR}")
    console.log(
        f"  {scheme}://{SERVER_CONFIG.get('host', '0.0.0.0')}:{SERVER_CONFIG.get('port', 8100)}"
    )
    yield
    console.log("[yellow]Media server shutting down")
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)


app = FastAPI(
    title="ircawp Media Server",
    description="OpenAI-compatible image generation service for ircawp bot",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    #    allow_origins=["http://localhost:5173", "https://fortyseven.github.io/chit-v2/"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ──────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "backend": DEFAULT_BACKEND}


@app.get("/backends")
async def backends():
    configured = sorted((CONFIG.get("backends") or {}).keys())
    return {
        "default": DEFAULT_BACKEND,
        "backends": configured,
        "capabilities": {
            backend_id: _backend_capabilities(backend_id)
            for backend_id in configured
        },
    }


@app.post("/backends/unload")
async def unload_backends():
    """Unload all loaded backends, freeing the model(s) from memory.

    The next generation will lazily re-load the backend. Rejects with 409
    if a generation is still in flight for the loaded backend(s).
    """
    if not _backend_cache:
        return {"unloaded": [], "message": "no backends loaded"}
    if _active_progress:
        raise HTTPException(
            status_code=409,
            detail="a generation is in flight — cancel it first",
        )

    unloaded = []
    for backend_id in list(_backend_cache):
        backend = _backend_cache.pop(backend_id)
        try:
            backend.dispose()
        except Exception as e:
            console.log(f"[yellow]Backend '{backend_id}' dispose warning: {e}")
        unloaded.append(backend_id)
    return {"unloaded": unloaded}


# ── LLM proxy ───────────────────────────────────────────────────
# All LLM inference (prompt rewriting, describe-images) is routed through
# here so the browser never talks to the LLM endpoint directly. Credentials
# live in config.yml (`llm` section), never in the frontend.


async def _llm_chat_completions(messages: list[dict]) -> str:
    """Call the configured OpenAI-compatible chat endpoint, return completion text."""
    endpoint = (LLM_CONFIG.get("endpoint") or "").strip()
    api_key = (LLM_CONFIG.get("api_key") or "").strip()
    if not endpoint:
        raise HTTPException(
            status_code=503,
            detail="LLM endpoint is not configured (config.yml: llm.endpoint)",
        )
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM API key is not configured (config.yml: llm.api_key)",
        )

    completion_url = (
        endpoint
        if endpoint.endswith("/chat/completions")
        else f"{endpoint.rstrip('/')}/chat/completions"
    )
    body = {"messages": messages}
    model = (LLM_CONFIG.get("model") or "").strip()
    if model:
        body["model"] = model

    verify = LLM_CONFIG.get("ssl_verify", True)
    timeout = LLM_CONFIG.get("timeout", 300)
    try:
        async with httpx.AsyncClient(
            verify=verify,
            timeout=httpx.Timeout(float(timeout)),
        ) as client:
            response = await client.post(
                completion_url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
    except httpx.HTTPError as e:
        console.log(f"[red]LLM request to {completion_url} failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")

    try:
        data = response.json()
    except (httpx.DecodingError, ValueError) as e:
        raise HTTPException(
            status_code=502, detail=f"LLM response was not JSON: {e}"
        )
    if response.status_code >= 400:
        detail = (
            data.get("detail")
            if isinstance(data, dict)
            else None
        ) or str(data)
        raise HTTPException(
            status_code=response.status_code,
            detail=f"LLM endpoint error ({response.status_code}): {detail}",
        )

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(
            status_code=502, detail=f"LLM response was malformed: {e}"
        )


@app.post("/prompt/rewrite")
async def prompt_rewrite(req: PromptRewriteRequest):
    """Proxy an LLM call (prompt rewrite / describe-images) to the
    configured endpoint and return the completion text."""
    from app.prompt_prompts import system_prompt_for

    if req.mode == "describe" and not req.images:
        raise HTTPException(
            status_code=400, detail="mode 'describe' requires at least one image"
        )
    if req.mode in ("generate", "edit") and not (req.prompt or "").strip():
        raise HTTPException(
            status_code=400, detail=f"mode '{req.mode}' requires a prompt"
        )
    if req.mode == "edit" and not req.images:
        raise HTTPException(
            status_code=400, detail="mode 'edit' requires at least one image"
        )

    # Build the user message: text (if any) + images in upload order,
    # matching the content shape the frontend used to send directly.
    user_content: list[dict] = []
    if req.mode != "describe" and (req.prompt or "").strip():
        user_content.append({"type": "text", "text": req.prompt.strip()})
    user_content.extend(
        {"type": "image_url", "image_url": {"url": img.image_url}}
        for img in req.images
        if img.image_url
    )

    messages = [
        {"role": "system", "content": system_prompt_for(req.mode)},
        {"role": "user", "content": user_content},
    ]
    text = await _llm_chat_completions(messages)
    return {"prompt": text}


@app.post("/images/cancellations/{request_id}", status_code=202)
async def cancel_image_request(request_id: str):
    cancellation_event = _active_cancellations.get(request_id)
    if cancellation_event is None:
        raise HTTPException(status_code=404, detail="Request not found")
    cancellation_event.set()
    return {"request_id": request_id, "status": "cancelling"}


@app.get("/images/progress/{request_id}")
async def get_image_progress(request_id: str):
    progress_state = _active_progress.get(request_id)
    if progress_state is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return {"request_id": request_id, **progress_state}


# ── POST /images/generations ────────────────────────────────────


@app.post("/images/generations", response_model=ImagesResponse)
async def images_generations(req: ImageGenerationRequest) -> ImagesResponse:
    """Create images from a text prompt (OpenAI-compatible)."""
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    backend_id = req.model or DEFAULT_BACKEND
    n = req.n

    # Validate n cap
    if n > 4:
        raise HTTPException(status_code=400, detail="n must be between 1 and 4")

    cancellation_event = _register_cancellation(req.request_id)
    progress_state = _register_progress(req.request_id, n)
    try:
        try:
            backend = get_backend(backend_id)
        except HTTPException:
            raise
        except Exception as e:
            console.log(f"[red]Backend '{backend_id}' failed to load: {e}")
            raise HTTPException(status_code=500, detail=f"Backend load failed: {e}")

        results = []

        for i in range(n):
            batch_id = i if n > 1 else None
            output_file = str(_new_temp_file())
            config = _build_backend_config(
                backend_id=backend_id,
                size=req.size,
                output_size=req.output_size,
                true_cfg_scale=req.true_cfg_scale,
                seed=req.seed,
                quality=req.quality,
                batch_id=batch_id,
                output_file=output_file,
                extra=_request_extra(req),
            )
            config["cancellation_event"] = cancellation_event
            if progress_state is not None:
                progress_state.update(image_index=i, step=0, total_steps=0)
            config["progress_state"] = progress_state

            console.log(
                f"[cyan]Generating ({i + 1}/{n}) with {backend_id}"
                + (f": {req.prompt}" if req.verbose else "")
            )

            try:
                result = await asyncio.to_thread(
                    backend.execute,
                    prompt=req.prompt.strip(),
                    config=config,
                    media=[],
                )

                # Handle both (path, prompt) tuple and single path return
                if isinstance(result, tuple):
                    image_path, final_prompt = result
                else:
                    image_path = result
                    final_prompt = None

                console.log(f"[green]Generated ({i + 1}/{n})")

                img = _image_to_response(image_path, final_prompt)
                results.append(img)

                # Remove temp file — image is already encoded in the response
                Path(image_path).unlink(missing_ok=True)

            except GenerationCancelled:
                Path(output_file).unlink(missing_ok=True)
                raise HTTPException(status_code=409, detail="Generation cancelled")
            except HTTPException:
                raise
            except Exception as e:
                console.log(f"[red]Generation ({i + 1}/{n}) failed: {e}")
                Path(output_file).unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail=str(e))

        return ImagesResponse(
            created=int(time.time()),
            data=results,
        )
    finally:
        _unregister_cancellation(req.request_id, cancellation_event)
        _unregister_progress(req.request_id)


# ── POST /images/edits ──────────────────────────────────────────


@app.post("/images/edits", response_model=ImagesResponse)
async def images_edits(req: ImageEditRequest) -> ImagesResponse:
    """Create edited/extended images from input images + prompt (OpenAI-compatible)."""
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    if not req.images:
        raise HTTPException(
            status_code=400, detail="At least one input image is required"
        )

    backend_id = req.model or DEFAULT_BACKEND
    n = req.n

    if n > 4:
        raise HTTPException(status_code=400, detail="n must be between 1 and 4")

    cancellation_event = _register_cancellation(req.request_id)
    progress_state = _register_progress(req.request_id, n)
    # Decode input images to temp files
    temp_dir = Path(tempfile.mkdtemp())
    temp_media_paths = []
    try:
        for img_ref in req.images:
            if not img_ref.image_url:
                continue

            image_bytes = _decode_image_url(img_ref.image_url)
            temp_path = temp_dir / f"input_{len(temp_media_paths)}.png"
            temp_path.write_bytes(image_bytes)
            temp_media_paths.append(str(temp_path))

        if not temp_media_paths:
            raise HTTPException(
                status_code=400, detail="No valid input images provided"
            )

        # Blank output size: match the input image's native longest edge.
        output_size = req.output_size
        if output_size is None:
            output_size = _input_max_edge(temp_media_paths)

        try:
            backend = get_backend(backend_id)
        except HTTPException:
            raise
        except Exception as e:
            console.log(f"[red]Backend '{backend_id}' failed to load: {e}")
            raise HTTPException(status_code=500, detail=f"Backend load failed: {e}")

        results = []

        for i in range(n):
            batch_id = i if n > 1 else None
            output_file = str(_new_temp_file())
            config = _build_backend_config(
                backend_id=backend_id,
                size=req.size,
                output_size=output_size,
                true_cfg_scale=req.true_cfg_scale,
                seed=req.seed,
                quality=req.quality,
                batch_id=batch_id,
                output_file=output_file,
                extra=_request_extra(req),
            )
            config["cancellation_event"] = cancellation_event
            if progress_state is not None:
                progress_state.update(image_index=i, step=0, total_steps=0)
            config["progress_state"] = progress_state

            console.log(
                f"[cyan]Editing ({i + 1}/{n}) with {backend_id}"
                + (f": {req.prompt}" if req.verbose else "")
            )

            try:
                result = await asyncio.to_thread(
                    backend.execute,
                    prompt=req.prompt.strip(),
                    config=config,
                    media=temp_media_paths,
                )

                if isinstance(result, tuple):
                    image_path, final_prompt = result
                else:
                    image_path = result
                    final_prompt = None

                console.log(f"[green]Edited ({i + 1}/{n})")

                img = _image_to_response(image_path, final_prompt)
                results.append(img)

                # Remove temp file — image is already encoded in the response
                Path(image_path).unlink(missing_ok=True)

            except GenerationCancelled:
                Path(output_file).unlink(missing_ok=True)
                raise HTTPException(status_code=409, detail="Generation cancelled")
            except HTTPException:
                raise
            except Exception as e:
                console.log(f"[red]Edit ({i + 1}/{n}) failed: {e}")
                Path(output_file).unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail=str(e))

        return ImagesResponse(
            created=int(time.time()),
            data=results,
        )

    finally:
        # Cleanup input temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        _unregister_cancellation(req.request_id, cancellation_event)
        _unregister_progress(req.request_id)


# ── Static Frontend Mount ───────────────────────────────────────
# Serve the built Svelte frontend (media-server/frontend/dist) at `/`.
# Mounted AFTER all API routes so API routes take precedence.
_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
else:
    console.log(
        f"[yellow]Frontend dist not found at {_FRONTEND_DIST} — serving API only"
    )


# ── CLI Entry Point ─────────────────────────────────────────────


def start():
    """CLI entry point."""
    import uvicorn

    ssl_cert = SERVER_CONFIG.get("ssl_cert")
    ssl_key = SERVER_CONFIG.get("ssl_key")

    # Expand ~ in paths — YAML stores them literally, ssl module won't
    if ssl_cert:
        ssl_cert = os.path.expanduser(ssl_cert)
    if ssl_key:
        ssl_key = os.path.expanduser(ssl_key)

    config = uvicorn.Config(
        app=app,
        host=SERVER_CONFIG.get("host", "0.0.0.0"),
        port=SERVER_CONFIG.get("port", 8100),
        ssl_certfile=ssl_cert or None,
        ssl_keyfile=ssl_key or None,
    )

    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    start()

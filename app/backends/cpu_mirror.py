"""Offload without the slow GPU-to-CPU copy.

Diffusers' ``enable_model_cpu_offload()`` parks each component on the CPU with
``module.to("cpu")`` when another one needs the GPU, and again for *every*
component at the end of each pipeline call (``maybe_free_model_hooks`` rebuilds
the hooks, and attaching a hook parks its module).  Copying many GB from the GPU
into freshly allocated pageable memory costs 4-5 s per large component, against
1-2 s for the copy back to the GPU.

The weights never change, so the copy is unnecessary: keep the CPU tensors the
module had while parked as *masters*, let ``module.to("cuda")`` make GPU copies
as usual, and "offload" by putting the masters back, which drops the GPU copies
without a transfer.  Results are bit-identical; only the copy is skipped.

``Module.to`` also moves bitsandbytes' ``quant_state`` tensors *in place*, so
those are snapshotted and restored too.
"""

import torch
from torch import nn


class CpuMirror:
    """CPU masters of one module's parameters, buffers and bitsandbytes state.

    The module must be on the CPU when this is created.  ``offload()`` restores
    the masters (and so frees the GPU copies); it can be called any number of
    times.  Create a new mirror if the module's structure changes (e.g. a LoRA
    adds parameters); tensors the mirror does not know about are left alone.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self._params = []
        self._buffers = []
        self._states = []
        seen = set()
        for owner in module.modules():
            for key, param in owner._parameters.items():
                if param is None:
                    continue
                self._require_cpu(param, key)
                self._params.append((owner, key, param, param.data))
                self._capture_state(getattr(param, "quant_state", None), seen)
            for key, buffer in owner._buffers.items():
                if buffer is None:
                    continue
                self._require_cpu(buffer, key)
                self._buffers.append((owner, key, buffer))

    @staticmethod
    def _require_cpu(tensor, key):
        if tensor.device.type != "cpu":
            raise RuntimeError(f"CpuMirror needs the module on the CPU, but {key} is on {tensor.device}")

    def _capture_state(self, obj, seen):
        # Only bitsandbytes' own state objects; their tensor attributes are what `.to` mutates.
        if obj is None or id(obj) in seen or not type(obj).__module__.startswith("bitsandbytes"):
            return
        seen.add(id(obj))
        for attr, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                self._states.append((obj, attr, value))
            else:
                self._capture_state(value, seen)

    def offload(self) -> None:
        for owner, key, param, data in self._params:
            owner._parameters[key] = param
            param.data = data
        for owner, key, buffer in self._buffers:
            owner._buffers[key] = buffer
        for obj, attr, tensor in self._states:
            setattr(obj, attr, tensor)


def _mirror_or_none(module: nn.Module):
    try:
        return CpuMirror(module)
    except RuntimeError:
        return None  # not parked on the CPU: leave the stock behaviour for this module


class FastOffload:
    """CpuMirror for every component of a pipeline that uses ``enable_model_cpu_offload()``.

    ``maybe_free_model_hooks`` calls ``enable_model_cpu_offload`` again after each
    request, discarding the hooks (and anything patched onto them), so this wraps
    that method on the pipeline instance.  Each time it runs:

    1. the GPU copies are dropped via the current masters, so the stock
       ``module.to("cpu")`` that attaching a hook performs has nothing to copy;
    2. the stock method rebuilds the hooks;
    3. masters are captured afresh (cheap: no tensor is copied) so structure
       changes such as a LoRA being added are picked up, and the new hooks'
       ``init_hook`` is replaced by the mirror's ``offload``.

    Modules that are not parked on the CPU at capture time keep the stock
    behaviour.  Install it after the first ``enable_model_cpu_offload()``.
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self._mirrors = {}
        self._original = pipe.enable_model_cpu_offload
        self._capture_and_patch()
        pipe.enable_model_cpu_offload = self._enable

    def _enable(self, *args, **kwargs):
        self.free_all()
        result = self._original(*args, **kwargs)
        self._capture_and_patch()
        return result

    def free_all(self) -> None:
        for mirror in self._mirrors.values():
            mirror.offload()

    def _capture_and_patch(self) -> None:
        self._mirrors = {}
        for user_hook in getattr(self.pipe, "_all_hooks", None) or []:
            mirror = _mirror_or_none(user_hook.model)
            if mirror is None:
                continue
            self._mirrors[id(user_hook.model)] = mirror
            user_hook.hook.init_hook = lambda module, mirror=mirror: mirror.offload() or module

    def park(self, module: nn.Module) -> None:
        """Move one of the pipeline's modules to the CPU, freeing its GPU copy for free if mirrored."""
        mirror = self._mirrors.get(id(module))
        if mirror is not None:
            mirror.offload()
        else:
            module.to("cpu")

    @property
    def mirrored(self) -> int:
        return len(self._mirrors)


def install_fast_offload(pipe):
    """Return a FastOffload for a pipeline already using ``enable_model_cpu_offload()``, else None."""
    hooks = getattr(pipe, "_all_hooks", None)
    if not isinstance(hooks, (list, tuple)) or not hooks:
        return None
    return FastOffload(pipe)

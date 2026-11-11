"""GPU-free contract for the shared CPU-mirror offload and its backend wiring.

Module.to(torch.float64) / .to("meta") stand in for moving to the GPU:
they replace tensor data the way a device move does.  Storage pointers prove
whether a copy happened.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn

from app.backends import cpu_mirror


class FakeQuantState:
    """Stands in for bitsandbytes' QuantState: `.to` moves its tensors in place."""

    __module__ = "bitsandbytes.functional"

    def __init__(self, absmax, nested=None):
        self.absmax = absmax
        self.blocksize = 64
        self.state2 = nested

    def to(self, dtype):
        self.absmax = self.absmax.to(dtype)
        if self.state2 is not None:
            self.state2.to(dtype)


class CpuMirrorTests(unittest.TestCase):
    """`.to(float64)` stands in for the move to the GPU: it replaces tensor data like a device move."""

    def module(self):
        torch.manual_seed(0)
        module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
        module.register_buffer("scale", torch.arange(3.0))
        module.register_buffer("transient", torch.ones(2), persistent=False)
        return module

    def test_offload_restores_the_original_tensors_without_copying_back(self):
        module = self.module()
        original = {k: v for k, v in module.state_dict().items()}
        parameter = module[0].weight
        master = parameter.data
        mirror = cpu_mirror.CpuMirror(module)
        module.to(torch.float64)  # "onload": new tensors replace the masters
        self.assertEqual(module[0].weight.dtype, torch.float64)
        self.assertIsNot(module[0].weight.data, master)
        mirror.offload()
        self.assertIs(module[0].weight, parameter)
        # `.data` builds a new wrapper on every access; the same storage means nothing was copied.
        self.assertEqual(module[0].weight.data.data_ptr(), master.data_ptr())
        for key, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]), key)
        self.assertEqual(module.transient.dtype, torch.float32)
        self.assertEqual(module.scale.dtype, torch.float32)

    def test_can_cycle_onload_and_offload_repeatedly(self):
        module = self.module()
        expected = module(torch.ones(1, 4))
        mirror = cpu_mirror.CpuMirror(module)
        for _ in range(3):
            module.to(torch.float64)
            self.assertEqual(module(torch.ones(1, 4, dtype=torch.float64)).dtype, torch.float64)
            mirror.offload()
            self.assertTrue(torch.equal(module(torch.ones(1, 4)), expected))

    def test_quantization_state_mutated_in_place_is_restored(self):
        module = self.module()
        nested = FakeQuantState(torch.full((2,), 7.0))
        state = FakeQuantState(torch.arange(4.0), nested)
        module[0].weight.quant_state = state
        absmax, inner = state.absmax, nested.absmax
        mirror = cpu_mirror.CpuMirror(module)
        state.to(torch.float64)
        module.to(torch.float64)
        mirror.offload()
        self.assertIs(state.absmax, absmax)
        self.assertIs(nested.absmax, inner)
        self.assertEqual(state.absmax.dtype, torch.float32)

    def test_unrelated_objects_are_left_alone(self):
        class Other:
            marker = torch.zeros(1)

        module = self.module()
        module[0].weight.quant_state = Other()  # not a bitsandbytes class: ignored, not captured
        mirror = cpu_mirror.CpuMirror(module)
        mirror.offload()  # must not raise

    def test_module_must_be_on_the_cpu_when_captured(self):
        module = torch.nn.Linear(2, 2).to("meta")
        with self.assertRaisesRegex(RuntimeError, "CPU"):
            cpu_mirror.CpuMirror(module)


class FakeAccelerateHook:
    """accelerate's CpuOffload: attaching parks the module with a (copying) .to("cpu")."""

    def init_hook(self, module):
        return module.to(torch.float32)


class FakePipeline:
    """Rebuilds its hooks on every enable_model_cpu_offload(), like maybe_free_model_hooks does."""

    def __init__(self, *modules):
        self.modules = list(modules)
        self._all_hooks = []
        self.enable_calls = 0
        self.enable_model_cpu_offload()

    def enable_model_cpu_offload(self, device="cuda"):
        self.enable_calls += 1
        self._all_hooks = []
        for module in self.modules:
            hook = FakeAccelerateHook()
            self._all_hooks.append(SimpleNamespace(model=module, hook=hook))
            hook.init_hook(module)  # attaching a hook parks its module


def block():
    torch.manual_seed(1)
    module = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    module.register_buffer("scale", torch.arange(3.0))
    return module


def pointers(module):
    return {k: v.data_ptr() for k, v in module.state_dict().items()}


class FastOffloadTests(unittest.TestCase):
    def test_cycle_drops_the_gpu_copy_without_copying_back(self):
        module = block()
        before = pointers(module)
        pipe = FakePipeline(module)
        fast = cpu_mirror.install_fast_offload(pipe)
        self.assertEqual(fast.mirrored, 1)
        module.to(torch.float64)  # a request onloads
        self.assertNotEqual(pointers(module), before)
        pipe.enable_model_cpu_offload()  # end of the call (maybe_free_model_hooks)
        self.assertEqual(pointers(module), before)  # the very same storage: nothing was copied
        self.assertEqual(module[0].weight.dtype, torch.float32)

    def test_stock_behaviour_copies_which_is_what_this_avoids(self):
        module = block()
        before = pointers(module)
        pipe = FakePipeline(module)
        module.to(torch.float64)
        pipe.enable_model_cpu_offload()
        self.assertNotEqual(pointers(module), before)  # a new allocation per cycle

    def test_hooks_rebuilt_every_cycle_are_patched_again(self):
        module = block()
        pipe = FakePipeline(module)
        cpu_mirror.install_fast_offload(pipe)
        for _ in range(3):
            first_hook = pipe._all_hooks[0].hook
            self.assertIn("init_hook", vars(first_hook))  # instance attribute: patched
            module.to(torch.float64)
            before_free = pointers(module)
            pipe._all_hooks[0].hook.init_hook(module)  # evicted by the next component's pre_forward
            self.assertEqual(module[0].weight.dtype, torch.float32)
            self.assertNotEqual(pointers(module), before_free)
            pipe.enable_model_cpu_offload()

    def test_values_are_identical_after_many_cycles(self):
        module = block()
        expected = {k: v.clone() for k, v in module.state_dict().items()}
        pipe = FakePipeline(module)
        cpu_mirror.install_fast_offload(pipe)
        for _ in range(4):
            module.to(torch.float64)
            pipe.enable_model_cpu_offload()
        for key, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), key)

    def test_structure_added_between_cycles_is_picked_up_like_a_lora(self):
        module = block()
        pipe = FakePipeline(module)
        cpu_mirror.install_fast_offload(pipe)
        module.to(torch.float64)
        module[0].extra = nn.Parameter(torch.ones(2, dtype=torch.float64))  # injected while "on the GPU"
        pipe.enable_model_cpu_offload()
        self.assertEqual(module[0].extra.dtype, torch.float32)  # the stock copy handled the unknown tensor
        added = module[0].extra.data.data_ptr()
        module.to(torch.float64)
        pipe.enable_model_cpu_offload()
        self.assertEqual(module[0].extra.data.data_ptr(), added)  # now mirrored: no copy

    def test_module_not_parked_on_the_cpu_keeps_the_stock_path(self):
        parked, away = block(), block().to("meta")
        pipe = FakePipeline.__new__(FakePipeline)
        pipe.modules, pipe.enable_calls = [parked, away], 0
        pipe._all_hooks = [
            SimpleNamespace(model=parked, hook=FakeAccelerateHook()),
            SimpleNamespace(model=away, hook=FakeAccelerateHook()),
        ]
        pipe.enable_model_cpu_offload = lambda device="cuda": None
        fast = cpu_mirror.install_fast_offload(pipe)
        self.assertEqual(fast.mirrored, 1)
        self.assertNotIn("init_hook", vars(pipe._all_hooks[1].hook))

    def test_park_is_free_for_mirrored_modules_and_a_plain_move_otherwise(self):
        module, other = block(), block()
        pipe = FakePipeline(module)
        fast = cpu_mirror.install_fast_offload(pipe)
        before = pointers(module)
        module.to(torch.float64)
        fast.park(module)
        self.assertEqual(pointers(module), before)
        other.to(torch.float64)
        fast.park(other)  # not part of the pipeline: ordinary .to("cpu")
        self.assertEqual(other[0].weight.device.type, "cpu")

    def test_pipelines_without_real_hooks_are_left_alone(self):
        self.assertIsNone(cpu_mirror.install_fast_offload(Mock()))
        self.assertIsNone(cpu_mirror.install_fast_offload(SimpleNamespace(_all_hooks=[])))
        self.assertIsNone(cpu_mirror.install_fast_offload(SimpleNamespace()))


class MingEncoderMirrorTests(unittest.TestCase):
    def encoder(self, **overrides):
        from app.backends.ming_loader import MingEncoder

        torch.manual_seed(2)
        host = SimpleNamespace(query_tokens_dict=nn.ParameterDict({"a": nn.Parameter(torch.ones(2))}))
        parts = [nn.Linear(3, 3) for _ in range(5)]
        encoder = MingEncoder(*parts, host, None, None, None, None, vision=nn.Linear(2, 2), linear_proj=nn.Linear(2, 2))
        for name, value in overrides.items():
            setattr(encoder, name, value)
        return encoder

    def test_default_is_the_stock_move(self):
        encoder = self.encoder()
        self.assertFalse(encoder.fast_offload)
        self.assertIsNone(encoder._mirrors)

    def test_fast_offload_frees_the_gpu_copy_with_no_transfer(self):
        encoder = self.encoder(fast_offload=True)
        before = [pointers(p) for p in encoder._parts()]
        encoder.to("meta")  # stands in for cuda
        self.assertEqual(encoder.llm.weight.device.type, "meta")
        self.assertIsNotNone(encoder._mirrors)
        encoder.to("cpu")
        self.assertEqual([pointers(p) for p in encoder._parts()], before)
        for _ in range(2):  # and again
            encoder.to("meta")
            encoder.to("cpu")
        self.assertEqual([pointers(p) for p in encoder._parts()], before)
        self.assertEqual(encoder.host.query_tokens_dict["a"].device.type, "cpu")

    def test_everything_the_stock_move_reaches_is_mirrored(self):
        encoder = self.encoder(fast_offload=True)
        encoder.to("meta")
        for part in encoder._parts():
            self.assertTrue(all(p.device.type == "meta" for p in part.parameters()))
        self.assertEqual(len(encoder._mirrors), len(encoder._parts()))

    def test_falls_back_to_the_stock_move_if_not_parked_on_the_cpu(self):
        encoder = self.encoder(fast_offload=True)
        encoder.llm.to("meta")  # not on the CPU when the first move happens
        encoder.to("meta")
        self.assertFalse(encoder.fast_offload)
        self.assertIsNone(encoder._mirrors)


class BackendWiringTests(unittest.TestCase):
    def test_qwen_parks_its_vae_through_the_mirror_when_installed(self):
        from app.backends.qwenimage21 import qwenimage21

        backend = object.__new__(qwenimage21)
        module = nn.Linear(2, 2)
        backend.fast_offload = None
        backend._park(module)  # no mirror: ordinary .to("cpu")
        self.assertEqual(module.weight.device.type, "cpu")
        backend.fast_offload = SimpleNamespace(park=Mock())
        backend._park(module)
        backend.fast_offload.park.assert_called_once_with(module)


if __name__ == "__main__":
    unittest.main()

"""Per-request scheduler selection across backends and the API."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from fastapi import HTTPException

from app import main
from app.backends.MediaBackend import MediaBackend
from app.backends.schedulers import build_scheduler, scheduler_names


class RegistryTests(unittest.TestCase):
    def test_flow_euler_a_enables_stochastic_sampling_and_keeps_config(self):
        base = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        scheduler = build_scheduler("flow", "euler_a", base.config)
        self.assertIsInstance(scheduler, FlowMatchEulerDiscreteScheduler)
        self.assertTrue(scheduler.config.stochastic_sampling)
        self.assertTrue(scheduler.config.use_dynamic_shifting)
        self.assertFalse(base.config.stochastic_sampling)

    def test_sd_family_builds_expected_classes(self):
        base = EulerAncestralDiscreteScheduler().config
        self.assertIsInstance(
            build_scheduler("sd", "euler", base), EulerDiscreteScheduler
        )
        self.assertIsInstance(
            build_scheduler("sd", "euler_a", base), EulerAncestralDiscreteScheduler
        )
        dpm = build_scheduler("sd", "dpmpp_2m", base)
        self.assertIsInstance(dpm, DPMSolverMultistepScheduler)
        self.assertFalse(dpm.config.use_karras_sigmas)
        self.assertTrue(
            build_scheduler("sd", "dpmpp_2m_karras", base).config.use_karras_sigmas
        )

    def test_unknown_name_and_family_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown scheduler"):
            build_scheduler("flow", "dpmpp_2m", {})
        with self.assertRaisesRegex(ValueError, "Unknown scheduler"):
            build_scheduler(None, "euler_a", {})

    def test_names_exclude_implicit_default(self):
        self.assertEqual(scheduler_names("flow"), ["euler_a"])
        self.assertNotIn("default", scheduler_names("sd"))
        self.assertEqual(scheduler_names(None), [])


class ApplySchedulerTests(unittest.TestCase):
    def backend(self, family="flow"):
        class Backend(MediaBackend):
            scheduler_family = family

        backend = Backend()
        backend.pipe = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler())
        return backend

    def test_omitted_or_default_leaves_pipeline_untouched(self):
        backend = self.backend()
        original = backend.pipe.scheduler
        for config in ({}, {"scheduler": "default"}, {"scheduler": ""}):
            with self.subTest(config=config):
                self.assertIsNone(backend._apply_scheduler(config, 1))
                self.assertIs(backend.pipe.scheduler, original)

    def test_selected_scheduler_is_restored_by_default_on_next_request(self):
        backend = self.backend()
        original = backend.pipe.scheduler
        self.assertEqual(
            backend._apply_scheduler({"scheduler": "euler_a"}, 1), "euler_a"
        )
        self.assertTrue(backend.pipe.scheduler.config.stochastic_sampling)
        self.assertIsNone(backend._apply_scheduler({}, 1))
        self.assertIs(backend.pipe.scheduler, original)

    def test_each_request_derives_from_the_default_not_the_previous_choice(self):
        backend = self.backend("sd")
        backend.pipe.scheduler = EulerAncestralDiscreteScheduler()
        backend._apply_scheduler({"scheduler": "dpmpp_2m_karras"})
        backend._apply_scheduler({"scheduler": "euler"})
        self.assertIsInstance(backend.pipe.scheduler, EulerDiscreteScheduler)

    def test_flow_selection_seeds_global_rng_for_reproducible_noise(self):
        import torch

        backend = self.backend()
        backend._apply_scheduler({"scheduler": "euler_a"}, 123)
        first = torch.rand(1)
        backend._apply_scheduler({"scheduler": "euler_a"}, 123)
        self.assertTrue(torch.equal(first, torch.rand(1)))

    def test_backend_without_family_ignores_selection(self):
        backend = self.backend(None)
        original = backend.pipe.scheduler
        self.assertIsNone(backend._apply_scheduler({"scheduler": "euler_a"}, 1))
        self.assertIs(backend.pipe.scheduler, original)


class ApiTests(unittest.TestCase):
    def build(self, backend_id, **config):
        with patch.object(main, "CONFIG", {"backends": {backend_id: config}}):
            return main._build_backend_config(
                backend_id=backend_id,
                size=None,
                output_size=None,
                true_cfg_scale=None,
                seed=None,
                quality=None,
                extra={"scheduler": "euler_a"},
            )

    def test_capabilities_advertise_schedulers(self):
        self.assertEqual(
            main._backend_capabilities("qwenimage21")["schedulers"], ["euler_a"]
        )
        self.assertIn("dpmpp_2m", main._backend_capabilities("sd15")["schedulers"])
        for backend_id in ("qwenimage21turbo", "mossimage", "sdxs", "unknown"):
            with self.subTest(backend_id=backend_id):
                self.assertEqual(
                    main._backend_capabilities(backend_id)["schedulers"], []
                )

    def test_valid_scheduler_is_forwarded(self):
        self.assertEqual(self.build("qwenimage21")["scheduler"], "euler_a")

    def test_unknown_scheduler_is_a_400(self):
        with patch.object(main, "CONFIG", {"backends": {"sd15": {}}}):
            with self.assertRaises(HTTPException) as ctx:
                main._build_backend_config(
                    backend_id="sd15",
                    size=None,
                    output_size=None,
                    true_cfg_scale=None,
                    seed=None,
                    quality=None,
                    extra={"scheduler": "nope"},
                )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_unsupported_backend_drops_scheduler(self):
        self.assertNotIn("scheduler", self.build("qwenimage21turbo"))

    def build_negative(self, backend_id, negative_prompt):
        with patch.object(main, "CONFIG", {"backends": {backend_id: {}}}):
            return main._build_backend_config(
                backend_id=backend_id,
                size=None,
                output_size=None,
                true_cfg_scale=None,
                seed=None,
                quality=None,
                extra={"negative_prompt": negative_prompt},
            )

    def test_capabilities_advertise_negative_prompt(self):
        for backend_id in ("qwenimage21", "sd15", "fibo2"):
            with self.subTest(backend_id=backend_id):
                self.assertTrue(
                    main._backend_capabilities(backend_id)["negative_prompt"]
                )
        for backend_id in ("flux3klein", "fibo2editturbo", "unknown"):
            with self.subTest(backend_id=backend_id):
                self.assertFalse(
                    main._backend_capabilities(backend_id)["negative_prompt"]
                )

    def test_negative_prompt_is_forwarded_stripped(self):
        config = self.build_negative("sd15", "  blurry  ")
        self.assertEqual(config["negative_prompt"], "blurry")

    def test_empty_negative_prompt_is_dropped(self):
        self.assertNotIn("negative_prompt", self.build_negative("sd15", "   "))

    def test_unsupported_backend_drops_negative_prompt(self):
        self.assertNotIn("negative_prompt", self.build_negative("flux3klein", "blurry"))

    def test_declared_families_match_backend_classes(self):
        from app.backends.flux2klein import flux2klein
        from app.backends.qwenimage21 import qwenimage21
        from app.backends.qwenimage21turbo import qwenimage21turbo
        from app.backends.zimageturbo import zimageturbo

        classes = {
            "qwenimage21": qwenimage21,
            "qwenimage21turbo": qwenimage21turbo,
            "flux2klein": flux2klein,
            "zimageturbo": zimageturbo,
        }
        for backend_id, cls in classes.items():
            with self.subTest(backend_id=backend_id):
                declared = main.BACKEND_CAPABILITIES[backend_id].get("scheduler_family")
                self.assertEqual(cls.scheduler_family, declared)


if __name__ == "__main__":
    unittest.main()

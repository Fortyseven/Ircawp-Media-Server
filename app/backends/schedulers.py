"""Per-request scheduler choices, grouped by the model family they are valid for.

Families are not interchangeable: flow-matching models (Qwen-Image, FLUX.2,
Z-Image) break under epsilon/v-prediction schedulers such as EulerAncestral.
"""


def _flow_euler_a(config):
    from diffusers import FlowMatchEulerDiscreteScheduler

    # Re-noises to the next sigma each step: the flow-matching analogue of Euler A.
    return FlowMatchEulerDiscreteScheduler.from_config(config, stochastic_sampling=True)


def _sd_euler(config):
    from diffusers import EulerDiscreteScheduler

    return EulerDiscreteScheduler.from_config(config)


def _sd_euler_a(config):
    from diffusers import EulerAncestralDiscreteScheduler

    return EulerAncestralDiscreteScheduler.from_config(config)


def _sd_dpmpp_2m(config):
    from diffusers import DPMSolverMultistepScheduler

    return DPMSolverMultistepScheduler.from_config(config)


def _sd_dpmpp_2m_karras(config):
    from diffusers import DPMSolverMultistepScheduler

    return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


FAMILIES = {
    "flow": {"euler_a": _flow_euler_a},
    "sd": {
        "euler": _sd_euler,
        "euler_a": _sd_euler_a,
        "dpmpp_2m": _sd_dpmpp_2m,
        "dpmpp_2m_karras": _sd_dpmpp_2m_karras,
    },
}


def scheduler_names(family: str | None) -> list[str]:
    """Selectable scheduler names for a family (excluding the implicit default)."""
    return list(FAMILIES.get(family, {}))


def build_scheduler(family: str | None, name: str, base_config):
    """Build scheduler `name` from the pipeline's default scheduler config."""
    factories = FAMILIES.get(family, {})
    if name not in factories:
        raise ValueError(
            f"Unknown scheduler {name!r} (available: {', '.join(factories) or 'none'})"
        )
    return factories[name](base_config)

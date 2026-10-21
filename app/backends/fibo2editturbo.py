"""Fibo 2.0 Edit Turbo: edit-only, four steps, no classifier-free guidance."""

from .fibo2 import fibo2


BASE_MODEL = "briaai/Fibo-2.0-edit-turbo"
TURBO_STEPS = 4


class fibo2editturbo(fibo2):
    model_id = "fibo2editturbo"
    base_model = BASE_MODEL
    default_steps = TURBO_STEPS
    default_cfg = 1.0
    edit_only = True

    def _resolve_steps(self, config: dict) -> int:
        # Distilled for exactly four steps; other counts are normalized.
        return TURBO_STEPS

    def _resolve_cfg(self, config: dict) -> float:
        return 1.0

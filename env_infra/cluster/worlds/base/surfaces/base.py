"""Common surface scaffolding shared by all surface base classes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from cluster.schemas import Action, ActionKind, ControlType, Observation
from cluster.worlds.base.adapter import WorldAdapter


class SurfaceAdapter(WorldAdapter):
    """Base for surface adapters: validates the surface's obs/action dialect.

    Subclasses set:
      - ``surface``                -> matches ``capabilities.environment``
      - ``default_modalities``     -> the modality keys this surface emits
      - ``allowed_action_kinds``   -> the :class:`ActionKind` values accepted
                                       (CONTROL is always allowed)
    """

    surface: str = "unknown"
    default_modalities: Tuple[str, ...] = ()
    allowed_action_kinds: Tuple[str, ...] = (ActionKind.CONTROL.value,)

    # -- action vocabulary -------------------------------------------------
    def validate_action(self, action: Action) -> None:
        """Raise ValueError if ``action`` is not valid for this surface."""
        allowed = set(self.allowed_action_kinds) | {ActionKind.CONTROL.value}
        if action.kind not in allowed:
            raise ValueError(
                f"{type(self).__name__} ({self.surface}) does not accept action "
                f"kind {action.kind!r}; allowed: {sorted(allowed)}"
            )
        if action.kind == ActionKind.CONTROL.value and action.type not in (
            ControlType.DONE.value,
            ControlType.FAIL.value,
            ControlType.WAIT.value,
        ):
            raise ValueError(f"unknown control action type: {action.type!r}")

    # -- observation helpers ----------------------------------------------
    def make_observation(
        self,
        modalities: Dict[str, Any],
        *,
        step: int = 0,
        runtime: str | None = None,
        artifacts: Iterable[Dict[str, Any]] | None = None,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> Observation:
        """Build an Observation, recording surface + declared default modalities.

        Default-modality keys absent from ``modalities`` are filled with ``None``
        so consumers can distinguish "this surface does not produce X" from a
        missing key.
        """
        merged: Dict[str, Any] = {k: None for k in self.default_modalities}
        merged.update(modalities or {})
        metadata: Dict[str, Any] = {"surface": self.surface, "step": step}
        if runtime:
            metadata["runtime"] = runtime
        if extra_metadata:
            metadata.update(extra_metadata)
        return Observation(
            modalities=merged,
            artifacts=list(artifacts or []),
            metadata=metadata,
        )

    def get_info(self) -> Dict[str, Any]:
        return {
            "surface": self.surface,
            "default_modalities": list(self.default_modalities),
            "allowed_action_kinds": list(self.allowed_action_kinds),
        }

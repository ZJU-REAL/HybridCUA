"""Transport-neutral message envelopes for the EnvSession protocol.

These dataclasses are the *only* vocabulary that crosses the runner <-> master
<-> node <-> sidecar boundaries. They are deliberately wide enough to describe
desktop, mobile, web and remote/online environments, yet narrow enough that
every world adapter can implement them.

Design rules:
- ``Observation`` carries a ``modalities`` map; a world only fills the keys it
  supports (a desktop world fills ``screenshot``/``accessibility_tree``, a
  mobile world fills ``screenshot``/``ui_tree``/``device_state``, ...).
- ``Action`` is typed (``kind`` + ``type`` + ``payload``); ``DONE``/``FAIL``/
  ``WAIT`` are platform-level *control* actions shared by every surface.
- Nothing here imports a benchmark SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ActionKind(str, Enum):
    """The coarse family an action belongs to (orthogonal to surface)."""

    GUI = "gui"          # desktop/web pixel + keyboard interaction (pyautogui, click, type)
    DEVICE = "device"    # mobile device gestures (tap, swipe, key, adb)
    TOOL = "tool"        # tool / API / MCP calls (browser nav, api_call, cli)
    CONTROL = "control"  # episode control: DONE / FAIL / WAIT


class ControlType(str, Enum):
    """Platform-level control verbs, valid on every surface."""

    DONE = "DONE"
    FAIL = "FAIL"
    WAIT = "WAIT"


@dataclass
class Observation:
    """A single observation, expressed as a modality map.

    ``modalities`` keys are world-defined but drawn from a shared vocabulary,
    e.g. ``screenshot`` (base64 PNG str), ``accessibility_tree`` (xml str),
    ``ui_tree``, ``terminal``, ``dom``, ``device_state`` (dict), ``instruction``.
    A world only populates the modalities it actually produces.
    """

    modalities: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modalities": dict(self.modalities),
            "artifacts": list(self.artifacts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        data = data or {}
        return cls(
            modalities=dict(data.get("modalities", {}) or {}),
            artifacts=list(data.get("artifacts", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class Action:
    """A typed action emitted by an agent and executed by a world.

    Examples::

        Action(kind="gui",     type="pyautogui", payload={"command": "pyautogui.click(10, 20)"})
        Action(kind="device",  type="tap",       payload={"x": 420, "y": 860})
        Action(kind="control", type="DONE",      payload={})
        Action(kind="tool",    type="api_call",  payload={"name": "search", "arguments": {}})
    """

    kind: str
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """True if this action ends the episode (DONE/FAIL control verbs)."""
        return self.kind == ActionKind.CONTROL.value and self.type in (
            ControlType.DONE.value,
            ControlType.FAIL.value,
        )

    def is_control(self) -> bool:
        return self.kind == ActionKind.CONTROL.value

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "type": self.type, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        if data is None:
            raise ValueError("Action.from_dict received None")
        return cls(
            kind=str(data["kind"]),
            type=str(data["type"]),
            payload=dict(data.get("payload", {}) or {}),
        )

    # Convenience constructors for the shared control verbs ------------------
    @classmethod
    def done(cls) -> "Action":
        return cls(kind=ActionKind.CONTROL.value, type=ControlType.DONE.value)

    @classmethod
    def fail(cls) -> "Action":
        return cls(kind=ActionKind.CONTROL.value, type=ControlType.FAIL.value)

    @classmethod
    def wait(cls) -> "Action":
        return cls(kind=ActionKind.CONTROL.value, type=ControlType.WAIT.value)


@dataclass
class StepResponse:
    """Result of executing one action."""

    observation: Observation
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "done": self.done,
            "info": dict(self.info),
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepResponse":
        return cls(
            observation=Observation.from_dict(data.get("observation", {})),
            done=bool(data.get("done", False)),
            info=dict(data.get("info", {}) or {}),
            reward=float(data.get("reward", 0.0) or 0.0),
        )


@dataclass
class EvaluationResult:
    """Outcome of an evaluation, world-neutral."""

    score: float = 0.0
    success: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "success": self.success,
            "metrics": dict(self.metrics),
            "reason": self.reason,
            "artifacts": list(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        data = data or {}
        return cls(
            score=float(data.get("score", 0.0) or 0.0),
            success=bool(data.get("success", False)),
            metrics=dict(data.get("metrics", {}) or {}),
            reason=str(data.get("reason", "") or ""),
            artifacts=list(data.get("artifacts", []) or []),
        )



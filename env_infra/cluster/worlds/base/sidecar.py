"""Standard HTTP sidecar that exposes a WorldAdapter over EnvSession REST.

A world sidecar wraps one :class:`WorldAdapter` instance behind the stable
endpoints the node talks to::

    GET  /health     -> {"ok": bool, "healthy": bool}
    POST /reset      {task_payload}      -> {ok, observation}
    POST /step       {action}            -> {ok, ...StepResponse}
    POST /observe                        -> {ok, observation}
    POST /evaluate                       -> {ok, ...EvaluationResult}
    POST /close                          -> {ok}
    GET  /info                           -> {ok, info}

This module is benchmark-neutral: it only depends on the protocol envelopes and
the adapter contract. Concrete worlds ship a thin ``sidecar.py`` that builds
their adapter and calls :func:`create_sidecar_app`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from flask import Flask, jsonify, request

from cluster.schemas import Action
from cluster.worlds.base.adapter import WorldAdapter

logger = logging.getLogger("cluster.sidecar")


def create_sidecar_app(adapter: WorldAdapter, name: str = "world-sidecar") -> Flask:
    """Build a Flask app serving the EnvSession protocol for ``adapter``."""
    app = Flask(name)

    def _body() -> Dict[str, Any]:
        return request.get_json(force=True, silent=True) or {}

    @app.get("/health")
    def health():  # type: ignore[unused-ignore]
        try:
            healthy = bool(adapter.health_check())
            return jsonify({"ok": True, "healthy": healthy})
        except Exception as exc:  # noqa: BLE001 - sidecar must always answer
            logger.exception("health_check failed")
            return jsonify({"ok": False, "healthy": False, "error": str(exc)}), 500

    @app.post("/reset")
    def reset():  # type: ignore[unused-ignore]
        try:
            task_payload = _body().get("task_payload", {}) or {}
            obs = adapter.reset(task_payload)
            return jsonify({"ok": True, "observation": obs.to_dict()})
        except Exception as exc:  # noqa: BLE001
            logger.exception("reset failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/step")
    def step():  # type: ignore[unused-ignore]
        try:
            action = Action.from_dict(_body().get("action"))
            resp = adapter.step(action)
            return jsonify({"ok": True, **resp.to_dict()})
        except Exception as exc:  # noqa: BLE001
            logger.exception("step failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/observe")
    def observe():  # type: ignore[unused-ignore]
        try:
            obs = adapter.observe()
            return jsonify({"ok": True, "observation": obs.to_dict()})
        except Exception as exc:  # noqa: BLE001
            logger.exception("observe failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/evaluate")
    def evaluate():  # type: ignore[unused-ignore]
        try:
            result = adapter.evaluate()
            return jsonify({"ok": True, **result.to_dict()})
        except Exception as exc:  # noqa: BLE001
            logger.exception("evaluate failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/close")
    def close():  # type: ignore[unused-ignore]
        try:
            adapter.close()
            return jsonify({"ok": True})
        except Exception as exc:  # noqa: BLE001
            logger.exception("close failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/info")
    def info():  # type: ignore[unused-ignore]
        try:
            return jsonify({"ok": True, "info": adapter.get_info()})
        except Exception as exc:  # noqa: BLE001
            logger.exception("info failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    return app

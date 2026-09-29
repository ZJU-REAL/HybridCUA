"""``CGuiMcpAgent`` -- c-gui plus MCP tools, as a subclass rather than a fork.

Only three hooks are overridden, and each one is a seam c_gui already provides:

    _build_tools_def    -> patch the single tool def (action mode only)
    _build_system_prompt-> splice the '# MCP tools' section in
    _parse_response     -> recognise action=mcp (action mode only)

Everything else -- the OpenAI call, image processing, history folding, CLI-output
threading, the XML tool-call contract -- is inherited untouched. That is the point:
MCP_MODE=off must be byte-identical to a plain c-gui run so the A/B baseline is
real, and `bash` mode must not perturb the action space at all.

The episode's tool catalog is injected by the run loop via ``set_episode_tools``
before the first predict(), and held constant for the episode. It is deliberately
NOT refreshed per step: a system prompt that changes every step throws away the
vLLM prefix cache for the whole conversation, and a task's related_apps is known
up front anyway.

This module lives in OSWorld-MCP and imports c_gui from env_infra's OSWorld
checkout, so nothing under env_infra/ is modified.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from mm_agents.c_gui_mcp.prompt import (MCP_MODES, build_example, build_mcp_section,
                        patch_tool_def)

logger = logging.getLogger("desktopenv.c_gui_mcp_agent")

try:
    from mm_agents.c_gui.agent import CGuiAgent
except Exception as _exc:
    CGuiAgent = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None

__all__ = ["CGuiMcpAgent", "build_mcp_agent"]


if CGuiAgent is not None:

    class CGuiMcpAgent(CGuiAgent):
        """c-gui with an MCP tool catalog. ``mcp_mode`` in {off, bash, action}."""

        def __init__(self, *args, mcp_mode: str = "off", **kwargs):
            super().__init__(*args, **kwargs)
            if mcp_mode not in MCP_MODES:
                raise ValueError(f"mcp_mode must be one of {MCP_MODES}, got {mcp_mode!r}")
            self.mcp_mode = mcp_mode
            self.mcp_tools: List[Dict] = []
            self.mcp_section: str = ""
            self.mcp_example: str = ""

        def set_episode_tools(self, tools: Optional[Sequence[Dict]]) -> None:
            """Install this episode's catalog and pre-render its prompt section."""
            self.mcp_tools = list(tools or [])
            self.mcp_section = build_mcp_section(
                self.mcp_tools, self.mcp_mode, tool_name=self.tool_name
            )
            self.mcp_example = (build_example(self.mcp_mode, tool_name=self.tool_name)
                                 if self.mcp_tools else "")
            logger.info("mcp agent: mode=%s tools=%d section=%d chars example=%d chars",
                        self.mcp_mode, len(self.mcp_tools), len(self.mcp_section),
                        len(self.mcp_example))

        def reset(self, _logger=None, *args, **kwargs):
            """c_gui's reset clears history. Keep the catalog: the run loop sets it
            once per episode and reset() runs inside that same episode."""
            super().reset(_logger, *args, **kwargs)

        def _build_tools_def(self, processed_width: int, processed_height: int) -> List[Dict]:
            defs = super()._build_tools_def(processed_width, processed_height)
            if self.mcp_mode == "off" or not self.mcp_tools:
                return defs
            return [patch_tool_def(d, self.mcp_mode, self.mcp_tools) for d in defs]

        def _build_system_prompt(self, tools_def: List[Dict]) -> str:
            prompt = super()._build_system_prompt(tools_def)
            if self.mcp_section:
                prompt = splice_section(prompt, self.mcp_section)
            if self.mcp_example:
                prompt = splice_example(prompt, self.mcp_example)
            return prompt

        def _parse_response(self, response: str) -> Tuple[str, List[Dict]]:
            """Delegate to c_gui, then add the mcp actions it cannot know about.

            c_gui's parser drops unknown `action` values (agent.py: "unknown action
            -> skip"), so in action mode an `action=mcp` call would silently vanish
            and the turn would look empty. We re-scan the response for those and
            merge them back, preserving the model's ordering.
            """
            low_level, actions = super()._parse_response(response)
            if self.mcp_mode != "action":
                return low_level, actions

            mcp_actions = self.parse_mcp_actions(response)
            if not mcp_actions:
                return low_level, actions

            merged = actions + mcp_actions
            n_mcp = len(mcp_actions)
            suffix = f"{n_mcp} mcp call(s)"
            low_level = f"{low_level} + {suffix}" if low_level else suffix
            return low_level, merged

        def parse_mcp_actions(self, response: str) -> List[Dict]:
            """Extract {"kind":"mcp", ...} from `action=mcp` tool calls."""
            from mm_agents.c_gui.parser import split_tool_calls

            out: List[Dict] = []
            for _fn, params in split_tool_calls(response, allowed=(self.tool_name,)):
                if str(params.get("action") or "").strip().lower() != "mcp":
                    continue
                name = str(params.get("name") or "").strip()
                if not name:
                    logger.warning("mcp agent: action=mcp with no name; skipping")
                    continue
                out.append({"kind": "mcp", "name": name,
                            "params": coerce_params(params.get("params"))})
            return out

else:

    class CGuiMcpAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "mm_agents.c_gui.agent.CGuiAgent could not be imported "
                f"({_IMPORT_ERROR}). Add env_infra/OSWorld to sys.path and install "
                "its deps (openai, PIL, qwen_agent)."
            )


def splice_section(prompt: str, section: str) -> str:
    """Insert the MCP block before '# Response format'.

    That position keeps the tool-call contract (# Tools) above it and leaves the
    worked output examples as the last thing the model reads, which is where they
    do the most good. Appends if the anchor ever moves.
    """
    anchor = "# Response format"
    i = prompt.find(anchor)
    if i == -1:
        logger.warning("mcp agent: '%s' anchor not found; appending section", anchor)
        return prompt.rstrip() + "\n\n" + section + "\n"
    return prompt[:i].rstrip() + "\n\n" + section + "\n\n" + prompt[i:]


def splice_example(prompt: str, example: str) -> str:
    """Insert the worked MCP example after the '## Wait' example's predecessor.

    Concretely: immediately BEFORE '## Wait (action=wait)', which puts it after
    the GUI and Shell examples and before the episode-control ones. Calling a tool
    is an action, so it reads alongside the other actions; appending at the very
    end would place it after 'Finish (action=terminate)', i.e. after the task is
    over.

    Falls back to appending if c_gui renames that heading -- a misplaced example
    is still better than none, and the warning says where to look.
    """
    anchor = "## Wait (action=wait)"
    i = prompt.find(anchor)
    if i == -1:
        logger.warning("mcp agent: '%s' anchor not found; appending example", anchor)
        return prompt.rstrip() + "\n\n" + example.rstrip() + "\n"
    return prompt[:i] + example.rstrip() + "\n\n" + prompt[i:]


def coerce_params(raw) -> Dict:
    """Turn the `params` parameter into a dict.

    The model writes this by hand inside XML, so it arrives as text and may be
    JSON, a python literal, or empty. Anything unparseable becomes {} -- better a
    no-arg call whose error the model can read than a crashed episode.
    """
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    import ast
    import json
    for parse in (json.loads, ast.literal_eval):
        try:
            val = parse(text)
        except Exception:
            continue
        if isinstance(val, dict):
            return val
    logger.warning("mcp agent: could not parse params %r; using {}", text[:120])
    return {}


def build_mcp_agent(args, **overrides):
    """Construct a CGuiMcpAgent from an argparse namespace shaped like
    run_c_gui_sharded.py's. Keeps the runner thin and the arg mapping in one place."""
    kwargs = dict(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        coordinate_type=args.coord,
        add_thought_prefix=args.add_thought_prefix,
        history_n=args.history_n,
        image_max=args.image_max,
        fold_size=args.fold_size,
        enable_thinking=args.enable_thinking,
        password=args.password,
        tool_name=args.tool_name,
        mcp_mode=getattr(args, "mcp_mode", "off"),
    )
    kwargs.update(overrides)
    return CGuiMcpAgent(**kwargs)

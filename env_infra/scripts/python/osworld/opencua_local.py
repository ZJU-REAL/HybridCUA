"""Platform-side glue for running OpenCUA on the local sharded-vLLM cluster.

``OSWorld/mm_agents/opencua/`` is a read-only vendor snapshot (CLAUDE.md #3), and two
things in it do not match this platform:

1. **The endpoint is hardcoded.** ``OpenCUAAgent.call_llm`` posts to
   ``https://{self.model}.app.msh.team/v1/chat/completions`` and authenticates with
   ``os.environ['OPENCUA_API_KEY']`` (opencua_agent.py:443-457) -- it never consults
   ``OPENAI_BASE_URL``. So the per-worker env-var trick that ``run_evocua_sharded.py``
   uses does not work here: ``self.model`` is both the URL host and the payload's
   ``model`` field, and no env var can redirect it. We subclass and replace only
   ``call_llm``, taking the endpoint as a constructor argument.

2. **The episode loop never logs ``results.json``.** ``lib_run_single.
   run_single_example_opencua`` (:397-448) ends with ``result.txt`` +
   ``recording.mp4`` but, unlike the stock ``run_single_example`` (:15-71), it never
   calls ``log_task_completion``. Without it there is no
   ``<result_dir>/summary/results.json``, and ``scripts/bash/osworld/
   eval_ckpt_serial.sh:75`` uses exactly that file to decide whether a rollout
   succeeded -- it would report ``NO results-.../summary/results.json (run likely
   failed)`` for a perfectly good run. ``run_single_example_opencua_logged`` below
   calls the vendor loop and then adds the missing line.

Nothing here modifies the vendored files; both pieces are wrappers.

Usage (normally via ``scripts/python/osworld/run_opencua_sharded.py``):
    agent = OpenCUALocalAgent(..., base_url="http://127.0.0.1:8000/v1", api_key="sk-local")
"""
import logging
import os

import lib_run_single
from lib_results_logger import log_task_completion
from mm_agents.opencua import OpenCUAAgent

# Reusing the Qwen family's client rather than writing another httpx loop buys us its
# 5x retry with backoff (OSWORLD_MAX_RETRY_TIMES). The vendor OpenCUA call_llm has no
# retry at all, and mm_agents/evocua's docstring flags that exact gap: "A transient
# endpoint hiccup kills the episode." Import-only -- the file is not modified.
from mm_agents.qwen.client import call_openai_compatible

logger = logging.getLogger("desktopenv.experiment")


class OpenCUALocalAgent(OpenCUAAgent):
    """``OpenCUAAgent`` pointed at an OpenAI-compatible endpoint instead of msh.team.

    Only ``call_llm`` differs. Everything else -- prompt construction, history
    handling, coordinate projection, response parsing -- is inherited unchanged, so
    behaviour on a given screenshot is identical to the vendored agent.
    """

    def __init__(self, *args, base_url: str = None, api_key: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Explicit args, not os.environ: the runner builds one agent per worker and
        # passes its shard directly, which keeps the endpoint visible in tracebacks
        # instead of hidden in the process environment.
        self.base_url = base_url
        self.api_key = api_key

    def call_llm(self, payload, model):
        """POST to the local vLLM shard. Returns the assistant message content.

        ``model`` is ignored on purpose: the vendored signature passes ``self.model``
        but only ever used it to build the URL. The payload already carries the
        ``model`` field, and call_openai_compatible reads it from there.
        """
        return call_openai_compatible(
            payload,
            model or self.model,
            base_url=self.base_url,
            api_key=self.api_key or "sk-local",
            default_max_tokens=self.max_tokens,
            default_temperature=self.temperature,
            default_top_p=self.top_p,
            logger=logger,
        )


def _read_score(example_result_dir: str) -> float:
    """Score the vendor loop just wrote to result.txt; 0.0 if it is missing/unreadable."""
    try:
        with open(os.path.join(example_result_dir, "result.txt"), "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        logger.warning("No readable result.txt in %s; logging score 0.0", example_result_dir)
        return 0.0


def run_single_example_opencua_logged(agent, env, example, max_steps, instruction,
                                      args, example_result_dir, scores):
    """Vendor OpenCUA episode loop + the ``log_task_completion`` it omits.

    Same 8-positional-arg contract as every other variant, so
    ``OSWorldEvalSource.run_episode`` (cluster/client/osworld/eval.py:174) calls it
    unchanged. If the loop raises, the worker's own error dump handles it
    (cluster/client/base/eval_worker.py:291-295) and nothing is logged here -- the
    same outcome as the stock loop, which also logs only on the success path.
    """
    lib_run_single.run_single_example_opencua(
        agent, env, example, max_steps, instruction, args, example_result_dir, scores
    )
    log_task_completion(example, _read_score(example_result_dir), example_result_dir, args)

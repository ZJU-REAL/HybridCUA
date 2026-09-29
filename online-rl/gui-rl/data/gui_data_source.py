from __future__ import annotations

import copy
import json
import logging
import os
import random
from pathlib import Path

import torch

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample

import config
from data.cua_gym_tasks import load_task


logger = logging.getLogger(__name__)


def _pop_first(buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


class _MetaDataSource(DataSource):
    """Shared machinery for meta-file GUI data sources.

    A subclass sets ``self.tasks`` (a list of task dicts) in ``__init__`` and
    implements ``_task_metadata``. Everything world-neutral — GRPO grouping,
    buffering, and rollout-state checkpointing — lives here. ``_STATE_PREFIX``
    names the checkpoint file so different sources don't collide on resume.
    """

    _STATE_PREFIX = "meta_state_dict_"

    def _init_state(self) -> None:
        self.buffer: list[list[Sample]] = []
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0

    def _task_metadata(self, task: dict) -> dict:
        """One entry of ``self.tasks`` -> that sample's ``metadata``.

        The rollout (``utils.rollout_helpers._sample_task_info``) reads four keys:
        ``instruction`` (also used as the prompt), ``task_config`` (forwarded
        verbatim as the env reset payload), and ``domain`` / ``example_id``
        (result-dir layout + logging).
        """
        raise NotImplementedError

    def _make_prompt_samples(self, num_samples: int) -> list[Sample]:
        out: list[Sample] = []
        for _ in range(num_samples):
            metadata = self._task_metadata(self._next_task())
            out.append(Sample(prompt=metadata["instruction"], label="", metadata=metadata))
        return out

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = _pop_first(self.buffer, num_samples)
        num_samples -= len(samples)
        if num_samples <= 0:
            return samples

        prompt_samples = self._make_prompt_samples(num_samples)
        groups: list[list[Sample]] = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                s = copy.deepcopy(prompt_sample)
                s.group_index = self.sample_group_index
                s.index = self.sample_index
                self.sample_index += 1
                group.append(s)
            self.sample_group_index += 1
            groups.append(group)
        return samples + groups

    def add_samples(self, samples: list[list[Sample]]):
        if samples:
            self.buffer.extend(samples)

    def save(self, rollout_id):
        state = {
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "sample_offset": self.sample_offset,
        }
        path = os.path.join(self.args.save, f"rollout/{self._STATE_PREFIX}{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

    def load(self, rollout_id=None):
        if self.args.load is None:
            return
        path = os.path.join(self.args.load, f"rollout/{self._STATE_PREFIX}{rollout_id}.pt")
        if not os.path.exists(path):
            return
        state = torch.load(path)
        self.epoch_id = state.get("epoch_id", 0)
        self.sample_group_index = state.get("sample_group_index", 0)
        self.sample_index = state.get("sample_index", 0)
        self.sample_offset = state.get("sample_offset", 0)

    def __len__(self) -> int:
        return len(self.tasks)

    def _next_task(self) -> dict:
        """Round-robin a task, reshuffling each time the list wraps."""
        task = self.tasks[self.sample_offset % len(self.tasks)]
        self.sample_offset += 1
        if self.sample_offset % len(self.tasks) == 0:
            self.epoch_id += 1
            random.Random(self.epoch_id).shuffle(self.tasks)
        return task


class GuiMetaDataSource(_MetaDataSource):
    """OSWorld GUI tasks from evaluation_examples meta files.

    Loads ``GUI_TRAIN_META_PATH`` (a ``{domain: [example_id, ...]}`` map) and the
    per-task configs under ``GUI_TEST_CONFIG_BASE_DIR/examples/<domain>/<id>.json``.
    """

    _STATE_PREFIX = "gui_meta_state_dict_"

    def __init__(self, args):
        self.args = args
        self._init_state()

        base_dir = Path(
            config.test_config_base_dir(str(Path(__file__).resolve().parent.parent / "evaluation_examples"))
        )
        meta_path = Path(config.train_meta_path(str(base_dir / "train_nochrome.json")))

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        tasks: list[dict] = []
        for domain, example_ids in meta.items():
            for example_id in example_ids:
                cfg_path = base_dir / "examples" / str(domain) / f"{example_id}.json"
                if not cfg_path.exists():
                    continue
                with open(cfg_path, "r", encoding="utf-8") as cf:
                    task_cfg = json.load(cf)
                tasks.append(
                    {
                        "domain": str(domain),
                        "example_id": str(example_id),
                        "instruction": task_cfg.get("instruction", ""),
                        "task_config": task_cfg,
                    }
                )

        if not tasks:
            raise RuntimeError(f"No tasks loaded from {meta_path}")

        self.tasks = tasks
        random.Random(self.epoch_id).shuffle(self.tasks)

    def _task_metadata(self, task: dict) -> dict:
        return {
            "domain": task["domain"],
            "example_id": task["example_id"],
            "instruction": task["instruction"],
            "task_config": task["task_config"],
            "cli_preferred": task["task_config"].get("cli_preferred"),
        }


class MobileWorldDataSource(_MetaDataSource):
    """MobileWorld tasks from an offline task-list JSON.

    The JSON is generated once on the env node (where ``mobile_world`` is
    importable) and shipped here, so the training node never imports the
    benchmark. Each sample carries ``task_config={"task_name": ...}``, which the
    session client forwards verbatim as the reset ``task_payload`` for the node's
    MobileWorld adapter to read.
    """

    _STATE_PREFIX = "mw_meta_state_dict_"
    _MCP_TAG = "agent-mcp"
    _USER_TAG = "agent-user-interaction"

    def __init__(self, args):
        self.args = args
        self._init_state()

        default = str(Path(__file__).resolve().parent.parent / "evaluation_examples" / "mw_tasks.json")
        path = Path(config.mw_task_list_path(default))
        enable_mcp = config.mw_enable_mcp()
        enable_user = config.mw_enable_user_interaction()

        doc = json.loads(path.read_text(encoding="utf-8"))
        tasks: list[dict] = []
        for t in doc["tasks"]:
            tags = set(t.get("tags", []))
            if (not enable_mcp and self._MCP_TAG in tags) or (not enable_user and self._USER_TAG in tags):
                continue
            name = t["task_name"]
            tasks.append({"task_name": name, "instruction": t.get("goal") or name})

        if not tasks:
            raise RuntimeError(f"No MobileWorld tasks loaded from {path}")

        self.tasks = tasks
        random.Random(self.epoch_id).shuffle(self.tasks)

    def _task_metadata(self, task: dict) -> dict:
        return {
            # Forwarded verbatim as the reset task_payload; the node's
            # MobileWorld adapter reads task_payload["task_name"].
            "task_config": {"task_name": task["task_name"]},
            "instruction": task["instruction"],
            # Decorative only (result dir / logging); no on-disk file.
            "domain": "mobileworld",
            "example_id": task["task_name"],
        }


class CuaGymDataSource(_MetaDataSource):
    """CUA-Gym desktop tasks from a flat directory of ``<uuid>/`` bundles.

    Task selection comes from an OSWorld-shaped meta JSON (``{app_type: [uuid,
    ...]}``, i.e. ``GUI_CUA_GYM_TASKS_META``), with ``app_type`` playing the role
    of ``domain``. Bundles are resolved under ``GUI_CUA_GYM_BUNDLES``.

    ``__init__`` only *stats* each bundle (10k entries on a network FS), leaving
    the actual read to :meth:`_task_metadata` — so startup stays cheap and each
    task's ``reward.py`` is read once, when it is first sampled.

    Bundles with an empty ``reward.py`` are skipped: the adapter scores them 0.0
    with "no reward code", which only flattens a GRPO group into a no-gradient
    constant. Web tasks needing a self-hosted CUA-Gym-Hub (their setup/reward
    carry ``__CUA_GYM_*_URL__`` placeholders, likewise always 0) are excluded by
    the default meta file rather than here.
    """

    _STATE_PREFIX = "cua_gym_meta_state_dict_"

    def __init__(self, args):
        self.args = args
        self._init_state()

        self.bundles_root = Path(config.cua_gym_bundles_dir())
        meta_path = Path(config.cua_gym_tasks_meta())
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        tasks: list[dict] = []
        skipped_missing = skipped_no_reward = 0
        for app_type, bundle_ids in meta.items():
            if not isinstance(bundle_ids, list):
                continue  # the meta file also carries string fields (_comment, description, ...)
            for bundle_id in bundle_ids:
                bundle = self.bundles_root / str(bundle_id)
                if not ((bundle / "task.json").exists() or (bundle / "config.json").exists()):
                    skipped_missing += 1
                    continue
                reward_py = bundle / "reward.py"
                if not reward_py.exists() or reward_py.stat().st_size == 0:
                    skipped_no_reward += 1
                    continue
                tasks.append({"domain": str(app_type), "example_id": str(bundle_id)})

        if not tasks:
            raise RuntimeError(f"No CUA-Gym tasks loaded from {meta_path} under {self.bundles_root}")
        logger.info(
            "CuaGymDataSource: %d tasks from %s (skipped %d missing bundle, %d empty reward.py)",
            len(tasks), meta_path, skipped_missing, skipped_no_reward,
        )

        self.tasks = tasks
        random.Random(self.epoch_id).shuffle(self.tasks)

    def _task_metadata(self, task: dict) -> dict:
        # load_task inlines reward.py as task_config["reward_code"]; CuaGymWorldAdapter
        # pops it at reset and runs it in the VM at evaluate.
        task_config = load_task(self.bundles_root / task["example_id"])
        return {
            "domain": task["domain"],
            "example_id": task["example_id"],
            "instruction": str(task_config.get("instruction", "")),
            "task_config": task_config,
           # b*: bundle-provided "this task benefits from CLI" flag; None until labeled.
            "cli_preferred": task_config.get("cli_preferred"),
        }


# Each device family: its runtime (env-server session routing), its agent class
# (prompt/parser), and the DataSource that produces its tasks.
_PLATFORMS = {
    "mobileworld": ("agents.qwen3vl_mobile_agent.Qwen3VLMobileAgentLocal", MobileWorldDataSource),
    "osworld": ("agents.qwen3vl_agent.Qwen3VLAgentLocal", GuiMetaDataSource),
    # Same desktop VM/agent surface as osworld — only the task source and the
    # reward mechanism (in-VM python script) differ, both of which are handled
    # node-side by CuaGymWorldAdapter.
    "cua_gym": ("agents.qwen35_hybrid_cua.Qwen35HybridCuaAgentLocal", CuaGymDataSource),
}


def _tag_platform(groups: list[list[Sample]], runtime: str, agent_class_path: str) -> list[list[Sample]]:
    """Stamp every sample in every group with its platform so the rollout worker
    routes the env session (runtime) and picks the agent (agent_class_path)."""
    for group in groups:
        for s in group:
            s.metadata = s.metadata or {}
            s.metadata["runtime"] = runtime
            s.metadata["agent_class_path"] = agent_class_path
    return groups


class _TaggedSource:
    """Wraps a platform's sub-source so every group it yields carries that
    platform's runtime + agent tag. One per platform; a multi-platform worker
    binds one of these and thus only ever produces its own platform's groups."""

    def __init__(self, source, runtime: str, agent_class_path: str):
        self.source = source
        self.runtime = runtime
        self.agent_class_path = agent_class_path

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return _tag_platform(self.source.get_samples(num_samples), self.runtime, self.agent_class_path)


class MultiPlatformDataSource(DataSource):
    """One sub-DataSource per device family (mobile, desktop, ...), for
    GUI-Owl-1.5 MRPO multi-platform training (Mix vs Interleaved).

    It is the object slime constructs (GUI_DATA_SOURCE_PATH) and drives for
    save/load/__len__ — so it owns every platform's cursor and checkpoint.

    Runs on the **semi-async** default rollout path (slime's ``generate_rollout``):
    each rollout synchronously pulls one batch through :meth:`get_samples`, which
    assembles that batch per ``--multi-platform-mode``:
      - ``mix``:         split the batch's groups evenly across platforms → each
                         training batch mixes platforms (baseline).
      - ``interleaved``: the whole batch comes from one platform on duty this
                         rollout, ``platforms[(rollout_id // K) % n]`` (ours).
      - ``single``:      first platform only (degenerate).

    ``rollout_id`` is not threaded into ``get_samples`` by slime, so the thin
    :func:`generate_rollout_multiplatform_semi` entrypoint stamps it onto
    ``current_rollout_id`` before each rollout; interleaved reads it here.
    """

    def __init__(self, args):
        self.args = args
        names = [n.strip() for n in os.getenv("GUI_MULTI_PLATFORM_LIST", "mobileworld,osworld").split(",") if n.strip()]
        unknown = [n for n in names if n not in _PLATFORMS]
        if unknown:
            raise ValueError(f"unknown platforms {unknown}; known: {list(_PLATFORMS)}")
        self.platforms = names
        self.sources = {n: _PLATFORMS[n][1](args) for n in names}
        self.agent_paths = {n: _PLATFORMS[n][0] for n in names}
        # Prebuilt tagged views so get_samples reuses the runtime/agent stamping.
        self._tagged = self.tagged_sources()
        # Set per-rollout by generate_rollout_multiplatform_semi; interleaved's
        # on-duty platform is derived from it. Persisted in save/load.
        self.current_rollout_id = 0

    def tagged_sources(self) -> dict[str, _TaggedSource]:
        """{platform: tagged sub-source}; each stamps its groups with runtime+agent."""
        return {n: _TaggedSource(self.sources[n], n, self.agent_paths[n]) for n in self.platforms}

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Assemble one batch of ``num_samples`` groups per multi-platform mode.

        ``num_samples`` counts GROUPS (each sub-source builds n_samples_per_prompt
        samples per group), so splitting by group count keeps every group intact.
        """
        mode = getattr(self.args, "multi_platform_mode", "single")
        if mode == "interleaved":
            idx = (self.current_rollout_id // self.args.alternating_switch_interval) % len(self.platforms)
            return self._tagged[self.platforms[idx]].get_samples(num_samples)
        if mode == "mix":
            # Even split of groups across platforms; remainder to the first ones.
            n = len(self.platforms)
            per, rem = divmod(num_samples, n)
            out: list[list[Sample]] = []
            for i, name in enumerate(self.platforms):
                k = per + (1 if i < rem else 0)
                if k:
                    out += self._tagged[name].get_samples(k)
            return out
        # single / fallback: first platform only.
        return self._tagged[self.platforms[0]].get_samples(num_samples)

    def add_samples(self, samples: list[list[Sample]]):
        # Route each group back to its origin platform (read from the tag).
        for group in samples:
            name = (group[0].metadata or {}).get("runtime") if group else None
            if name in self.sources:
                self.sources[name].add_samples([group])

    def save(self, rollout_id):
        for src in self.sources.values():
            src.save(rollout_id)
        # Persist the interleaved cursor so resume lands on the right platform.
        path = os.path.join(self.args.save, f"rollout/mp_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"current_rollout_id": self.current_rollout_id}, path)

    def load(self, rollout_id=None):
        for src in self.sources.values():
            src.load(rollout_id)
        if self.args.load is None:
            return
        path = os.path.join(self.args.load, f"rollout/mp_state_dict_{rollout_id}.pt")
        if os.path.exists(path):
            self.current_rollout_id = torch.load(path).get("current_rollout_id", 0)

    def __len__(self) -> int:
        return sum(len(src) for src in self.sources.values())


def generate_rollout_multiplatform_semi(args, rollout_id, data_source, evaluation: bool = False):
    """Semi-async multi-platform ``--rollout-function-path`` entrypoint.

    slime's default ``generate_rollout`` does not thread ``rollout_id`` into
    ``data_source.get_samples`` (it passes the bound method). This thin wrapper
    stamps ``rollout_id`` onto the source — so interleaved knows which platform is
    on duty — then delegates to the stock synchronous rollout unchanged. Using the
    real ``rollout_id`` (not a per-call counter) keeps the on-duty platform stable
    even if over-sampling calls ``get_samples`` more than once per rollout.
    """
    from slime.rollout.sglang_rollout import generate_rollout

    if not evaluation and hasattr(data_source, "current_rollout_id"):
        data_source.current_rollout_id = rollout_id
    return generate_rollout(args, rollout_id, data_source, evaluation=evaluation)

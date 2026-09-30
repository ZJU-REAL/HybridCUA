<div align="center">

# HybridCUA

**Learning to Orchestrate GUI and CLI for Computer-Use Agents**

📄 [Paper](https://arxiv.org/abs/2604.00000) |
🌐 [Project Page](https://zjureal.com/HybridCUA/) |
🤗 [Dataset](https://huggingface.co/collections/077lukamagic/hybridcua) |
🤖 [Models](https://huggingface.co/collections/077lukamagic/hybridcua)

[![arXiv](https://img.shields.io/badge/arXiv-2604.00000-b31b1b.svg)](https://arxiv.org/abs/2604.00000)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HybridCUA--8K-ffc107.svg)](https://huggingface.co/collections/077lukamagic/hybridcua)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-HybridCUA--9B-ff9800.svg)](https://huggingface.co/collections/077lukamagic/hybridcua)
[![License](https://img.shields.io/badge/License-Apache%202.0-4caf50.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776ab.svg)](https://www.python.org/downloads/)

</div>

Computer-use agents act through clicks and keystrokes — general, but slow and
prone to cascading errors. Application APIs are faster but must be built per
application. HybridCUA instead pairs the GUI with the **command line**, which
ships with the OS and can collapse a long GUI sequence into one command.

The hard part is not shell *access* but knowing **when and how** to use it:
exposing a CLI to an untrained agent *costs* 2.5–11.5 points on OSWorld.
HybridCUA closes that gap with hybrid training data and CLI-aware rewards.

**HybridCUA-9B reaches 53.6% on OSWorld in 14.0 average steps** — +14.8 points
over its base model — and transfers to OSWorld-MCP (47.1%) and Windows (36.0%).

## Repository layout

```
HybridCUA/
├── env_infra/    desktop/mobile environment platform (OSWorld, MobileWorld, CUA-Gym)
├── online-rl/    GRPO training (slime + Megatron-LM actor, SGLang rollout)
└── site/         project page (static, no build step)
```

The two halves talk over HTTP: `env_infra` serves environments through a unified
`/v1/sessions` protocol, and `online-rl` only speaks to that endpoint — the
trainer never launches VMs or containers itself.

| Directory | Read next |
|---|---|
| `env_infra/` | [`env_infra/CLAUDE.md`](env_infra/CLAUDE.md) — architecture, session protocol, world adapters |
| `online-rl/` | [`online-rl/CLAUDE.md`](online-rl/CLAUDE.md) — rollout pipeline, the two layers of asynchrony, every `GUI_*` variable |
| `site/` | [`site/README.md`](site/README.md) — preview and deploy the project page |

## Method in brief

**One action space.** Every executable interaction goes through the same
`bash` action — a direct shell command, or a quoted Python heredoc of
`pyautogui` calls. Switching interfaces costs the model no extra grammar.
Keeping the two in separate tools instead costs 7.2 accuracy points.

**HybridCUA-8K.** 5,023 supervised trajectories across 11 application domains
in three modes (GUI only 870, CLI only 3,155, interleaved 998), plus 3,000
verified RLVR tasks, each labelled by whether CLI use has a clear execution
advantage.

**Two-stage training.** SFT on the mixed corpus, then GRPO with two CLI-aware
rewards:

- `R_CLI` (trajectory level, λ=0.1) — rewards a successful rollout only when its
  CLI usage matches the task's label. Teaches **when**.
- `r_exec` (step level, λ=0.3) — penalises the tokens of a command that fails at
  the shell level. Teaches **how**.

Ablating them separates the two effects: without `R_CLI` the efficiency gain
stalls (18.2% shorter trajectories instead of 29.3%); without `r_exec`
execution errors climb to 16.5% instead of settling at 11.5%.

## Results

| Model | Action space | OSWorld Acc. ↑ | Avg. steps ↓ |
|---|---|---|---|
| Qwen3.5-9B | GUI | 38.8 | 31.6 |
| Qwen3.5-9B | GUI+CLI | 18.4 | 22.1 |
| ToolCUA-8B | GUI+API | 46.8 | 14.9 |
| AutoGLM-OS-9B | GUI+API | 48.9 | — |
| UltraCUA-32B | GUI+API | 43.7 | — |
| **HybridCUA-9B** | **GUI+CLI** | **53.6** | **14.0** |

Out-of-distribution: **47.1%** on OSWorld-MCP (+9.1 over the base model) and
**36.0%** on WindowsAgentArena (+4.0) — issuing PowerShell commands despite
training only on Linux shells.

Full tables, per-domain results and step-by-step rollouts are on the
[project page](https://zjureal.com/HybridCUA/).

## Release

Trajectories, RLVR tasks, the data-generation and training pipelines, and the
HybridCUA-9B model. See the [Hugging Face collection](https://huggingface.co/collections/077lukamagic/hybridcua).

## Safety

Shell access amplifies what an agent can do in a single action. All trajectory
collection, training and evaluation run in sandboxed VMs that are reset between
episodes, with no access to real user accounts or credentials. Real-world
deployment is out of scope for this work and would need further evaluation,
safeguards, and human oversight.

## Citation

```bibtex
@article{chen2026hybridcua,
  title   = {HybridCUA: Learning to Orchestrate GUI and CLI for Computer-Use Agents},
  author  = {Chen, Tongbo and Niu, Junbo and Lu, Zhengxi and Lian, Niu and
             Tang, Fei and Yan, Yuchen and Hong, Yike and Du, Yong and
             Liu, Yizhou and Chen, Bofan and Shen, Yongliang},
  journal = {arXiv preprint},
  year    = {2026}
}
```

"""Guest-side bring-up of the MCP stack, driven over a DesktopEnv's ``run_code``.

setup_mcp_guest.py does the same thing over the container's mapped :5000 /execute
port, which is fine for manual poking but useless from inside a runner: the runner
holds an ``env`` object and has no idea which host port its VM landed on. This
module is the run_code-driven twin, and it is what ``mcp_runtime.provision_mcp``
calls once per episode.

The ordering and every non-obvious workaround below are explained in
setup_mcp_guest.py's module docstring; the short version:

  1. python deps  -- fastmcp MUST be 2.11.3 (4.x moved ``fastmcp.tools.tool``)
  2. node + uv    -- only needed by the two distractor servers, but the benchmark
                     counts their 28 tools, so a run without them is not comparable
  3. bundle       -- pushed as base64 over run_code; ~59KB gzipped
  4. soffice UNO  -- :2002 must be up BEFORE the server, else all 68 LibreOffice
                     tools fail. Probe the PORT, never ``pgrep -f accept=socket``
                     (that pattern matches the bash -lc carrying it).
  5. mcp server   -- :9292, started with cwd=mcp_server (relative json paths)

Everything is idempotent: the container pool recycles VMs and env.reset() may roll
a snapshot back, so each step checks before acting and a warm guest costs a handful
of greps.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import shlex
import tarfile
import time
from typing import Optional

logger = logging.getLogger("desktopenv.mcp_guest_ops")

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_mcp_src() -> str:
    """Locate the benchmark's mcp/ tree (mcp_server/ + osworld_mcp_client.py).

    It lives in the OSWorld-MCP checkout and is deliberately NOT copied into this
    package: it is benchmark source, shipped to the guest verbatim, and the two
    copies of this package would drift. So the path is resolved at runtime, which
    means it differs per copy:

        OSWorld-MCP/osworld_mcp_agents/mcp_common/  -> ../../mcp
        env_infra/OSWorld/mm_agents/mcp_common/     -> ../../../../OSWorld-MCP/mcp

    Returns "" when nothing is found; callers must treat that as fatal.
    """
    env_override = os.environ.get("OSWORLD_MCP_SRC", "").strip()
    candidates = [env_override] if env_override else []
    candidates += [
        os.path.join(HERE, "..", "..", "mcp"),
        os.path.join(HERE, "..", "..", "..", "..", "OSWorld-MCP", "mcp"),
    ]
    for cand in candidates:
        path = os.path.abspath(cand)
        if os.path.isfile(os.path.join(path, "osworld_mcp_client.py")):
            return path
    return ""


MCP_SRC = _find_mcp_src()
GUEST_HOME = "/home/user"


def preflight() -> None:
    """Fail before any container is acquired if the bundle cannot be built.

    A missing mcp/ tree makes provision_mcp return False, and every episode then
    degrades to GUI-only with tools=0 -- a full run's worth of results labelled as
    an MCP arm with no MCP in it. Cheap to check, so check it up front where the
    error is still attributable.
    """
    if not MCP_SRC:
        raise FileNotFoundError(
            "MCP bundle source not found. Expected OSWorld-MCP/mcp/ containing "
            "mcp_server/ and osworld_mcp_client.py, searched relative to "
            f"{HERE}. Set OSWORLD_MCP_SRC to override.")
    missing = [p for p in ("osworld_mcp_client.py", "mcp_server")
               if not os.path.exists(os.path.join(MCP_SRC, p))]
    if missing:
        raise FileNotFoundError(f"{MCP_SRC} is incomplete, missing: {missing}")

FASTMCP_PIN = "fastmcp==2.11.3"
NODE_VERSION = "v22.18.0"
NODE_BIN = f"{GUEST_HOME}/.nvm/versions/node/{NODE_VERSION}/bin"
UV_BIN = f"{GUEST_HOME}/.local/bin"
GUEST_PROXY = "http://star-proxy.oa.com:3128"

ENV_PREFIX = (
    f'export PATH="{NODE_BIN}:{UV_BIN}:$PATH" '
    f'http_proxy={GUEST_PROXY} https_proxy={GUEST_PROXY} '
    f'no_proxy=localhost,127.0.0.1,::1; '
)

__all__ = ["bring_up_via_run_code", "mcp_is_serving", "bundle_b64"]


def guest_sh(env, cmd: str, timeout: Optional[int] = 180) -> str:
    """One bash command in the guest; returns stdout+stderr, never raises."""
    try:
        res = env.run_code(ENV_PREFIX + cmd, lang="bash", timeout=timeout) or {}
    except TypeError:
        res = env.run_code(ENV_PREFIX + cmd, lang="bash") or {}
    except Exception as exc:
        logger.warning("guest cmd failed: %s", exc)
        return ""
    return (res.get("output") or "") + (res.get("error") or "")


def poll_guest(env, cmd: str, want: str, tries: int, delay: int, label: str,
          log=logger) -> bool:
    for i in range(tries):
        if want in guest_sh(env, cmd, timeout=120):
            return True
        log.info("mcp bring-up: waiting on %s (%d/%d)", label, i + 1, tries)
        time.sleep(delay)
    return False


def bundle_b64() -> str:
    """mcp_server/ + osworld_mcp_client.py as a base64 gzipped tar.

    The client is shipped VERBATIM: its three-server config is what produces the
    ``osworld_mcp_*`` tool names and the 28 distractor tools the benchmark scores.
    """
    buf = io.BytesIO()
    if not MCP_SRC:
        raise FileNotFoundError(
            "cannot locate the benchmark's mcp/ tree (mcp_server/ + "
            "osworld_mcp_client.py). Set OSWORLD_MCP_SRC=/path/to/OSWorld-MCP/mcp")
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        root = os.path.join(MCP_SRC, "mcp_server")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if fn.endswith(".pyc"):
                    continue
                full = os.path.join(dirpath, fn)
                tf.add(full, arcname=os.path.join(
                    "mcp_server", os.path.relpath(full, root)))
        tf.add(os.path.join(MCP_SRC, "osworld_mcp_client.py"),
               arcname="osworld_mcp_client.py")
    return base64.b64encode(buf.getvalue()).decode()


def mcp_is_serving(env) -> bool:
    """True when :9292 is bound AND the client can complete a handshake.

    Binding happens before serving, and a bound-but-not-serving port is exactly the
    state that makes the first tool call fail, so check both.
    """
    if "UP" not in guest_sh(env, "(ss -ltn 2>/dev/null||netstat -ltn)|grep -q 9292 "
                            "&& echo UP || echo DOWN", timeout=60):
        return False
    out = guest_sh(env, f"cd {GUEST_HOME} && python3 -c \""
                   "from osworld_mcp_client import OsworldMcpClient as C;"
                   "t=C.list_tools(tool_name=None, shuffle=False, rag=False);"
                   "print('SERVING' if t else 'EMPTY')\" 2>&1 | "
                   "grep -oE 'SERVING|EMPTY' | tail -1", timeout=180)
    return "SERVING" in out


def ensure_python_deps(env, log) -> bool:
    probe = ("python3 -c \"import fastmcp; from fastmcp.tools.tool import Tool; "
             "import playwright; print('PY-OK')\" 2>&1 | tail -1")
    if "PY-OK" in guest_sh(env, probe, timeout=120):
        return True
    log.info("mcp bring-up: installing %s + playwright", FASTMCP_PIN)
    guest_sh(env, "nohup bash -c 'pip3 uninstall -y fastmcp fastmcp-slim mcp >/dev/null 2>&1; "
             f"pip3 install \"{FASTMCP_PIN}\" playwright' > /tmp/mcp_deps_py.log 2>&1 & echo go",
        timeout=120)
    return poll_guest(env, probe, "PY-OK", tries=40, delay=15, label="python deps", log=log)


def ensure_node_and_uv(env, log) -> bool:
    """node + uv for the filesystem/git distractor servers. Non-fatal if missing."""
    ok = True
    if "NODE-OK" not in guest_sh(env, "node -v >/dev/null 2>&1 && npx --version >/dev/null 2>&1 "
                                 "&& echo NODE-OK", timeout=60):
        log.info("mcp bring-up: installing node %s", NODE_VERSION)
        tgz = f"node-{NODE_VERSION}-linux-x64"
        guest_sh(env, "nohup bash -c '"
                 f"mkdir -p {GUEST_HOME}/.nvm/versions/node && cd /tmp && "
                 f"curl -fsSL -o {tgz}.tar.xz "
                 f"https://nodejs.org/dist/{NODE_VERSION}/{tgz}.tar.xz && "
                 f"tar xf {tgz}.tar.xz && rm -rf {GUEST_HOME}/.nvm/versions/node/{NODE_VERSION} && "
                 f"mv {tgz} {GUEST_HOME}/.nvm/versions/node/{NODE_VERSION}"
                 "' > /tmp/mcp_deps_node.log 2>&1 & echo go", timeout=120)
        ok &= poll_guest(env, "node -v >/dev/null 2>&1 && npx --version >/dev/null 2>&1 && echo NODE-OK",
                    "NODE-OK", tries=30, delay=10, label="node", log=log)

    if "UV-OK" not in guest_sh(env, "uvx --version >/dev/null 2>&1 && echo UV-OK", timeout=60):
        log.info("mcp bring-up: installing uv")
        guest_sh(env, "nohup bash -c 'curl -fsSL https://astral.sh/uv/install.sh | sh' "
                 "> /tmp/mcp_deps_uv.log 2>&1 & echo go", timeout=120)
        ok &= poll_guest(env, "uvx --version >/dev/null 2>&1 && echo UV-OK", "UV-OK",
                    tries=24, delay=10, label="uv", log=log)
    return ok


def ensure_distractor_servers(env, log) -> bool:
    """Pre-install both distractor servers so fastmcp's per-Client spawn is local.

    fastmcp spawns `npx -y @modelcontextprotocol/server-filesystem` and
    `uvx mcp-server-git` on EVERY Client() construction. On a cold cache that
    download happens inside the first list_tools, which blocks long enough that the
    guest's HTTP layer 500s and every handshake attempt fails. Observed directly:
    `npm exec @modelcontextprotocol/server-filesystem` wedged for the whole window.
    """
    probe = ("command -v mcp-server-filesystem >/dev/null 2>&1 && "
             "command -v mcp-server-git >/dev/null 2>&1 && echo DIST-OK")
    if "DIST-OK" in guest_sh(env, probe, timeout=60):
        return True
    log.info("mcp bring-up: pre-installing the two distractor servers")
    guest_sh(env, f"npm config set proxy {GUEST_PROXY}; npm config set https-proxy {GUEST_PROXY}; "
             "npm config set noproxy localhost,127.0.0.1; echo set", timeout=120)
    guest_sh(env, "nohup bash -c '"
             "npm install -g @modelcontextprotocol/server-filesystem > /tmp/mcp_dist_fs.log 2>&1; "
             "UV_HTTP_TIMEOUT=120 uv tool install mcp-server-git > /tmp/mcp_dist_git.log 2>&1"
             "' > /dev/null 2>&1 & echo go", timeout=120)
    return poll_guest(env, probe, "DIST-OK", tries=30, delay=15, label="distractor servers", log=log)


def push_bundle(env, log) -> bool:
    blob = bundle_b64()
    log.info("mcp bring-up: pushing bundle (%d b64 chars)", len(blob))
    guest_sh(env, f"rm -rf {GUEST_HOME}/mcp_server {GUEST_HOME}/mcp_b64", timeout=120)
    CHUNK = 100_000
    for i in range(0, len(blob), CHUNK):
        guest_sh(env, f"printf '%s' {shlex.quote(blob[i:i + CHUNK])} >> {GUEST_HOME}/mcp_b64",
            timeout=300)
    out = guest_sh(env, f"cd {GUEST_HOME} && base64 -d mcp_b64 > mcp.tgz && tar xzf mcp.tgz && "
                   "rm -f mcp_b64 mcp.tgz && "
                   "echo PUSHED:$(ls mcp_server/tools/apis/*.json 2>/dev/null | wc -l)",
              timeout=300)
    if "PUSHED:10" not in out:
        log.error("mcp bring-up: bundle unpack looks wrong -- %s", out.strip()[:200])
        return False
    return True


def start_soffice_uno(env, log) -> bool:
    """Start the UNO listener on :2002, the way upstream's launch_soffice.sh does.

    Upstream is one line, and the two things it does NOT do are load-bearing:

      no pkill      -- LibreOffice is single-instance, so a task's setup opening
                       a document REUSES this process. Killing soffice.bin here
                       kills whatever document the task just opened. Measured:
                       41/46 libreoffice_calc episodes began on an empty desktop
                       and the domain scored 33.3% vs 57.4% on the no-MCP baseline.
      no --headless -- with it, every document opened afterwards is invisible.
                       OSWorld is a screenshot-driven benchmark; an invisible
                       LibreOffice is an unusable one.

    Both were in this function and together cost ~24 accuracy points.

    Prefer baking the stack into the image (bake_mcp_image.sh) so this runs before
    any task setup, exactly like upstream's install-once flow.
    """
    listening = ("(ss -ltn 2>/dev/null||netstat -ltn)|grep -q ':2002' "
                 "&& echo UNO-UP || echo UNO-DOWN")
    if "UNO-UP" in guest_sh(env, listening, timeout=60):
        return True
    log.info("mcp bring-up: starting soffice UNO :2002")
    guest_sh(env, f"cd {GUEST_HOME} && (nohup setsid soffice "
             "--accept='socket,host=localhost,port=2002;urp;' "
             "--norestore --nologo --nodefault > /tmp/soffice_uno.log 2>&1 &); "
             "echo go", timeout=180)
    ok = poll_guest(env, listening, "UNO-UP", tries=18, delay=5,
                    label="soffice UNO", log=log)
    if not ok:
        log.error("mcp bring-up: UNO log tail: %s",
                  guest_sh(env, "tail -5 /tmp/soffice_uno.log 2>&1", timeout=60).strip()[:300])
    return ok


def start_server(env, log) -> bool:
    # Bracketed for the same self-match reason as start_soffice_uno.
    guest_sh(env, "pkill -f '[p]ython3 server.py' 2>/dev/null; sleep 1; echo ok", timeout=120)
    guest_sh(env, f"cd {GUEST_HOME}/mcp_server && "
             "(nohup python3 server.py > /tmp/mcp_server.log 2>&1 &); echo go", timeout=120)
    if not poll_guest(env, "(ss -ltn 2>/dev/null||netstat -ltn)|grep -q 9292 && echo BOUND",
                 "BOUND", tries=18, delay=10, label=":9292 bind", log=log):
        log.error("mcp bring-up: :9292 never bound -- %s",
                  guest_sh(env, "tail -5 /tmp/mcp_server.log", timeout=60).strip()[:300])
        return False
    for i in range(10):
        if mcp_is_serving(env):
            return True
        log.info("mcp bring-up: waiting on handshake (%d/10)", i + 1)
        time.sleep(10)
    log.error("mcp bring-up: bound but not serving -- %s",
              guest_sh(env, "tail -5 /tmp/mcp_server.log", timeout=60).strip()[:300])
    return False


def probe_only(env, log) -> bool:
    """Confirm a baked image already serves the stack. Installs nothing.

    Costs two `ss -ltn` plus one handshake (~2s) against ~5min for a full install,
    and it cannot perturb task state -- which the install path can, since anything
    that restarts soffice races the task's own document.
    """
    if mcp_is_serving(env):
        log.info("mcp: image-baked stack already serving")
        return True
    ports = guest_sh(env, "(ss -ltn 2>/dev/null||netstat -ltn)"
                     "|grep -oE ':(2002|9292)'|sort -u|tr '\\n' ' '", timeout=60).strip()
    log.error("mcp: MCP_BRINGUP=image but the guest is not serving (listening: %s). "
              "Either the node is not on a baked qcow2 (PATH_TO_VM) or the bake is "
              "stale -- rebuild with OSWorld-MCP/bake_mcp_image.sh, or set "
              "MCP_BRINGUP=runcode to install per episode.", ports or "none")
    return False


def bring_up_via_run_code(env, log=logger, timeout: int = 600,
                          skip_deps: bool = False,
                          require_distractors: bool = True) -> bool:
    """Bring the whole MCP stack up in the guest. Idempotent. Never raises.

    `require_distractors=True` makes a missing node/uv a hard failure, because
    without the 28 distractor tools TIR is not comparable to the published numbers.
    Set False to run degraded on purpose.

    MCP_BRINGUP selects how the stack gets there:
      image   -- probe only; the qcow2 already has it (bake_mcp_image.sh). Matches
                 upstream's flow, where the README installs once before evaluation.
      runcode -- install per episode. Works without a baked image, but it runs
                 AFTER env.reset(), so anything touching soffice races the task's
                 document.
      auto    -- probe first, fall back to runcode (default; safe either way).
    """
    mode = os.environ.get("MCP_BRINGUP", "auto").strip().lower()
    if mode not in ("auto", "image", "runcode"):
        log.warning("mcp: unknown MCP_BRINGUP=%r, treating as auto", mode)
        mode = "auto"

    if mode == "image":
        return probe_only(env, log)
    if mode == "auto" and mcp_is_serving(env):
        log.info("mcp: stack already serving (baked image or warm guest)")
        return True

    if not skip_deps:
        if not ensure_python_deps(env, log):
            log.error("mcp bring-up: python deps failed")
            return False
        deps_ok = ensure_node_and_uv(env, log) and ensure_distractor_servers(env, log)
        if not deps_ok:
            msg = ("node/uv/distractor servers unavailable -- the 28 distractor tools "
                   "will be missing, so TIR is NOT comparable to the paper")
            if require_distractors:
                log.error("mcp bring-up: %s", msg)
                return False
            log.warning("mcp bring-up: %s", msg)

    if not push_bundle(env, log):
        return False
    if not start_soffice_uno(env, log):
        log.error("mcp bring-up: UNO :2002 down -- all 68 LibreOffice tools will fail")
    return start_server(env, log)

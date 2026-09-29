#!/usr/bin/env bash
# Prefetch OSWorld task assets into <OSWorld>/cache/<task_id>/... via
# scripts/python/osworld/cache.py
#
# 平台版:锚定 env_infra 自带的 OSWorld 子模块数据 (env_infra/OSWorld/evaluation_examples),
# 调用复制到 scripts/python/osworld/ 下的 cache.py。上游 OSWorld/scripts/ 只读、勿用。
#
# cache.py 用 requests/curl 下载，二者都尊重 http_proxy/https_proxy 环境变量，
# 所以代理通过下面的 export 注入（脚本本身无 --proxy 参数）。
#
# 用法 (可在任意目录执行):
#   bash scripts/bash/osworld/prefetch_cache.sh                 # 默认 test_all.json 全量预取
#   bash scripts/bash/osworld/prefetch_cache.sh --dry-run       # 只解析清单不下载
#   TEST_META=evaluation_examples/test_small.json \
#       bash scripts/bash/osworld/prefetch_cache.sh             # 换任务清单(相对 OSWorld 根)
#   WORKERS=4 bash scripts/bash/osworld/prefetch_cache.sh       # 调并发
#   PROXY=http://127.0.0.1:3128 bash scripts/bash/osworld/prefetch_cache.sh  # 显式指定本机代理
#   PROXY= bash scripts/bash/osworld/prefetch_cache.sh           # 强制直连(本机不通,会失败)
#   bash scripts/bash/osworld/prefetch_cache.sh --use-curl      # 用 curl 而非 requests
# 任何额外参数都会原样透传给 cache.py。

set -euo pipefail

# --- 代理 ---------------------------------------------------------------
# 默认使用本机代理；用 PROXY= 禁用代理，或传入其他代理地址。
PROXY="${PROXY-http://127.0.0.1:3128}"
export http_proxy="$PROXY"
export https_proxy="$PROXY"
export HTTP_PROXY="$PROXY"
export HTTPS_PROXY="$PROXY"
# 本地/内网不要走代理
export no_proxy="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export NO_PROXY="$no_proxy"

# 可选: 走 HF 国内镜像 (cache.py 的 maybe_replace_hf_endpoint 会识别 hf-mirror.com)
# 默认关闭(已有代理直连 huggingface.co)。要启用就把下一行取消注释:
# export HF_ENDPOINT="https://hf-mirror.com"

# --- 路径解析 ----------------------------------------------------------
# 脚本在 env_infra/scripts/bash/osworld/ 下,往上三级 = env_infra 根
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_INFRA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OSWORLD_ROOT="$ENV_INFRA_ROOT/OSWorld"
CACHE_PY="$ENV_INFRA_ROOT/scripts/python/osworld/cache.py"

if [[ ! -f "$CACHE_PY" ]]; then
  echo "[prefetch] ERROR: 找不到 cache.py: $CACHE_PY" >&2
  exit 1
fi
if [[ ! -d "$OSWORLD_ROOT/evaluation_examples" ]]; then
  echo "[prefetch] ERROR: 找不到数据目录: $OSWORLD_ROOT/evaluation_examples" >&2
  exit 1
fi

# 以 OSWorld 根为工作目录 (cache.py 的相对默认路径都相对它解析)
cd "$OSWORLD_ROOT"

# --- 可调参数 (环境变量覆盖, 路径相对 OSWorld 根) ----------------------
TEST_META="${TEST_META:-evaluation_examples/test_all.json}"
EXAMPLES_DIR="${EXAMPLES_DIR:-evaluation_examples/examples}"
CACHE_DIR="${CACHE_DIR:-cache}"
WORKERS="${WORKERS:-2}"          # 代理不稳就保持低并发
RETRIES="${RETRIES:-8}"
TIMEOUT="${TIMEOUT:-900}"
MANIFEST_OUT="${MANIFEST_OUT:-cache/prefetch_manifest.json}"
PYTHON="${PYTHON:-python3}"

echo "[prefetch] env_infra  = $ENV_INFRA_ROOT"
echo "[prefetch] osworld    = $OSWORLD_ROOT"
echo "[prefetch] cache.py   = $CACHE_PY"
echo "[prefetch] proxy      = ${PROXY:-<直连>}"
# CACHE_DIR 相对 OSWorld 根解析(上面已 cd 过去), 打印绝对路径避免误会:
# 这个位置必须和运行时一致 —— world.yaml:52 的 cache_dir 默认 "OSWorld/cache",
# 是相对【env_infra 根】(node 的 cwd, start_cluster_node.sh:8 cd 过去的)解析的,
# 正好等于这里的 <OSWorld 根>/cache。两者对得上, 预取才有意义。
echo "[prefetch] cache_dir  = $OSWORLD_ROOT/$CACHE_DIR"
echo "[prefetch]              (运行时 world.yaml cache_dir=OSWorld/cache 解析到同一目录 ✅)"
echo "[prefetch] test_meta  = $TEST_META   workers=$WORKERS retries=$RETRIES timeout=$TIMEOUT"
echo "[prefetch] 验证连通性..."
# 注意: PROXY 为空时不能传 -x "" —— 那和「不传 -x」语义不同, 某些 curl 版本会报错。
# 所以分开构造。
if [[ -n "$PROXY" ]]; then
  _probe=(curl -sS -m 15 -x "$PROXY")
else
  _probe=(curl -sS -m 15 --noproxy '*')
fi
if "${_probe[@]}" -o /dev/null \
        -w "  -> huggingface.co HTTP=%{http_code} time=%{time_total}s\n" \
        https://huggingface.co/ ; then
  :
else
  echo "  !! 探测失败，下面的下载很可能会失败。请检查网络/代理 (${PROXY:-直连}) 是否可用。" >&2
fi

exec "$PYTHON" "$CACHE_PY" \
  --test-meta   "$TEST_META" \
  --examples-dir "$EXAMPLES_DIR" \
  --cache-dir   "$CACHE_DIR" \
  --workers     "$WORKERS" \
  --retries     "$RETRIES" \
  --timeout     "$TIMEOUT" \
  --manifest-out "$MANIFEST_OUT" \
  "$@"

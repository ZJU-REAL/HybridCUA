#!/usr/bin/env bash
# =============================================================================
# 把 Blender 和 CUA-Gym 任务运行依赖烤进 golden qcow2。
#
# 【为什么必须烤进镜像】providers 把 qcow2 以 :ro 挂进容器作为 /System.qcow2，而
#   /run/install.sh:191 无条件执行
#       qemu-img create -f qcow2 -b /System.qcow2 -F qcow2 /boot.qcow2
#   qemu 只 boot `-hda /boot.qcow2`。也就是说挂进去的那份 qcow2 **永远只是
#   backing file**，guest 的写入全落在容器 overlayfs 上的 /boot.qcow2 里，随
#   `docker rm` 一起消失。改挂载模式(rw)没有任何用 —— 每个 reset 都是全新容器，
#   所以 episode 期间装的东西活不过一次 reset。仓库既有先例：
#   cluster/client/cua_gym/tasks.py:115「deps are pre-installed in the custom qcow2」。
#
# 【两种模式，分别在不同机器上跑】
#   bake   —— 用 virt-customize 离线改盘，**不需要启动 VM**。
#             这是 cua-h/docs/CLAUDE.md:527 记的项目「主方案」，
#             Ubuntu-cua-gym.qcow2 本身就是这么烤出来的。
#             需要：一台普通机器(裸机/VM，内核与 /lib/modules 匹配) + libguestfs-tools。
#   verify —— 起一个一次性 osworld 容器，在**生产同款环境**里验收渲染。
#             需要：docker + 本仓库 env_infra。
#
#   bake 回答不了「EEVEE 在这台无 GPU 的 guest 上到底渲不渲得出来」——
#   那必须启动 VM 才知道。所以两个模式都要跑。
#
# 【用法】
#   # A 机：前置自检（不改任何东西）
#   bash bake_blender_image.sh --preflight
#   # A 机：烤（约 20-40 分钟，大部分是 apt 下载和 virt-sparsify）
#   bash bake_blender_image.sh bake
#   # B 机：验收（把新镜像拷回来之后）
#   bash bake_blender_image.sh verify
#
# 【环境变量】
#   BASE_QCOW   源镜像   默认 <env_infra>/cua_gym_data/Ubuntu-cua-gym.qcow2
#   GOLD_QCOW   产出镜像 默认 <env_infra>/cua_gym_data/Ubuntu-cua-gym-blender.qcow2
#   MIRROR      首选 apt 镜像站                默认 https://mirrors.tuna.tsinghua.edu.cn
#   APT_MIRRORS offline 模式依次尝试的 apt 镜像（空格分隔）。默认在 MIRROR
#               失败后尝试腾讯、阿里、中科大和 Ubuntu 官方源。
#   PIP_INDEX   首选 pip 镜像                  默认清华 PyPI
#   PIP_INDEXES offline 模式依次尝试的 pip index（空格分隔）。
#   PROXY       需要走代理时设，例如 http://127.0.0.1:3128；默认空
#   INSTALL_MODE 依赖安装方式：auto（默认，探测 guest 出站网络）、online、offline。
#                offline 在宿主下载 deb/wheel，再复制进 qcow2 安装，适合
#                libguestfs appliance 没有默认路由的机器。
#   HOST_PYTHON offline 模式在宿主下载 wheel 使用的 Python。默认自动寻找
#               一个支持 `-m pip` 的系统/Conda Python；sudo 清理 PATH 时可显式传入。
#   OFFLINE_CACHE_DIR  宿主离线包缓存目录；默认与产出镜像同目录。
#   RUNTIME_APT_PACKAGES / RUNTIME_PIP_PACKAGES
#               烤入 guest 的系统包 / Python 包。默认覆盖 cua-gym-local
#               中启用的 skills（忽略 drawio/excalidraw/grafana/mock_websites/
#               openshot/overleaf/penpot）及当前 RLVR reward 的运行依赖。
#               Playwright 只安装 Python/CDP 客户端，不下载自带 Chromium；
#               任务连接镜像已有的 Chrome 130。
#   SKIP_SPARSIFY=1  跳过 virt-sparsify（省一次 23G 拷贝，但镜像会更大）
#   PORT/VNC_PORT    verify 模式用的宿主端口，默认 15000/18006
#                    ★ 必须避开 pool 的分配区间 5000-9999，否则会和真实任务抢端口
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_INFRA="${ENV_INFRA:-$(cd "${HERE}/../../.." && pwd)}"

BASE_QCOW="${BASE_QCOW:-${ENV_INFRA}/cua_gym_data/Ubuntu-cua-gym.qcow2}"
GOLD_QCOW="${GOLD_QCOW:-${ENV_INFRA}/cua_gym_data/Ubuntu-cua-gym-blender.qcow2}"
MIRROR="${MIRROR:-https://mirrors.tuna.tsinghua.edu.cn}"
# pip 源单独一个变量：沿用仓库既有约定（tasks.py:84 / cua-h/docs/CLAUDE.md:537 都用这个）。
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
APT_MIRRORS="${APT_MIRRORS:-${MIRROR} https://mirrors.cloud.tencent.com https://mirrors.aliyun.com https://mirrors.ustc.edu.cn http://archive.ubuntu.com}"
PIP_INDEXES="${PIP_INDEXES:-${PIP_INDEX} https://mirrors.cloud.tencent.com/pypi/simple https://mirrors.aliyun.com/pypi/simple https://pypi.org/simple}"
PROXY="${PROXY:-}"
GUEST_DNS="${GUEST_DNS:-}"
INSTALL_MODE="${INSTALL_MODE:-auto}"
HOST_PYTHON="${HOST_PYTHON:-}"
OFFLINE_CACHE_DIR="${OFFLINE_CACHE_DIR:-$(dirname "${GOLD_QCOW}")/.bake-offline-cache-jammy-amd64}"
SKIP_SPARSIFY="${SKIP_SPARSIFY:-0}"
# xcftools 最后发布于旧版 Ubuntu，Jammy 仓库没有该包。XCF 文件由已安装的
# GIMP 和 Python gimpformats 处理，不把 xcfinfo/xcf2png 当作 Jammy 硬依赖。
# libreoffice/cv2/skimage 已在基础镜像中，继续要求对应 apt 元包会额外拉入约
# 2.2G 的 Java、LLVM、OpenCV/GDAL 等依赖。门禁仍会验证命令和 Python import。
RUNTIME_APT_PACKAGES="${RUNTIME_APT_PACKAGES:-blender gimp vlc ffmpeg poppler-utils xclip xdotool wmctrl socat sqlite3 pulseaudio-utils dconf-cli libglib2.0-bin libsndfile1 fonts-dejavu-core fonts-liberation gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-libav gir1.2-gstreamer-1.0 python3-pip python3-dev build-essential cython3 python3-uno python3-gi python3-cairo python3-pyatspi python3-xlib}"
# odfpy 只发布 sdist，没有可供 --only-binary 下载的 wheel；基础镜像已有 odf，
# 且 python3-odf 已随系统依赖存在，因此不再通过 pip 重装。
# Ubuntu 22.04 的 python3-opencv 是按 NumPy 1.x ABI 构建的。NumPy 2.x 会让
# cv2 在门禁或 reward 运行时出现 ABI 导入错误，因此固定到最新的 1.x。
RUNTIME_PIP_PACKAGES="${RUNTIME_PIP_PACKAGES:-Pillow numpy==1.26.4 openpyxl pandas python-pptx python-docx PyMuPDF pikepdf reportlab requests beautifulsoup4 lxml rapidfuzz==3.10.1 tldextract==5.1.3 librosa==0.10.2.post1 scipy scikit-image ImageHash==4.3.2 playwright==1.48.0 pypdf==5.1.0 PyPDF2 pdfplumber PyAutoGUI}"
RUNTIME_PIP_BUILD_PACKAGES="${RUNTIME_PIP_BUILD_PACKAGES:-fastdtw==0.3.4}"

IMAGE="${IMAGE:-happysixd/osworld-docker}"
NAME="${NAME:-blender-bake}"
PORT="${PORT:-15000}"
VNC_PORT="${VNC_PORT:-18006}"
BOOT_WAIT="${BOOT_WAIT:-180}"     # /execute 开始应答（约 15s）
X_WAIT="${X_WAIT:-180}"           # 额外等 X :0 socket 出现（gnome 会话登录后才有）

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for cand in "${ENV_INFRA}/../venvs/env_infra/bin/python" "$(command -v python3 || true)"; do
    [[ -x "${cand}" ]] && { PYTHON_BIN="${cand}"; break; }
  done
fi

LOG_DIR="${LOG_DIR:-${ENV_INFRA}/logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BAKE_LOG="${LOG_DIR}/bake-blender-${STAMP}.log"

log()  { printf '\n[bake %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\n[bake %s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '\n[bake] ERROR: %s\n' "$*" >&2; exit 1; }

MODE="${1:-}"
[[ "${MODE}" == "--preflight" || "${MODE}" == "preflight" ]] && MODE=preflight
[[ "${MODE}" == "--verify-only" ]] && MODE=verify
[[ -z "${MODE}" ]] && MODE=bake
case "${MODE}" in bake|verify|preflight) ;; *) die "未知模式 '${MODE}'（bake | verify | preflight）";; esac

# ===========================================================================
#  A 机 —— 前置自检
# ===========================================================================
preflight() {
  log "=== preflight ==="
  log "     apt runtime packages: ${RUNTIME_APT_PACKAGES}"
  log "     pip runtime packages: ${RUNTIME_PIP_PACKAGES}"
  log "     pip compiled packages: ${RUNTIME_PIP_BUILD_PACKAGES}"
  local ok=0

  # libguestfs 的硬前提：appliance 是按 `uname -r` 找内核和模块来构建的。
  # 容器里跑通常就死在这 —— 镜像里的 /lib/modules 和宿主内核对不上。
  local krel; krel="$(uname -r)"
  if [[ -d "/lib/modules/${krel}" ]]; then
    log "OK   kernel modules: /lib/modules/${krel}"
  else
    warn "FAIL /lib/modules/${krel} 不存在（宿主内核与镜像不匹配 -> supermin 无法构建 appliance）"
    warn "     现有: $(ls /lib/modules/ 2>/dev/null | tr '\n' ' ')"
    warn "     这是容器环境最典型的坑。换一台普通机器/VM。"
    ok=1
  fi
  if [[ -f "/boot/vmlinuz-${krel}" ]]; then
    log "OK   kernel image: /boot/vmlinuz-${krel}"
  else
    warn "FAIL /boot/vmlinuz-${krel} 不存在"; ok=1
  fi

  for c in virt-customize qemu-img; do
    if command -v "$c" >/dev/null; then log "OK   $c -> $(command -v $c)"
    else warn "FAIL 缺 $c（apt-get install -y libguestfs-tools qemu-utils）"; ok=1; fi
  done
  command -v virt-sparsify >/dev/null && log "OK   virt-sparsify" \
    || warn "缺 virt-sparsify（设 SKIP_SPARSIFY=1 可跳过瘦身）"

  [[ "${SKIP_SPARSIFY}" == "1" ]] || command -v virt-sparsify >/dev/null || true

  if [[ -w /dev/kvm ]]; then log "OK   /dev/kvm 可写（appliance 走 KVM，快）"
  else warn "/dev/kvm 不可写 —— appliance 会退化成 TCG，慢十倍但不报错"; fi

  [[ "$(id -u)" == "0" ]] && log "OK   root" || { warn "FAIL 需要 root（libguestfs 要挂文件系统）"; ok=1; }

  # 磁盘：cp 一份 23G + apt 增量 ~2G + virt-sparsify 再一份 ~24G
  local need_kb avail_kb base_size_bytes
  base_size_bytes="$(stat -c%s "${BASE_QCOW}" 2>/dev/null || echo 24942870528)"
  need_kb=$(( (base_size_bytes / 1024) * 2 + 4 * 1024 * 1024 ))
  avail_kb="$(df -Pk "$(dirname "${GOLD_QCOW}")" | tail -1 | awk '{print $4}')"
  log "     workdir $(dirname "${GOLD_QCOW}") 可用 $((avail_kb/1024/1024))G, 需要约 $((need_kb/1024/1024))G"
  [[ "${avail_kb}" -ge "${need_kb}" ]] || { warn "FAIL 空间不足"; ok=1; }

  avail_kb="$(df -Pk "${TMPDIR:-/tmp}" | tail -1 | awk '{print $4}')"
  log "     TMPDIR ${TMPDIR:-/tmp} 可用 $((avail_kb/1024/1024))G（appliance 需要 ~3G）"
  [[ "${avail_kb}" -ge $((3*1024*1024)) ]] || warn "TMPDIR 偏小，建议 export TMPDIR=/var/tmp"

  [[ -n "${PYTHON_BIN}" ]] && log "OK   python: ${PYTHON_BIN}" || warn "缺 python3（verify 模式需要）"
  command -v docker >/dev/null && log "OK   docker（verify 模式需要）" || warn "缺 docker（verify 模式需要）"

  if [[ "${ok}" != "0" ]]; then
    die "preflight 未通过 —— 别急着跑 bake"
  fi
  log "preflight 通过。接下来："
  log "  export LIBGUESTFS_BACKEND=direct LIBGUESTFS_MEMSIZE=2048 TMPDIR=/var/tmp"
  log "  bash $0 bake"
}

# 在宿主准备 guest 可直接安装的 deb/wheel。APT 使用 guest 的 dpkg status 做
# 依赖求解，因此只下载 guest 真正缺少或需要升级的包。整个缓存可跨失败重试复用。
prepare_offline_payload() {
  command -v apt-get >/dev/null || die "离线安装需要宿主有 apt-get"
  command -v dpkg-deb >/dev/null || die "离线安装需要宿主有 dpkg-deb"

  # sudo 通常会用 secure_path 覆盖调用者的 Conda PATH，所以不能写死
  # `python3 -m pip`。优先使用调用者显式传入的 HOST_PYTHON，否则逐个寻找
  # 真正带 pip 的 Python；失败的候选（如 Ubuntu 未安装 python3-pip 的
  # /usr/bin/python3）继续跳过。
  local host_python="${HOST_PYTHON}" cand
  if [[ -n "${host_python}" ]]; then
    [[ -x "${host_python}" ]] || die "HOST_PYTHON 不可执行: ${host_python}"
    "${host_python}" -m pip --version >/dev/null 2>&1 \
      || die "HOST_PYTHON 没有可用的 pip: ${host_python}"
  else
    local host_python_candidates=(
      "$(command -v python3 2>/dev/null || true)"
      "$(command -v python 2>/dev/null || true)"
      /root/miniconda3/bin/python
      /root/miniforge3/bin/python
      /root/anaconda3/bin/python
      /opt/conda/bin/python
      /usr/local/bin/python3
      /usr/bin/python3
    )
    # 兼容自定义名称的 root 下 Conda 安装目录。
    for cand in /root/*conda*/bin/python /root/*forge*/bin/python; do
      [[ -e "${cand}" ]] && host_python_candidates+=("${cand}")
    done
    host_python=""
    for cand in "${host_python_candidates[@]}"; do
      [[ -n "${cand}" && -x "${cand}" ]] || continue
      if "${cand}" -m pip --version >/dev/null 2>&1; then
        host_python="${cand}"
        break
      fi
    done
    [[ -n "${host_python}" ]] || die "离线安装需要宿主有带 pip 的 Python；请在 sudo env 中传 HOST_PYTHON=\"\$(command -v python3)\""
  fi
  log "    宿主 wheel 下载 Python: ${host_python} ($("${host_python}" --version 2>&1))"

  local cache="${OFFLINE_CACHE_DIR}"
  local deb_dir="${cache}/debs" wheel_dir="${cache}/wheels"
  local manifest wanted
  wanted="$(printf '%s\n%s\n%s\n%s\n%s\n' "${APT_MIRRORS}" "${PIP_INDEXES}" "${RUNTIME_APT_PACKAGES}" "${RUNTIME_PIP_PACKAGES}" "${RUNTIME_PIP_BUILD_PACKAGES}" | sha256sum | awk '{print $1}')"
  manifest="${cache}/.complete"
  if [[ -s "${manifest}" && "$(cat "${manifest}")" == "${wanted}" ]] \
      && compgen -G "${deb_dir}/*.deb" >/dev/null \
      && compgen -G "${wheel_dir}/*" >/dev/null; then
    log "    复用宿主离线包缓存: ${cache}"
    return
  fi

  log "    guest 无出站网络；在宿主准备离线 deb/wheel: ${cache}"
  # 缓存按阶段保留。前一次若只在 pip 下载失败，已下载的几百 MB deb 会被
  # apt 直接复用，不再因为一次 wheel 错误全部重下。
  rm -f "${manifest}"
  mkdir -p "${deb_dir}" "${wheel_dir}" "${cache}/state/lists/partial" \
    "${cache}/apt-cache/archives/partial"

  # 有 virt-cat 时让宿主 APT 看见 guest 已安装包，避免重新下载整套桌面系统。
  if command -v virt-cat >/dev/null; then
    virt-cat -a "${GOLD_QCOW}" /var/lib/dpkg/status > "${cache}/status"
  else
    warn "宿主缺 virt-cat；APT 将按空系统求依赖，下载量会明显增大"
    : > "${cache}/status"
  fi

  local apt_host_opts=(
    -o "Dir::Etc::sourcelist=${cache}/sources.list"
    -o "Dir::Etc::sourceparts=-"
    -o "Dir::State=${cache}/state"
    -o "Dir::State::status=${cache}/status"
    -o "Dir::Cache=${cache}/apt-cache"
    -o "APT::Architecture=amd64"
    -o "APT::Architectures=amd64"
    -o "APT::Sandbox::User=root"
  )
  [[ -n "${PROXY}" ]] && apt_host_opts+=(
    -o "Acquire::http::Proxy=${PROXY}"
    -o "Acquire::https::Proxy=${PROXY}"
  )
  # 将前一次成功下载的 deb 放回 APT archive。apt-get 会校验并只补缺失文件。
  find "${deb_dir}" -maxdepth 1 -type f -name '*.deb' \
    -exec mv -t "${cache}/apt-cache/archives" {} + 2>/dev/null || true
  local apt_mirror apt_repo selected_apt_mirror=""
  for apt_mirror in ${APT_MIRRORS}; do
    case "${apt_mirror%/}" in
      */ubuntu) apt_repo="${apt_mirror%/}" ;;
      *) apt_repo="${apt_mirror%/}/ubuntu" ;;
    esac
    cat > "${cache}/sources.list" <<EOF
deb [arch=amd64] ${apt_repo} jammy main restricted universe multiverse
deb [arch=amd64] ${apt_repo} jammy-updates main restricted universe multiverse
deb [arch=amd64] ${apt_repo} jammy-security main restricted universe multiverse
deb [arch=amd64] ${apt_repo} jammy-backports main restricted universe multiverse
EOF
    rm -rf "${cache}/state/lists"
    mkdir -p "${cache}/state/lists/partial"
    log "    尝试宿主 APT 镜像: ${apt_repo}"
    if apt-get "${apt_host_opts[@]}" -o APT::Update::Error-Mode=any update; then
      selected_apt_mirror="${apt_repo}"
      break
    fi
    warn "APT 镜像不可用，尝试下一个: ${apt_repo}"
  done
  [[ -n "${selected_apt_mirror}" ]] \
    || die "所有宿主 APT 镜像均不可用；可用 APT_MIRRORS='https://可访问镜像' 覆盖"
  log "    宿主 APT 镜像可用: ${selected_apt_mirror}"
  apt-get "${apt_host_opts[@]}" --download-only --no-upgrade install -y ${RUNTIME_APT_PACKAGES}
  find "${cache}/apt-cache/archives" -maxdepth 1 -type f -name '*.deb' -exec mv -t "${deb_dir}" {} +
  compgen -G "${deb_dir}/*.deb" >/dev/null || die "宿主 APT 没有产出 deb；检查 ${cache}/status 和软件源"
  # 生成 file: APT 仓库。即使缓存里保留了旧一轮下载的额外 deb，guest 也只会
  # 按 RUNTIME_APT_PACKAGES 求解并安装真正需要的包。
  : > "${deb_dir}/Packages"
  local deb
  for deb in "${deb_dir}"/*.deb; do
    dpkg-deb -f "${deb}" >> "${deb_dir}/Packages"
    printf 'Filename: ./%s\nSize: %s\nSHA256: %s\n\n' \
      "$(basename "${deb}")" "$(stat -c%s "${deb}")" \
      "$(sha256sum "${deb}" | awk '{print $1}')" >> "${deb_dir}/Packages"
  done
  gzip -9c "${deb_dir}/Packages" > "${deb_dir}/Packages.gz"

  # 目标是 Ubuntu 22.04 的 CPython 3.10 amd64。主依赖只收 wheel；PyAutoGUI
  # 及其纯 Python 依赖、fastdtw 单独收 sdist，在 guest 里用已烤入的编译链构建。
  local binary_pkgs=() pkg
  for pkg in ${RUNTIME_PIP_PACKAGES}; do
    case "${pkg,,}" in pyautogui*) ;; *) binary_pkgs+=("${pkg}");; esac
  done
  local pip_proxy=()
  [[ -n "${PROXY}" ]] && pip_proxy=(--proxy "${PROXY}")
  local pip_index selected_pip_index=""
  for pip_index in ${PIP_INDEXES}; do
    mkdir -p "${wheel_dir}"
    log "    尝试宿主 pip 镜像: ${pip_index}"
    if "${host_python}" -m pip download -d "${wheel_dir}" -i "${pip_index}" "${pip_proxy[@]}" \
        --timeout 120 --retries 10 \
        --only-binary=:all: --python-version 310 --implementation cp \
        --abi cp310 --abi abi3 --abi none \
        --platform manylinux_2_35_x86_64 --platform manylinux_2_34_x86_64 \
        --platform manylinux_2_31_x86_64 --platform manylinux_2_28_x86_64 \
        --platform manylinux2014_x86_64 --platform manylinux2010_x86_64 \
        --platform manylinux1_x86_64 --platform linux_x86_64 \
        "${binary_pkgs[@]}" \
      && "${host_python}" -m pip download -d "${wheel_dir}" -i "${pip_index}" "${pip_proxy[@]}" \
        --timeout 120 --retries 10 \
        --no-deps --no-binary=:all: \
        PyAutoGUI==0.9.54 PyMsgBox PyTweening PyScreeze PyGetWindow MouseInfo pyrect pyperclip ${RUNTIME_PIP_BUILD_PACKAGES}; then
      selected_pip_index="${pip_index}"
      break
    fi
    warn "pip 镜像不可用或包不完整，尝试下一个: ${pip_index}"
  done
  [[ -n "${selected_pip_index}" ]] \
    || die "所有宿主 pip 镜像均不可用；可用 PIP_INDEXES='https://可访问索引/simple' 覆盖"
  log "    宿主 pip 镜像可用: ${selected_pip_index}"

  printf '%s\n' "${wanted}" > "${manifest}"
  log "    离线包就绪: deb=$(find "${deb_dir}" -name '*.deb' | wc -l), python=$(find "${wheel_dir}" -type f | wc -l)"
}

# ===========================================================================
#  A 机 —— virt-customize 烤镜像
# ===========================================================================
bake() {
  [[ "$(id -u)" == "0" ]] || die "bake 需要 root"
  command -v virt-customize >/dev/null || die "缺 virt-customize（apt-get install -y libguestfs-tools qemu-utils）"
  command -v qemu-img >/dev/null      || die "缺 qemu-img（apt-get install -y qemu-utils）"
  [[ -f "${BASE_QCOW}" ]] || die "源镜像不存在: ${BASE_QCOW}"

  export LIBGUESTFS_BACKEND="${LIBGUESTFS_BACKEND:-direct}"
  export LIBGUESTFS_MEMSIZE="${LIBGUESTFS_MEMSIZE:-2048}"
  export TMPDIR="${TMPDIR:-/var/tmp}"

  mkdir -p "${LOG_DIR}"

  # ---------------------------------------------------------------- 复制
  if [[ -f "${GOLD_QCOW}" ]]; then
    if [[ "${REUSE_GOLD:-0}" == "1" ]]; then
      warn "复用已存在的产出（REUSE_GOLD=1）：${GOLD_QCOW}"
    else
      die "产出已存在: ${GOLD_QCOW}
     要重烤请先删掉它；若上次没走到最后、确定没写入，用 REUSE_GOLD=1 复用。"
    fi
  else
    log "=== 复制源镜像 -> 产出（$(du -h "${BASE_QCOW}" | cut -f1)，几分钟）==="
    log "    ★ 复制这一步就是「不用停 node」的关键：改的是副本，"
    log "      绝不会碰到正被几十个容器当 backing file 的那份活镜像。"
    cp --reflink=auto "${BASE_QCOW}" "${GOLD_QCOW}.partial"
    mv "${GOLD_QCOW}.partial" "${GOLD_QCOW}"
    log "复制完成: $(du -h "${GOLD_QCOW}" | cut -f1)"
  fi

  # ---------------------------------------------------------------- Step 1 侦察
  # 每条 --run-command 都以一条必定成功的命令收尾，否则 virt-customize 会中途 abort。
  log "=== Step 1/4 侦察（结果见 ${BAKE_LOG}） ==="
  virt-customize -a "${GOLD_QCOW}" \
    --run-command '. /etc/os-release; echo "PROBE_OS: ${PRETTY_NAME}"' \
    --run-command 'echo "PROBE_ARCH: $(dpkg --print-architecture)"' \
    --run-command 'echo "PROBE_BLENDER_BIN: $(command -v blender || echo NONE)"; blender --version 2>&1 | head -2 || true' \
    --run-command 'for m in cv2 numpy PIL imagehash skimage playwright pypdf openpyxl pandas pptx docx odf pymupdf fitz pikepdf reportlab requests bs4 lxml rapidfuzz tldextract librosa fastdtw scipy PyPDF2 pdfplumber Xlib gi uno pyatspi pyautogui; do python3 -c "import $m" 2>/dev/null && echo "PROBE_PY_OK: $m" || echo "PROBE_PY_MISSING: $m"; done; true' \
    --run-command 'for c in blender gimp vlc libreoffice ffmpeg ffprobe xcfinfo xcf2png pdftoppm xclip xdotool wmctrl socat sqlite3 pactl dconf gsettings google-chrome code; do command -v "$c" >/dev/null && echo "PROBE_BIN_OK: $c=$(command -v "$c")" || echo "PROBE_BIN_MISSING: $c"; done; true' \
    --run-command 'echo "PROBE_FREE_B_ROOT: $(df --output=avail -B1 / | tail -1)"' \
    --run-command 'echo "PROBE_DF:"; df -h / /home 2>&1 | head -5' \
    --run-command 'echo "PROBE_XSOCK: $(ls /tmp/.X11-unix/X0 2>/dev/null || echo NONE)"' \
    --run-command 'echo "PROBE_XAUTH: $(ls /run/user/1000/gdm/Xauthority 2>/dev/null || echo NONE)"' \
    --run-command 'echo "PROBE_DISPLAY_MGR: $(cat /etc/X11/default-display-manager 2>/dev/null || echo NONE)"' \
    --run-command 'echo "PROBE_DONE"' \
    2>&1 | tee -a "${BAKE_LOG}" | grep -E 'PROBE_|^\[bake' || true

  # ---------------------------------------------------------------- Step 2 安装
  log "=== Step 2/4 安装 blender + RLVR 运行依赖（这是最慢的一步） ==="
  local apt_env="export DEBIAN_FRONTEND=noninteractive"
  local proxy_apt=""
  # 离线 guest 的 /etc/resolv.conf 往往指向 127.0.0.53，但 appliance 中没有
  # systemd-resolved。优先透传宿主实际 DNS；GUEST_DNS 可传空格/逗号分隔列表。
  local guest_dns="${GUEST_DNS}"
  if [[ -z "${guest_dns}" ]]; then
    guest_dns="$(awk '$1 == "nameserver" && $2 !~ /^127\./ { printf "%s ", $2 }' /etc/resolv.conf 2>/dev/null)"
  fi
  guest_dns="${guest_dns//,/ }"
  guest_dns="$(xargs <<<"${guest_dns}" 2>/dev/null || true)"
  [[ -n "${guest_dns}" ]] || guest_dns="10.0.2.3 223.5.5.5 8.8.8.8"
  log "    guest 临时 DNS: ${guest_dns}（可用 GUEST_DNS=... 覆盖）"
  local mirror_host pip_host host ip guest_hosts=""
  mirror_host="${MIRROR#*://}"; mirror_host="${mirror_host%%/*}"
  pip_host="${PIP_INDEX#*://}"; pip_host="${pip_host%%/*}"
  for host in "${mirror_host}" "${pip_host}"; do
    ip="$(getent ahostsv4 "${host}" 2>/dev/null | awk 'NR == 1 { print $1 }')"
    [[ -n "${ip}" ]] && guest_hosts+="${ip} ${host}"$'\n'
  done
  if [[ -n "${guest_hosts}" ]]; then
    log "    宿主预解析 hosts: $(tr '\n' ';' <<<"${guest_hosts}")"
  else
    warn "宿主未能预解析 apt/pip 镜像域名；将只依赖 guest DNS"
  fi
  if [[ -n "${PROXY}" ]]; then
    proxy_apt="printf 'Acquire::http::Proxy \"${PROXY}\";\nAcquire::https::Proxy \"${PROXY}\";\n' > /etc/apt/apt.conf.d/99bake-proxy"
  fi
  # 系统侧覆盖启用 skills 需要的桌面应用和 CLI。Chrome 与 VS Code 由基础
  # 镜像提供，门禁只检查，不从 Ubuntu 源替换。pip 只补 apt/基础镜像缺口。
  # 优先用 apt 的 python3-opencv/python3-skimage 和桌面绑定，保持 guest ABI。
  # ImageHash / playwright / pypdf 在当前生产 VM 缺失：
  #   - ImageHash 被媒体 reward 直接或可选地用于感知哈希；
  #   - playwright 只提供 Python/CDP 客户端，复用已有 google-chrome，不执行
  #     `playwright install`，避免再烤一套浏览器；
  #   - pypdf 补齐新包名，旧任务仍可继续使用已存在的 PyPDF2 fallback。
  # jammy 的 pip 22.0.2 不受 PEP668 限制。
  local pip_args="--no-cache-dir"
  [[ -n "${PROXY}" ]] && pip_args="${pip_args} --proxy ${PROXY}"

  local install_mode="${INSTALL_MODE}"
  case "${install_mode}" in
    auto)
      local probe_ip="${ip:-1.1.1.1}"
      log "    探测 libguestfs appliance 出站网络 (${probe_ip}:443)"
      if virt-customize --network -a "${GOLD_QCOW}" \
          --run-command "python3 -c 'import socket; s=socket.create_connection((\"${probe_ip}\",443),8); s.close()'" \
          >/dev/null 2>&1; then
        install_mode=online
      else
        install_mode=offline
      fi
      ;;
    online|offline) ;;
    *) die "INSTALL_MODE=${INSTALL_MODE} 无效，只能是 auto、online、offline" ;;
  esac
  log "    依赖安装模式: ${install_mode}"

  if [[ "${install_mode}" == "offline" ]]; then
    prepare_offline_payload
    local offline_guest_dir=/opt/bake-offline
    virt-customize -a "${GOLD_QCOW}" \
      --mkdir "${offline_guest_dir}" \
      --copy-in "${OFFLINE_CACHE_DIR}/debs:${offline_guest_dir}" \
      --copy-in "${OFFLINE_CACHE_DIR}/wheels:${offline_guest_dir}" \
      --run-command "printf 'deb [trusted=yes] file:${offline_guest_dir}/debs ./\\n' > /tmp/bake-offline.list; ${apt_env}; apt-get -o Dir::Etc::sourcelist=/tmp/bake-offline.list -o Dir::Etc::sourceparts=- update; apt-get -o Dir::Etc::sourcelist=/tmp/bake-offline.list -o Dir::Etc::sourceparts=- --no-upgrade install -y ${RUNTIME_APT_PACKAGES}" \
      --run-command "python3 -m pip install ${pip_args} --no-index --find-links=${offline_guest_dir}/wheels --find-links=/opt/cua_gym/vm_wheels --upgrade-strategy only-if-needed ${RUNTIME_PIP_PACKAGES}" \
      --run-command "python3 -m pip install ${pip_args} --no-index --find-links=${offline_guest_dir}/wheels --find-links=/opt/cua_gym/vm_wheels --no-build-isolation --force-reinstall --no-deps ${RUNTIME_PIP_BUILD_PACKAGES}" \
      --run-command "rm -rf ${offline_guest_dir} /tmp/bake-offline.list /tmp/bake-resolv.conf /tmp/bake-resolv.link /tmp/bake-hosts /tmp/bake-restore-resolv.sh /tmp/bake-sources.list; test -e /etc/resolv.conf || ln -s ../run/systemd/resolve/stub-resolv.conf /etc/resolv.conf; blender --version 2>&1 | head -3; echo INSTALL_DONE" \
      2>&1 | tee -a "${BAKE_LOG}"
  else
    virt-customize --network -a "${GOLD_QCOW}" \
    --run-command "${apt_env}
cp -n /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true
if test ! -e /tmp/bake-resolv.conf && test ! -e /tmp/bake-resolv.link; then
  if test -L /etc/resolv.conf; then
    readlink /etc/resolv.conf > /tmp/bake-resolv.link
  else
    cp /etc/resolv.conf /tmp/bake-resolv.conf 2>/dev/null || true
  fi
fi
test -e /tmp/bake-hosts || cp /etc/hosts /tmp/bake-hosts
rm -f /etc/resolv.conf
for ns in ${guest_dns}; do printf 'nameserver %s\n' "\$ns"; done > /etc/resolv.conf
printf 'options timeout:5 attempts:2\n' >> /etc/resolv.conf
cat >> /etc/hosts <<'BAKE_STATIC_HOSTS'
${guest_hosts}
BAKE_STATIC_HOSTS
cat > /tmp/bake-sources.list <<'BAKE_APT_SOURCES'
deb [arch=amd64] ${MIRROR%/}/ubuntu jammy main restricted universe multiverse
deb [arch=amd64] ${MIRROR%/}/ubuntu jammy-updates main restricted universe multiverse
deb [arch=amd64] ${MIRROR%/}/ubuntu jammy-security main restricted universe multiverse
deb [arch=amd64] ${MIRROR%/}/ubuntu jammy-backports main restricted universe multiverse
BAKE_APT_SOURCES
cat > /tmp/bake-restore-resolv.sh <<'BAKE_RESTORE_DNS'
#!/bin/sh
rm -f /etc/resolv.conf
if test -s /tmp/bake-resolv.link; then
  ln -s \"\$(cat /tmp/bake-resolv.link)\" /etc/resolv.conf
elif test -s /tmp/bake-resolv.conf; then
  cp /tmp/bake-resolv.conf /etc/resolv.conf
else
  ln -s ../run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
fi
rm -f /tmp/bake-resolv.conf /tmp/bake-resolv.link
if test -s /tmp/bake-hosts; then cp /tmp/bake-hosts /etc/hosts; fi
rm -f /tmp/bake-hosts
BAKE_RESTORE_DNS
chmod 755 /tmp/bake-restore-resolv.sh
${proxy_apt}
cat /tmp/bake-sources.list
echo 'DNS_CONFIG:'; cat /etc/resolv.conf" \
    --run-command "${apt_env}; for i in 1 2 3; do apt-get -o Dir::Etc::sourcelist=/tmp/bake-sources.list -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0 -o APT::Update::Error-Mode=any update && break; if test \"\$i\" = 3; then /tmp/bake-restore-resolv.sh; exit 1; fi; echo \"apt update attempt \$i failed; retrying\"; sleep 3; done" \
    --run-command "apt-cache show blender >/dev/null || { echo 'APT_GATE_FAIL: blender 不在当前索引中'; /tmp/bake-restore-resolv.sh; exit 1; }" \
    --run-command "${apt_env}; apt-get -o Dir::Etc::sourcelist=/tmp/bake-sources.list -o Dir::Etc::sourceparts=- install -y ${RUNTIME_APT_PACKAGES} || { /tmp/bake-restore-resolv.sh; exit 1; }" \
    --run-command "for i in 1 2 3; do python3 -m pip install -i ${PIP_INDEX} ${pip_args} --upgrade-strategy only-if-needed ${RUNTIME_PIP_PACKAGES} && break; if test \"\$i\" = 3; then /tmp/bake-restore-resolv.sh; exit 1; fi; echo \"pip runtime dependency attempt \$i failed; retrying\"; sleep 3; done" \
    --run-command "for i in 1 2 3; do python3 -m pip install -i ${PIP_INDEX} ${pip_args} --no-build-isolation --force-reinstall --no-deps ${RUNTIME_PIP_BUILD_PACKAGES} && break; if test \"\$i\" = 3; then /tmp/bake-restore-resolv.sh; exit 1; fi; echo \"pip compiled dependency attempt \$i failed; retrying\"; sleep 3; done" \
    --run-command '/tmp/bake-restore-resolv.sh; rm -f /tmp/bake-restore-resolv.sh /tmp/bake-sources.list' \
    --run-command 'blender --version 2>&1 | head -3 || true' \
    --run-command 'echo "INSTALL_DONE"' \
    2>&1 | tee -a "${BAKE_LOG}"
  fi

  # ---------------------------------------------------------------- Step 3 清理
  log "=== Step 3/4 清理（瘦身 + 移除临时代理配置） ==="
  virt-customize -a "${GOLD_QCOW}" \
    --run-command 'export DEBIAN_FRONTEND=noninteractive; apt-get clean; rm -rf /var/lib/apt/lists/*' \
    --run-command 'rm -rf /root/.cache/pip /home/user/.cache/pip 2>/dev/null; true' \
    --run-command 'rm -f /etc/apt/apt.conf.d/99bake-proxy' \
    --run-command 'echo "CLEAN_DONE"' \
    2>&1 | tee -a "${BAKE_LOG}" | grep -E 'CLEAN_DONE|^\[bake'

  # ---------------------------------------------------------------- Step 4 硬门禁
  # --run-command 非零退出会让 virt-customize 整体失败，所以这一组就是 CI 门。
  log "=== Step 4/4 硬门禁 ==="
  virt-customize -a "${GOLD_QCOW}" \
    --run-command 'out=$(blender --version 2>&1); printf "%s\n" "$out" | head -5
v=$(printf "%s\n" "$out" | grep -m1 -oE "Blender [0-9]+\\.[0-9]+\\.[0-9]+" || true); echo "GATE_VERSION: ${v:-NOT_FOUND}"
case "$v" in
  "Blender 3.0"*) ;;
  *) echo "GATE_FAIL: 期望 Blender 3.0.x，提取结果 [${v:-NOT_FOUND}]"; exit 1;;
esac' \
    --run-command 'python3 -c "import cv2,numpy,PIL,imagehash,skimage,pypdf,librosa,fastdtw,rapidfuzz,tldextract,bs4; from importlib.metadata import version; from playwright.sync_api import sync_playwright; from PIL import Image; assert str(imagehash.phash(Image.new(\"L\",(8,8),0))); assert callable(sync_playwright); assert version(\"PyAutoGUI\"); print(\"GATE_DEPS: runtime imports ok; NumPy=%s ImageHash=%s Playwright=%s PyAutoGUI=%s pypdf=%s\" % (numpy.__version__,version(\"ImageHash\"),version(\"playwright\"),version(\"PyAutoGUI\"),version(\"pypdf\")))"' \
    --run-command 'python3 -c "import uno; from com.sun.star.beans import PropertyValue; assert PropertyValue is not None; print(\"GATE_UNO: com.sun.star bridge ok\")"' \
    --run-command 'for c in blender gimp vlc libreoffice ffmpeg ffprobe pdftoppm xclip xdotool wmctrl socat sqlite3 pactl dconf gsettings google-chrome code; do command -v "$c" >/dev/null || { echo "GATE_BIN_MISSING: $c"; exit 1; }; done; echo "GATE_BINS: all skill commands present"' \
    --run-command 'test ! -e /etc/apt/apt.conf.d/99bake-proxy && echo "GATE_NO_STALE_PROXY: ok"' \
    --run-command 'echo "GATE_APT_HISTORY_TAIL:"; tail -12 /var/log/apt/history.log 2>/dev/null || echo "(no apt history)"' \
    --run-command 'echo "GATE_DONE"' \
    2>&1 | tee -a "${BAKE_LOG}" | grep -E 'GATE_|^\[bake|Error'

  # ---------------------------------------------------------------- 瘦身
  if [[ "${SKIP_SPARSIFY}" == "1" ]]; then
    warn "SKIP_SPARSIFY=1，跳过瘦身"
  else
    command -v virt-sparsify >/dev/null || die "缺 virt-sparsify（设 SKIP_SPARSIFY=1 跳过）"
    log "=== virt-sparsify（需要再一份完整拷贝的空间） ==="
    virt-sparsify --compress --tmp "${TMPDIR}" "${GOLD_QCOW}" "${GOLD_QCOW}.sparse"
    mv "${GOLD_QCOW}.sparse" "${GOLD_QCOW}"
  fi

  log "=== 镜像自检 ==="
  qemu-img check "${GOLD_QCOW}" || die "qemu-img check 报错，这份镜像不要用"
  qemu-img info  "${GOLD_QCOW}"
  # backing file 必须为空 —— 有 backing 的话部署到别的机器就起不来。
  if qemu-img info "${GOLD_QCOW}" | grep -qiE '^backing file:'; then
    die "产出仍带 backing file，不能作为独立镜像部署"
  fi
  log "backing file: 无（独立镜像）"

  cat <<EOF

============================================================
烤制完成
  产出:   ${GOLD_QCOW}  ($(du -h "${GOLD_QCOW}" | cut -f1))
  日志:   ${BAKE_LOG}

下一步（在 B 机上，生产同款容器环境里验收渲染）:
  bash $(basename "$0") verify

  ★ bake 只证明了「包装进去了」。EEVEE 在无 GPU 的 QEMU guest 上到底
    渲不渲得出来，必须启动 VM 才知道 —— 那是 verify 的活。
============================================================
EOF
}

# ===========================================================================
#  B 机 —— 生产同款容器验收
# ===========================================================================
exec_guest() {
  # $1 = shell 片段  $2 = 超时秒数。走 /execute（服务端硬上限 120s）。
  local payload
  payload="$("${PYTHON_BIN}" -c '
import json,sys
print(json.dumps({"command":["bash","-lc",sys.argv[1]],"shell":False}))' "$1")"
  curl -s -m "${2:-120}" -X POST "http://127.0.0.1:${PORT}/execute" \
    -H "Content-Type: application/json" -d "${payload}" \
  | "${PYTHON_BIN}" -c 'import json,sys
try: print(json.load(sys.stdin).get("output",""))
except Exception: pass'
}

run_guest_script() {
  # $1 = 多行 bash 脚本内容  $2 = 超时。走 /run_bash_script（timeout 可调，stderr 合并进 stdout）。
  local payload
  payload="$("${PYTHON_BIN}" -c '
import json,sys
print(json.dumps({"script":sys.argv[1],"timeout":int(sys.argv[2])}))' "$1" "$2")"
  curl -s -m "$(( $2 + 30 ))" -X POST "http://127.0.0.1:${PORT}/run_bash_script" \
    -H "Content-Type: application/json" -d "${payload}" \
  | "${PYTHON_BIN}" -c 'import json,sys
try: print(json.load(sys.stdin).get("output",""))
except Exception: pass'
}

start_container() {
  local mode="$1" qcow="$2"
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  local devices=()
  if [[ -e /dev/kvm ]]; then
    devices=(--device /dev/kvm)
  else
    warn "无 /dev/kvm，QEMU 走 TCG，会非常慢"
  fi
  log "启动 ${NAME} (${mode}) on $(basename "${qcow}")，端口 ${PORT}/${VNC_PORT}"
  # ${devices[@]+...} 是为了在数组为空时也不触发 set -u（bash < 4.4）
  docker run -d --name "${NAME}" ${devices[@]+"${devices[@]}"} --cap-add NET_ADMIN \
    -v "${qcow}:/System.qcow2:${mode}" \
    -p "127.0.0.1:${PORT}:5000" -p "127.0.0.1:${VNC_PORT}:8006" \
    "${IMAGE}" >/dev/null
}

cleanup_container() { docker rm -f "${NAME}" >/dev/null 2>&1 || true; }

verify() {
  command -v docker >/dev/null || die "verify 需要 docker"
  [[ -n "${PYTHON_BIN}" ]] || die "verify 需要 python3"
  [[ -f "${GOLD_QCOW}" ]] || die "找不到镜像: ${GOLD_QCOW}（先拷回来，或设 GOLD_QCOW=）"

  # 端口必须避开 pool 的 5000-9999（server 5000+/vnc 6000+/vlc 7000+/chromium 9000+），
  # 否则这个临时容器会和真实任务抢端口。
  if (( PORT >= 5000 && PORT <= 9999 )); then
    die "PORT=${PORT} 落在 pool 的分配区间 5000-9999 内，会和真实任务抢端口。换一个，例如 15000。"
  fi

  mkdir -p "${LOG_DIR}"
  local vlog="${LOG_DIR}/verify-blender-${STAMP}.log"
  trap cleanup_container EXIT

  start_container ro "${GOLD_QCOW}"

  # ---- 等 /execute 应答（轮询，不要单次采样） ----
  log "等待 guest :${PORT} 应答（最多 ${BOOT_WAIT}s）"
  local i
  for ((i = 0; i < BOOT_WAIT / 5; i++)); do
    [[ "$(exec_guest 'echo READY' 15 2>/dev/null)" == *READY* ]] && { log "guest 已在 $((i*5))s 后应答"; break; }
    sleep 5
  done
  [[ "$(exec_guest 'echo READY' 15 2>/dev/null)" == *READY* ]] || die "guest 始终没在 :${PORT} 应答"

  # ---- 等 X :0 起来（/execute 应答得比 gnome 会话登录早得多） ----
  # 用 X socket 存在性判断，零依赖（guest 里未必有 xdpyinfo/xset）。
  log "等待 X 显示 :0（最多 ${X_WAIT}s）"
  for ((i = 0; i < X_WAIT / 5; i++)); do
    [[ "$(exec_guest 'test -S /tmp/.X11-unix/X0 && echo X_READY' 20 2>/dev/null)" == *X_READY* ]] \
      && { log "X :0 已就绪（$((i*5))s）"; break; }
    sleep 5
  done

  # ---- 环境事实，先记下来 ----
  log "=== 环境探测 ==="
  run_guest_script 'set +e
echo "OS: $(. /etc/os-release; echo "$PRETTY_NAME")"
echo "BLENDER: $(command -v blender || echo NONE)"
blender --version 2>&1 | head -2
echo "XSOCK: $(ls /tmp/.X11-unix/X0 2>/dev/null || echo NONE)"
echo "XAUTH_GDM: $(ls /run/user/1000/gdm/Xauthority 2>/dev/null || echo NONE)"
echo "XAUTH_ANY: $(ls /run/user/1000/*/Xauthority /home/user/.Xauthority 2>/dev/null | head -3 | tr "\n" " ")"
echo "DM: $(cat /etc/X11/default-display-manager 2>/dev/null || echo NONE)"
echo "GL: $(command -v glxinfo >/dev/null && DISPLAY=:0 glxinfo -B 2>&1 | grep -iE "opengl version|renderer" || echo no-glxinfo)"
' 120 2>&1 | tee -a "${vlog}"

  # ---- Python 运行依赖门禁 ----
  # 在真正启动后的生产同款 guest 里再验一次，防止 virt-customize 使用的
  # Python/pip 与桌面会话运行 reward.py 时使用的 Python 不一致。
  log "=== RLVR Python 依赖验收 ==="
  local deps_script deps_output
deps_script="$(cat <<'DEPS'
set -e
export DISPLAY="${DISPLAY:-:0}"
XA="$(ls /run/user/1000/*/Xauthority /home/user/.Xauthority 2>/dev/null | head -1 || true)"
[[ -n "${XA}" ]] && export XAUTHORITY="${XA}"
python3 - <<'"'"'PY'"'"'
import importlib
from PIL import Image

required = [
    "cv2", "numpy", "PIL", "skimage", "imagehash", "pypdf",
    "openpyxl", "docx", "pptx", "fitz", "odf", "pikepdf",
    "pymupdf", "reportlab", "pdfplumber", "pandas", "requests",
    "bs4", "lxml", "rapidfuzz", "tldextract", "librosa", "fastdtw",
    "scipy", "PyPDF2", "Xlib", "gi", "uno", "pyatspi", "pyautogui",
]
for name in required:
    module = importlib.import_module(name)
    print("DEP_OK:", name, getattr(module, "__version__", ""))

from importlib.metadata import version
from playwright.sync_api import sync_playwright
assert callable(sync_playwright)
# Starting the Playwright driver validates its bundled Node transport without
# downloading or launching Playwright's Chromium. CUA-Gym tasks connect to the
# image's existing Chrome 130 over CDP.
with sync_playwright() as playwright:
    assert playwright.chromium.name == "chromium"
import uno
from com.sun.star.beans import PropertyValue
assert PropertyValue is not None
import imagehash
assert str(imagehash.phash(Image.new("L", (8, 8), 0)))
print("DEP_OK: ImageHash", version("ImageHash"))
print("DEP_OK: playwright", version("playwright"), "driver")
print("DEP_OK: pypdf", version("pypdf"))
print("DEP_FUNCTIONAL_OK")
PY
DEPS
)"
  deps_output="$(run_guest_script "${deps_script}" 180)"
  printf '%s\n' "${deps_output}" | tee -a "${vlog}"
  [[ "${deps_output}" == *DEP_FUNCTIONAL_OK* ]] \
    || die "RLVR Python 依赖验收失败；不要把这份镜像接进 pool"

  # ---- 渲染烟测 ----
  # 这个脚本刻意只用 SKILL.md 教的那套 API（图元、Principled BSDF 的 input 名、
  # BLENDER_EEVEE、Euler 弧度）—— 它跑通就等于 skill 的 Blender 3.0 假设全部成立。
  local smoke
  smoke="$(cat <<'SMOKE'
set +e
mkdir -p /tmp/blender_verify
cat > /tmp/blender_verify/smoke.py <<'PY'
import bpy
from mathutils import Euler

# 清掉 startup file 里的一切，保证结果确定，不依赖 ~/.config/blender 有没有东西
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
obj = bpy.context.active_object
bpy.ops.object.shade_smooth()

mat = bpy.data.materials.new("Red")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 0, 1)
obj.data.materials.append(mat)

cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 35
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam)
cam.location = (4, -4, 3)
cam.rotation_euler = Euler((1.1, 0, 0.785))
bpy.context.scene.camera = cam

sun = bpy.data.lights.new("Sun", "SUN")
sun.energy = 5.0
so = bpy.data.objects.new("Sun", sun)
bpy.context.collection.objects.link(so)
so.rotation_euler = Euler((0.8, 0.2, -0.5))

sc = bpy.context.scene
sc.render.engine = 'BLENDER_EEVEE'
sc.render.resolution_x = 320
sc.render.resolution_y = 240
sc.render.image_settings.file_format = 'PNG'
sc.render.filepath = '/tmp/blender_verify/out.png'
bpy.ops.render.render(write_still=True)
print("SMOKE_RENDER_DONE")
PY

run_once() {
  local label="$1"; shift
  rm -f /tmp/blender_verify/out.png
  echo "----- smoke [$label] env: $* -----"
  env "$@" blender --background --python /tmp/blender_verify/smoke.py 2>&1 | tail -20
  python3 - <<'PY' | tee -a /tmp/blender_verify/results.txt
import os, sys
p = "/tmp/blender_verify/out.png"
if not os.path.exists(p):
    print("SMOKE_RESULT: NO_PNG"); sys.exit(0)
try:
    import cv2, numpy as np
except ImportError as e:
    print("SMOKE_RESULT: PNG_OK_BUT_NO_CV2", e, os.path.getsize(p)); sys.exit(0)
im = cv2.imread(p)
if im is None:
    print("SMOKE_RESULT: PNG_UNREADABLE"); sys.exit(0)
print("SMOKE_RESULT: STD=%.3f" % float(np.std(im)))
PY
}

: > /tmp/blender_verify/results.txt

# 1) 先按 SKILL.md §14.6 的原配方跑（DISPLAY=:0 + gdm 的 XAUTHORITY）
XA="$(ls /run/user/1000/*/Xauthority 2>/dev/null | head -1)"
run_once "skill-recipe" DISPLAY=:0 XAUTHORITY="${XA:-/run/user/1000/gdm/Xauthority}"

# 2) skill 记的 XAUTHORITY 路径是照上游 Aliyun VM 抄的，这份 cua-gym VM 未必一致。
#    原配方不出图时再试一次只给 DISPLAY —— 这一次的结果决定要不要改 skill。
if ! grep -q 'STD=' /tmp/blender_verify/results.txt 2>/dev/null; then
  echo "----- 原配方未出图，回退：只给 DISPLAY=:0 -----"
  run_once "display-only" DISPLAY=:0
fi
echo "SMOKE_MARKER_DONE"
SMOKE
)"

  log "=== 渲染烟测（EEVEE，320x240） ==="
  run_guest_script "${smoke}" 600 2>&1 | tee -a "${vlog}"

  # ---- 判定 ----
  log "=== 判定 ==="
  local version std
  version="$(grep -m1 -oE 'Blender [0-9]+\.[0-9]+\.[0-9]+' "${vlog}" || true)"
  std="$(grep -oE 'SMOKE_RESULT: STD=[0-9.]+' "${vlog}" | tail -1 | sed 's/.*=//' || true)"

  echo
  echo "  blender 版本 : ${version:-<未能获取>}"
  echo "  渲染标准差   : ${std:-<未渲染出图>}   （> 10 才算真出了内容，全黑/全透明都是 0）"
  echo "  完整日志     : ${vlog}"
  echo

  local fail=0
  [[ "${version}" == "Blender 3.0"* ]] || { warn "版本不是 3.0.x —— SKILL.md 里所有 EEVEE/BSDF/modifier 的写法都可能失效"; fail=1; }
  if [[ -z "${std}" ]]; then
    warn "没有渲染出图。看 ${vlog} 里的 blender stderr："
    warn "  'Unable to open a display' -> DISPLAY/XAUTHORITY 不对，改 SKILL.md §14.6"
    warn "  GL/EEVEE 相关报错          -> 无 GPU 软渲染跑不起来，退 CYCLES(CPU)，"
    warn "                                并同步改 SKILL.md §7/§11 的渲染章节"
    fail=1
  elif "${PYTHON_BIN}" -c "import sys; sys.exit(0 if float('${std}') > 10 else 1)"; then
    log "渲染出图，标准差 ${std} > 10 —— OK"
  else
    warn "渲染出来了但标准差 ${std} <= 10，画面基本是纯色（全黑/全白）。"
    warn "相机没对准、没光源、或 EEVEE 在软渲染下没真正出图。"
    fail=1
  fi

  [[ "${fail}" == "0" ]] || die "验收未通过 —— 别把这个镜像接进 pool"
  log "VERIFY OK —— 可以接进 pool 了"
  cat <<EOF

接线（不要改 world.yaml 默认值：cua_gym 和 osworld 两个 world 都指向同一份，
改了等于同时动 OSWorld 基线）：

  PATH_TO_VM=${GOLD_QCOW} \\
    python -m cluster.node.server --port 18080 --max-envs 72 ...

建议先跑仓库既有的 §6 Tier 0（4 个任务 0 个 env）冒烟，再切全量。
EOF
}

case "${MODE}" in
  preflight) preflight ;;
  bake)      bake ;;
  verify)    verify ;;
esac

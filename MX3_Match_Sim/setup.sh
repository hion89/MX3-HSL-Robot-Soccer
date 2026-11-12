#!/usr/bin/env bash
# MX3 경기 시뮬 설치 — Ubuntu 24.04 + ROS 2 Jazzy 가 깔린 PC 에서 이 폴더째로 한 번 실행한다 (README 참고).
#   ./setup.sh             필요한 apt 패키지가 없으면 sudo 로 설치 → 파이썬 라이브러리 (.venv) → 빌드
#   SKIP_APT=1 ./setup.sh  apt 설치를 건너뜀
set -eo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS"
if [[ -n "${CONDA_PREFIX:-}" ]]; then echo "conda 가 켜져 있습니다. conda deactivate 후 다시 실행하세요" >&2; exit 1; fi
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy 가 없습니다. README 의 '1. ROS 2 Jazzy 설치' 부터 하세요" >&2; exit 1
fi
. /etc/os-release
[[ "${VERSION_ID:-}" == 24.04 ]] || echo "경고: Ubuntu ${VERSION_ID:-?} — 돌려 본 환경은 24.04 입니다" >&2

APT=(ros-jazzy-ros-base ros-jazzy-robot-state-publisher ros-jazzy-twist-mux ros-jazzy-vision-msgs
     ros-jazzy-tf2-eigen ros-jazzy-tf2-geometry-msgs ros-jazzy-kdl-parser ros-jazzy-orocos-kdl-vendor
     ros-jazzy-eigen3-cmake-module ros-jazzy-visualization-msgs ros-jazzy-std-srvs ros-jazzy-nav-msgs
     libeigen3-dev python3-colcon-common-extensions python3-venv python3-numpy python3-yaml python3-pil)
MISSING=()
for p in "${APT[@]}"; do
  dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || MISSING+=("$p")
done
if (( ${#MISSING[@]} == 0 )); then
  echo "[1/3] apt 패키지: 모두 있음"
elif [[ "${SKIP_APT:-0}" == 1 ]]; then
  echo "[1/3] apt 패키지: 건너뜀 (SKIP_APT=1) — 없는 것: ${MISSING[*]}" >&2
else
  echo "[1/3] apt 패키지 설치: ${MISSING[*]}"
  sudo apt-get update
  sudo apt-get install -y "${MISSING[@]}"
fi

echo "[2/3] 파이썬 라이브러리 (.venv: mujoco, onnxruntime)"
# --system-site-packages: 시스템 numpy·yaml·PIL 을 그대로 쓰고 없는 것만 받는다.
# ROS 노드는 /usr/bin/python3 로 돌기 때문에 실행 스크립트가 .venv 의 site-packages 를 PYTHONPATH 로 넣는다 (가상환경을 켤 필요 없음).
[[ -x .venv/bin/pip ]] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import mujoco, onnxruntime; print('  mujoco', mujoco.__version__, '/ onnxruntime', onnxruntime.__version__)"

echo "[3/3] 빌드 (colcon, 1~3 분)"
set +u
source /opt/ros/jazzy/setup.bash
colcon build --base-paths src --packages-up-to hsl_bringup humanoid_sim --cmake-args -DCMAKE_BUILD_TYPE=Release
echo
echo "완료. 경기 시작: ./run_match.sh   (끄기: 그 터미널에서 Ctrl+C)"

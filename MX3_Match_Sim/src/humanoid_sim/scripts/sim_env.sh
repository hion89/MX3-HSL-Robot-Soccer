# 시뮬 스크립트 공통 환경 — source 해서 쓴다. 부르기 전에 WS (colcon 워크스페이스 폴더) 를 정해 둘 것.
#   ROS 2 Jazzy + 워크스페이스 install + 파이썬 라이브러리 (mujoco·onnxruntime)
# 파이썬 라이브러리: ROS 노드는 시스템 Python 3.12 (/usr/bin/python3) 로 돌아서 가상환경을 켜는 것만으로는 안 잡힌다.
#   그래서 워크스페이스의 .venv (setup.sh 가 만듦) 또는 pylib (pip --target) 를 PYTHONPATH 로 넣는다.
if [[ -n "${CONDA_PREFIX:-}" ]]; then echo "conda deactivate 후 실행하세요" >&2; exit 1; fi
if [[ ! -f "$WS/install/setup.bash" ]]; then
  echo "빌드 결과가 없습니다: $WS/install — 먼저 빌드하거나 SIM_WS=<워크스페이스 폴더> 로 지정하세요" >&2; exit 1
fi
set +u
source /opt/ros/jazzy/setup.bash
# Fast DDS 공유 메모리 전송을 쓰지 않는다 (UDP 만). 한 PC 에 로봇 8 대 × 노드 ~25 개가 동시에 뜨면 가끔 한 프로세스가
# 통신망에서 고립됐다 (파랑 4 번 보행 러너가 관절 상태를 한 번도 못 받음). 실기(로봇 1 대)에는 해당 없음.
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source "$WS/install/setup.bash"
export SIM_WS="$WS"
for _d in "$WS"/.venv/lib/python3*/site-packages "$WS/pylib"; do
  [[ -d "$_d" ]] && export PYTHONPATH="$_d:$PYTHONPATH"
done
unset _d

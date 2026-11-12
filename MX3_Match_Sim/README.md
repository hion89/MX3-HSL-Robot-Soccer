# MX3 휴머노이드 축구 경기 시뮬레이터 (MuJoCo + ROS 2)

MX3(XM) 로봇 8 대가 4 대 4 로 RoboCup 휴머노이드 경기를 하는 시뮬레이터입니다.
물리는 MuJoCo 가 계산하고, 각 로봇은 실제 로봇과 같은 ROS 2 소프트웨어로 움직입니다.

- 로봇 하위 계층 = 실기 소스 `mx3_src` 그대로 (보행·킥 강화학습 정책 러너, 넘어짐 관리, 목 제어, 게임컨트롤러 수신)
- 로봇 상위 계층 = `hsl_src` (B-Human 방식 측위·공 추적·행동·팀 통신)
- 시뮬 = `humanoid_sim` (MuJoCo 브리지: 모터·IMU·카메라 검출을 대신함) + 심판 (GameController 패킷을 실제처럼 보냄)
- 정책: 보행 v0_6 model_13500, 킥 v1_5b model_22300 (`src/hsl_src/hsl_bringup/models/`)

## 필요한 것

| | |
|---|---|
| OS | **Ubuntu 24.04** (다른 버전은 확인 안 함) |
| ROS | **ROS 2 Jazzy** |
| CPU | 로봇 8 대 + 물리를 실시간으로 돌린다. 만든 PC 는 i7-12700F (20 스레드) 에서 실시간 1.0 배. 8 코어 이하면 느려질 수 있다 |
| GPU | 필요 없음 (화면을 띄우려면 OpenGL 되는 데스크톱) |
| 디스크 | 약 1 GB (빌드 + 파이썬 라이브러리) |

## 설치

### 1. ROS 2 Jazzy 설치 (이미 있으면 건너뜀)

공식 안내 https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html 대로 저장소를 추가한 뒤
`sudo apt install ros-jazzy-ros-base` (또는 `ros-jazzy-desktop`). 나머지 패키지는 2 단계가 알아서 설치한다.

### 2. 이 폴더 설치

이 폴더(`MX3_Match_Sim`)를 원하는 곳에 두고 (경로에 공백이 없는 곳 권장), 터미널에서:

```bash
cd MX3_Match_Sim
./setup.sh
```

하는 일 (처음 1~5 분):
1. 없는 apt 패키지 설치 (sudo 비밀번호를 물을 수 있음) — twist_mux, vision_msgs, kdl_parser, colcon, python3-venv 등
2. `.venv` 가상환경을 만들고 mujoco 3.17.0 · onnxruntime 1.31.1 설치 (`requirements.txt`)
3. `colcon build` 로 로봇 소프트웨어 빌드 → `build/`, `install/` 생성

**conda 가 켜져 있으면 먼저 `conda deactivate`** (ROS 2 는 시스템 Python 3.12 를 써야 한다).

> 가상환경을 직접 켤(activate) 필요는 없다. ROS 노드는 시스템 Python 으로 실행되기 때문에,
> 실행 스크립트가 `.venv` 의 라이브러리 경로를 알아서 넣어 준다.

## 실행

```bash
./run_match.sh
```

- MuJoCo 창이 뜨고, 약 30 초 동안 로봇 소프트웨어를 띄운 뒤 (화면 가운데 PREPARING MATCH) 경기가 시작된다.
- **끄기: 그 터미널에서 Ctrl+C.** 터미널을 그냥 닫았거나 뭔가 남은 것 같으면 `./stop_match.sh`.
- 기본은 5 분 뒤 자동 종료. 전반(경기 시간 10 분)을 끝까지 보려면 `DURATION=1200 ./run_match.sh`.

설정은 환경 변수로 준다:

| 변수 | 기본 | 뜻 |
|---|---|---|
| `SCENE` | `4v4` | `4v4` / `4v0` (한 팀만) / `1v0` (로봇 1 대) |
| `VIEWER` | `1` | MuJoCo 화면 (0 = 화면 없이) |
| `DURATION` | `300` | 몇 초 뒤 전부 끌지 |
| `HALF` | `600` | 경기 시간(PLAYING 누적) [s], 지나면 FINISHED |
| `READY_S` | `40` | 첫 READY 시간 [s] |
| `KICKOFF` | `35` | 첫 킥오프 팀 (35 / 36) |
| `LOG` | `logs/match_<장면>_<시각>` | 로그 폴더 |
| `STACK_ARGS` | (없음) | 로봇 스택 launch 인자 (아래) |

예:
```bash
SCENE=4v0 VIEWER=0 DURATION=180 ./run_match.sh        # 화면 없이 한 팀 4 대, 3 분
STACK_ARGS="kick_actor_path:=/경로/다른킥.onnx" ./run_match.sh   # 다른 킥 정책 (관측 265 walk_kick v1 계열만)
```

`STACK_ARGS` 로 줄 수 있는 것: `kick_actor_path:=<onnx>` (`runner` = mx3_src 러너 사본 18200), `walk_actor_path:=<onnx>`,
`kick_duration:=<s>` (기본 2.0), `walk_csv_log:=true` (보행 러너 CSV 를 `~/policy_logs` 에 저장, 기본 끔),
`team_send_rate_hz:=<Hz>` (팀 통신, 기본 2).

### 화면 보는 법

- 몸통 색 = 팀: 35 팀 파랑 (골키퍼 청록), 36 팀 빨강 (골키퍼 주황). 머리 위 숫자 = 선수 번호.
- 발밑 고리 = 지금 공을 맡은 로봇.
- 왼쪽 위 점수, 오른쪽 위 GameController 상태 (상태·남은 시간·킥 팀·세트플레이·페널티·마지막 판정).
- 마우스: 왼쪽 드래그 회전, 오른쪽 드래그 이동, 휠 확대. 패널은 Tab / Shift+Tab. **W 를 누르면 와이어프레임**이 되니 주의.

### 심판이 하는 것

INITIAL 3 s → READY → SET 5 s → PLAYING. 골 → 센터, 실점 팀 킥오프. 터치라인 아웃 → 상대 KICK-IN, 골라인 아웃 →
CORNER / GOAL KICK. 반칙 (페널티 30 s): 라인 밖 이탈, 밀기 (PUSHING), SET 순간 위치 위반, SET 중 움직임,
20 s 넘게 못 일어남. 공이 45 s 멈춰 있으면 센터 근처로 옮긴다. 시뮬에서 넘어진 로봇은 3 초 뒤 세워 준다 (실제 기상 동작 아님).

## 로그

`logs/match_<장면>_<시각>/`:
- `monitor.txt` — 10 초마다 로봇별 상태 표, 끝에 요약 (아래 읽는 법)
- `monitor.jsonl` — 1 초마다 시계열·이벤트 (분석용)
- `t35p1.log` … — 로봇별 소프트웨어 로그, `bridge.log` (물리), `referee.log` (심판)

```
t35p3  PLAY suppor A play_ball_align   gt(-0.4,-0.2)   loc  0.07m    2° S ball 0.05 kick 1 fall 0
 이름  상태  역할  공담당  행동          실제 위치       측위 오차(위치/방향) 품질  공 오차 킥 수 넘어짐
```
좌표는 각 팀 기준 (+x = 그 팀이 공격하는 방향). 로그는 판당 3~11 MB 씩 쌓이니 가끔 지운다.

## 폴더 구성

```
MX3_Match_Sim/
  README.md          이 문서
  setup.sh           설치 (apt → .venv → 빌드)
  run_match.sh       경기 시작
  stop_match.sh      남은 프로세스 정리
  requirements.txt   파이썬 라이브러리 버전
  src/
    humanoid_sim/    MuJoCo 브리지, 공통 실행 환경 (scripts/sim_env.sh)
    hsl_src/         상위 계층 5 개 패키지 + 설계 문서 DESIGN.md, 라이선스
    mx3_src/         실기 하위 계층 중 시뮬에 필요한 8 개 패키지 (수정 없이 복사)
  (설치 후) .venv/ build/ install/ log/ logs/
```

더 자세한 설계·실험 기록: `src/hsl_src/DESIGN.md`, `src/hsl_src/README.md`, `src/humanoid_sim/README.md`
(이 문서들 안의 `~/sim_ws`, `~/Documents/ros2_ws/...` 경로는 만든 PC 기준 — 여기서는 이 폴더가 그 워크스페이스다).

## 문제 해결

| 증상 | 확인 |
|---|---|
| `conda deactivate 후 실행하세요` | conda 끄기. `.bashrc` 에서 자동으로 켜지면 새 터미널마다 `conda deactivate` |
| `이미 시뮬 브리지가 돌고 있습니다` | 이전 경기가 남아 있음 → `./stop_match.sh` |
| `⚠ t36p4 보행 러너가 아직 정책을 안 돌림` | 그 로봇만 통신이 끊김. 끄고 다시 시작 (`./stop_match.sh` 후 `./run_match.sh`) |
| `bridge.log` 에 지연 경고가 많다 | CPU 부족. 다른 무거운 작업을 끄거나 `VIEWER=0`, `SCENE=4v0` |
| 창이 안 뜬다 / OpenGL 오류 | 원격 접속(SSH) 이면 화면이 없다 → `VIEWER=0`. 데스크톱에서 직접 실행할 것 |
| 로봇이 READY 에서 안 움직임 | 같은 네트워크에 진짜 GameController 나 다른 시뮬이 돌고 있지 않은지 (UDP 3838) |
| 빌드 실패 | `setup.sh` 를 다시 실행 (apt 설치가 중간에 끊긴 경우), 오류 메시지의 패키지 이름을 `sudo apt install ros-jazzy-<이름>` |

로봇마다 ROS_DOMAIN_ID 를 따로 쓴다 (35 팀 11~14, 36 팀 21~24). 같은 네트워크의 다른 ROS 2 장비와 도메인이 겹치지 않게 할 것.

## 라이선스

This product includes software developed by B-Human (http://www.b-human.de).
B-Human 코드 이식 부분의 조건은 `src/hsl_src/LICENSE_BHUMAN.md`. 나머지는 만든 사람의 허락 없이 재배포하지 말 것.

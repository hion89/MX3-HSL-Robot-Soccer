#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# control.launch.py  (humanoid_bringup)
# --------------------------------------------------------------------------
# "제어(control)" 레이어 노드들을 함께 띄운다. 명령 흐름은:
#
#   teleop/Nav2/FSM ─(각 입력 토픽)→ twist_mux ─/cmd_vel_raw→
#       lateral_drift_correction ─/cmd_vel→ actor(보행+킥) ─/joint_goals→ (하드웨어)
#
# 포함 노드:
#   1) twist_mux                          : 여러 cmd 소스를 우선순위로 다중화
#   2) lateral_drift_correction_node      : vy bias 보정(plant inner loop)
#   3) actor_onnx_walk_residual_kick_node : 보행 + residual 킥 통합 정책 추론
#   4) fall_manager_node                  : 넘어짐 보호 + /robot_state 발행
#   5) keyboard_teleop_node               : (선택) 수동 개입용. use_teleop 로 토글
#
# ── 로봇별 파라미터 ──
#   config/robot_params.yaml 의 'robot' 섹션을 골라 actor 노드에 적용한다.
#   robot 인자 기본값 "" → 비면 hostname 앞부분 사용(예: robo04-desktop → robo04).
#   적용 우선순위: 노드 내부 기본값 < common < <robot> 섹션.
#
# 인자:
#   robot ("")            : robot_params.yaml 에서 고를 섹션. 비면 hostname 자동.
#   use_twist_mux (true)  : twist_mux 사용 여부.
#   use_teleop    (false) : 같은 터미널에서 키보드 텔레옵 동시 실행 여부.
#   use_correction(true)  : 횡 drift 보정 노드 사용 여부.
#   node_output   (screen): 이 레이어 노드들의 출력 위치('screen' 또는 'log').
#   walk_actor_path / residual_actor_path / reference_csv_path : 모델/레퍼런스 경로.
#
# [주의] fall_manager 는 하드웨어 /u2d2/* 서비스를 호출하므로 bringup 으로
#        hardware 와 함께 띄우는 것을 전제로 한다.
# ==========================================================================

import os
import socket
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.logging import get_logger
from launch_ros.actions import Node


# ── 횡/전진 drift 보정 노드(lateral_drift_correction_node)가 쓰는 파라미터 키 ──
#  robot_params.yaml(common/<robot>)에서 이 키들만 골라 correction 노드에 적용한다.
#  actor 노드에는 이 키들을 "빼고" 넘긴다 — actor 가 declare 안 한 vy_trim 등을 받아
#  거부/경고하는 일을 막기 위함(노드별로 아는 파라미터만 전달).
#
#  [2026-08-24] vx_trim(제자리 후진 drift 보정) 키들을 추가했다. 동시에
#  vx_ref 가 빠져 있던 것도 채웠다 — scale_with_vx 를 켜도 기준속도를 YAML 로
#  못 넘겨 노드 기본값 0.3 만 쓰이던 구멍이었다.
CORRECTION_KEYS = (
    # 횡(vy) drift 보정 — "걷는 중" 에만 적용
    "vy_trim",
    "vx_activation",
    "scale_with_vx",
    "vx_ref",
    # 전진(vx) drift 보정 — "제자리" 에만 적용. 자세한 가드는 노드 docstring 참조.
    "vx_trim",
    "vx_trim_max_cmd",
    "vx_trim_clamp",
    "vx_trim_gait_modes",
    "vx_trim_require_lock",
    # 회전(wz) drift 보정 — "걷는 중" 에만 적용 (2026-08-28 추가).
    #   상수 오픈루프 바이어스다. IMU 를 보는 폐루프 피드백이 아니다.
    "wz_trim",
    "wz_trim_limit",
    "wz_trim_min_cmd",
    "wz_out_limit",
    "wz_trim_scale_with_vx",
)
#  robot_params 에 해당 키가 없을 때 쓰는 기본값(기존 하드코딩과 동일).
#  ★ vx_trim / wz_trim 계열은 여기 넣지 않는다 ★ 노드 기본값(둘 다 0.0 =
#  비활성)이 그대로 쓰이게 두어, YAML 에 명시하지 않으면 기존 동작과 100%
#  같게 만든다.
CORRECTION_DEFAULTS = {"vy_trim": -0.1, "vx_activation": 0.05, "scale_with_vx": False}


# ── actor(v1s 러너)에 넘길 robot_params 키 화이트리스트 ─────────────────
# 2026-08 정책 교체: actor_onnx_walk_residual_kick_runner → actor_onnx_walking_runner
#
# ★ "correction 키만 빼고 전부 넘기기" 를 하면 안 된다 ★
#   robot_params.yaml 은 이전 킥 러너 기준으로 작성되어 있어서,
#   v1s 러너가 declare 하지 않은 키(kick_cutoff_s, walk_resume_hold_seconds,
#   kick_grav_offset_*, player_number, imu_mount_* 등)가 섞여 있다.
#   이걸 그대로 넘기면 노드가 "declare 하지 않은 파라미터" 로 거부하며 죽는다.
#
#   더 위험한 건 이름은 같은데 의미가 다른 키다:
#     cmd_vx  : 킥 러너에서는 "킥 중 obs 고정 전진명령" 이지만
#               v1s 에서는 "/cmd_vel 수신 전 초기 속도명령" 이다.
#               common 의 0.30 을 그대로 넘기면 기동 직후 hold 가 풀리는
#               순간 아무도 명령하지 않았는데 0.3m/s 로 전진한다.
#     resume_hold_seconds : 킥 러너에서는 "킥 진입 전 정지 안정화",
#               v1s 에서는 "낙상 복귀 후 hold". 값의 근거가 다르다.
#
#   → 의미가 v1s 에서도 동일하다고 확인된 키만 명시적으로 통과시킨다.
#     새 키를 추가할 때는 v1s 러너의 declare_parameter 목록과 의미를
#     반드시 대조할 것.
ACTOR_ROBOT_KEYS = (
    "grav_offset_x",        # 가상 중력 트림 (앞뒤 쏠림)
    "grav_offset_y",        # 가상 중력 트림 (좌우 쏠림)
    "grav_offset_z",
    "pelvis_pitch_offset",  # 골반 pitch 기구 트림 (로봇별)
)


def _resolve_robot_name(name: str) -> str:
    """robot 인자가 비어있으면 hostname 앞부분으로 (robo04-desktop → robo04)."""
    name = (name or "").strip()
    if name:
        return name
    host = socket.gethostname()
    return host.split("-")[0].split(".")[0]


def _load_robot_params(robot: str):
    """config/robot_params.yaml 에서 common + <robot> 병합(robot 우선)."""
    cfg = os.path.join(
        get_package_share_directory("humanoid_bringup"),
        "config", "robot_params.yaml")
    data = {}
    if os.path.isfile(cfg):
        with open(cfg, "r") as f:
            data = yaml.safe_load(f) or {}
    common = data.get("common", {}) or {}
    robocfg = data.get(robot, {}) or {}
    merged = dict(common)
    merged.update(robocfg)
    return merged, (robot in data), cfg


def _servo_kick_wait_timeout() -> float:
    """vision_servo.yaml 의 kick_wait_timeout_s 를 읽어온다(불변식 검사용).

    walking 러너의 킥 양보 워치독은 이 값보다 작아야 한다. 두 값이 다른
    패키지에 흩어져 있어 한쪽만 고치면 조용히 어긋나므로, 여기서 실제
    yaml 을 읽어 러너에 넘겨 기동 시점에 검사시킨다.
    파일이 없거나 파싱에 실패하면 0.0 을 돌려 검사를 생략한다(기동은 막지 않는다).
    """
    try:
        path = os.path.join(
            get_package_share_directory("humanoid_localization"),
            "config", "vision_servo.yaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return float(data["visual_servo_kick_node"]["ros__parameters"]
                     ["kick_wait_timeout_s"])
    except Exception:
        return 0.0


def _robot_nodes_setup(context, *args, **kwargs):
    """robot 인자 해석 → YAML 로드 → correction/actor 노드를 robot 파라미터로 생성.

    robot_params.yaml(common+<robot>)을 한 번 읽어 두 노드에 "나눠" 적용한다:
      - correction 노드 : CORRECTION_KEYS(vy_trim 등)만 추출(없으면 기본값) → 로봇별 횡보정
      - actor 노드      : 나머지 키(cmd_vx/kick_*/grav_* 등). correction 키는 제외.
    노드별로 자신이 declare 한 키만 받게 분리하는 게 핵심(서로 모르는 키 거부 방지).
    """
    robot = _resolve_robot_name(LaunchConfiguration("robot").perform(context))
    node_output = LaunchConfiguration("node_output").perform(context)
    walk_actor_path = LaunchConfiguration("walk_actor_path").perform(context)
    kick_actor_path = LaunchConfiguration("kick_actor_path").perform(context)
    onnx_provider = LaunchConfiguration("onnx_provider").perform(context)
    stop_delay_s = float(LaunchConfiguration("stop_delay_s").perform(context))
    policy_variant = LaunchConfiguration("policy_variant").perform(context)
    gait_mode = LaunchConfiguration("gait_mode").perform(context)
    gait_freq = float(LaunchConfiguration("gait_freq").perform(context))
    kick_resume_hold = float(
        LaunchConfiguration("kick_resume_hold").perform(context))
    # ── 킥(walkkick_v1) 인자 ──
    kick_ball_x = float(LaunchConfiguration("kick_ball_x").perform(context))
    kick_cmd_vx = float(LaunchConfiguration("kick_cmd_vx").perform(context))
    kick_duration = float(LaunchConfiguration("kick_duration").perform(context))
    kick_start_hold = float(
        LaunchConfiguration("kick_start_hold").perform(context))
    # [2026-08 CPU 최적화] projected gravity 취득 방식.
    #   derive/subscribe/verify. 자세한 설명은 아래 DeclareLaunchArgument 참조.
    gravity_source = LaunchConfiguration("gravity_source").perform(context)
    kick_strict_gate = (LaunchConfiguration("kick_strict_gate")
                        .perform(context).lower() in ("1", "true", "yes"))
    kick_yield_margin = float(
        LaunchConfiguration("kick_yield_margin").perform(context))
    use_correction = (LaunchConfiguration("use_correction").perform(context)
                      .strip().lower() in ("true", "1", "yes", "on"))

    robot_params, found, cfg = _load_robot_params(robot)
    log = get_logger("control.launch")
    if found:
        log.info(f"[robot_params] robot='{robot}' 적용 ({cfg}): {robot_params}")
    else:
        log.warn(f"[robot_params] robot='{robot}' 섹션 없음 → common/노드기본값만: {robot_params}")

    nodes = []

    # ── 2) 횡/전진 drift 보정 노드 : robot_params 의 vy_trim/vx_trim 등 적용 ──
    #  use_correction=false 면 띄우지 않는다(대신 twist_mux_direct 가 /cmd_vel 직결).
    #  ★ 이 노드는 twist_mux '뒤' 에 있다 ★ 즉 game_stop_lock 이 0 을 만든 뒤에
    #    보정이 더해진다. vx_trim 은 정지 구간에 붙는 보정이라 그대로 두면
    #    페널티 중에 로봇이 기어간다 — 노드가 락 토픽을 직접 구독해 막는다.
    #    경기 스택에서는 vx_trim_require_lock: true 를 권장한다.
    if use_correction:
        corr = dict(CORRECTION_DEFAULTS)
        for k in CORRECTION_KEYS:
            if k in robot_params:
                corr[k] = robot_params[k]
        log.info(f"[robot_params] correction(vy_trim/vx_trim 등) robot='{robot}': {corr}")
        nodes.append(Node(
            package="humanoid_policy_runner",
            executable="lateral_drift_correction_node",
            name="lateral_drift_correction_node",
            output=node_output,
            parameters=[corr],
        ))

    # ── 3) actor 노드 (v1s 러너) ──
    #  구조적 파라미터(경로/모드/학습cfg)는 여기서, 로봇별 tunable 은 YAML 에서.
    #  parameters 리스트는 뒤가 override → actor_params(YAML) 가 최종 우선.
    #
    #  ★ 화이트리스트로 걸러서 넘긴다 ★ (사유는 ACTOR_ROBOT_KEYS 주석 참조)
    actor_params = {k: v for k, v in robot_params.items() if k in ACTOR_ROBOT_KEYS}
    dropped = sorted(set(robot_params) - set(actor_params) - set(CORRECTION_KEYS))
    if dropped:
        # 조용히 버리면 "YAML 에 적었는데 왜 반영이 안 되지?" 로 시간을 잃는다.
        log.info(f"[robot_params] actor 미전달 키(v1s 미지원/의미상이): {dropped}")
    log.info(f"[robot_params] actor(v1s) robot='{robot}': {actor_params}")

    structural = {
        # ── 모델 ──
        # policy_variant 는 ONNX 관측 차원과 반드시 일치해야 한다.
        #   v0_2 : 관측 240, v0_1 + 제자리 걷기, 주파수 2.0 고정
        # ★ v0_2 와 v1s 는 관측 차원이 240 으로 같아 자동 판별이 안 된다 ★
        #   틀리면 에러 없이 조용히 오동작하므로 모델 교체 시 반드시 확인.
        "policy_variant": policy_variant,
        "actor_path": walk_actor_path,
        "onnx_provider": onnx_provider,

        # ── 학습 cfg 와 일치시켜야 하는 값 (velocity_v0_2_env_cfg.py) ──
        "policy_rate_hz": 50.0,
        "action_scale": 0.25,
        "gait_frequency": gait_freq,
        "gait_frequency_scale": 2.0,
        "gait_stop_rate": 2.0,
        "standing_threshold": 0.1,

        # ── 보행 모드 ──
        # 경기 파이프라인 기본은 auto: /cmd_vel 속도에 따라 보행/정지가
        # 자동 전환된다. march(제자리 계속걷기)/stop(강제 정지)은 벤치
        # 테스트용이며, 상위 제어 노드(전략 FSM)와 충돌할 수 있다.
        "gait_mode": gait_mode,
        "stop_delay_s": stop_delay_s,

        # ── hold ──
        "startup_hold_seconds": 1.0,
        "resume_hold_seconds": 1.0,
        # 킥 복귀 전용 hold. -1.0 이면 위 resume_hold_seconds(1.0) 를 따른다.
        #   킥 직후는 서 있는 상태에서 자세만 흐트러진 것이라, 바닥에서
        #   일어나는 낙상 복귀보다 짧아도 된다. 0.0 이면 hold 없이 즉시
        #   보행으로 넘어간다(테스트용 — last_action 이 0 으로 리셋된 채
        #   첫 추론이 나가므로 발산 위험이 있다).
        "kick_resume_hold_seconds": kick_resume_hold,
        # ── 킥 양보 워치독 (자동 계산) ──
        #   워치독 = kick_duration + kick_yield_margin.
        #   walking 러너는 킥 러너의 kick_duration_s 를 직접 읽을 수 없으므로
        #   (노드가 분리되어 있다) launch 가 같은 값을 양쪽에 넘겨 짝을 맞춘다.
        #   ★ 순서 불변식 ★ 이 워치독 < vision_servo.yaml 의
        #     kick_wait_timeout_s 여야 한다. 서보가 먼저 포기하면 /kick_done
        #     없이 DONE 으로 가버려 FSM 의 ALIGN_KICK 하드 래치가 풀리지 않는다.
        #     walking 이 먼저 복귀하면서 복구용 /kick_done 을 쏘도록 되어 있다.
        "kick_yield_timeout_s": -1.0,        # -1.0 = 자동
        "kick_duration_hint_s": kick_duration,
        "kick_yield_margin_s": kick_yield_margin,
        # 불변식 검사용 참조값. vision_servo.yaml 에서 직접 읽어 넘기므로
        # 한쪽만 고쳐서 어긋나는 일이 없다. 파일을 못 찾으면 0(검사 생략).
        "servo_kick_wait_timeout_ref_s": _servo_kick_wait_timeout(),

        # 킥 핸드오프 진단용. CSV 마지막 3컬럼(run_state/publishing/robot_state)에
        # 양보·hold 구간까지 남으므로, 킥 전후 gait/cmd 추이를 한 파일에서 본다.
        # 경기 전에는 false 로 되돌릴 것(파일이 계속 쌓인다).
        "enable_csv_log": True,

        # [2026-08 CPU 최적화] projected gravity 취득 방식.
        #   /imu/projected_gravity 구독을 없애고 /imu/data.orientation 에서
        #   파생 계산한다(깨우기 400 → 200/초). /imu/data 는 200Hz 그대로라
        #   관측 신선도는 변하지 않는다.
        "gravity_source": gravity_source,
    }
    nodes.append(Node(
        package="humanoid_policy_runner",
        executable="actor_onnx_walking_runner",
        name="actor_onnx_walking_node",
        output=node_output,
        parameters=[structural, actor_params],
    ))

    # ── 킥 러너 (walkkick_v1) ────────────────────────────────────────────
    #  [왜 여기 있나]
    #    예전 actor_onnx_walk_residual_kick_runner 는 보행+킥을 한 노드에서
    #    전환했다(residual 킥이라 보행 정책 위에 얹을 수 있었다). 지금 킥은
    #    접근 보행까지 포함한 독립 정책(관측 265)이라 얹을 수 없어 노드가
    #    분리된다. 두 노드가 /joint_goals 를 공유하므로 반드시 소유권을
    #    주고받아야 하고, 그 짝을 한곳에서 관리하려고 control.launch 에 둔다.
    #    (match / control_joypad 양쪽이 use_kick 으로 켜기만 하면 된다)
    #
    #  [소유권 핸드오프]
    #    /kick_trigger → walking 러너가 발행 중단(양보)
    #                  → 이 노드가 접근 보행 + 킥
    #    /kick_done    → walking 러너 복귀(kick_resume_hold ease-in)
    #    어느 순간에도 /joint_goals 발행자는 하나뿐이다.
    #
    #  [공 위치]
    #    ball_source=param 이고, 트리거 순간의 값이 freeze 되어 "목표 명세"
    #    가 된다. 경기에서는 visual_servo_kick_node 가 트리거 직전에 비전
    #    실측 공위치를 SetParameters 로 이 노드에 밀어넣는다
    #    (vision_servo.yaml 의 push_ball_pos_to_kicker / kick_node_name).
    #    조이패드 단독 테스트에서는 push 가 없으므로 아래 기본값이 그대로 쓰인다.
    kick_params = {
        "actor_path": kick_actor_path,
        "onnx_provider": onnx_provider,
        "ball_source": "param",
        "ball_pos_b_x": kick_ball_x,
        "ball_pos_b_y": 0.0,
        # 비전 서보가 로봇–공–골대를 일직선으로 정렬한 뒤 트리거하므로,
        # "정면"이 곧 골대 방향이다. 별도 목표 방향 소스가 필요 없다.
        "target_source": "param",
        # 학습 분포는 공 거리 0.20~0.50 m. 비전 서보의 정렬 목표
        # (target_ball_x ± kick_pos_tol_x)가 그 안에 들어오므로 게이트를
        # 켜둔 채로 운용할 수 있다. OOD 발산을 막는 마지막 방어선이라
        # 특별한 이유 없이 끄지 말 것.
        "strict_ood_gate": kick_strict_gate,
        # 학습 분포 vx 0.1~0.5.
        # ⚠ 하한 0.1 은 gait standing_threshold(0.1)와 같아, 0.15 미만이면
        #   명령 노름이 임계에 걸려 freq=0 이 되고 킥 중에 걷지 않는다.
        "cmd_vx": kick_cmd_vx,
        "cmd_vy": 0.0,
        "cmd_wz": 0.0,
        # ★ /cmd_vel 구독 차단 — 필수 ★
        #   조이패드/전략이 /cmd_vel 을 계속 발행하므로, 구독을 켜두면
        #   트리거 순간의 /cmd_vel(스틱을 놓았다면 0)이 cmd_init 으로
        #   얼어붙어 공까지 걸어가지 못한다. 위 cmd_vx 를 지키려면 꺼야 한다.
        "use_cmd_vel_topic": False,
        # 트리거 직후 default 자세 hold(급발진 방지). 킥 '전' 구간이다.
        "resume_hold_seconds": kick_start_hold,
        "kick_duration_s": kick_duration,
        # ★ 핸드오프 필수 ★ True 면 킥이 아닐 때도 default 자세를 계속
        #   발행해 walking 러너와 /joint_goals 를 두고 싸운다.
        "idle_publish_default_pose": False,
        # 트리거는 조이패드 A 버튼 또는 visual_servo_kick_node 가 준다.
        "auto_start_kick": False,
        # 중력 관측 트림 비활성(원본 grav 사용).
        #   walking 러너는 robot_params.yaml 의 grav_offset_x 를 받지만 이
        #   노드는 받지 않는다. 두 러너의 계산식이 달라(walking=뺄셈 /
        #   여기=y축 회전) 부호가 뒤집히므로, 값을 옮길 때는
        #     grav_pitch_offset_rad = -arcsin(grav_offset_x)
        #   변환이 필요하다. 지금은 영향을 주지 않도록 명시적으로 0 으로 둔다.
        "grav_pitch_offset_rad": 0.0,
        "enable_csv_log": False,

        # [2026-08 CPU 최적화] walking 러너와 동일. 두 러너를 따로 두면
        # 한쪽만 롤백해 관측이 어긋나는 사고가 나므로 같은 인자를 공유한다.
        "gravity_source": gravity_source,
    }
    nodes.append(Node(
        package="humanoid_policy_runner",
        executable="actor_onnx_walkkick_v1_runner",
        name="actor_onnx_walkkick_v1_node",
        output=node_output,
        parameters=[kick_params],
        condition=IfCondition(LaunchConfiguration("use_kick")),
    ))
    return nodes


def generate_launch_description():
    use_twist_mux = LaunchConfiguration("use_twist_mux")
    use_correction = LaunchConfiguration("use_correction")
    use_teleop = LaunchConfiguration("use_teleop")
    node_output = LaunchConfiguration("node_output")

    twist_mux_cfg = os.path.join(
        get_package_share_directory("humanoid_bringup"),
        "config", "twist_mux_topics.yaml",
    )
    # 정책/레퍼런스 기본 경로 — 노드 내부 기본값과 동일하게 패키지 models/ 기준.
    pol_models = os.path.join(
        get_package_share_directory("humanoid_policy_runner"), "models")

    # ── 1) twist_mux : cmd_vel_out → /cmd_vel_raw ──
    twist_mux_node = Node(
        package="twist_mux", executable="twist_mux", name="twist_mux",
        output=node_output, parameters=[twist_mux_cfg],
        remappings=[("cmd_vel_out", "/cmd_vel_raw")],
        condition=IfCondition(use_twist_mux),
    )
    # ── 2) 횡 drift 보정 노드는 _robot_nodes_setup(OpaqueFunction) 안에서 생성 ──
    #  robot_params.yaml 의 vy_trim 등을 로봇별로 적용하기 위해 actor 와 함께 만든다.
    #  use_correction=false 면 OpaqueFunction 이 correction 을 빼고, 아래 direct 가 직결.
    # use_correction=false → mux 출력을 /cmd_vel 로 직접
    twist_mux_direct = Node(
        package="twist_mux", executable="twist_mux", name="twist_mux",
        output=node_output, parameters=[twist_mux_cfg],
        remappings=[("cmd_vel_out", "/cmd_vel")],
        condition=UnlessCondition(use_correction),
    )
    # ── 4) 넘어짐 보호 + 상태머신 ──
    # [2026-08 CPU 최적화] IMU 토픽을 인자로 받는다.
    #   이 노드는 그동안 launch 에서 파라미터를 전혀 받지 않았다. 그래서
    #   imu_topic 을 노드에 만들어도 launch 로 바꿀 수 없었다(= 롤백 불가).
    #   여기서 parameters 를 뚫어 두면 재빌드 없이 200Hz 로 되돌릴 수 있다.
    fall_manager_node = Node(
        package="humanoid_serial_bridge", executable="fall_manager_node",
        name="fall_manager_node", output=node_output,
        parameters=[{"imu_topic": LaunchConfiguration("fall_manager_imu_topic")}],
    )
    # ── 5) (선택) 키보드 텔레옵 ──
    teleop_node = Node(
        package="humanoid_policy_runner", executable="keyboard_teleop_node",
        name="keyboard_teleop_node", output="screen",
        remappings=[("cmd_vel", "cmd_vel_teleop")],
        condition=IfCondition(use_teleop),
    )

    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value=""),  # 비면 hostname 앞부분
        DeclareLaunchArgument("use_twist_mux", default_value="true"),
        DeclareLaunchArgument("use_teleop", default_value="false"),
        DeclareLaunchArgument("use_correction", default_value="true"),
        DeclareLaunchArgument("node_output", default_value="screen"),
        # ── 보행 정책 (v1s 러너) ──
        # 모델은 humanoid_policy_runner 패키지 share 안의 models/ 에서 찾는다.
        # 절대경로 하드코딩 금지 — 로봇마다 홈 디렉터리가 달라 깨진다.
        # 모델 교체는 src/humanoid_policy_runner/models/walk_policy.onnx 를
        # 갈아끼우고 colcon build 하면 된다(--symlink-install 이면 즉시 반영).
        DeclareLaunchArgument("walk_actor_path",
                              default_value=os.path.join(pol_models, "walk_policy.onnx"),
                              description="보행 정책 ONNX 경로. 기본값은 패키지 share/models"),
        DeclareLaunchArgument("policy_variant", default_value="v0_2",
                              description="정책 변형. ONNX 관측 차원과 반드시 일치 "
                                          "(v0_1|v0_2|v1|v1s). v0_2 와 v1s 는 240 으로 "
                                          "차원이 같아 자동 판별 불가"),
        DeclareLaunchArgument("kick_actor_path",
                              default_value=os.path.join(pol_models, "walkkick_v1_policy.onnx")),
        DeclareLaunchArgument("onnx_provider", default_value="auto"),
        DeclareLaunchArgument("stop_delay_s", default_value="1.0"),
        DeclareLaunchArgument("gait_mode", default_value="auto",
                              description="보행 모드. auto=속도에 따라 자동 보행/정지"
                                          "(경기 기본), march=제자리 계속걷기, "
                                          "stop=강제 정지. auto 외에는 상위 제어 노드와 "
                                          "충돌할 수 있으니 벤치 테스트에서만 쓸 것"),
        DeclareLaunchArgument("gait_freq", default_value="2.0",
                              description="보행 주파수 [Hz]. v0_2 는 2.0 고정 학습"),
        # ★ 기본값을 0.0 으로 바꿨다 (2026-08-20) ★
        #   -1.0(=resume_hold_seconds 1.0초 상속)이던 시절, bringup 경로가
        #   이 인자를 전달하지 않아 비전 서보/match 에서만 킥 후 1.0초
        #   정지가 남았다(로봇이 default 자세로 굳음). 조이패드 경로는
        #   control_joypad 가 0.0 을 직접 넘겨 증상이 없었다.
        #   킥 복귀는 서 있는 상태에서 자세만 흐트러진 것이라 hold 가
        #   필요 없다(낙상 복귀 hold 는 resume_hold_seconds 로 별개).
        DeclareLaunchArgument("kick_resume_hold", default_value="0.0",
                              description="킥 종료 후 보행 복귀 hold [s]. "
                                          "-1.0=resume_hold_seconds(1.0) 상속, "
                                          "0.0=hold 없이 즉시 보행(테스트용). "
                                          "낙상 복귀 hold 와는 별개다"),
        # ── 킥(walkkick_v1) 인자 ──
        #   use_kick=false 가 기본(opt-in)이다. match.launch.py 와
        #   control_joypad.launch.py 가 명시적으로 true 를 넘긴다.
        DeclareLaunchArgument("use_kick", default_value="false",
                              description="킥 러너(actor_onnx_walkkick_v1_runner) "
                                          "실행 여부. /kick_trigger 구독자가 "
                                          "되며, 없으면 비전 서보가 트리거를 "
                                          "쏴도 아무도 듣지 않는다"),
        DeclareLaunchArgument("kick_ball_x", default_value="0.30",
                              description="공까지 전방 거리 [m] (base_link). "
                                          "학습 분포 0.20~0.50 m. 경기에서는 "
                                          "visual_servo 가 트리거 직전에 실측값으로 "
                                          "덮어쓰므로 이 값은 조이패드 단독용이다"),
        DeclareLaunchArgument("kick_cmd_vx", default_value="0.30",
                              description="킥 중 접근 전진 명령 [m/s]. 학습 분포 "
                                          "0.1~0.5 의 중앙. ⚠ 하한 0.1 은 gait "
                                          "standing_threshold(0.1)와 같아 0.15 "
                                          "미만이면 킥 중에 걷지 않는다"),
        # 2026-08-20 실측: 3.0 은 실제 킥 모션보다 길어, 남는 구간에서
        #   어정쩡하게 움직이는 끊김이 생겼다. 1.8~2.0 이 적정.
        #   ※ 늘리면 워치독(= kick_duration + margin)도 함께 늘어난다.
        #     vision_servo.yaml 의 kick_wait_timeout_s(5.0)보다 커지지
        #     않도록 확인할 것.
        DeclareLaunchArgument("kick_duration", default_value="2.0",
                              description="트리거 후 킥 구동 시간 [s]. 이 시간이 "
                                          "지나면 /kick_done 을 내고 보행에 "
                                          "소유권을 돌려준다. kick_ball_x 를 "
                                          "늘리면 함께 늘려야 한다"),
        DeclareLaunchArgument("kick_start_hold", default_value="0.3",
                              description="[킥 전] 트리거 직후 default 자세 hold [s]. "
                                          "급발진 방지. 킥 후 hold 는 kick_resume_hold"),
        DeclareLaunchArgument("kick_yield_margin", default_value="1.0",
                              description="킥 양보 워치독 여유 [s]. "
                                          "워치독 = kick_duration + 이 값. "
                                          "★ 결과가 vision_servo.yaml 의 "
                                          "kick_wait_timeout_s 보다 작아야 "
                                          "한다(순서 불변식)"),
        # ── [2026-08 CPU 최적화] IMU 깨우기 절감 관련 인자 ──────────────
        #  배경: rclpy 는 executor 가 깨어날 때마다 wait set 을 Python 에서
        #        재구축한다. 그래서 비용은 "콜백이 무슨 일을 하느냐"가 아니라
        #        "몇 번 깨어나느냐"에 붙는다. 200Hz IMU 를 50Hz 로만 쓰는
        #        노드들이 깨우기의 대부분을 차지하고 있었다.
        DeclareLaunchArgument(
            "gravity_source", default_value="derive",
            description="정책 러너의 projected gravity 취득 방식. "
                        "derive=/imu/data.orientation 에서 파생 계산(기본, "
                        "구독 없음) / subscribe=/imu/projected_gravity 구독"
                        "(롤백 경로) / verify=둘 다 받아 차이를 5초마다 로그"
                        "(검증 전용, CPU 절감 없음). "
                        "오프라인 대조에서 파생값은 원본과 비트 단위로 동일"),
        DeclareLaunchArgument(
            "fall_manager_imu_topic", default_value="/imu/data_50hz",
            description="fall_manager 가 구독할 IMU 토픽. FSM 이 50Hz 라 "
                        "200Hz 는 4개 중 3개를 버렸다. 롤백하려면 "
                        "/imu/data 로 지정"),
        DeclareLaunchArgument("kick_strict_gate", default_value="true",
                              description="학습 분포 게이트(공 거리 0.20~0.50 m, "
                                          "방위 ±60°). OOD 발산을 막는 마지막 "
                                          "방어선이라 특별한 이유 없이 끄지 말 것"),
        # ── (미사용) 구 킥 러너용 인자 ──
        # actor_onnx_residual_kick_power_runner 는 2026-08-18 삭제됐고
        # residual_kick_power_policy.onnx 도 models/ 에 없다. 그래도
        # control_teleop.launch.py / control_joypad.launch.py 가 아직 이
        # 인자를 전달하므로 선언만 남긴다(제거하면 "선언되지 않은 인자"로
        # 실패한다). 세 런치에서 전달을 모두 끊은 뒤 함께 지울 것.
        DeclareLaunchArgument("residual_actor_path", default_value="",
                              description="(미사용) 구 킥 러너용. 러너가 삭제되어 "
                                          "무시됨. 전달처를 모두 끊은 뒤 제거할 것"),
        DeclareLaunchArgument("reference_csv_path", default_value="",
                              description="(미사용) 구 킥 러너용. 러너가 삭제되어 무시됨"),
        twist_mux_node,
        twist_mux_direct,
        # ── 2)+3) correction + actor (robot YAML 적용) ──
        OpaqueFunction(function=_robot_nodes_setup),
        fall_manager_node,
        teleop_node,
    ])

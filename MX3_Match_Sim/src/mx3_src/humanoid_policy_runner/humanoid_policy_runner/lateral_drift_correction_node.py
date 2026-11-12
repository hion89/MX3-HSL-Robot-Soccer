#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lateral_drift_correction_node.py
==========================================================================
보행 정책 실행 시 로봇이 "명령대로 안 가고 한쪽으로 미끄러지듯 밀려가는"
현상을 ROS 단에서 보정하는 미들웨어 노드.

두 가지 보정을 한다. 적용 조건이 정반대라는 점이 핵심이다.

    [1] vy_trim  (횡 drift, 기존)  — "걷는 중"에만 적용
    [2] vx_trim  (제자리 후진 drift, 2026-08-24 추가) — "제자리"에서만 적용
    [3] wz_trim  (yaw drift, 2026-08-28 추가) — "걷는 중"에만 적용

──────────────────────────────────────────────────────────────────────────
[1] 횡 drift 보정 (vy_trim) — 기존 기능
──────────────────────────────────────────────────────────────────────────
[진단 배경]
    처음엔 yaw(회전)가 누적되는 문제로 봤으나, 실측 결과 로봇은 정면
    heading 을 유지한 채 옆으로 평행이동(lateral translation)하는 형태였다.
      - 증거: vy 에 반대 방향 상수(-0.1)를 주니 직진성이 회복됨.
              순수 yaw 휨이면 vy 로는 상쇄되지 않았을 것이다.
    따라서 보정은 wz(회전)가 아니라 vy(횡속도)에 trim 을 더하는 방식이 맞다.

    ※ 위 진단은 그 시점의 기체·정책 기준이다. 2026-08-28 에 별도로
      "5m 주행 중 yaw 가 약 4.2 deg/s 로 누적되는" 증상이 확인되어
      [3] wz_trim 을 추가했다. 두 현상은 별개이며 동시에 존재할 수 있다.
      어느 쪽인지는 IMU yaw 변화를 보면 갈린다 — yaw 가 변하면 [3],
      yaw 는 그대로인데 옆으로 밀리면 [1] 이다.

    ※ [갱신 2026-08-28] 원래 이 자리에는 "회전(wz) 보정은 쓰지 않는다"는
      주석이 있었다. 그 근거는 "보행은 single-support 마다 몸통이 좌우로
      돌아가 yaw-rate 가 매 스텝 자연 진동하므로, 여기에 wz 피드백을 걸면
      정책이 학습한 보행 리듬과 충돌해 발산한다" 였다.

      ★ 그 판단은 여전히 옳고, [3] wz_trim 은 그것과 다른 것이다 ★
      위 경고의 대상은 "측정한 yaw-rate 를 되먹이는 폐루프 피드백" 이다.
      wz_trim 은 IMU 를 전혀 보지 않는 상수 오픈루프 바이어스로,
      vy_trim 과 완전히 같은 성격이다. 루프가 없으므로 보행 리듬과
      싸울 수 없다.
      → yaw drift 에 대해 IMU 피드백 제어를 붙이는 것은 지금도 금지다.

──────────────────────────────────────────────────────────────────────────
[3] yaw drift 보정 (wz_trim) — 2026-08-28 추가
──────────────────────────────────────────────────────────────────────────
[진단 배경]
    5m 직선 주행(전진 명령 0.30) 중 heading 이 한쪽으로 계속 누적되어,
    도착 시점에 90도 가까이 틀어지는 것이 확인됐다.
    약 4.2 deg/s = 0.074 rad/s 수준이다.

    키보드 A/D 로는 잡을 수 없다. wz_min 이 0.50 이라 한 번만 눌러도
    필요한 보정량의 5배가 걸린다.

[보정식]
    전진 중(vx >= vx_activation)이고 입력 명령이 실제로 걷는 크기일 때만:
        wz_out = wz_user + wz_trim
    부호(ROS REP-103: +z = 반시계 = 좌회전):
        로봇이 왼쪽으로 휘면 wz_trim 은 음수(우회전)로 상쇄한다.

    출발점 계산: 실제 wz ≈ 0.7 × 명령 (nav_cmd_bridge_node 실측)이므로,
    0.074 rad/s 를 상쇄하려면 명령 단위로 약 0.106 이다. 거기서 ±0.02 씩
    조정해 찾는다.

[★ 가드가 왜 필요한가 ★]
    (a) 명령 노름 하한 (wz_trim_min_cmd)
        러너는 norm(vx, vy, wz) < standing_threshold(0.1) 를 standing 으로
        본다. vx_activation(0.05) 만 조건으로 두면, vx=0.05 인 "사실상
        정지" 구간에서 wz_trim 이 더해져 노름이 0.1 을 넘고, 서 있어야 할
        로봇이 제자리에서 돌기 시작한다. 그래서 입력 명령 노름이
        standing_threshold 위(기본 0.12)일 때만 적용한다.
        → 정지락이 걸려 속도가 0 이 된 구간도 이 조건에서 자동으로 걸러진다.
          (vx_trim 과 달리 별도 락 가드가 필요 없는 이유)

    (b) trim 크기 제한 (wz_trim_limit)
        오타 하나로 로봇이 계속 도는 사고를 막는다. 기본 0.20.

    (c) 출력 제한 (wz_out_limit)
        사용자가 이미 최대 회전을 주고 있을 때 trim 이 더해지면 학습 분포
        밖으로 나간다. 보정 후 |wz| 를 기본 1.5 로 클램프한다.

[★ 이건 증상 보정이지 원인 수정이 아니다 ★]
    yaw drift 의 근본 원인은 보통 좌우 다리 비대칭이다. 관절 영점 오프셋
    (혼 조립 톱니 어긋남), 링크 길이 차, 또는 정책 자체의 비대칭.
    진단 순서:
      1) march 모드 제자리걸음 30초 — 제자리에서도 도는가?
      2) 로봇을 180도 돌려 반복 — 로봇 기준인가 필드 기준(바닥 경사)인가?
      3) 다른 기체에서도 같은 방향인가? — 같으면 정책, 다르면 기체
    기본값 wz_trim=0.0 이라 아무것도 안 넣으면 기존 동작과 100% 동일하다.

[보정식]
    전진 중(vx >= vx_activation)일 때만:
        vy_out = vy_user + vy_trim
    그 외(정지/후진/이미 횡이동 명령 중)에는 보정하지 않는다.
        - 정지 상태에서 vy_trim 을 더하면 제자리에서 옆으로 기어가므로 제외.
    부호(ROS REP-103: +y = 왼쪽):
        로봇이 왼쪽(+y)으로 밀리면 vy_trim 은 음수(오른쪽)로 상쇄한다.
        (실측에서 vy=-0.1 로 직진성 회복 → vy_trim 의 출발점이 -0.1)

──────────────────────────────────────────────────────────────────────────
[2] 제자리 후진 drift 보정 (vx_trim) — 2026-08-24 추가
──────────────────────────────────────────────────────────────────────────
[무엇을 하나]
    제자리걸음(march)에서 로봇이 뒤로 밀릴 때, 명령이 사실상 0 인 구간에
    한해 아주 작은 전진 명령을 더해 상쇄한다.
        vx_out = vx_user + vx_trim      (조건 만족 시에만)

[★ 왜 가드가 이렇게 많은가 ★]
    vy_trim 은 "걷는 중"에만 붙으므로 정지·정지락 구간을 건드리지 않는다.
    그런데 vx_trim 은 정확히 그 반대로 "명령이 0 인 구간"에 붙어야 효과가
    난다. 즉 시스템에서 가장 안전해야 할 구간을 건드린다. 그래서 아래 네
    가지를 전부 통과할 때만 적용한다.

      (a) 정지락 바이패스
          이 노드는 twist_mux **뒤**에 있다. game_stop_lock / estop_lock 이
          속도를 0 으로 눌러도 그 뒤에서 vx_trim 을 더하면 Penalized /
          Initial / Finished 상태에서 로봇이 앞으로 기어간다(반칙이고,
          긴급정지도 무력화된다). → 락이 걸려 있으면 무조건 적용 금지.

      (b) gait_mode 제한
          기본값은 march 에서만. auto 에서 스틱을 놓은 뒤의 정지 구간까지
          건드리면 "정지 신호" 의미가 흐려진다. 필요하면
          vx_trim_gait_modes:=["march","auto"] 로 넓힐 수 있다.

      (c) 제자리 판정
          입력 명령의 노름이 vx_trim_max_cmd(기본 0.02) 이하일 때만.
          즉 사람이나 FSM 이 실제로 명령을 주고 있으면 절대 끼어들지 않는다.

      (d) 출력 노름 클램프  ★ 이게 제일 중요하다 ★
          러너는 norm(vx, vy, wz) < standing_threshold(0.1) 를 "standing"
          으로 본다. 이 경계를 넘기면 auto 모드에서 로봇이 영원히 멈추지
          않는다. 그래서 보정 후 출력 노름을 vx_trim_clamp(기본 0.09) 로
          하드 클램프해 0.1 을 절대 넘지 못하게 한다.
          → march 에서는 freq 가 유지되므로 보정이 실제로 먹고,
            auto 에서는 여전히 standing 으로 판정되어 정지 로직이 그대로다.

[★ 이건 증상 보정이지 원인 수정이 아니다 ★]
    제자리에서 뒤로 밀리는 근본 원인은 보통 자세(pitch) 또는 학습 default
    자세 불일치다. grav_offset_x / imu_mount_pitch_deg / leg_init_q 를 먼저
    확인하고, 그래도 남는 잔여 drift 를 메우는 용도로 쓸 것.
    기본값 vx_trim=0.0 이라 아무것도 안 넣으면 기존 동작과 100% 동일하다.

──────────────────────────────────────────────────────────────────────────
[전체 구조]
    twist_mux --(/cmd_vel_raw)--> [이 노드] --(/cmd_vel)--> actor_onnx_runner
    - actor 러너 코드는 건드리지 않는다(기존대로 /cmd_vel 구독).
    - 이 노드를 끄면(use_correction:=false) 보정이 사라진다.
    - IMU 는 사용하지 않는다. 순수 오픈루프 trim 이라 피드백 루프가 없어
      보행 리듬과 싸우지 않는다.

[RViz / Foxglove 디버그]
    /cmd_vel_correction/debug/state          (std_msgs/String)
        OFF / ACTIVE / BLOCK:LOCK / BLOCK:MODE / BLOCK:CMD
    /cmd_vel_correction/debug/vx_applied     (std_msgs/Float32)
    /cmd_vel_correction/debug/vy_applied     (std_msgs/Float32)
    /cmd_vel_correction/debug/wz_applied     (std_msgs/Float32)
        실제로 더해진 wz 보정량 [rad/s 명령 단위]. 오도메트리 실험 중에는
        이 값을 기록해 두어야 "그 구간에 회전 바이어스가 얼마나 섞였는지"
        를 나중에 되짚을 수 있다.
    /cmd_vel_correction/debug/markers        (visualization_msgs/MarkerArray)
        base_link 기준 화살표(적용 중인 보정 벡터) + 상태 텍스트.
        초록=적용중 / 빨강=정지락 차단 / 회색=조건 미충족.
    RViz 에서 MarkerArray 를 /cmd_vel_correction/debug/markers 로 추가하면
    "언제 보정이 들어갔는지" 를 눈으로 확인하면서 튜닝할 수 있다.
==========================================================================
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray


class LateralDriftCorrectionNode(Node):
    def __init__(self):
        super().__init__("lateral_drift_correction_node")

        # ══════════════════════════════════════════════════════════════
        # [1] 횡 drift 보정 파라미터 (기존)
        # ══════════════════════════════════════════════════════════════
        # 전진 시 vy 에 더해지는 고정 보정값(명령 스케일).
        # 왼쪽으로 밀리면 음수. 실측 출발점은 -0.1.
        self.declare_parameter("vy_trim", -0.1)

        # 이 값 이상으로 전진할 때만 vy trim 을 적용한다.
        # 정지/후진 시에는 옆으로 기어가는 것을 막기 위해 보정하지 않는다.
        self.declare_parameter("vx_activation", 0.05)

        # True 면 vy_trim 을 전진속도에 비례시킨다: vy_corr = vy_trim * vx/vx_ref
        # 빠르게 걸을수록 더 휘는 경향이 있으면 켠다. 기본은 False(고정 상수).
        # 단, 비례는 "휨이 vx 에 선형"이라는 추가 가정이므로, 여러 속도에서
        # 실측해 보고 정말 그럴 때만 켜는 것을 권장한다.
        self.declare_parameter("scale_with_vx", False)
        self.declare_parameter("vx_ref", 0.3)  # vy_trim 을 측정한 기준 전진속도

        # ══════════════════════════════════════════════════════════════
        # [2] 제자리 후진 drift 보정 파라미터 (신규)
        # ══════════════════════════════════════════════════════════════
        # 제자리 구간에 더할 전진 명령 [m/s]. 0.0 = 완전 비활성(기본).
        # 뒤로 밀리면 양수. 0.02 정도부터 0.01 씩 올려가며 찾는다.
        self.declare_parameter("vx_trim", 0.0)

        # [제자리 판정] 입력 명령 노름 sqrt(vx^2+vy^2+wz^2) 이 이 값 이하일
        # 때만 적용한다. 사람이나 FSM 이 실제 명령을 주는 중이면 끼어들지
        # 않게 하는 가드다. 0.02 는 "사실상 0" 을 뜻한다.
        self.declare_parameter("vx_trim_max_cmd", 0.02)

        # [출력 노름 클램프] ★ 안전 핵심 ★
        # 러너의 standing_threshold(0.1) 미만을 반드시 유지해야 auto 모드의
        # 정지 로직(속도 0 이 stop_delay_s 지속 → freq 0)이 살아남는다.
        # 0.1 이상으로 키우지 말 것.
        self.declare_parameter("vx_trim_clamp", 0.09)

        # [모드 제한] 이 gait_mode 목록에 있을 때만 적용한다.
        # 빈 리스트면 모드와 무관하게 적용(권장하지 않음).
        #   ["march"]         : 제자리걸음에서만 (기본, 가장 안전)
        #   ["march","auto"]  : auto 의 정지 대기 구간까지 포함
        self.declare_parameter("vx_trim_gait_modes", ["march"])

        # [정지락 미수신 시 동작]
        #   False(기본) : 락 토픽을 한 번도 못 받았으면 벤치 테스트로 보고 적용.
        #                 (game_controller 없이 control_joypad 만 띄운 상황)
        #   True        : 락 상태를 모르면 적용하지 않는다. ★ 경기 스택에서는
        #                 반드시 true 로 둘 것 ★
        self.declare_parameter("vx_trim_require_lock", False)

        # 락 값이 이 시간 이상 갱신되지 않으면 "미수신" 으로 본다 [s].
        # game_state_to_lock_node 는 타이머로 주기 재발행하므로 1.5s 면 충분.
        self.declare_parameter("lock_timeout_s", 1.5)

        # 구독할 정지락 토픽. twist_mux_topics.yaml 의 locks 와 같은 이름이어야
        # 의미가 있다(같은 신호를 봐야 하므로).
        self.declare_parameter("game_stop_lock_topic", "game_stop_lock")
        self.declare_parameter("estop_lock_topic", "estop_lock")

        # /gait_mode 를 아무도 발행하지 않았을 때 가정할 초기 모드.
        # 러너/조이패드 기본값과 같은 "auto" 로 두면, 모드 정보가 없을 때
        # (기본 설정에서는) vx_trim 이 꺼진 채로 시작한다.
        self.declare_parameter("gait_mode_initial", "auto")
        self.declare_parameter("gait_mode_topic", "/gait_mode")

        # ══════════════════════════════════════════════════════════════
        # [3] yaw drift 보정 파라미터 (2026-08-28 신규)
        # ══════════════════════════════════════════════════════════════
        # 걷는 중 wz 에 더해지는 고정 보정값(명령 스케일). 0.0 = 비활성(기본).
        # 로봇이 왼쪽(+z, 반시계)으로 휘면 음수를 넣는다.
        #
        # [출발점 잡는 법]
        #   측정한 drift 를 rad/s 로 환산한 뒤 실측 스케일 0.7 로 나눈다.
        #     4.2 deg/s = 0.0733 rad/s  →  0.0733 / 0.7 ≈ 0.105
        #   부호는 휘는 방향의 반대. 거기서 ±0.02 씩 조정한다.
        self.declare_parameter("wz_trim", 0.0)

        # [크기 제한] |wz_trim| 을 이 값으로 자른다. 오타(예: 1.05)로 로봇이
        # 계속 도는 사고를 막는 안전장치다. 정상적인 drift 보정은 0.2 를
        # 넘을 이유가 없다 — 넘어야 한다면 기구 문제이므로 그쪽을 고칠 것.
        self.declare_parameter("wz_trim_limit", 0.20)

        # [명령 노름 하한] ★ 안전 핵심 ★
        # 입력 명령 노름 sqrt(vx^2+vy^2+wz^2) 이 이 값 이상일 때만 적용한다.
        # 러너의 standing_threshold(0.1) 보다 위여야 한다. 안 그러면 사실상
        # 정지 상태(vx=0.05 등)에 trim 이 더해져 노름이 0.1 을 넘고, 서
        # 있어야 할 로봇이 제자리에서 회전한다.
        # 정지락으로 속도가 0 이 된 구간도 이 조건에서 자동으로 걸러진다.
        self.declare_parameter("wz_trim_min_cmd", 0.12)

        # [출력 제한] 보정 후 |wz| 를 이 값으로 클램프한다.
        # 사용자가 이미 최대 회전을 주는 중이면 trim 이 더해져 학습 분포
        # 밖으로 나갈 수 있다. 조이패드 wz_max 기본값과 맞춰 1.5 로 둔다.
        self.declare_parameter("wz_out_limit", 1.5)

        # True 면 wz_trim 을 전진속도에 비례시킨다: wz_trim * vx/vx_ref
        # 기본 False 를 권한다. yaw drift 는 보통 걸음당 비대칭에서 오고,
        # 걸음 주파수는 vx 와 무관하게 거의 일정하므로(gait_freq 고정),
        # drift 는 rad/s 단위로 대체로 상수다. 여러 속도에서 실측해 보고
        # 정말 비례할 때만 켤 것. vx_ref 는 [1] 과 공유한다.
        self.declare_parameter("wz_trim_scale_with_vx", False)

        # ══════════════════════════════════════════════════════════════
        # [4] 디버그 시각화 파라미터
        # ══════════════════════════════════════════════════════════════
        self.declare_parameter("debug_enable", True)
        self.declare_parameter("debug_rate_hz", 10.0)     # 마커/토픽 발행 상한
        self.declare_parameter("debug_frame", "base_link")
        # 보정 벡터는 최대 0.09 m/s 라 그대로 그리면 화살표가 9cm 밖에 안 된다.
        # RViz 에서 보이라고 배율을 곱한다(길이 = 속도 * 이 값 [m/(m/s)]).
        self.declare_parameter("debug_arrow_scale", 5.0)

        # ── 파라미터 캐싱 ──
        self.vy_trim = float(self.get_parameter("vy_trim").value)
        self.vx_activation = float(self.get_parameter("vx_activation").value)
        self.scale_with_vx = bool(self.get_parameter("scale_with_vx").value)
        self.vx_ref = float(self.get_parameter("vx_ref").value)

        self.vx_trim = float(self.get_parameter("vx_trim").value)
        self.vx_trim_max_cmd = float(self.get_parameter("vx_trim_max_cmd").value)
        self.vx_trim_clamp = float(self.get_parameter("vx_trim_clamp").value)
        self.vx_trim_modes = [str(m).lower() for m in
                              (self.get_parameter("vx_trim_gait_modes").value or [])]
        self.vx_trim_require_lock = bool(
            self.get_parameter("vx_trim_require_lock").value)
        self.lock_timeout_s = float(self.get_parameter("lock_timeout_s").value)

        # ── wz_trim 캐싱 ──
        # 한도는 여기서 한 번, 그리고 파라미터 갱신 콜백에서 다시 적용한다.
        # 런타임에 rqt 로 큰 값을 넣어도 막혀야 하기 때문이다.
        self.wz_trim_limit = abs(float(self.get_parameter("wz_trim_limit").value))
        self.wz_trim = self._clamp_wz_trim(
            float(self.get_parameter("wz_trim").value))
        self.wz_trim_min_cmd = float(self.get_parameter("wz_trim_min_cmd").value)
        self.wz_out_limit = abs(float(self.get_parameter("wz_out_limit").value))
        self.wz_trim_scale_with_vx = bool(
            self.get_parameter("wz_trim_scale_with_vx").value)

        self.debug_enable = bool(self.get_parameter("debug_enable").value)
        self.debug_rate_hz = float(self.get_parameter("debug_rate_hz").value)
        self.debug_frame = str(self.get_parameter("debug_frame").value)
        self.debug_arrow_scale = float(self.get_parameter("debug_arrow_scale").value)

        # ── 런타임 상태 ──
        self.gait_mode = str(self.get_parameter("gait_mode_initial").value).lower()
        # 락 토픽별 (최신값, 수신 시각). 한 번도 못 받았으면 키가 없다.
        self._locks = {}
        self._last_debug_t = 0.0
        self._warned_no_lock_t = 0.0

        # 걷는 걸 보면서 rqt 등으로 실시간 튜닝할 수 있도록 콜백 등록.
        self.add_on_set_parameters_callback(self._on_param_update)

        # ── 명령 경로 (지연 최소화가 최우선) ──
        self.sub = self.create_subscription(
            Twist, "/cmd_vel_raw", self._cb_cmd_raw, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ── gait_mode 구독 ──
        # 조이패드/러너와 동일한 TRANSIENT_LOCAL 이어야 이 노드가 나중에 떠도
        # latch 된 마지막 모드를 받는다. 안 맞추면 다음 모드 변경까지
        # gait_mode_initial 값으로 돌아 vx_trim 이 엉뚱한 때 켜질 수 있다.
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            String, str(self.get_parameter("gait_mode_topic").value),
            self._cb_gait_mode, latched_qos)

        # ── 정지락 구독 ──
        for key in ("game_stop_lock_topic", "estop_lock_topic"):
            topic = str(self.get_parameter(key).value)
            if not topic:
                continue
            self.create_subscription(
                Bool, topic,
                lambda msg, t=topic: self._cb_lock(t, msg), 10)

        # ── 디버그 퍼블리셔 ──
        if self.debug_enable:
            self.dbg_state_pub = self.create_publisher(
                String, "/cmd_vel_correction/debug/state", 10)
            self.dbg_vx_pub = self.create_publisher(
                Float32, "/cmd_vel_correction/debug/vx_applied", 10)
            self.dbg_vy_pub = self.create_publisher(
                Float32, "/cmd_vel_correction/debug/vy_applied", 10)
            self.dbg_wz_pub = self.create_publisher(
                Float32, "/cmd_vel_correction/debug/wz_applied", 10)
            self.dbg_marker_pub = self.create_publisher(
                MarkerArray, "/cmd_vel_correction/debug/markers", 10)

        self.get_logger().info(
            "lateral_drift_correction_node 시작: /cmd_vel_raw -> (vx/vy 보정) -> /cmd_vel")
        self.get_logger().info(
            f"  [vy] vy_trim={self.vy_trim:.4f}  vx_activation={self.vx_activation:.3f}  "
            f"scale_with_vx={self.scale_with_vx}  vx_ref={self.vx_ref:.3f}")
        self.get_logger().info(
            f"  [vx] vx_trim={self.vx_trim:.4f}  max_cmd={self.vx_trim_max_cmd:.3f}  "
            f"clamp={self.vx_trim_clamp:.3f}  modes={self.vx_trim_modes}  "
            f"require_lock={self.vx_trim_require_lock}")
        if self.vx_trim != 0.0:
            # 켜져 있다는 사실을 기동 로그에서 반드시 눈에 띄게 남긴다.
            # (경기 중 "왜 가만히 있는데 앞으로 가지?" 를 몇 시간 뒤에
            #  추적하는 일을 막기 위함)
            self.get_logger().warn(
                f"★ vx_trim 활성 ({self.vx_trim:+.4f} m/s) — 제자리 구간에 전진 "
                f"바이어스가 들어갑니다. 원인 수정(grav_offset_x / imu_mount_pitch_deg "
                f"/ leg_init_q) 이 끝나면 0 으로 되돌리세요.")
        if self.vx_trim_clamp >= 0.1:
            self.get_logger().warn(
                f"★ vx_trim_clamp={self.vx_trim_clamp:.3f} 는 러너의 "
                f"standing_threshold(0.1) 이상입니다. auto 모드에서 로봇이 "
                f"멈추지 않을 수 있습니다.")

        self.get_logger().info(
            f"  [wz] wz_trim={self.wz_trim:+.4f}  limit={self.wz_trim_limit:.3f}  "
            f"min_cmd={self.wz_trim_min_cmd:.3f}  out_limit={self.wz_out_limit:.2f}  "
            f"scale_with_vx={self.wz_trim_scale_with_vx}")
        if self.wz_trim != 0.0:
            # 켜져 있다는 사실을 기동 로그에서 반드시 눈에 띄게 남긴다.
            # 회전 바이어스는 증상이 "왜 자꾸 휘지" 로 나타나서, 켜둔 걸
            # 잊으면 원인 추적이 매우 어려워진다.
            side = "우회전(시계)" if self.wz_trim < 0 else "좌회전(반시계)"
            self.get_logger().warn(
                f"★ wz_trim 활성 ({self.wz_trim:+.4f} → {side}) — 걷는 중 "
                f"회전 바이어스가 들어갑니다. 이건 증상 보정입니다. "
                f"좌우 다리 비대칭(관절 영점/링크 길이)을 찾으면 0 으로 "
                f"되돌리세요.")
        if self.wz_trim_min_cmd < 0.1:
            self.get_logger().warn(
                f"★ wz_trim_min_cmd={self.wz_trim_min_cmd:.3f} 가 러너의 "
                f"standing_threshold(0.1) 미만입니다. 사실상 정지 상태에서 "
                f"보정이 더해져 로봇이 제자리에서 돌 수 있습니다.")

    # ──────────────────────────────────────────────────────────
    # 콜백
    # ──────────────────────────────────────────────────────────
    def _on_param_update(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "vy_trim":
                self.vy_trim = float(p.value)
            elif p.name == "vx_activation":
                self.vx_activation = float(p.value)
            elif p.name == "scale_with_vx":
                self.scale_with_vx = bool(p.value)
            elif p.name == "vx_ref":
                self.vx_ref = float(p.value)
            elif p.name == "vx_trim":
                self.vx_trim = float(p.value)
            elif p.name == "vx_trim_max_cmd":
                self.vx_trim_max_cmd = float(p.value)
            elif p.name == "vx_trim_clamp":
                self.vx_trim_clamp = float(p.value)
            elif p.name == "vx_trim_gait_modes":
                self.vx_trim_modes = [str(m).lower() for m in (p.value or [])]
            elif p.name == "vx_trim_require_lock":
                self.vx_trim_require_lock = bool(p.value)
            elif p.name == "lock_timeout_s":
                self.lock_timeout_s = float(p.value)
            elif p.name == "wz_trim_limit":
                # 한도를 먼저 갱신하고, 현재 trim 을 새 한도로 다시 자른다.
                self.wz_trim_limit = abs(float(p.value))
                self.wz_trim = self._clamp_wz_trim(self.wz_trim)
            elif p.name == "wz_trim":
                self.wz_trim = self._clamp_wz_trim(float(p.value))
            elif p.name == "wz_trim_min_cmd":
                self.wz_trim_min_cmd = float(p.value)
            elif p.name == "wz_out_limit":
                self.wz_out_limit = abs(float(p.value))
            elif p.name == "wz_trim_scale_with_vx":
                self.wz_trim_scale_with_vx = bool(p.value)
            elif p.name == "debug_enable":
                self.debug_enable = bool(p.value)
            elif p.name == "debug_rate_hz":
                self.debug_rate_hz = float(p.value)
            elif p.name == "debug_arrow_scale":
                self.debug_arrow_scale = float(p.value)
        self.get_logger().info(
            f"[param 갱신] vy_trim={self.vy_trim:.4f} vx_trim={self.vx_trim:.4f} "
            f"wz_trim={self.wz_trim:+.4f}")
        return SetParametersResult(successful=True)

    def _clamp_wz_trim(self, value: float) -> float:
        """wz_trim 을 안전 한도로 자른다.

        한도를 넘는 값이 들어오면 경고를 남기고 자른다. 조용히 자르면
        "설정했는데 왜 효과가 없지" 로 시간을 버리게 된다.
        오타(1.05 를 0.105 대신) 하나로 로봇이 계속 도는 사고를 막는 것이
        목적이라, 거부가 아니라 클램프로 처리한다 — 거부하면 이전 값이
        남아 무슨 값이 적용 중인지 더 헷갈린다.
        """
        lim = self.wz_trim_limit
        if abs(value) > lim:
            self.get_logger().warn(
                f"wz_trim={value:+.4f} 이 한도 ±{lim:.3f} 를 넘어 잘렸습니다. "
                f"이만큼 큰 보정이 필요하면 기구 쪽을 먼저 확인하세요.")
            return math.copysign(lim, value)
        return value

    def _cb_gait_mode(self, msg: String):
        """조이패드/전략이 발행하는 /gait_mode 를 따라간다."""
        mode = str(msg.data).strip().lower()
        if mode != self.gait_mode:
            self.get_logger().info(f"[gait_mode] {self.gait_mode} -> {mode}")
        self.gait_mode = mode

    def _cb_lock(self, topic: str, msg: Bool):
        """정지락(game_stop_lock / estop_lock) 최신값과 수신 시각을 보관."""
        self._locks[topic] = (bool(msg.data), time.time())

    # ──────────────────────────────────────────────────────────
    # vx_trim 적용 판정
    # ──────────────────────────────────────────────────────────
    def _lock_state(self):
        """(잠김 여부, 락 정보를 알고 있는지) 를 돌려준다.

        신선도(lock_timeout_s)를 넘긴 값은 "모른다" 로 취급한다. 락 노드가
        죽었는데 마지막 False 를 계속 믿는 상황을 막기 위함이다.
        """
        now = time.time()
        known = False
        for _topic, (val, stamp) in self._locks.items():
            if (now - stamp) > self.lock_timeout_s:
                continue          # 오래된 값 → 모르는 것으로
            known = True
            if val:
                return True, True   # 하나라도 잠겨 있으면 잠김
        return False, known

    def _vx_trim_state(self, cmd_norm: float) -> str:
        """이번 메시지에 vx_trim 을 적용할지 판정하고 상태 문자열을 돌려준다.

        Returns:
            "OFF"        : vx_trim=0 (기능 자체가 꺼짐)
            "BLOCK:LOCK" : 정지락이 걸렸거나(또는 require_lock 인데 미수신)
            "BLOCK:MODE" : 현재 gait_mode 가 허용 목록 밖
            "BLOCK:CMD"  : 사용자가 실제 명령을 주는 중(제자리가 아님)
            "ACTIVE"     : 적용
        """
        if abs(self.vx_trim) < 1e-9:
            return "OFF"

        # (a) 정지락 — 가장 먼저 본다.
        locked, known = self._lock_state()
        if locked:
            return "BLOCK:LOCK"
        if self.vx_trim_require_lock and not known:
            now = time.time()
            if (now - self._warned_no_lock_t) > 10.0:
                self._warned_no_lock_t = now
                self.get_logger().warn(
                    "vx_trim_require_lock=true 인데 정지락 토픽을 못 받고 "
                    "있습니다 — 보정을 적용하지 않습니다. "
                    "game_controller 스택이 떠 있는지 확인하세요.")
            return "BLOCK:LOCK"

        # (b) gait_mode 제한 (빈 목록이면 제한 없음)
        if self.vx_trim_modes and self.gait_mode not in self.vx_trim_modes:
            return "BLOCK:MODE"

        # (c) 제자리 판정
        if cmd_norm > self.vx_trim_max_cmd:
            return "BLOCK:CMD"

        return "ACTIVE"

    # ──────────────────────────────────────────────────────────
    # 메인 경로
    # ──────────────────────────────────────────────────────────
    def _cb_cmd_raw(self, msg: Twist):
        """원본 명령을 받아 보정 후 즉시 재발행한다(지연 최소).

        판정에 필요한 상태(gait_mode / 락)는 별도 콜백에서 갱신되므로
        여기서는 비교만 한다. 무거운 일(마커 생성)은 발행률을 제한한다.
        """
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        wz = float(msg.angular.z)

        out = Twist()
        out.linear.x = vx
        out.linear.y = vy
        out.angular.z = wz       # wz 는 그대로 통과

        # ── [1] 횡 trim : 전진 중일 때만 ──
        vy_applied = 0.0
        if vx >= self.vx_activation:
            if self.scale_with_vx and abs(self.vx_ref) > 1e-6:
                vy_applied = self.vy_trim * (vx / self.vx_ref)
            else:
                vy_applied = self.vy_trim
            out.linear.y += vy_applied

        # ── [1-b] yaw trim : 전진 중 + 실제로 걷는 크기일 때만 ──
        # ★ 입력 노름을 먼저 본다 ★
        # vx_activation(0.05) 만으로 판정하면, 러너가 standing 으로 보는
        # 구간(노름 < 0.1)에 회전 바이어스가 더해져 노름이 0.1 을 넘고,
        # 서 있어야 할 로봇이 제자리에서 돌기 시작한다.
        # 정지락으로 속도가 0 이 된 구간도 이 조건에서 함께 걸러진다.
        in_norm = math.sqrt(vx * vx + vy * vy + wz * wz)
        wz_applied = 0.0
        if (self.wz_trim != 0.0
                and vx >= self.vx_activation
                and in_norm >= self.wz_trim_min_cmd):
            if self.wz_trim_scale_with_vx and abs(self.vx_ref) > 1e-6:
                wz_applied = self.wz_trim * (vx / self.vx_ref)
            else:
                wz_applied = self.wz_trim
            out.angular.z += wz_applied

            # 출력 제한 — 사용자가 이미 최대 회전을 주는 중이면 trim 이
            # 더해져 학습 분포 밖으로 나갈 수 있다.
            if abs(out.angular.z) > self.wz_out_limit:
                out.angular.z = math.copysign(
                    self.wz_out_limit, out.angular.z)
                # 클램프가 깎은 만큼까지 반영해야 디버그 값이 "실제로
                # 나간 보정" 을 가리킨다.
                wz_applied = out.angular.z - wz

        # ── [2] 전진 trim : 제자리일 때만 ──
        # 러너의 standing 판정과 같은 노름을 쓴다(norm(vx, vy, wz)).
        # 같은 기준으로 봐야 "러너가 standing 이라고 보는 구간" 과
        # "여기서 제자리라고 보는 구간" 이 어긋나지 않는다.
        # 위 [1-b] 에서 이미 같은 값을 계산했으므로 재사용한다.
        cmd_norm = in_norm
        state = self._vx_trim_state(cmd_norm)
        vx_applied = 0.0
        if state == "ACTIVE":
            out.linear.x += self.vx_trim
            vx_applied = self.vx_trim

            # (d) 출력 노름 하드 클램프 — standing_threshold 를 넘지 않게.
            #     세 성분을 같은 비율로 줄여 방향은 보존한다.
            n = math.sqrt(out.linear.x ** 2 + out.linear.y ** 2
                          + out.angular.z ** 2)
            if n > self.vx_trim_clamp > 0.0 and n > 1e-9:
                s = self.vx_trim_clamp / n
                out.linear.x *= s
                out.linear.y *= s
                out.angular.z *= s
                # 클램프가 실제로 깎은 양까지 반영해야 디버그 화살표가
                # "실제로 나간 보정" 을 가리킨다.
                vx_applied = out.linear.x - vx
                vy_applied = out.linear.y - vy
                # vx_trim(노름 <= 0.02)과 wz_trim(노름 >= 0.12)은 조건이
                # 배타적이라 여기서 wz_applied 는 실질적으로 항상 0 이다.
                # 그래도 값을 맞춰 두어야 디버그 토픽이 거짓말하지 않는다.
                wz_applied = out.angular.z - wz

        self.pub.publish(out)

        if self.debug_enable:
            self._publish_debug(state, vx_applied, vy_applied, wz_applied)

    # ──────────────────────────────────────────────────────────
    # 디버그 시각화
    # ──────────────────────────────────────────────────────────
    def _publish_debug(self, state: str, vx_applied: float,
                       vy_applied: float, wz_applied: float = 0.0):
        """RViz/Foxglove 용 상태·마커 발행 (debug_rate_hz 로 제한).

        명령 경로(/cmd_vel)는 위에서 이미 발행했으므로, 여기서 시간을 써도
        제어 지연에는 영향이 없다. 그래도 50Hz 로 마커를 만들면 낭비라
        기본 10Hz 로 낮춘다.
        """
        now = time.time()
        min_dt = 1.0 / max(1e-6, self.debug_rate_hz)
        if (now - self._last_debug_t) < min_dt:
            return
        self._last_debug_t = now

        m = String()
        m.data = (f"{state} vx={vx_applied:+.4f} vy={vy_applied:+.4f} "
                  f"wz={wz_applied:+.4f} mode={self.gait_mode}")
        self.dbg_state_pub.publish(m)

        f = Float32()
        f.data = float(vx_applied)
        self.dbg_vx_pub.publish(f)
        f = Float32()
        f.data = float(vy_applied)
        self.dbg_vy_pub.publish(f)
        f = Float32()
        f.data = float(wz_applied)
        self.dbg_wz_pub.publish(f)

        # ── 마커: 화살표(보정 벡터) + 텍스트(상태) ──
        # 색: 초록=적용중 / 빨강=정지락 차단 / 노랑=조건 미충족 / 회색=꺼짐
        if state == "ACTIVE":
            rgba = (0.15, 0.85, 0.30, 0.95)
        elif state == "BLOCK:LOCK":
            rgba = (0.90, 0.15, 0.15, 0.95)
        elif state == "OFF":
            rgba = (0.55, 0.55, 0.55, 0.60)
        else:
            rgba = (0.90, 0.80, 0.15, 0.80)

        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        arrow = Marker()
        arrow.header.frame_id = self.debug_frame
        arrow.header.stamp = stamp
        arrow.ns = "cmd_vel_correction"
        arrow.id = 0
        arrow.type = Marker.ARROW
        length = math.hypot(vx_applied, vy_applied) * self.debug_arrow_scale
        if length < 1e-3:
            # 길이 0 화살표는 RViz 가 경고를 뱉으므로 아예 지운다.
            arrow.action = Marker.DELETE
        else:
            arrow.action = Marker.ADD
            # base_link 원점에서 보정 방향으로. 발밑에 묻히지 않게 살짝 띄운다.
            p0 = Point(x=0.0, y=0.0, z=0.05)
            p1 = Point(x=vx_applied * self.debug_arrow_scale,
                       y=vy_applied * self.debug_arrow_scale,
                       z=0.05)
            arrow.points = [p0, p1]
            # ARROW + points 모드에서 scale 의미:
            #   x=자루 지름, y=머리 지름, z=머리 길이(0이면 자동)
            arrow.scale.x = 0.015
            arrow.scale.y = 0.035
            arrow.scale.z = 0.05
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = rgba
        arr.markers.append(arrow)

        text = Marker()
        text.header.frame_id = self.debug_frame
        text.header.stamp = stamp
        text.ns = "cmd_vel_correction"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = 0.0
        text.pose.position.y = 0.0
        text.pose.position.z = 0.45
        text.pose.orientation.w = 1.0
        text.scale.z = 0.06          # 글자 높이 [m]
        text.color.r, text.color.g, text.color.b, text.color.a = rgba
        text.text = (f"{state}\n"
                     f"vx{vx_applied:+.3f} vy{vy_applied:+.3f}\n"
                     f"wz{wz_applied:+.3f}\n"
                     f"mode={self.gait_mode}")
        arr.markers.append(text)

        self.dbg_marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = LateralDriftCorrectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():        # ← 이 가드 추가
            rclpy.shutdown()


if __name__ == "__main__":
    main()

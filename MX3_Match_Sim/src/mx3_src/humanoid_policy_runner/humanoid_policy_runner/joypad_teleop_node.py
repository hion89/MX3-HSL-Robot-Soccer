#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joypad_teleop_node.py

조이패드(8BitDo 등) 텔레옵 노드.
    /joy → /cmd_vel_teleop + /gait_mode + /gait_frequency + /kick_trigger

[개명 이력]
    2026-08-18  joypad_teleop_v1s_node → joypad_teleop_node
        · 파일   joypad_teleop_v1s_node.py → joypad_teleop_node.py
        · 실행/노드명 joypad_teleop_v1s_node → joypad_teleop_node
        · 클래스 JoypadTeleopV1SNode → JoypadTeleopNode
        원래는 v0 정책용 구 joypad_teleop_node 와 공존시키려고 v1s 를
        붙였는데, 구 노드를 삭제하면서 접미사가 무의미해졌다. 짝이 되는
        keyboard_teleop_node 와 대칭도 맞췄다.
        ※ 이 패키지에서 _runner 접미사는 ONNX 정책 러너 전용이다.
          이 노드는 정책을 돌리지 않으므로 _node 가 맞다.

[주요 기능]
    - 축 매핑, 축별 데드존(대각선 새어듦 방지), 전/후진 비대칭 속도 범위
    - 데드맨(enable_button), 킥 rising edge
    - 자체 타이머 재발행 (twist_mux teleop timeout 유지)

[v1s 전용 추가]
    - /gait_frequency (std_msgs/Float32) : 보행 주파수 실시간 조절
    - /gait_mode      (std_msgs/String)  : auto | march | stop | walk

[추가 매핑 — XInput 기준 기본값. 반드시 `ros2 topic echo /joy` 로 검증할 것]
    D-pad 상/하 (axes[7]) : 주파수 -0.1 / +0.1 Hz
    D-pad 좌/우 (axes[6]) : 보행 모드 순환 (역방향 / 정방향)
    Y 버튼 (buttons[4])   : 주파수를 기본값으로 리셋
    B 버튼 (buttons[1])   : 긴급 정지 (즉시 gait_mode=stop + 속도 0)

    ※ 축/버튼 기본값은 humanoid_bringup/launch/control_joypad.launch.py
      의 실측값과 동일하게 맞췄다. 8BitDo Ultimate 2 (XInput) 기준
      axes: 0=좌스틱LR, 1=좌스틱UD, 2=우스틱LR, 3=우스틱UD, 6/7=D-pad
      buttons: A=0, B=1, X=3, Y=4

    ※ 8BitDo Ultimate 2 는 D-pad 가 axes 로 오는 모드와 buttons 로 오는
      모드가 있다. buttons 로 온다면 axis_dpad_* 를 -1 로 두고
      button_freq_up/down, button_mode_next/prev 를 대신 지정하라.

[주파수 범위]
    학습 분포는 velocity_v1s_env_cfg.py 의 frequency_range=(1.8, 2.2) 다.
    freq_min/freq_max 기본값을 여기에 맞춰 두었다. 이 밖으로 나가면
    분포 외라 성능이 보장되지 않으므로, 넓히려면 훈련 범위를 먼저 넓혀라.

[데드맨 / 긴급정지]
    enable_button 기본값은 -1(비활성)이다. 조작성 때문에 껐다.
    대신 B 버튼이 즉시 gait_mode='stop' 으로 전환하고 속도를 0 으로 만든다.
    데드맨을 다시 쓰려면 enable_button:=5 (RB) 로 실행하라.

    ★ 모드/주파수 조작은 데드맨과 무관하게 항상 동작한다 ★
    데드맨이 모드 전환까지 막으면, RB 를 놓은 상태에서 stop 으로 갈 수 없어
    위상이 계속 돌고 제자리 걷기가 이어진다. 안전상 역효과다.
    데드맨은 cmd_vel 과 킥에만 적용된다.

[보행 모드 의미 — 러너의 resolve_gait_frequency 참조]
    auto  : 속도 0 이 stop_delay_s 이상 지속되면 완전 정지 (기본, 경기용)
    march : 속도 0 이어도 제자리 걷기 유지 (리듬 유지 → 즉시 출발)
    stop  : 속도와 무관하게 완전 정지 (위상이 0 으로 수렴)
    walk  : 속도와 무관하게 주파수 유지 (디버그 전용)

[QoS]
    /gait_frequency 와 /gait_mode 는 TRANSIENT_LOCAL 로 발행한다.
    값이 바뀔 때만 보내되, 러너가 나중에 떠도 마지막 값을 받도록 하기
    위함이다. 매 주기 재발행하면 러너 콜백이 불필요하게 자주 돈다.

[수정 이력]
    2026-08-28  LT/RT 트리거 홀드 회전 부스트 추가.
        배경: 최대 회전속도를 3.0 rad/s 까지 쓰고 싶지만, 항상 3.0 이면
              저속 미세 조작 구간이 좁아지고 오조작 위험도 커진다.
              평소에는 wz_max(1.5), 트리거를 당기고 있는 동안만
              wz_max_boost(3.0) 를 쓰는 2 단 구성으로 만들었다.
        조치: · axis_wz_boost_l / axis_wz_boost_r 로 LT/RT 축을 지정
              · 둘 중 하나라도 임계를 넘으면 부스트 ON (OR 조건)
              · 히스테리시스(on=0.60 / off=0.40)로 임계 근처 채터링 방지
              · 놓으면 즉시 원래 wz_max 로 복귀 (홀드 방식, 토글 아님)
        ★ 무장(arming) 래치가 핵심 안전장치다 ★
              xpad 계열 드라이버는 트리거를 '한 번도 당기지 않은' 동안
              축을 0.0 으로 보고한다. rest=+1.0 규약에서 0.0 은 press=0.5
              에 해당해 임계(0.60) 바로 아래라, 노이즈만으로도 부스트가
              걸릴 수 있는 위험 구간이다. 그래서 press <= arm_below(0.15)
              인 '진짜 릴리스' 값을 최소 1 회 본 축만 부스트를 허용한다.
              기동 후 각 트리거를 한 번씩 당겼다 놓아야 활성화된다.
              (덤으로, axis 인덱스를 잘못 지정하면 대부분 무장이 안 돼
               부스트가 조용히 꺼진 상태로 남는다 — fail-safe)
        주의: 부스트 해제 시 목표가 3.0 → 1.5 로 떨어지는 것은 기존
              슬루 리미터(decel_wz)가 처리한다. 급변하지 않는다.
        주의: 부스트 값은 학습 분포 밖일 수 있다. wz_max_boost 는
              반드시 매단 상태에서 단계적으로 올려 확인할 것.

    2026-08-18  슬루 레이트 리미터(명령 스무딩) 추가.
        배경: 스틱을 100% 밀다 놓으면 목표가 max → 0 으로 한 번에 떨어진다.
        조치: 목표(vx/vy/wz)와 발행값(_out_*)을 분리하고, publish_cmd()
              에서 축별 가감속 한계 안에서만 발행값을 이동시킨다.
              가속/감속 한계를 따로 두어 조작 응답성과 정지 부드러움을
              동시에 잡는다. slew_enable:=false 로 끌 수 있다.
        안전: E-STOP(B 버튼), 데드맨 해제, 축 매핑 오류는 슬루를 우회해
              즉시 0 을 발행한다. 램프를 타면 긴급 정지가 아니게 된다.

    2026-08-18  데드존 통일 + 최소속도 제거.
        배경: 스틱을 놓는 순간 로봇이 휘청이는 증상.
        원인: _map_axis() 가 데드존 경계에서 곧바로 out_min 을 내보내므로
              vx 는 0↔0.15, wz 는 0↔0.50 의 계단이 생긴다. 이 계단이
              velocity_commands 관측에 그대로 들어갔다.
        조치: · deadzone 전 축 0.20 통일 (전역 0.15→0.20, vy 0.35→상속)
              · vx_pos_min 0.15→0, vx_neg_min 0.10→0, vy_min 0.10→0
              · wz_min/max 는 요청에 따라 0.50/1.50 유지 (계단 잔존)
        효과: 스틱 변위 0~100% 가 0~max 에 대응 → 미세 조작 해상도 개선.

    2026-08-17  gait_modes 파라미터 타입 오류 수정.
        증상: control_joypad.launch.py 로 기동하면 이 노드만 즉사.
              InvalidParameterTypeException: Trying to set parameter
              'gait_modes' to '['auto', 'march']' of type 'STRING_ARRAY',
              expecting type 'BYTE_ARRAY'
        원인: declare_parameter('gait_modes', []) 의 빈 리스트 기본값을
              rclpy 가 BYTE_ARRAY 로 추론. launch 의 STRING_ARRAY 와 불일치.
        조치: Parameter.Type.STRING_ARRAY + ParameterDescriptor 로 타입 명시.
              기본값이 사라지므로 get_parameter_or() 로 안전하게 읽는다.
              "비워두면 enable_walk_mode 기본값" 하위호환은 그대로 유지.

[테스트 순서 권장]
    1) 로봇을 매달고 gait_mode=stop 으로 두고 노드만 띄워 축/버튼 확인
    2) march 로 전환 → 제자리 걷기 확인 (/gait/phase 톱니파)
    3) stop 으로 전환 → 두 발이 모이며 멈추는지 확인
    4) auto + 왼쪽 스틱 살짝 → 전진
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String


# 모드 순환 순서. walk 는 디버그용이라 기본 순환에서 제외한다
# (enable_walk_mode=True 로 켜면 포함).
GAIT_MODES_DEFAULT = ["auto", "march", "stop"]
GAIT_MODES_WITH_WALK = ["auto", "march", "stop", "walk"]


class JoypadTeleopNode(Node):
    def __init__(self):
        super().__init__('joypad_teleop_node')

        # ── 토픽 파라미터 ─────────────────────────────────────────
        self.declare_parameter('joy_topic', 'joy')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('kick_topic', '/kick_trigger')
        # v1s 전용. 절대경로 → remap 영향 없이 러너와 직결.
        self.declare_parameter('gait_freq_topic', '/gait_frequency')
        self.declare_parameter('gait_mode_topic', '/gait_mode')

        joy_topic = str(self.get_parameter('joy_topic').value)
        cmd_topic = str(self.get_parameter('cmd_vel_topic').value)
        kick_topic = str(self.get_parameter('kick_topic').value)
        freq_topic = str(self.get_parameter('gait_freq_topic').value)
        mode_topic = str(self.get_parameter('gait_mode_topic').value)

        # ── 축 인덱스 매핑 (XInput/xpad 표준 기본값) ──────────────
        # 8BitDo Ultimate 2 (XInput) 실측값.
        # humanoid_bringup/launch/control_joypad.launch.py 와 동일하게 맞춘다.
        #   axes: 0=좌스틱 좌우, 1=좌스틱 상하, 2=우스틱 좌우, 3=우스틱 상하
        # axis_wz 를 3(우스틱 상하)으로 두면 앞뒤로 밀어야 회전한다.
        self.declare_parameter('axis_vx', 1)
        self.declare_parameter('axis_vy', 0)
        self.declare_parameter('axis_wz', 2)
        self.axis_vx = int(self.get_parameter('axis_vx').value)
        self.axis_vy = int(self.get_parameter('axis_vy').value)
        self.axis_wz = int(self.get_parameter('axis_wz').value)

        self.declare_parameter('invert_vx', False)
        self.declare_parameter('invert_vy', False)
        self.declare_parameter('invert_wz', False)
        self.inv_vx = -1.0 if bool(self.get_parameter('invert_vx').value) else 1.0
        self.inv_vy = -1.0 if bool(self.get_parameter('invert_vy').value) else 1.0
        self.inv_wz = -1.0 if bool(self.get_parameter('invert_wz').value) else 1.0

        # ── 버튼 인덱스 매핑 ──────────────────────────────────────
        # 버튼 실측 (control_joypad.launch.py 주석 참조): A=0, B=1, X=3, Y=4
        self.declare_parameter('button_kick', 0)      # A
        # 데드맨(enable_button): 기본 비활성(-1).
        # 조작성 때문에 껐다. 대신 button_estop(B) 으로 즉시 stop 모드에
        # 진입할 수 있게 해 두었다. 바닥 테스트에서 필요하면 5(RB) 로 켤 것.
        self.declare_parameter('enable_button', -1)
        self.button_kick = int(self.get_parameter('button_kick').value)
        self.enable_button = int(self.get_parameter('enable_button').value)

        # ── v1s: D-pad 매핑 (axes 방식) ──────────────────────────
        # 대부분의 XInput 드라이버에서 D-pad 는 axes[6](좌우), axes[7](상하).
        # 값은 -1 / 0 / +1 로 오며, 방향에 따라 부호가 뒤집힌 패드도 있다.
        self.declare_parameter('axis_dpad_ud', 7)   # 상하 → 주파수
        self.declare_parameter('axis_dpad_lr', 6)   # 좌우 → 모드 순환
        self.declare_parameter('invert_dpad_ud', False)
        self.declare_parameter('invert_dpad_lr', False)
        self.axis_dpad_ud = int(self.get_parameter('axis_dpad_ud').value)
        self.axis_dpad_lr = int(self.get_parameter('axis_dpad_lr').value)
        self.inv_dpad_ud = -1.0 if bool(self.get_parameter('invert_dpad_ud').value) else 1.0
        self.inv_dpad_lr = -1.0 if bool(self.get_parameter('invert_dpad_lr').value) else 1.0

        # ── v1s: D-pad 가 buttons 로 오는 패드용 대체 매핑 ────────
        # axis_dpad_* 를 -1 로 두고 아래를 지정하면 버튼 방식으로 동작한다.
        self.declare_parameter('button_freq_up', -1)
        self.declare_parameter('button_freq_down', -1)
        self.declare_parameter('button_mode_next', -1)
        self.declare_parameter('button_mode_prev', -1)
        self.button_freq_up = int(self.get_parameter('button_freq_up').value)
        self.button_freq_down = int(self.get_parameter('button_freq_down').value)
        self.button_mode_next = int(self.get_parameter('button_mode_next').value)
        self.button_mode_prev = int(self.get_parameter('button_mode_prev').value)

        # 주파수 기본값 복귀 버튼 (Y=4). X 는 3 이므로 혼동 주의.
        self.declare_parameter('button_freq_reset', 4)
        self.button_freq_reset = int(self.get_parameter('button_freq_reset').value)

        # 긴급 정지 버튼 (B). 누르는 즉시 gait_mode='stop' 으로 전환하고
        # 속도 명령을 0 으로 만든다. 데드맨을 껐으므로 이 버튼이 유일한
        # 즉시 정지 수단이다. 위치를 반드시 손에 익혀 둘 것.
        self.declare_parameter('button_estop', 1)
        self.button_estop = int(self.get_parameter('button_estop').value)

        # ── 데드존 ────────────────────────────────────────────────
        # ★ 2026-08-18 개편: 전 축 0.20 통일 ★
        #   최소속도(vx_pos_min 등)를 0 으로 내렸기 때문에 데드존의 역할이
        #   바뀌었다. 예전에는 "최소속도 계단이 발동하는 문턱"이었고, 지금은
        #   "스틱 미세 떨림이 그대로 작은 속도로 새어나가는 것을 막는 문턱"이다.
        #   min=0 이면 문턱 바로 위의 출력이 0 근처라 문턱을 넉넉히 잡아도
        #   조작감 손해가 없다. 그래서 0.05 → 0.20 으로 키웠다.
        #
        #   [joy_node 데드존과 혼동하지 말 것]
        #   joy_node 에도 deadzone(0.05) 이 있지만 성격이 다르다. 그쪽은
        #   스틱 물리 중립 오차를 없애는 용도이고, 잘라낸 뒤 다시 ±1.0 으로
        #   재정규화하므로(joy.cpp: scale_ = -1/(1-deadzone)/32767) /joy 의
        #   범위는 줄지 않는다. 실제 속도 매핑 문턱은 이 노드의 값 하나뿐이다.
        #   따라서 launch 에서 joy_deadzone 을 이 파라미터에 넘기면 안 된다.
        self.declare_parameter('deadzone', 0.20)
        # 축별 개별 데드존. -1.0 이면 위의 전역값을 상속한다.
        # vy 는 예전에 0.35 로 크게 잡아 "전진 중 대각선 새어듦"을 막았으나,
        # vy_min=0 이 되면서 새어나가는 크기 자체가 0 근처로 작아졌으므로
        # 전역값(0.20) 상속으로 통일했다. 실기에서 여전히 대각선으로 흐르면
        # 이 값만 0.30~0.35 로 되돌리면 된다.
        self.declare_parameter('deadzone_vx', -1.0)
        self.declare_parameter('deadzone_vy', -1.0)
        self.declare_parameter('deadzone_wz', -1.0)
        # D-pad 는 -1/0/+1 이산값이라 임계값만 넘으면 눌린 것으로 본다.
        self.declare_parameter('dpad_threshold', 0.5)

        dz_global = float(self.get_parameter('deadzone').value)
        self.dz_global = dz_global   # 기동 로그 출력용

        def _resolve_dz(name: str) -> float:
            v = float(self.get_parameter(name).value)
            return dz_global if v < 0.0 else v

        self.dz_vx = _resolve_dz('deadzone_vx')
        self.dz_vy = _resolve_dz('deadzone_vy')
        self.dz_wz = _resolve_dz('deadzone_wz')
        self.dpad_th = float(self.get_parameter('dpad_threshold').value)

        # ── 축별 속도 범위 ────────────────────────────────────────
        # ★ 2026-08-18 개편: vx/vy 의 최소속도(*_min)를 0 으로 내렸다 ★
        #   _map_axis() 는 데드존을 넘는 순간 out_min 부터 시작하므로,
        #   out_min > 0 이면 스틱을 살짝만 밀어도 출력이 0 → out_min 으로
        #   수직 점프한다. 놓을 때도 out_min → 0 으로 그대로 낙하한다.
        #   이 계단이 정책 관측(velocity_commands)에 그대로 들어가 스틱을
        #   놓는 순간 휘청이는 원인이었다.
        #   현재 정책은 저속에서도 잘 걷는 것이 확인되어 바닥을 없앴다.
        #   부수 효과로 스틱 변위 0~100% 전 구간이 0~max 에 대응하게 되어
        #   미세 조작 해상도도 좋아진다.
        self.declare_parameter('vx_pos_min', 0.00)
        self.declare_parameter('vx_pos_max', 0.50)
        self.declare_parameter('vx_neg_min', 0.00)
        self.declare_parameter('vx_neg_max', 0.30)
        self.declare_parameter('vy_min', 0.00)
        self.declare_parameter('vy_max', 0.30)
        # ⚠ wz 는 사용자 요청으로 기존 0.50~1.50 을 유지한다.
        #   즉 회전축에는 여전히 0.50 rad/s 짜리 계단이 남아 있다.
        #   (데드존 0.20 을 넘는 순간 wz 가 0 → 0.50 으로 점프)
        #   릴리스 휘청임이 vx 를 0 으로 내린 뒤에도 남는다면, 원인은
        #   거의 확실히 이 wz 계단이다. 그때 wz_min 을 0 으로 내릴 것.
        self.declare_parameter('wz_min', 0.50)
        self.declare_parameter('wz_max', 1.50)

        self.vx_pos_min = float(self.get_parameter('vx_pos_min').value)
        self.vx_pos_max = float(self.get_parameter('vx_pos_max').value)
        self.vx_neg_min = float(self.get_parameter('vx_neg_min').value)
        self.vx_neg_max = float(self.get_parameter('vx_neg_max').value)
        self.vy_min = float(self.get_parameter('vy_min').value)
        self.vy_max = float(self.get_parameter('vy_max').value)
        self.wz_min = float(self.get_parameter('wz_min').value)
        self.wz_max = float(self.get_parameter('wz_max').value)

        # ── 회전 부스트 (LT/RT 홀드) ─────────────────────────────
        # 트리거를 당기고 있는 동안만 wz 상한을 wz_max_boost 로 올린다.
        # 놓으면 즉시 wz_max 로 돌아온다(홀드 방식, 토글 아님).
        # 홀드로 만든 이유: 손을 떼면 무조건 안전한 쪽으로 복귀해야 한다.
        # 토글이면 "지금 몇 단인지" 를 사람이 기억해야 하고, 그 기억이
        # 틀리는 순간이 사고다.
        #
        # 0.0 이하이면 기능 자체를 끈다(= 항상 wz_max).
        self.declare_parameter('wz_max_boost', 0.0)

        # ── 트리거 축 인덱스 ──
        # ★ 반드시 `ros2 topic echo /joy` 로 실측할 것 ★
        # 8BitDo Ultimate 2 는 연결 방식(2.4G/BT)과 드라이버(xpad /
        # hid-generic)에 따라 트리거가 오는 축이 달라진다. 기본값 4/5 는
        # hid-generic 계열에서 흔한 값(ABS_Z / ABS_RZ)이지 보장값이 아니다.
        # -1 이면 그 트리거는 사용하지 않는다.
        self.declare_parameter('axis_wz_boost_l', -1)   # LT
        self.declare_parameter('axis_wz_boost_r', -1)   # RT

        # ── 트리거 '미조작' 상태의 축 값 ──
        # +1.0 : 놓으면 +1, 끝까지 당기면 -1 (xpad / joy_node 일반 규약)
        # -1.0 : 놓으면 -1, 끝까지 당기면 +1 (반대로 오는 패드)
        # 이 값 하나로 두 규약을 모두 흡수한다(_boost_press 참조).
        # echo /joy 에서 트리거를 놓았을 때 보이는 값을 그대로 넣으면 된다.
        self.declare_parameter('wz_boost_rest', 1.0)

        # ── 임계값 (0~1 로 정규화된 '당긴 정도') ──
        # on > off 로 히스테리시스를 준다. 같은 값이면 임계 근처에서
        # 부스트가 깜빡이며 wz 목표가 1.5 ↔ 3.0 을 오가고, 그 진동이
        # 그대로 velocity_commands 관측에 들어간다.
        self.declare_parameter('wz_boost_on', 0.60)
        self.declare_parameter('wz_boost_off', 0.40)

        # ── 무장(arming) 임계 ──
        # press 가 이 값 이하인 '진짜 릴리스' 를 1 회 이상 본 축만
        # 부스트를 허용한다. 위 [수정 이력] 의 설명 참조.
        self.declare_parameter('wz_boost_arm_below', 0.15)

        self.wz_max_boost = float(self.get_parameter('wz_max_boost').value)
        self.axis_wz_boost_l = int(self.get_parameter('axis_wz_boost_l').value)
        self.axis_wz_boost_r = int(self.get_parameter('axis_wz_boost_r').value)
        self.wz_boost_rest = float(self.get_parameter('wz_boost_rest').value)
        self.wz_boost_on = float(self.get_parameter('wz_boost_on').value)
        self.wz_boost_off = float(self.get_parameter('wz_boost_off').value)
        self.wz_boost_arm_below = float(self.get_parameter('wz_boost_arm_below').value)

        # rest 는 부호만 의미가 있다. 0 근처면 press 계산이 발산하므로
        # 안전한 기본값(+1.0)으로 되돌린다.
        if abs(self.wz_boost_rest) < 0.5:
            self.get_logger().warn(
                f'wz_boost_rest={self.wz_boost_rest:.2f} 는 유효하지 않다 '
                '(+1.0 또는 -1.0 이어야 함) → +1.0 으로 대체')
            self.wz_boost_rest = 1.0
        self.wz_boost_rest = math.copysign(1.0, self.wz_boost_rest)

        # on <= off 면 히스테리시스가 성립하지 않는다. 강제로 벌려 준다.
        if self.wz_boost_off >= self.wz_boost_on:
            new_off = max(0.0, self.wz_boost_on - 0.20)
            self.get_logger().warn(
                f'wz_boost_off({self.wz_boost_off:.2f}) >= '
                f'wz_boost_on({self.wz_boost_on:.2f}) — 히스테리시스가 '
                f'성립하지 않는다 → off={new_off:.2f} 로 보정')
            self.wz_boost_off = new_off

        # 부스트 상한이 기본 상한보다 낮으면 '부스트' 가 아니다. 오타일
        # 가능성이 높으므로 경고만 남기고 값은 존중한다(감속용으로 쓸
        # 수도 있으니 막지는 않는다).
        if 0.0 < self.wz_max_boost < self.wz_max:
            self.get_logger().warn(
                f'wz_max_boost({self.wz_max_boost:.2f}) < '
                f'wz_max({self.wz_max:.2f}) — 트리거를 당기면 오히려 '
                '회전이 느려진다. 의도한 것인지 확인하라.')

        # 부스트를 켰는데 축을 하나도 안 잡아 두면 영원히 동작하지 않는다.
        self.wz_boost_enabled = (
            self.wz_max_boost > 0.0
            and (self.axis_wz_boost_l >= 0 or self.axis_wz_boost_r >= 0))
        if self.wz_max_boost > 0.0 and not self.wz_boost_enabled:
            self.get_logger().warn(
                'wz_max_boost 가 설정됐지만 axis_wz_boost_l/r 이 둘 다 -1 '
                '이라 부스트가 동작하지 않는다.')

        # ── v1s: 보행 주파수 파라미터 ────────────────────────────
        # 기본 범위는 학습 분포(1.8~2.2 Hz)에 맞춘다.
        # 이 밖으로 나가면 정책이 본 적 없는 조건이 된다.
        self.declare_parameter('gait_freq_default', 2.0)
        self.declare_parameter('gait_freq_min', 1.8)
        self.declare_parameter('gait_freq_max', 2.2)
        self.declare_parameter('gait_freq_step', 0.1)
        # D-pad 를 누르고 있을 때 연속 증감 간격 [s]. 0 이면 1 회만.
        self.declare_parameter('gait_freq_repeat_s', 0.25)

        self.gait_freq_default = float(self.get_parameter('gait_freq_default').value)
        self.gait_freq_min = float(self.get_parameter('gait_freq_min').value)
        self.gait_freq_max = float(self.get_parameter('gait_freq_max').value)
        self.gait_freq_step = float(self.get_parameter('gait_freq_step').value)
        self.gait_freq_repeat_s = float(self.get_parameter('gait_freq_repeat_s').value)

        self.declare_parameter('gait_mode_default', 'auto')
        self.declare_parameter('enable_walk_mode', False)
        # ── D-pad 좌우로 순환할 모드 목록 ──
        # 비워두면 enable_walk_mode 에 따른 기존 기본값을 쓴다(하위호환).
        # 목록에서 모드를 빼면 D-pad 로 그 모드에 진입할 수 없다.
        #
        # 예) stop 을 빼고 auto/march 만 쓰기:
        #       gait_modes: ["auto", "march"]
        #     auto 에서 스틱을 놓으면 정책이 알아서 발을 모으고 멈추므로
        #     stop 모드는 사실상 중복이다. 순환 단계가 줄어 조작이 단순해진다.
        #
        # ★ 타입 명시가 필수인 이유 (2026-08-17 버그 수정) ★
        #   기존 코드는 `declare_parameter('gait_modes', [])` 였다. rclpy 는
        #   기본값에서 파라미터 타입을 추론하는데, 빈 파이썬 리스트는 원소가
        #   없어 문자열 배열인지 알 방법이 없으므로 BYTE_ARRAY 로 추론된다.
        #   그 상태에서 control_joypad.launch.py 가 ["auto", "march"]
        #   (STRING_ARRAY) 를 오버라이드로 넘기면 선언 타입과 불일치하여
        #       InvalidParameterTypeException: Trying to set parameter
        #       'gait_modes' to '['auto', 'march']' of type 'STRING_ARRAY',
        #       expecting type 'BYTE_ARRAY'
        #   가 발생하고 노드가 __init__ 단계에서 죽는다. 이때 러너
        #   (actor_onnx_walking_runner) 는 정상 기동하므로 "정책은 도는데
        #   조이패드만 안 먹는" 증상으로 보인다.
        #
        #   해결: 기본값 대신 Parameter.Type 을 넘겨 타입만 선언하고,
        #   descriptor 로 STRING_ARRAY 를 못박는다. 이러면 "기본값 없음"
        #   상태가 되어 오버라이드가 없을 때 .value 가 None 을 반환하므로,
        #   아래 get_parameter_or() + `or []` 로 빈 목록 폴백을 유지한다.
        #   (기존의 "비워두면 enable_walk_mode 기본값 사용" 동작은 그대로다)
        self.declare_parameter(
            'gait_modes',
            rclpy.Parameter.Type.STRING_ARRAY,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=('D-pad 좌우로 순환할 gait 모드 목록. '
                             '비우면 enable_walk_mode 기반 기본값 사용.'),
            ),
        )
        # 오버라이드가 없으면 파라미터가 NOT_SET 이라 get_parameter() 가
        # 예외를 던질 수 있다. get_parameter_or() 로 안전하게 읽고,
        # NOT_SET 일 때의 .value(None) 도 `or []` 로 흡수한다.
        _modes_raw = self.get_parameter_or(
            'gait_modes', rclpy.Parameter('gait_modes')).value
        _modes_param = [str(m).strip().lower()
                        for m in (_modes_raw or [])
                        if str(m).strip()]
        mode_default = str(self.get_parameter('gait_mode_default').value).lower()
        if _modes_param:
            self.gait_modes = _modes_param
            # 진단용: 목록이 파라미터에서 왔는지 기본값 폴백인지 로그에 남긴다.
            # 타입 오류로 오버라이드가 무시되는 상황을 조기에 알아채기 위함.
            self._gait_modes_src = 'param'
        else:
            self.gait_modes = (GAIT_MODES_WITH_WALK
                               if bool(self.get_parameter('enable_walk_mode').value)
                               else GAIT_MODES_DEFAULT)
            self._gait_modes_src = 'default'
        if mode_default not in self.gait_modes:
            self.get_logger().warn(
                f"gait_mode_default={mode_default} 가 순환 목록에 없음 "
                f"→ '{self.gait_modes[0]}' 사용")
            mode_default = self.gait_modes[0]

        # ── 발행 주기 ─────────────────────────────────────────────
        self.declare_parameter('publish_rate_hz', 20.0)
        pub_hz = float(self.get_parameter('publish_rate_hz').value)

        # ── 슬루 레이트 리미터 (명령 스무딩) ─────────────────────
        # 스틱을 100% 밀다 놓으면 목표 속도가 max → 0 으로 한 번에 떨어진다.
        # 이 계단이 velocity_commands 관측에 그대로 들어가 릴리스 순간
        # 로봇이 휘청인다. 여기서 '목표'와 '실제 발행값'을 분리하고,
        # 발행값이 목표를 향해 정해진 가감속 한계 안에서만 움직이게 한다.
        #
        # [가속과 감속을 따로 두는 이유]
        #   가속은 빠를수록 조작감이 좋고(입력 지연처럼 느껴짐), 감속은
        #   느릴수록 부드럽다. 두 값을 하나로 묶으면 둘 중 하나를 포기해야
        #   한다. 기본값은 가속을 감속보다 크게 잡았다.
        #
        # [학습 분포와의 관계]
        #   램프 도중의 중간 속도값은 모두 학습 시 샘플되던 범위 안이므로
        #   분포 밖 입력이 아니다. 다만 관측 히스토리(5프레임=0.1s)에
        #   기울기가 생기는데, 학습 때도 command resample 순간에 유사한
        #   변화를 겪으므로 문제가 되지 않는다. 그래도 감속을 지나치게
        #   느리게(1.0 미만) 잡으면 정지 지시가 늦어져 위험하다.
        #
        # [단위] accel/decel_vx, vy = m/s^2 · wz = rad/s^2
        #        0 이하로 두면 그 축은 제한 없음(즉시 반영)
        self.declare_parameter('slew_enable', True)
        self.declare_parameter('accel_vx', 3.0)   # 0 → 0.5 m/s 까지 약 0.17s
        self.declare_parameter('decel_vx', 2.0)   # 0.5 → 0 m/s 까지 약 0.25s
        self.declare_parameter('accel_vy', 2.0)
        self.declare_parameter('decel_vy', 1.5)
        self.declare_parameter('accel_wz', 6.0)   # 0 → 1.5 rad/s 까지 약 0.25s
        self.declare_parameter('decel_wz', 4.0)   # 1.5 → 0 rad/s 까지 약 0.38s

        self.slew_enable = bool(self.get_parameter('slew_enable').value)
        self.accel_vx = float(self.get_parameter('accel_vx').value)
        self.decel_vx = float(self.get_parameter('decel_vx').value)
        self.accel_vy = float(self.get_parameter('accel_vy').value)
        self.decel_vy = float(self.get_parameter('decel_vy').value)
        self.accel_wz = float(self.get_parameter('accel_wz').value)
        self.decel_wz = float(self.get_parameter('decel_wz').value)

        # ── 내부 상태 ─────────────────────────────────────────────
        # vx/vy/wz 는 '목표' 속도(스틱이 지시한 값)다.
        # 실제로 발행되는 값은 _out_* 이며, publish_cmd() 에서 슬루 제한을
        # 거쳐 목표를 향해 조금씩 이동한다.
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._out_vx = 0.0
        self._out_vy = 0.0
        self._out_wz = 0.0
        # 슬루를 건너뛰고 즉시 0 을 내보내야 하는 상황(E-STOP, 데드맨 해제)
        # 을 표시한다. publish_cmd() 가 소비한 뒤 스스로 내린다.
        self._slew_bypass = False
        # 슬루 계산에 쓸 직전 발행 시각. 타이머 지연이 있어도 실제 경과
        # 시간으로 계산해야 가감속이 설정값과 일치한다.
        self._last_pub_t = None
        self._prev_kick_pressed = False
        self._warned_axes = False
        self._warned_btns = False
        self._enabled = True

        # ── 회전 부스트 상태 ──
        # _wz_boost      : 현재 부스트 ON 여부 (LT/RT 의 OR)
        # _boost_active  : 축별 ON/OFF. 히스테리시스 판정에 직전 상태가
        #                  필요하므로 축마다 따로 들고 있어야 한다.
        # _boost_armed   : 축별 무장 여부. press <= arm_below 를 한 번
        #                  본 뒤에야 True 가 되고, 이후 내려가지 않는다.
        # 키는 축 인덱스가 아니라 'L'/'R' 로 둔다. 두 트리거가 같은
        # 인덱스로 잘못 설정돼도 상태가 뒤섞이지 않는다.
        self._wz_boost = False
        self._boost_active = {'L': False, 'R': False}
        self._boost_armed = {'L': False, 'R': False}
        self._warned_boost_axes = False
        self._warned_boost_unarmed = False

        # v1s 상태
        self.gait_freq = self.gait_freq_default
        self.gait_mode = mode_default
        self._prev_dpad_ud = 0     # -1 / 0 / +1
        self._prev_dpad_lr = 0
        self._prev_freq_reset = False
        self._prev_estop = False
        self._last_freq_repeat_t = 0.0
        self._joy_logged = False       # 첫 /joy 수신 진단 로그 1 회용

        # ── 퍼블리셔 / 구독 ───────────────────────────────────────
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.kick_pub = self.create_publisher(Bool, kick_topic, 10)

        # 러너가 나중에 떠도 마지막 값을 받도록 TRANSIENT_LOCAL.
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.freq_pub = self.create_publisher(Float32, freq_topic, latched_qos)
        self.mode_pub = self.create_publisher(String, mode_topic, latched_qos)

        # ── 부스트 디버그 발행 ────────────────────────────────────
        # Foxglove/RViz(Plot 패널)에서 cmd_vel 의 angular.z 와 같은
        # 시간축에 겹쳐 볼 수 있어야 한다. "회전이 이상했는데 그때
        # 부스트가 걸려 있었나?" 를 로그 뒤지지 않고 바로 확인하려는
        # 목적이므로, 상태가 바뀔 때만이 아니라 발행 주기마다 낸다.
        # (이산 신호는 변화 시점만 발행하면 그래프가 끊겨 보인다)
        # 이름 앞의 '~/' 는 노드 프라이빗 네임스페이스다.
        #   → /joypad_teleop_node/wz_boost_active
        #   → /joypad_teleop_node/wz_max_current
        self.boost_pub = self.create_publisher(Bool, '~/wz_boost_active', 10)
        self.wzmax_pub = self.create_publisher(Float32, '~/wz_max_current', 10)

        self.create_subscription(Joy, joy_topic, self.cb_joy, 10)
        self.timer = self.create_timer(1.0 / pub_hz, self.publish_cmd)

        # 초기값 1 회 발행 (latched 이므로 이후 구독자도 받는다)
        self._publish_gait_freq()
        self._publish_gait_mode()

        self.get_logger().info(
            'joypad_teleop_v1s 시작. '
            f'axes(vx={self.axis_vx}, vy={self.axis_vy}, wz={self.axis_wz}), '
            f'kick=btn{self.button_kick}, enable_button={self.enable_button}')
        self.get_logger().info(
            f'gait: freq={self.gait_freq:.2f}Hz '
            f'[{self.gait_freq_min:.2f}~{self.gait_freq_max:.2f}, '
            f'step={self.gait_freq_step:.2f}] mode={self.gait_mode} '
            f'modes={self.gait_modes} (출처={self._gait_modes_src})')
        self.get_logger().info(
            f'dpad: ud={self.axis_dpad_ud}(주파수) lr={self.axis_dpad_lr}(모드), '
            f'freq_reset=btn{self.button_freq_reset}, '
            f'E-STOP=btn{self.button_estop}')
        # 데드존과 축별 속도 범위를 한 줄로 확인할 수 있게 찍는다.
        # min 이 0 이 아니면 그 값이 곧 "데드존 경계에서 튀는 계단 크기"다.
        self.get_logger().info(
            f'deadzone: vx={self.dz_vx:.2f} vy={self.dz_vy:.2f} '
            f'wz={self.dz_wz:.2f} (전역={self.dz_global:.2f})')
        self.get_logger().info(
            f'range: vx+[{self.vx_pos_min:.2f}~{self.vx_pos_max:.2f}] '
            f'vx-[{self.vx_neg_min:.2f}~{self.vx_neg_max:.2f}] '
            f'vy[{self.vy_min:.2f}~{self.vy_max:.2f}] '
            f'wz[{self.wz_min:.2f}~{self.wz_max:.2f}]')
        if self.wz_boost_enabled:
            self.get_logger().info(
                f'wz boost: 트리거 홀드 시 wz_max {self.wz_max:.2f} → '
                f'{self.wz_max_boost:.2f} rad/s '
                f'(axes L={self.axis_wz_boost_l} R={self.axis_wz_boost_r}, '
                f'rest={self.wz_boost_rest:+.0f}, '
                f'on={self.wz_boost_on:.2f}/off={self.wz_boost_off:.2f})')
            # 무장 절차를 반드시 알려야 한다. 이걸 모르면 "트리거를
            # 당겨도 아무 일이 없다" 는 증상만 남는다.
            self.get_logger().warn(
                'wz boost 무장 필요 — 기동 후 각 트리거를 한 번씩 끝까지 '
                '당겼다 놓아야 활성화된다(미조작 시 0.0 을 보고하는 '
                '드라이버 대응). 무장 전에는 부스트가 걸리지 않는다.')
        else:
            self.get_logger().info('wz boost: 비활성 (wz_max 고정)')
        # min > 0 인 축은 스틱 릴리스 시 그 크기만큼 명령이 수직 낙하한다.
        # 정책 관측(velocity_commands)에 계단으로 들어가므로 경고로 남긴다.
        _steps = [(n, v) for n, v in (
            ('vx+', self.vx_pos_min), ('vx-', self.vx_neg_min),
            ('vy', self.vy_min), ('wz', self.wz_min)) if v > 1e-6]
        if _steps:
            self.get_logger().warn(
                '릴리스 시 명령 계단이 남는 축: '
                + ', '.join(f'{n}={v:.2f}' for n, v in _steps)
                + ' — 휘청임이 있으면 해당 *_min 을 0 으로 내릴 것.')
        if self.slew_enable:
            self.get_logger().info(
                f'slew: vx(a={self.accel_vx:.1f}/d={self.decel_vx:.1f}) '
                f'vy(a={self.accel_vy:.1f}/d={self.decel_vy:.1f}) '
                f'wz(a={self.accel_wz:.1f}/d={self.decel_wz:.1f}) '
                '— E-STOP/데드맨 해제는 즉시 0(우회)')
        else:
            self.get_logger().warn('slew_enable=False — 명령 스무딩 비활성')
        self.get_logger().info(
            '축/버튼이 반대로 움직이면 `ros2 topic echo /joy` 로 확인 후 '
            'axis_*/invert_*/button_* 파라미터를 조정하라.')

    # ── 아날로그 한 축을 속도로 매핑 ─────────────────────────────
    def _map_axis(self, raw: float, deadzone: float,
                  out_min: float, out_max: float) -> float:
        a = abs(raw)
        if a < deadzone:
            return 0.0
        span = 1.0 - deadzone
        if span <= 0.0:
            return 0.0
        t = (a - deadzone) / span
        if t > 1.0:
            t = 1.0
        mag = out_min + t * (out_max - out_min)
        return math.copysign(mag, raw)

    # ── 회전 부스트 (LT/RT 홀드) ─────────────────────────────────
    def _boost_press(self, raw: float) -> float:
        """트리거 축의 raw 값을 '당긴 정도' 0.0~1.0 으로 변환한다.

        rest(미조작)와 full(끝까지 당김)이 어느 부호인지가 드라이버마다
        다르므로, wz_boost_rest 하나로 두 규약을 모두 흡수한다.
        full = -rest 라고 보면

            press = (rest - raw) / (rest - full) = (rest - raw) / (2*rest)

        · rest=+1 : raw=+1 → 0.0 , raw=-1 → 1.0
        · rest=-1 : raw=-1 → 0.0 , raw=+1 → 1.0

        범위를 벗어난 값(캘리브레이션 오차)은 0~1 로 잘라낸다.
        """
        press = (self.wz_boost_rest - raw) / (2.0 * self.wz_boost_rest)
        if press < 0.0:
            return 0.0
        if press > 1.0:
            return 1.0
        return press

    def _update_one_boost(self, axes, key: str, axis_idx: int) -> bool:
        """트리거 한 개의 부스트 상태를 갱신하고 결과를 반환한다.

        key      : 'L' / 'R' (상태 딕셔너리 키)
        axis_idx : 축 인덱스. -1 이거나 범위 밖이면 항상 False.
        """
        if axis_idx < 0:
            return False
        if axis_idx >= len(axes):
            # 인덱스가 범위 밖이면 부스트를 걸 수 없다. 조용히 꺼두되
            # 원인을 알 수 있게 한 번은 경고한다.
            if not self._warned_boost_axes:
                self._warned_boost_axes = True
                self.get_logger().warn(
                    f'[boost] axis_wz_boost_{key.lower()}={axis_idx} 가 '
                    f'axes 길이({len(axes)}) 범위를 벗어남 — 부스트 비활성. '
                    '`ros2 topic echo /joy` 로 트리거 축을 확인하라.')
            self._boost_active[key] = False
            return False

        press = self._boost_press(float(axes[axis_idx]))

        # ── 무장 판정 ──
        # '진짜 릴리스' 를 한 번 본 뒤에야 부스트를 허용한다.
        # (미조작 시 0.0 을 보고하는 드라이버 대응 — 파일 상단 이력 참조)
        if not self._boost_armed[key]:
            if press <= self.wz_boost_arm_below:
                self._boost_armed[key] = True
                self.get_logger().info(
                    f'[boost] {key} 트리거 무장 완료 (press={press:.2f})')
            else:
                if not self._warned_boost_unarmed:
                    self._warned_boost_unarmed = True
                    self.get_logger().warn(
                        f'[boost] {key} 트리거 미무장 (press={press:.2f}). '
                        '트리거를 끝까지 당겼다 놓으면 무장된다. '
                        '계속 이 상태면 axis 인덱스나 wz_boost_rest 부호가 '
                        '틀린 것이다.')
                self._boost_active[key] = False
                return False

        # ── 히스테리시스 판정 ──
        # ON 중이면 off 임계 아래로 내려가야 꺼지고,
        # OFF 중이면 on 임계 위로 올라가야 켜진다.
        if self._boost_active[key]:
            if press < self.wz_boost_off:
                self._boost_active[key] = False
        else:
            if press > self.wz_boost_on:
                self._boost_active[key] = True
        return self._boost_active[key]

    def _update_wz_boost(self, axes) -> None:
        """LT/RT 를 모두 읽어 self._wz_boost 를 갱신한다.

        둘 중 하나라도 눌려 있으면 부스트 ON (OR 조건). 어느 쪽 손으로도
        걸 수 있게 하려는 의도다.

        데드맨이 꺼져 있으면 부스트도 해제한다. 이때 cmd 는 어차피 0 이라
        직접적인 영향은 없지만, 데드맨을 다시 잡는 순간 3.0 상한이 이미
        걸려 있는 상태가 되는 것을 막는다.
        """
        if not self.wz_boost_enabled or not self._enabled:
            self._wz_boost = False
            return

        left = self._update_one_boost(axes, 'L', self.axis_wz_boost_l)
        right = self._update_one_boost(axes, 'R', self.axis_wz_boost_r)
        new_state = bool(left or right)

        if new_state != self._wz_boost:
            # 상태 변화는 반드시 로그로 남긴다. 실기에서 "왜 갑자기
            # 빨리 돌았지" 를 사후에 추적할 수 있어야 한다.
            if new_state:
                self.get_logger().info(
                    f'[boost] ON — wz_max {self.wz_max:.2f} → '
                    f'{self.wz_max_boost:.2f} rad/s '
                    f'(L={left}, R={right})')
            else:
                self.get_logger().info(
                    f'[boost] OFF — wz_max {self.wz_max:.2f} rad/s 복귀')
        self._wz_boost = new_state

    def _wz_max_now(self) -> float:
        """현재 적용할 wz 상한. 부스트 여부에 따라 갈린다."""
        return self.wz_max_boost if self._wz_boost else self.wz_max

    # ── v1s: 주파수 / 모드 발행 ──────────────────────────────────
    def _publish_gait_freq(self):
        m = Float32()
        m.data = float(self.gait_freq)
        self.freq_pub.publish(m)

    def _publish_gait_mode(self):
        m = String()
        m.data = str(self.gait_mode)
        self.mode_pub.publish(m)

    def _adjust_freq(self, direction: int):
        """주파수를 step 만큼 증감하고 범위로 클램프한다.

        direction: +1(증가) / -1(감소)
        """
        new_f = self.gait_freq + direction * self.gait_freq_step
        new_f = max(self.gait_freq_min, min(self.gait_freq_max, new_f))
        if abs(new_f - self.gait_freq) < 1e-9:
            # 이미 한계값. 사용자에게 한 번은 알려준다.
            self.get_logger().info(
                f'[gait_freq] 한계 도달: {self.gait_freq:.2f}Hz '
                f'(범위 {self.gait_freq_min:.2f}~{self.gait_freq_max:.2f})')
            return
        self.gait_freq = new_f
        self._publish_gait_freq()
        self.get_logger().info(f'[gait_freq] {self.gait_freq:.2f} Hz')

    def _cycle_mode(self, direction: int):
        """보행 모드를 순환한다. direction: +1(다음) / -1(이전)"""
        try:
            idx = self.gait_modes.index(self.gait_mode)
        except ValueError:
            idx = 0
        idx = (idx + direction) % len(self.gait_modes)
        self.gait_mode = self.gait_modes[idx]
        self._publish_gait_mode()
        self.get_logger().info(f'[gait_mode] {self.gait_mode}')

    def _read_dpad_state(self, axes, buttons, is_ud: bool) -> int:
        """D-pad 한 축의 현재 상태를 -1 / 0 / +1 로 반환.

        axis 방식(axes[6], axes[7])과 button 방식 둘 다 지원한다.
        axis_dpad_* 가 0 이상이면 axis 방식, 아니면 button 방식.
        """
        axis_idx = self.axis_dpad_ud if is_ud else self.axis_dpad_lr
        inv = self.inv_dpad_ud if is_ud else self.inv_dpad_lr

        if axis_idx >= 0:
            if axis_idx >= len(axes):
                return 0
            v = inv * float(axes[axis_idx])
            if v > self.dpad_th:
                return 1
            if v < -self.dpad_th:
                return -1
            return 0

        # button 방식
        b_pos = self.button_freq_up if is_ud else self.button_mode_next
        b_neg = self.button_freq_down if is_ud else self.button_mode_prev
        pos = (0 <= b_pos < len(buttons)) and (buttons[b_pos] == 1)
        neg = (0 <= b_neg < len(buttons)) and (buttons[b_neg] == 1)
        if pos and not neg:
            return 1
        if neg and not pos:
            return -1
        return 0

    def _handle_gait_inputs(self, axes, buttons, now: float):
        """D-pad / 리셋 버튼을 읽어 주파수·모드를 갱신한다.

        데드맨을 쓰는 경우 활성 상태에서만 반응한다 (오조작 방지).
        """
        # ★ 데드맨과 무관하게 항상 동작한다 ★
        # 데드맨은 "로봇을 움직이는 것"(cmd_vel/킥)만 막아야 한다.
        # 모드 전환까지 막으면 RB 를 놓은 상태에서 stop 으로 갈 수 없어
        # 위상이 계속 돌고 제자리 걷기가 이어진다. 안전상 역효과다.

        # ── 긴급 정지 (B): 즉시 정지 모드 + 속도 0 ──
        # 순환 목록에 'stop' 이 없는 구성(예: auto/march 만 사용)에서는
        # 목록의 첫 모드로 되돌린다. auto 에서 속도 0 이면 정책이 발을 모으고
        # 멈추므로 stop 모드가 없어도 긴급 정지 기능은 그대로 성립한다.
        estop_pressed = False
        if 0 <= self.button_estop < len(buttons):
            estop_pressed = (buttons[self.button_estop] == 1)
        if estop_pressed and not self._prev_estop:
            safe_mode = 'stop' if 'stop' in self.gait_modes else self.gait_modes[0]
            if self.gait_mode != safe_mode:
                self.gait_mode = safe_mode
                self._publish_gait_mode()
                self.get_logger().warn(f'[E-STOP] gait_mode -> {safe_mode}')
            # 속도도 즉시 0 으로. publish_cmd 가 다음 주기에 반영한다.
            # ★ 슬루 우회 필수 ★ 긴급 정지가 램프를 타고 서서히 줄어들면
            #   E-STOP 이 아니다. bypass 를 세워 다음 주기에 0 을 직접 낸다.
            self.vx = self.vy = self.wz = 0.0
            self._slew_bypass = True
        self._prev_estop = estop_pressed

        # ── 주파수: D-pad 상하 ──
        ud = self._read_dpad_state(axes, buttons, True)
        if ud != 0:
            if self._prev_dpad_ud == 0:
                # rising edge → 즉시 1 회 반영
                self._adjust_freq(ud)
                self._last_freq_repeat_t = now
            elif self.gait_freq_repeat_s > 0.0:
                # 누르고 있는 동안 일정 간격으로 반복
                if (now - self._last_freq_repeat_t) >= self.gait_freq_repeat_s:
                    self._adjust_freq(ud)
                    self._last_freq_repeat_t = now
        self._prev_dpad_ud = ud

        # ── 모드: D-pad 좌우 (rising edge 만, 반복 없음) ──
        lr = self._read_dpad_state(axes, buttons, False)
        if lr != 0 and self._prev_dpad_lr == 0:
            # 오른쪽(+) = 다음 모드, 왼쪽(-) = 이전 모드
            self._cycle_mode(lr)
        self._prev_dpad_lr = lr

        # ── 주파수 리셋 버튼 ──
        reset_pressed = False
        if 0 <= self.button_freq_reset < len(buttons):
            reset_pressed = (buttons[self.button_freq_reset] == 1)
        if reset_pressed and not self._prev_freq_reset:
            self.gait_freq = self.gait_freq_default
            self._publish_gait_freq()
            self.get_logger().info(f'[gait_freq] 리셋 → {self.gait_freq:.2f} Hz')
        self._prev_freq_reset = reset_pressed

    def _log_joy_diagnostics(self, axes, buttons):
        """첫 /joy 수신 때 배열 길이와 D-pad 감지 방식을 한 번 찍는다.

        D-pad 가 반응하지 않을 때 원인이 인덱스 초과인지 부호 반전인지
        구분하기 위한 진단이다. 이게 없으면 '눌러도 아무 일이 없다' 는
        증상만 남아 원인을 좁힐 수 없다.
        """
        self.get_logger().info(
            f'[joy] 첫 수신: axes={len(axes)}개, buttons={len(buttons)}개')

        need_axis = max(self.axis_vx, self.axis_vy, self.axis_wz)
        if need_axis >= len(axes):
            self.get_logger().error(
                f'[joy] 이동 축 인덱스({need_axis})가 axes 범위를 벗어남. '
                'axis_vx/vy/wz 를 확인하라.')

        if self.axis_dpad_ud >= 0 or self.axis_dpad_lr >= 0:
            ok_ud = 0 <= self.axis_dpad_ud < len(axes)
            ok_lr = 0 <= self.axis_dpad_lr < len(axes)
            if ok_ud and ok_lr:
                self.get_logger().info(
                    f'[joy] D-pad = axes 방식 (ud={self.axis_dpad_ud}, '
                    f'lr={self.axis_dpad_lr})')
            else:
                self.get_logger().error(
                    f'[joy] D-pad axes 인덱스가 범위 밖 '
                    f'(ud={self.axis_dpad_ud} ok={ok_ud}, '
                    f'lr={self.axis_dpad_lr} ok={ok_lr}, axes={len(axes)}개). '
                    'D-pad 가 buttons 로 오는 패드일 수 있다. '
                    'axis_dpad_ud/lr 을 -1 로 두고 '
                    'button_freq_up/down, button_mode_next/prev 를 지정하라.')
        else:
            self.get_logger().info(
                f'[joy] D-pad = buttons 방식 '
                f'(freq_up={self.button_freq_up}, freq_down={self.button_freq_down}, '
                f'mode_next={self.button_mode_next}, mode_prev={self.button_mode_prev})')

        for label, idx in [('button_kick', self.button_kick),
                           ('button_estop', self.button_estop),
                           ('button_freq_reset', self.button_freq_reset)]:
            if idx >= 0 and idx >= len(buttons):
                self.get_logger().error(
                    f'[joy] {label}={idx} 가 buttons 범위({len(buttons)})를 벗어남.')

        if self.enable_button < 0:
            self.get_logger().warn(
                '[joy] 데드맨 비활성. 즉시 정지는 '
                f'B 버튼(btn{self.button_estop}) 을 쓸 것.')

        # ── 부스트 축 진단 ──
        # 첫 수신 때 각 트리거의 현재 raw 값과 그것이 press 로 얼마인지를
        # 같이 찍는다. 놓은 상태인데 press 가 0 이 아니면 wz_boost_rest
        # 부호가 틀렸거나 축 인덱스가 틀린 것이라, 이 한 줄로 판별된다.
        if self.wz_boost_enabled:
            for key, idx in (('L', self.axis_wz_boost_l),
                             ('R', self.axis_wz_boost_r)):
                if idx < 0:
                    continue
                if idx >= len(axes):
                    self.get_logger().error(
                        f'[joy] axis_wz_boost_{key.lower()}={idx} 가 axes '
                        f'범위({len(axes)})를 벗어남 — 부스트 비활성.')
                    continue
                raw = float(axes[idx])
                self.get_logger().info(
                    f'[joy] {key} 트리거 axes[{idx}]={raw:+.3f} → '
                    f'press={self._boost_press(raw):.2f} '
                    '(놓은 상태라면 press 가 0 에 가까워야 한다. '
                    '1 에 가까우면 wz_boost_rest 부호를 뒤집어라)')

    def cb_joy(self, msg: Joy):
        axes = msg.axes
        buttons = msg.buttons
        now = self.get_clock().now().nanoseconds * 1e-9

        if not self._joy_logged:
            self._log_joy_diagnostics(axes, buttons)
            self._joy_logged = True

        # ── 데드맨 처리 ──
        if self.enable_button >= 0:
            if self.enable_button < len(buttons):
                self._enabled = (buttons[self.enable_button] == 1)
            else:
                self._enabled = False

        # ── 킥 버튼 rising edge ──
        kick_pressed = False
        if 0 <= self.button_kick < len(buttons):
            kick_pressed = (buttons[self.button_kick] == 1)
        elif not self._warned_btns:
            self.get_logger().warn(
                f'button_kick={self.button_kick} 가 buttons 길이({len(buttons)}) '
                '범위를 벗어남. 파라미터 확인 필요.')
            self._warned_btns = True

        if kick_pressed and not self._prev_kick_pressed:
            if self._enabled:
                self.kick_pub.publish(Bool(data=True))
                self.get_logger().info('[KICK] /kick_trigger 발행')
        self._prev_kick_pressed = kick_pressed

        # ── v1s: 보행 주파수 / 모드 ──
        self._handle_gait_inputs(axes, buttons, now)

        # ── 이동 축 읽기 ──
        need = max(self.axis_vx, self.axis_vy, self.axis_wz)
        if need >= len(axes):
            if not self._warned_axes:
                self.get_logger().warn(
                    f'axes 길이({len(axes)})가 매핑 인덱스보다 짧음. '
                    'axis_* 파라미터/패드 모드를 확인하라.')
                self._warned_axes = True
            # 매핑이 깨진 상태다. 서서히 줄일 이유가 없으니 즉시 0.
            self.vx = self.vy = self.wz = 0.0
            # 부스트 상태도 같이 내린다. 매핑이 복구됐을 때 이전 부스트가
            # 살아 있으면 첫 프레임부터 3.0 상한으로 시작하게 된다.
            self._wz_boost = False
            self._slew_bypass = True
            return

        # ── 회전 부스트 갱신 ──
        # ★ 반드시 _map_axis 호출 '전' 에 갱신해야 한다 ★
        # 아래 self.wz 계산이 _wz_max_now() 를 읽으므로, 순서가 뒤바뀌면
        # 부스트가 한 프레임(=최대 50ms) 늦게 반영된다.
        self._update_wz_boost(axes)

        raw_vx = self.inv_vx * float(axes[self.axis_vx])
        raw_vy = self.inv_vy * float(axes[self.axis_vy])
        raw_wz = self.inv_wz * float(axes[self.axis_wz])

        if raw_vx >= 0.0:
            self.vx = self._map_axis(raw_vx, self.dz_vx,
                                     self.vx_pos_min, self.vx_pos_max)
        else:
            self.vx = self._map_axis(raw_vx, self.dz_vx,
                                     self.vx_neg_min, self.vx_neg_max)

        self.vy = self._map_axis(raw_vy, self.dz_vy, self.vy_min, self.vy_max)
        # wz 상한만 부스트에 따라 갈린다. 하한(wz_min)은 그대로 둔다 —
        # 부스트로 하한까지 올리면 데드존 경계의 계단이 같이 커진다.
        self.wz = self._map_axis(raw_wz, self.dz_wz,
                                 self.wz_min, self._wz_max_now())

    # ── 슬루 레이트 제한 ─────────────────────────────────────────
    @staticmethod
    def _slew(cur: float, target: float,
              accel: float, decel: float, dt: float) -> float:
        """cur 를 target 쪽으로 dt 동안 허용된 만큼만 이동시킨다.

        가속(0 에서 멀어짐)과 감속(0 에 가까워짐)에 서로 다른 한계를 쓴다.
        부호가 뒤집히는 경우(전진 → 후진)는 일단 감속으로 취급해 0 을
        거쳐 가게 한다. 0 을 지난 뒤에는 자동으로 가속 판정으로 넘어간다.

        accel/decel 이 0 이하면 그 방향은 제한하지 않는다(즉시 반영).
        """
        delta = target - cur
        if abs(delta) < 1e-9:
            return target
        # 0 에서 멀어지고 있고 부호도 같으면 가속, 그 외는 전부 감속.
        # (cur == 0.0 은 정지에서 출발하는 경우라 가속)
        if abs(target) > abs(cur) and (cur == 0.0 or target * cur > 0.0):
            rate = accel
        else:
            rate = decel
        if rate <= 0.0:
            return target
        max_step = rate * dt
        if abs(delta) <= max_step:
            return target
        return cur + math.copysign(max_step, delta)

    def publish_cmd(self):
        # ── 목표 속도 결정 ──
        # 데드맨이 꺼져 있으면 목표는 0 이다. 이때는 안전을 위해 슬루를
        # 건너뛴다(손을 뗀 즉시 멈춰야 한다).
        if self._enabled:
            tgt_vx, tgt_vy, tgt_wz = self.vx, self.vy, self.wz
        else:
            tgt_vx = tgt_vy = tgt_wz = 0.0

        bypass = self._slew_bypass or (not self._enabled) or (not self.slew_enable)

        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_pub_t is None:
            dt = 0.0
        else:
            dt = now - self._last_pub_t
        self._last_pub_t = now
        # 타이머가 크게 밀렸을 때 한 번에 튀지 않도록 상한을 둔다.
        # (0.2s = 발행 주기 20Hz 기준 4 주기분)
        if dt > 0.2:
            dt = 0.2

        if bypass or dt <= 0.0:
            self._out_vx, self._out_vy, self._out_wz = tgt_vx, tgt_vy, tgt_wz
            # 우회 요청은 1 회만 쓰고 내린다. 다음 주기부터 다시 슬루 적용.
            self._slew_bypass = False
        else:
            self._out_vx = self._slew(self._out_vx, tgt_vx,
                                      self.accel_vx, self.decel_vx, dt)
            self._out_vy = self._slew(self._out_vy, tgt_vy,
                                      self.accel_vy, self.decel_vy, dt)
            self._out_wz = self._slew(self._out_wz, tgt_wz,
                                      self.accel_wz, self.decel_wz, dt)

        msg = Twist()
        msg.linear.x = self._out_vx
        msg.linear.y = self._out_vy
        msg.angular.z = self._out_wz
        self.pub.publish(msg)

        # ── 부스트 디버그 발행 ──
        # cmd_vel 과 같은 주기·같은 함수에서 내보내야 Foxglove Plot 에서
        # angular.z 와 시간축이 정확히 맞는다. 별도 타이머로 빼면 두
        # 신호의 샘플 시각이 어긋나 "부스트가 먼저인지 회전이 먼저인지"
        # 를 판단할 수 없게 된다.
        self.boost_pub.publish(Bool(data=bool(self._wz_boost)))
        self.wzmax_pub.publish(Float32(data=float(self._wz_max_now())))


def main(args=None):
    rclpy.init(args=args)
    node = JoypadTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

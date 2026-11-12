# -*- coding: utf-8 -*-
# =============================================================================
#  pose_goal_controller_node.py   (제네릭 go-to-pose 보행 컨트롤러 + 회피 합성)
#
#  Nav2(글로벌 플래너 + RPP)를 대체해, "목표 pose 한 점"을 향해 reactive 하게
#  걸어가는 컨트롤러. 매 주기 목표를 base_link 로 TF 변환해 재조준하므로
#  목표가 움직여도(예: 굴러가는 공의 standoff pose) 부드럽게 추종한다.
#
#  [왜 만드는가 — Nav2 와의 차이]
#  --------------------------------------------------------------------------
#   · Nav2 RPP 는 (vx, ω)만 내고 곡률 lookahead 가 진동하며, 데드존 경계를
#     계속 넘나들어 보행이 출렁이다 못해 무너질 때가 있다(start/stop 채터).
#     또 goal 을 줄 때마다 replan 하며 끊긴다. 이 컨트롤러는 플래너/코스트맵/
#     replan 이 없고, 명령을 visual_servo 와 동일한 floor 로 정형해 거동이 매끄럽다.
#   · 이 로봇은 홀로노믹(정책이 vy 를 받음) → RPP 가 못 쓰는 횡이동(strafe)으로
#     작은 측면오차를 호(arc) 회전 없이 평행이동으로 죽인다.
#
#   ★정확도는 Nav2 와 동급(목표가 map 좌표면 결국 localization 정확도에 묶임).
#     이 노드의 이점은 "정확도"가 아니라 "보행 안정성"이다 — 솔직히 못 박아둔다.
#
#  [입출력]
#  --------------------------------------------------------------------------
#   입력:
#     - goal_topic (geometry_msgs/PoseStamped) : 목표 pose.
#         · frame_id = 'odom'(부드러움, 짧은 홉) 또는 'map'(필드 좌표, localization
#           점프 가능) 또는 'base_link'(상대 명령). 매 주기 base_link 로 변환해 추종.
#         · 새 메시지가 오기 전까지 마지막 goal 을 유지(set-point). 멈추려면 enable=false.
#     - obstacle_topic (geometry_msgs/Vector3Stamped) : 회피 힌트(선택).
#         · depth_obstacle_detector_node 가 발행. 계약:
#             vector.x = 전방 코리도 최근접 장애물 거리[m] (없으면 큰 값, 예 99.0)
#             vector.y = 횡회피 방향·세기, 부호 포함 [-1, 1]
#                        (+ = 로봇이 +y(좌)로 비켜야 함 / 0 = 비킬 필요 없음)
#             vector.z = 예약(0)
#         · 이 노드가 "단일 권한"으로 추종 vy 와 회피 vy 를 한곳에서 합성한다
#           (vy 두 주인 충돌 방지). 힌트가 끊기면(timeout) "장애물 없음"으로 폴백.
#     - TF: goal.header.frame_id → base_frame (매 주기 최신 TF 폴백 lookup).
#   출력:
#     - cmd_vel_topic (geometry_msgs/Twist) : 기본 'cmd_vel_nav'(twist_mux navigation
#       슬롯, priority 20). Nav2 가 있던 자리에 그대로 드롭인 → 상위 FSM 핸드오프 로직 무변경.
#     - state_topic (std_msgs/String) : IDLE/TURN/TRAVEL/AVOID/SETTLE/ARRIVED/BLOCKED/NO_TF
#     - marker_topic (visualization_msgs/MarkerArray) : 목표/방위/주행선/허용원/회피/
#       상태텍스트(base_link, RViz). 추종·회피를 눈으로 즉시 검증.
#   서비스:
#     - ~/enable (std_srvs/SetBool) : 오케스트레이터가 on/off (visual_servo 와 동일 패턴).
#
#  [제어법 — turn-then-go + strafe 보조]
#  --------------------------------------------------------------------------
#   매 주기 base_link 에서:
#     g=(gx,gy)=목표 위치, ρ=|g|=거리, α=atan2(gy,gx)=목표점 방위,
#     ψ=목표 최종 heading(base 기준 회전오차).
#   상태:
#     TURN    : 목표가 옆/뒤(|α|>turn_in_place) → 제자리 회전으로 목표를 정면에 둔다
#               (긴 횡이동·후진 회피).
#     TRAVEL  : 목표를 대체로 바라봄 →
#                 vx = k_rho·ρ·cosα (전진성분)
#                 vy = k_lat·ρ·sinα (작은 방위오차를 strafe 로)
#                 wz = k_face·α     (코를 목표에 고정)
#     AVOID   : TRAVEL 중 전방 장애물 → vx 감속 + 여유 쪽 vy 사이드스텝(아래 합성).
#     SETTLE  : 위치 도착(ρ≤pos_tol) but 방위 미정렬(|ψ|>yaw_tol) → 제자리 회전으로 ψ 정렬.
#     ARRIVED : 위치·방위 모두 도착 → 정지(0) 유지.
#   ※ vy 는 보행 중(vx 또는 wz 활성)에만 정책이 듣는다 → 보행 게이트는 호출부에서 처리.
#
#  [회피 합성 — 단일 권한]
#   장애물 거리 c 가 react_dist..stop_dist 사이면:
#     scale = (c−stop_dist)/(react_dist−stop_dist) ∈ [0,1]   (가까울수록 0)
#     vx *= scale                                            (다가갈수록 감속, stop_dist 에서 0)
#     vy += avoid_gain·dir·(1−scale)                         (가까울수록 강하게 비킴)
#   c ≤ stop_dist 면 전진 0, 횡이동 최대로 비킨다. 코(wz)는 목표를 향한 채 둬서
#   장애물이 빠지면 곧장 직진 추종으로 복귀한다.
#
#  [한계 — 솔직하게]
#   - reactive only: 기둥·로봇 한둘 옆으로 비키는 수준. 긴 벽/U자 트랩은 못 빠져나옴.
#   - map 목표 추종 정확도 = localization 정확도. map→base_link 점프 시 명령이 튈 수
#     있어(짧은 홉은 odom 프레임 목표 권장). 본 노드는 floor·게이트로 거동만 매끄럽게 한다.
#   - 회피는 목(pan/tilt)이 전방을 봐야 작동(공추적으로 숙이면 눈 감음) → 상위 통합에서
#     네비 구간 목 정면 고정이 필요(이 노드 범위 밖, 별도 통합 단계).
# =============================================================================

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import (
    Twist, PoseStamped, PointStamped, Vector3Stamped, Point)
from std_msgs.msg import String, ColorRGBA, Bool
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener, TransformException
# do_transform_point 를 import 하면 PointStamped 변환이 tf2 에 등록된다.
from tf2_geometry_msgs import do_transform_point

# 공용 명령 정형(데드밴드/floor) — visual_servo 와 동일 거동을 위해 공유 모듈 사용.
from humanoid_utils.cmd_shaping import (
    clamp, wrap_angle, floor_vx, floor_vy, floor_wz)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """quaternion → yaw[rad] (평면 가정. roll/pitch 무시)."""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


class PoseGoalControllerNode(Node):
    """목표 pose 한 점을 reactive 하게 추종하는 보행 컨트롤러(+회피 합성)."""

    def __init__(self) -> None:
        super().__init__('pose_goal_controller_node')

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 토픽/프레임
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('goal_topic', '/pose_goal')
        # 기본 출력 = twist_mux navigation 슬롯(priority 20). Nav2 자리 드롭인.
        self.declare_parameter('cmd_vel_topic', 'cmd_vel_nav')
        self.declare_parameter('obstacle_topic', '/obstacle/avoid_hint')
        self.declare_parameter('state_topic', '/pose_goal/state')
        self.declare_parameter('marker_topic', '/pose_goal/markers')
        self.declare_parameter('base_frame', 'base_link')

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 명령 범위(정책 학습/실측 데드존; visual_servo 와 동일값)
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('vx_min', 0.20)
        self.declare_parameter('vx_max', 0.50)
        self.declare_parameter('vy_min', 0.10)
        self.declare_parameter('vy_max', 0.30)
        self.declare_parameter('wz_min', 0.50)
        self.declare_parameter('wz_max', 1.50)

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 제어 게인
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('k_rho', 0.8)         # 전진 ← 거리 ρ
        self.declare_parameter('k_lat', 1.2)         # 횡이동 ← 방위오차의 측면성분
        self.declare_parameter('k_face', 1.5)        # 회전 ← 목표점 방위 α(코 고정)
        self.declare_parameter('k_yaw_final', 1.5)   # 회전 ← 최종 heading 오차 ψ(SETTLE)
        # 직진 경로에서 작은 좌우 오차가 vy_min 으로 증폭되어 지그재그가 나는 것을 막는다.
        # 장애물 회피 vy 는 이 데드밴드 뒤에 더해져서 그대로 살아난다.
        self.declare_parameter('straight_lateral_deadband_m', 0.08)

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 임계/도착
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('arrive_pos_tol', 0.15)     # 위치 도착 허용[m] (래치 진입)
        self.declare_parameter('arrive_yaw_tol_deg', 12.0)  # 방위 도착 허용[deg] (래치 진입)
        # ── 도착 래치(터미널) ───────────────────────────────────────
        # 한번 ARRIVED 되면 새/이동 목표가 발행되기 전까지 무조건 정지 유지.
        #   · 오차 기반 재진입 없음 → localization 흔들림으로 안 깨지고,
        #     로봇을 손으로 들어 옮겨도(새 목표 발행이 없으면) 다시 안 쫓아간다.
        #   · 재출발은 '_on_goal' 에서 목표가 충분히 달라졌을 때만(아래 두 임계).
        self.declare_parameter('goal_update_pos_tol', 0.15)     # 새 목표가 이 이상 떨어져야 재출발[m]
        self.declare_parameter('goal_update_yaw_tol_deg', 20.0)  # 또는 방위가 이 이상 달라야[deg]
        # ── 정착영역 위치 히스테리시스 ──────────────────────────────
        # 위치 도착(rho≤arrive_pos_tol) 하면 'settling' 으로 진입해 yaw 만 맞춘다.
        # rho 가 이 값을 넘어야 다시 접근(TURN/TRAVEL)으로 빠진다. 목표 바로 위에서
        # alpha(목표 방위)가 노이즈로 ±180° 튀어 헛 TURN/strafe 하는 것을 막는다.
        self.declare_parameter('settle_exit_tol', 0.30)         # 정착영역 이탈 임계[m]
        # 최종 heading 을 맞출지. False 면 위치 도착만으로 ARRIVED(yaw 정착 생략).
        # ★localization yaw 가 노이즈 심하면(정지 중 psi 가 수십° 출렁) False 권장.
        self.declare_parameter('require_goal_yaw', True)
        self.declare_parameter('turn_in_place_deg', 60.0)   # |α| 이상이면 제자리 회전 먼저
        self.declare_parameter('control_rate_hz', 20.0)
        # 목표 신선도. 0 이면 만료 없음(마지막 goal 유지). >0 이면 그 시간 지나면 IDLE.
        self.declare_parameter('goal_timeout_s', 0.0)
        self.declare_parameter('auto_start', False)         # True 면 켜자마자 enable
        self.declare_parameter('auto_enable_on_goal', True)  # goal 수신 시 자동 enable
        # 정지락(game_stop_lock) 연동: True 면 락이 걸리는 순간 추종 중이던 목표를
        # 버린다. 퇴장/재진입 대기 중 락이 풀릴 때(복귀) 옛 목표로 튀어나가는 것을
        # 막는다(strategy 가 새 목표를 보낼 때까지 정지 유지).
        self.declare_parameter('game_stop_lock_topic', 'game_stop_lock')
        self.declare_parameter('clear_goal_on_lock', True)
        self.declare_parameter('hold_on_arrive', True)       # 도착 시 0 유지(False=정지명령만)

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 장애물 회피
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('avoid_enable', True)
        self.declare_parameter('obstacle_timeout_s', 0.5)   # 힌트 끊기면 '장애물 없음' 폴백
        self.declare_parameter('react_dist', 0.9)           # 이 거리부터 감속·비킴 시작[m]
        self.declare_parameter('stop_dist', 0.35)           # 이 거리면 전진 0[m]
        self.declare_parameter('avoid_gain', 1.0)           # 힌트 dir → vy 변환 게인
        # 전방이 막혀 vx≈0 인데 strafe 가 필요할 때 보행 게이트를 살리는 방법:
        #   여유가 이보다 크면 살짝 전진 creep 으로 게이트 충족, 아니면 회전으로 충족.
        self.declare_parameter('creep_min_dist', 0.5)
        # ── 목표 막힘 → 도착 간주(교착 방지) ────────────────────────
        # 목표 근처(rho≤block_near_dist)에서 장애물에 막혀(전방 clearance≤stop_dist)
        # 이 시간 이상 갇히면 '도착 간주'하고 멈춘다(터미널). 상태는 BLOCKED 로 알린다.
        # 0 이면 비활성(계속 시도). 경로 중간 장애물은 거리 게이트로 제외(계속 우회).
        self.declare_parameter('block_timeout_s', 2.5)
        self.declare_parameter('block_near_dist', 0.5)   # 이 거리 안에서 막힐 때만 포기[m]

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 명령 슬루(가속도 제한)
        #   정지→보행 급출발(0→0.50)·급회전(0→1.50)으로 정책이 휘청여 넘어지는 것 방지.
        #   매 주기 명령 변화량을 (accel/rate) 로 제한해 부드럽게 올린다.
        #   ※데드존 때문에 0→vx_min(0.20) 한 발은 불가피(그 밑은 정책이 안 걸음).
        #     그 첫 발만 vx_min 으로 부드럽게 출발하고, 그 위로는 천천히 가속한다.
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('slew_enable', True)
        self.declare_parameter('accel_lin', 0.4)    # 전·횡 가속 한계[m/s^2] (작을수록 천천히)
        self.declare_parameter('decel_lin', 0.8)    # 전·횡 감속 한계[m/s^2] (정지는 더 빨리)
        self.declare_parameter('accel_ang', 2.0)    # 회전 가속 한계[rad/s^2]
        self.declare_parameter('decel_ang', 4.0)    # 회전 감속 한계[rad/s^2]

        # ──────────────────────────────────────────────────────────
        # 파라미터 캐싱
        # ──────────────────────────────────────────────────────────
        g = self.get_parameter
        self._goal_topic = str(g('goal_topic').value)
        self._cmd_topic = str(g('cmd_vel_topic').value)
        self._obs_topic = str(g('obstacle_topic').value)
        state_topic = str(g('state_topic').value)
        marker_topic = str(g('marker_topic').value)
        self._base_frame = str(g('base_frame').value)

        self._vx_min = float(g('vx_min').value)
        self._vx_max = float(g('vx_max').value)
        self._vy_min = float(g('vy_min').value)
        self._vy_max = float(g('vy_max').value)
        self._wz_min = float(g('wz_min').value)
        self._wz_max = float(g('wz_max').value)

        self._k_rho = float(g('k_rho').value)
        self._k_lat = float(g('k_lat').value)
        self._k_face = float(g('k_face').value)
        self._k_yaw_final = float(g('k_yaw_final').value)
        self._straight_lat_deadband = float(g('straight_lateral_deadband_m').value)

        self._pos_tol = float(g('arrive_pos_tol').value)
        self._yaw_tol = math.radians(float(g('arrive_yaw_tol_deg').value))
        self._goal_upd_pos = float(g('goal_update_pos_tol').value)
        self._goal_upd_yaw = math.radians(
            float(g('goal_update_yaw_tol_deg').value))
        self._settle_exit = float(g('settle_exit_tol').value)
        self._require_yaw = bool(g('require_goal_yaw').value)
        self._turn_in_place = math.radians(float(g('turn_in_place_deg').value))
        self._rate = float(g('control_rate_hz').value)
        self._goal_timeout = float(g('goal_timeout_s').value)
        self._auto_enable_on_goal = bool(g('auto_enable_on_goal').value)
        self._game_stop_lock_topic = str(g('game_stop_lock_topic').value)
        self._clear_goal_on_lock = bool(g('clear_goal_on_lock').value)
        self._hold_on_arrive = bool(g('hold_on_arrive').value)

        self._avoid_enable = bool(g('avoid_enable').value)
        self._obs_timeout = float(g('obstacle_timeout_s').value)
        self._react_dist = float(g('react_dist').value)
        self._stop_dist = float(g('stop_dist').value)
        self._avoid_gain = float(g('avoid_gain').value)
        self._creep_min = float(g('creep_min_dist').value)
        self._block_timeout = float(g('block_timeout_s').value)
        self._block_near = float(g('block_near_dist').value)

        self._slew_enable = bool(g('slew_enable').value)
        self._accel_lin = float(g('accel_lin').value)
        self._decel_lin = float(g('decel_lin').value)
        self._accel_ang = float(g('accel_ang').value)
        self._decel_ang = float(g('decel_ang').value)

        # 런타임 튜닝(게인/임계/회피 즉시 반영).
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ──────────────────────────────────────────────────────────
        # 내부 상태
        # ──────────────────────────────────────────────────────────
        self._enabled = bool(g('auto_start').value)
        self._state = 'IDLE'
        self._arrived = False        # 도착 래치(터미널). 새 목표 전까지 멈춤 유지
        self._settling = False       # 정착영역(위치 도착, yaw 정렬 중) 히스테리시스 플래그
        self._latched_goal = None    # 도착 시점의 목표 pose(새 목표 변경 판정 기준)
        self._blocked = False        # 막힘으로 도착 간주됐는지(상태 BLOCKED 표시용)
        self._block_since = None     # 막힘 시작 시각(타임아웃 측정)
        # 직전 '발행된' 명령(슬루 연속성). 정지/안전정지는 즉시 0 으로 리셋된다.
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._prev_wz = 0.0
        self._goal = None            # 최신 목표 PoseStamped | None
        self._goal_stamp = None      # 목표 수신 시각(node clock)
        self._game_locked = False    # game_stop_lock 최신값(True=정지락)
        self._obs_c = None           # 최신 장애물 거리[m] | None
        self._obs_dir = 0.0          # 최신 횡회피 방향·세기 [-1,1]
        self._obs_stamp = None       # 장애물 힌트 수신 시각
        # 마커/텔레메트리용 마지막 계산값(콜백 간 캐시).
        self._last_goal_b = None     # (gx, gy) base | None
        self._last_psi = 0.0         # 최종 heading 오차[rad]

        # ──────────────────────────────────────────────────────────
        # TF / Pub / Sub / Srv
        # ──────────────────────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._cmd_pub = self.create_publisher(Twist, self._cmd_topic, 10)
        self._state_pub = self.create_publisher(String, state_topic, 10)
        self._marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)

        self.create_subscription(PoseStamped, self._goal_topic, self._on_goal, 10)
        self.create_subscription(
            Vector3Stamped, self._obs_topic, self._on_obstacle, 10)
        # 정지락 구독 — 락이 걸리면 추종 목표를 버려 복귀 시 튀어나감 방지.
        self.create_subscription(
            Bool, self._game_stop_lock_topic, self._on_game_lock, 10)

        self.create_service(SetBool, '~/enable', self._on_enable)

        self._timer = self.create_timer(1.0 / max(self._rate, 1.0), self._on_timer)

        self.get_logger().info(
            f"pose_goal_controller 시작 — goal='{self._goal_topic}', "
            f"cmd='{self._cmd_topic}', obstacle='{self._obs_topic}', "
            f"avoid={self._avoid_enable}, auto_start={self._enabled}")

    # =========================================================================
    #  런타임 파라미터 변경
    # =========================================================================
    def _on_set_parameters(self, params):
        m = {
            'vx_min': ('_vx_min', float), 'vx_max': ('_vx_max', float),
            'vy_min': ('_vy_min', float), 'vy_max': ('_vy_max', float),
            'wz_min': ('_wz_min', float), 'wz_max': ('_wz_max', float),
            'k_rho': ('_k_rho', float), 'k_lat': ('_k_lat', float),
            'k_face': ('_k_face', float), 'k_yaw_final': ('_k_yaw_final', float),
            'straight_lateral_deadband_m': ('_straight_lat_deadband', float),
            'arrive_pos_tol': ('_pos_tol', float),
            'goal_timeout_s': ('_goal_timeout', float),
            'avoid_enable': ('_avoid_enable', bool),
            'obstacle_timeout_s': ('_obs_timeout', float),
            'react_dist': ('_react_dist', float),
            'stop_dist': ('_stop_dist', float),
            'avoid_gain': ('_avoid_gain', float),
            'creep_min_dist': ('_creep_min', float),
            'block_timeout_s': ('_block_timeout', float),
            'block_near_dist': ('_block_near', float),
            'slew_enable': ('_slew_enable', bool),
            'accel_lin': ('_accel_lin', float),
            'decel_lin': ('_decel_lin', float),
            'accel_ang': ('_accel_ang', float),
            'decel_ang': ('_decel_ang', float),
            'hold_on_arrive': ('_hold_on_arrive', bool),
            'auto_enable_on_goal': ('_auto_enable_on_goal', bool),
        }
        for p in params:
            if p.name in m:
                attr, cast = m[p.name]
                setattr(self, attr, cast(p.value))
            elif p.name == 'arrive_yaw_tol_deg':
                self._yaw_tol = math.radians(float(p.value))
            elif p.name == 'goal_update_pos_tol':
                self._goal_upd_pos = float(p.value)
            elif p.name == 'goal_update_yaw_tol_deg':
                self._goal_upd_yaw = math.radians(float(p.value))
            elif p.name == 'settle_exit_tol':
                self._settle_exit = float(p.value)
            elif p.name == 'require_goal_yaw':
                self._require_yaw = bool(p.value)
            elif p.name == 'turn_in_place_deg':
                self._turn_in_place = math.radians(float(p.value))
        return SetParametersResult(successful=True)

    # =========================================================================
    #  콜백
    # =========================================================================
    def _on_enable(self, req: SetBool.Request, res: SetBool.Response):
        self._enabled = bool(req.data)
        self._arrived = False                  # enable 전환 시 래치 초기화(상태 신선화)
        self._settling = False
        self._latched_goal = None
        self._blocked = False
        self._block_since = None
        if not self._enabled:
            self._publish_cmd(0.0, 0.0, 0.0)   # 끄면 즉시 정지
            self._set_state('IDLE')
        res.success = True
        res.message = f'pose_goal {"ENABLED" if self._enabled else "DISABLED"}'
        self.get_logger().info(res.message)
        return res

    def _on_game_lock(self, msg: Bool):
        """game_stop_lock(twist_mux 정지락) 구독.

        lock=True(정지)로 '전환'되는 순간, 추종 중이던 목표를 완전히 버린다.
        퇴장(penalty)/재진입 대기 중 이 노드가 뒤에서 계속 추종하던 stale 목표를
        비워, 락이 풀리는 순간(복귀) 옛 목표로 튀어나가지 않게 한다. 락이 풀린 뒤엔
        strategy 가 새 목표를 보내야 다시 움직인다.
        lock=False(해제)로의 전환은 아무것도 하지 않는다(새 goal 을 기다림).
        """
        locked = bool(msg.data)
        if locked and not self._game_locked and self._clear_goal_on_lock:
            if self._goal is not None:
                self.get_logger().info(
                    "game_stop_lock ON → 추종 목표 클리어(복귀 시 옛 목표 돌진 방지)")
            self._clear_goal()
            self._enabled = False
            self._publish_cmd(0.0, 0.0, 0.0)   # 즉시 0속도
            self._set_state('IDLE')
        self._game_locked = locked

    def _clear_goal(self):
        """추종 중이던 목표와 관련 래치를 모두 버려 '목표 없음' 상태로 만든다."""
        self._goal = None
        self._goal_stamp = None
        self._latched_goal = None
        self._last_goal_b = None
        self._arrived = False
        self._settling = False
        self._blocked = False
        self._block_since = None

    def _on_goal(self, msg: PoseStamped) -> None:
        """새 목표 수신 → 저장.

        ★터미널 래치: 이미 ARRIVED 상태면, 새 목표가 도착 목표와 '충분히 다를' 때만
          재출발한다. 같은 자리 재발행(예: step-3 standoff 연속 발행)은 무시해
          도착점에서의 떨림을 막는다.
        """
        if not msg.header.frame_id:
            self.get_logger().warn('goal frame_id 비어 있음 — 무시(프레임 지정 필수)')
            return
        # 정지락 중엔 새 목표를 받지 않는다(복귀 후 strategy 가 다시 보냄).
        # 락 중 들어온 목표를 저장해두면 복귀 순간 그리로 튀어나가므로 무시한다.
        if self._game_locked:
            return
        if self._arrived and self._latched_goal is not None and \
                self._goal_is_close(msg, self._latched_goal):
            return  # 같은 목표 재발행 — 도착 유지(재출발 안 함)

        self._goal = msg
        self._goal_stamp = self.get_clock().now()
        self._arrived = False        # 새/이동 목표 → 래치 해제
        self._settling = False
        self._latched_goal = None
        self._blocked = False
        self._block_since = None
        if self._auto_enable_on_goal and not self._enabled:
            self._enabled = True
            self.get_logger().info('goal 수신 → 자동 enable')

    def _goal_is_close(self, a: PoseStamped, b: PoseStamped) -> bool:
        """두 목표 pose 가 (같은 프레임에서) 위치·방위 모두 갱신 임계 안인지."""
        if a.header.frame_id != b.header.frame_id:
            return False  # 프레임 다르면 다른 목표로 간주
        dx = a.pose.position.x - b.pose.position.x
        dy = a.pose.position.y - b.pose.position.y
        if math.hypot(dx, dy) > self._goal_upd_pos:
            return False
        ya = yaw_from_quaternion(a.pose.orientation.x, a.pose.orientation.y,
                                 a.pose.orientation.z, a.pose.orientation.w)
        yb = yaw_from_quaternion(b.pose.orientation.x, b.pose.orientation.y,
                                 b.pose.orientation.z, b.pose.orientation.w)
        return abs(wrap_angle(ya - yb)) <= self._goal_upd_yaw

    def _on_obstacle(self, msg: Vector3Stamped) -> None:
        """회피 힌트 수신 → 거리/방향 갱신."""
        self._obs_c = float(msg.vector.x)
        self._obs_dir = float(clamp(msg.vector.y, -1.0, 1.0))
        self._obs_stamp = self.get_clock().now()

    # =========================================================================
    #  목표 → base_link 변환
    # =========================================================================
    def _goal_in_base(self):
        """현재 목표를 base_link 기준 (gx, gy, psi) 로 변환. 실패 시 None.

        psi = 목표 최종 heading 의 base 기준 회전오차(0 이면 로봇이 목표 방향을 봄).
        동적 목/주행으로 exact-time 변환은 자주 실패 → 최신 TF 폴백을 쓴다.
        """
        if self._goal is None:
            return None
        src = self._goal.header.frame_id
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, src, RclpyTime())
        except TransformException as e:
            self.get_logger().warn(
                f'{src}→{self._base_frame} TF 없음({e}) — TF 트리/로컬라이제이션 확인',
                throttle_duration_sec=2.0)
            return None

        # 위치 변환(목표점 → base).
        ps = PointStamped()
        ps.header.frame_id = src
        ps.point = self._goal.pose.position
        try:
            pb = do_transform_point(ps, tf).point
        except Exception:  # noqa: BLE001 (변환 실패 방어)
            return None
        gx, gy = float(pb.x), float(pb.y)
        if not (math.isfinite(gx) and math.isfinite(gy)):
            return None

        # 방위 변환: 목표 heading(소스 프레임) + 변환의 회전(yaw) = base 기준 heading.
        q = self._goal.pose.orientation
        yaw_goal = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        qt = tf.transform.rotation
        yaw_tf = yaw_from_quaternion(qt.x, qt.y, qt.z, qt.w)
        psi = wrap_angle(yaw_goal + yaw_tf)
        return gx, gy, psi

    # =========================================================================
    #  제어 타이머 — FSM + 제어법
    # =========================================================================
    def _on_timer(self) -> None:
        if not self._enabled:
            return  # IDLE: 아무것도 발행 안 함(낮은 우선순위 소스에 양보)

        now = self.get_clock().now()

        # ── 목표 유무/신선도 ──
        if self._goal is None:
            self._set_state('IDLE')
            return
        if self._goal_timeout > 0.0 and self._goal_stamp is not None and \
                (now - self._goal_stamp).nanoseconds * 1e-9 > self._goal_timeout:
            self.get_logger().info(
                'goal timeout — IDLE(set-point 만료)', throttle_duration_sec=1.0)
            self._goal = None
            self._arrived = False
            self._settling = False
            self._latched_goal = None
            self._blocked = False
            self._block_since = None
            self._publish_cmd(0.0, 0.0, 0.0)
            self._set_state('IDLE')
            return

        # ── 목표 base 변환 ──
        gb = self._goal_in_base()
        if gb is None:
            self._set_state('NO_TF')
            self._publish_cmd(0.0, 0.0, 0.0)   # TF 없으면 안전 정지
            self._publish_markers(now)
            return
        gx, gy, psi = gb
        self._last_goal_b = (gx, gy)
        self._last_psi = psi

        rho = math.hypot(gx, gy)
        alpha = math.atan2(gy, gx)

        # ── 1) 터미널 도착 래치 ──────────────────────────────────────
        # 한번 ARRIVED 면 무조건 정지 유지. 오차로 안 풀린다(localization 흔들림·
        # 손으로 들어 옮겨도 재출발 안 함). 재출발은 _on_goal 에서 '다른 목표'가
        # 발행될 때만 일어난다(같은 자리 재발행은 무시 → step3 연속 standoff 호환).
        if self._arrived:
            if self._hold_on_arrive:
                self._publish_motion(0.0, 0.0, 0.0)
            self._set_state('BLOCKED' if self._blocked else 'ARRIVED')
            self._publish_markers(now)
            return

        # ── 2) 정착영역 위치 히스테리시스 ────────────────────────────
        # 위치 도착하면 settling 진입. settle_exit 를 넘어야 다시 접근으로 빠진다.
        # 목표 바로 위에서 alpha 가 노이즈로 ±180° 튀어 헛 TURN/strafe 하는 것을 차단.
        if rho <= self._pos_tol:
            self._settling = True
        elif rho >= self._settle_exit:
            self._settling = False

        if self._settling:
            # 위치는 도착 — yaw 만 맞춘다(또는 require_goal_yaw=False 면 바로 도착).
            if (not self._require_yaw) or abs(psi) <= self._yaw_tol:
                self._arrived = True
                self._latched_goal = self._goal   # 도착 목표 기억(재출발 판정 기준)
                if self._hold_on_arrive:
                    self._publish_motion(0.0, 0.0, 0.0)
                self._set_state('ARRIVED')
                self._publish_markers(now)
                return
            # yaw 정렬(제자리 회전). alpha 는 쓰지 않는다(목표 위라 의미 없음).
            wz = floor_wz(self._k_yaw_final * psi, self._wz_min, self._wz_max)
            self._set_state('SETTLE')
            self._publish_motion(0.0, 0.0, wz)
            self._telemetry(rho, alpha, psi, 0.0, 0.0, wz)
            self._publish_markers(now)
            return

        # ── 3) 접근: 목표가 옆/뒤면 제자리 회전 먼저(TURN) ──
        if abs(alpha) > self._turn_in_place:
            wz = floor_wz(self._k_face * alpha, self._wz_min, self._wz_max)
            self._set_state('TURN')
            self._publish_motion(0.0, 0.0, wz)
            self._telemetry(rho, alpha, psi, 0.0, 0.0, wz)
            self._publish_markers(now)
            return

        # ── TRAVEL: 전진 + 横 strafe + 코 고정 ──
        vx_des = self._k_rho * rho * math.cos(alpha)   # 전진성분
        vy_des = self._k_lat * rho * math.sin(alpha)   # 측면성분(strafe)
        wz = floor_wz(self._k_face * alpha, self._wz_min, self._wz_max)
        if abs(gy) < self._straight_lat_deadband:
            vy_des = 0.0

        # ── 회피 합성(단일 권한) ──
        avoiding = False
        c = self._obstacle_clearance(now) if self._avoid_enable else None
        if c is not None and c < self._react_dist:
            avoiding = True
            # 감속 scale: react_dist 에서 1 → stop_dist 에서 0.
            span = max(self._react_dist - self._stop_dist, 1e-3)
            scale = clamp((c - self._stop_dist) / span, 0.0, 1.0)
            vx_des *= scale
            # 여유 쪽으로 사이드스텝(가까울수록 강하게).
            vy_des += self._avoid_gain * self._obs_dir * (1.0 - scale)

        # ── 목표 막힘 → 도착 간주(교착 방지) ──
        #   목표 근처(rho≤block_near)에서 전방이 stop_dist 안으로 막혀 일정 시간
        #   지속되면 '도착 간주'(터미널, 상태 BLOCKED). 경로 중간 장애물은 rho 게이트로
        #   제외 → 계속 우회. block_timeout=0 이면 비활성.
        blocked_now = (self._block_timeout > 0.0 and avoiding and c is not None
                       and c <= self._stop_dist and rho <= self._block_near)
        if blocked_now:
            if self._block_since is None:
                self._block_since = now
            elif (now - self._block_since).nanoseconds * 1e-9 >= self._block_timeout:
                self._arrived = True
                self._blocked = True
                self._latched_goal = self._goal
                self._block_since = None
                self.get_logger().info(
                    f'목표 막힘 {self._block_timeout:.1f}s → 도착 간주(BLOCKED)')
                self._publish_motion(0.0, 0.0, 0.0)
                self._set_state('BLOCKED')
                self._publish_markers(now)
                return
        else:
            self._block_since = None   # 막힘 해소 → 타이머 리셋

        # ── floor + 보행 게이트(vy 는 보행 중에만 적용) ──
        vx = floor_vx(vx_des, self._vx_min, self._vx_max)
        walking = (vx >= self._vx_min) or (abs(wz) >= self._wz_min)
        vy_floored = floor_vy(vy_des, self._vy_min, self._vy_max)

        if not walking and vy_floored != 0.0:
            # 전방이 막혀 vx≈0·wz≈0 인데 strafe 가 필요한 경우:
            #   여유가 있으면 살짝 creep, 없으면 여유 쪽으로 회전해 게이트를 살린다.
            if avoiding and (self._obs_c is not None) and \
                    self._obs_c <= self._creep_min:
                wz = math.copysign(self._wz_min, self._obs_dir)  # 비킬 쪽으로 회전
            else:
                vx = self._vx_min                               # 게이트용 creep
            walking = True
        vy = vy_floored if walking else 0.0

        self._set_state('AVOID' if avoiding else 'TRAVEL')
        self._publish_motion(vx, vy, wz)
        self._telemetry(rho, alpha, psi, vx, vy, wz)
        self._publish_markers(now)

    # =========================================================================
    #  장애물 신선도
    # =========================================================================
    def _obstacle_clearance(self, now):
        """신선한 장애물 거리[m] 반환. 없거나 stale 이면 None(=장애물 없음 폴백)."""
        if self._obs_c is None or self._obs_stamp is None:
            return None
        if (now - self._obs_stamp).nanoseconds * 1e-9 > self._obs_timeout:
            return None
        if not math.isfinite(self._obs_c):
            return None
        return self._obs_c

    # =========================================================================
    #  발행 헬퍼
    # =========================================================================
    def _publish_cmd(self, vx: float, vy: float, wz: float) -> None:
        """즉시(슬루 없이) 발행. 안전 정지(disable/NO_TF/IDLE)용.

        직전 명령 캐시도 발행값으로 갱신해 다음 슬루의 출발점을 맞춘다(보통 0).
        """
        t = Twist()
        t.linear.x = float(vx)
        t.linear.y = float(vy)
        t.angular.z = float(wz)
        self._cmd_pub.publish(t)
        self._prev_vx, self._prev_vy, self._prev_wz = vx, vy, wz

    def _slew(self, prev, target, accel, decel, vmin, vmax):
        """데드존 인식 가속도 제한. prev·target 은 0 또는 |.|∈[vmin,vmax](부호 포함).

        · target 0          → 0 쪽으로 decel 감속, vmin 밑이면 정지(데드존엔 못 머묾)
        · 정지/데드존서 출발 → 첫 발은 vmin 으로 부드럽게(그 밑은 정책이 안 걸음)
        · 부호 반전          → 일단 0 경유(급반전 방지)
        · 같은 방향 보행 중  → 증가는 accel, 감소는 decel 만큼만 변화, [vmin,vmax] 클램프
        """
        up = accel / max(self._rate, 1.0)     # 이번 주기 허용 증가량
        dn = decel / max(self._rate, 1.0)     # 이번 주기 허용 감소량
        if target == 0.0:
            n = prev + clamp(-prev, -dn, dn)
            return 0.0 if abs(n) < vmin else n
        if prev == 0.0 or abs(prev) < vmin:
            return math.copysign(vmin, target)         # 부드러운 출발(최소 보행속도)
        if (prev > 0.0) != (target > 0.0):             # 부호 반전 → 0 경유
            n = prev + clamp(-prev, -dn, dn)
            return 0.0 if abs(n) < vmin else n
        if abs(target) > abs(prev):                    # 가속
            n = prev + math.copysign(min(up, abs(target) - abs(prev)), target)
        else:                                          # 감속
            n = prev - math.copysign(min(dn, abs(prev) - abs(target)), prev)
        return math.copysign(clamp(abs(n), vmin, vmax), n)

    def _publish_motion(self, vx_t: float, vy_t: float, wz_t: float) -> None:
        """슬루(가속도 제한)를 적용해 발행. 보행 단계(TURN/SETTLE/TRAVEL/ARRIVED)용."""
        if not self._slew_enable:
            self._publish_cmd(vx_t, vy_t, wz_t)
            return
        vx = self._slew(self._prev_vx, vx_t, self._accel_lin, self._decel_lin,
                        self._vx_min, self._vx_max)
        vy = self._slew(self._prev_vy, vy_t, self._accel_lin, self._decel_lin,
                        self._vy_min, self._vy_max)
        wz = self._slew(self._prev_wz, wz_t, self._accel_ang, self._decel_ang,
                        self._wz_min, self._wz_max)
        self._publish_cmd(vx, vy, wz)

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self.get_logger().info(f'상태: {self._state} → {state}')
            self._state = state
        self._state_pub.publish(String(data=state))

    def _telemetry(self, rho, alpha, psi, vx, vy, wz) -> None:
        c = self._obs_c if self._obs_c is not None else float('inf')
        self.get_logger().info(
            f'TELEM [{self._state}] rho={rho:.2f}m a={math.degrees(alpha):+5.0f} '
            f'psi={math.degrees(psi):+5.0f} clr={c:.2f} '
            f'cmd=(vx{vx:+.2f},vy{vy:+.2f},wz{wz:+.2f})',
            throttle_duration_sec=0.5)

    def _publish_markers(self, now) -> None:
        """목표/방위/주행선/허용원/회피/상태텍스트를 base_link 에 발행(RViz)."""
        arr = MarkerArray()
        stamp = now.to_msg()
        life = Duration(seconds=0.5).to_msg()

        def base_marker(mid, mtype):
            m = Marker()
            m.header.frame_id = self._base_frame
            m.header.stamp = stamp
            m.ns = 'pose_goal'
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.lifetime = life
            return m

        if self._last_goal_b is not None:
            gx, gy = self._last_goal_b

            # (1) 목표점(파랑 구).
            s = base_marker(1, Marker.SPHERE)
            s.pose.position.x, s.pose.position.y, s.pose.position.z = gx, gy, 0.05
            s.scale.x = s.scale.y = s.scale.z = 0.10
            s.color = ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9)
            arr.markers.append(s)

            # (2) 목표 최종 heading 화살표(목표점에서 ψ 방향).
            hd = base_marker(2, Marker.ARROW)
            hx = gx + 0.3 * math.cos(self._last_psi)
            hy = gy + 0.3 * math.sin(self._last_psi)
            hd.points = [Point(x=float(gx), y=float(gy), z=0.05),
                         Point(x=float(hx), y=float(hy), z=0.05)]
            hd.scale.x, hd.scale.y, hd.scale.z = 0.02, 0.05, 0.06
            hd.color = ColorRGBA(r=0.2, g=0.9, b=0.9, a=0.9)
            arr.markers.append(hd)

            # (3) 주행선(로봇 원점 → 목표점).
            ln = base_marker(3, Marker.LINE_STRIP)
            ln.scale.x = 0.015
            ln.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.7)
            ln.points = [Point(x=0.0, y=0.0, z=0.02),
                         Point(x=float(gx), y=float(gy), z=0.02)]
            arr.markers.append(ln)

            # (4) 위치 허용원(초록 원기둥, 도착 판정 반경).
            ring = base_marker(4, Marker.CYLINDER)
            ring.pose.position.x, ring.pose.position.y = gx, gy
            ring.pose.position.z = 0.01
            ring.scale.x = ring.scale.y = 2.0 * self._pos_tol
            ring.scale.z = 0.01
            ring.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.25)
            arr.markers.append(ring)

        # (5) 회피 표시: AVOID 일 때 비킬 방향 화살표(원점에서 ±y).
        if self._state == 'AVOID':
            av = base_marker(5, Marker.ARROW)
            sign = 1.0 if self._obs_dir >= 0.0 else -1.0
            av.points = [Point(x=0.1, y=0.0, z=0.06),
                         Point(x=0.1, y=float(0.3 * sign), z=0.06)]
            av.scale.x, av.scale.y, av.scale.z = 0.03, 0.06, 0.08
            av.color = ColorRGBA(r=1.0, g=0.3, b=0.1, a=0.95)
            arr.markers.append(av)

        # (6) 상태 텍스트.
        tx = base_marker(6, Marker.TEXT_VIEW_FACING)
        tx.pose.position.x, tx.pose.position.y, tx.pose.position.z = 0.0, 0.0, 0.45
        tx.scale.z = 0.08
        tx.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        tx.text = self._state
        arr.markers.append(tx)

        self._marker_pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseGoalControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

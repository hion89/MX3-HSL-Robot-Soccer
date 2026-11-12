# -*- coding: utf-8 -*-
# =============================================================================
#  ball_follow_nav_node.py
#
#  "map 프레임 공 위치"를 받아서 standoff 목적지를 /pose_goal 로 발행해
#  pose_goal_controller(go-to-pose 보행 컨트롤러)가 공 근처까지 접근하게 하는
#  Phase 2 노드(러프 접근 전용). 정밀 정렬/킥(Phase 3)은 다루지 않는다.
#  ※예전엔 Nav2(NavigateToPose)로 보냈으나, RL 보행정책에 더 안정적인
#    pose_goal_controller 로 교체했다(go-to-pose + 회피 + 도착래치 + 막힘처리 내장).
#
#  [두 가지 접근 모드 — align_to_goal 로 전환]
#  --------------------------------------------------------------------------
#   (A) align_to_goal = False  (레거시: 로봇→공 직선상 standoff)
#       goal 위치 = 공 − standoff × (로봇→공 단위벡터)
#       → 로봇이 어디 있든 "공 바로 앞에 못미쳐" 멈춘다. 골대 정렬과 무관.
#
#   (B) align_to_goal = True   (신규: 골대–공–로봇 일직선 정렬)  ★ 기본값
#       기준선을 "로봇→공"이 아니라 "공격 골대 G → 공 B 연장선"으로 잡는다.
#         u  = (B − G) / |B − G|        # 골대에서 공을 향하는 단위벡터(=골대 반대쪽)
#         P  = B + standoff × u         # 공보다 골대에서 더 먼 쪽(=공 뒤)에 목적지
#         yaw= atan2(G.y−P.y, G.x−P.x)  # 공을 관통해 골대를 바라보는 방향
#       → 도착하면 로봇–공–골대가 한 줄에 놓이고, 로봇은 골대 정면을 본다.
#         그대로 전진/킥하면 공이 골대로 향한다(슈팅 셋업).
#
#       검산: G=(4.5,0), B=(1.0,0), standoff=0.45 → u=(−1,0) → P=(0.55,0),
#             yaw=0(+x=골대). 로봇(0.55)–공(1.0)–골대(4.5) 정렬. OK.
#
#  [무엇을 받고 무엇을 하는가]
#  --------------------------------------------------------------------------
#   입력:
#     - /ball/map_point  (geometry_msgs/PointStamped, map 프레임)
#         · ball_map_projector_node 가 발행. 공이 안 보이면 발행이 끊긴다
#           → 마지막 수신 시각으로 '분실'을 판정한다.
#     - TF: map → base_link  (로봇 현재 pose. 로컬라이제이션 스택이 발행)
#     - 공격 골대 좌표(target_goal_x/y): map 프레임 파라미터. 좌표계는 팀 상대다.
#         · 필드 컨벤션(field_lines_node): 원점=정중앙, +x=상대 골대 방향, 골라인 x=±4.5.
#         · 재진입 시드(sideline_reentry_pose)가 우리 골을 항상 -x 로 심으므로,
#           하프타임에 물리 사이드가 바뀌어도 상대 골=+x 가 유지된다. 따라서
#           target_goal_x 는 4.5 고정이며 사이드 배정에 따라 뒤집지 않는다
#           (뒤집으면 공을 우리 골 방향으로 차게 됨). 파라미터는 다른 필드
#           프로파일(연습장 1.95 등) override 용으로만 둔다.
#   동작:
#     - 주기(update_period)마다 위 (A)/(B) 중 한 방식으로 standoff 목적지(PoseStamped,
#       map)를 계산해 /pose_goal 로 발행한다. pose_goal_controller 가 매 주기 재조준해
#       추종하므로 공이 움직여도 부드럽게 따라간다.
#     - 직전 발행 goal 에서 resend_threshold[m] 이상 움직였을 때만 재발행(토픽 스팸 방지).
#     - 공 분실(수신 끊김 timeout) 시 pose controller 를 꺼 로봇을 세운다.
#   ★핸드오프(Phase2→Phase3) — pose controller enable 소유:
#     - ball_follow 가 pose_goal_controller 의 ~/enable 을 직접 토글한다. 접근 중엔 ON,
#       도착/막힘/분실/disable 시 OFF. OFF 하면 pose controller 가 cmd_vel_nav(우선순위
#       20)를 비우므로, visual_servo(cmd_vel_behavior, 10)가 비로소 구동된다.
#     - 상위 전략 FSM 은 종전대로 ball_follow/visual_servo 의 ~/enable 만 토글하면 된다
#       (pose controller 는 ball_follow 가 대신 관리 → 전략 FSM 무변경).
#   출력(시각화/연계용):
#     - /ball_follow/status       (std_msgs/String)        : IDLE/FOLLOWING/ARRIVED/LOST
#         · pose controller 가 BLOCKED(장애물로 목표 근처서 막힘)면 ARRIVED 로 보고
#           핸드오프(최대한 근접). 전략 FSM 은 종전처럼 'ARRIVED' 만 보면 된다.
#     - /ball_follow/goal_marker  (visualization_msgs/Marker)     : 현재 goal 화살표(RViz)
#     - /ball_follow/align_markers(visualization_msgs/MarkerArray) : 골대–공–목적지
#         정렬선 + 세 지점 구(球). align_to_goal=True 일 때만 의미 있는 직선이 보인다
#         → 일직선 정렬 여부를 RViz 에서 눈으로 바로 확인.
#
#  [설계 근거 — 왜 standoff 인가]
#  --------------------------------------------------------------------------
#   공 위치 자체를 goal 로 주면 로봇이 공을 밟으러 간다. Phase 2 는 "근처까지"만
#   하고 정밀 정렬은 Phase 3(visual)이 맡으므로, 공보다 standoff 만큼 (모드 B 에선
#   골대 반대쪽으로) 떨어진 곳에서 멈춰 Phase 3 가 다룰 공간을 남긴다.
#   도착 판정(arrived_distance)에서 status='ARRIVED' 를 내면 상위가 Phase3 로 넘긴다.
#
#  [한계 — 솔직하게]
#   - 접근 정확도 = 로컬라이제이션 정확도. map→base_link 가 틀어지면 계산된 P 가
#     아니라 "틀어진 만큼 옆"에 도착한다(러프 접근이라 Phase 3 가 보정).
#   - 모드 B 에서 공이 골대에 거의 붙으면(|B−G|≈0) 기준선이 정의되지 않는다 →
#     이 경우는 로봇→공 방향(모드 A)으로 안전 폴백한다(throttle 경고).
#   - 장애물 회피는 pose_goal_controller 의 회피(/obstacle/avoid_hint) 가 담당한다.
# =============================================================================

import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time as RclpyTime
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Bool, ColorRGBA
from std_srvs.srv import SetBool

from tf2_ros import Buffer, TransformListener, TransformException


# 모드 B 에서 "공이 골대에 붙었다"고 보고 폴백할 거리[m]. 이보다 |B−G| 가 작으면
# 골대→공 방향이 수치적으로 불안정해지므로 로봇→공 방향(모드 A)으로 안전 폴백한다.
_GOAL_BALL_MIN_DIST = 0.05


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """2D yaw[rad] → 평면 quaternion(roll=pitch=0)."""
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """quaternion → yaw[rad] (평면 가정)."""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


class BallFollowNavNode(Node):
    """map 공 위치를 standoff 목적지로 변환해 /pose_goal 발행, pose controller로 접근하는 노드(Phase 2)."""

    def __init__(self) -> None:
        super().__init__('ball_follow_nav_node')

        # ──────────────────────────────────────────────────────────
        # 파라미터 선언
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('ball_point_topic', '/ball/map_point')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        # ── pose controller 연동(Nav2 대체) ────────────────────────
        # standoff 목적지를 PoseStamped 로 이 토픽에 발행 → pose_goal_controller 가 추종.
        self.declare_parameter('pose_goal_topic', '/pose_goal')
        # pose controller 의 상태 토픽(도착/막힘 판정 보조).
        self.declare_parameter('pose_state_topic', '/pose_goal/state')
        # ★핸드오프 핵심: ball_follow 가 pose controller 의 enable 을 소유한다.
        #   Phase2 끌 때 pose controller 를 꺼서 cmd_vel_nav(우선순위20)를 비워야
        #   visual_servo(cmd_vel_behavior, 10)가 구동된다.
        self.declare_parameter(
            'pose_ctrl_enable_srv', '/pose_goal_controller_node/enable')

        # ── 정렬 모드 ──────────────────────────────────────────────
        # True  : 골대–공–로봇 일직선 정렬(공 뒤에서 골대를 보게)  ★ 기본
        # False : 레거시(로봇→공 직선상에서 공 앞 standoff)
        self.declare_parameter('align_to_goal', True)
        # 공격 골대 중심(map 프레임). 좌표계가 팀 상대라 상대 골=+x 고정.
        # 재진입 시드가 우리 골을 늘 -x 로 심으므로 하프타임에도 이 값은 그대로다.
        # → 사이드 배정에 따라 뒤집지 말 것(뒤집으면 킥 방향이 우리 골로 반대).
        self.declare_parameter('target_goal_x', 4.5)   # S-Field 골라인(=half_length). 연습장=1.95
        self.declare_parameter('target_goal_y', 0.0)

        # 공에서 몇 m 떨어진 곳에서 멈출지(모드 B: 골대 반대쪽 / 모드 A: 로봇 쪽).
        self.declare_parameter('standoff_m', 0.45)
        # goal 갱신 주기[s]. "1초에 한 번이면 충분"의 그 주기.
        self.declare_parameter('update_period_s', 1.0)
        # 직전 전송 goal 에서 이 거리[m] 이상 목적지가 움직였을 때만 재전송(replan 끊김 방지).
        self.declare_parameter('resend_threshold_m', 0.15)
        # 공 수신이 이 시간[s] 끊기면 분실로 판정 → 진행 goal 취소(정지).
        self.declare_parameter('ball_lost_timeout_s', 1.0)
        # 로봇이 "목적지 P"에 이 거리[m] 이내로 이미 들어와 있으면 접근 불필요 → ARRIVED.
        #  · 모드 B 에선 'P 까지의 거리'로 판정한다(공까지 거리로 판정하면 공의 잘못된
        #    쪽에 있어도 도착으로 오인하므로). 모드 A 에선 종전대로 '공까지 거리'.
        self.declare_parameter('arrived_distance_m', 0.55)

        # 출력 토픽
        self.declare_parameter('status_topic', '/ball_follow/status')
        self.declare_parameter('goal_marker_topic', '/ball_follow/goal_marker')
        self.declare_parameter('align_marker_topic', '/ball_follow/align_markers')
        self.declare_parameter('game_stop_lock_topic', 'game_stop_lock')

        # ── enable 게이트 ──────────────────────────────────────────
        # True  : 노드가 떠 있는 동안 항상 공을 추종(레거시 단독 동작).
        # False : 상위 오케스트레이터(전략 FSM)가 ~/enable 서비스로 켤 때만 추종. ★기본
        #   step3 통합: owner 로봇만 Phase2 를 켜고, 나머지/비-PLAYING 구간엔 끈다.
        self.declare_parameter('auto_start', False)

        # ──────────────────────────────────────────────────────────
        # 파라미터 캐싱
        # ──────────────────────────────────────────────────────────
        ball_topic = str(self.get_parameter('ball_point_topic').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        pose_goal_topic = str(self.get_parameter('pose_goal_topic').value)
        pose_state_topic = str(self.get_parameter('pose_state_topic').value)
        self._pose_ctrl_srv = str(
            self.get_parameter('pose_ctrl_enable_srv').value)

        self._align = bool(self.get_parameter('align_to_goal').value)
        self._goal_x = float(self.get_parameter('target_goal_x').value)
        self._goal_y = float(self.get_parameter('target_goal_y').value)

        self._standoff = float(self.get_parameter('standoff_m').value)
        self._update_period = float(self.get_parameter('update_period_s').value)
        self._resend_thr = float(self.get_parameter('resend_threshold_m').value)
        self._lost_timeout = float(self.get_parameter('ball_lost_timeout_s').value)
        self._arrived_dist = float(self.get_parameter('arrived_distance_m').value)

        status_topic = str(self.get_parameter('status_topic').value)
        marker_topic = str(self.get_parameter('goal_marker_topic').value)
        align_topic = str(self.get_parameter('align_marker_topic').value)
        game_stop_lock_topic = str(self.get_parameter('game_stop_lock_topic').value)

        # 런타임 튜닝 콜백(standoff/threshold/timeout/arrived/정렬모드/골대좌표를
        # ros2 param set 으로 즉시 반영).
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ──────────────────────────────────────────────────────────
        # 내부 상태
        # ──────────────────────────────────────────────────────────
        self._ball_xy = None            # 최신 공 map 좌표 (x, y) | None
        self._ball_stamp = None         # 최신 공 수신 시각(rclpy Time) | None
        self._last_sent_goal_xy = None  # 마지막 발행 goal 위치(재발행 판정용)
        self._pose_state = None         # pose controller 최신 상태 문자열(ARRIVED/BLOCKED/...)
        self._pose_ctrl_on = None       # pose_ctrl enable 캐시(None=미상→첫 호출 강제)
        self._state = 'IDLE'            # IDLE / FOLLOWING / ARRIVED / LOST
        self._game_stop_locked = False
        # enable 게이트: auto_start 가 기본 상태. ~/enable 서비스로 런타임 토글.
        self._enabled = bool(self.get_parameter('auto_start').value)

        # ──────────────────────────────────────────────────────────
        # TF / Action / Pub / Sub
        # ──────────────────────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # standoff 목적지 발행 + pose controller enable 소유 + 상태 구독.
        self._goal_pub = self.create_publisher(PoseStamped, pose_goal_topic, 10)
        self._pose_ctrl_client = self.create_client(SetBool, self._pose_ctrl_srv)
        self.create_subscription(
            String, pose_state_topic, self._on_pose_state, 10)

        self._status_pub = self.create_publisher(String, status_topic, 10)
        self._marker_pub = self.create_publisher(Marker, marker_topic, 10)
        self._align_pub = self.create_publisher(MarkerArray, align_topic, 10)

        self._ball_sub = self.create_subscription(
            PointStamped, ball_topic, self._on_ball_point, 10)
        self.create_subscription(
            Bool, game_stop_lock_topic, self._on_game_stop_lock, 10)

        # ~/enable : 상위 오케스트레이터가 Phase2 추종을 켜고 끄는 서비스(SetBool).
        #   data=true  → 추종 시작, data=false → 정지(pose_ctrl OFF) 후 IDLE.
        self._enable_srv = self.create_service(
            SetBool, '~/enable', self._on_enable)

        # 주기 타이머 — goal 계산/발행/분실 판정/pose_ctrl 토글을 담당.
        self._timer = self.create_timer(self._update_period, self._on_timer)

        self.get_logger().info(
            f"ball_follow(pose_goal) 시작 — ball='{ball_topic}', "
            f"goal→'{pose_goal_topic}', pose_ctrl='{self._pose_ctrl_srv}', "
            f"enabled={self._enabled}(auto_start), "
            f"align_to_goal={self._align}, goal=({self._goal_x:.2f},{self._goal_y:.2f}), "
            f"standoff={self._standoff}m, update={self._update_period}s")

    # =========================================================================
    #  런타임 파라미터 변경 콜백
    # =========================================================================
    def _on_set_parameters(self, params):
        for p in params:
            if p.name == 'align_to_goal':
                self._align = bool(p.value)
            elif p.name == 'target_goal_x':
                self._goal_x = float(p.value)
            elif p.name == 'target_goal_y':
                self._goal_y = float(p.value)
            elif p.name == 'standoff_m':
                self._standoff = float(p.value)
            elif p.name == 'resend_threshold_m':
                self._resend_thr = float(p.value)
            elif p.name == 'ball_lost_timeout_s':
                self._lost_timeout = float(p.value)
            elif p.name == 'arrived_distance_m':
                self._arrived_dist = float(p.value)
            elif p.name == 'update_period_s':
                # 타이머 주기는 즉시 못 바꾸므로 타이머를 재생성한다.
                self._update_period = float(p.value)
                self._timer.destroy()
                self._timer = self.create_timer(self._update_period, self._on_timer)
        return SetParametersResult(successful=True)

    # =========================================================================
    #  공 수신 콜백 — 최신 map 좌표/시각만 저장(처리는 타이머가)
    # =========================================================================
    def _on_ball_point(self, msg: PointStamped) -> None:
        # ball_map_projector 가 이미 map 프레임으로 발행하므로 그대로 사용.
        # (혹시 다른 프레임이면 경고만 하고 좌표는 그대로 둠 — 보통 발생 안 함)
        if msg.header.frame_id and msg.header.frame_id != self._map_frame:
            self.get_logger().warn(
                f"공 좌표 frame='{msg.header.frame_id}' 가 map_frame 과 다름 — 변환 없이 사용",
                throttle_duration_sec=5.0)
        x = float(msg.point.x)
        y = float(msg.point.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return  # 이상값 방어
        self._ball_xy = (x, y)
        self._ball_stamp = self.get_clock().now()

    # =========================================================================
    #  목적지(P)·방위(yaw) 계산 — 모드 A/B 분기. (rx, ry)=로봇 현재 위치
    #  반환: (gx, gy, gyaw, used_align)
    #    used_align : 실제로 모드 B 정렬을 적용했는지(폴백 시 False)
    # =========================================================================
    def _compute_goal(self, rx, ry, bx, by):
        if self._align:
            # ── 모드 B: 골대 G → 공 B 연장선 위에서 공 뒤로 standoff ──
            gx0, gy0 = self._goal_x, self._goal_y           # 공격 골대 중심
            vx, vy = (bx - gx0), (by - gy0)                 # 골대→공 벡터
            n = math.hypot(vx, vy)
            if n >= _GOAL_BALL_MIN_DIST:
                ux, uy = vx / n, vy / n                     # 골대→공 단위(=골대 반대쪽)
                gx = bx + self._standoff * ux               # 공보다 골대에서 더 먼 쪽
                gy = by + self._standoff * uy
                # 공을 관통해 골대를 바라보는 방위(= −u 방향).
                gyaw = math.atan2(gy0 - gy, gx0 - gx)
                return gx, gy, gyaw, True
            # 공이 골대에 거의 붙음 → 기준선 불안정. 모드 A 로 폴백.
            self.get_logger().warn(
                f'공이 공격 골대에 너무 가까움(|B−G|={n:.3f}m<{_GOAL_BALL_MIN_DIST}m) '
                f'— 로봇→공 방향으로 폴백',
                throttle_duration_sec=2.0)

        # ── 모드 A(또는 폴백): 로봇→공 직선상에서 공 앞 standoff ──
        dx, dy = (bx - rx), (by - ry)
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            # 로봇이 공 위에 거의 겹침 — 방향 정의 불가. 현재 위치/임의 방위 반환.
            return rx, ry, 0.0, False
        ux, uy = dx / dist, dy / dist                       # 로봇→공 단위
        gx = bx - self._standoff * ux                       # 공에서 로봇 쪽으로 못미쳐
        gy = by - self._standoff * uy
        gyaw = math.atan2(dy, dx)                            # 공을 향하는 방위
        return gx, gy, gyaw, False

    # =========================================================================
    #  enable 서비스 콜백 — 상위 오케스트레이터가 Phase2 추종을 토글
    # =========================================================================
    def _on_pose_state(self, msg: String) -> None:
        """pose controller 상태 캐시(ARRIVED/BLOCKED 핸드오프 판정 보조)."""
        self._pose_state = msg.data

    def _on_game_stop_lock(self, msg: Bool) -> None:
        locked = bool(msg.data)
        if locked and not self._game_stop_locked:
            self._abort_for_game_stop_lock()
        self._game_stop_locked = locked

    def _abort_for_game_stop_lock(self) -> None:
        """Drop stale approach goal/cache when game_stop_lock turns on."""
        self._enabled = False
        self._set_pose_ctrl(False)
        self._pose_ctrl_on = None
        self._ball_xy = None
        self._ball_stamp = None
        self._pose_state = None
        self._last_sent_goal_xy = None
        self._set_state('IDLE')
        self.get_logger().warn(
            'game_stop_lock ON -> ball_follow goal/cache cleared')

    # =========================================================================
    #  pose controller enable 소유 — 변화 시에만 SetBool 호출(서버 미준비 시 보류)
    # =========================================================================
    def _set_pose_ctrl(self, on: bool) -> None:
        if on and self._game_stop_locked:
            return
        if self._pose_ctrl_on == on:
            return
        if not self._pose_ctrl_client.service_is_ready():
            self.get_logger().warn(
                f"pose_ctrl enable 서비스 '{self._pose_ctrl_srv}' 대기 중 "
                f"— pose_goal_controller 떠있는지 확인",
                throttle_duration_sec=2.0)
            return  # 캐시 갱신 안 함 → 다음 tick 재시도
        req = SetBool.Request()
        req.data = on
        self._pose_ctrl_client.call_async(req)
        self._pose_ctrl_on = on
        self.get_logger().info(f'pose_ctrl enable → {"ON" if on else "OFF"}')

    # =========================================================================
    #  enable 서비스 콜백 — 상위 오케스트레이터가 Phase2 추종을 토글
    # =========================================================================
    def _on_enable(self, request, response):
        want = bool(request.data)
        if want and self._game_stop_locked:
            response.success = False
            response.message = 'game_stop_lock active'
            return response
        if want == self._enabled:
            response.success = True
            response.message = f'already {"enabled" if want else "disabled"}'
            return response
        self._enabled = want
        if not want:
            # 끌 때: 발행 중단 + pose controller OFF(정지·nav 슬롯 비움 → visual_servo 인계).
            self._set_pose_ctrl(False)
            self._last_sent_goal_xy = None
            self._set_state('IDLE')
        else:
            # 켤 때: 직전 접근의 stale 상태 제거(조기 ARRIVED/BLOCKED 오판 방지).
            self._pose_state = None
            self._last_sent_goal_xy = None
        self.get_logger().info(f'~/enable → {"ON" if want else "OFF"}')
        response.success = True
        response.message = 'enabled' if want else 'disabled'
        return response

    # =========================================================================
    #  주기 타이머 — goal 계산/발행/분실 처리/pose_ctrl 토글
    # =========================================================================
    def _on_timer(self) -> None:
        if self._game_stop_locked:
            return
        # 0) enable 게이트 — 꺼져 있으면 아무 것도 하지 않는다(_on_enable 에서 정지 처리).
        if not self._enabled:
            return

        now = self.get_clock().now()

        # 1) 분실 판정 — 공 수신이 끊기면 pose_ctrl OFF(정지) 후 종료.
        fresh = (self._ball_stamp is not None
                 and (now - self._ball_stamp).nanoseconds * 1e-9 <= self._lost_timeout)
        if not fresh:
            self._set_pose_ctrl(False)       # 낡은 공으로 헛걸음 방지(정지)
            self._last_sent_goal_xy = None
            self._set_state('LOST')
            return

        # 2) 로봇 현재 pose(map) 조회 — 방향/도착 판정에 필요.
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, RclpyTime())
            rx = tf.transform.translation.x
            ry = tf.transform.translation.y
        except TransformException as e:
            self.get_logger().warn(
                f'map→{self._base_frame} TF 없음({e}) — 로컬라이제이션 확인. 이번 주기 건너뜀',
                throttle_duration_sec=2.0)
            return

        bx, by = self._ball_xy

        # 3) 목적지 P·방위 계산(모드 A/B 분기).
        gx, gy, gyaw, used_align = self._compute_goal(rx, ry, bx, by)

        # 4) 도착 판정.
        #    · 거리 기반(arrived_dist): 모드 B 는 'P 까지', 모드 A 는 '공 까지' 거리.
        #    · pose controller 가 BLOCKED(장애물로 목표 근처서 막힘)면 '최대한 근접'으로
        #      간주해 ARRIVED 처리 → 상위가 visual_servo 로 인계.
        if used_align:
            arrive_err = math.hypot(gx - rx, gy - ry)
        else:
            arrive_err = math.hypot(bx - rx, by - ry)
        arrived = (arrive_err <= self._arrived_dist) or (self._pose_state == 'BLOCKED')

        if arrived:
            # 도착 → pose_ctrl OFF(standoff 에서 정지). status ARRIVED → 전략이 핸드오프.
            self._set_pose_ctrl(False)
            self._last_sent_goal_xy = None
            self._set_state('ARRIVED')
            self._publish_goal_marker(gx, gy, gyaw)
            self._publish_align_markers(bx, by, gx, gy, used_align)
            return

        # 5) 접근 중 — pose controller ON 보장.
        self._set_pose_ctrl(True)

        # 6) goal 발행: 직전 발행에서 resend_thr 이상 움직였을 때만(토픽 스팸 방지).
        need_send = True
        if self._last_sent_goal_xy is not None:
            lgx, lgy = self._last_sent_goal_xy
            if math.hypot(gx - lgx, gy - lgy) < self._resend_thr:
                need_send = False
        if need_send:
            self._publish_pose_goal(gx, gy, gyaw)
            self._last_sent_goal_xy = (gx, gy)
        self._set_state('FOLLOWING')

        # 7) 시각화 — 현재 goal 화살표 + 골대–공–목적지 정렬선(map 프레임).
        self._publish_goal_marker(gx, gy, gyaw)
        self._publish_align_markers(bx, by, gx, gy, used_align)

    # =========================================================================
    #  standoff 목적지를 PoseStamped 로 발행(pose_goal_controller 가 추종)
    # =========================================================================
    def _publish_pose_goal(self, x: float, y: float, yaw: float) -> None:
        ps = PoseStamped()
        ps.header.frame_id = self._map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation = yaw_to_quaternion(yaw)
        self._goal_pub.publish(ps)

    # =========================================================================
    #  상태/시각화 발행
    # =========================================================================
    def _set_state(self, state: str) -> None:
        if state != self._state:
            self.get_logger().info(f'상태: {self._state} → {state}')
        self._state = state
        msg = String()
        msg.data = state
        self._status_pub.publish(msg)

    def _publish_goal_marker(self, x: float, y: float, yaw: float) -> None:
        """현재 goal 을 화살표 마커로 발행(RViz, map 프레임)."""
        m = Marker()
        m.header.frame_id = self._map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'ball_follow_goal'
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.02
        m.pose.orientation = yaw_to_quaternion(yaw)
        # ARROW: scale.x=길이, y=폭, z=높이
        m.scale.x = 0.30
        m.scale.y = 0.05
        m.scale.z = 0.05
        # 상태별 색: FOLLOWING=청록, ARRIVED=초록 (한눈에 진행 상태 파악)
        if self._state == 'ARRIVED':
            m.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.9)
        else:
            m.color = ColorRGBA(r=0.1, g=0.7, b=0.9, a=0.9)
        m.lifetime = Duration(seconds=max(1.0, 2.0 * self._update_period)).to_msg()
        self._marker_pub.publish(m)

    def _publish_align_markers(self, bx, by, gx, gy, used_align) -> None:
        """골대–공–목적지 정렬선 + 세 지점 구(球)를 MarkerArray 로 발행(RViz).

        align_to_goal=True 로 정렬이 적용됐으면 세 점이 한 직선 위에 놓이므로
        선이 곧게 보인다 → 일직선 정렬을 눈으로 즉시 검증할 수 있다.
        모드 A/폴백일 땐 선을 회색으로 흐리게 표시해 '정렬 미적용'을 구분한다.
        """
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        # 정렬 적용 여부에 따라 선 색을 바꿔 한눈에 구분(적용=자홍, 미적용=회색).
        line_color = (ColorRGBA(r=1.0, g=0.2, b=0.9, a=0.9) if used_align
                      else ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.6))

        # (1) 골대 → 공 → 목적지 LINE_STRIP
        line = Marker()
        line.header.frame_id = self._map_frame
        line.header.stamp = stamp
        line.ns = 'ball_follow_align'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.02  # 선 두께
        line.color = line_color
        line.pose.orientation.w = 1.0
        for px, py in ((self._goal_x, self._goal_y), (bx, by), (gx, gy)):
            pt = Point()
            pt.x = float(px)
            pt.y = float(py)
            pt.z = 0.01
            line.points.append(pt)
        line.lifetime = Duration(seconds=max(1.0, 2.0 * self._update_period)).to_msg()
        arr.markers.append(line)

        # (2) 세 지점 구: 골대(노랑) / 공(주황) / 목적지(초록)
        spheres = (
            (1, self._goal_x, self._goal_y, ColorRGBA(r=1.0, g=0.85, b=0.1, a=0.9)),
            (2, bx, by,                     ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)),
            (3, gx, gy,                     ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.9)),
        )
        for sid, sx, sy, col in spheres:
            s = Marker()
            s.header.frame_id = self._map_frame
            s.header.stamp = stamp
            s.ns = 'ball_follow_align'
            s.id = sid
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.pose.position.x = float(sx)
            s.pose.position.y = float(sy)
            s.pose.position.z = 0.03
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.06
            s.color = col
            s.lifetime = Duration(seconds=max(1.0, 2.0 * self._update_period)).to_msg()
            arr.markers.append(s)

        self._align_pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BallFollowNavNode()
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

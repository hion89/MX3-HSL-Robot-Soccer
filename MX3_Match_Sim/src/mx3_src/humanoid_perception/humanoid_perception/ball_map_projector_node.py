# -*- coding: utf-8 -*-
# =============================================================================
#  ball_map_projector_node.py
#
#  "카메라 광학프레임에서 검출된 공"을 map 프레임 절대 좌표로 변환해
#  (1) RViz 마커  (2) 전략/행동 노드용 토픽 으로 발행하는 노드.
#
#  [무엇을 받고 무엇을 내는가]
#  --------------------------------------------------------------------------
#   입력:
#     - /perception/detections_3d  (vision_msgs/Detection3DArray)
#         · yolo_detector_3d_node 가 발행. 헤더 frame_id 는 보통
#           'camera_color_optical_frame' (x오른쪽, y아래, z전방).
#         · 각 Detection3D.results[0].hypothesis.class_id 에 "이름 문자열"
#           ('ball','goalpost'...)이 들어있다. score 도 같이.
#         · bbox.center.position = 광학프레임 3D 점(X,Y,Z)[m].
#     - TF: map → ... → camera_color_optical_frame
#         · 로컬라이제이션 스택(EKF)이 map→base_link 를, robot_state_publisher 가
#           base_link→카메라 광학프레임을 발행하므로 체인이 완성돼 있어야 한다.
#   출력:
#     - <ball_point_topic>   (geometry_msgs/PointStamped, map 프레임)
#         · 공의 map 좌표. 전략/행동(킥 정렬 등)에서 소비.
#     - <ball_marker_topic>  (visualization_msgs/Marker, map 프레임, SPHERE)
#         · RViz 에서 지도(필드) 위에 공을 띄우는 디버그 마커.
#     - <text_marker_topic>  (visualization_msgs/Marker, map 프레임, TEXT_VIEW_FACING)
#         · 공 좌표를 숫자로 표시하는 디버그 마커. map 좌표와 로봇(base) 기준
#           거리/방위를 함께 띄우고, walkkick OOD 게이트(0.3~1.0m, ±60도)
#           통과 여부를 [kick O/X] 로 보여준다. publish_text_marker=False 로 끔.
#
#  [핵심 원리]
#  --------------------------------------------------------------------------
#   - 광학프레임 점을 직접 map 으로 do_transform_point 한다.
#     축 변환(광학 x오른쪽/y아래/z전방 → map x전/y좌/z상)은 TF 가 알아서 처리하므로
#     이 노드에서 수동으로 부호를 만지지 않는다(실수 원천 제거).
#   - 변환 시각은 "검출 헤더 stamp" 를 쓴다. 그 시점 TF 가 아직 없으면(아주 최근 프레임)
#     최신 TF(Time(0)) 로 폴백한다. 로컬라이저와 동일한 정책으로 통일.
#   - 공은 지면 위에 있으므로 project_to_ground=True 면 map z 를 ball_radius 로 고정한다.
#     (깊이 z 잡음이 "높이"로 새는 것을 차단 — 평면 위치는 그대로 둠)
#   - EMA 로 map 평면 좌표를 평활해 마커 떨림을 줄인다(분실 후 재포착 시 리셋).
#   - lost_timeout 동안 새 검출이 없으면 마커를 DELETE 하고 pose 발행을 멈춘다.
#
#  [정확도에 대한 솔직한 주의]
#  --------------------------------------------------------------------------
#   공의 map 좌표 = (로봇 self-localization 정확도) + (카메라 depth 정확도) 의 합.
#     · 로봇 pose 가 틀어져 있으면 공도 그만큼 통째로 틀어진다(같은 오차 전파).
#     · D405 depth 는 ~1m 넘어가면 부정확 → 먼 공의 map 평면 위치도 부정확.
#   즉 이 노드는 "더 좋은 센서를 만들지 않는다". 있는 정보를 map 으로 옮길 뿐.
#   먼 공은 EMA/ground projection 으로 떨림만 줄지 근본 오차는 남는다 → max_range 로 컷.
#
#  [의도적으로 단순하게 둔 것]
#   - 공은 1개라고 가정하고 "최고 점수 1개"만 추적한다(다중 공/오검출 군집 처리 안 함).
#   - 속도/궤적 추정(칼만 등)은 하지 않는다. 필요해지면 별도 트래커로 확장.
# =============================================================================

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time as RclpyTime

from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

from tf2_ros import Buffer, TransformListener, TransformException
# tf2_geometry_msgs 를 import 하면 PointStamped 에 대한 do_transform 이 등록된다.
from tf2_geometry_msgs import do_transform_point


class BallMapProjectorNode(Node):
    """공 검출(카메라 광학프레임)을 map 프레임으로 투영해 마커/토픽으로 발행."""

    def __init__(self) -> None:
        super().__init__('ball_map_projector_node')

        # ──────────────────────────────────────────────────────────
        # 파라미터 선언 (모두 런치/CLI 에서 덮어쓸 수 있게 노출)
        # ──────────────────────────────────────────────────────────
        # 입력 검출 토픽 — yolo_detector_3d_node 의 출력.
        self.declare_parameter('detections_topic', '/perception/detections_3d')
        # 출력 토픽 — 전략/행동이 구독할 공의 map 좌표.
        self.declare_parameter('ball_point_topic', '/ball/map_point')
        # 출력 토픽 — RViz 마커.
        self.declare_parameter('ball_marker_topic', '/ball/marker')

        # 변환 목표 프레임(지도). 로컬라이저의 map_frame 과 동일해야 함.
        self.declare_parameter('map_frame', 'map')

        # 검출 노드는 class_id 에 '이름 문자열'을 넣는다('ball','goalpost'...).
        self.declare_parameter('ball_class_name', 'ball')
        # 이 점수 미만 검출은 무시(오검출 컷).
        self.declare_parameter('min_score', 0.40)
        # 광학프레임 Z(=깊이) 가 이 값 이하면 깊이 무효(투영 불가) → 버림.
        self.declare_parameter('min_valid_z_m', 0.05)
        # 카메라→공 직선거리가 이 값 초과면 depth 부정확 → 버림(연습장 대각선 ~4.7m).
        self.declare_parameter('max_range_m', 5.0)

        # 공 반지름[m] — 마커 크기 + 지면 투영 높이. RoboCup KidSize 공 ≈ Ø0.08~0.10m.
        self.declare_parameter('ball_radius_m', 0.05)
        # True 면 map z 를 ball_radius 로 고정(공을 필드 평면에 앉힘, 깊이 높이잡음 제거).
        self.declare_parameter('project_to_ground', True)

        # EMA 평활 계수 a∈(0,1]. 새값 비중 a, 과거 비중 (1-a).
        #   1.0 = 평활 끔(즉시 반영), 작을수록 부드럽지만 지연↑. 0.5 가 무난.
        self.declare_parameter('ema_alpha', 0.5)
        # 마지막 검출 후 이 시간(s) 지나면 '분실' → 마커 DELETE, pose 발행 중단.
        self.declare_parameter('lost_timeout_s', 0.5)
        # 분실 감시 타이머 주기[Hz]. 검출 콜백과 별개로 분실/마커수명을 관리.
        self.declare_parameter('check_rate_hz', 10.0)

        # ── TF 조회 / 신선도 (2026-08-20 추가) ──────────────────────────
        #  [배경 — 실측으로 확인된 병목]
        #    검출은 26.6Hz(지연 35ms)로 정상 도착하는데 /ball/map_point 는
        #    15.9Hz 밖에 안 나갔다. 원인은 아래 1순위 TF 조회의 타임아웃이
        #    0.05s 로 하드코딩되어 있던 것.
        #      · 1순위 성공률 실측 0.0% — 매번 50ms 를 꽉 채우고 폴백
        #      · 콜백 1회가 61ms 소요 -> 처리율 1/0.061 = 16.4Hz 로 제한
        #      · 입력 37ms 주기 > 처리 61ms 이므로 큐가 계속 쌓여,
        #        콜백이 꺼낼 때는 이미 ~370ms 낡은 검출이 된다
        #    즉 파이프라인이 느린 게 아니라 이 노드가 스스로 지연을 만들었다.
        #
        #  [왜 1순위가 안 됐나 — 구 구조 기준]
        #    base_link->camera 체인이 목 관절을 지난다. 당시 목 상태는 MCU
        #    시리얼로 들어와 지연이 커서, 넥 joint_states 를 실측 시각으로
        #    크게 back-date 해 발행했다. 그 결과 최신 TF 조차 검출 시각보다
        #    과거라 1순위는 미래 외삽이 되어 항상 실패했다.
        #
        #  [2026-08 목 이관 후 — 전제가 바뀌었다]
        #    목이 U2D2 버스로 합쳐져 body_motor_controller 의 SyncRead 에
        #    실려 50Hz 로 나온다. back-date 폭이 SyncRead 왕복(수 ms)으로
        #    줄어, 이제 TF 버퍼가 검출 시각(≈50ms 과거)을 여유 있게 덮는다.
        #    → 1순위(시각 정렬)가 정상 성공해야 정상이다.
        #    stats_period_s 로그의 "tier1 %" 가 높게 나오는지 반드시 확인할 것.
        #    tier1 이 여전히 0% 라면 목 이관이 제대로 안 붙은 것이므로
        #    /joint_states 의 stamp 와 tf_lookup_timeout_s 를 다시 볼 것.
        #
        #  [해결] 1순위를 '있으면 쓰고 없으면 즉시 폴백'으로 바꾼다.
        #    /tf 가 76Hz 라 최신 TF 는 최대 13ms 낡았을 뿐이고, 검출 35ms 를
        #    더해도 ~50ms 다. 370ms 백로그보다 압도적으로 정확하다.
        #    0 이하로 두면 대기 없이 곧바로 폴백한다.
        self.declare_parameter('tf_lookup_timeout_s', 0.005)
        # 검출 헤더가 이보다 오래됐으면 버린다(0 이하 = 검사 안 함).
        #   낡은 위치로 서보를 돌리면 이미 지나간 자리를 쫓아 진동한다.
        #   차라리 발행하지 않아 LOST 로 떨어지는 편이 안전하다.
        #   정상 지연이 ~50ms 이므로 0.15s 면 충분한 여유다.
        self.declare_parameter('max_detection_age_s', 0.15)
        # 처리 통계를 이 주기[s]로 로그에 남긴다(0 이하 = 끔).
        #   1순위/2순위/실패 비율과 콜백 소요시간을 노드가 스스로 보고한다.
        self.declare_parameter('stats_period_s', 10.0)

        # 마커 색(RGBA). 기본 주황 — 필드 초록과 대비되어 공으로 잘 읽힘.
        self.declare_parameter('marker_color_rgba', [1.0, 0.5, 0.0, 0.9])

        # ── 좌표 텍스트 마커 (RViz 에서 숫자를 바로 읽기 위한 디버그) ──
        # 공 위에 map 좌표와 로봇(base) 기준 좌표를 함께 띄운다.
        # 킥 정렬 검증용: walkkick 러너의 OOD 게이트가 base 기준
        # 거리 0.3~1.0m / 방위 ±60° 이므로, 그 숫자를 화면에서 바로 봐야
        # "지금 트리거해도 되는지" 판단할 수 있다.
        self.declare_parameter('publish_text_marker', True)
        self.declare_parameter('text_marker_topic', '/ball/text_marker')
        # 로봇 기준 좌표를 계산할 프레임. walkkick 러너의 base_frame 과 맞출 것.
        self.declare_parameter('base_frame', 'base_link')
        # 텍스트 크기[m] 와 공 위로 띄울 높이[m].
        self.declare_parameter('text_height_m', 0.08)
        self.declare_parameter('text_offset_z_m', 0.18)

        # ──────────────────────────────────────────────────────────
        # 파라미터 값 캐싱
        # ──────────────────────────────────────────────────────────
        det_topic = self.get_parameter('detections_topic').value
        point_topic = self.get_parameter('ball_point_topic').value
        marker_topic = self.get_parameter('ball_marker_topic').value

        self._map_frame = str(self.get_parameter('map_frame').value)
        self._ball_name = str(self.get_parameter('ball_class_name').value)
        self._min_score = float(self.get_parameter('min_score').value)
        self._min_valid_z = float(self.get_parameter('min_valid_z_m').value)
        self._max_range = float(self.get_parameter('max_range_m').value)
        self._ball_radius = float(self.get_parameter('ball_radius_m').value)
        self._project_ground = bool(self.get_parameter('project_to_ground').value)
        self._ema = float(self.get_parameter('ema_alpha').value)
        self._lost_timeout = float(self.get_parameter('lost_timeout_s').value)
        check_hz = float(self.get_parameter('check_rate_hz').value)
        color = list(self.get_parameter('marker_color_rgba').value)
        self._pub_text = bool(self.get_parameter('publish_text_marker').value)
        text_topic = str(self.get_parameter('text_marker_topic').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        self._text_h = float(self.get_parameter('text_height_m').value)
        self._text_dz = float(self.get_parameter('text_offset_z_m').value)
        self._color = ColorRGBA(
            r=float(color[0]), g=float(color[1]),
            b=float(color[2]), a=float(color[3]))

        # ──────────────────────────────────────────────────────────
        # 내부 상태
        # ──────────────────────────────────────────────────────────
        # 평활된 map 좌표(없으면 None). 분실 후 재포착 시 리셋해 잔상 제거.
        self._ema_x = None
        self._ema_y = None
        self._ema_z = None
        # 마지막으로 '유효 공'을 발행한 시각(rclpy Time). 분실 판정 기준.
        self._last_stamp = None
        # 마커가 현재 RViz 에 떠있는지(=ADD 발행됨). 분실 시 한 번만 DELETE 하려는 플래그.
        self._marker_alive = False

        # ──────────────────────────────────────────────────────────
        # TF / Pub / Sub
        # ──────────────────────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # 출력 publisher
        self._point_pub = self.create_publisher(PointStamped, point_topic, 10)
        self._marker_pub = self.create_publisher(Marker, marker_topic, 10)
        self._text_pub = (self.create_publisher(Marker, text_topic, 10)
                          if self._pub_text else None)

        # 입력 subscriber — 검출 콜백에서 투영/발행 수행.
        self._det_sub = self.create_subscription(
            Detection3DArray, det_topic, self._on_detections, 10)

        # 분실 감시 타이머 — 검출이 끊겨도 주기적으로 돌며 timeout 시 마커를 지운다.
        period = 1.0 / max(1.0, check_hz)
        self._check_timer = self.create_timer(period, self._on_check_lost)

        # ── 처리 통계 (진단용) ──
        #   1순위/2순위/실패 비율과 콜백 소요시간을 주기적으로 보고한다.
        #   프로브 스크립트를 따로 돌리지 않아도 병목이 재발하면 바로 보인다.
        self._tf_timeout = float(
            self.get_parameter('tf_lookup_timeout_s').value)
        self._max_age = float(
            self.get_parameter('max_detection_age_s').value)
        self._stat_n = 0        # 수신 검출 프레임
        self._stat_t1 = 0       # 1순위(시각 정렬) 성공
        self._stat_t2 = 0       # 2순위(최신 TF) 폴백
        self._stat_fail = 0     # TF 조회 실패
        self._stat_stale = 0    # 너무 낡아 버림
        self._stat_cb_s = 0.0   # 콜백 누적 소요시간
        _sp = float(self.get_parameter('stats_period_s').value)
        if _sp > 0.0:
            self._stat_timer = self.create_timer(_sp, self._report_stats)

        # 검출 frame 이 광학프레임이 아닐 때 1회만 경고하기 위한 플래그.
        self._warned_frame = False

        self.get_logger().info(
            f"ball_map_projector 시작 — '{det_topic}' → map='{self._map_frame}', "
            f"ball='{self._ball_name}', min_score={self._min_score}, "
            f"max_range={self._max_range}m, project_to_ground={self._project_ground}")

    # =========================================================================
    #  검출 콜백 — 최고 점수 'ball' 1개를 골라 map 으로 투영해 발행
    # =========================================================================
    def _on_detections(self, msg: Detection3DArray) -> None:
        """검출 콜백 래퍼 — 어느 경로로 return 하든 소요시간을 집계한다.

        조기 return 지점이 여러 곳이라, 각 지점마다 계측 코드를 넣으면
        하나만 빠뜨려도 통계가 조용히 틀어진다. try/finally 로 묶는다.
        """
        _cb_t0 = time.time()
        self._stat_n += 1
        try:
            self._on_detections_impl(msg)
        finally:
            self._stat_cb_s += time.time() - _cb_t0

    def _on_detections_impl(self, msg: Detection3DArray) -> None:
        # ── 신선도 검사 ──
        #   낡은 검출로 서보를 돌리면 이미 지나간 자리를 쫓아 진동한다.
        #   버려서 LOST 로 떨어지는 편이 안전하다.
        if self._max_age > 0.0:
            age = (self.get_clock().now()
                   - RclpyTime.from_msg(msg.header.stamp)).nanoseconds * 1e-9
            if age > self._max_age:
                self._stat_stale += 1
                self.get_logger().warn(
                    f'검출이 {age * 1e3:.0f}ms 낡음(>{self._max_age * 1e3:.0f}ms) '
                    '— 버림. 상류(카메라/YOLO) 지연 또는 큐 적체를 확인하라.',
                    throttle_duration_sec=2.0)
                return
        src_frame = msg.header.frame_id or ''
        if not src_frame:
            # frame_id 없으면 어느 프레임에서 변환할지 알 수 없음 → 건너뜀.
            self.get_logger().warn(
                '검출 메시지에 frame_id 가 없음 — map 변환 불가, 건너뜀',
                throttle_duration_sec=2.0)
            return

        # 광학프레임이 아니면 깊이/축 가정이 깨질 수 있음 — 1회 경고(동작은 계속).
        if (not self._warned_frame) and ('optical' not in src_frame):
            self.get_logger().warn(
                f"검출 frame_id='{src_frame}' 가 광학프레임이 아닐 수 있음. "
                f"TF 변환 자체는 frame 기준으로 수행되지만 깊이 규약을 확인할 것.")
            self._warned_frame = True

        # --- 'ball' 중 최고 점수 1개 선택 ---
        best_p = None
        best_score = -1.0
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            # 검출 노드가 class_id 에 이름 문자열을 넣음('ball' 등).
            if str(hyp.hypothesis.class_id) != self._ball_name:
                continue
            score = float(hyp.hypothesis.score)
            if score < self._min_score:
                continue

            p = det.bbox.center.position   # 광학프레임 3D 점
            # NaN/inf 좌표(브리지 결손/깊이 이상)는 즉시 버린다.
            #   NaN 은 모든 비교가 False 라 'z < thr' 만으론 안 걸러지므로 isfinite 먼저.
            if not (math.isfinite(p.x) and math.isfinite(p.y)
                    and math.isfinite(p.z)):
                continue
            if float(p.z) < self._min_valid_z:
                continue                   # 깊이 무효(투영 불가) → 버림
            # 카메라→공 직선거리(광학프레임 원점 기준)로 원거리 컷.
            rng = math.sqrt(float(p.x) ** 2 + float(p.y) ** 2 + float(p.z) ** 2)
            if rng > self._max_range:
                continue                   # 너무 먼 공 — depth 부정확 → 버림

            if score > best_score:
                best_score = score
                best_p = p

        if best_p is None:
            # 이번 프레임엔 쓸만한 공이 없음. 분실 판정/마커삭제는 타이머가 담당.
            return

        # --- TF: 광학프레임 점 → map 점 ---
        ps = PointStamped()
        ps.header.frame_id = src_frame
        ps.header.stamp = msg.header.stamp
        ps.point.x = float(best_p.x)
        ps.point.y = float(best_p.y)
        ps.point.z = float(best_p.z)

        stamp = RclpyTime.from_msg(msg.header.stamp)
        try:
            # 1순위: 검출 시각의 TF(시간 정렬). 짧은 타임아웃 안에서 대기.
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, src_frame, stamp,
                timeout=Duration(seconds=max(0.0, self._tf_timeout)))
            self._stat_t1 += 1
        except TransformException:
            try:
                # 2순위: 최신 TF 폴백. /tf 가 76Hz 라 최대 13ms 낡았을 뿐이므로
                # 정확도 손실은 무시할 만하다. 1순위를 오래 기다리는 것보다
                # 훨씬 낫다(기다리면 큐가 쌓여 수백 ms 지연이 된다).
                tf = self._tf_buffer.lookup_transform(
                    self._map_frame, src_frame, RclpyTime())
                self._stat_t2 += 1
            except TransformException as e:
                self._stat_fail += 1
                self.get_logger().warn(
                    f'TF {src_frame}->{self._map_frame} 조회 실패({e}) — 이 프레임 건너뜀. '
                    f'(EKF map→base_link 가 살아있는지 확인)',
                    throttle_duration_sec=2.0)
                return

        pm = do_transform_point(ps, tf)   # map 프레임 점
        mx, my, mz = float(pm.point.x), float(pm.point.y), float(pm.point.z)

        # --- 지면 투영(옵션): 높이를 공 반지름으로 고정 ---
        if self._project_ground:
            mz = self._ball_radius

        # --- EMA 평활 (분실 후 재포착이면 리셋) ---
        now = self.get_clock().now()
        fresh = (self._last_stamp is not None
                 and (now - self._last_stamp).nanoseconds * 1e-9 <= self._lost_timeout
                 and self._ema_x is not None)
        if fresh:
            a = self._ema
            self._ema_x = a * mx + (1.0 - a) * self._ema_x
            self._ema_y = a * my + (1.0 - a) * self._ema_y
            self._ema_z = a * mz + (1.0 - a) * self._ema_z
        else:
            # 재포착 첫 프레임: 평활 없이 즉시 반영(이전 위치 잔상 방지).
            self._ema_x, self._ema_y, self._ema_z = mx, my, mz

        self._last_stamp = now

        # --- 발행 (map 프레임) ---
        self._publish_point(msg.header.stamp)
        self._publish_marker(msg.header.stamp, add=True)
        self._publish_text_marker(msg.header.stamp, add=True)
        self._marker_alive = True

        self.get_logger().info(
            f'ball_map: x={self._ema_x:.3f} y={self._ema_y:.3f} '
            f'(score={best_score:.2f})',
            throttle_duration_sec=1.0)

    # =========================================================================
    #  분실 감시 타이머 — timeout 지나면 마커 DELETE + 발행 중단
    # =========================================================================
    def _report_stats(self):
        """처리 통계를 로그로 남긴다(진단용).

        [읽는 법]
          · tier2 가 대부분이어도 정상이다 — 최신 TF 는 최대 13ms 낡았을 뿐.
          · cb 평균이 10ms 를 넘으면 다시 큐가 쌓이기 시작한다는 신호다.
            (입력 26.6Hz = 37ms 주기이므로 그 안에 끝나야 한다)
          · stale 이 늘면 상류(카메라/YOLO)가 실제로 느려진 것이다.
        """
        if self._stat_n == 0:
            return
        n = self._stat_n
        self.get_logger().info(
            f'[proj] {n}건  tier1 {100.0 * self._stat_t1 / n:.0f}%  '
            f'tier2 {100.0 * self._stat_t2 / n:.0f}%  '
            f'fail {100.0 * self._stat_fail / n:.0f}%  '
            f'stale {100.0 * self._stat_stale / n:.0f}%  '
            f'cb평균 {1e3 * self._stat_cb_s / n:.1f}ms')
        self._stat_n = 0
        self._stat_t1 = 0
        self._stat_t2 = 0
        self._stat_fail = 0
        self._stat_stale = 0
        self._stat_cb_s = 0.0

    def _on_check_lost(self) -> None:
        if self._last_stamp is None:
            return  # 아직 한 번도 공을 못 봄 — 할 일 없음.
        age = (self.get_clock().now() - self._last_stamp).nanoseconds * 1e-9
        if age <= self._lost_timeout:
            return  # 아직 '신선' — 유지.
        # 분실: 떠있던 마커가 있으면 한 번만 DELETE 한다.
        if self._marker_alive:
            now_stamp = self.get_clock().now().to_msg()
            self._publish_marker(now_stamp, add=False)
            self._publish_text_marker(now_stamp, add=False)
            self._marker_alive = False
            self.get_logger().info('공 분실 — 마커 삭제', throttle_duration_sec=2.0)
        # PointStamped 는 그냥 발행을 멈춘다(소비측이 stamp 나이로 분실 판정 가능).

    # =========================================================================
    #  발행 헬퍼
    # =========================================================================
    def _publish_point(self, stamp) -> None:
        """공의 map 좌표를 PointStamped 로 발행(전략/행동 소비용)."""
        out = PointStamped()
        out.header.stamp = stamp           # 검출 시각 그대로(소비측 시간정렬용)
        out.header.frame_id = self._map_frame
        out.point.x = float(self._ema_x)
        out.point.y = float(self._ema_y)
        out.point.z = float(self._ema_z)
        self._point_pub.publish(out)

    def _publish_marker(self, stamp, add: bool) -> None:
        """RViz SPHERE 마커 발행. add=False 면 DELETE(분실 시)."""
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self._map_frame
        m.ns = 'ball'
        m.id = 0                            # 단일 공 — id 고정(ADD/DELETE 가 같은 마커 가리킴)
        m.type = Marker.SPHERE

        if not add:
            # 분실 — 마커 제거.
            m.action = Marker.DELETE
            self._marker_pub.publish(m)
            return

        m.action = Marker.ADD
        m.pose.position.x = float(self._ema_x)
        m.pose.position.y = float(self._ema_y)
        m.pose.position.z = float(self._ema_z)
        m.pose.orientation.w = 1.0          # identity (tf2 0벡터 quaternion 경고 방지)
        # 지름 = 2×반지름 (Marker.scale 은 '지름' 단위).
        d = 2.0 * self._ball_radius
        m.scale.x = d
        m.scale.y = d
        m.scale.z = d
        m.color = self._color
        # 수명 = lost_timeout. 노드가 죽어도 RViz 가 자동으로 잔상을 지우게 하는 보강책.
        m.lifetime = Duration(seconds=self._lost_timeout).to_msg()
        self._marker_pub.publish(m)

    def _publish_text_marker(self, stamp, add: bool) -> None:
        """공 좌표를 숫자로 띄우는 TEXT_VIEW_FACING 마커.

        RViz 는 토픽 값을 화면에 숫자로 보여주지 못하므로, 좌표를 직접
        텍스트 마커로 그려준다. 공 위 text_offset_z_m 만큼 띄운다.

        [무엇을 보여주나]
          map  : 필드 절대 좌표 (전략/로컬라이제이션 확인용)
          base : 로봇 기준 좌표 + 거리 + 방위 (킥 정렬 확인용)

        base 기준 값은 walkkick 러너가 트리거 시 freeze 하는 값과 같은
        의미라, 이 숫자를 보고 "지금 트리거하면 OOD 게이트를 통과할지"
        (거리 0.3~1.0m, 방위 ±60°) 를 눈으로 바로 판단할 수 있다.

        base TF 조회에 실패하면 map 좌표만 표시한다 — 로컬라이제이션이
        아직 안 붙은 상태에서도 이 마커는 계속 쓸 수 있어야 한다.
        """
        if self._text_pub is None:
            return

        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self._map_frame
        m.ns = 'ball_text'
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING

        if not add:
            m.action = Marker.DELETE
            self._text_pub.publish(m)
            return

        # ── 로봇(base) 기준 좌표 계산 ──
        base_line = 'base: TF 없음'
        try:
            # 최신 TF 사용. 텍스트는 디버그 표시라 시간정렬을 엄격히 할 필요가 없고,
            # 오히려 stamp 를 맞추려다 조회 실패로 숫자가 사라지는 게 더 불편하다.
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._map_frame, RclpyTime(),
                timeout=Duration(seconds=0.05))
            pt = PointStamped()
            pt.header.frame_id = self._map_frame
            pt.point.x = float(self._ema_x)
            pt.point.y = float(self._ema_y)
            pt.point.z = float(self._ema_z)
            pb = do_transform_point(pt, tf)
            bx, by = float(pb.point.x), float(pb.point.y)
            dist = math.hypot(bx, by)
            bearing = math.degrees(math.atan2(by, bx))
            # 킥 게이트(0.3~1.0m, ±60°) 통과 여부를 기호로 함께 표시.
            ok = (0.30 <= dist <= 1.00) and (abs(bearing) <= 60.0)
            mark = 'O' if ok else 'X'
            base_line = (f'base: x={bx:+.3f} y={by:+.3f}\\n'
                         f'      d={dist:.3f}m  {bearing:+.1f}deg  [kick {mark}]')
        except TransformException:
            pass

        m.action = Marker.ADD
        m.pose.position.x = float(self._ema_x)
        m.pose.position.y = float(self._ema_y)
        m.pose.position.z = float(self._ema_z) + self._text_dz
        m.pose.orientation.w = 1.0
        # TEXT_VIEW_FACING 은 scale.z 만 글자 높이로 쓴다.
        m.scale.z = self._text_h
        m.color = self._color
        m.text = (f'map:  x={float(self._ema_x):+.3f} '
                  f'y={float(self._ema_y):+.3f}\n{base_line}')
        m.lifetime = Duration(seconds=self._lost_timeout).to_msg()
        self._text_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BallMapProjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy.shutdown() 은 런치 컨텍스트에 따라 이미 내려갔을 수 있어 방어적으로 호출.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

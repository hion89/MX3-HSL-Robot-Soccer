#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  field_landmark_localizer_node.py
#
#  D405 YOLO 3D 검출(/perception/detections_3d)을 받아, 경기장 안에서 로봇의
#  "지도(map) 절대 위치"를 측정해 /localization/vision_pose 로 발행하는 노드.
#
#  [이 노드의 책임은 딱 하나 — "측정값 한 줄"]
#  --------------------------------------------------------------------------
#  과거 2번의 실패는 필터/다층 구조의 복잡도에 버그가 숨었던 것이 원인이었다.
#  이번 구조에서 "필터링/융합"은 robot_localization(EKF)이 전부 담당하고,
#  이 노드는 오직 "지금 본 랜드마크들로 추정한 절대 pose"라는 측정값만 낸다.
#  → 이 노드 안에는 칼만필터도, 다층 레이어도 없다. 측정 → 발행. 끝.
#
#  [데이터 흐름]
#     /perception/detections_3d (Detection3DArray, camera_color_optical_frame)
#        │  ① 랜드마크 클래스(goalpost/X/T/L)만 추리고, 깊이 무효(Z<=0) 버림
#        │  ② TF 로 검출 3D 점을 base_link 로 변환
#        │  ③ base_link 기준 (range, bearing) 관측으로 변환
#        ▼
#     LandmarkPoseSolver (Gauss-Newton, 시드 근처 정밀화)
#        │  시드: map→base_link TF(EKF 피드백) > 직전 발행 pose > /initialpose
#        ▼
#     /localization/vision_pose (PoseWithCovarianceStamped, map frame)
#        → robot_localization 의 pose 입력으로 들어감
#
#  [중요 전제]
#   - base_link → camera_color_optical_frame TF 가 살아 있어야 함
#     (humanoid_perception/robot_tf_bringup.launch.py + /joint_states).
#   - 랜드마크 지도는 field_profile(연습장/대회)에서 자동 생성되므로,
#     경기장 전환은 파라미터 field_profile 한 개로 끝난다.
#   - solver 는 "시드 근처 지역 보정기"다(수렴 basin 실측 ±0.12m/±5°).
#     정상 동작 중 시드는 EKF 가 주므로 충분히 가깝고, 시작 시에는 /initialpose
#     또는 initial_* 파라미터로 대략 맞춰 주면 된다.
# =============================================================================

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time as RclpyTime
from rclpy.executors import ExternalShutdownException
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy)

from geometry_msgs.msg import (
    Point, PointStamped, PoseWithCovarianceStamped, Quaternion)
from std_msgs.msg import Bool, ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3DArray
from nav_msgs.msg import Odometry   # 회전속도(yaw rate) 읽어 회전중 비전 게이트용

from tf2_ros import Buffer, TransformListener, TransformException
# tf2_geometry_msgs 를 import 하면 PointStamped 에 대한 do_transform 이 등록된다.
from tf2_geometry_msgs import do_transform_point

from .field_geometry import field_geometry_for_profile
from .field_landmarks import generate_field_landmarks, count_landmarks
from .landmark_pose_solver import (
    Observation, LandmarkSolverConfig, LandmarkPoseSolver, normalize_angle)


# =============================================================================
#  쿼터니언 ↔ yaw 헬퍼 (walk_odometry 와 동일 규약)
# =============================================================================
def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """쿼터니언(x,y,z,w)에서 yaw(Z축 회전, rad)만 추출."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    """yaw(rad)만 갖는 평면 쿼터니언 생성(roll=pitch=0)."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def bearing_range_from_base_point(x: float, y: float):
    """base_link 기준 평면 점(x 전방, y 좌측)에서 (range, bearing) 계산.

    range  = 평면 거리 = hypot(x, y)   [m]
    bearing= 로봇 정면(+x) 기준 방위각 = atan2(y, x)  [rad] (좌 +, 우 -)
    z(높이)는 2D 위치추정에 무관하므로 무시한다(평면 투영).
    """
    return math.hypot(x, y), math.atan2(y, x)


def build_pose_covariance(var_xy: float, var_yaw: float):
    """PoseWithCovariance 의 6x6(36) 공분산(행우선)을 만든다.

    이 측정은 2D(x, y, yaw)만 의미 있으므로:
      - x, y, yaw 에는 실제 분산을 넣고,
      - z, roll, pitch 에는 매우 큰 값(1e6)을 넣어 EKF 가 무시하게 한다.
    인덱스: x=0, y=7, z=14, roll=21, pitch=28, yaw=35 (대각 성분).
    """
    cov = [0.0] * 36
    big = 1e6
    cov[0] = var_xy     # x
    cov[7] = var_xy     # y
    cov[14] = big       # z      (사용 안 함)
    cov[21] = big       # roll   (사용 안 함)
    cov[28] = big       # pitch  (사용 안 함)
    cov[35] = var_yaw   # yaw
    return cov


class FieldLandmarkLocalizerNode(Node):
    """D405 3D 검출 → 지도 절대 pose 측정 노드."""

    def __init__(self):
        super().__init__('field_landmark_localizer_node')

        # =====================================================================
        #  파라미터 선언
        # =====================================================================
        # --- 경기장 프로파일: 연습장("practice") / 대회("competition") ---
        #     랜드마크 지도가 이 프로파일에서 자동 생성된다. 전환은 이 한 개로 끝.
        self.declare_parameter('field_profile', 'practice')

        # --- 토픽 / 프레임 ---
        self.declare_parameter('detections_topic', '/perception/detections_3d')
        self.declare_parameter('pose_topic', '/localization/vision_pose')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        # --- 검출 클래스 이름(문자열) → 랜드마크 class_id(정수) 매핑 ---
        #     yolo_detector_3d_node 는 class_id 를 "이름 문자열"로 채운다.
        #     (class_names = ['ball','goalpost','robot','field_edge','line',
        #                     'X-Intersection','T-Intersection','L-Intersection','obstacle'])
        #     아래 두 리스트를 zip 해 "이름→정수id" 사전을 만든다. 검출기 클래스명이
        #     바뀌면 이 두 파라미터만 맞추면 된다.
        self.declare_parameter(
            'det_class_names',
            ['goalpost', 'X-Intersection', 'T-Intersection', 'L-Intersection'])
        self.declare_parameter('det_class_ids', [1, 5, 6, 7])

        # --- 실제로 앵커로 쓸 랜드마크 클래스(생성된 것 중 일부만 쓰고 싶을 때) ---
        #     예) 골대 높이 편향이 의심되면 [5,6,7] 로 goalpost(1) 제외 가능.
        self.declare_parameter('landmark_classes', [1, 5, 6, 7])

        # --- 검출 필터 ---
        self.declare_parameter('min_score', 0.40)       # 신뢰도 하한(검출기 외 추가 게이트)
        self.declare_parameter('min_valid_z_m', 0.05)   # 깊이 무효(Z<=0) 검출 제거 임계
        self.declare_parameter('max_range_m', 5.0)      # 이 거리 초과 관측은 버림(원거리 부정확)
        # 추정 품질 게이트: 매칭 관측의 평균 재투영 잔차[m] 가 이보다 크면
        # "틀린 추정"으로 보고 발행/시드갱신을 건너뛴다(EKF 는 odom coast).
        # 잘 맞으면 보통 0.05~0.15m, 오연관/오염되면 커진다.
        self.declare_parameter('max_residual_m', 0.25)

        # --- solver 파라미터 (LandmarkSolverConfig 로 전달) ---
        self.declare_parameter('min_observations', 2)
        self.declare_parameter('max_iterations', 6)
        self.declare_parameter('range_gate_m', 1.5)
        self.declare_parameter('bearing_gate_rad', 0.9)
        self.declare_parameter('sigma_range_m', 0.25)
        self.declare_parameter('sigma_bearing_rad', 0.20)
        # 거리 의존 range 가중치: 먼 마커의 거리 측정을 덜 신뢰(σ 팽창).
        # falloff[m] 에서 σ 2배. 0=끔. 작을수록 먼 마커를 강하게 깎음.
        self.declare_parameter('range_sigma_falloff_m', 2.0)
        self.declare_parameter('max_step_xy_m', 0.30)
        self.declare_parameter('max_step_yaw_rad', 0.25)

        # --- 발행 pose 공분산(기본값) 및 발행 게이트 ---
        self.declare_parameter('pose_sigma_xy', 0.05)       # 기본 위치 표준편차[m]
        self.declare_parameter('pose_sigma_yaw', 0.05)      # 기본 방위 표준편차[rad]
        self.declare_parameter('vision_recovery_boost_sec', 4.0)
        self.declare_parameter('vision_recovery_pose_sigma_xy', 0.08)
        self.declare_parameter('vision_recovery_pose_sigma_yaw', 0.06)
        self.declare_parameter('vision_recovery_boost_on_robot_recovery', True)
        self.declare_parameter('vision_recovery_boost_on_kick_done', True)
        self.declare_parameter('robot_state_topic', '/robot_state')
        self.declare_parameter('kick_done_topic', '/kick_done')
        self.declare_parameter('min_trust_matches', 3)      # 이 수 이상이면 기본 신뢰, 미만이면 공분산 팽창
        self.declare_parameter('min_publish_matches', 2)    # 이 수 미만이면 측정 미발행(EKF 가 odom 으로 coast)

        # --- 시드 전략 ---
        #     True 면 map→base_link TF(=EKF 피드백)를 시드로 우선 사용.
        #     EKF 가 아직 없으면 조용히 직전 발행 pose 로 폴백한다(standalone 검증용).
        self.declare_parameter('use_tf_seed', True)
        # /initialpose(2D Pose Estimate) 직후, use_tf_seed 의 stale TF 피드백을
        # 무시하고 "클릭한 pose" 를 강제 시드로 쓸 프레임 수.
        # → vision↔EKF 가 틀린 위치에 함께 잠긴 상태에서 탈출시키는 핵심.
        self.declare_parameter('seed_override_frames', 5)
        # 부트스트랩 초기 pose(첫 프레임 시드). /initialpose 가 오면 그걸로 덮어쓴다.
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)

        # --- 회전속도 게이트(rotation gate) ---
        #   회전 중엔 비전 위치추정이 신뢰 불가:
        #     모션블러 / 랜드마크 매칭 급변 / 보행 틸트 / 필드대칭(미러)으로 yaw 가 튐.
        #   → EKF yaw 가 튀고, 컨트롤러가 좌/우 회전을 번갈아 → 제자리 진동(무한반복).
        #   대책: |yaw rate| 가 임계 초과면 이 프레임 비전 발행을 건너뛰고(시드도 유지)
        #         회전을 직접 측정하는 odom 에 yaw 를 맡긴다(EKF odom coast).
        #         회전이 멈춰 충분히 안정되면 비전이 다시 보정한다.
        self.declare_parameter('odom_topic', '/odom')              # 회전속도 소스
        self.declare_parameter('max_yaw_rate_for_vision', 0.45)    # [rad/s] 평활화된 yaw rate 가 이 값 초과 시 비전 일시정지
        self.declare_parameter('yaw_rate_filter_tau_sec', 0.8)     # yaw rate 저역필터 시간상수[s]. 보행 좌우흔들림(부호교번) 상쇄용
        self.declare_parameter('vision_resume_settle_sec', 0.4)    # 회전 멈춘 뒤 이 시간 더 정지(잔여 흔들림 대기)

        # --- RViz 디버그 시각화(검토용) ---
        #     랜드마크 지도 + 이번 프레임 매칭선 + 관측점 + 추정 화살표를 MarkerArray 로 발행.
        #     실제 카메라 입력으로 "무엇을 보고 어디로 추정했는지"를 눈으로 검증하기 위함.
        self.declare_parameter('debug_markers', True)
        self.declare_parameter('debug_topic', '/localization/vision_debug')

        # =====================================================================
        #  파라미터 읽기
        # =====================================================================
        profile = self.get_parameter('field_profile').value
        self._detections_topic = self.get_parameter('detections_topic').value
        self._pose_topic = self.get_parameter('pose_topic').value
        self._initialpose_topic = self.get_parameter('initialpose_topic').value
        self._map_frame = self.get_parameter('map_frame').value
        self._base_frame = self.get_parameter('base_frame').value

        names = list(self.get_parameter('det_class_names').value)
        ids = list(self.get_parameter('det_class_ids').value)
        # 이름→정수id 사전. 길이 불일치는 명확히 에러.
        if len(names) != len(ids):
            raise RuntimeError(
                f'det_class_names({len(names)}) 와 det_class_ids({len(ids)}) 길이 불일치')
        self._name_to_id = {str(n): int(i) for n, i in zip(names, ids)}

        self._use_classes = set(int(c) for c in self.get_parameter('landmark_classes').value)
        self._min_score = float(self.get_parameter('min_score').value)
        self._min_valid_z = float(self.get_parameter('min_valid_z_m').value)
        self._max_range = float(self.get_parameter('max_range_m').value)
        self._max_residual = float(self.get_parameter('max_residual_m').value)

        self._pose_sigma_xy = float(self.get_parameter('pose_sigma_xy').value)
        self._pose_sigma_yaw = float(self.get_parameter('pose_sigma_yaw').value)
        self._vision_boost_duration = Duration(
            seconds=float(self.get_parameter('vision_recovery_boost_sec').value))
        self._vision_boost_sigma_xy = float(
            self.get_parameter('vision_recovery_pose_sigma_xy').value)
        self._vision_boost_sigma_yaw = float(
            self.get_parameter('vision_recovery_pose_sigma_yaw').value)
        self._vision_boost_on_robot_recovery = bool(
            self.get_parameter('vision_recovery_boost_on_robot_recovery').value)
        self._vision_boost_on_kick_done = bool(
            self.get_parameter('vision_recovery_boost_on_kick_done').value)
        self._robot_state_topic = str(self.get_parameter('robot_state_topic').value)
        self._kick_done_topic = str(self.get_parameter('kick_done_topic').value)
        self._vision_boost_until = None
        self._last_robot_state = None
        self._min_trust_matches = int(self.get_parameter('min_trust_matches').value)
        self._min_publish_matches = int(self.get_parameter('min_publish_matches').value)

        self._use_tf_seed = bool(self.get_parameter('use_tf_seed').value)
        self._seed_override_frames = int(
            self.get_parameter('seed_override_frames').value)

        # --- 회전속도 게이트 상태 ---
        self._max_yaw_rate = float(self.get_parameter('max_yaw_rate_for_vision').value)
        self._yaw_tau = float(self.get_parameter('yaw_rate_filter_tau_sec').value)
        self._resume_settle = Duration(
            seconds=float(self.get_parameter('vision_resume_settle_sec').value))
        self._last_yaw_rate = 0.0          # odom 원시 yaw rate(로그/관찰용)
        self._yaw_rate_ema = 0.0           # 평활화된 yaw rate(게이트 판정에 사용)
        self._last_odom_stamp = None       # EMA dt 계산용 직전 odom 수신 시각
        self._last_rotating_stamp = None   # 마지막으로 "회전중"이라 판정된 시각(settle 계산용)

        # =====================================================================
        #  랜드마크 지도 생성 (경기장 치수 → GP/X/T/L 점 랜드마크)
        # =====================================================================
        geom = field_geometry_for_profile(profile)
        all_landmarks = generate_field_landmarks(geom, landmark_classes=self._use_classes)
        self._landmarks = all_landmarks
        self.get_logger().info(
            f"필드 프로파일='{profile}' (길이 {geom.length_m}m x 폭 {geom.width_m}m), "
            f"랜드마크 {count_landmarks(all_landmarks)}점 생성 "
            f"(클래스 {sorted(all_landmarks.keys())})")

        # =====================================================================
        #  solver 구성
        # =====================================================================
        cfg = LandmarkSolverConfig(
            min_observations=int(self.get_parameter('min_observations').value),
            max_iterations=int(self.get_parameter('max_iterations').value),
            range_gate_m=float(self.get_parameter('range_gate_m').value),
            bearing_gate_rad=float(self.get_parameter('bearing_gate_rad').value),
            sigma_range_m=float(self.get_parameter('sigma_range_m').value),
            sigma_bearing_rad=float(self.get_parameter('sigma_bearing_rad').value),
            range_sigma_falloff_m=float(
                self.get_parameter('range_sigma_falloff_m').value),
            max_step_xy_m=float(self.get_parameter('max_step_xy_m').value),
            max_step_yaw_rad=float(self.get_parameter('max_step_yaw_rad').value),
            # 관측을 base_link 로 변환해 넣으므로 카메라 오프셋은 0.
            camera_x_m=0.0,
            camera_y_m=0.0,
        )
        self._solver = LandmarkPoseSolver(self._landmarks, cfg)

        # =====================================================================
        #  시드 상태 (직전 발행 pose / 부트스트랩 초기값)
        # =====================================================================
        self._last_pose = (
            float(self.get_parameter('initial_x').value),
            float(self.get_parameter('initial_y').value),
            float(self.get_parameter('initial_yaw').value),
        )
        # /initialpose 직후 "강제 시드" 상태:
        #   _seed_override_count > 0 인 동안은 _seed_override_pose 를 시드로 강제 사용.
        self._seed_override_pose = None
        self._seed_override_count = 0
        # 검출 점수 텍스트 마커 정리용: 직전 프레임에 그린 텍스트 개수
        self._prev_score_count = 0

        # =====================================================================
        #  TF / 구독 / 발행
        # =====================================================================
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # 검출 토픽 QoS: 발행측(yolo_detector_3d_node)이 기본(RELIABLE, depth 10)이라 맞춘다.
        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            Detection3DArray, self._detections_topic, self._cb_detections, det_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, self._initialpose_topic, self._cb_initialpose, 10)
        # 회전속도 게이트용 odom 구독(twist.angular.z 만 사용). sensor data QoS(best effort) 로
        # 받아 최신값만 추적 — 위치 융합에는 안 쓰고 "회전중 여부" 판정에만 쓴다.
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._cb_odom, 10)
        self.create_subscription(
            String, self._robot_state_topic, self._cb_robot_state, 10)
        self.create_subscription(
            Bool, self._kick_done_topic, self._cb_kick_done, 10)

        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, self._pose_topic, 10)

        # --- 디버그 마커 발행자(검토용) ---
        self._debug_markers = bool(self.get_parameter('debug_markers').value)
        self._debug_pub = None
        if self._debug_markers:
            # transient_local(latched): RViz 가 늦게 떠도 마지막 마커를 받아 바로 보이게 함
            debug_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self._debug_pub = self.create_publisher(
                MarkerArray, self.get_parameter('debug_topic').value, debug_qos)

        # 클래스별 색(RViz 마커): goalpost=파랑, X=초록, T=주황, L=빨강
        self._class_color = {
            1: ColorRGBA(r=0.10, g=0.45, b=1.00, a=0.85),  # goalpost
            5: ColorRGBA(r=0.20, g=0.90, b=0.25, a=0.85),  # X-Intersection
            6: ColorRGBA(r=1.00, g=0.70, b=0.05, a=0.85),  # T-Intersection
            7: ColorRGBA(r=0.95, g=0.20, b=0.20, a=0.85),  # L-Intersection
        }

        self.get_logger().info(
            f"field_landmark_localizer 시작: 검출='{self._detections_topic}' → "
            f"pose='{self._pose_topic}' (map='{self._map_frame}', base='{self._base_frame}'), "
            f"use_tf_seed={self._use_tf_seed}, debug_markers={self._debug_markers}")

        # 시작 시 랜드마크 지도를 한 번 발행 → 검출 전에도 RViz 에 랜드마크가 보임(latched)
        if self._debug_pub is not None:
            self._publish_debug(self.get_clock().now().to_msg(), est_pose=None, matches=[])

    # =========================================================================
    #  /initialpose: 시작점/재위치 지정 (RViz 2D Pose Estimate 등)
    # =========================================================================
    def _cb_initialpose(self, msg: PoseWithCovarianceStamped):
        """초기/재위치 pose 로 시드를 리셋한다.

        받은 pose 의 frame 이 map 이 아니면 TF 로 map 으로 변환해 반영한다.
        (보통 RViz 는 map 프레임으로 보낸다)
        """
        src = msg.header.frame_id or self._map_frame
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        if src != self._map_frame:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._map_frame, src, RclpyTime())
                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                tq = tf.transform.rotation
                tyaw = yaw_from_quaternion(tq.x, tq.y, tq.z, tq.w)
                c, s = math.cos(tyaw), math.sin(tyaw)
                px, py = tx + c * px - s * py, ty + s * px + c * py
                yaw = normalize_angle(tyaw + yaw)
            except TransformException as e:
                self.get_logger().warn(
                    f'initialpose: {src}->{self._map_frame} 변환 실패({e}). 그대로 사용.')

        self._last_pose = (px, py, yaw)
        # ★ 클릭 pose 를 다음 N 프레임 동안 "강제 시드" 로 설정.
        #   use_tf_seed 의 stale TF(틀린 EKF 위치)를 무시하고 여기서 재수렴시킨다.
        self._seed_override_pose = (px, py, yaw)
        self._seed_override_count = self._seed_override_frames
        self.get_logger().info(
            f'initialpose 적용: x={px:.3f}, y={py:.3f}, yaw={math.degrees(yaw):.1f}deg '
            f'(다음 {self._seed_override_frames}프레임 강제 시드)')

    # =========================================================================
    #  odom: 회전속도(yaw rate) 추적 — 회전 게이트 판정에만 사용
    # =========================================================================
    def _cb_odom(self, msg: Odometry):
        # twist.angular.z = 현재 yaw rate[rad/s]. 위치 융합엔 안 쓰고 회전 여부만 본다.
        wz = float(msg.twist.twist.angular.z)
        self._last_yaw_rate = wz   # 원시값(로그/관찰용)

        # --- 시간상수 기반 저역통과(EMA) ---
        #   보행 좌우흔들림은 부호가 빠르게 교번(평균 0) → EMA 에서 상쇄.
        #   지속 회전은 같은 부호로 쌓임 → EMA 가 회전속도까지 자라남.
        #   → "직진 보행의 출렁임"과 "실제 회전"을 분리한다.
        now = self.get_clock().now()
        if self._last_odom_stamp is None:
            self._yaw_rate_ema = wz
        else:
            dt = (now - self._last_odom_stamp).nanoseconds * 1e-9
            if dt <= 0.0:
                dt = 1e-3  # 비정상 dt 방어
            alpha = dt / (self._yaw_tau + dt)          # 시간상수 → 가중치
            self._yaw_rate_ema += alpha * (wz - self._yaw_rate_ema)
        self._last_odom_stamp = now

        # 평활화된 yaw rate 가 임계 초과면 "회전중" 으로 보고 시각 기록(이후 settle 유지)
        if abs(self._yaw_rate_ema) > self._max_yaw_rate:
            self._last_rotating_stamp = now

    def _cb_robot_state(self, msg: String):
        state = str(msg.data)
        prev = self._last_robot_state
        self._last_robot_state = state
        if (self._vision_boost_on_robot_recovery
                and state == 'NORMAL'
                and prev is not None
                and prev != 'NORMAL'):
            self._trigger_vision_boost('robot_state recovery')

    def _cb_kick_done(self, msg: Bool):
        if self._vision_boost_on_kick_done and bool(msg.data):
            self._trigger_vision_boost('kick_done')

    def _trigger_vision_boost(self, reason: str):
        if self._vision_boost_duration.nanoseconds <= 0:
            return
        now = self.get_clock().now()
        until = now + self._vision_boost_duration
        if (self._vision_boost_until is None
                or until.nanoseconds > self._vision_boost_until.nanoseconds):
            self._vision_boost_until = until
        self.get_logger().info(
            f'vision recovery boost: {reason}, '
            f'{self._vision_boost_duration.nanoseconds * 1e-9:.1f}s '
            f'(sigma_xy={self._vision_boost_sigma_xy:.3f}, '
            f'sigma_yaw={self._vision_boost_sigma_yaw:.3f})')

    def _current_pose_sigmas(self):
        if self._vision_boost_until is not None:
            now = self.get_clock().now()
            if now.nanoseconds < self._vision_boost_until.nanoseconds:
                return self._vision_boost_sigma_xy, self._vision_boost_sigma_yaw, True
            self._vision_boost_until = None
        return self._pose_sigma_xy, self._pose_sigma_yaw, False

    def _is_vision_paused_by_rotation(self):
        """평활화된 회전속도가 임계 초과거나, 회전 멈춘 뒤 settle 시간 이내면 True(=비전 정지)."""
        # ① 평활화 yaw rate 가 임계 초과 → 정지 (보행 출렁임은 EMA 에서 걸러져 여기 안 걸림)
        if abs(self._yaw_rate_ema) > self._max_yaw_rate:
            return True
        # ② 직전 회전 후 settle 시간 이내 → 잔여 흔들림이 가라앉을 때까지 더 정지
        if self._last_rotating_stamp is not None:
            if (self.get_clock().now() - self._last_rotating_stamp) < self._resume_settle:
                return True
        return False

    # =========================================================================
    #  메인: 검출 → 관측 → solver → 발행
    # =========================================================================
    def _cb_detections(self, msg: Detection3DArray):
        if not msg.detections:
            return  # 빈 프레임은 무시(EKF 가 odom 으로 coast)

        # --- ★ 회전속도 게이트 ---
        #   회전중(또는 멈춘 직후 settle 동안)엔 비전 추정이 신뢰 불가 → 이 프레임 미발행.
        #   "시드도 갱신하지 않는다"(_last_pose 유지) → 튄 yaw 가 다음 시드를 오염시키지 않음.
        #   EKF 는 측정 없이 odom 으로 yaw 를 이어가 회전을 매끄럽게 통과한다.
        #   RViz: 비전 화살표를 지워 "회전중 비전 정지"가 눈에 보이게 한다.
        if self._is_vision_paused_by_rotation():
            self.get_logger().info(
                f'회전중(yaw rate EMA={self._yaw_rate_ema:+.2f}>{self._max_yaw_rate:.2f}, '
                f'원시={self._last_yaw_rate:+.2f}) — 비전 일시정지, EKF odom coast',
                throttle_duration_sec=1.0)
            if self._debug_pub is not None:
                # est_pose=None → 추정 화살표 제거(비전이 이번엔 기여 안 함을 표시)
                self._publish_debug(msg.header.stamp, est_pose=None, matches=[])
            return

        src_frame = msg.header.frame_id
        if not src_frame:
            self.get_logger().warn_once('검출 메시지에 frame_id 가 비어있음 — TF 변환 불가')
            return

        # --- 검출 frame → base_link TF 한 번만 조회(메시지 내 모든 검출이 같은 frame) ---
        #     검출 stamp 로 우선 시도(정확), 실패하면 최신(Time())으로 폴백(목 천천히 움직이면 충분).
        stamp = RclpyTime.from_msg(msg.header.stamp)
        tf = None
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, src_frame, stamp, timeout=Duration(seconds=0.05))
        except TransformException:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._base_frame, src_frame, RclpyTime())
            except TransformException as e:
                self.get_logger().warn(
                    f'TF {src_frame}->{self._base_frame} 조회 실패({e}). 이 프레임 건너뜀.',
                    throttle_duration_sec=2.0)
                return

        # --- 각 검출을 base_link 관측(range, bearing)으로 변환 ---
        #   신뢰도 게이트를 range/bearing 계산 "뒤"에 둬서, 통과/탈락 검출 모두를
        #   RViz 에 점수와 함께 표시(min_score 튜닝용)할 수 있게 한다.
        observations = []
        score_debug = []   # (range, bearing, score, accepted) — RViz 점수 텍스트용
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            # class_id 는 "이름 문자열" → 정수 id 로 매핑. 매핑에 없으면(=랜드마크 아님) 스킵.
            name = hyp.hypothesis.class_id
            class_id = self._name_to_id.get(str(name))
            if class_id is None or class_id not in self._use_classes:
                continue  # 랜드마크 클래스가 아니면 표시 의미도 없음 → 스킵
            score = float(hyp.hypothesis.score)

            # 3D 점(검출 frame). 깊이 무효 시 (0,0,0) 으로 오므로 Z<=임계면 버린다.
            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            cz = float(det.bbox.center.position.z)
            if cz <= self._min_valid_z:
                continue  # 깊이 측정 실패 — 위치 계산 불가라 표시도 스킵

            # base_link 로 변환 → range/bearing
            ps = PointStamped()
            ps.header.frame_id = src_frame
            ps.header.stamp = msg.header.stamp
            ps.point.x, ps.point.y, ps.point.z = cx, cy, cz
            pb = do_transform_point(ps, tf)
            rng, brg = bearing_range_from_base_point(pb.point.x, pb.point.y)
            if rng > self._max_range:
                continue  # 너무 먼 관측은 부정확 → 제외

            # ★ 신뢰도 게이트: 통과한 것만 solver 관측으로. RViz 표시는 통과/탈락 모두.
            accepted = (score >= self._min_score)
            score_debug.append((rng, brg, score, accepted))
            if accepted:
                observations.append(
                    Observation(class_id=class_id, range_m=rng, bearing_rad=brg))

        if len(observations) < self._min_publish_matches:
            # 쓸만한 관측이 부족 — 이번 프레임은 측정 미발행
            self.get_logger().debug(
                f'유효 관측 부족({len(observations)}개) — 미발행')
            return

        # --- 시드 결정: TF(EKF 피드백) > 직전 발행 pose ---
        seed = self._get_seed()

        # --- solver: 시드에서 출발해 관측에 맞는 pose 추정 ---
        est, used = self._solver.solve_pose_from_seed(
            seed[0], seed[1], seed[2], observations)
        x, y, yaw = float(est[0]), float(est[1]), normalize_angle(float(est[2]))

        # --- 추정 품질 평가: 매칭쌍 + 평균 재투영 잔차(m) ---
        #   잔차 = 관측이 둔 랜드마크 위치와 실제 랜드마크 위치의 평균 거리.
        #   작을수록 검출들이 추정 pose 와 잘 맞음(품질 좋음).
        final_matches = self._solver.associate_observations(
            np.array([x, y, yaw], dtype=float), observations)
        residual = self._compute_residual(x, y, yaw, final_matches)

        # --- ★ 품질 게이트 ---
        #   매칭 부족 또는 잔차가 크면(=검출들이 서로 안 맞음=틀린 추정) 발행하지 않고
        #   "시드도 갱신하지 않는다". 나쁜 값이 다음 시드를 오염시키는 연쇄를 끊는다.
        #   이때 EKF 는 측정 없이 odom 으로 coast → 부드럽게 유지된다.
        good = (used >= self._min_publish_matches
                and len(final_matches) >= self._min_publish_matches
                and residual <= self._max_residual)

        if not good:
            self.get_logger().warn(
                f'추정 품질 미달 (매칭 {used}, 잔차 {residual:.3f}m '
                f'> {self._max_residual:.3f}) — 미발행/시드유지, EKF odom coast',
                throttle_duration_sec=1.0)
            # 나쁜 추정도 RViz 엔 "빨간" 화살표 + 잔차로 표시(왜 버렸는지 보이게)
            if self._debug_pub is not None:
                self._publish_debug(msg.header.stamp, est_pose=(x, y, yaw),
                                    matches=final_matches, score_debug=score_debug,
                                    residual=residual, accepted=False)
            return

        # --- 통과: 직전 pose 갱신(다음 프레임 시드 연속성) ---
        self._last_pose = (x, y, yaw)

        # --- 공분산: 매칭 적거나 잔차 크면 팽창(덜 신뢰) ---
        match_scale = max(1.0, float(self._min_trust_matches) / max(1, used))
        res_scale = max(1.0, residual / max(1e-3, 0.5 * self._max_residual))
        sigma_xy, sigma_yaw, boost_active = self._current_pose_sigmas()
        var_xy = (sigma_xy ** 2) * match_scale * res_scale
        var_yaw = (sigma_yaw ** 2) * match_scale * res_scale

        # --- 발행 (map 프레임 절대 pose) ---
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp  # 검출 시각 그대로(EKF 시간정렬용)
        out.header.frame_id = self._map_frame
        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation = quaternion_from_yaw(yaw)
        out.pose.covariance = build_pose_covariance(var_xy, var_yaw)
        self._pose_pub.publish(out)

        self.get_logger().info(
            f'vision_pose: x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}deg '
            f'(매칭 {used}/{len(observations)}, 잔차 {residual:.3f}m'
            f'{", boost" if boost_active else ""})',
            throttle_duration_sec=1.0)

        # --- RViz 디버그 마커 (통과: 주황 화살표) ---
        if self._debug_pub is not None:
            self._publish_debug(msg.header.stamp, est_pose=(x, y, yaw),
                                matches=final_matches, score_debug=score_debug,
                                residual=residual, accepted=True)

    # =========================================================================
    #  추정 품질: 매칭 관측의 평균 재투영 잔차[m]
    # =========================================================================
    def _compute_residual(self, x, y, yaw, matches):
        """매칭쌍의 "거리 가중" 평균 재투영 오차[m] 를 반환한다.

        각 관측이 둔 랜드마크 위치와 실제 위치의 거리(오차)를, 그 관측의 신뢰도
        (가까울수록 ↑, 멀수록 ↓)로 가중평균한다. → 먼 마커의 큰(예상된) 오차가
        품질 점수를 부풀려 잘못 REJECT 시키는 것을 막는다.
        매칭이 없으면 inf(=최악).
        """
        if not matches:
            return float('inf')
        sr0 = self._solver.config.sigma_range_m
        wsum = 0.0
        esum = 0.0
        for obs, lm in matches:
            ox = x + obs.range_m * math.cos(yaw + obs.bearing_rad)
            oy = y + obs.range_m * math.sin(yaw + obs.bearing_rad)
            err = math.hypot(ox - lm.x, oy - lm.y)
            # 가중치: (기준σ / 거리의존σ)² → 가까우면 ≈1, 멀수록 작아짐
            sr = self._solver.range_sigma(obs.range_m)
            w = (sr0 / sr) ** 2 if sr > 0 else 1.0
            esum += w * err
            wsum += w
        return esum / wsum if wsum > 0.0 else float('inf')

    # =========================================================================
    #  시드 획득: map→base_link TF 우선, 없으면 직전 발행 pose
    # =========================================================================
    def _get_seed(self):
        """solver 시드(map 프레임 x, y, yaw)를 결정한다.

        우선순위:
          0) /initialpose 직후 N 프레임: 클릭 pose 강제(stale TF 피드백 무시)
          1) use_tf_seed 이고 map→base_link TF 가 있으면 그것(=EKF 융합 결과 반영)
          2) 아니면 직전 발행 pose(연속성). 첫 프레임은 initial_* 파라미터.
        EKF 가 아직 없으면 1)은 조용히 실패하고 2)로 폴백한다(standalone 검증).
        """
        # 0) 클릭 직후 강제 시드 — use_tf_seed 보다 우선해 틀린 잠금에서 탈출시킨다.
        if self._seed_override_count > 0 and self._seed_override_pose is not None:
            self._seed_override_count -= 1
            return self._seed_override_pose

        if self._use_tf_seed:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._map_frame, self._base_frame, RclpyTime())
                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                q = tf.transform.rotation
                tyaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
                return (tx, ty, tyaw)
            except TransformException:
                # EKF 미실행 등으로 map→base_link 가 없으면 폴백(정상 상황일 수 있음)
                self.get_logger().debug('map→base_link TF 없음 — 직전 pose 를 시드로 사용')
        return self._last_pose

    # =========================================================================
    #  RViz 디버그 마커 (검토용) — 실제 카메라 입력을 눈으로 검증하기 위함
    # =========================================================================
    def _publish_debug(self, stamp, est_pose, matches, score_debug=None,
                       residual=None, accepted=True):
        """랜드마크 지도 + 매칭선 + 관측점 + 추정 화살표 + 검출 점수를 MarkerArray 로 발행.

        - landmarks(영구): 솔버가 아는 랜드마크 31점. 클래스별 색. 검출 전에도 보임.
        - matched(1s)    : 추정 로봇 → 매칭된 랜드마크 선. 이번 프레임에 무엇을 썼는지.
        - observed(1s)   : 측정이 그 랜드마크를 둔 위치(추정 pose 기준 range/bearing 투영).
                          실제 랜드마크 구와의 간격 = 잔차(residual). 잘 맞으면 거의 겹친다.
        - robot(1s)      : 추정 pose 위치/방향 화살표.
        - det_score(1s)  : 각 검출의 YOLO 신뢰도 텍스트. 통과=초록, 탈락=빨강.
                          → 실제 데이터로 min_score 임계를 눈으로 튜닝하기 위함.
        (동적 마커는 lifetime 1s — 검출이 끊기면 자동으로 사라진다)
        """
        if self._debug_pub is None:
            return

        arr = MarkerArray()
        forever = Duration(seconds=0).to_msg()   # lifetime 0 = 영구
        ttl = Duration(seconds=1).to_msg()        # 동적 마커 수명 1s

        # A) 랜드마크 지도 — 클래스별 SPHERE_LIST (영구)
        for class_id, lms in self._landmarks.items():
            m = Marker()
            m.header.frame_id = self._map_frame
            m.header.stamp = stamp
            m.ns = 'landmarks'
            m.id = int(class_id)
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.08
            m.color = self._class_color.get(
                class_id, ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.85))
            m.lifetime = forever
            m.pose.orientation.w = 1.0
            for lm in lms:
                m.points.append(Point(x=lm.x, y=lm.y, z=0.02))
            arr.markers.append(m)

        # 시작 시(est_pose=None)에는 랜드마크만, 그 외엔 동적 마커도 추가
        if est_pose is not None:
            x, y, yaw = est_pose

            # B) 매칭선 (LINE_LIST, cyan, 1s)
            line = Marker()
            line.header.frame_id = self._map_frame
            line.header.stamp = stamp
            line.ns = 'matched'
            line.id = 0
            line.type = Marker.LINE_LIST
            line.action = Marker.ADD
            line.scale.x = 0.012
            line.color = ColorRGBA(r=0.0, g=0.9, b=0.9, a=0.9)
            line.lifetime = ttl
            line.pose.orientation.w = 1.0

            # C) 관측점 (SPHERE_LIST, magenta, 1s)
            obsm = Marker()
            obsm.header.frame_id = self._map_frame
            obsm.header.stamp = stamp
            obsm.ns = 'observed'
            obsm.id = 0
            obsm.type = Marker.SPHERE_LIST
            obsm.action = Marker.ADD
            obsm.scale.x = obsm.scale.y = obsm.scale.z = 0.07
            obsm.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)
            obsm.lifetime = ttl
            obsm.pose.orientation.w = 1.0

            for obs, lm in matches:
                # 로봇 → 실제 랜드마크 선
                line.points.append(Point(x=x, y=y, z=0.02))
                line.points.append(Point(x=lm.x, y=lm.y, z=0.02))
                # 측정이 둔 랜드마크 위치 = 추정 pose 에서 (range, bearing) 투영
                ox = x + obs.range_m * math.cos(yaw + obs.bearing_rad)
                oy = y + obs.range_m * math.sin(yaw + obs.bearing_rad)
                obsm.points.append(Point(x=ox, y=oy, z=0.02))

            arr.markers.append(line)
            arr.markers.append(obsm)

            # D) 추정 화살표 (ARROW, orange, 1s)
            arrow = Marker()
            arrow.header.frame_id = self._map_frame
            arrow.header.stamp = stamp
            arrow.ns = 'robot'
            arrow.id = 0
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.025   # 화살대 지름
            arrow.scale.y = 0.05    # 화살촉 지름
            # 통과=주황, 품질 미달(잔차 큼)=빨강 → 한눈에 "지금 추정을 쓰는지" 보임
            arrow.color = (ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.95) if accepted
                           else ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.95))
            arrow.lifetime = ttl
            arrow.pose.orientation.w = 1.0
            arrow.points.append(Point(x=x, y=y, z=0.03))
            arrow.points.append(Point(
                x=x + 0.25 * math.cos(yaw), y=y + 0.25 * math.sin(yaw), z=0.03))
            arr.markers.append(arrow)

            # D2) 잔차 텍스트 (추정 화살표 위) — 추정 품질 + 채택 여부 한눈에.
            #     통과=주황 "res=...", 탈락=빨강 "res=... REJECT".
            if residual is not None:
                rt = Marker()
                rt.header.frame_id = self._map_frame
                rt.header.stamp = stamp
                rt.ns = 'residual'
                rt.id = 0
                rt.type = Marker.TEXT_VIEW_FACING
                rt.action = Marker.ADD
                rt.scale.z = 0.10   # 글자 높이[m]
                rt.color = (ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.95) if accepted
                            else ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.95))
                rt.lifetime = ttl
                rt.pose.position.x = x
                rt.pose.position.y = y
                rt.pose.position.z = 0.32
                rt.pose.orientation.w = 1.0
                res_txt = f'{residual:.3f}' if residual != float('inf') else 'inf'
                rt.text = f'res={res_txt}' + ('' if accepted else ' REJECT')
                arr.markers.append(rt)

            # E) 검출 신뢰도 텍스트 (TEXT_VIEW_FACING, 1s) — min_score 튜닝용.
            #    숫자 = YOLO score. 통과(>=min_score)=초록, 탈락=빨강.
            #    추정 pose 기준 (range, bearing) 으로 map 에 투영해 그 자리에 표시.
            n_score = 0
            if score_debug:
                for i, (rng, brg, score, accepted) in enumerate(score_debug):
                    tx = x + rng * math.cos(yaw + brg)
                    ty = y + rng * math.sin(yaw + brg)
                    t = Marker()
                    t.header.frame_id = self._map_frame
                    t.header.stamp = stamp
                    t.ns = 'det_score'
                    t.id = i
                    t.type = Marker.TEXT_VIEW_FACING
                    t.action = Marker.ADD
                    t.scale.z = 0.09   # 글자 높이[m]
                    t.color = (ColorRGBA(r=0.1, g=1.0, b=0.1, a=0.95) if accepted
                               else ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.95))
                    t.lifetime = ttl
                    t.pose.position.x = tx
                    t.pose.position.y = ty
                    t.pose.position.z = 0.15
                    t.pose.orientation.w = 1.0
                    t.text = f'{score:.2f}'
                    arr.markers.append(t)
                n_score = len(score_debug)

            # 이번 프레임에 안 쓴 직전 텍스트 id 는 DELETE 로 정리(스테일 숫자 방지)
            for i in range(n_score, self._prev_score_count):
                d = Marker()
                d.header.frame_id = self._map_frame
                d.header.stamp = stamp
                d.ns = 'det_score'
                d.id = i
                d.action = Marker.DELETE
                arr.markers.append(d)
            self._prev_score_count = n_score

        self._debug_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FieldLandmarkLocalizerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # 종료 시퀀스 중 executor race — context 가 이미 내려갔으면 무해하게 삼킴
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

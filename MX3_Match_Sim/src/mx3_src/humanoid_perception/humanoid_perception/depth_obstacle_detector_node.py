#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_obstacle_detector_node.py
==========================================================================
깊이(depth) 이미지 기반 기하 장애물 검출 — YOLO/학습과 완전 독립.

[원리 — 사용자 직관 그대로]
    "바닥은 거리에 따라 깊이가 매끄럽게(선형적으로) 증가하고, 장애물은 그
     자리에서 앞으로 튀어나와(=깊이가 갑자기 가까워져) 바닥 위로 솟는다."
    → 깊이를 3D 점으로 펴서 base_link(몸 기준)로 옮기면, 바닥은 낮은 z 로
      이어지고 장애물은 그 위로 솟은 점들이 된다. "바닥보다 일정 이상 높은 점"
      을 장애물로 본다.

[YOLO 대비 장점]
    - 클래스 무관: 넘어진 로봇, 심판 다리, 처음 보는 물체도 잡는다.
    - 가까워져도 안 사라짐: 바로 앞 바닥이 안 보이면 그게 곧 장애물 신호.
    학습 검출(YOLO)의 "가까우면 실명" 약점을 메우는 보완 센서다.

[잔디 lean 대응 — ★]
    바닥을 "수평 평면(z=상수)" 으로 가정하면 카메라가 기울거나 잔디가 ~7°
    기울어진 경우 먼 바닥이 장애물로 오검출된다. 그래서 바닥을 **전방거리(bx)에
    대한 선** 으로 피팅한다: z_floor = m·bx + b. (= 사용자가 말한 "바닥은 선형")
    각 거리 구간의 낮은 분위수(percentile)를 바닥으로 잡아 장애물 점에 안 휩쓸리게.
    그 선보다 height_threshold 이상 솟은 점이 장애물.

[성능]
    depth 를 step_px 간격으로 다운샘플 후 numpy 벡터 연산 → Orin 부하 최소화.

[이 노드의 범위 — 검출 전용]
    지금은 회피에 연결하지 않는다. RViz 마커로 "검출이 잘 되는지" 만 먼저 본다:
      /depth_obstacle/markers (MarkerArray):
        · 장애물 점들(빨강 POINTS)
        · (옵션) 바닥 점들(초록, show_ground)
        · 전방 회랑(파랑 사각형)
        · 회랑 내 최근접 장애물(주황 구)
    검출이 확인되면 그 다음에 회피로 연결한다(별도 단계).

[입력]
    depth_topic       sensor_msgs/Image 16UC1 (mm), aligned_depth_to_color
    camera_info_topic sensor_msgs/CameraInfo (K: fx,fy,cx,cy)
    TF                depth frame(camera_color_optical_frame) → base_link
==========================================================================
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Vector3Stamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import (
    Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose,
)

from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener, TransformException


def _quat_to_rot(x, y, z, w):
    """쿼터니언 → 3x3 회전행렬(numpy)."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float32)


class DepthObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__("depth_obstacle_detector_node")

        # ── 토픽/프레임 ──
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("markers_topic", "/depth_obstacle/markers")
        self.declare_parameter("target_frame", "base_link")

        # ── 회피 연결용 검출 출력 ──
        #   회랑 최근접 장애물을 Detection3DArray(base_link)로 발행 → 회피 노드가
        #   detections_topic 만 이쪽으로 바꿔 그대로 먹는다(회피 노드 코드 무수정).
        self.declare_parameter("detections_topic", "/depth_obstacle/detections")
        #   회피 노드의 avoid_class_names 와 맞출 클래스명. 기본 'obstacle'.
        self.declare_parameter("obstacle_class", "obstacle")
        #   최근접 대표점: 최근접 x 부터 이 폭(m) 안의 점들 median 으로 안정화.
        self.declare_parameter("nearest_band_m", 0.15)

        # ── 다운샘플(성능) ──
        # depth 를 이 픽셀 간격으로 솎아낸다. 클수록 가볍고 거칠다.
        self.declare_parameter("step_px", 8)

        # ── 전방거리/높이 임계 ──
        self.declare_parameter("min_range", 0.1)   # 카메라 depth 최소 유효(m)
        self.declare_parameter("max_range", 0.5)   # 최대 처리 거리(m) — 단거리 범퍼
        # ※예전 불안정의 주범은 max_range 가 길어(2.5m) 먼 잔디·진동 노이즈까지 본 것.
        #   0.5m 단거리면 가까운 정확한 깊이만 봐서 보행 중에도 안정적이다.
        # 바닥 선보다 이만큼 솟으면 장애물(잔디/노이즈 위로). 아래는 무시.
        self.declare_parameter("min_obstacle_height", 0.06)
        self.declare_parameter("max_obstacle_height", 1.0)  # 천장/팔 등 제외

        # ── 바닥 선 피팅(잔디 lean 대응) ──
        self.declare_parameter("ground_bin_m", 0.2)       # 거리 구간 폭(m)
        self.declare_parameter("ground_percentile", 20.0)  # 구간 내 바닥 분위수
        self.declare_parameter("ground_min_pts", 10)       # 구간 유효 최소 점수

        # ── 회랑(시각화/최근접) ──
        self.declare_parameter("corridor_half", 0.20)
        # ── 회피 힌트(pose_goal_controller 가 소비하는 Vector3Stamped) ──
        #   /obstacle/avoid_hint: x=전방 최근접거리[m], y=±회피방향(+1=좌로 비킴), z=0
        #   (기존 nearest 결과를 그대로 단순화해 발행 — pose controller 단일권한 합성용)
        self.declare_parameter("hint_topic", "/obstacle/avoid_hint")
        # 장애물 횡중심 |y|<이면 좌우 모호 → 직전 방향 유지(부호 떨림 방지)[m]
        self.declare_parameter("hint_center_deadband", 0.05)
        # 첫 검출/중앙 모호일 때 기본 회피쪽(+1=좌, -1=우)
        self.declare_parameter("hint_default_dir", 1.0)

        # ── 공간 밀집도 필터(★ 바닥 순간 튐 차단) ──
        #   바닥 튐은 외딴 점 몇 개로 나타난다. 최근접 대표점 주변 density_radius
        #   안에 장애물 점이 min_cluster_points 개 미만이면 "외딴 튐" 으로 보고
        #   인정하지 않는다(발행/마커 둘 다 안 함). 진짜 장애물은 면이라 밀집.
        self.declare_parameter("density_radius", 0.15)
        self.declare_parameter("min_cluster_points", 8)

        # ── 마커/동작 ──
        self.declare_parameter("max_points", 2000)   # 마커 점 상한(과다 방지)
        self.declare_parameter("point_size", 0.02)
        self.declare_parameter("show_ground", False)  # 바닥 점도 그릴지(튜닝용)
        self.declare_parameter("tf_timeout", 0.1)
        self.declare_parameter("debug_markers", True)

        self._load_params()
        self.add_on_set_parameters_callback(self._on_param_update)

        # ── intrinsics 캐시(CameraInfo 도착 전엔 None) ──
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

        self._bridge = CvBridge()

        # ── TF ──
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── QoS: 카메라 토픽은 BEST_EFFORT (yolo_3d 와 동일) ──
        cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_info, cam_qos)
        self.create_subscription(
            Image, self.depth_topic, self._on_depth, cam_qos)

        self.pub_markers = self.create_publisher(
            MarkerArray, self.markers_topic, 10)
        self.pub_det = self.create_publisher(
            Detection3DArray, self.detections_topic, 10)
        self.pub_hint = self.create_publisher(
            Vector3Stamped, self.hint_topic, 10)

        self.get_logger().info(
            f"depth_obstacle_detector_node 시작: {self.depth_topic} -> "
            f"{self.markers_topic} (frame={self.target_frame})")
        self.get_logger().info(
            f"  step_px={self.step_px}  range=[{self.min_range:.2f},"
            f"{self.max_range:.2f}]m  h_thresh={self.min_obstacle_height:.2f}m")

    def _load_params(self):
        g = self.get_parameter
        self.depth_topic = str(g("depth_topic").value)
        self.camera_info_topic = str(g("camera_info_topic").value)
        self.markers_topic = str(g("markers_topic").value)
        self.target_frame = str(g("target_frame").value)
        self.detections_topic = str(g("detections_topic").value)
        self.hint_topic = str(g("hint_topic").value)
        self.hint_deadband = float(g("hint_center_deadband").value)
        self.hint_default_dir = float(g("hint_default_dir").value)
        self._last_hint_dir = self.hint_default_dir  # 방향 유지(떨림 방지)
        self.obstacle_class = str(g("obstacle_class").value)
        self.nearest_band_m = float(g("nearest_band_m").value)
        self.step_px = int(g("step_px").value)
        self.min_range = float(g("min_range").value)
        self.max_range = float(g("max_range").value)
        self.min_obstacle_height = float(g("min_obstacle_height").value)
        self.max_obstacle_height = float(g("max_obstacle_height").value)
        self.ground_bin_m = float(g("ground_bin_m").value)
        self.ground_percentile = float(g("ground_percentile").value)
        self.ground_min_pts = int(g("ground_min_pts").value)
        self.corridor_half = float(g("corridor_half").value)
        self.density_radius = float(g("density_radius").value)
        self.min_cluster_points = int(g("min_cluster_points").value)
        self.max_points = int(g("max_points").value)
        self.point_size = float(g("point_size").value)
        self.show_ground = bool(g("show_ground").value)
        self.tf_timeout = float(g("tf_timeout").value)
        self.debug_markers = bool(g("debug_markers").value)

    def _on_param_update(self, params):
        from rcl_interfaces.msg import SetParametersResult
        # 토픽/프레임 외 튜닝값만 실시간 반영(재구독 필요한 건 재시작).
        keys = {
            "step_px": ("step_px", int),
            "min_range": ("min_range", float),
            "max_range": ("max_range", float),
            "min_obstacle_height": ("min_obstacle_height", float),
            "max_obstacle_height": ("max_obstacle_height", float),
            "ground_bin_m": ("ground_bin_m", float),
            "ground_percentile": ("ground_percentile", float),
            "ground_min_pts": ("ground_min_pts", int),
            "corridor_half": ("corridor_half", float),
            "density_radius": ("density_radius", float),
            "min_cluster_points": ("min_cluster_points", int),
            "max_points": ("max_points", int),
            "point_size": ("point_size", float),
            "show_ground": ("show_ground", bool),
            "tf_timeout": ("tf_timeout", float),
            "debug_markers": ("debug_markers", bool),
        }
        for p in params:
            if p.name in keys:
                attr, cast = keys[p.name]
                setattr(self, attr, cast(p.value))
        self.get_logger().info(
            f"[param 갱신] h_thresh={self.min_obstacle_height:.2f}  "
            f"max_range={self.max_range:.2f}  show_ground={self.show_ground}")
        return SetParametersResult(successful=True)

    def _on_info(self, msg: CameraInfo):
        """intrinsics 캐시. K = [fx 0 cx; 0 fy cy; 0 0 1]."""
        k = msg.k
        if k[0] <= 0.0 or k[4] <= 0.0:
            return
        self._fx = float(k[0])
        self._fy = float(k[4])
        self._cx = float(k[2])
        self._cy = float(k[5])

    def _lookup_tf(self, target, source, stamp):
        timeout = Duration(seconds=self.tf_timeout)
        try:
            return self._tf_buffer.lookup_transform(target, source, stamp, timeout)
        except TransformException:
            pass
        try:
            return self._tf_buffer.lookup_transform(target, source, Time(), timeout)
        except TransformException as ex:
            self.get_logger().warn(
                f"TF 변환 실패 {target}<-{source}: {ex}", throttle_duration_sec=2.0)
            return None

    # =====================================================================
    #  깊이 콜백 — 역투영 → base_link 변환 → 바닥선 피팅 → 장애물 분류
    # =====================================================================
    def _on_depth(self, msg: Image):
        if self._fx is None:
            return  # intrinsics 아직 없음

        src_frame = msg.header.frame_id
        if not src_frame:
            return
        tf = self._lookup_tf(self.target_frame, src_frame, msg.header.stamp)
        if tf is None:
            return

        # ── depth → numpy (mm) → m, 다운샘플 ──
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as ex:  # noqa: BLE001
            self.get_logger().warn(f"depth 변환 실패: {ex}", throttle_duration_sec=2.0)
            return
        h, w = depth.shape
        s = max(self.step_px, 1)
        ds = depth[::s, ::s].astype(np.float32) / 1000.0  # m
        vs = np.arange(0, h, s, dtype=np.float32)[:, None]
        us = np.arange(0, w, s, dtype=np.float32)[None, :]
        U = np.broadcast_to(us, ds.shape)
        V = np.broadcast_to(vs, ds.shape)

        # 유효 depth (카메라 z = 전방거리).
        valid = (ds > self.min_range) & (ds < self.max_range)
        if not np.any(valid):
            self._publish_markers(None, None, None, None, msg.header.stamp)
            self._publish_detections(None, msg.header.stamp)
            return
        z = ds[valid]
        u = U[valid]
        v = V[valid]

        # ── 역투영(카메라 광학 frame): X=오른쪽, Y=아래, Z=전방 ──
        xc = (u - self._cx) * z / self._fx
        yc = (v - self._cy) * z / self._fy
        pts_cam = np.stack([xc, yc, z], axis=1)  # (N,3)

        # ── base_link 로 변환 (p_base = R·p_cam + t) ──
        tr = tf.transform.translation
        ro = tf.transform.rotation
        R = _quat_to_rot(ro.x, ro.y, ro.z, ro.w)
        t = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
        pts_base = pts_cam @ R.T + t
        bx = pts_base[:, 0]   # 전방(+)
        bz = pts_base[:, 2]   # 높이(+위)

        # 전방 작업 범위로 제한.
        work = (bx > self.min_range) & (bx < self.max_range)
        if not np.any(work):
            self._publish_markers(None, None, None, None, msg.header.stamp)
            self._publish_detections(None, msg.header.stamp)
            return

        # ── 바닥선 피팅: z_floor = m·bx + b (거리 구간별 낮은 분위수) ──
        m_g, b_g = self._fit_ground_line(bx[work], bz[work])
        ground_at = m_g * bx + b_g
        height = bz - ground_at  # 바닥선 위로 솟은 높이

        # ── 장애물 분류: 바닥선 위 [min,max] 높이 + 작업범위 ──
        obst = work & (height > self.min_obstacle_height) \
            & (height < self.max_obstacle_height)

        obst_pts = pts_base[obst]
        ground_pts = pts_base[work & ~obst] if self.show_ground else None

        # ── 회랑 내 최근접 장애물(대표점) ──
        #   단일 최근접 픽셀은 노이즈가 크므로, 최근접 x 부터 nearest_band 안의
        #   회랑 점들 median 을 대표 위치로 써서 안정화한다.
        nearest = None
        if obst_pts.shape[0] > 0:
            oy = obst_pts[:, 1]
            corr = np.abs(oy) < self.corridor_half
            if np.any(corr):
                cpts = obst_pts[corr]
                x_min = float(np.min(cpts[:, 0]))
                band = cpts[cpts[:, 0] <= x_min + self.nearest_band_m]
                cand = np.median(band, axis=0)  # (x, y, z) 대표점 후보

                # ── 공간 밀집도 필터(★ 바닥 튐 차단) ──
                #   대표점 주변 density_radius(xy) 안의 장애물 점 개수를 센다.
                #   min_cluster_points 미만이면 외딴 튐으로 보고 기각.
                dx = obst_pts[:, 0] - cand[0]
                dy = obst_pts[:, 1] - cand[1]
                n_near = int(np.count_nonzero(
                    dx * dx + dy * dy <= self.density_radius ** 2))
                if n_near >= self.min_cluster_points:
                    nearest = cand
                else:
                    self.get_logger().debug(
                        f"외딴 튐 기각: 대표점 주변 {n_near}개 "
                        f"< {self.min_cluster_points}")

        if self.debug_markers:
            self._publish_markers(obst_pts, ground_pts, nearest, None,
                                  msg.header.stamp)

        # ── 회피 연결용 Detection3DArray 발행(base_link, 항상 발행) ──
        #   최근접 장애물 1개를 담거나(있을 때), 없으면 빈 배열.
        #   빈 배열도 보내야 회피 노드 timeout 이 갱신되어 깔끔히 0 으로 감쇠.
        self._publish_detections(nearest, msg.header.stamp)

        if nearest is not None:
            self.get_logger().info(
                f"전방 최근접 장애물: x={nearest[0]:.2f} y={nearest[1]:.2f}m "
                f"(장애물 점 {obst_pts.shape[0]}개)",
                throttle_duration_sec=1.0)

    def _fit_ground_line(self, fwd, hgt):
        """전방거리(fwd) 대비 바닥 높이(hgt) 선형 피팅. 구간별 낮은 분위수 사용."""
        edges = np.arange(self.min_range, self.max_range + self.ground_bin_m,
                          self.ground_bin_m)
        cs, fs = [], []
        for i in range(len(edges) - 1):
            sel = (fwd >= edges[i]) & (fwd < edges[i + 1])
            if np.count_nonzero(sel) >= self.ground_min_pts:
                cs.append(0.5 * (edges[i] + edges[i + 1]))
                fs.append(np.percentile(hgt[sel], self.ground_percentile))
        if len(cs) >= 2:
            A = np.vstack([np.array(cs), np.ones(len(cs))]).T
            m_g, b_g = np.linalg.lstsq(A, np.array(fs), rcond=None)[0]
            return float(m_g), float(b_g)
        # 구간이 부족하면 평면(상수) 폴백.
        b_g = float(np.percentile(hgt, self.ground_percentile)) if hgt.size else 0.0
        return 0.0, b_g

    # =====================================================================
    #  RViz 마커
    # =====================================================================
    def _publish_markers(self, obst_pts, ground_pts, nearest, _unused, stamp):
        arr = MarkerArray()
        frame = self.target_frame

        clear = Marker()
        clear.header.frame_id = frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        # (1) 바닥 점(옵션, 초록) — 튜닝용.
        if ground_pts is not None and ground_pts.shape[0] > 0:
            arr.markers.append(self._points_marker(
                ground_pts, 10, frame, stamp,
                ColorRGBA(r=0.2, g=0.8, b=0.3, a=0.5)))

        # (2) 장애물 점(빨강).
        if obst_pts is not None and obst_pts.shape[0] > 0:
            arr.markers.append(self._points_marker(
                obst_pts, 11, frame, stamp,
                ColorRGBA(r=0.95, g=0.2, b=0.2, a=0.9)))

        # (3) 전방 회랑(파랑 사각형).
        corr = Marker()
        corr.header.frame_id = frame
        corr.header.stamp = stamp
        corr.ns = "depth_obst"
        corr.id = 12
        corr.type = Marker.LINE_STRIP
        corr.action = Marker.ADD
        corr.scale.x = 0.01
        corr.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.8)
        d = self.max_range
        c = self.corridor_half
        corr.points = [
            Point(x=self.min_range, y=-c, z=0.0),
            Point(x=d, y=-c, z=0.0),
            Point(x=d, y=c, z=0.0),
            Point(x=self.min_range, y=c, z=0.0),
            Point(x=self.min_range, y=-c, z=0.0),
        ]
        arr.markers.append(corr)

        # (4) 회랑 내 최근접 장애물(주황 구).
        near = Marker()
        near.header.frame_id = frame
        near.header.stamp = stamp
        near.ns = "depth_obst"
        near.id = 13
        if nearest is not None:
            near.type = Marker.SPHERE
            near.action = Marker.ADD
            near.pose.position.x = float(nearest[0])
            near.pose.position.y = float(nearest[1])
            near.pose.position.z = float(nearest[2])
            near.pose.orientation.w = 1.0
            near.scale.x = near.scale.y = near.scale.z = 0.12
            near.color = ColorRGBA(r=1.0, g=0.6, b=0.1, a=0.95)
        else:
            near.action = Marker.DELETE
        arr.markers.append(near)

        # (5) 상태 텍스트(BLOCK 거리·회피쪽 / clear) — 한눈에 보기용.
        txt = Marker()
        txt.header.frame_id = frame
        txt.header.stamp = stamp
        txt.ns = "depth_obst"
        txt.id = 14
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = self.max_range
        txt.pose.position.z = 0.25
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.07
        txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        if nearest is not None:
            y = float(nearest[1])
            if abs(y) < self.hint_deadband:
                ddir = self._last_hint_dir
            else:
                ddir = -1.0 if y > 0.0 else 1.0
            side = "좌로" if ddir > 0 else "우로"
            txt.text = f"BLOCK {nearest[0]:.2f}m  {side} 회피"
        else:
            ddir = 0.0
            txt.text = "clear"
        arr.markers.append(txt)

        # (6) 회피 방향 화살표(막혔을 때 비킬 쪽).
        av = Marker()
        av.header.frame_id = frame
        av.header.stamp = stamp
        av.ns = "depth_obst"
        av.id = 15
        if nearest is not None and ddir != 0.0:
            av.type = Marker.ARROW
            av.action = Marker.ADD
            av.points = [Point(x=0.15, y=0.0, z=0.05),
                         Point(x=0.15, y=float(0.3 * ddir), z=0.05)]
            av.scale.x, av.scale.y, av.scale.z = 0.03, 0.06, 0.08
            av.color = ColorRGBA(r=1.0, g=0.3, b=0.1, a=0.95)
            av.pose.orientation.w = 1.0
        else:
            av.action = Marker.DELETE
        arr.markers.append(av)

        self.pub_markers.publish(arr)

    def _publish_detections(self, nearest, stamp):
        """회랑 최근접 장애물을 Detection3DArray(base_link)로 발행. 없으면 빈 배열.

        회피 노드가 detections_topic 만 이쪽으로 바꿔 그대로 소비한다
        (frame=base_link 이므로 회피 노드의 TF 변환은 identity).
        """
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame
        if nearest is not None:
            det = Detection3D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.obstacle_class
            hyp.hypothesis.score = 1.0
            hyp.pose.pose.position.x = float(nearest[0])
            hyp.pose.pose.position.y = float(nearest[1])
            hyp.pose.pose.position.z = float(nearest[2])
            hyp.pose.pose.orientation.w = 1.0
            det.results.append(hyp)
            bbox = BoundingBox3D()
            bbox.center.position.x = float(nearest[0])
            bbox.center.position.y = float(nearest[1])
            bbox.center.position.z = float(nearest[2])
            bbox.center.orientation.w = 1.0
            # 박스 크기는 명목값(회피는 중심점만 사용).
            bbox.size.x = 0.2
            bbox.size.y = 0.2
            bbox.size.z = 0.4
            det.bbox = bbox
            msg.detections.append(det)
        self.pub_det.publish(msg)
        # 같은 nearest 로 회피 힌트도 발행(pose_goal_controller 소비).
        self._publish_hint(nearest, stamp)

    def _publish_hint(self, nearest, stamp):
        """nearest → Vector3Stamped 회피 힌트.

        x = 전방 최근접거리[m] (장애물 없으면 99.0)
        y = ±회피방향 — 장애물이 좌(+y)면 우(-1)로, 우(-y)면 좌(+1)로 비킨다.
            횡중심이 데드밴드 안(중앙)이면 직전 방향 유지(부호 떨림 방지).
        z = 0
        """
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame
        if nearest is None:
            msg.vector.x = 99.0
            msg.vector.y = 0.0
        else:
            msg.vector.x = float(nearest[0])
            y = float(nearest[1])
            if abs(y) < self.hint_deadband:
                direction = self._last_hint_dir          # 중앙 모호 → 유지
            else:
                direction = -1.0 if y > 0.0 else 1.0      # 좌 장애물→우, 우 장애물→좌
                self._last_hint_dir = direction
            msg.vector.y = direction
        msg.vector.z = 0.0
        self.pub_hint.publish(msg)

    def _points_marker(self, pts, mid, frame, stamp, color):
        """(N,3) 점들을 POINTS 마커로. max_points 초과 시 솎아냄."""
        if pts.shape[0] > self.max_points:
            idx = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[idx]
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns = "depth_obst"
        m.id = mid
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = self.point_size
        m.scale.y = self.point_size
        m.color = color
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        return m


def main(args=None):
    rclpy.init(args=args)
    node = DepthObstacleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
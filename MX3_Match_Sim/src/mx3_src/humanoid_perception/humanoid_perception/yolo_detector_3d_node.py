#!/usr/bin/env python3
"""
yolo_detector_3d_node.py
========================

YOLOv8 (TensorRT engine) 기반 객체 검출 + 3D Cartesian 좌표 계산 노드.

기존 `yolo_detector_node` 와 독립적으로 동작하며, 한 번의 YOLO 추론으로
각 검출 객체의 3D 위치(X, Y, Z) 를 카메라 광학(optical) 좌표계 기준으로 publish 한다.

좌표계 (camera color optical frame, ROS REP-103 convention):
    - X: 오른쪽 (right)
    - Y: 아래쪽 (down)
    - Z: 앞쪽   (forward, 카메라가 바라보는 방향)
즉, 책상 위에 정면을 보고 놓인 카메라 기준에서 +Z 가 객체 거리에 해당.

핵심 원리 (pinhole deprojection):
    bbox 중심 픽셀 (u, v) + 그 픽셀의 깊이 z (m) →
        X = (u - cx_p) * z / fx
        Y = (v - cy_p) * z / fy
        Z = z
    여기서 fx, fy 는 초점거리(픽셀 단위), (cx_p, cy_p) 는 주점(principal point).
    이 값들은 `/camera/.../color/camera_info` 토픽의 K 행렬에서 얻는다.
    aligned_depth_to_color 를 쓰고 있으므로 depth 픽셀에도 동일한 intrinsics 가 유효.

입력:
  /camera/camera/color/image_raw                   (sensor_msgs/Image, bgr8)
  /camera/camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image, 16UC1, 단위 mm)
  /camera/camera/color/camera_info                 (sensor_msgs/CameraInfo)

출력:
  /perception/detections_3d  (vision_msgs/Detection3DArray)
                             각 검출의 3D 중심(X,Y,Z) + 크기(near_z 기반 추정) + class.
  /perception/annotated_3d   (sensor_msgs/Image)
                             bbox + 라벨 + (X,Y,Z) 텍스트가 그려진 시각화 이미지.

파라미터 (대부분 기존 노드와 동일):
  engine_path          TensorRT engine 파일 경로 (.engine). 필수.
  conf_threshold       신뢰도 임계값 (기본 0.25)
  iou_threshold        NMS IoU 임계값 (기본 0.45)
  imgsz                추론 입력 해상도 (기본 640)
  publish_annotated    시각화 이미지 발행 여부 (기본 true)
  color_topic          컬러 입력 토픽
  depth_topic          aligned depth 입력 토픽
  camera_info_topic    CameraInfo 토픽
  output_frame_id      3D 좌표가 표현되는 frame_id (빈 문자열이면 컬러 헤더 그대로 사용)
                       D405 + realsense2_camera 의 표준값은 'camera_color_optical_frame'
  min_depth_m          유효 거리 하한 (기본 0.07, D405 MinZ)
  max_depth_m          유효 거리 상한 (기본 3.0, D405 권장 범위 상한 근처)
  depth_patch_radius   bbox 중심에서 깊이 추출 시 patch 반경(px) (기본 3 → 7×7 patch)
  drop_invalid_depth   깊이 측정 실패한 detection 을 결과에서 제외할지 여부 (기본 true)
                       true  : 깊이를 못 구한 검출은 발행하지 않는다.
                       false : 2D bbox 만 있는 detection 도 X=Y=Z=0 으로 publish.
                       ★ false 는 위험하다 — (0,0,0) 은 '깊이 미상'이라는 뜻인데
                         메시지에는 그 구분이 없다. 자체 min_valid_z 게이트가 없는
                         소비자(visual_servo_kick_node)가 이를 실제 좌표로 읽어
                         '로봇 머리 밑에 공이 있다'고 오인한다. 기본 true 유지 권장.
  class_names          클래스 인덱스→이름 매핑 리스트
"""
import time  # 추론 시간 측정 및 FPS 계산용

import numpy as np  # 깊이 patch 처리 (median, masking)
import cv2          # 시각화 (bbox, text 그리기)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Image: 컬러/깊이 입력, CameraInfo: intrinsics (fx, fy, cx, cy) 수신
from sensor_msgs.msg import Image, CameraInfo

# Detection3DArray: 3D 박스 + class id + score 를 담는 표준 ROS 메시지.
# - Detection3D.bbox 는 BoundingBox3D 타입이며 center(Pose) + size(Vector3) 필드를 가짐.
# - Detection3D.results 는 ObjectHypothesisWithPose 의 list. 여기에 class_id, score, pose 가 들어감.
from vision_msgs.msg import (
    Detection3DArray, Detection3D, BoundingBox3D,
    ObjectHypothesisWithPose,
)

from cv_bridge import CvBridge  # ROS Image <-> OpenCV ndarray 변환

from ultralytics import YOLO  # YOLOv8 추론 엔진 (.engine 파일 직접 로드 가능)


# ──────────────────────────────────────────────────────────────────────
# 클래스 ID 별 시각화 색상 (BGR) — 기존 노드와 동일하게 맞춤
#   0=ball, 1=goalpost, 2=robot, 3=field_edge, 4=line,
#   5=X-Intersection, 6=T-Intersection, 7=L-Intersection, 8=obstacle
# ──────────────────────────────────────────────────────────────────────
DEFAULT_COLORS_BY_ID = {
    0: (255,   0, 255),  # ball             - magenta
    1: (  0, 255, 255),  # goalpost         - yellow
    2: (255, 255,   0),  # robot            - cyan
    3: (128, 128, 255),  # field_edge       - light red/pink
    4: (255, 255, 255),  # line             - white
    5: (  0,   0, 255),  # X-Intersection   - red
    6: (255,   0,   0),  # T-Intersection   - blue
    7: (  0, 255,   0),  # L-Intersection   - green
    8: (  0, 128, 255),  # obstacle         - orange
}
FALLBACK_COLOR = (180, 180, 180)  # 정의되지 않은 class_id 가 들어오면 회색


def color_for_class(cls_id: int):
    """class_id 에 대응되는 BGR 색상 tuple 을 반환. 정의되지 않으면 fallback 회색."""
    return DEFAULT_COLORS_BY_ID.get(int(cls_id), FALLBACK_COLOR)


class YoloDetector3D(Node):
    """
    YOLO 검출 + 카메라 intrinsics 역투영으로 3D Cartesian 좌표를 산출하는 노드.

    동기화 정책:
        - depth, color, camera_info 를 각각 독립 콜백으로 받고 최신 값을 캐시한다.
          (message_filters ApproximateTimeSynchronizer 사용도 가능하지만
           카메라 fps 가 충분히 높고 D405 의 align 후 시점이 거의 같아 단순 캐시로 충분.)
        - color 콜백에서만 추론 + publish 수행.
    """

    def __init__(self):
        super().__init__('yolo_detector_3d')

        # ──────────────────────────────────────────────────────────
        # Parameters
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('engine_path', '')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('color_topic',
                               '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        # aligned_depth_to_color 사용 시 color 의 intrinsics 가 depth 픽셀에도 그대로 적용된다.
        # (즉 depth 픽셀 (u,v) 가 color 픽셀 (u,v) 와 같은 광선을 본다.)
        self.declare_parameter('camera_info_topic',
                               '/camera/camera/color/camera_info')
        # output_frame_id: 비어있으면 컬러 이미지 헤더의 frame_id 를 그대로 사용.
        # realsense2_camera 기본값은 'camera_color_optical_frame'.
        # 다운스트림 TF 변환을 일관되게 하려면 명시 지정 권장.
        self.declare_parameter('output_frame_id', '')
        # 깊이 유효 범위 — D405 spec: MinZ ≈ 0.07m, 권장 작동 0.07~0.5m, ~1m 까지 신뢰
        # 3m 까지 받아두되 RoboCup 필드 원거리 공은 depth 부정확함을 인지할 것.
        self.declare_parameter('min_depth_m', 0.07)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('depth_patch_radius', 3)
        # ── 깊이 측정 실패 검출 처리 ────────────────────────────────
        # (2026-08 변경) 기본값 False → True
        #
        # [왜 바꿨나 — 버그였다]
        #   False 일 때 깊이 측정에 실패한 검출은 버려지지 않고 X=Y=Z=0 으로
        #   발행된다. 이 (0,0,0) 은 "원점에 물체가 있다"가 아니라 "깊이를
        #   모른다"는 뜻인데, 메시지에는 그 구분이 담기지 않는다.
        #
        #   다운스트림 소비자 5개 중 3개(ball_map_projector /
        #   neck_ball_tracker / field_landmark_localizer)는 자체적으로
        #   min_valid_z 게이트가 있어 (0,0,0) 을 버린다. 그러나
        #   visual_servo_kick_node 에는 그 게이트가 없다.
        #   → (0,0,0) 을 base_link 로 TF 변환하면 '카메라 광학원점의
        #     base_link 좌표'(= 로봇 머리 근처)가 나오고, 그 값이 그대로
        #     공 위치(_ball_b)로 들어간다. 로봇이 자기 머리 밑에 공이
        #     있다고 믿고 정렬을 시도하는 오동작이 된다.
        #
        # [잃는 것이 있나 — 없다]
        #   깊이 없는 검출(2D bbox 만 있는 검출)을 실제로 활용하는 소비자가
        #   하나도 없다. 전부 min_valid_z 로 버리거나(3개), 잘못 쓰거나(1개),
        #   구독조차 하지 않는다(walkkick 은 ball_source=param).
        #   따라서 True 로 바꿔도 정상 경로의 동작은 변하지 않고,
        #   위의 오동작 경로만 사라진다.
        #
        # [되돌리려면]
        #   launch 인자는 그대로 남아 있다:
        #     ros2 launch humanoid_perception yolo_3d.launch.py \
        #       drop_invalid_depth:=false
        #   (깊이 파이프라인 자체를 의심할 때 원시 검출을 보려는 용도)
        self.declare_parameter('drop_invalid_depth', True)
        # ── 로봇 자기 몸 오검출 방지: ball 근거리 게이트 ─────────────
        # 로봇의 팔/몸통/발이 카메라 시야(특히 목을 아래로 숙였을 때)에 들어와
        # 'ball' 로 오검출되는 문제가 있다. 실제 경기 공은 카메라 광학중심에서
        # 일정 거리 이상 떨어져 있으므로, ball 로 검출됐어도 그 3D 거리(Z, m)가
        # ball_min_dist_m 보다 가까우면 자기 몸일 가능성이 높다고 보고 버린다.
        #   - 값은 '카메라 광학중심 ~ 공 중심'의 직선거리(Z) 기준이지, 발 앞
        #     수평거리가 아님에 유의(카메라가 머리에 있어 직선거리가 더 큼).
        #   - 킥 대상(발 앞) 공까지 걸러지지 않도록 실측 후 조정할 것.
        #     0.40m 는 보수적 시작값. 자기 몸 오검출이 남으면 키우고, 발 앞
        #     공이 사라지면 줄인다.
        #   - depth 측정 실패(Z 미상)한 검출에는 이 게이트를 적용하지 않는다.
        self.declare_parameter('ball_min_dist_m', 0.30)
        # 클래스 이름 매핑 — 학습된 .pt 의 m.names 와 동일해야 함
        self.declare_parameter(
            'class_names',
            ['ball', 'goalpost', 'robot', 'field_edge', 'line',
             'X-Intersection', 'T-Intersection', 'L-Intersection', 'obstacle'])

        # ──────────────────────────────────────────────────────────
        # Parameter 값 캐싱 (콜백마다 get_parameter 호출 피하기)
        # ──────────────────────────────────────────────────────────
        engine_path = self.get_parameter('engine_path').value
        if not engine_path:
            raise RuntimeError(
                "engine_path 파라미터가 비어있습니다. "
                "예: --ros-args -p engine_path:=/path/to/best.engine")

        self.conf  = float(self.get_parameter('conf_threshold').value)
        self.iou   = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.publish_annotated = bool(self.get_parameter('publish_annotated').value)
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_frame_id = str(self.get_parameter('output_frame_id').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.depth_patch_radius = int(self.get_parameter('depth_patch_radius').value)
        self.drop_invalid_depth = bool(self.get_parameter('drop_invalid_depth').value)
        self.class_names = list(self.get_parameter('class_names').value)

        # ── ball 근거리 게이트 관련 캐시 ───────────────────────────
        self.ball_min_dist_m = float(self.get_parameter('ball_min_dist_m').value)
        # 'ball' 의 클래스 인덱스를 이름 매핑에서 찾아 캐시(하드코딩 0 대신).
        # 학습 시 클래스 순서가 바뀌어도 이름만 맞으면 정상 동작한다.
        # 이름 목록에 'ball' 이 없으면 -1 → 게이트가 아무것도 거르지 않음(안전).
        try:
            self._ball_class_id = self.class_names.index('ball')
        except ValueError:
            self._ball_class_id = -1
            self.get_logger().warn(
                "class_names 에 'ball' 이 없어 근거리 게이트가 비활성화됩니다.")
        # 근거리 게이트로 버린 ball 검출 누적 개수(진단 로그용).
        self._skipped_near_ball = 0
        # 깊이 측정 실패로 버려진 검출 누적 수 (drop_invalid_depth=True 일 때만 증가)
        self._skipped_no_depth = 0
        self.get_logger().info(
            f"ball 근거리 게이트: dist < {self.ball_min_dist_m:.2f}m 인 ball 검출 무시 "
            f"(ball_class_id={self._ball_class_id})")

        # ──────────────────────────────────────────────────────────
        # Model load (.engine) + warmup
        # ──────────────────────────────────────────────────────────
        self.get_logger().info(f"Loading TensorRT engine: {engine_path}")
        # ultralytics YOLO 는 .engine 확장자를 자동 인식해 TensorRT 백엔드로 로드.
        # task='detect' 명시 — 분류/세그멘테이션 엔진을 잘못 로드하지 않도록.
        self.model = YOLO(engine_path, task='detect')
        # warmup — TensorRT 의 첫 추론은 메모리 할당/커널 캐싱 때문에 매우 느림.
        # 더미 이미지 한 번 흘려서 첫 추론 지연을 미리 소화한다.
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        self.get_logger().info("Model warmed up.")

        # ──────────────────────────────────────────────────────────
        # Runtime state caches
        # ──────────────────────────────────────────────────────────
        # 최신 depth (np.uint16, 단위 mm). aligned_depth_to_color 이므로 color 와 픽셀 1:1 대응.
        self._latest_depth = None
        self._latest_depth_stamp = None

        # 카메라 intrinsics — CameraInfo 가 도착하기 전엔 None.
        # (fx, fy, cx_p, cy_p): 초점거리(px) 와 주점(px) 좌표.
        # 변수명에 _p 를 붙인 이유: bbox 중심 cx, cy 와 헷갈리지 않기 위함.
        self._intr = None
        self._intr_logged = False  # CameraInfo 첫 수신 시 한 번만 로그

        # ──────────────────────────────────────────────────────────
        # CV bridge — ROS Image <-> numpy 변환기
        # ──────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ──────────────────────────────────────────────────────────
        # QoS: 카메라 토픽은 BEST_EFFORT (드롭 허용, 지연 최소)
        # CameraInfo 도 같은 QoS 로 받음 — 거의 안 변하니 한 번만 받으면 됨.
        # ──────────────────────────────────────────────────────────
        cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ──────────────────────────────────────────────────────────
        # Subscribers / Publishers
        # ──────────────────────────────────────────────────────────
        self.create_subscription(Image, self.color_topic, self._on_color, cam_qos)
        self.create_subscription(Image, self.depth_topic, self._on_depth, cam_qos)
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, cam_qos)

        # 3D detection publisher — KEEP_LAST depth=10 (다운스트림이 좀 늦어도 손실 방지)
        self.pub_det3d = self.create_publisher(
            Detection3DArray, '/perception/detections_3d', 10)
        if self.publish_annotated:
            self.pub_vis = self.create_publisher(
                Image, '/perception/annotated_3d', 10)

        # ──────────────────────────────────────────────────────────
        # Statistics — 2초마다 평균 추론시간/실제 FPS 로그
        # ──────────────────────────────────────────────────────────
        self._frame_count = 0
        self._t_inf_ms_sum = 0.0
        self._t_window_start = None
        self.create_timer(2.0, self._print_stats)

        self.get_logger().info(
            f"YOLO 3D detector ready. classes={self.class_names}, "
            f"conf={self.conf}, iou={self.iou}, imgsz={self.imgsz}, "
            f"depth_range=[{self.min_depth_m}, {self.max_depth_m}]m, "
            f"drop_invalid_depth={self.drop_invalid_depth}")

    # ════════════════════════════════════════════════════════════════
    # Callbacks
    # ════════════════════════════════════════════════════════════════

    def _on_depth(self, msg: Image):
        """깊이 이미지 콜백 — 캐시만 갱신. 추론은 color 콜백에서."""
        try:
            # 16UC1: 16-bit unsigned, 값은 mm 단위. 0 = invalid / no measurement.
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().warn(f"depth convert failed: {e}")
            return
        self._latest_depth = depth
        self._latest_depth_stamp = msg.header.stamp

    def _on_camera_info(self, msg: CameraInfo):
        """
        CameraInfo 콜백 — intrinsics (fx, fy, cx, cy) 를 캐시.

        CameraInfo.k 는 row-major 3x3 카메라 행렬:
            [ fx   0   cx ]
            [  0  fy   cy ]
            [  0   0    1 ]
        인덱스: 0=fx, 4=fy, 2=cx, 5=cy.

        aligned_depth_to_color 사용 시 이 intrinsics 가 depth 픽셀의 역투영에도 그대로 유효.
        """
        k = msg.k
        # 카메라가 캘리브레이션 안 됐을 때 fx=0, fy=0 으로 publish 되는 경우가 있어 방어
        if k[0] <= 0.0 or k[4] <= 0.0:
            self.get_logger().warn_once(
                "CameraInfo 의 fx/fy 가 0 이하 — 캘리브레이션 확인 필요")
            return
        fx = float(k[0])
        fy = float(k[4])
        cx_p = float(k[2])
        cy_p = float(k[5])
        self._intr = (fx, fy, cx_p, cy_p)

        # 첫 수신 한 번만 상세 로그
        if not self._intr_logged:
            self.get_logger().info(
                f"CameraInfo received: fx={fx:.2f}, fy={fy:.2f}, "
                f"cx={cx_p:.2f}, cy={cy_p:.2f}, "
                f"size={msg.width}x{msg.height}, "
                f"frame_id='{msg.header.frame_id}'")
            self._intr_logged = True

    def _on_color(self, msg: Image):
        """컬러 이미지 콜백 — 추론 + 3D 좌표 계산 + publish."""
        # 처리 FPS 측정 윈도우 시작 시각 — 첫 콜백 도착 시점 기록
        if self._t_window_start is None:
            self._t_window_start = time.perf_counter()

        try:
            # ultralytics 는 BGR numpy ndarray 입력을 받음
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"color convert failed: {e}")
            return

        # ── YOLO 추론 ─────────────────────────────────────────
        t0 = time.perf_counter()
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]  # batch=1 이므로 첫 결과만
        t1 = time.perf_counter()
        self._frame_count += 1
        self._t_inf_ms_sum += (t1 - t0) * 1000.0

        # ── Detection3DArray 메시지 헤더 ──────────────────────
        det_msg = Detection3DArray()
        det_msg.header = msg.header
        # output_frame_id 가 지정돼있으면 헤더의 frame_id 를 덮어쓴다.
        # 일반적으로 'camera_color_optical_frame' 으로 강제 지정해 TF 변환을 일관되게.
        if self.output_frame_id:
            det_msg.header.frame_id = self.output_frame_id

        # 현재 캐시된 depth 와 intrinsics 스냅샷 — 콜백 진행 중 갱신 방지
        depth = self._latest_depth
        intr  = self._intr
        h_img, w_img = frame.shape[:2]

        # ── 박스 처리 + 3D 변환 ───────────────────────────────
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            # GPU tensor → CPU numpy 한 번에 변환 (반복 호출 비용 절감)
            xyxy   = boxes.xyxy.cpu().numpy()                    # (N, 4)
            cls    = boxes.cls.cpu().numpy().astype(int)         # (N,)
            scores = boxes.conf.cpu().numpy()                    # (N,)

            for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scores):
                # 2D bbox 중심 + 크기 (픽셀 단위)
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                bw = (x2 - x1)
                bh = (y2 - y1)

                # 깊이 추정 (없으면 None)
                z_m = self._sample_depth_m(depth, cx, cy, w_img, h_img)

                # 깊이 측정 실패 시 옵션에 따라 스킵
                # 카운터를 세는 이유: drop_invalid_depth=True 로 바뀌면서
                # 이 검출들이 조용히 사라진다. 나중에 "공이 안 잡힌다" 를
                # 디버깅할 때 'YOLO 가 못 봤다' 와 '깊이를 못 구했다' 를
                # 구분할 수단이 필요하다. 2초 통계 로그에 같이 찍는다.
                if z_m is None and self.drop_invalid_depth:
                    self._skipped_no_depth += 1
                    continue

                # ── 로봇 자기 몸(ball 오검출) 근거리 게이트 ──────────
                # ball 로 검출됐지만 거리가 ball_min_dist_m 보다 가까우면
                # 자기 몸일 가능성이 높다고 보고 결과에서 제외한다.
                # (시각화 _draw 에서도 동일 판정으로 회색 처리하여 관찰 가능)
                if self._is_near_self_ball(int(c), z_m):
                    self._skipped_near_ball += 1
                    continue

                # ── 3D 중심 좌표 계산 ──────────────────────────
                # intrinsics 와 depth 가 모두 있을 때만 정상 X/Y/Z 계산.
                # 그 외엔 (0, 0, 0) — drop_invalid_depth=False 일 때 발생 가능.
                X = Y = Z = 0.0
                if z_m is not None and intr is not None:
                    fx, fy, cx_p, cy_p = intr
                    # pinhole deprojection — 왜곡 무시 (D405 color 는 거의 무왜곡)
                    #   X = (u - cx_p) * z / fx   (왼쪽: -, 오른쪽: +)
                    #   Y = (v - cy_p) * z / fy   (위:   -, 아래:  +)
                    #   Z = z                       (앞:   +)
                    X = (cx - cx_p) * z_m / fx
                    Y = (cy - cy_p) * z_m / fy
                    Z = z_m

                # ── 3D bbox 크기 추정 ────────────────────────
                # 정확한 깊이가 없으므로 bbox 픽셀 크기를 z 거리에서 metric 으로 환산.
                # size_x ≈ bw * z / fx, size_y ≈ bh * z / fy.
                # size_z (깊이 방향 두께) 는 알 수 없으므로 0 으로 둔다.
                size_x = size_y = size_z = 0.0
                if z_m is not None and intr is not None:
                    fx, fy, _, _ = intr
                    size_x = float(bw) * z_m / fx
                    size_y = float(bh) * z_m / fy

                # ── ObjectHypothesisWithPose 채우기 ─────────
                # 이 메시지는 class_id, score, 그리고 (선택적으로) pose 를 담는다.
                # 다운스트림이 hypothesis 단위로 pose 를 보는 경우(예: vision_msgs 표준 흐름)
                # bbox.center 와 동일한 값을 여기에도 채워주면 호환성이 높다.
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = self._class_name(int(c))
                hyp.hypothesis.score = float(s)
                hyp.pose.pose.position.x = float(X)
                hyp.pose.pose.position.y = float(Y)
                hyp.pose.pose.position.z = float(Z)
                # quaternion identity — tf2 가 0벡터 quaternion 에 경고 띄우므로 안전하게 채움
                hyp.pose.pose.orientation.w = 1.0

                # ── BoundingBox3D 채우기 ────────────────────
                bbox3d = BoundingBox3D()
                bbox3d.center.position.x = float(X)
                bbox3d.center.position.y = float(Y)
                bbox3d.center.position.z = float(Z)
                bbox3d.center.orientation.w = 1.0
                bbox3d.size.x = float(size_x)
                bbox3d.size.y = float(size_y)
                bbox3d.size.z = float(size_z)

                # ── Detection3D 조립 ────────────────────────
                det = Detection3D()
                det.header = det_msg.header  # array 와 동일 frame_id 사용
                det.bbox = bbox3d
                det.results.append(hyp)
                det_msg.detections.append(det)

        # ── publish 3D detections ─────────────────────────────
        self.pub_det3d.publish(det_msg)

        # ── 시각화 (옵션) ────────────────────────────────────
        if self.publish_annotated:
            vis = self._draw(frame, results, depth, intr, w_img, h_img)
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            vis_msg.header = msg.header  # 원본 컬러 헤더 그대로 (TF는 detection 쪽 frame_id 사용)
            self.pub_vis.publish(vis_msg)

    # ════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════

    def _class_name(self, idx: int) -> str:
        """class_id 정수를 사람이 읽는 이름으로 변환. 매핑 밖이면 'class_<idx>'."""
        if 0 <= idx < len(self.class_names):
            name = self.class_names[idx]
            if name:  # 빈 문자열 방어
                return name
        return f"class_{idx}"

    def _is_near_self_ball(self, cls_id: int, z_m) -> bool:
        """ball 근거리 게이트 판정 — True 면 그 검출을 버려야 한다.

        로봇의 팔/몸통/발이 'ball' 로 오검출되는 것을 막기 위한 필터.
        아래를 모두 만족할 때만 True:
          - 게이트 활성 (self._ball_class_id >= 0; 'ball' 이름이 없으면 -1=비활성)
          - 검출 클래스가 ball
          - 거리(z_m) 측정 성공
          - z_m 이 ball_min_dist_m 보다 가까움
        depth 측정 실패(z_m is None)면 거리를 알 수 없으므로 거르지 않는다(False).
        """
        if self._ball_class_id < 0:
            return False
        if cls_id != self._ball_class_id:
            return False
        if z_m is None:
            return False
        return z_m < self.ball_min_dist_m

    def _sample_depth_m(self, depth, cx, cy, w_img, h_img):
        """
        bbox 중심 (cx, cy) 주변 작은 patch 의 median 으로 거리(m) 추정.

        - depth: np.uint16 mm. None 이면 None 반환.
        - 단일 픽셀은 stereo hole 에 빠지면 0 이 나오므로 patch median 이 robust.
        - 유효 범위 [min_depth_m, max_depth_m] 밖이면 None.

        반환: float(m) 또는 None.
        """
        if depth is None:
            return None
        # depth 와 color 의 해상도가 다르면 픽셀 매핑이 깨진 것 — aligned topic 이 아니거나 설정 이상.
        if depth.shape[0] != h_img or depth.shape[1] != w_img:
            return None

        # 픽셀 좌표 반올림 + 경계 클리핑
        cx_i, cy_i = int(round(cx)), int(round(cy))
        r = self.depth_patch_radius
        x0 = max(0, cx_i - r); x1 = min(w_img, cx_i + r + 1)
        y0 = max(0, cy_i - r); y1 = min(h_img, cy_i + r + 1)
        patch = depth[y0:y1, x0:x1]

        # 0 은 stereo 매칭 실패(no measurement) — 유효 픽셀만 추림
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        # mm → m. median 은 hole 잡음에 robust.
        z_m = float(np.median(valid)) * 1e-3
        if z_m < self.min_depth_m or z_m > self.max_depth_m:
            return None
        return z_m

    def _draw(self, frame, results, depth, intr, w_img, h_img):
        """
        시각화 — 원본 컬러 이미지 위에 bbox + 라벨 + (X,Y,Z) 텍스트를 그린다.
        라벨 포맷:  '<class> <conf>  (X,Y,Z)m'
        depth 또는 intrinsics 가 없으면 좌표는 생략 또는 거리만 표시.
        """
        out = frame.copy()
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return out

        xyxy   = boxes.xyxy.cpu().numpy()
        cls    = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scores):
            # 3D 좌표/근거리 게이트 판정을 위해 중심 깊이를 먼저 구한다.
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            z_m = self._sample_depth_m(depth, cx, cy, w_img, h_img)

            # 근거리 게이트로 '버려질' ball 인지 판정.
            # 버려질 검출은 실제 publish(Detection3DArray)에서는 제외되지만,
            # 시각화에는 회색 + [near-drop] 로 남겨 '왜 이 공이 무시됐는지'를
            # 눈으로 확인할 수 있게 한다(디버깅/튜닝용).
            near_self = self._is_near_self_ball(int(c), z_m)

            color = (90, 90, 90) if near_self else color_for_class(int(c))
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(out, p1, p2, color, 2)

            # 라벨 작성: 기본은 'name score' (+근거리 게이트 대상이면 표식)
            label = f"{self._class_name(int(c))} {s:.2f}"
            if near_self:
                label += " [near-drop]"

            # 깊이/intrinsics 기반 3D 좌표를 라벨 뒤에 덧붙임
            if z_m is not None:
                if intr is not None:
                    fx, fy, cx_p, cy_p = intr
                    X = (cx - cx_p) * z_m / fx
                    Y = (cy - cy_p) * z_m / fy
                    # +X=right, +Y=down, +Z=forward (optical frame)
                    label += f"  ({X:+.2f},{Y:+.2f},{z_m:.2f})m"
                else:
                    # intrinsics 미수신 — 거리만
                    label += f"  z={z_m:.2f}m"

            # 라벨 배경 박스 + 텍스트
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out,
                          (p1[0], max(0, p1[1] - th - 6)),
                          (p1[0] + tw + 4, p1[1]),
                          color, -1)
            cv2.putText(out, label,
                        (p1[0] + 2, p1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _print_stats(self):
        """2초마다 호출되는 통계 로그. 평균 추론 시간 + 실제/이론 FPS."""
        if self._frame_count == 0:
            return
        avg_ms = self._t_inf_ms_sum / self._frame_count
        # 이론 최대 FPS (추론만 고려, 카메라 fps 무시)
        fps_capacity = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        # 실제 처리 FPS (콜백 도착 윈도우 기준)
        now = time.perf_counter()
        elapsed = now - self._t_window_start if self._t_window_start else 0.0
        fps_actual = self._frame_count / elapsed if elapsed > 0 else 0.0

        # intrinsics 수신 여부도 함께 노출 — 처음 디버깅 시 매우 유용
        intr_ok = "OK" if self._intr is not None else "WAITING"
        self.get_logger().info(
            f"Inference: {avg_ms:.1f} ms/frame  "
            f"(actual {fps_actual:.1f} FPS, capacity ~{fps_capacity:.1f} FPS) "
            f"| intrinsics={intr_ok}  over {self._frame_count} frames"
            f" | near-ball dropped(누적)={self._skipped_near_ball}"
            f" | no-depth dropped(누적)={self._skipped_no_depth}"
        )
        # 윈도우 리셋
        self._frame_count = 0
        self._t_inf_ms_sum = 0.0
        self._t_window_start = None


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector3D()
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
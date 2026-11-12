#!/usr/bin/env python3
"""MuJoCo 하드웨어 대체 브리지 — mx3_src 경기 스택을 실기 대신 MuJoCo 에 붙인다.

대체하는 실기 노드 (dynamixel_system.launch.py 의 하드웨어 쪽 + 카메라/YOLO):
  body_motor_controller : /joint_states 50 Hz, /joint_goals, /u2d2/* 토픽·서비스
  imu_parser            : /imu/data 200 Hz, /imu/data_50hz, /imu/projected_gravity
  button_input          : /buttons/stop_pressed(래치 False), /buttons/start_event (/sim/start 로 누름)
  RealSense + YOLO      : /perception/detections_3d (정답 위치 + 노이즈, 모드 A)
그대로 쓰는 실기 노드: neck_control, fall_manager, 정책 러너, 측위, 전략 … (sim_robot.launch.py)

로봇마다 ROS_DOMAIN_ID 를 따로 연다 (스택이 절대 토픽 이름을 쓰므로 도메인으로 분리).
벽시계 1 배속으로 돈다 — 러너가 time.time() 을 쓰고 /joint_states 1 개당 정책 1 스텝이라서다.

실행 (sim_ws/scripts/run_bridge.sh 가 PYTHONPATH 를 잡아 준다):
  run_bridge.sh --scene ~/sim_ws/src/humanoid_sim/models/scene_1v0
"""
import argparse
import json
import math
import re
import threading
import time

import mujoco
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont   # 뷰어 '준비 중' 판 (없으면 위쪽 글자로 대신)
except ImportError:
    Image = None
import rclpy
from rclpy.context import Context
from rclpy.event_handler import PublisherEventCallbacks, SubscriptionEventCallbacks
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import (PointStamped, PoseStamped, PoseWithCovarianceStamped,
                               TransformStamped, Vector3Stamped)
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Empty, Float32, String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
try:   # 필드 선 검출 메시지 (hsl_src). 없으면 선 검출 시뮬을 끈다
    from hsl_interfaces.msg import LineSegment, LineSegmentArray, StrategyState
except ImportError:
    LineSegmentArray = StrategyState = None
from visualization_msgs.msg import Marker, MarkerArray

from humanoid_behavior_interfaces.srv import (GetJointLimits, MotorReset, RobotInit, StandUp,
                                              TorqueEnable, TorqueEnableJoints)
from humanoid_localization.vision.field_geometry import field_geometry_for_profile
from humanoid_localization.vision.field_landmarks import generate_field_landmarks

# ── 실기 규약 (servo_map.yaml / u2d2_hardware.py / robot_init_behavior.py) ─────────────
# /joint_states 순서 = 다이나믹셀 ID 순
JS_ORDER = [
    "l_ankle_roll_joint", "r_ankle_roll_joint", "l_ankle_pitch_joint", "r_ankle_pitch_joint",
    "l_knee_pitch_joint", "r_knee_pitch_joint", "l_pelvis_yaw_joint", "r_pelvis_yaw_joint",
    "l_pelvis_roll_joint", "r_pelvis_roll_joint", "l_pelvis_pitch_joint", "r_pelvis_pitch_joint",
    "l_shoulder_pitch_joint", "r_shoulder_pitch_joint", "l_shoulder_roll_joint", "r_shoulder_roll_joint",
    "l_elbow_pitch_joint", "r_elbow_pitch_joint", "neck_yaw_joint", "neck_pitch_joint",
]
NECK = {"neck_yaw_joint", "neck_pitch_joint"}
INIT_POSE = {  # robot_init_behavior.py
    "r_pelvis_pitch_joint": 0.62, "r_pelvis_roll_joint": 0.0, "r_pelvis_yaw_joint": 0.0,
    "r_knee_pitch_joint": 1.2, "r_ankle_pitch_joint": 0.62, "r_ankle_roll_joint": 0.0,
    "l_pelvis_pitch_joint": -0.62, "l_pelvis_roll_joint": 0.0, "l_pelvis_yaw_joint": 0.0,
    "l_knee_pitch_joint": -1.2, "l_ankle_pitch_joint": -0.62, "l_ankle_roll_joint": 0.0,
    "r_shoulder_pitch_joint": -0.3, "r_shoulder_roll_joint": 0.0, "r_elbow_pitch_joint": 1.0,
    "l_shoulder_pitch_joint": 0.3, "l_shoulder_roll_joint": 0.0, "l_elbow_pitch_joint": -1.0,
    "neck_yaw_joint": 0.0, "neck_pitch_joint": 0.4363,
}
RPM = 2.0 * math.pi / 60.0
PROFILE_MAX_RPM = 45.0          # body_motor_controller profile_velocity_limit_rpm
NECK_DEFAULT_RPM = 10.0

# 카메라 TF 체인 (XM_robot03_tf_min.urdf) — 스택의 robot_state_publisher 와 같은 값
NECK_YAW_ORIGIN = np.array([0.0485, 0.00059264, 0.091889])
NECK_PITCH_ORIGIN = np.array([-0.013831, 0.00125, 0.063])
CAMERA_ORIGIN = np.array([0.045, 0.0, 0.020])
# camera_link(x 앞, y 왼, z 위) → optical(x 오른, y 아래, z 앞)
R_LINK_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# D405 FOV (수평 87°, 수직 58°), YOLO 깊이 게이트 (yolo_3d.launch.py)
HFOV, VFOV = math.radians(87.0), math.radians(58.0)
MIN_DEPTH, MAX_DEPTH, BALL_MIN_DEPTH = 0.07, 3.0, 0.30
CLASS_NAME = {1: "goalpost", 5: "X-Intersection", 6: "T-Intersection", 7: "L-Intersection"}
CLASS_SIZE = {"ball": 0.13, "goalpost": 0.10, "X-Intersection": 0.15, "T-Intersection": 0.15,
              "L-Intersection": 0.15, "robot": 0.30}
GOALPOST_Z = 0.30               # 기둥 검출점 높이 (bbox 중심 근사)
BALL_LIN_DAMP = BALL_ANG_DAMP = 0.2


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def heading_of(R):
    """누워 있어도 맞는 '바라보던 방향'. 몸 x 축이 거의 수직이면(앞/뒤로 넘어짐) 몸 z 축으로 판단한다.
    yaw_of 만 쓰면 뒤로 누운 로봇을 세울 때 180° 뒤집혀 세워졌다 (4v4_g 에서 확인)."""
    if math.hypot(R[0, 0], R[1, 0]) > 0.5:
        return yaw_of(R)
    s = -1.0 if R[2, 0] > 0 else 1.0       # x 축이 위 = 뒤로 넘어짐 → 바라보던 방향 = −(몸 z 축)
    return math.atan2(s * R[1, 2], s * R[0, 2])


FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def text_rects(text, path, size):
    """글자를 PIL 로 작은 흑백 그림에 그려 켜진 화소를 직사각형으로 묶는다 (뷰어 3D 글자용: 직사각형 하나 = 막대 하나).
    반환: (폭, 높이 [화소], [(x0, y0, x1, y1), ...]  y 는 아래로)."""
    try:
        font = ImageFont.truetype(path, size)
    except OSError:
        font = ImageFont.load_default(size=size)
    left, top, right, bottom = font.getbbox(text)
    img = Image.new("L", (right - left + 2, bottom - top + 2), 0)
    ImageDraw.Draw(img).text((1 - left, 1 - top), text, font=font, fill=255)
    on = np.asarray(img) > 110
    rects, open_runs = [], {}                       # (x0, x1) → 시작 행: 같은 구간이 이어지는 행은 하나로 합친다
    for y, row in enumerate(on):
        edges = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(np.int8), [0]])))
        runs = list(zip(edges[::2], edges[1::2]))
        nxt = {run: open_runs.pop(run, y) for run in runs}
        rects += [(x0, y0, x1, y) for (x0, x1), y0 in open_runs.items()]
        open_runs = nxt
    rects += [(x0, y0, x1, on.shape[0]) for (x0, x1), y0 in open_runs.items()]
    return on.shape[1], on.shape[0], rects


class RobotHW:
    """로봇 1 대의 실기 대체 I/O. 자기 ROS 도메인에서 노드 하나를 연다."""

    def __init__(self, sim, m, d, meta, args):
        self.sim, self.m, self.d, self.args = sim, m, d, args
        self.name, self.p = meta["name"], meta["prefix"]
        self.ctx = Context()
        rclpy.init(context=self.ctx, domain_id=int(meta["domain"]))
        # 실행 비용 줄이기: 파라미터 서비스·QoS 이벤트 핸들러를 안 만든다. rclpy 는 executor 가 돌 때마다
        # 모든 대기 객체로 wait set 을 새로 만드는데, 발행자마다 붙는 이벤트 핸들러가 그 대부분이었다 (cProfile).
        self.node = rclpy.create_node(f"sim_hw_{self.name}", context=self.ctx, start_parameter_services=False)
        _cp, _cs = self.node.create_publisher, self.node.create_subscription
        self.node.create_publisher = lambda *a, **k: _cp(
            *a, event_callbacks=PublisherEventCallbacks(use_default_callbacks=False), **k)
        self.node.create_subscription = lambda *a, **k: _cs(
            *a, event_callbacks=SubscriptionEventCallbacks(use_default_callbacks=False), **k)
        self.exe = SingleThreadedExecutor(context=self.ctx)
        self.exe.add_node(self.node)
        self.log = self.node.get_logger()
        self.rng = np.random.default_rng(int(meta["domain"]))

        # ── 모델 인덱스 ──
        self.jid = {j: m.joint(self.p + j).id for j in JS_ORDER}
        self.qadr = np.array([m.jnt_qposadr[self.jid[j]] for j in JS_ORDER])
        self.vadr = np.array([m.jnt_dofadr[self.jid[j]] for j in JS_ORDER])
        self.aid = np.array([m.actuator(self.p + j).id for j in JS_ORDER])
        self.gain0 = m.actuator_gainprm[self.aid, 0].copy()
        self.bias0 = m.actuator_biasprm[self.aid, :3].copy()
        self.lo = np.array([m.jnt_range[self.jid[j]][0] for j in JS_ORDER])
        self.hi = np.array([m.jnt_range[self.jid[j]][1] for j in JS_ORDER])
        self.bid = m.body(self.p + "base_link").id
        root = m.joint(self.p + "root")
        self.rq, self.rv = m.jnt_qposadr[root.id], m.jnt_dofadr[root.id]
        s = lambda n: (m.sensor_adr[m.sensor(self.p + n).id], m.sensor_dim[m.sensor(self.p + n).id])
        self.s_gyro, self.s_acc, self.s_quat = s("gyro"), s("acc"), s("quat")
        self.idx = {j: i for i, j in enumerate(JS_ORDER)}
        self.neck_i = (self.idx["neck_yaw_joint"], self.idx["neck_pitch_joint"])

        # ── 서보 상태 ──
        init = np.array([INIT_POSE[j] for j in JS_ORDER])
        self.target = init.copy()        # Goal Position
        self.setpoint = init.copy()      # 프로파일 속도로 따라가는 내부 궤적
        self.prof = np.array([(NECK_DEFAULT_RPM if j in NECK else PROFILE_MAX_RPM) * RPM for j in JS_ORDER])
        self.torque = np.ones(20, bool)  # 물리적으로는 처음부터 잡고 있다 (사람이 들고 있는 상태)
        self.torque_reported = False     # 스택에는 START 전까지 토크 OFF 로 보인다
        self.pending = {}
        self.lock = threading.Lock()
        # 진단: 정책 goal 이 몇 주기 뒤에 쓰였나 (goal header.stamp = 그 goal 을 만든 /joint_states 의 stamp).
        #   1 = 정상 (다음 주기에 씀). 30 s 마다 1 이 아닌 게 있으면 경고로 내고 비운다.
        self.goal_stamp = None
        self.cycle = 0
        self.cycle_of = {}
        self.goal_lag = {}
        self.had_goal = False
        self.behavior = dict(behavior="none", active=False, phase="", phase_idx=0, total_phases=0)
        self.behavior_until = None
        self.init_settle = None
        self.started = False             # START 를 눌렀나 (뷰어 '준비 중' 표시용)
        self.ball_owner_t = -1.0         # 행동 노드가 '내가 공 담당' 이라고 마지막으로 알린 벽시계 시각 (뷰어 고리용)
        self.yaw_ref = None
        self.spawn = (float(meta["xy"][0]), float(meta["xy"][1]), math.radians(float(meta["yaw_deg"])))
        self.cam_bid = m.body(self.p + "neck_pitch_link").id     # 가림 광선 검사에서 제외할 머리
        self.reset_request = None

        # ── 퍼블리셔 ──
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        n = self.node
        self.pub_js = n.create_publisher(JointState, "/joint_states", 10)
        self.pub_imu = n.create_publisher(Imu, "/imu/data", 10)
        self.pub_imu50 = n.create_publisher(Imu, "/imu/data_50hz", 10)
        self.pub_grav = n.create_publisher(Vector3Stamped, "/imu/projected_gravity", 10)
        self.pub_torque = n.create_publisher(Bool, "/u2d2/torque_enabled", latched)
        self.pub_behavior = n.create_publisher(String, "/u2d2/behavior_status", 10)
        self.pub_diag = n.create_publisher(String, "/motor_diagnostics", 10)
        self.pub_stop = n.create_publisher(Bool, "/buttons/stop_pressed", latched)
        self.pub_start = n.create_publisher(Empty, "/buttons/start_event", 10)
        self.pub_det = n.create_publisher(Detection3DArray, "/perception/detections_3d", 10)
        self.pub_lines = (n.create_publisher(LineSegmentArray, "/perception/field_lines", 10)
                          if LineSegmentArray is not None and not args.no_lines else None)
        self.pub_initpose = n.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.pub_gt_pose = n.create_publisher(PoseStamped, "/sim/gt/base_pose", 10)
        self.pub_gt_ball = n.create_publisher(PointStamped, "/sim/gt/ball", 10)
        self.pub_event = n.create_publisher(String, "/sim/event", 10)
        self.pub_gt_markers = n.create_publisher(MarkerArray, "/sim/gt/markers", 10)
        self.pub_ball_vis = n.create_publisher(String, "/sim/debug/ball_vis", 10)
        self.pub_torque.publish(Bool(data=False))
        self.pub_stop.publish(Bool(data=False))

        # ── 서브스크라이버 ──
        n.create_subscription(JointState, "/joint_goals", self.cb_goals, 10)
        n.create_subscription(Float32, "/u2d2/set_profile_velocity", self.cb_profile_all, 10)
        n.create_subscription(JointState, "/u2d2/set_joint_profile_velocity", self.cb_profile_joints, 10)
        n.create_subscription(JointState, "/u2d2/set_joint_goal_pwm", lambda msg: None, 10)
        if StrategyState is not None:   # hsl_behavior 가 떠 있으면 공 담당 표시 (뷰어 발밑 고리)
            n.create_subscription(StrategyState, "/hsl/strategy_state", self.cb_strategy, 10)

        # ── 서비스 (body_motor_controller 와 같은 이름·타입) ──
        n.create_service(RobotInit, "/u2d2/robot_init", self.srv_robot_init)
        n.create_service(RobotInit, "/u2d2/motor_zero", lambda rq, rs: self._ok(rs, "sim: no-op"))
        n.create_service(TorqueEnable, "/u2d2/torque_enable", self.srv_torque)
        n.create_service(TorqueEnableJoints, "/u2d2/torque_enable_joints", self.srv_torque_joints)
        n.create_service(GetJointLimits, "/u2d2/get_joint_limits", self.srv_limits)
        n.create_service(StandUp, "/u2d2/standup", self.srv_standup)
        n.create_service(MotorReset, "/u2d2/motor_reset", lambda rq, rs: self._ok(rs, "sim: no errors"))
        n.create_service(Trigger, "/sim/start", self.srv_start)
        n.create_service(Trigger, "/sim/set_initialpose", self.srv_initpose)

        # ── 정적 TF: camera_link → optical (realsense2_camera 가 내던 것) ──
        self.stf = StaticTransformBroadcaster(n)
        tfs = []
        for child, q in (("camera_color_frame", (0.0, 0.0, 0.0, 1.0)),
                         ("camera_color_optical_frame", (-0.5, 0.5, -0.5, 0.5))):
            t = TransformStamped()
            t.header.stamp = n.get_clock().now().to_msg()
            t.header.frame_id = "camera_link" if child == "camera_color_frame" else "camera_color_frame"
            t.child_frame_id = child
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
            tfs.append(t)
        self.stf.sendTransform(tfs)
        self.log.info(f"[{self.name}] domain {meta['domain']} ready — START 는 /sim/start")

    # ────────────────────────── 콜백 ──────────────────────────
    @staticmethod
    def _ok(resp, msg):
        resp.success, resp.message = True, msg
        return resp

    def cb_goals(self, msg):
        with self.lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self.idx and math.isfinite(pos):
                    self.pending[name] = pos
            if len(msg.name) >= 12:
                self.goal_stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

    def cb_profile_all(self, msg):
        rpm = min(max(float(msg.data), 0.0), PROFILE_MAX_RPM)
        for j, i in self.idx.items():
            if j not in NECK:
                self.prof[i] = rpm * RPM if rpm > 0 else 0.0

    def cb_profile_joints(self, msg):
        for name, rpm in zip(msg.name, msg.velocity):
            if name in self.idx and math.isfinite(rpm) and rpm >= 0:
                self.prof[self.idx[name]] = rpm * RPM

    def set_torque(self, idxs, on):
        for i in idxs:
            self.torque[i] = on
            a = self.aid[i]
            self.m.actuator_gainprm[a, 0] = self.gain0[i] if on else 0.0
            self.m.actuator_biasprm[a, :3] = self.bias0[i] if on else 0.0
            if on:  # 켜는 순간의 자세를 목표로 (튀지 않게)
                self.setpoint[i] = self.target[i] = self.d.qpos[self.qadr[i]]

    def srv_torque(self, req, resp):
        self.set_torque(range(20), req.enable)
        self.torque_reported = bool(req.enable)
        self.pub_torque.publish(Bool(data=self.torque_reported))
        return self._ok(resp, f"sim torque {'ON' if req.enable else 'OFF'}")

    def srv_torque_joints(self, req, resp):
        idxs = [self.idx[n] for n in req.joint_names if n in self.idx]
        self.set_torque(idxs, req.enable)
        resp.applied_joints = [JS_ORDER[i] for i in idxs]
        return self._ok(resp, f"sim torque {len(idxs)} joints")

    def srv_limits(self, req, resp):
        resp.joint_names = list(JS_ORDER)
        resp.lower_limits_rad = [float(v) for v in self.lo]
        resp.upper_limits_rad = [float(v) for v in self.hi]
        return self._ok(resp, "sim limits")

    def start_robot_init(self, init_rpm=20.0, final_rpm=45.0, tol=0.05, settle=10, timeout=5.0):
        self.set_torque([i for i in range(20) if not self.torque[i]], True)
        self.torque_reported = True
        self.pub_torque.publish(Bool(data=True))
        for j, i in self.idx.items():
            self.prof[i] = (NECK_DEFAULT_RPM if j in NECK else min(init_rpm, PROFILE_MAX_RPM)) * RPM
            self.target[i] = INIT_POSE[j]
        self.init_settle = dict(final=min(final_rpm, PROFILE_MAX_RPM), tol=tol, need=settle, count=0,
                                deadline=time.monotonic() + timeout)
        self.behavior = dict(behavior="robot_init", active=True, phase="moving", phase_idx=0, total_phases=1)

    def srv_robot_init(self, req, resp):
        self.start_robot_init(req.init_rpm or 20.0, req.final_rpm or 45.0, req.pos_tolerance_rad or 0.05,
                              req.settle_count or 10, req.timeout_sec or 5.0)
        return self._ok(resp, "started")

    def srv_standup(self, req, resp):
        cmd = (req.command or "start").strip().lower()
        if cmd != "start":   # fall_manager 는 STOPPED 진입 시 "stop" 을 보낸다 → 예약된 기상 취소
            self.behavior_until = None
            self.behavior = dict(behavior="none", active=False, phase="", phase_idx=0, total_phases=0)
            resp.current_phase, resp.phase_index, resp.total_phases = "", 0, 0
            return self._ok(resp, f"sim standup '{cmd}'")
        # ★ 임시: 기상 동작 대신 3 s 뒤 그 자리에 세워 둔다 (기상 정책/키프레임은 다음 단계)
        self.behavior = dict(behavior="standup", active=True, phase="sim_teleport", phase_idx=0, total_phases=1)
        self.behavior_until = time.monotonic() + 3.0
        resp.detected_direction = req.fall_direction or "unknown"
        resp.current_phase, resp.phase_index, resp.total_phases = "sim_teleport", 0, 1
        self.log.warn(f"[{self.name}] standup 요청 → sim 임시 처리 (3 s 후 제자리 기립)")
        return self._ok(resp, "started")

    def cb_strategy(self, msg):
        if msg.active:
            self.ball_owner_t = time.time()

    def srv_start(self, req, resp):
        """START 버튼: start_event + robot_init (button_input_node 와 같은 인자).
        사람이 로봇을 사이드라인 출발 위치에 세워 놓고 누르는 것과 같게, 출발 pose 에 바로 세운다
        (START 전 대기 중 기울거나 넘어진 경우가 있었다 — 4v4_g 에서 1 대 전도). 물리 스텝 사이에 적용."""
        self.reset_request = self.spawn
        self.started = True
        self.pub_start.publish(Empty())
        self.start_robot_init(20.0, 45.0, 0.05, 10, 5.0)
        return self._ok(resp, "START pressed (sim)")

    def srv_initpose(self, req, resp):
        self.publish_initialpose()
        return self._ok(resp, "initialpose = ground truth")

    # ────────────────────────── 시뮬 쪽 동작 ──────────────────────────
    def base_pose(self):
        return self.d.xpos[self.bid].copy(), self.d.xmat[self.bid].reshape(3, 3).copy()

    def place_upright(self, pose=None):
        """넘어진 로봇을 그 자리에 기본자세로 세운다 (임시 기상). pose=(x, y, yaw) 면 그 자리에."""
        pos, R = self.base_pose()
        yaw = heading_of(R)
        if pose is not None:
            pos, yaw = (pose[0], pose[1]), pose[2]
        self.d.qpos[self.rq:self.rq + 3] = [pos[0], pos[1], 0.42]
        self.d.qpos[self.rq + 3:self.rq + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.d.qvel[self.rv:self.rv + 6] = 0.0
        for j, i in self.idx.items():
            self.d.qpos[self.qadr[i]] = INIT_POSE[j]
            self.d.qvel[self.vadr[i]] = 0.0
            self.target[i] = self.setpoint[i] = INIT_POSE[j]

    def begin_cycle(self, stamp):
        """20 ms 주기 시작: 쌓인 goal 을 한 번에 쓰고(SyncWrite) 상태를 읽어 낸다(SyncRead)."""
        with self.lock:
            goals, self.pending = self.pending, {}
            gst, self.goal_stamp = self.goal_stamp, None
        for name, pos in goals.items():
            self.target[self.idx[name]] = min(max(pos, -math.pi), math.pi)
        self.cycle += 1
        if gst is not None:
            lag = self.cycle - self.cycle_of[gst] if gst in self.cycle_of else "?"
            self.goal_lag[lag] = self.goal_lag.get(lag, 0) + 1
        elif self.had_goal:
            self.goal_lag["빈"] = self.goal_lag.get("빈", 0) + 1     # 직전 주기엔 goal 이 왔는데 이번엔 없음
        self.had_goal = gst is not None
        sn = stamp.sec * 10**9 + stamp.nanosec
        self.cycle_of[sn] = self.cycle
        if len(self.cycle_of) > 50:
            self.cycle_of.pop(next(iter(self.cycle_of)))
        if self.cycle % 1500 == 0:
            late = sum(v for k, v in self.goal_lag.items() if k not in (1, "빈"))
            if late:        # 정상 (전부 1 주기) 이면 조용히. 2026-11-02 4v4 실측은 전부 1 주기였다
                self.log.warn(f"[{self.name}] goal 지연(주기) {dict(sorted(self.goal_lag.items(), key=str))}")
            self.goal_lag = {}
        js = JointState()
        js.header.stamp = stamp
        js.name = list(JS_ORDER)
        js.position = [float(v) for v in self.d.qpos[self.qadr]]
        js.velocity = [float(v) for v in self.d.qvel[self.vadr]]
        self.pub_js.publish(js)

    def pre_step(self, dt):
        step = np.where(self.prof > 0, self.prof * dt, np.inf)
        self.setpoint += np.clip(self.target - self.setpoint, -step, step)
        self.d.ctrl[self.aid] = np.clip(self.setpoint, self.lo, self.hi)

    def imu_msg(self, stamp):
        q = self.d.sensordata[self.s_quat[0]:self.s_quat[0] + 4].copy()       # body→world (w,x,y,z)
        if self.yaw_ref is None:
            self.yaw_ref = yaw_of(self.base_pose()[1])
        h = -self.yaw_ref / 2
        q = quat_mul(np.array([math.cos(h), 0.0, 0.0, math.sin(h)]), q)       # 시작 yaw 기준 (Madgwick 처럼)
        if self.args.imu_yaw_drift_dps:   # 실기 6 축 IMU 처럼 yaw 가 서서히 흐르게 (자이로 바이어스 적분)
            hd = math.radians(self.args.imu_yaw_drift_dps) * self.d.time / 2
            q = quat_mul(np.array([math.cos(hd), 0.0, 0.0, math.sin(hd)]), q)
        w, x, y, z = q
        msg = Imu()
        msg.header.stamp, msg.header.frame_id = stamp, "imu_link"
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = x, y, z, w
        g = self.d.sensordata[self.s_gyro[0]:self.s_gyro[0] + 3]
        a = self.d.sensordata[self.s_acc[0]:self.s_acc[0] + 3]
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (float(v) for v in g)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (float(v) for v in a)
        grav = Vector3Stamped()
        grav.header = msg.header
        grav.vector.x = -2.0 * (x * z - w * y)
        grav.vector.y = -2.0 * (w * x + y * z)
        grav.vector.z = -(w * w - x * x - y * y + z * z)
        return msg, grav

    def post_step(self, k, stamp):
        imu, grav = self.imu_msg(stamp)
        self.pub_imu.publish(imu)
        self.pub_grav.publish(grav)
        if k % 4 == 0:
            self.pub_imu50.publish(imu)
        now = time.monotonic()
        if self.init_settle is not None:
            s = self.init_settle
            err = np.abs(self.d.qpos[self.qadr] - self.target).max()
            s["count"] = s["count"] + 1 if err < s["tol"] else 0
            if (k % 4 == 0 and s["count"] >= s["need"] * 4) or now > s["deadline"]:
                for j, i in self.idx.items():
                    if j not in NECK:
                        self.prof[i] = s["final"] * RPM
                self.init_settle = None
                self.behavior = dict(behavior="none", active=False, phase="", phase_idx=0, total_phases=0)
                self.log.info(f"[{self.name}] robot_init 완료 (max err {err:.3f} rad)")
        if self.behavior_until is not None and now > self.behavior_until:
            self.place_upright()
            self.behavior_until = None
            self.behavior = dict(behavior="none", active=False, phase="", phase_idx=0, total_phases=0)

    # ── 카메라 + 검출 (모드 A: 정답 위치) ──
    def camera_pose(self):
        pos, R = self.base_pose()
        qy = self.d.qpos[self.qadr[self.neck_i[0]]]
        qp = self.d.qpos[self.qadr[self.neck_i[1]]]
        R1 = R @ rz(qy)
        p1 = pos + R @ NECK_YAW_ORIGIN
        R2 = R1 @ ry(qp)
        p2 = p1 + R1 @ NECK_PITCH_ORIGIN
        pc = p2 + R2 @ CAMERA_ORIGIN
        return pc, R2 @ R_LINK_OPT

    def publish_lines(self, stamp, field_lines):
        """필드 선 검출 시뮬 (모드 A: 정답 + 노이즈). 바닥 위 선분을 카메라 시야(87×58°)·깊이(0.2 m~max_range)로
        자르고 끝점에 검출과 같은 오차 모델, 선분별 누락 10 %, 로봇·공 가림(근사). 센터서클은 없음."""
        if self.pub_lines is None:
            return
        pc, Rwo = self.camera_pose()
        tx, ty = math.tan(HFOV / 2), math.tan(VFOV / 2)
        arr = LineSegmentArray()
        arr.header.stamp, arr.header.frame_id = stamp, "camera_color_optical_frame"
        for a_w, b_w in field_lines:
            a = Rwo.T @ (a_w - pc)
            b = Rwo.T @ (b_w - pc)
            d = b - a
            t0, t1 = 0.0, 1.0
            # g(p) >= 0 꼴 제약 6 개 (깊이 최소/최대, 좌우·상하 시야)
            for ga, gd in ((a[2] - 0.2, d[2]), (self.args.max_range - a[2], -d[2]),
                           (tx * a[2] - a[0], tx * d[2] - d[0]), (tx * a[2] + a[0], tx * d[2] + d[0]),
                           (ty * a[2] - a[1], ty * d[2] - d[1]), (ty * a[2] + a[1], ty * d[2] + d[1])):
                if abs(gd) < 1e-12:
                    if ga < 0:
                        t0, t1 = 1.0, 0.0
                        break
                    continue
                t = -ga / gd
                if gd > 0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
            if t1 - t0 <= 0.0:
                continue
            p0, p1 = a + t0 * d, a + t1 * d
            if np.linalg.norm(p1 - p0) < 0.3 or self.rng.random() < 0.10:
                continue
            # 가림: 선분 위 5 점 중 3 점 이상 가려지면 버림 (부분 가림은 근사)
            samples = [Rwo @ (p0 + f * (p1 - p0)) + pc for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
            if sum(self.occluded(pc, samples, [-1] * 5)) >= 3:
                continue
            seg = LineSegment()
            for P, out in ((p0, seg.start), (p1, seg.end)):
                P = self.noisy(P)
                out.x, out.y, out.z = (float(v) for v in P)
            arr.segments.append(seg)
        self.pub_lines.publish(arr)

    # ── 검출 오차 모델 (사용자 실측 경험, 2026-10-31) ──
    #   깊이(카메라→물체 방향) σ: 2.5 m 까지 선형으로 10 cm, 그 뒤 급증해 4 m 에서 50 cm (작은 로봇이라 먼 물체 픽셀이 적다)
    #   옆 방향(방위) σ: 0.5°. 오검출은 넣지 않는다 (실기에서도 공에 가끔뿐).
    def range_sigma(self, d):
        ks = self.sim.depth_knots
        if d <= ks[0][0]:
            return ks[0][1]
        for (d0, s0), (d1, s1) in zip(ks, ks[1:]):
            if d <= d1:
                return s0 + (s1 - s0) * (d - d0) / (d1 - d0)
        (d0, s0), (d1, s1) = ks[-2], ks[-1]
        return s1 + (s1 - s0) * (d - d1) / (d1 - d0)

    def noisy(self, p):
        if self.args.noise_model == "legacy":
            dist = float(np.linalg.norm(p))
            return p + self.rng.normal(0.0, self.args.det_sigma0 + self.args.det_sigma_k * dist * dist, 3)
        d = float(np.linalg.norm(p))
        u = p / max(d, 1e-6)
        e1 = np.cross(u, [0.0, 1.0, 0.0])
        e1 /= max(np.linalg.norm(e1), 1e-6)
        e2 = np.cross(u, e1)
        sl = d * math.tan(math.radians(self.args.bearing_sigma_deg))
        return p + u * self.rng.normal(0.0, self.range_sigma(d)) + e1 * self.rng.normal(0.0, sl) \
            + e2 * self.rng.normal(0.0, sl)

    def occluded(self, pc, targets, owners):
        """가림: 카메라에서 각 목표점까지 광선이 다른 로봇·공에 먼저 맞으면 True. 경기장(정적 geom)은 무시,
        자기 머리는 제외. 목표 자신(공·그 로봇)에 맞은 건 보이는 것으로 친다."""
        n = len(targets)
        if n == 0 or self.args.no_occlusion:
            return [False] * n
        vec = np.ascontiguousarray((np.asarray(targets) - pc).reshape(-1), dtype=np.float64)
        geomid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n, dtype=np.float64)
        mujoco.mj_multiRay(self.m, self.d, pc.astype(np.float64), vec, None, 0, self.cam_bid, geomid, dist, None, n, 1e6)
        out = []
        sim = self.sim
        for i in range(n):
            g = geomid[i]
            if g < 0 or dist[i] < 0:
                out.append(False)
                continue
            seg_len = float(np.linalg.norm(vec[3 * i:3 * i + 3]))
            if dist[i] * seg_len > seg_len - 0.12:          # 목표 바로 앞(12 cm)까지는 가림 아님
                out.append(False)
                continue
            own = owners[i]
            if own == -2 and g in sim.ball_geom_set:          # 공 자신
                out.append(False)
            elif own >= 0 and sim.geom_robot[g] == own:      # 그 로봇 자신
                out.append(False)
            else:
                out.append(True)
        return out

    def publish_detections(self, stamp, objects):
        pc, Rwo = self.camera_pose()
        arr = Detection3DArray()
        arr.header.stamp, arr.header.frame_id = stamp, "camera_color_optical_frame"
        tx, ty = math.tan(HFOV / 2), math.tan(VFOV / 2)
        cands = []
        for cls, pw, owner in objects:
            po = Rwo.T @ (pw - pc)
            zdep = po[2]
            # 어떤 게이트에 걸렸는지 (공만 진단 토픽으로 낸다)
            if zdep <= 0:
                why = "behind"
            elif abs(po[0]) > tx * zdep:
                why = "out_hfov"
            elif abs(po[1]) > ty * zdep:
                why = "out_vfov"
            elif zdep < MIN_DEPTH:
                why = "too_close_depth"
            elif zdep > self.args.max_range:
                why = "too_far"
            elif cls == "ball" and zdep < BALL_MIN_DEPTH:
                why = "ball_min_dist"
            elif self.rng.random() < self.args.det_dropout:
                why = "dropout"
            else:
                why = "ok"
            cands.append([cls, pw, owner, po, why])
        ok = [c for c in cands if c[4] == "ok"]
        occ = self.occluded(pc, [c[1] for c in ok], [c[2] for c in ok])
        for c, o in zip(ok, occ):
            if o:
                c[4] = "occluded"
        for cls, pw, owner, po, why in cands:
            if cls == "ball":
                zdep = po[2]
                pb, Rb = self.base_pose()
                bb = Rb.T @ (pw - pb)
                self.pub_ball_vis.publish(String(data=json.dumps(dict(
                    why=why, z=round(float(zdep), 3),
                    h_deg=round(math.degrees(math.atan2(po[0], zdep)), 1) if zdep > 0 else None,
                    v_deg=round(math.degrees(math.atan2(po[1], zdep)), 1) if zdep > 0 else None,
                    ball_b=[round(float(bb[0]), 3), round(float(bb[1]), 3)],
                    neck_deg=[round(math.degrees(self.d.qpos[self.qadr[i]]), 1) for i in self.neck_i]))))
            if why != "ok":
                continue
            po = self.noisy(po)
            det = Detection3D()
            det.header = arr.header
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = cls, 0.9
            h.pose.pose.position.x, h.pose.pose.position.y, h.pose.pose.position.z = (float(v) for v in po)
            h.pose.pose.orientation.w = 1.0
            det.results.append(h)
            det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.center.position.z = (float(v) for v in po)
            det.bbox.center.orientation.w = 1.0
            det.bbox.size.x = det.bbox.size.y = CLASS_SIZE.get(cls, 0.1)
            arr.detections.append(det)
        self.pub_det.publish(arr)

    def publish_gt(self, stamp, ball_w):
        pos, R = self.base_pose()
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp, "map"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in pos)
        q = self.d.xquat[self.bid]
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = (float(v) for v in q)
        self.pub_gt_pose.publish(ps)
        pt = PointStamped()
        pt.header = ps.header
        pt.point.x, pt.point.y, pt.point.z = (float(v) for v in ball_w)
        self.pub_gt_ball.publish(pt)
        # RViz 용 정답 표시: 초록 원기둥 = 실제 로봇, 초록 화살표 = 실제 방향, 흰 공 = 실제 공
        yaw = yaw_of(R)
        arr = MarkerArray()
        for i, (typ, sx, sy, sz, z, rgba) in enumerate((
                (Marker.CYLINDER, 0.22, 0.22, 0.02, 0.01, (0.1, 0.9, 0.2, 0.6)),
                (Marker.ARROW, 0.45, 0.05, 0.05, 0.05, (0.1, 0.9, 0.2, 1.0)),
                (Marker.SPHERE, 0.13, 0.13, 0.13, 0.065, (1.0, 1.0, 1.0, 1.0)))):
            mk = Marker()
            mk.header = ps.header
            mk.ns, mk.id, mk.type, mk.action = "sim_gt", i, typ, Marker.ADD
            if typ == Marker.SPHERE:
                mk.pose.position.x, mk.pose.position.y = float(ball_w[0]), float(ball_w[1])
            else:
                mk.pose.position.x, mk.pose.position.y = float(pos[0]), float(pos[1])
                mk.pose.orientation.z, mk.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            mk.pose.position.z = z
            mk.scale.x, mk.scale.y, mk.scale.z = sx, sy, sz
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = rgba
            arr.markers.append(mk)
        txt = Marker()
        txt.header = ps.header
        txt.ns, txt.id, txt.type, txt.action = "sim_gt", 9, Marker.TEXT_VIEW_FACING, Marker.ADD
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(pos[0]), float(pos[1]), 0.6
        txt.scale.z = 0.15
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f"GT {self.name}"
        arr.markers.append(txt)
        self.pub_gt_markers.publish(arr)

    def publish_initialpose(self):
        pos, R = self.base_pose()
        yaw = yaw_of(R)
        msg = PoseWithCovarianceStamped()
        msg.header.stamp, msg.header.frame_id = self.node.get_clock().now().to_msg(), "map"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(pos[0]), float(pos[1])
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.01
        cov[35] = 0.01
        msg.pose.covariance = cov
        self.pub_initpose.publish(msg)
        self.log.info(f"[{self.name}] /initialpose = 정답 ({pos[0]:+.2f}, {pos[1]:+.2f}, {math.degrees(yaw):+.0f}°)")

    def publish_slow(self):
        self.pub_behavior.publish(String(data=json.dumps(self.behavior)))
        # 토크 상태는 바뀔 때만 보내던 latched 토픽이라, 한 번 전달이 빠지면 fall_manager 가 계속 STOPPED 로 남았다
        # (4v4 에서 파랑 4 번이 START 뒤에도 정지). 1 s 마다 다시 보낸다 (fall_manager 는 바뀔 때만 반응).
        self._slow_n = getattr(self, "_slow_n", 0) + 1
        if self._slow_n % 20 == 0:
            self.pub_torque.publish(Bool(data=self.torque_reported))

    def publish_diag(self):
        motors = [dict(id=i + 1, name=j, pwm=0, current_ma=0, voltage_v=12.0, temp_c=35, hw_error=0)
                  for i, j in enumerate(JS_ORDER)]
        self.pub_diag.publish(String(data=json.dumps(dict(stamp=time.time(), motors=motors))))


class Sim:
    def __init__(self, args):
        self.args = args
        meta = json.load(open(args.scene + ".json"))
        self.m = mujoco.MjModel.from_xml_path(args.scene + ".xml")
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.depth_knots = [tuple(float(v) for v in kv.split(":")) for kv in args.depth_sigma.split(",")]
        self.robots = [RobotHW(self, self.m, self.d, r, args) for r in meta["robots"]]
        bj = self.m.joint("ball_root")
        self.bq, self.bv = self.m.jnt_qposadr[bj.id], self.m.jnt_dofadr[bj.id]
        if args.spawn:   # 첫 로봇 시작 위치 덮어쓰기 "x,y,yaw_deg" (반복 시험용)
            x, y, yd = (float(v) for v in args.spawn.split(","))
            r0 = self.robots[0]
            r0.spawn = (x, y, math.radians(yd))
            self.d.qpos[r0.rq:r0.rq + 3] = [x, y, 0.44]
            h = math.radians(yd) / 2
            self.d.qpos[r0.rq + 3:r0.rq + 7] = [math.cos(h), 0.0, 0.0, math.sin(h)]
            mujoco.mj_forward(self.m, self.d)
        if args.ball:
            bx, by = (float(v) for v in args.ball.split(","))
            self.d.qpos[self.bq:self.bq + 3] = [bx, by, 0.065]
        self.paint_teams()
        for r in self.robots:   # 기본자세로 세워 두고 시작
            r.place_upright()
            self.d.qpos[r.rq + 2] = 0.44
        mujoco.mj_forward(self.m, self.d)
        self.ball_mass = self.m.body("ball").mass[0]
        self.ball_I = self.m.body("ball").inertia.copy()
        geom = field_geometry_for_profile(args.field_profile)
        lms = generate_field_landmarks(geom, landmark_classes=(1, 5, 6, 7))
        hl, hw = geom.half_length_m, geom.half_width_m
        pa, pw = geom.penalty_area_inner_x_m, geom.penalty_area_half_width_m
        ga, gw = geom.goal_area_inner_x_m, geom.goal_area_half_width_m
        segs = [(-hl, hw, hl, hw), (-hl, -hw, hl, -hw), (0, -hw, 0, hw)]
        for sx in (-1, 1):
            segs += [(sx * hl, -hw, sx * hl, hw), (sx * pa, -pw, sx * pa, pw), (sx * pa, pw, sx * hl, pw),
                     (sx * pa, -pw, sx * hl, -pw), (sx * ga, -gw, sx * ga, gw), (sx * ga, gw, sx * hl, gw),
                     (sx * ga, -gw, sx * hl, -gw)]
        self.field_lines = [(np.array([x0, y0, 0.0]), np.array([x1, y1, 0.0])) for x0, y0, x1, y1 in segs]
        self.landmarks = [(CLASS_NAME[c], np.array([lm.x, lm.y, GOALPOST_Z if c == 1 else 0.0]))
                          for c, L in lms.items() for lm in L]
        self.ball_out = None
        self.score = {"GOAL_US": 0, "GOAL_THEM": 0}
        # 심판 스크립트용: 공 재배치 (첫 로봇 도메인의 /sim/place_ball, 월드 좌표 x, y)
        self.ball_request = None
        n0 = self.robots[0].node
        n0.create_subscription(PointStamped, "/sim/place_ball", self.cb_place_ball, 10)
        # 심판 스크립트용 (첫 로봇 도메인): 로봇 재배치 "이름 x y yaw_deg", 마지막 공 접촉, 로봇 위치, 점수판
        n0.create_subscription(String, "/sim/place_robot", self.cb_place_robot, 10)
        n0.create_subscription(String, "/sim/scoreboard", self.cb_scoreboard, 10)
        self.pub_touch = n0.create_publisher(String, "/sim/ball_touch", 10)
        self.pub_robots = n0.create_publisher(String, "/sim/robots", 10)
        # 공 접촉 판정: geom → 로봇 번호 (-1 = 로봇 아님)
        prefix_of = [r.p for r in self.robots]
        self.geom_robot = np.full(self.m.ngeom, -1, dtype=int)
        for g in range(self.m.ngeom):
            bname = self.m.body(self.m.geom_bodyid[g]).name
            for i, pf in enumerate(prefix_of):
                if bname.startswith(pf):
                    self.geom_robot[g] = i
                    break
        ball_bid = self.m.body("ball").id
        self.ball_geoms = np.where(self.m.geom_bodyid == ball_bid)[0]
        self.ball_geom_set = set(int(g) for g in self.ball_geoms)
        self.last_touch = None
        self.scoreboard = None
        self.scoreboard_shown = None
        self.loading_msg = None
        self.loading_cache = (None, None)
        self.viewer = None
        if args.viewer:
            from mujoco import viewer as mj_viewer
            # 좌우 패널은 기본으로 숨김 (화면에서 Tab = 왼쪽, Shift+Tab = 오른쪽 패널 다시 켜기). --viewer_ui 면 표시
            ui = bool(getattr(args, "viewer_ui", False))
            self.viewer = mj_viewer.launch_passive(self.m, self.d, show_left_ui=ui, show_right_ui=ui)
            with self.viewer.lock():                      # 처음 화면: 경기장 전체를 비스듬히 내려다보기
                self.viewer.cam.lookat[:] = [0.0, -0.5, 0.0]
                self.viewer.cam.distance = 10.0
                self.viewer.cam.elevation = -50.0
                self.viewer.cam.azimuth = 90.0

    # 팀 색 (장면 json 순서대로 첫 팀 = 파랑, 둘째 팀 = 빨강). 골키퍼(1 번)는 다른 색 — 실제 경기 유니폼 규칙처럼
    TEAM_COLORS = [((0.15, 0.35, 0.90), (0.10, 0.80, 0.85)),    # 필드 선수, 골키퍼
                   ((0.90, 0.18, 0.15), (1.00, 0.62, 0.10))]

    def paint_teams(self):
        """로봇 이름 t<팀>p<번호> 로 팀을 나눠 몸통(base_link)만 팀 색으로 칠한다 (나머지 링크는 기본 회색).
        골키퍼(1 번)는 다른 색. 머리 위 번호 표시(draw_labels)에 쓸 색·번호도 여기서 정한다."""
        teams = []
        for r in self.robots:
            mt = re.match(r"t(\d+)p(\d+)", r.name)
            if mt and int(mt.group(1)) not in teams:
                teams.append(int(mt.group(1)))
        for r in self.robots:
            r.team_rgb, r.number = None, None
            mt = re.match(r"t(\d+)p(\d+)", r.name)
            if not mt or teams.index(int(mt.group(1))) >= len(self.TEAM_COLORS):
                continue
            field, gk = self.TEAM_COLORS[teams.index(int(mt.group(1)))]
            r.number = int(mt.group(2))
            r.team_rgb = np.array(gk if r.number == 1 else field)
            for g in range(self.m.ngeom):
                if self.m.body(self.m.geom_bodyid[g]).name == r.p + "base_link" and self.m.geom_rgba[g, 3] > 0:
                    self.m.geom_rgba[g, :3] = r.team_rgb
            r.head_bid = self.m.body(r.p + "neck_pitch_link").id

    # 7 세그먼트 숫자: 세그먼트 → (가로 위치 ×폭, 세로 위치 ×높이, 가로 막대인가)
    _SEG = {"a": (0, .5, 1), "b": (.5, .25, 0), "c": (.5, -.25, 0), "d": (0, -.5, 1), "e": (-.5, -.25, 0),
            "f": (-.5, .25, 0), "g": (0, 0, 1)}
    _DIGIT = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc", 5: "afgcd", 6: "afgedc", 7: "abc",
              8: "abcdefg", 9: "abcdfg"}

    def draw_labels(self):
        """머리 위 선수 표시 (게임처럼): 반투명 팀 색 역삼각뿔 + 팀 색 7 세그먼트 번호 (항상 카메라 쪽을 본다).
        MuJoCo 라벨 글자는 뷰어 글꼴 크기에 묶여 못 키워서 숫자를 막대로 그린다. 뷰어 사용자 장면에 매번 다시 그린다."""
        if self.viewer is None:
            return
        try:
            marker = 2 * self.m.mesh("player_marker").id     # mjvGeom 의 메시 dataid = 2 × 메시 id
        except KeyError:
            marker = -1
        cam = self.viewer.cam
        az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
        fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= max(np.linalg.norm(right), 1e-6)
        up = np.cross(right, fwd)
        M = np.column_stack([right, up, fwd]).flatten()
        k = 1.0                                  # 고정 크기 (카메라 거리 비례 확대는 너무 커서 뺐다, 2026-10-31 사용자)
        H, W = 0.19 * k, 0.095 * k              # 숫자 높이·폭 [m]
        t = H * 0.07                            # 막대 굵기
        edge = 0.010                            # 흰 테두리 두께 [m]

        def add(scn, typ, size, pos, mat, rgba, dataid=-1, emission=0.0):
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), mat,
                                np.asarray(rgba, np.float32))
            if dataid >= 0:
                g.dataid = dataid
            g.emission = emission
            g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)   # 장식: 바닥에 그림자 안 드리움
            g.transparent = 0
            scn.ngeom += 1

        with self.viewer.lock():
            scn = self.viewer.user_scn
            scn.ngeom = 0
            for r in self.robots:
                if r.number is None or scn.ngeom + 16 >= scn.maxgeom:
                    continue
                if time.time() - r.ball_owner_t < 0.3 and scn.ngeom + 16 + self.RING_SEG < scn.maxgeom:
                    self.draw_owner_ring(scn, r)
                base = self.d.xpos[r.head_bid] + np.array([0.0, 0.0, 0.20 + 0.12 * k])
                if marker >= 0:                  # 메시 크기는 회전 행렬에 배율을 곱해 바꾼다
                    add(scn, mujoco.mjtGeom.mjGEOM_MESH, [1, 1, 1], base, (k * np.eye(3)).flatten(),
                        [*r.team_rgb, 0.55], dataid=marker, emission=0.3)
                c0 = base + np.array([0.0, 0.0, 0.05 * k + H / 2])
                for ch in self._DIGIT.get(r.number % 10, ""):
                    u, w, horiz = self._SEG[ch]
                    pos = c0 + right * u * W + up * w * H
                    # 흰 테두리: 조금 큰 흰 막대를 (카메라에서 볼 때) 바로 뒤에, 그 앞에 팀 색 막대
                    add(scn, mujoco.mjtGeom.mjGEOM_BOX,
                        [W / 2 + edge, t + edge, t] if horiz else [t + edge, H / 4 + edge, t],
                        pos + fwd * (t + 0.003), M, [1.0, 1.0, 1.0, 1.0], emission=1.0)
                    add(scn, mujoco.mjtGeom.mjGEOM_BOX, [W / 2, t, t] if horiz else [t, H / 4, t],
                        pos, M, [*r.team_rgb, 1.0], emission=0.6)   # 몸통과 같은 색
            if self.loading_msg and Image is not None:
                self.draw_loading(scn, add, cam, fwd, right, up, M)

    RING_SEG = 28

    def draw_owner_ring(self, scn, r, radius=0.24, width=0.035):
        """공 담당 로봇 발밑 고리 (팀 색). 각 로봇이 /hsl/strategy_state.active 로 '내가 담당' 이라 알린 동안만 그린다
        — 팀원끼리 판단이 엇갈리는 순간(담당 교대)에는 두 개가 잠깐 보일 수 있다 (분산 결정 그대로 보여 줌)."""
        cx, cy = self.d.xpos[r.bid][:2]
        z = 0.004
        pts = [(cx + radius * math.cos(a), cy + radius * math.sin(a), z)
               for a in np.linspace(0.0, 2 * math.pi, self.RING_SEG + 1)]
        rgba = np.array([*r.team_rgb, 1.0], np.float32)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width / 2, np.array(p0), np.array(p1))
            g.emission = 0.5
            g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            g.transparent = 0
            scn.ngeom += 1

    def draw_loading(self, scn, add, cam, fwd, right, up, M):
        """'PREPARING MATCH' 판: 카메라 바로 앞 1 m 에 카메라를 향해 그려 화면 정가운데에 보인다 (카메라를 돌려도 따라옴).
        MuJoCo 글자 판(set_texts)은 정가운데 자리가 없고 빈 줄로 내리면 배경이 위부터 길게 칠해지며,
        그림 판(set_images)은 교체할 때 뷰어가 죽어서 (3.17, 메모리 손상) 글자를 화소 막대로 그린다."""
        if self.loading_cache[0] != self.loading_msg:
            self.loading_cache = (self.loading_msg, (text_rects("PREPARING MATCH", FONT_BOLD, 26),
                                                     text_rects(self.loading_msg, FONT_REGULAR, 16)))
        (w1, h1, r1), (w2, h2, r2) = self.loading_cache[1]
        if scn.ngeom + len(r1) + len(r2) + 1 >= scn.maxgeom:
            return
        dist = 1.0                                               # 카메라 앞 거리 [m] (near 면보다 충분히 멀다)
        half_h = dist * math.tan(math.radians(self.m.vis.global_.fovy) / 2)
        px = 2 * half_h * 0.08 / h1                              # 제목 높이 = 화면 높이의 8 %
        gap, pad = 0.45 * h1, 0.6 * h1                           # 줄 간격·여백 [화소]
        vp = self.viewer.viewport
        if vp is not None and vp.height > 0:                     # 창이 좁으면 폭에 맞춰 줄인다
            px = min(px, 2 * half_h * vp.width / vp.height * 0.9 / (max(w1, w2) + 2 * pad))
        block_h = h1 + gap + h2
        centre = np.array(cam.lookat) - fwd * (cam.distance - dist)   # 자유 카메라 위치 + 앞으로 dist
        depth = 0.3 * px
        for rects, w, y_top, rgb in ((r1, w1, block_h / 2, (1.0, 1.0, 1.0)),
                                     (r2, w2, block_h / 2 - h1 - gap, (0.85, 0.87, 0.9))):
            for x0, y0, x1, y1 in rects:
                pos = centre + right * (px * ((x0 + x1) / 2 - w / 2)) + up * (px * (y_top - (y0 + y1) / 2))
                add(scn, mujoco.mjtGeom.mjGEOM_BOX, [px * (x1 - x0) / 2, px * (y1 - y0) / 2, depth],
                    pos, M, [*rgb, 1.0], emission=1.0)
        add(scn, mujoco.mjtGeom.mjGEOM_BOX,                      # 반투명 어두운 배경 (글자 바로 뒤)
            [px * (max(w1, w2) / 2 + pad), px * (block_h / 2 + pad), depth], centre + fwd * (3 * depth),
            M, [0.05, 0.06, 0.07, 0.6])

    def referee(self, t_sim):
        """골·아웃 판정만 한다 (공 재배치·GC 연동은 아직 사람이 한다). +x 골 = 우리 팀 득점."""
        x, y, z = self.d.qpos[self.bq:self.bq + 3]
        r, hx, hy, ghw = 0.065, 4.5, 3.0, 0.9
        if abs(x) > hx + r and abs(y) < ghw and z < 1.0:
            ev = "GOAL_US" if x > 0 else "GOAL_THEM"
        elif abs(x) > hx + r or abs(y) > hy + r:
            ev = "OUT_GOAL_LINE" if abs(x) > hx + r else "OUT_TOUCH_LINE"
        else:
            self.ball_out = None
            return
        if self.ball_out == ev or (self.ball_out or "").startswith("GOAL"):
            return
        self.ball_out = ev
        if ev.startswith("GOAL"):
            self.score[ev] += 1
        msg = (f"{ev} t={t_sim:.1f}s ball=({x:+.2f},{y:+.2f}) score US {self.score['GOAL_US']} : "
               f"{self.score['GOAL_THEM']} THEM last={self.last_touch or '-'}")
        self.robots[0].log.warn(f"[referee] {msg}")
        for rb in self.robots:
            rb.pub_event.publish(String(data=msg))

    def cb_place_robot(self, msg):
        try:
            name, x, y, yd = msg.data.split()
            r = next(r for r in self.robots if r.name == name)
            r.reset_request = (float(x), float(y), math.radians(float(yd)))
        except (ValueError, StopIteration):
            self.robots[0].log.warn(f"/sim/place_robot 형식 오류: '{msg.data}' (이름 x y yaw_deg)")

    def cb_scoreboard(self, msg):
        self.scoreboard = msg.data

    def update_touch(self):
        """공에 닿은 로봇 (마지막 접촉) — 세트플레이(킥인·골킥·코너킥) 판정용."""
        nc = self.d.ncon
        if nc == 0:
            return
        g1 = self.d.contact.geom1[:nc]
        g2 = self.d.contact.geom2[:nc]
        b1 = np.isin(g1, self.ball_geoms)
        b2 = np.isin(g2, self.ball_geoms)
        other = np.concatenate([g2[b1], g1[b2]])
        idx = self.geom_robot[other]
        idx = idx[idx >= 0]
        if len(idx):
            name = self.robots[int(idx[0])].name
            if name != self.last_touch:
                self.last_touch = name
                self.pub_touch.publish(String(data=name))

    def publish_robots(self):
        out = {}
        for r in self.robots:
            pos, R = r.base_pose()
            out[r.name] = [round(float(pos[0]), 3), round(float(pos[1]), 3), round(math.degrees(yaw_of(R)), 1),
                           bool(R[2, 2] > 0.7)]
        self.pub_robots.publish(String(data=json.dumps(out)))

    def show_loading(self):
        """심판 점수판이 오기 전(경기 준비 중) 화면 정가운데 표시: 로봇 소프트웨어 기동 → START·기립 → GC 대기.
        글자는 draw_loading 이 3D 로 그린다 (PIL 이 없으면 위쪽 가운데 큰 글자)."""
        n = len(self.robots)
        up = sum(1 for r in self.robots if r.pub_js.get_subscription_count() >= 2)   # 스택이 관절 상태를 구독
        started = sum(1 for r in self.robots if r.started)
        ready = sum(1 for r in self.robots if r.started and r.init_settle is None)
        if started == 0:
            msg = f"robot software starting   {up}/{n}"
        elif ready < n:
            msg = f"robots standing up   {ready}/{n}"
        else:
            msg = "waiting for GameController"
        self.loading_msg = msg
        if Image is None and msg != self.scoreboard_shown:
            self.scoreboard_shown = msg
            self.viewer.set_texts([(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOP,
                                    f"PREPARING MATCH\n{msg}", "")])

    def show_scoreboard(self):
        """뷰어 점수판: 왼쪽 위 = 점수 (큰 글자), 오른쪽 위 = 게임컨트롤러 상태 (심판이 /sim/scoreboard 로 보낸 JSON).
        점수판이 오기 전에는 '준비 중' 표시."""
        if self.viewer is None:
            return
        if self.scoreboard is None:
            self.show_loading()
            return
        if self.scoreboard == self.scoreboard_shown:
            return
        self.scoreboard_shown = self.scoreboard
        self.loading_msg = None
        try:
            sb = json.loads(self.scoreboard)
        except ValueError:
            return
        col = lambda rows, i: "\n".join(str(r[i]) for r in rows)  # noqa: E731
        texts = []
        if sb.get("score"):
            texts.append((mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          col(sb["score"], 0), col(sb["score"], 1)))
        if sb.get("gc"):
            texts.append((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                          col(sb["gc"], 0), col(sb["gc"], 1)))
        self.viewer.set_texts(texts)

    def cb_place_ball(self, msg):
        self.ball_request = (msg.point.x, msg.point.y)      # 물리 스텝 사이(메인 루프)에서 적용

    def apply_place_ball(self):
        if self.ball_request is None:
            return
        x, y = self.ball_request
        self.ball_request = None
        msg = PointStamped()
        msg.point.x, msg.point.y = x, y
        self.last_touch = None            # 놓인 공은 아무도 안 건드린 상태로
        self.d.qpos[self.bq:self.bq + 3] = [msg.point.x, msg.point.y, 0.065]
        self.d.qpos[self.bq + 3:self.bq + 7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[self.bv:self.bv + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.ball_out = None
        self.robots[0].log.info(f"[referee] 공 재배치 ({msg.point.x:+.2f}, {msg.point.y:+.2f})")

    def ball_damping(self):
        v = self.d.qvel[self.bv:self.bv + 6]
        self.d.qfrc_applied[self.bv:self.bv + 3] = -BALL_LIN_DAMP * self.ball_mass * v[:3]
        self.d.qfrc_applied[self.bv + 3:self.bv + 6] = -BALL_ANG_DAMP * self.ball_I * v[3:]

    def objects_for(self, robot):
        ball = self.d.qpos[self.bq:self.bq + 3].copy()
        objs = [("ball", ball, -2)] + [(c, p, -1) for c, p in self.landmarks]
        for i, r in enumerate(self.robots):
            if r is not robot:
                objs.append(("robot", r.base_pose()[0] - np.array([0.0, 0.0, 0.15]), i))
        return objs, ball

    def run(self):
        dt = self.m.opt.timestep
        # 로봇마다 executor 를 스레드에서 블로킹 대기로 돌린다 (메시지가 올 때만 깨어남).
        # 예전엔 메인 루프가 남는 시간 내내 spin_once 를 돌려 8 대에서 실시간의 87 % 로 느려졌다.
        for r in self.robots:
            threading.Thread(target=r.exe.spin, daemon=True).start()
        t0 = time.monotonic()
        k = 0
        next_det = next_gt = next_slow = next_diag = 0.0
        lag_warned = 0.0
        while all(rclpy.ok(context=r.ctx) for r in self.robots):
            t_sim = k * dt
            stamp = self.robots[0].node.get_clock().now().to_msg()
            if k % 4 == 0:
                for r in self.robots:
                    r.begin_cycle(stamp)
            for r in self.robots:
                r.pre_step(dt)
            self.ball_damping()
            mujoco.mj_step(self.m, self.d)
            k += 1
            for r in self.robots:
                r.post_step(k, stamp)
            if t_sim >= next_det:
                next_det += 1.0 / self.args.det_hz
                for r in self.robots:
                    objs, _ = self.objects_for(r)
                    r.publish_detections(stamp, objs)
                    r.publish_lines(stamp, self.field_lines)
            if k % 4 == 0:
                self.update_touch()
            if t_sim >= next_gt:
                next_gt += 0.1
                self.referee(t_sim)
                if int(round(t_sim * 10)) % 2 == 0:      # 5 Hz (심판의 밀기·장외 판정용)
                    self.publish_robots()
                ball = self.d.qpos[self.bq:self.bq + 3]
                for r in self.robots:
                    r.publish_gt(stamp, ball)
            if t_sim >= next_slow:
                next_slow += 0.05
                for r in self.robots:
                    r.publish_slow()
            if t_sim >= next_diag:
                next_diag += 0.5
                for r in self.robots:
                    r.publish_diag()
            if self.viewer is not None and k % 8 == 0:
                self.show_scoreboard()
                self.draw_labels()
                self.viewer.sync()
            # ── 벽시계 1 배속: 남는 시간은 잔다 (콜백은 로봇별 executor 스레드가 처리) ──
            self.apply_place_ball()
            for r in self.robots:
                if r.reset_request is not None:
                    r.place_upright(r.reset_request)
                    r.reset_request = None
                    mujoco.mj_forward(self.m, self.d)
            deadline = t0 + k * dt
            remain = deadline - time.monotonic()
            if remain > 0:
                time.sleep(remain)
            lag = time.monotonic() - deadline
            if lag > 0.05 and time.monotonic() - lag_warned > 5.0:
                self.robots[0].log.warn(f"실시간보다 {lag * 1000:.0f} ms 뒤처짐 — 따라잡지 않고 기준 시각을 옮긴다")
                lag_warned = time.monotonic()
                t0 += lag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="models/<name> (.xml/.json 확장자 없이)")
    ap.add_argument("--field_profile", default="s_field")
    ap.add_argument("--det_hz", type=float, default=30.0)
    ap.add_argument("--det_dropout", type=float, default=0.05)
    ap.add_argument("--det_sigma0", type=float, default=0.01, help="검출 위치 노이즈 σ = σ0 + k·d² [m]")
    ap.add_argument("--det_sigma_k", type=float, default=0.01)
    ap.add_argument("--viewer", action="store_true", help="MuJoCo 화면 띄우기")
    ap.add_argument("--viewer_ui", action="store_true", help="뷰어 좌우 패널도 표시 (기본 숨김, 화면에서 Tab/Shift+Tab)")
    ap.add_argument("--no_lines", action="store_true", help="필드 선 검출(/perception/field_lines) 시뮬 끄기")
    ap.add_argument("--noise_model", default="range", choices=["range", "legacy"],
                    help="range = 깊이 방향 오차(--depth_sigma)+방위 오차, legacy = 등방 σ0+k·d² (예전)")
    ap.add_argument("--depth_sigma", default="0:0.005,2.5:0.10,4.0:0.50",
                    help="깊이 방향 σ 매듭 '거리:σ,...' [m] (사이는 선형, 끝은 마지막 기울기로 외삽)")
    ap.add_argument("--bearing_sigma_deg", type=float, default=0.5)
    ap.add_argument("--max_range", type=float, default=4.5, help="검출 최대 깊이 [m]")
    ap.add_argument("--no_occlusion", action="store_true", help="로봇·공 가림 끄기")
    ap.add_argument("--imu_yaw_drift_dps", type=float, default=0.0, help="IMU yaw 드리프트 [deg/s] (실기 자이로 바이어스 흉내)")
    ap.add_argument("--spawn", default=None, help="첫 로봇 시작 위치 x,y,yaw_deg (장면 값 덮어쓰기)")
    ap.add_argument("--ball", default=None, help="공 시작 위치 x,y")
    args = ap.parse_args()
    sim = Sim(args)
    try:
        sim.run()
    except KeyboardInterrupt:
        pass
    finally:
        for r in sim.robots:
            r.node.destroy_node()
            rclpy.try_shutdown(context=r.ctx)


if __name__ == "__main__":
    main()

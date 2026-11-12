#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_logger_node.py

오도메트리 캘리브레이션 실험용 데이터 로거.

[무엇을 하는 노드인가]
    보행 중의 여러 토픽을 하나의 시간축에 정렬해 CSV 두 장으로 남긴다.

      1) <prefix>_raw.csv      : /joint_states 매 프레임(50Hz) 한 줄
      2) <prefix>_segments.csv : 측정 구간 하나당 한 줄 (요약)

    keyboard_teleop_node 의 ',' / '.' 키가 발행하는 /experiment/marker 를
    받아 구간을 구분한다.

[왜 rosbag 이 아니라 CSV 인가]
    rosbag 은 토픽별로 따로 저장되므로, 분석할 때 시간 정렬 코드를 매번
    따로 써야 한다. 이 실험은 데이터가 100MB 도 안 되는 규모라 rosbag 의
    범용성이 별로 살아나지 않는 반면, "pandas 로 바로 읽힌다" 는 이점이
    크다. 정렬을 기록 시점에 한 번 해두면 분석이 훨씬 단순해진다.

[왜 러너의 CSV 를 안 쓰고 별도 노드인가]
    actor_onnx_walking_runner 에도 CSV 로거가 있지만 세 가지 이유로
    부적합하다.
      1) 관절값이 정책 관측용으로 정렬·오프셋된 값이고, 관절을 '실제로
         읽은 시각'(body_motor_controller 가 back-date 한 stamp)이 없다.
         기구학 계산에는 원본 rad 와 그 stamp 가 필요하다.
      2) 50Hz 제어 루프 안에서 도는 실시간 노드다. 컬럼을 늘리면 제어
         주기에 영향을 주고, 실험 때문에 보행이 바뀌면 데이터가 오염된다.
      3) 킥 핸드오프 진단용으로 이미 자리잡은 파일이다.
    이 노드는 실시간 제약이 없고, 죽어도 로봇에 아무 영향이 없다.

[시간축 — 중요]
    ★ 행의 시각은 /joint_states 의 header.stamp 다 ★
    body_motor_controller_node 가 이 stamp 를 'SyncRead 시작 시각'으로
    back-date 해 둔다(stamp_at_read_start=True). 수신 시각을 쓰면 그
    back-date 가 날아가고 시리얼 왕복 지연이 그대로 오차로 들어온다.

    나머지 토픽은 '가장 최근에 받은 값'을 그대로 붙인다(zero-order hold).
    /joint_states 가 50Hz 로 가장 빠르고 다른 신호는 그보다 느리거나
    같으므로, 이 방식의 정렬 오차는 최대 한 프레임(20ms)이다.

[구간 마커 페이로드]
    SEG,<START|END|ABORT>,<seg_id>,<vx>,<vy>,<wz>,<stamp_sec>

[RViz / Foxglove 시각화]
    /experiment/viz (visualization_msgs/MarkerArray) 로 아래를 발행한다.
      · 구간 시작점 구체
      · 구간 중 주행 궤적 LINE_STRIP  ← 직선으로 갔는지 눈으로 확인
      · 구간 번호·명령·경과시간 텍스트
    5m 를 갔다고 생각했는데 실제로는 휘어졌더라, 같은 상황을 CSV 를
    열기 전에 현장에서 알아채기 위한 것이다.

[사용법]
    ros2 launch humanoid_bringup experiment_odom.launch.py vx_pos_max:=0.30
"""

import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """쿼터니언에서 yaw(rad)만 추출한다."""
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


class ExperimentLoggerNode(Node):

    def __init__(self):
        super().__init__('experiment_logger_node')

        # ── 파라미터 ─────────────────────────────────────────────
        self.declare_parameter('output_dir', '~/odom_experiments')
        # 파일명 앞에 붙는 라벨. 세션 성격을 적어두면 나중에 학습용과
        # 검증용을 파일 단위로 나눌 수 있다(예: fwd_ladder, square).
        self.declare_parameter('label', 'run')
        # 로봇 식별자. 조립 편차가 있어 데이터는 기체별이다.
        self.declare_parameter('robot', '')

        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('imu_topic', '/imu/data_50hz')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('marker_topic', '/experiment/marker')
        self.declare_parameter('viz_topic', '/experiment/viz')

        # 구간 밖(측정 중이 아닐 때)도 기록할지 여부.
        #   true  : 전 구간 기록. 파일이 커지지만 나중에 "구간 사이에
        #           무슨 일이 있었나"를 볼 수 있다.
        #   false : 구간 안에서만 기록. 파일이 작고 분석이 단순하다.
        # 기본 true — 50Hz × 약 900B 라 한 시간 찍어도 160MB 수준이고,
        # 나중에 필요해졌을 때 다시 찍는 비용이 훨씬 크다.
        self.declare_parameter('log_outside_segments', True)

        # 궤적 마커에 남길 최대 점 개수. 5m 를 0.2m/s 로 가면 25초,
        # 50Hz 면 1250점이다. 2000이면 충분하고 RViz 부담도 없다.
        self.declare_parameter('viz_max_points', 2000)
        self.declare_parameter('viz_rate_hz', 5.0)
        self.declare_parameter('viz_frame', 'odom')

        out_dir = os.path.expanduser(
            str(self.get_parameter('output_dir').value))
        label = str(self.get_parameter('label').value)
        robot = str(self.get_parameter('robot').value)
        if not robot:
            # 지정이 없으면 hostname 앞부분을 쓴다(robot_params.yaml 과 동일 규칙).
            robot = os.uname().nodename.split('.')[0]

        self._log_outside = bool(
            self.get_parameter('log_outside_segments').value)
        self._viz_max_pts = int(self.get_parameter('viz_max_points').value)
        self._viz_frame = str(self.get_parameter('viz_frame').value)

        # ── 최신값 보관 (zero-order hold) ────────────────────────
        self._cmd = (0.0, 0.0, 0.0)
        self._imu_quat = (0.0, 0.0, 0.0, 1.0)
        self._imu_gyro = (0.0, 0.0, 0.0)
        self._imu_acc = (0.0, 0.0, 0.0)
        self._odom_pose = (0.0, 0.0, 0.0)      # x, y, yaw
        self._odom_twist = (0.0, 0.0, 0.0)     # vx, vy, wz
        self._gait_phase = 0.0
        self._gait_freq = 0.0
        self._gait_state = ''

        # ── 구간 상태 ────────────────────────────────────────────
        self._seg_id = 0            # 0 = 구간 밖
        self._seg_cmd = (0.0, 0.0, 0.0)
        self._seg_t0 = 0.0
        self._seg_pose0 = (0.0, 0.0, 0.0)
        self._seg_track = []        # [(x, y), ...] 궤적 마커용
        self._seg_rows = 0
        self._pending_event = ''    # 이번 행에 남길 마커 이벤트

        # ── 관절 이름 순서 고정 ──────────────────────────────────
        # 첫 메시지의 순서를 잠그고, 이후 다른 순서가 오면 경고한다.
        # SyncRead 실패로 일부 관절이 빠지면 컬럼이 밀리는데, 그걸
        # 조용히 넘기면 분석 단계에서야 알게 된다.
        self._joint_names = None
        self._warned_names = False
        self._name_index = {}

        # ── 파일 준비 ────────────────────────────────────────────
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%m%d_%H%M%S')
        base = f'{label}_{robot}_{ts}'
        self._raw_path = os.path.join(out_dir, f'{base}_raw.csv')
        self._seg_path = os.path.join(out_dir, f'{base}_segments.csv')

        self._raw_file = None
        self._raw_writer = None
        self._raw_rows = 0

        self._seg_file = open(self._seg_path, 'w', newline='',
                              encoding='utf-8')
        self._seg_writer = csv.writer(self._seg_file)
        self._seg_writer.writerow([
            'seg_id', 'result', 'cmd_vx', 'cmd_vy', 'cmd_wz',
            't_start', 't_end', 'duration_s',
            'odom_dx', 'odom_dy', 'odom_dist', 'odom_dyaw_deg',
            'n_rows',
        ])
        self._seg_file.flush()

        # ── QoS ──────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── 구독 ─────────────────────────────────────────────────
        p = self.get_parameter
        self.create_subscription(
            JointState, str(p('joint_states_topic').value),
            self._cb_joints, 20)
        self.create_subscription(
            Twist, str(p('cmd_vel_topic').value), self._cb_cmd, 10)
        self.create_subscription(
            Imu, str(p('imu_topic').value), self._cb_imu, sensor_qos)
        self.create_subscription(
            Odometry, str(p('odom_topic').value), self._cb_odom, 10)
        self.create_subscription(
            String, str(p('marker_topic').value), self._cb_marker, 20)
        self.create_subscription(
            Float32, '/gait/phase', self._cb_phase, 10)
        self.create_subscription(
            Float32, '/gait/effective_frequency', self._cb_freq, 10)
        self.create_subscription(
            String, '/gait/state', self._cb_state, 10)

        # ── 시각화 ───────────────────────────────────────────────
        self._viz_pub = self.create_publisher(
            MarkerArray, str(p('viz_topic').value), 10)
        viz_hz = float(p('viz_rate_hz').value)
        self.create_timer(1.0 / max(0.5, viz_hz), self._publish_viz)

        self.get_logger().info(f'실험 로거 시작 — 기체={robot}, 라벨={label}')
        self.get_logger().info(f'  raw      : {self._raw_path}')
        self.get_logger().info(f'  segments : {self._seg_path}')
        self.get_logger().info(
            "키보드 노드에서 ',' 시작 / '.' 종료 / 'x' 취소")

    # =====================================================================
    #  최신값 콜백 — 저장만 한다
    # =====================================================================
    def _cb_cmd(self, msg: Twist):
        self._cmd = (float(msg.linear.x), float(msg.linear.y),
                     float(msg.angular.z))

    def _cb_imu(self, msg: Imu):
        q = msg.orientation
        self._imu_quat = (float(q.x), float(q.y), float(q.z), float(q.w))
        w = msg.angular_velocity
        self._imu_gyro = (float(w.x), float(w.y), float(w.z))
        a = msg.linear_acceleration
        self._imu_acc = (float(a.x), float(a.y), float(a.z))

    def _cb_odom(self, msg: Odometry):
        pp = msg.pose.pose
        q = pp.orientation
        self._odom_pose = (float(pp.position.x), float(pp.position.y),
                           yaw_from_quat(q.x, q.y, q.z, q.w))
        t = msg.twist.twist
        self._odom_twist = (float(t.linear.x), float(t.linear.y),
                            float(t.angular.z))

    def _cb_phase(self, msg: Float32):
        self._gait_phase = float(msg.data)

    def _cb_freq(self, msg: Float32):
        self._gait_freq = float(msg.data)

    def _cb_state(self, msg: String):
        self._gait_state = str(msg.data)

    # =====================================================================
    #  구간 마커
    # =====================================================================
    def _cb_marker(self, msg: String):
        """'SEG,<event>,<id>,<vx>,<vy>,<wz>,<stamp>' 를 파싱한다.

        형식이 깨진 메시지는 경고만 남기고 무시한다. 실험 중에 로거가
        죽으면 그 세션이 통째로 날아가므로, 파싱 실패로는 절대 죽지 않게
        한다.
        """
        try:
            parts = msg.data.split(',')
            if len(parts) != 7 or parts[0] != 'SEG':
                raise ValueError('필드 수/헤더 불일치')
            event = parts[1]
            seg_id = int(parts[2])
            cmd = (float(parts[3]), float(parts[4]), float(parts[5]))
            stamp = float(parts[6])
        except Exception as e:            # noqa: BLE001
            self.get_logger().warn(f'마커 파싱 실패: "{msg.data}" ({e})')
            return

        if event == 'START':
            self._start_segment(seg_id, cmd, stamp)
        elif event in ('END', 'ABORT'):
            self._finish_segment(event, seg_id, stamp)
        else:
            self.get_logger().warn(f'알 수 없는 마커 이벤트: {event}')

    def _start_segment(self, seg_id: int, cmd: tuple, stamp: float):
        if self._seg_id != 0:
            # 키보드 노드가 ABORT 를 먼저 쏘므로 정상적으로는 안 온다.
            # 마커가 유실됐을 때의 방어다.
            self.get_logger().warn(
                f'#{self._seg_id} 가 열린 채로 #{seg_id} 시작 → 앞 구간 취소')
            self._finish_segment('ABORT', self._seg_id, stamp)

        self._seg_id = seg_id
        self._seg_cmd = cmd
        self._seg_t0 = stamp
        self._seg_pose0 = self._odom_pose
        self._seg_track = []
        self._seg_rows = 0
        self._pending_event = 'START'
        self.get_logger().info(
            f'[SEG #{seg_id}] 시작  cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, '
            f'{cmd[2]:+.2f})')

    def _finish_segment(self, event: str, seg_id: int, stamp: float):
        if self._seg_id == 0:
            self.get_logger().warn('열린 구간이 없는데 종료 마커가 왔습니다')
            return

        dt = stamp - self._seg_t0
        x0, y0, yaw0 = self._seg_pose0
        x1, y1, yaw1 = self._odom_pose
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        dyaw = math.degrees(
            math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))

        self._seg_writer.writerow([
            self._seg_id, event,
            f'{self._seg_cmd[0]:.4f}', f'{self._seg_cmd[1]:.4f}',
            f'{self._seg_cmd[2]:.4f}',
            f'{self._seg_t0:.6f}', f'{stamp:.6f}', f'{dt:.4f}',
            f'{dx:.4f}', f'{dy:.4f}', f'{dist:.4f}', f'{dyaw:.2f}',
            self._seg_rows,
        ])
        self._seg_file.flush()

        # 참고용으로 현재 odom 이 말하는 거리도 같이 띄운다. 실제 5m 와
        # 얼마나 다른지가 지금 오도메트리 오차의 1 차 지표다.
        self.get_logger().info(
            f'[SEG #{self._seg_id}] {event}  {dt:.2f}s  '
            f'odom거리={dist:.3f}m  Δyaw={dyaw:+.1f}deg  '
            f'({self._seg_rows}행)')

        self._pending_event = event
        self._seg_id = 0

    # =====================================================================
    #  메인 — /joint_states 가 행의 클럭
    # =====================================================================
    def _cb_joints(self, msg: JointState):
        if self._joint_names is None:
            self._lock_joint_order(msg)
            self._init_raw_csv()

        if self._seg_id == 0 and not self._log_outside:
            self._pending_event = ''
            return

        # ★ 수신 시각이 아니라 header.stamp ★ (파일 상단 [시간축] 참조)
        t = (msg.header.stamp.sec
             + msg.header.stamp.nanosec * 1e-9)

        pos, vel = self._extract_joints(msg)

        row = [
            self._raw_rows,
            f'{t:.6f}',
            self._seg_id,
            self._pending_event,
        ]
        row += [f'{v:.5f}' for v in self._cmd]
        row += [f'{v:.5f}' for v in self._odom_twist]
        row += [f'{v:.5f}' for v in self._odom_pose]
        row += [f'{v:.6f}' for v in self._imu_quat]
        row += [f'{v:.6f}' for v in self._imu_gyro]
        row += [f'{v:.6f}' for v in self._imu_acc]
        row += [f'{self._gait_phase:.5f}', f'{self._gait_freq:.4f}',
                self._gait_state]
        row += pos
        row += vel

        self._raw_writer.writerow(row)
        self._raw_rows += 1
        self._pending_event = ''

        if self._seg_id != 0:
            self._seg_rows += 1
            self._seg_track.append((self._odom_pose[0], self._odom_pose[1]))
            if len(self._seg_track) > self._viz_max_pts:
                self._seg_track.pop(0)

        # 구간 경계에서 데이터가 잘리면 그 구간을 못 쓰므로, 구간 안에서는
        # 자주 flush 한다. 구간 밖에서는 부하를 줄인다.
        if self._seg_id != 0:
            if self._raw_rows % 25 == 0:
                self._raw_file.flush()
        elif self._raw_rows % 250 == 0:
            self._raw_file.flush()

    def _lock_joint_order(self, msg: JointState):
        self._joint_names = list(msg.name)
        self._name_index = {n: i for i, n in enumerate(self._joint_names)}
        self.get_logger().info(
            f'관절 순서 고정: {len(self._joint_names)}개 — '
            f'{", ".join(self._joint_names)}')

    def _extract_joints(self, msg: JointState):
        """고정된 순서대로 position/velocity 를 뽑는다.

        메시지의 관절 구성이 첫 프레임과 다르면(SyncRead 실패 등) 빠진
        칸을 빈 문자열로 두고 경고한다. 순서를 그대로 믿고 쓰면 컬럼이
        밀려서, 분석 단계에 가서야 이상을 알아채게 된다.
        """
        n = len(self._joint_names)
        if list(msg.name) == self._joint_names:
            pos = [f'{v:.6f}' for v in msg.position[:n]]
            vel = ([f'{v:.6f}' for v in msg.velocity[:n]]
                   if len(msg.velocity) >= n else [''] * n)
            # 길이가 모자라면 채운다(방어).
            pos += [''] * (n - len(pos))
            return pos, vel

        if not self._warned_names:
            self._warned_names = True
            self.get_logger().warn(
                f'관절 구성이 첫 프레임과 다릅니다 '
                f'({len(msg.name)}개 vs {n}개). 이름으로 매칭합니다 — '
                'SyncRead 실패가 반복되면 데이터에 빈칸이 생깁니다.')

        pos = [''] * n
        vel = [''] * n
        for i, name in enumerate(msg.name):
            j = self._name_index.get(name)
            if j is None:
                continue
            if i < len(msg.position):
                pos[j] = f'{msg.position[i]:.6f}'
            if i < len(msg.velocity):
                vel[j] = f'{msg.velocity[i]:.6f}'
        return pos, vel

    def _init_raw_csv(self):
        self._raw_file = open(self._raw_path, 'w', newline='',
                              encoding='utf-8')
        self._raw_writer = csv.writer(self._raw_file)

        header = ['row', 't_ros', 'seg_id', 'event']
        header += ['cmd_vx', 'cmd_vy', 'cmd_wz']
        header += ['odom_vx', 'odom_vy', 'odom_wz']
        header += ['odom_x', 'odom_y', 'odom_yaw']
        header += ['imu_qx', 'imu_qy', 'imu_qz', 'imu_qw']
        header += ['imu_wx', 'imu_wy', 'imu_wz']
        header += ['imu_ax', 'imu_ay', 'imu_az']
        header += ['gait_phase', 'gait_freq', 'gait_state']
        header += [f'q_{n}' for n in self._joint_names]
        header += [f'qd_{n}' for n in self._joint_names]

        self._raw_writer.writerow(header)
        self._raw_file.flush()
        self.get_logger().info(f'raw CSV 헤더 {len(header)}컬럼 작성')

    # =====================================================================
    #  RViz / Foxglove 시각화
    # =====================================================================
    def _publish_viz(self):
        """구간 시작점, 주행 궤적, 상태 텍스트를 발행한다.

        CSV 를 열어보기 전에 현장에서 "이번 구간이 직선이었나"를 눈으로
        확인하기 위한 것이다. 궤적이 휘었으면 그 회차는 다시 하면 된다.
        """
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        # (1) 궤적
        track = Marker()
        track.header.frame_id = self._viz_frame
        track.header.stamp = now
        track.ns = 'experiment'
        track.id = 0
        track.type = Marker.LINE_STRIP
        track.action = Marker.ADD if self._seg_track else Marker.DELETE
        track.scale.x = 0.02
        track.color.r = 0.1
        track.color.g = 0.8
        track.color.b = 0.4
        track.color.a = 0.9
        track.pose.orientation.w = 1.0
        for x, y in self._seg_track:
            track.points.append(Point(x=x, y=y, z=0.01))
        arr.markers.append(track)

        # (2) 시작점
        start = Marker()
        start.header.frame_id = self._viz_frame
        start.header.stamp = now
        start.ns = 'experiment'
        start.id = 1
        start.type = Marker.SPHERE
        start.action = Marker.ADD if self._seg_id != 0 else Marker.DELETE
        start.pose.position.x = self._seg_pose0[0]
        start.pose.position.y = self._seg_pose0[1]
        start.pose.position.z = 0.02
        start.pose.orientation.w = 1.0
        start.scale.x = start.scale.y = start.scale.z = 0.08
        start.color.r = 0.9
        start.color.g = 0.5
        start.color.b = 0.1
        start.color.a = 0.9
        arr.markers.append(start)

        # (3) 상태 텍스트 — 로봇 위에 띄운다
        txt = Marker()
        txt.header.frame_id = self._viz_frame
        txt.header.stamp = now
        txt.ns = 'experiment'
        txt.id = 2
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = self._odom_pose[0]
        txt.pose.position.y = self._odom_pose[1]
        txt.pose.position.z = 0.75
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.10
        txt.color.r = txt.color.g = txt.color.b = 1.0
        txt.color.a = 0.95
        if self._seg_id != 0:
            el = self.get_clock().now().nanoseconds * 1e-9 - self._seg_t0
            dx = self._odom_pose[0] - self._seg_pose0[0]
            dy = self._odom_pose[1] - self._seg_pose0[1]
            txt.text = (f'REC #{self._seg_id}  vx={self._seg_cmd[0]:+.2f}\n'
                        f'{el:.1f}s   odom {math.hypot(dx, dy):.2f}m')
        else:
            txt.text = 'idle'
        arr.markers.append(txt)

        self._viz_pub.publish(arr)

    # =====================================================================
    def close(self):
        try:
            if self._raw_file is not None:
                self._raw_file.flush()
                self._raw_file.close()
            self._seg_file.flush()
            self._seg_file.close()
        except Exception:      # noqa: BLE001
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentLoggerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # 열린 구간이 있으면 닫아 둔다(파일이 반쯤 열린 채 끝나지 않게).
        if node._seg_id != 0:
            stamp = node.get_clock().now().nanoseconds * 1e-9
            node._finish_segment('ABORT', node._seg_id, stamp)
        node.get_logger().info(
            f'기록 종료 — raw {node._raw_rows}행: {node._raw_path}')
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy, math, time
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3Stamped
from rcl_interfaces.msg import SetParametersResult
from collections import deque


# ── Madgwick 필터 ──────────────────────
class MadgwickAHRS:
    def __init__(self, sample_period=1/100.0, beta=0.1):
        self.sample_period = float(sample_period)
        self.beta = float(beta)
        self.q1, self.q2, self.q3, self.q4 = 1.0, 0.0, 0.0, 0.0  # w,x,y,z

    def update_imu(self, gx, gy, gz, ax, ay, az):
        q1, q2, q3, q4 = self.q1, self.q2, self.q3, self.q4

        # 가속도 정규화
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm == 0.0:
            return
        ax, ay, az = ax/norm, ay/norm, az/norm

        # 보조 상수
        _2q1 = 2.0*q1; _2q2 = 2.0*q2; _2q3 = 2.0*q3; _2q4 = 2.0*q4
        _4q1 = 4.0*q1; _4q2 = 4.0*q2; _4q3 = 4.0*q3
        _8q2 = 8.0*q2; _8q3 = 8.0*q3
        q1q1 = q1*q1; q2q2 = q2*q2; q3q3 = q3*q3; q4q4 = q4*q4

        s1 = _4q1*q3q3 + _2q3*ax + _4q1*q2q2 - _2q2*ay
        s2 = _4q2*q4q4 - _2q4*ax + 4.0*q1q1*q2 - _2q1*ay - _4q2 + _8q2*q2q2 + _8q2*q3q3 + _4q2*az
        s3 = 4.0*q1q1*q3 + _2q1*ax + _4q3*q4q4 - _2q4*ay - _4q3 + _8q3*q2q2 + _8q3*q3q3 + _4q3*az
        s4 = 4.0*q2q2*q4 - _2q2*ax + 4.0*q3q3*q4 - _2q3*ay

        norm_s = math.sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4)
        if norm_s != 0.0:
            s1, s2, s3, s4 = s1/norm_s, s2/norm_s, s3/norm_s, s4/norm_s

        qDot1 = 0.5 * (-q2*gx - q3*gy - q4*gz) - self.beta*s1
        qDot2 = 0.5 * ( q1*gx + q3*gz - q4*gy) - self.beta*s2
        qDot3 = 0.5 * ( q1*gy - q2*gz + q4*gx) - self.beta*s3
        qDot4 = 0.5 * ( q1*gz + q2*gy - q3*gx) - self.beta*s4

        q1 += qDot1 * self.sample_period
        q2 += qDot2 * self.sample_period
        q3 += qDot3 * self.sample_period
        q4 += qDot4 * self.sample_period

        norm_q = math.sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4)
        if norm_q == 0.0:
            return
        self.q1, self.q2, self.q3, self.q4 = q1/norm_q, q2/norm_q, q3/norm_q, q4/norm_q

    def quaternion(self):
        return self.q2, self.q3, self.q4, self.q1  # (x,y,z,w)

    def gravity_vector_body(self):
        """
        Isaac-style projected gravity in body frame.
        정자세일 때 대체로 (0, 0, -1)
        """
        q1, q2, q3, q4 = self.q1, self.q2, self.q3, self.q4

        gx = 2.0 * (q2 * q4 - q1 * q3)
        gy = 2.0 * (q1 * q2 + q3 * q4)
        gz = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4

        return -gx, -gy, -gz


# ── ROS Node ───────────────────────────
class ImuParserNode(Node):
    def __init__(self):
        super().__init__('imu_parser_node')
        self.ahrs = MadgwickAHRS(sample_period=1/200.0, beta=0.1)

        # ── 마운팅 정렬(extrinsic) 보정 ──────────────────────────
        # IMU 장착 방향의 고정 기울기(roll/pitch)를 보정한다. 값은
        # humanoid_bringup/config/robot_params.yaml 의 <로봇> 섹션에서 오며,
        # dynamixel_system.launch.py 가 hostname 으로 섹션을 골라 전달한다.
        # (2026-08 이전에는 런치가 전달하지 않아 항상 0 이었다 — 수정됨)
        # khuber_imu_calib 도구가 '수평 기준자세'에서 측정해 출력한 값을 넣는다.
        # accel/gyro 에 R=Ry(pitch)·Rx(roll) 을 곱해 base_link 로 정렬한다
        # (→ /imu/data · projected_gravity · EKF 모두 혜택).
        self.declare_parameter('imu_mount_roll_deg', 0.0)
        self.declare_parameter('imu_mount_pitch_deg', 0.0)
        _mroll = float(self.get_parameter('imu_mount_roll_deg').value)
        _mpitch = float(self.get_parameter('imu_mount_pitch_deg').value)
        self._mount_R = self._build_mount_R(_mroll, _mpitch)
        self._log_mount(_mroll, _mpitch)
        # 라이브 튜닝(부호 검증용): ros2 param set 으로 즉시 반영
        self.add_on_set_parameters_callback(self._on_set_params)

        # full-scale 단위 테이블
        self.accel_sens = {0:0.061, 1:0.122, 2:0.244, 3:0.488}    # mg/LSB
        self.gyro_sens  = {0:4.375, 1:8.75, 2:17.5, 3:35.0, 4:70.0, 0xC:140.0} # mdps/LSB

        self.sub = self.create_subscription(
            ByteMultiArray, '/mcu/raw_rx', self.on_raw, 10)

        self.pub_imu = self.create_publisher(Imu, '/imu/data', 10)
        self.pub_gravity = self.create_publisher(Vector3Stamped, '/imu/projected_gravity', 10)

        # ── [2026-08 CPU 최적화] 저레이트 IMU 채널 ────────────────────────
        #   ★ 200Hz 원본(/imu/data)은 절대 건드리지 않는다.
        #     정책 러너가 그걸 쓰고, 관측 신선도가 바뀌면 학습 시 설정한
        #     action delay 와 어긋나 검증된 보행 베이스라인이 깨진다.
        #     이 토픽은 '추가'일 뿐이다.
        #
        #   왜 필요한가:
        #     50Hz 로만 소비하면서 200Hz 를 구독하던 노드들이 있었다.
        #       walk_odometry_node       : 타이머 50Hz 로만 사용 (저장만 하고 버림)
        #       fall_manager_node        : FSM 50Hz
        #       body_motor_controller    : standup 서비스에서 가끔 1회 사용
        #     rclpy 는 깨어날 때마다 wait set 을 Python 에서 재구축하므로,
        #     콜백이 대입 한 줄이어도 깨우기 비용은 그대로 다 낸다.
        #     4프레임 중 3개는 깨어나서 버리는 순수 낭비였다.
        #
        #   decimate_n=4 → 200Hz / 4 = 50Hz. 제어 주기와 맞춘 값이다.
        #     ※ 발행만 솎아낸다. Madgwick 갱신(update_imu)은 아래에서 매 프레임
        #       200Hz 로 그대로 돈다 → 필터 반응성 손실이 전혀 없다.
        #       솎아낸 시점의 필터 출력을 그대로 내보내는 것뿐이다.
        self.declare_parameter('decimate_n', 4)
        self._decimate_n = max(1, int(self.get_parameter('decimate_n').value))
        self._decimate_i = 0
        self.pub_imu_50hz = self.create_publisher(Imu, '/imu/data_50hz', 10)

        # ── [2026-08 CPU 최적화] projected gravity 발행 토글 ──────────────
        #   /imu/projected_gravity 는 /imu/data.orientation 만의 순수 함수라
        #   정보량이 0 인 중복 토픽이다(imu_math.projected_gravity_from_quat 참조).
        #   주 구독자들은 파생 계산으로 전환했지만, 아직 이 토픽을 쓰는
        #   온디맨드/디버그 도구가 남아 있어 기본값은 True 로 둔다.
        #       standup_node.py       (온디맨드)
        #       policy_logger_node.py (디버그)
        #       tilt_monitor_tui.py   (TUI 도구)
        #   구독자가 0 이면 DDS 가 전달을 생략하므로 유지 비용은 메시지 생성 +
        #   publish 호출뿐이다(200Hz 기준 대략 1~2%). 그 1~2%까지 필요해지면
        #   launch 에서 False 로 끌 것.
        self.declare_parameter('publish_projected_gravity', True)
        self._pub_grav_enabled = bool(
            self.get_parameter('publish_projected_gravity').value)

        self.get_logger().info(
            f"IMU 발행: /imu/data 200Hz, /imu/data_50hz 1/{self._decimate_n}, "
            f"/imu/projected_gravity="
            f"{'ON' if self._pub_grav_enabled else 'OFF'}")

        # sliding window 타임스탬프 기록
        self.timestamps = deque(maxlen=500)
        self.last_sample = None
        self.create_timer(1.0, self.print_rate)

        self.last_msg_time = None

    # ── 마운팅 정렬(extrinsic) 보정 헬퍼 ─────────────────────
    @staticmethod
    def _build_mount_R(roll_deg, pitch_deg):
        """R = Ry(pitch)·Rx(roll) (3x3). IMU 프레임 벡터를 base_link 로 회전.

        khuber_imu_calib 도구가 '수평 기준자세'에서 측정한 roll/pitch(도)를
        받아, 측정된 중력 up 벡터를 (0,0,1)로 되돌리는 보정 회전을 만든다.
        (수치 검증: 임의 틸트로 만든 up 벡터에 이 R 을 곱하면 (0,0,1)로 복원)
        """
        r = math.radians(roll_deg)
        p = math.radians(pitch_deg)
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        # Ry(p) @ Rx(r)
        return (
            (cp, sp * sr, sp * cr),
            (0.0, cr, -sr),
            (-sp, cp * sr, cp * cr),
        )

    @staticmethod
    def _apply_R(R, x, y, z):
        """3x3 회전 R 을 벡터 (x,y,z)에 곱한 결과 (x',y',z') 반환."""
        return (
            R[0][0] * x + R[0][1] * y + R[0][2] * z,
            R[1][0] * x + R[1][1] * y + R[1][2] * z,
            R[2][0] * x + R[2][1] * y + R[2][2] * z,
        )

    def _log_mount(self, roll_deg, pitch_deg):
        """현재 마운팅 보정값을 로그로 남긴다."""
        if abs(roll_deg) < 1e-9 and abs(pitch_deg) < 1e-9:
            self.get_logger().info("IMU 마운팅 보정: 없음 (roll=pitch=0)")
        else:
            self.get_logger().info(
                f"IMU 마운팅 보정 적용: roll={roll_deg:+.2f}deg "
                f"pitch={pitch_deg:+.2f}deg (R=Ry(pitch)·Rx(roll))")

    def _on_set_params(self, params):
        """ros2 param set 으로 마운팅 값 변경 시 회전행렬을 즉시 재구성(부호 검증용)."""
        roll = float(self.get_parameter('imu_mount_roll_deg').value)
        pitch = float(self.get_parameter('imu_mount_pitch_deg').value)
        changed = False
        for p in params:
            if p.name == 'imu_mount_roll_deg':
                roll = float(p.value)
                changed = True
            elif p.name == 'imu_mount_pitch_deg':
                pitch = float(p.value)
                changed = True
        if changed:
            self._mount_R = self._build_mount_R(roll, pitch)
            self._log_mount(roll, pitch)
        return SetParametersResult(successful=True)

    def on_raw(self, msg: ByteMultiArray):
        data = b''.join(msg.data)
        if len(data) < 19:
            return
        if data[0] != 0x53:
            return
        if data[2] != 0x20 or data[4] != 0x02:
            return

        # === parameter에서 범위 추출 ===
        param = data[5]
        accel_fs = param & 0x0F
        gyro_fs  = (param >> 4) & 0x0F

        accel_g_per_lsb = (self.accel_sens.get(accel_fs, 0.244)) / 1000.0
        gyro_dps_per_lsb = (self.gyro_sens.get(gyro_fs, 35.0)) / 1000.0

        # === payload 파싱 ===
        def s16(lo, hi):
            v = (hi << 8) | lo
            return v - 0x10000 if v & 0x8000 else v

        base = 7
        ax = s16(data[base+0], data[base+1]) * accel_g_per_lsb * 9.81
        ay = s16(data[base+2], data[base+3]) * accel_g_per_lsb * 9.81
        az = s16(data[base+4], data[base+5]) * accel_g_per_lsb * 9.81
        gx = math.radians(s16(data[base+6], data[base+7]) * gyro_dps_per_lsb)
        gy = math.radians(s16(data[base+8], data[base+9]) * gyro_dps_per_lsb)
        gz = math.radians(s16(data[base+10], data[base+11]) * gyro_dps_per_lsb)

        # 마운팅 정렬 보정: accel/gyro 를 base_link 프레임으로 회전(장착 기울기 제거).
        # 회전은 순수 선형이라 단위 변환(m/s^2, rad/s) 이후 적용해도 동일하다.
        ax, ay, az = self._apply_R(self._mount_R, ax, ay, az)
        gx, gy, gz = self._apply_R(self._mount_R, gx, gy, gz)

        # 실제 dt 반영
        now_sec = time.time()
        if self.last_msg_time is not None:
            dt = now_sec - self.last_msg_time
            if 0.001 <= dt <= 0.1:
                self.ahrs.sample_period = dt
        self.last_msg_time = now_sec

        # Madgwick 업데이트
        self.ahrs.update_imu(gx, gy, gz, ax/9.81, ay/9.81, az/9.81)
        qx, qy, qz, qw = self.ahrs.quaternion()

        # projected gravity 계산
        pgx, pgy, pgz = self.ahrs.gravity_vector_body()

        stamp = self.get_clock().now().to_msg()

        # ROS IMU 퍼블리시
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = 'imu_link'
        imu.linear_acceleration.x = ax
        imu.linear_acceleration.y = ay
        imu.linear_acceleration.z = az
        imu.angular_velocity.x = gx
        imu.angular_velocity.y = gy
        imu.angular_velocity.z = gz
        imu.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        self.pub_imu.publish(imu)

        # ── [2026-08 CPU 최적화] 저레이트 채널 발행 ────────────────────────
        #   같은 imu 객체를 그대로 재발행한다. publish() 시 rclpy 가 내부적으로
        #   직렬화 복사를 하므로 객체 공유는 안전하고, Imu 메시지를 한 번 더
        #   만들 필요가 없다.
        #   ★ 여기 오는 값은 항상 200Hz 로 갱신된 최신 필터 출력이다.
        #     '오래된 값을 내보내는' 것이 아니라 '최신값을 덜 자주 내보내는' 것.
        self._decimate_i += 1
        if self._decimate_i >= self._decimate_n:
            self._decimate_i = 0
            self.pub_imu_50hz.publish(imu)

        # projected gravity 퍼블리시
        #   주 구독자들은 imu_math.projected_gravity_from_quat() 로 파생 계산하도록
        #   전환됐다. 이 발행은 온디맨드/디버그 도구 호환용이며 파라미터로 끌 수 있다.
        if self._pub_grav_enabled:
            grav = Vector3Stamped()
            grav.header.stamp = stamp
            grav.header.frame_id = 'imu_link'
            grav.vector.x = pgx
            grav.vector.y = pgy
            grav.vector.z = pgz
            self.pub_gravity.publish(grav)

        # sliding window에 수신 시각 기록
        self.timestamps.append(now_sec)
        self.last_sample = (ax, ay, az, gx, gy, gz, qx, qy, qz, qw, pgx, pgy, pgz)

    def print_rate(self):
        now = time.time()
        self.timestamps = deque([t for t in self.timestamps if now - t <= 1.0], maxlen=500)
        if len(self.timestamps) > 1:
            dt = self.timestamps[-1] - self.timestamps[0]
            hz = (len(self.timestamps)-1) / dt if dt > 0 else 0.0
        else:
            hz = 0.0

        self.get_logger().info(f"IMU rate: {hz:.1f} Hz")
        if self.last_sample:
            ax, ay, az, gx, gy, gz, qx, qy, qz, qw, pgx, pgy, pgz = self.last_sample
            self.get_logger().info(
                f"acc=({ax:+.3f},{ay:+.3f},{az:+.3f}) "
                f"gyro=({math.degrees(gx):+.2f},{math.degrees(gy):+.2f},{math.degrees(gz):+.2f}) "
                f"quat=({qx:+.3f},{qy:+.3f},{qz:+.3f},{qw:+.3f}) "
                f"grav=({pgx:+.3f},{pgy:+.3f},{pgz:+.3f})"
            )


def main():
    rclpy.init()
    node = ImuParserNode()
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khuber_imu_calib.py
===================
KHUBER v2 IMU 캘리브레이션 통합 도구 (자이로 바이어스 + 마운팅 정렬).

mcu_bridge_node 가 떠 있는 상태에서, 기존 시리얼 브리지를 그대로 활용한다.
(하드웨어를 직접 잡지 않으므로 khuber_dxl_* 와 달리 ROS 스택 실행이 필요.)
  - 송신: /mcu/raw_tx  (ByteMultiArray) → 브리지가 MCU로 write
  - 수신: /mcu/raw_rx  (ByteMultiArray) ← 브리지가 MCU 프레임을 발행(firehose)

두 캘리는 성격이 다르다:
  1) 자이로 바이어스 — 펌웨어가 측정/적용(RAM). 정지만 하면 되고 매 부팅 실행.
                       부작용 있음(펌웨어 상태를 바꿈).
  2) 마운팅 정렬     — IMU 장착 기울기(roll/pitch)를 '측정/출력'만 한다. 파일은
                       건드리지 않으며, 출력값을 사용자가 robot_params.yaml 에
                       직접 기입한다. 반드시 '수평 기준자세'(스탠드)에서 1회.
                       실제 회전 적용은 imu_parser_node(다음 단계)에서 한다.

마운팅 측정은 /imu/data(imu_parser 거친 값)가 아니라 /mcu/raw_rx 의 IMU 스트림
프레임에서 raw accel 을 직접 뽑는다. 이유: 나중에 imu_parser 가 마운트 회전을
적용하면 /imu/data 는 이미 보정돼 재측정 시 ~0 이 나온다. raw 를 읽으면 imu_parser
설정과 무관하게 항상 순수 기울기를 재고, 각도는 벡터 방향의 비율이라 FS 스케일
변환도 필요 없다.

동작 모드:
  (기본)        자이로 START(적용) → 마운트 측정(출력). 셋업 세션용(스탠드, 수평).
  --gyro-only   자이로만. 매 부팅용(수평 아니어도 됨).

사용 예 (mcu_bridge_node 실행 중이어야 함):
  ros2 run humanoid_utils khuber_imu_calib               # 자이로+마운트 (셋업)
  ros2 run humanoid_utils khuber_imu_calib --gyro-only   # 자이로만 (부팅)

주의:
  - 두 측정 모두 로봇 정지 필수. 마운트는 추가로 '수평 기준자세'가 필수다.
  - 자이로 바이어스는 플래시에 저장하지 않는다(온도 의존성). 매 부팅 재실행.
"""
import sys
import math
import time
import socket
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray


# ── 프로토콜 상수 (펌웨어 vcp_proto.h / 브리지와 일치) ────────────────────
STX = 0x53
TARGET_ID_ANY = 0xFE            # 펌웨어가 ID 를 검사하지 않는 명령(관례상 0xFE)
DXL_INST_READ = 0x02           # GET / IMU READ
DXL_INST_WRITE = 0x03          # START

# 자이로 바이어스 캘리 (CMD=0x30)
CALIB_TARGET_COMMAND = 0x30
CALIB_PARAM_GYRO_BIAS = 0x01
CALIB_STATUS = {
    0x00: "OK",       # 성공(바이어스 갱신됨)
    0x01: "MOVED",    # 측정 중 움직임 감지 → 폐기
    0x02: "TIMEOUT",  # 샘플 수집 실패
    0x03: "INVALID",  # GET 시 아직 캘리 안 됨
}
GYRO_DPS_PER_LSB = 70.0 / 1000.0   # ±2000dps 기준 70 mdps/LSB (FS 변경 시 수정)

# IMU 스트림 (CMD=0x20). 마운트 측정용 raw accel 을 여기서 뽑는다.
IMU_TARGET_COMMAND = 0x20
IMU_STREAM_START_PARAM = 0x10
IMU_STREAM_ON_PAYLOAD = [0x01]
IMU_STREAM_OFF_PAYLOAD = [0x00]   # payload[0]==0 → STREAM_STOP (펌웨어 cmdRx 규약)

# 마운트 측정 파라미터
MOUNT_SAMPLES = 300              # ~1.5s @ 200Hz
MOUNT_MAX_WAIT_S = 8.0           # 샘플 수집 상한
MOUNT_ACC_STD_THRESH_LSB = 150   # 정지 게이트(축별 accel std). ~0.07g @ ±16g. FS 변경 시 재조정
MOUNT_LEVEL_WARN_DEG = 20.0      # 수직 대비 총기울기가 이보다 크면 기준자세 경고


def build_frame(command, target_id, instruction, parameter, payload=None):
    """한 프레임 빌드. 펌웨어 VCP_BuildPacket / 브리지 build_packet 과 동일.

      [STX][LEN][CMD][ID][INST][PARAM][PLEN][payload..][CHK]
      LEN = 8 + len(payload)
      CHK = sum(bytes[0..LEN-2]) & 0xFF
    """
    if payload is None:
        payload = []
    plen = len(payload)
    length = 8 + plen
    pkt = [STX, length, command, target_id, instruction, parameter, plen] + list(payload)
    pkt.append(sum(pkt) & 0xFF)
    return pkt


def s16_le(lo, hi):
    """리틀엔디안 2바이트 → 부호 있는 16비트 정수."""
    v = (hi << 8) | lo
    return v - 0x10000 if v & 0x8000 else v


def fmt_bias(bias):
    """자이로 bias(LSB 3축)를 LSB + dps 로 보기 좋게 문자열화한다."""
    lsb = "[{:+d}, {:+d}, {:+d}] LSB".format(*bias)
    dps = "[{:+.3f}, {:+.3f}, {:+.3f}] dps".format(
        *[b * GYRO_DPS_PER_LSB for b in bias])
    return f"{lsb}  ({dps})"


def compute_mount(samples):
    """raw accel 샘플들(LSB 3축)에서 평균/표준편차와 roll/pitch/총기울기를 계산.

    accel 은 정지 시 '비력(specific force)' = 위쪽을 향하므로, 수평·정렬 상태면
    대략 (0, 0, +g). x/y 성분이 장착 기울기를 나타낸다.
      roll  = atan2(my, mz)                      (x축 회전)
      pitch = atan2(-mx, sqrt(my^2 + mz^2))      (y축 회전)
    반환: (mean3, std3, roll_deg, pitch_deg, total_tilt_deg)
    """
    n = len(samples)
    mx = sum(s[0] for s in samples) / n
    my = sum(s[1] for s in samples) / n
    mz = sum(s[2] for s in samples) / n

    def std(idx, mean):
        return math.sqrt(sum((s[idx] - mean) ** 2 for s in samples) / n)

    stds = (std(0, mx), std(1, my), std(2, mz))

    roll = math.degrees(math.atan2(my, mz))
    pitch = math.degrees(math.atan2(-mx, math.sqrt(my * my + mz * mz)))
    mag = math.sqrt(mx * mx + my * my + mz * mz)
    total_tilt = math.degrees(math.acos(max(-1.0, min(1.0, mz / mag)))) if mag > 0 else 0.0
    return (mx, my, mz), stds, roll, pitch, total_tilt


class ImuCalibClient(Node):
    """브리지 토픽으로 자이로/마운트 캘리를 수행하는 일회성 클라이언트."""

    def __init__(self):
        """퍼블리셔/서브스크립션과 수신 버퍼를 초기화한다."""
        super().__init__('khuber_imu_calib')
        self.pub = self.create_publisher(ByteMultiArray, '/mcu/raw_tx', 10)
        # firehose(200Hz+) 를 받으므로 큐를 넉넉히
        self.sub = self.create_subscription(
            ByteMultiArray, '/mcu/raw_rx', self._on_rx, 50)
        self._last_resp = None        # 최근 CALIB(0x30) 응답 (status, bias3)
        self._accel_buf = []          # 마운트용 raw accel 수집 버퍼
        self._collecting_accel = False

    # ── 송신 헬퍼 ────────────────────────────────────────────────────────
    def _send(self, frame_bytes):
        """정수 리스트 프레임을 ByteMultiArray 로 감싸 /mcu/raw_tx 로 발행한다."""
        msg = ByteMultiArray()
        msg.data = [bytes([b]) for b in frame_bytes]   # 브리지 _on_tx 규약
        self.pub.publish(msg)

    def send_gyro_start(self):
        """자이로 바이어스 측정 시작(START, INST=WRITE)."""
        self._last_resp = None
        self._send(build_frame(CALIB_TARGET_COMMAND, TARGET_ID_ANY,
                               DXL_INST_WRITE, CALIB_PARAM_GYRO_BIAS))

    def send_gyro_get(self):
        """저장된 자이로 바이어스 조회(GET, INST=READ)."""
        self._last_resp = None
        self._send(build_frame(CALIB_TARGET_COMMAND, TARGET_ID_ANY,
                               DXL_INST_READ, CALIB_PARAM_GYRO_BIAS))

    def send_imu_on(self):
        """IMU 스트림 ON(브리지 _enable_imu_stream 과 동일 프레임)."""
        self._send(build_frame(IMU_TARGET_COMMAND, TARGET_ID_ANY,
                               DXL_INST_READ, IMU_STREAM_START_PARAM,
                               IMU_STREAM_ON_PAYLOAD))

    def send_imu_off(self):
        """IMU 스트림 OFF(STREAM_STOP). 자이로 캘리를 idle 경로에서 돌리기 위함.

        펌웨어 자이로 캘리(CMD 0x30)는 스트림 OFF(idle) 상태에서 안정적으로
        응답한다. 스트리밍 중이면(이전 실행이 켜둔 스트림 등) 캘리 응답이
        누락될 수 있어, START 전에 스트림을 끈다.
        """
        self._send(build_frame(IMU_TARGET_COMMAND, TARGET_ID_ANY,
                               DXL_INST_READ, IMU_STREAM_START_PARAM,
                               IMU_STREAM_OFF_PAYLOAD))

    # ── 수신 콜백 ────────────────────────────────────────────────────────
    def _on_rx(self, msg: ByteMultiArray):
        """CALIB(0x30) 응답과 (수집 중일 때) IMU(0x20) raw accel 을 라우팅한다."""
        data = b''.join(msg.data)
        if len(data) < 8 or data[0] != STX:
            return
        cmd = data[2]
        if cmd == CALIB_TARGET_COMMAND:
            # [STX][LEN][CMD][ID][INST][PARAM][PLEN=7][payload7][CHK] = 15바이트
            if len(data) < 15 or data[6] != 7:
                return
            status = data[7]
            bias = (
                s16_le(data[8], data[9]),
                s16_le(data[10], data[11]),
                s16_le(data[12], data[13]),
            )
            self._last_resp = (status, bias)
        elif cmd == IMU_TARGET_COMMAND and self._collecting_accel:
            # [STX][LEN][0x20][ID][INST=0x02][PARAM(FS)][PLEN=12][accel6+gyro6][CHK]
            if len(data) < 20 or data[4] != DXL_INST_READ or data[6] != 12:
                return
            ax = s16_le(data[7], data[8])
            ay = s16_le(data[9], data[10])
            az = s16_le(data[11], data[12])
            self._accel_buf.append((ax, ay, az))

    # ── 대기/수집 헬퍼 ───────────────────────────────────────────────────
    def wait_response(self, timeout_s):
        """timeout_s 안에 CALIB 응답이 오면 (status, bias) 를, 없으면 None 을 반환."""
        deadline = time.time() + timeout_s
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._last_resp is not None:
                return self._last_resp
        return None

    def wait_bridge(self, timeout_s=5.0):
        """브리지가 /mcu/raw_tx 를 구독할 때까지 대기(첫 메시지 유실 방지)."""
        deadline = time.time() + timeout_s
        while rclpy.ok() and time.time() < deadline:
            if self.pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def collect_accel(self, n, timeout_s):
        """IMU 스트림을 켜고 raw accel n개(또는 timeout)까지 수집해 리스트로 반환."""
        self._accel_buf = []
        self._collecting_accel = True
        self.send_imu_on()
        deadline = time.time() + timeout_s
        while rclpy.ok() and time.time() < deadline and len(self._accel_buf) < n:
            rclpy.spin_once(self, timeout_sec=0.02)
        self._collecting_accel = False
        return list(self._accel_buf)


def _run_gyro(node):
    """자이로 바이어스 START → 응답 파싱 → (성공 시) GET 재확인. 종료코드 반환."""
    # 자이로 캘리는 스트림 OFF(idle) 상태에서 안정적이다. 스트리밍 중이면
    # (이전 실행의 마운트 단계가 켜둔 스트림 등) START 응답이 누락될 수 있어,
    # 먼저 스트림을 끄고 잠깐 안정화한다. 캘리(~1.5s)는 브리지 워치독(2s)보다
    # 짧아, 캘리 도중 스트림이 재활성화되지 않는다.
    node.send_imu_off()
    time.sleep(0.3)
    node.get_logger().info(
        "자이로 바이어스 캘리 시작(START). 로봇을 정지 상태로 유지하세요 (~1.5s)...")
    node.send_gyro_start()
    resp = node.wait_response(timeout_s=6.0)   # 측정 ~1.5s + 여유
    if resp is None:
        node.get_logger().error("START 응답 없음(타임아웃). 브리지/펌웨어 상태를 확인하세요.")
        return 1

    status, bias = resp
    name = CALIB_STATUS.get(status, f"UNKNOWN(0x{status:02X})")
    rc = 0
    if status == 0x00:
        node.get_logger().info(f"자이로 캘리 성공! status=OK  bias={fmt_bias(bias)}")
        node.get_logger().info(
            "바이어스가 즉시 적용됩니다. 정지 시 /imu/data 의 angular_velocity 가 "
            "0 근처인지 확인하세요.")
    elif status == 0x01:
        node.get_logger().warn(
            "status=MOVED — 측정 중 움직임 감지로 폐기. 더 확실히 고정하고 다시 실행하세요.")
        rc = 2
    elif status == 0x02:
        node.get_logger().warn(
            "status=TIMEOUT — 샘플 수집 실패. IMU 스트림/FIFO 상태를 확인하세요.")
        rc = 2
    else:
        node.get_logger().warn(f"status={name}  bias={fmt_bias(bias)}")
        rc = 2

    if status == 0x00:
        time.sleep(0.2)
        node.send_gyro_get()
        g = node.wait_response(timeout_s=2.0)
        if g is not None and g[0] == 0x00:
            node.get_logger().info(f"[확인] GET bias={fmt_bias(g[1])}")
    return rc


def _run_mount(node):
    """마운트 틸트 측정(raw accel) → roll/pitch 출력. 파일은 건드리지 않는다."""
    node.get_logger().info(
        "마운팅 정렬 측정: 로봇을 '수평 기준자세'(스탠드, 상체 수평)로 두고 정지하세요...")
    samples = node.collect_accel(MOUNT_SAMPLES, MOUNT_MAX_WAIT_S)
    if len(samples) < MOUNT_SAMPLES:
        node.get_logger().error(
            f"IMU 스트림 샘플 부족({len(samples)}/{MOUNT_SAMPLES}). "
            "mcu_bridge_node/IMU 스트림 상태를 확인하세요.")
        return 1

    mean, stds, roll, pitch, total_tilt = compute_mount(samples)

    # 정지 게이트
    if max(stds) > MOUNT_ACC_STD_THRESH_LSB:
        std_str = "(%.0f, %.0f, %.0f)" % stds
        node.get_logger().warn(
            f"측정 중 흔들림 감지(accel std={std_str} LSB > {MOUNT_ACC_STD_THRESH_LSB}). "
            "더 확실히 고정하고 다시 실행하세요.")
        return 2

    # 수평 기준자세 확인
    if total_tilt > MOUNT_LEVEL_WARN_DEG:
        node.get_logger().warn(
            f"수직 대비 총기울기가 {total_tilt:.1f}° 입니다. 로봇이 수평 기준자세인지 "
            "확인하세요. 이 상태 값은 마운트 캘리로 부적절할 수 있습니다.")

    robot = socket.gethostname().split('-')[0]
    node.get_logger().info(
        "마운트 측정 완료: "
        f"roll={roll:+.2f}°  pitch={pitch:+.2f}°  "
        f"(총기울기 {total_tilt:.2f}°, accel_mean_LSB=(%.0f, %.0f, %.0f))" % mean)
    node.get_logger().info(
        "아래를 robot_params.yaml 의 해당 로봇 섹션에 붙여넣으세요 "
        "(imu_parser 회전 적용은 다음 단계, 그때 부호를 실측 검증):\n"
        f"{robot}:\n"
        f"  imu_mount_roll_deg: {roll:.2f}\n"
        f"  imu_mount_pitch_deg: {pitch:.2f}")
    return 0


def main(args=None):
    """엔트리 포인트: 기본은 자이로+마운트, --gyro-only 는 자이로만."""
    parser = argparse.ArgumentParser(description="KHUBER v2 IMU 캘리 도구(자이로+마운트)")
    parser.add_argument('--gyro-only', action='store_true',
                        help="자이로 바이어스만 실행(매 부팅용). 마운트 측정 생략")
    ns = parser.parse_args(args=args)

    rclpy.init(args=args)
    node = ImuCalibClient()

    if not node.wait_bridge(timeout_s=5.0):
        node.get_logger().error(
            "브리지(/mcu/raw_tx 구독자)를 찾지 못했습니다. mcu_bridge_node 가 떠 있나요?")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    try:
        rc = _run_gyro(node)
        if not ns.gyro_only:
            rc_mount = _run_mount(node)
            if rc == 0:
                rc = rc_mount
        # 자이로 캘리가 잠시 껐던 스트림을 정상(ON)으로 복구하고 종료.
        # (기본 모드는 마운트 단계가 이미 켜두지만, 상태 일관성을 위해 재전송)
        node.send_imu_on()
        time.sleep(0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()

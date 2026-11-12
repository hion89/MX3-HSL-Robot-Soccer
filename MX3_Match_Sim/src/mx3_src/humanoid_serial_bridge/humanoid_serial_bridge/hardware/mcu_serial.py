#!/usr/bin/env python3
"""
mcu_serial.py
=============

MCU (제어 보드) 와의 USB CDC 시리얼 통신 캡슐화.

ROS 의존성이 없는 순수 IO 레이어. 로깅은 주입된 콜백으로 처리.
mock_serial 로 교체해서 unit test 가능.

프레임 포맷 (양방향 공통):
    [0x53][LEN][CMD][ID][INST][PARAM][PLEN][PAYLOAD..][CHK]
    LEN = 전체 프레임 바이트 수 = 8 + PLEN
    CHK = bytes[0..LEN-2] 의 8-bit sum

책임:
  - 시리얼 포트 open/close
  - raw byte write
  - byte 누적 → 프레임 동기화 (STX, 길이, 체크섬 검증)
  - 프레임 빌드 (build_packet)

비책임 (Node 가 담당):
  - ROS topic 송수신
  - 타이머
  - 부팅 시 명령 전송 정책
"""
import os
import time
import serial
from typing import Callable, Optional


STX = 0x53
MIN_FRAME_LEN = 7    # STX+LEN+CMD+ID+INST+PARAM+CHK (PLEN=0 도 7바이트 안 됨, 안전 하한)
MAX_FRAME_LEN = 128  # 폭주 패킷 방어용 상한


def build_packet(command: int,
                 target_id: int,
                 instruction: int,
                 parameter: int,
                 payload: Optional[list] = None) -> bytes:
    """
    한 프레임 빌드.

      [STX=0x53][LEN][CMD][ID][INST][PARAM][PLEN][payload..][CHK]
      LEN = 8 + len(payload)
      CHK = sum(bytes[0..LEN-2]) & 0xFF
    """
    if payload is None:
        payload = [0x00]
    plen = len(payload)
    length = 8 + plen
    pkt = [STX, length, command, target_id, instruction, parameter, plen] + list(payload)
    chk = sum(pkt) & 0xFF
    pkt.append(chk)
    return bytes(pkt)


class McuSerial:
    """
    USB CDC 시리얼 wrapper.

    사용 패턴:
        mcu = McuSerial(port='/dev/ttyACM0', baud=115200,
                        log_info=node.get_logger().info,
                        log_error=node.get_logger().error)
        mcu.open()
        mcu.write_bytes(frame)
        for pkt in mcu.read_frames():
            ...
        mcu.close()
    """

    def __init__(self,
                 port: str = '/dev/ttyACM0',
                 baud: int = 115200,
                 timeout: float = 0.001,
                 log_info: Callable[[str], None] = print,
                 log_error: Callable[[str], None] = print,
                 on_reopen: Optional[Callable[[], None]] = None):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._log_info = log_info
        self._log_error = log_error
        # 재연결 직후 호출되는 콜백(노드가 IMU 스트림 재요청 등에 사용)
        self._on_reopen = on_reopen

        self.ser: Optional[serial.Serial] = None
        self._rx_buffer = bytearray()

        # 자동 재연결 throttle: 마지막 시도 시각 / 최소 간격(초)
        self._last_open_attempt = 0.0
        self._reopen_interval = 1.0

    # ───────────────────────────────────────────
    # Lifecycle
    # ───────────────────────────────────────────
    def open(self):
        """포트 열기. 실패 시 RuntimeError.

        exclusive=True: 같은 포트를 다른 프로세스가 이미 열고 있으면 열기를 거부한다.
        → 이전 인스턴스가 안 죽고 포트를 잡고 있는 'multiple access' 상황을 차단.
        """
        if not os.path.exists(self.port):
            raise RuntimeError(
                f"시리얼 포트 {self.port} 가 존재하지 않습니다. USB 연결 확인."
            )
        try:
            self.ser = serial.Serial(
                self.port, self.baud, timeout=self.timeout, exclusive=True)
        except Exception as e:
            raise RuntimeError(
                f"시리얼 포트 열기 실패(다른 프로세스가 점유 중이거나 권한 문제일 수 있음): {e}")
        self._log_info(f"시리얼 포트 {self.port} 연결됨 (baud={self.baud})")

    def is_open(self) -> bool:
        return self.ser is not None

    def _ensure_open(self):
        """ser 가 닫혀 있으면 throttle 간격으로 재연결을 시도한다.

        - 포트가 아직 없으면(USB 미연결) 조용히 다음 주기에 다시 시도.
        - 재연결에 성공하면 수신 버퍼를 비우고 on_reopen 콜백을 호출한다
          (노드가 IMU 스트림을 다시 켜는 등 복구 작업을 하도록).
        """
        if self.ser is not None:
            return
        now = time.monotonic()
        if now - self._last_open_attempt < self._reopen_interval:
            return
        self._last_open_attempt = now
        if not os.path.exists(self.port):
            return  # USB 가 아직 안 꽂힘 — 조용히 대기
        try:
            self.ser = serial.Serial(
                self.port, self.baud, timeout=self.timeout, exclusive=True)
            self._rx_buffer.clear()
            self._log_info(f"시리얼 포트 {self.port} 재연결됨")
            if self._on_reopen is not None:
                self._on_reopen()
        except Exception as e:
            # 점유/권한 등 — 다음 주기에 재시도(throttle 되어 로그 폭주 없음)
            self._log_error(f"재연결 대기 중({self.port}): {e}")

    def _handle_io_error(self, e):
        """read/write 중 IO 오류 발생 시 연결을 정리하고 재연결 모드로 전환."""
        self._log_error(f"시리얼 IO 오류, 연결 해제 처리(자동 재연결 시도 예정): {e}")
        self.close()
        self._rx_buffer.clear()

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    # ───────────────────────────────────────────
    # Raw IO
    # ───────────────────────────────────────────
    def write_bytes(self, data: bytes) -> bool:
        """raw 바이트 송신. 성공 True, 실패 False(연결 끊김 시 재연결 모드 전환)."""
        if self.ser is None:
            return False
        try:
            self.ser.write(data)
            return True
        except (serial.SerialException, OSError) as e:
            self._handle_io_error(e)
            return False

    # ───────────────────────────────────────────
    # Frame-level read
    # ───────────────────────────────────────────
    def read_frames(self) -> list[bytes]:
        """
        가능한 만큼 read해서 완전한 프레임만 리스트로 반환.

        깨진 프레임(헤더 어긋남, 길이 오류, 체크섬 불일치)은 조용히 폐기.
        반환되는 각 프레임은 STX 부터 CHK 까지 포함.
        """
        # 연결이 끊긴 상태면 재연결을 시도하고 이번 주기는 비운다.
        if self.ser is None:
            self._ensure_open()
            return []

        try:
            n = self.ser.in_waiting
            if n:
                self._rx_buffer.extend(self.ser.read(n))
        except (serial.SerialException, OSError) as e:
            # 일시적 끊김/포트 충돌 등 — 노드를 죽이지 않고 재연결 모드로.
            self._handle_io_error(e)
            return []

        frames = []
        while len(self._rx_buffer) >= 3:
            # 1) 헤더 동기화
            if self._rx_buffer[0] != STX:
                self._rx_buffer.pop(0)
                continue

            # 2) 길이 검증
            packet_length = self._rx_buffer[1]
            if packet_length < MIN_FRAME_LEN or packet_length > MAX_FRAME_LEN:
                self._rx_buffer.pop(0)
                continue

            # 3) 충분한 바이트가 들어왔는지
            if len(self._rx_buffer) < packet_length:
                break

            # 4) 한 프레임 분리
            pkt = bytes(self._rx_buffer[:packet_length])
            del self._rx_buffer[:packet_length]

            # 5) 체크섬 검증
            if (sum(pkt[:-1]) & 0xFF) != pkt[-1]:
                continue

            frames.append(pkt)

        return frames
#!/usr/bin/env python3
"""
led_control_node.py
===================

MCU 보드의 LED1~3을 USB(VCP)로 ON/OFF/점멸 제어하고 상태를 읽는 노드.
neck_control_node 와 동일하게 /mcu/raw_tx 로 프레임을 내보낸다.

[CPU 최적화]
  상태읽기 응답은 /mcu/raw_rx 로 오는데, 이 토픽은 IMU(200Hz)+목(50Hz)+ping
  까지 섞여 고레이트로 흐른다. 이걸 '상시' 구독하면 모든 메시지마다 콜백+
  역직렬화 비용이 들어 코어 1/3 가까이 낭비된다(top 기준 ~32%).
  LED 상태조회는 /led/query 받을 때만 일시적으로 필요하므로, query 직후에만
  raw_rx 를 잠깐 구독하고 응답을 받거나 0.5s 지나면 구독을 해제한다.
  → 평소(조회 안 할 때) raw_rx 구독이 없어 CPU ~0.

프레임 (CMD=LED_COMMAND=0x60):
  WRITE(설정): [0x53][LEN][0x60][0xFE][0x03][PARAM=led_id][PLEN=1][mode][CHK]
      - led_id : 1,2,3 또는 0xFF(전체),  mode : 0=OFF, 1=ON, 2=BLINK
  READ(상태조회 요청): [0x53][LEN][0x60][0xFE][0x02][0x00][PLEN=0][CHK]
  READ 응답(MCU→host): [...][0x60][0xFE][0x02][0x00][PLEN=3][m1][m2][m3][CHK]

토픽:
  구독 /led/set    (std_msgs/UInt8MultiArray, data=[led_id, mode])
  구독 /led/query  (std_msgs/Empty)  → 상태조회 요청
  발행 /led/status (std_msgs/UInt8MultiArray, data=[m1, m2, m3])
  발행 /mcu/raw_tx (std_msgs/ByteMultiArray)
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray, UInt8MultiArray, Empty

# 프로토콜 상수 (펌웨어 vcp_proto.h 와 일치)
STX             = 0x53
LED_COMMAND     = 0x60
DXL_INST_READ   = 0x02
DXL_INST_WRITE  = 0x03
LED_BROADCAST   = 0xFF          # PARAM=0xFF → 전체 LED
LED_MODE_OFF    = 0
LED_MODE_ON     = 1
LED_MODE_BLINK  = 2

# 상태조회 후 raw_rx 를 잠깐만 듣는 창(window). 이 시간 내 응답 없으면 구독 해제.
STATUS_LISTEN_SEC = 0.5


def build_frame(cmd: int, motor_id: int, inst: int, param: int,
                payload: bytes = b'') -> bytes:
    """[STX, LEN, CMD, ID, INST, PARAM, PLEN] + payload + [CHK].
    LEN = 8 + payload_len, CHK = 앞 (LEN-1) 바이트의 단순 SUM."""
    plen = len(payload)
    length = 8 + plen
    body = bytes([STX, length, cmd, motor_id, inst, param, plen]) + payload
    chk = sum(body) & 0xFF
    return body + bytes([chk])


class LedControlNode(Node):
    def __init__(self):
        super().__init__('led_control_node')

        self.pub_tx     = self.create_publisher(ByteMultiArray, '/mcu/raw_tx', 10)
        self.pub_status = self.create_publisher(UInt8MultiArray, '/led/status', 10)

        # /led/set, /led/query 만 상시 구독(저레이트). raw_rx 는 조회할 때만.
        self.create_subscription(UInt8MultiArray, '/led/set',   self.on_set,   10)
        self.create_subscription(Empty,           '/led/query', self.on_query, 10)

        self._rx_sub     = None     # raw_rx 구독 핸들 (조회 동안만 생성)
        self._stop_timer = None     # 구독 자동 해제 타이머

        self.get_logger().info(
            'led_control_node 시작. /led/set [led_id, mode] (mode 0=OFF/1=ON/2=BLINK, '
            'led_id 1~3 또는 255=전체), /led/query 로 상태조회. '
            'raw_rx 는 조회 시에만 일시 구독(평소 CPU 0).')

    # ── 프레임 송신 헬퍼 ─────────────────────────────────────────
    def _send(self, frame: bytes):
        msg = ByteMultiArray()
        msg.data = [bytes([b]) for b in frame]
        self.pub_tx.publish(msg)

    # ── /led/set → WRITE 프레임 ──────────────────────────────────
    def on_set(self, msg: UInt8MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn('/led/set 은 [led_id, mode] 2개 필요')
            return
        led_id = int(msg.data[0])
        mode   = int(msg.data[1])
        if mode > LED_MODE_BLINK:
            self.get_logger().warn(f'잘못된 mode={mode} (0/1/2만 허용)')
            return
        if led_id != LED_BROADCAST and not (1 <= led_id <= 3):
            self.get_logger().warn(f'잘못된 led_id={led_id} (1~3 또는 255)')
            return
        # PARAM=led_id, payload=[mode]
        self._send(build_frame(LED_COMMAND, 0xFE, DXL_INST_WRITE,
                               led_id, bytes([mode])))
        self.get_logger().debug(f'LED set: id={led_id} mode={mode}')

    # ── /led/query → READ 요청 + raw_rx 일시 구독 ────────────────
    def on_query(self, msg: Empty):
        # 응답을 받을 동안만 raw_rx 를 듣는다(평소엔 미구독 → CPU 0).
        if self._rx_sub is None:
            self._rx_sub = self.create_subscription(
                ByteMultiArray, '/mcu/raw_rx', self.on_raw, 10)
        # 이전 타이머 정리 후 새 해제 타이머 시작.
        if self._stop_timer is not None:
            self.destroy_timer(self._stop_timer)
            self._stop_timer = None
        self._stop_timer = self.create_timer(STATUS_LISTEN_SEC, self._stop_listening)
        # 조회 프레임 송신
        self._send(build_frame(LED_COMMAND, 0xFE, DXL_INST_READ, 0x00, b''))

    # ── raw_rx 구독 해제(응답 수신 또는 타임아웃 시) ──────────────
    def _stop_listening(self):
        if self._stop_timer is not None:
            self._stop_timer.cancel()   # 자기 콜백 안에서는 cancel만(파괴는 다음 query 때)
        if self._rx_sub is not None:
            self.destroy_subscription(self._rx_sub)
            self._rx_sub = None

    # ── /mcu/raw_rx → LED 상태 응답 파싱 (조회 창에서만 호출됨) ────
    def on_raw(self, msg: ByteMultiArray):
        data = b''.join(msg.data)
        if len(data) < 11 or data[0] != STX:
            return
        if data[2] != LED_COMMAND or data[4] != DXL_INST_READ:
            return
        if data[6] < 3:
            return
        out = UInt8MultiArray()
        out.data = [data[7], data[8], data[9]]
        self.pub_status.publish(out)
        self.get_logger().debug(f'LED status: {data[7]} {data[8]} {data[9]}')
        self._stop_listening()   # 받았으면 즉시 해제


def main():
    rclpy.init()
    node = LedControlNode()
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
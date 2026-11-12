#!/usr/bin/env python3
"""
dip_status_node.py
==================

Listens to /mcu/raw_rx for PING response frames sent by the neck-MCU
firmware and republishes the DIP-switch state on /board/dip_status.

DIP 스위치는 제어 보드(PCB)의 입력이지 목 모터의 일부가 아니므로
/neck/* 가 아니라 /board/* 네임스페이스에 둔다. 향후 다른 보드 IO
(BAT_ALRT, 버튼, 부저 등)도 같은 네임스페이스에 추가하면 응집도가 좋다.

Frame layout (PING response, 9 bytes total):
    [0x53][LEN=9][CMD=0xF0][ID=0xFF][INST=0x00][PARAM=0x00][PLEN=1][DIP][CHK]

DIP bitmap (1 byte):
    bit 0 = DIP_SW1
    bit 1 = DIP_SW2
    bit 2 = DIP_SW3
    bit 3 = DIP_SW4
    bit 4..7 = reserved

Published topics:
    /board/dip_status (std_msgs/UInt8)        - raw bitmap
    /board/dip/sw1..sw4 (std_msgs/Bool)       - 개별 스위치
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray, UInt8, Bool

PING_CMD = 0xF0
STX      = 0x53


class DipStatusNode(Node):
    def __init__(self):
        super().__init__('dip_status_node')

        # [CPU 최적화] firehose(/mcu/raw_rx) 가 아니라 PING 전용 저레이트 토픽을 구독.
        #   브릿지(mcu_bridge_node)가 CMD=0xF0(PING 응답) 프레임만 골라 이 토픽으로
        #   보내준다. 따라서 IMU(200Hz)/목(50Hz) 프레임 때문에 콜백이 깨어나지 않고,
        #   오직 PING 응답(~10Hz)에만 깨어난다 → 콜백 호출이 ~26배 감소.
        #   (DIP 스위치는 거의 안 바뀌므로 10Hz 만으로도 차고 넘친다.)
        self.sub = self.create_subscription(
            ByteMultiArray, '/mcu/raw_rx/ping', self.on_raw, 10)

        self.pub_bitmap = self.create_publisher(UInt8, '/board/dip_status', 10)
        self.pub_sw1 = self.create_publisher(Bool, '/board/dip/sw1', 10)
        self.pub_sw2 = self.create_publisher(Bool, '/board/dip/sw2', 10)
        self.pub_sw3 = self.create_publisher(Bool, '/board/dip/sw3', 10)
        self.pub_sw4 = self.create_publisher(Bool, '/board/dip/sw4', 10)

        # 마지막으로 본 상태를 기억해서 변경된 경우에만 로그 출력
        self._last_bitmap = None

        self.get_logger().info(
            'DIP status node started. Listening for PING responses (CMD=0xF0) '
            'on /mcu/raw_rx.')

    def on_raw(self, msg: ByteMultiArray):
        # [CPU 최적화] join 하기 전에 싼 검사부터.
        #   /mcu/raw_rx 는 IMU(200Hz)+목(50Hz)+ping 이 섞인 고레이트 토픽이라,
        #   대부분(IMU/목)이 우리가 원하는 PING 응답이 아니다. 매번 b''.join()
        #   으로 전체를 합치면 낭비이므로, CMD 바이트(인덱스 2)만 먼저 보고
        #   PING(0xF0)이 아니면 join 없이 즉시 반환한다.
        #   (ByteMultiArray.data 는 1바이트 bytes 객체들의 리스트라 인덱싱이 쌈)
        if len(msg.data) < 9:
            return
        if msg.data[0] != b'\x53' or msg.data[2] != b'\xF0':
            return

        data = b''.join(msg.data)

        # 최소 길이 (PING 응답은 9바이트 고정)
        if len(data) < 9:
            return
        if data[0] != STX:
            return
        if data[2] != PING_CMD:
            return
        # PLEN 필드 검증
        if data[6] != 1:
            return

        dip = data[7]

        # bitmap 발행
        bm = UInt8()
        bm.data = dip
        self.pub_bitmap.publish(bm)

        # 개별 스위치 발행
        self.pub_sw1.publish(Bool(data=bool(dip & 0x01)))
        self.pub_sw2.publish(Bool(data=bool(dip & 0x02)))
        self.pub_sw3.publish(Bool(data=bool(dip & 0x04)))
        self.pub_sw4.publish(Bool(data=bool(dip & 0x08)))

        # 변경 시에만 INFO 로그 (스팸 방지)
        if dip != self._last_bitmap:
            self._last_bitmap = dip
            self.get_logger().info(
                f'DIP changed: 0x{dip:02X}  '
                f'[SW1={bool(dip & 0x01)} SW2={bool(dip & 0x02)} '
                f'SW3={bool(dip & 0x04)} SW4={bool(dip & 0x08)}]')


def main(args=None):
    rclpy.init(args=args)
    node = DipStatusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C는 정상 종료 흐름. 스택트레이스 안 찍음.
        pass
    finally:
        node.destroy_node()
        # 시그널 핸들러가 이미 shutdown 했을 수 있으니 가드
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
    
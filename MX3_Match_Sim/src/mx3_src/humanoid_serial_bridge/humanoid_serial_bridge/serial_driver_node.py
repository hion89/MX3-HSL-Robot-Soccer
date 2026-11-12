# serial_driver_node.py
#!/usr/bin/env python3
import serial, rclpy, os
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray

CMD_DXL_TARGET = 0x10
CMD_IMU_TARGET = 0x20



class SerialDriver(Node):
    def __init__(self):
        super().__init__('serial_driver')
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value

        # --- USB 연결 확인 ---
        if not os.path.exists(port):
            self.get_logger().error(f"시리얼 포트 {port} 가 존재하지 않습니다. USB 연결을 확인하세요.")
            raise SystemExit

        try:
            self.ser = serial.Serial(port, baud, timeout=0.001)
            self.get_logger().info(f"시리얼 포트 {port} 에 연결되었습니다.")
        except Exception as e:
            self.get_logger().error(f"시리얼 포트 열기 실패: {e}")
            raise SystemExit

        # --- MCU에 데이터 송신 명령 전송 ---
        self.enable_data_stream()


        # --- ROS I/O ---
        self.sub = self.create_subscription(ByteMultiArray,
                                            '/dynamixel/raw_tx',
                                            self.write_serial, 10)
        self.pub = self.create_publisher(ByteMultiArray,
                                         '/dynamixel/raw_rx', 10)
        self.create_timer(0.005, self.read_serial)  # 200 Hz poll
        # 10 Hz ping
        self.create_timer(0.1, self.send_ping)
        self.rx_buffer = bytearray()


    def write_serial(self, msg: ByteMultiArray):
        try:
            self.ser.write(b''.join(msg.data))
        except serial.SerialException as e:
            self.get_logger().error(f'Write error: {e}')

    # def read_serial(self):
    #     n = self.ser.in_waiting
    #     if n:
    #         data = self.ser.read(n)
    #         # data는 이미 bytes 타입 → 바로 list로 바꾸지 말기
    #         msg = ByteMultiArray()
    #         msg.data = [bytes([b]) for b in data]  # 각 원소를 bytes로 감싸기
    #         self.pub.publish(msg)
    def read_serial(self):
        n = self.ser.in_waiting
        if not n:
            return

        data = self.ser.read(n)
        self.rx_buffer.extend(data)

        while len(self.rx_buffer) >= 3:
            # 헤더 동기화
            if self.rx_buffer[0] != 0x53:
                self.rx_buffer.pop(0)
                continue

            packet_length = self.rx_buffer[1]
            if packet_length < 7 or packet_length > 128:
                self.rx_buffer.pop(0)
                continue

            if len(self.rx_buffer) < packet_length:
                break

            pkt = self.rx_buffer[:packet_length]
            self.rx_buffer = self.rx_buffer[packet_length:]

            # 체크섬 확인
            if (sum(pkt[:-1]) & 0xFF) != pkt[-1]:
                continue

            # 퍼블리시
            msg = ByteMultiArray()
            msg.data = [bytes([b]) for b in pkt]
            self.pub.publish(msg)


    # ──────────────────────────────────────
    # MCU로 데이터 요청 패킷 전송
    # ──────────────────────────────────────
    def enable_data_stream(self):
        # DATA ON
        # data_pkt = self.create_packet(CMD_DXL_TARGET, 0xFE, 0x02, 0x50, [0x01])
        # self.ser.write(data_pkt)
        # self.get_logger().info("DATA ON 명령 전송")

        # IMU ON
        imu_pkt = self.create_packet(CMD_IMU_TARGET, 0xFE, 0x02, 0x10, [0x01])
        self.ser.write(imu_pkt)
        self.get_logger().info("IMU ON 명령 전송")
        
    def create_packet(self, command, target_id, instruction, parameter, payload=None):
        START_BYTE = 0x53
        if payload is None: 
            payload = [0x00]
        PAYLOAD_LENGTH = len(payload)
        PACKET_LENGTH = 8 + PAYLOAD_LENGTH
        pkt = [
            START_BYTE, PACKET_LENGTH, command, target_id,
            instruction, parameter, PAYLOAD_LENGTH
        ] + payload
        checksum = sum(pkt) & 0xFF
        pkt.append(checksum)
        return bytes(pkt)
    
    def send_ping(self):
        pkt = self.create_packet(0xF0, 0x00, 0x00, 0x00, [0x00])
        try:
            self.ser.write(pkt)
        except serial.SerialException as e:
            self.get_logger().error(f'Ping write error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SerialDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ser.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
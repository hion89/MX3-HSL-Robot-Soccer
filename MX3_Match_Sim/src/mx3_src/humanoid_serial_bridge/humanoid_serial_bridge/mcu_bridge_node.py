#!/usr/bin/env python3
"""
mcu_bridge_node.py
==================

ROS 2 노드: 제어 보드 MCU 와 ROS 토픽 사이의 브릿지.

USB CDC 시리얼로 MCU 와 양방향 통신, raw 프레임을 토픽으로 노출:
  - 수신: MCU → /mcu/raw_rx (체크섬 검증된 완전한 프레임만)
  - 송신: /mcu/raw_tx → MCU
  - 자동: 10Hz PING (MCU watchdog 유지), 부팅 시 IMU 스트림 ON

이 채널 위에 얹힌 의미별 노드:
  - imu_parser_node    : CMD=0x20 (IMU)  → /imu/*
  - dip_status_node    : CMD=0xF0 (PING resp) → /board/dip_*
  (2026-08) 목 모터가 U2D2 버스로 이관되면서 CMD=0x50(NECK_TARGET_COMMAND)
  경로는 사라졌다. 목 제어는 neck_control_node → /joint_goals 로 간다.

시리얼 IO 자체는 hardware/mcu_serial.py 의 McuSerial 에 위임한다.

────────────────────────────────────────────────────────────────────────────
[2026-08 CPU 최적화] IMU watchdog 의 /imu/data 구독 제거
────────────────────────────────────────────────────────────────────────────
■ 무엇이 문제였나

  이 노드는 IMU 스트림 생존 확인을 위해 /imu/data(200Hz)를 구독하고 있었다.
  그런데 그 토픽은 다음 경로로 만들어진다.

      MCU ──시리얼──> mcu_bridge._poll_rx
                        └─> publish /mcu/raw_rx
                              └─> imu_parser.on_raw (파싱)
                                    └─> publish /imu/data
                                          └─> mcu_bridge._on_imu_rx  ★되돌아옴
                                                └─> self._last_imu_rx = now

  즉 자기가 방금 내보낸 데이터를 DDS 한 바퀴 돌려 다시 받아, 변수 하나에
  시각을 대입하고 있었다. 그것도 초당 200번.

■ 왜 비쌌나

  rclpy 의 SingleThreadedExecutor 는 spin 반복마다 wait set 을 Python 레벨에서
  처음부터 재구축한다(노드의 모든 subscription/timer/client/QoS event 순회).
  따라서 비용은 "콜백이 무슨 일을 하느냐"가 아니라 "몇 번 깨어나느냐"에 붙는다.
  콜백 본문이 대입 한 줄이어도 깨우기 비용은 그대로 다 낸다.

  실측(py-spy, 2026-08, robo02):
      mcu_bridge 전체 샘플의 74.6% 가 rclpy/executors.py:678 (wait set 재구축)
      실제 본업(시리얼 read + publish)은 합쳐서 1.5%
      깨우기 ~411 회/초 중 200 회가 이 왕복
      → 노드 CPU 27.9% 중 약 14%p

■ 무엇을 바꿨나

  되돌아오는 화살표를 끊고, _poll_rx 가 이미 손에 쥐고 있는 원본 프레임에서
  직접 판정한다. 판정 조건은 imu_parser_node.on_raw 의 필터와 동일하다.

      MCU ──시리얼──> mcu_bridge._poll_rx
                        ├─> IMU 프레임이면 _imu_frame_seen = True   ← 추가(공짜)
                        └─> publish /mcu/raw_rx
                              └─> imu_parser → /imu/*   (이하 전부 그대로)

  ★ /imu/data, /imu/projected_gravity 의 발행 주기(200Hz)와 내용은 전혀
    바뀌지 않는다. Madgwick 필터도 imu_parser 안에서 200Hz 로 그대로 돈다.
    정책 러너가 받는 자이로/중력벡터의 신선도에 영향 없음.

■ 감시 대상이 달라지는 점 (의도된 변경)

      기존   : MCU 프레임 도착 + imu_parser 파싱 성공 + 발행 성공
      변경 후: MCU 프레임 도착

  즉 imu_parser 가 죽어도 이 워치독은 울리지 않는다. 그러나 워치독의 조치는
  _enable_imu_stream(), 즉 MCU 에 IMU ON 을 재전송하는 것뿐이다. imu_parser 가
  죽은 상황에서 MCU 에 IMU ON 을 보내봐야 아무것도 고쳐지지 않는다. 기존 코드는
  그 상황에서 2초마다 무의미한 경고와 재전송을 반복할 뿐이었다.
  → "고칠 수 있는 것만 감시한다" 로 정리된 것이므로 기능적 후퇴가 아니다.
  → imu_parser 자체의 생존 감시가 필요하면 /diagnostics 나 별도 heartbeat 로
    분리할 것. 200Hz 토픽 구독으로 할 일이 아니다.
────────────────────────────────────────────────────────────────────────────
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray

from humanoid_serial_bridge.hardware.mcu_serial import McuSerial, build_packet


# Protocol command bytes
CMD_DXL_TARGET = 0x10
CMD_IMU_TARGET = 0x20
CMD_PING       = 0xF0

# IMU 스트림 응답 프레임의 INST 바이트(pkt[4]).
#   imu_parser_node.on_raw 가 쓰는 필터 조건과 동일한 값이다.
#     imu_parser: if data[2] != 0x20 or data[4] != 0x02: return
#   두 곳이 어긋나면 워치독이 오동작하므로, imu_parser 의 필터를 고칠 일이
#   생기면 여기도 반드시 같이 고칠 것.
INST_IMU_STREAM = 0x02


class McuBridge(Node):
    def __init__(self):
        super().__init__('mcu_bridge')

        # ── Params ─────────────────────────────
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)

        # ── Hardware ──────────────────────────
        self.mcu = McuSerial(
            port=self.get_parameter('port').value,
            baud=int(self.get_parameter('baud').value),
            log_info=self.get_logger().info,
            log_error=self.get_logger().error,
            # 재연결되면 IMU 스트림을 자동으로 다시 켠다
            on_reopen=self._enable_imu_stream,
        )
        try:
            self.mcu.open()
            # 부팅 시 IMU 스트림 시작 요청
            self._enable_imu_stream()
        except RuntimeError as e:
            # 포트가 없거나 점유 중이어도 노드를 죽이지 않는다.
            # 아래 _poll_rx 타이머가 주기적으로 자동 재연결을 시도한다.
            self.get_logger().warn(f"{e}\n→ 노드는 살아있고 자동 재연결을 계속 시도합니다.")

        # ── ROS I/O ────────────────────────────
        self.sub = self.create_subscription(
            ByteMultiArray, '/mcu/raw_tx', self._on_tx, 10)
        self.pub = self.create_publisher(
            ByteMultiArray, '/mcu/raw_rx', 10)
        # [CPU 최적화] PING 응답(=DIP 상태 운반) 전용 저레이트 토픽.
        #   /mcu/raw_rx 는 IMU(200Hz)+PING(10Hz)≈210Hz 가 섞인 firehose 라,
        #   이걸 그대로 구독하는 노드는 필요 없는 프레임에도 매번 깨어나
        #   executor 재진입 + ByteMultiArray 역직렬화 비용을 낸다. DIP 상태는
        #   PING 응답(CMD=0xF0)에만 실려 오므로, 그 프레임만 따로 이 토픽으로
        #   빼서 dip_status_node 가 이것만 구독하게 한다
        #   → DIP 노드 콜백 호출이 ~210Hz 에서 ~10Hz 로 감소.
        #   (2026-08 목 이관으로 CMD=0x50 이 사라져 firehose 는 260 → 210Hz)
        self.pub_ping = self.create_publisher(
            ByteMultiArray, '/mcu/raw_rx/ping', 10)

        # 200Hz 폴링, 10Hz ping
        self.create_timer(0.005, self._poll_rx)
        self.create_timer(0.1,   self._send_ping)

        # ── IMU 수신 watchdog ──────────────────
        #   재부팅 직후 첫 IMU ON 명령을 MCU 가 놓치면 IMU 스트림이 영영 안 온다.
        #   IMU 프레임이 일정 시간 안 들어오면 IMU ON 을 자동 재전송해 복구한다.
        #   (이러면 MCU 를 손으로 리셋할 필요가 없다)
        self.declare_parameter('imu_watchdog_timeout', 2.0)  # 이 시간 IMU 없으면 재전송
        self._imu_timeout = float(self.get_parameter('imu_watchdog_timeout').value)
        self._last_imu_rx = None   # 마지막 IMU 프레임 확인 시각(초). None = 아직 없음
        self._start_time = self.get_clock().now().nanoseconds * 1e-9

        # [CPU 최적화] IMU 프레임 목격 플래그.
        #   _poll_rx(200Hz 핫패스)에서는 bool 대입만 한다. 시각 기록은 1Hz 인
        #   _imu_watchdog 에 맡긴다.
        #   ★ 핫패스에서 get_clock().now() 를 부르지 않는 이유
        #     rclpy 의 시계 호출은 C 확장 왕복이라 공짜가 아니다. 200Hz 로
        #     부르면 그 자체가 측정에 잡힌다(구버전 _on_imu_rx 에서 실제로 잡혔다).
        #     워치독 타임아웃이 2.0s 인데 점검 주기는 1Hz 이므로, 수신 시각을
        #     1Hz 해상도로만 갱신해도 판정 결과는 달라지지 않는다.
        #     (최악의 경우 감지가 1초 늦어지는데, 2s 타임아웃 대비 무의미하다)
        self._imu_frame_seen = False

        self.create_timer(1.0, self._imu_watchdog)   # 1Hz 점검

    # ───────────────────────────────────────────
    # ROS → Serial (송신)
    # ───────────────────────────────────────────
    def _on_tx(self, msg: ByteMultiArray):
        self.mcu.write_bytes(b''.join(msg.data))

    # ───────────────────────────────────────────
    # Serial → ROS (수신)
    # ───────────────────────────────────────────
    def _poll_rx(self):
        for pkt in self.mcu.read_frames():
            out = ByteMultiArray()
            out.data = [bytes([b]) for b in pkt]
            # (기존 동작) 전체 프레임 firehose. imu_parser 가 구독.
            # (2026-08) 목이 U2D2 로 이관되어 neck_state_parser 는 삭제됨.
            #           MCU 는 더 이상 목 프레임(CMD=0x50)을 주고받지 않는다.
            self.pub.publish(out)

            # [CPU 최적화] CMD 바이트(pkt[2])로 라우팅한다.
            #   pkt 는 read_frames() 가 돌려준 체크섬 검증된 완전 프레임(bytes).
            #   프레임 레이아웃: [STX][LEN][CMD][ID][INST][PARAM][PLEN]...[CHK]
            #                     0     1    2    3    4      5     6
            #   분기 비용은 정수 비교 1~2 회뿐이라 사실상 공짜다.
            if len(pkt) >= 3 and pkt[2] == CMD_PING:
                # PING 응답(0xF0)만 전용 저레이트 토픽으로 추가 발행.
                #   같은 out 객체를 재사용하므로 ByteMultiArray 리스트를 다시
                #   만들지 않는다 (publish 시 rclpy 가 내부적으로 복사하므로
                #   객체 공유는 안전).
                self.pub_ping.publish(out)

            elif (len(pkt) >= 5
                  and pkt[2] == CMD_IMU_TARGET
                  and pkt[4] == INST_IMU_STREAM):
                # IMU 스트림 프레임 목격 → watchdog 리셋 신호.
                #   과거에는 이 신호를 /imu/data 구독으로 받았다(초당 200회
                #   DDS 왕복 + executor 재진입). 원본 프레임이 바로 여기 있으므로
                #   구독 없이 판정한다. 파일 상단 주석의 배경 설명 참조.
                #   여기서는 플래그만 세운다 — 시각 기록은 _imu_watchdog(1Hz).
                self._imu_frame_seen = True

    # ───────────────────────────────────────────
    # MCU 측 watchdog / 스트림 활성화
    # ───────────────────────────────────────────
    def _send_ping(self):
        self.mcu.write_bytes(build_packet(CMD_PING, 0x00, 0x00, 0x00, [0x00]))

    def _imu_watchdog(self):
        """IMU 프레임이 timeout 이상 안 들어오면 IMU ON 을 재전송한다.

        - 시리얼이 닫혀 있으면(재연결 대기 중) 건너뛴다.
        - 시작 직후엔 _start_time 기준으로 grace 를 준다.
        - IMU 가 다시 흐르기 시작하면 _last_imu_rx 가 갱신되어 자동으로 멈춘다.
        - IMU ON 은 idempotent(펌웨어가 명시적 STREAM_START 처리)라 여러 번
          보내도 안전.
        """
        if not self.mcu.is_open():
            # 재연결 대기 중에는 판정 자체를 건너뛴다.
            # 단, 플래그는 지워둔다. 포트가 닫힌 동안 남아 있던 묵은 플래그가
            # 재연결 직후 "IMU 살아있음"으로 오판되는 것을 막는다.
            self._imu_frame_seen = False
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # 지난 1초 사이에 IMU 프레임을 한 번이라도 봤으면 수신 시각을 갱신하고
        # 플래그를 소비한다(다음 주기를 위해 내린다).
        if self._imu_frame_seen:
            self._last_imu_rx = now
            self._imu_frame_seen = False

        ref = self._last_imu_rx if self._last_imu_rx is not None else self._start_time
        if (now - ref) > self._imu_timeout:
            self.get_logger().warn(
                f"IMU 데이터 {self._imu_timeout:.0f}s 이상 수신 없음 — IMU ON 재전송")
            self._enable_imu_stream()

    def _enable_imu_stream(self):
        # IMU ON: CMD=0x20, broadcast(0xFE), INST=0x02(READ?), PARAM=0x10(start),
        #         payload=[0x01]
        pkt = build_packet(CMD_IMU_TARGET, 0xFE, 0x02, 0x10, [0x01])
        self.mcu.write_bytes(pkt)
        self.get_logger().info("IMU ON 명령 전송")


def main(args=None):
    rclpy.init(args=args)
    node = McuBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mcu.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

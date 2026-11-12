#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  button_input_node.py
# -----------------------------------------------------------------------------
#  역할
#    KHUBER 의 물리 버튼(정지/시작) 입력 계층.
#    Jetson Orin NX(reComputer J4012) 40핀 헤더에 외부 풀업으로 연결한
#    두 개의 모멘터리 푸시버튼을 폴링하여, 그 상태/이벤트를 ROS 2 토픽으로
#    발행한다. 추가로 "정지" 버튼을 누르면 안전을 위해 곧바로 토크 오프
#    (소프트웨어 e-stop)를 호출한다.
#
#    이 노드는 "런치를 띄우고 내리는" 노드가 아니다. 부팅 시 systemd 가
#    런치를 한 번 올리면, 이 노드는 다른 노드들과 함께 항상 떠 있는 상태로
#    대기하며, 버튼은 오직 "행동 활성/비활성" 신호만 준다.
#    => 런치 재기동에 따른 포트/모델 재초기화 지연 없이 즉각(수 ms) 반응.
#
#  하드웨어 전제 (test_buttons.py 와 동일한 배선)
#    - 정지(STOP) 스위치  -> 40핀 헤더 11번 핀 (BOARD)
#    - 시작(START) 스위치 -> 40핀 헤더 13번 핀 (BOARD)
#    - 공통 한쪽은 GND, 외부 풀업(10kΩ)으로 3.3V ~ GPIO 구성
#    - active-low : 평상시 HIGH(1, 뗌), 누르면 LOW(0, 눌림)
#    - 외부 풀업을 이미 달았으므로 내부 풀업/풀다운은 끈다(PUD_OFF).
#
#  발행 토픽
#    /buttons/stop_pressed   (std_msgs/Bool)
#        정지 버튼의 "현재 눌림 상태"를 그대로 미러링(눌림=True/뗌=False).
#        Latched(transient_local) QoS 로 발행하여 늦게 뜬 구독자도
#        마지막 상태를 즉시 받을 수 있게 한다.
#    /buttons/start_event    (std_msgs/Empty)
#        시작 버튼이 "눌리는 순간(하강 에지)"에 1회 펄스로 발행.
#        상태 미러링이 아니라 이벤트라는 점에 유의(엣지 트리거).
#    /buttons/stop_event     (std_msgs/Empty)
#        정지 버튼이 눌리는 순간 1회 펄스로 발행(엣지 트리거).
#
#  서비스 클라이언트
#    /u2d2/torque_enable (humanoid_behavior_interfaces/TorqueEnable)
#        정지 버튼이 눌리면 enable=False 로 호출하여 전신 토크를 끈다.
#        (fall_manager_node 와 동일한 서비스를 공유)
#    /u2d2/robot_init    (humanoid_behavior_interfaces/RobotInit)
#        시작 버튼이 눌리면 호출하여 정책 기본자세(robot_init)로 정렬한다.
#        (fall_manager_node 의 복구 단계와 동일한 서비스/파라미터 사용)
#
#  현재 동작 (FSM 없이 단순 동작)
#    - 시작 버튼 누름 -> /u2d2/robot_init 호출 (기본자세로 정렬) + 이벤트 발행
#    - 정지 버튼 누름 -> /u2d2/torque_enable(False) 호출 (토크오프) + 이벤트 발행
#
#  주의: 정지로 토크오프되어 로봇이 무너진 뒤 시작을 누르면, robot_init 은
#    "현재 누운/주저앉은 자세"에서 기본자세로 당기려 한다. 평지에서 가볍게
#    주저앉은 정도면 복귀되지만, 완전히 넘어진 경우엔 standup 이 필요하다.
#    추후 GameState FSM 을 붙여 STOPPED -> REINITIALIZING -> PLAYING 식의
#    안전한 복귀 시퀀스로 발전시키는 것을 전제로 한 임시 단순 동작이다.
#
#  다른 노드와의 연동(권장, FSM 도입 시)
#    - 행동 FSM / 보행 정책은 /buttons/stop_event(또는 stop_pressed)를
#      구독하여 즉시 cmd_vel 을 멈추고 출력을 중단한다.
#    - /buttons/start_event 를 구독하여 INITIAL <-> READY 토글 등
#      RoboCup HSL 상태머신 전환에 사용한다.
#    - FSM 을 붙일 때는 노드 내부의 robot_init/torque_off 직접 호출을 끄고
#      (call_robot_init_on_start / auto_torque_off 파라미터를 False)
#      이벤트 토픽만 FSM 이 받아 처리하도록 전환하면 된다.
#
#  파라미터
#    stop_pin        (int)   기본 11   정지 버튼 BOARD 핀 번호
#    start_pin       (int)   기본 13   시작 버튼 BOARD 핀 번호
#    poll_hz         (float) 기본 200  폴링 주파수(Hz)
#    debounce_sec    (float) 기본 0.05 소프트웨어 디바운스 시간(초)
#    auto_torque_off (bool)  기본 True 정지 버튼 누름 시 토크 오프 자동 호출 여부
#    call_robot_init_on_start (bool) 기본 True 시작 버튼 누름 시 robot_init 자동 호출 여부
#    robot_init_rpm  (float) 기본 40   robot_init 모터 동작 중 Profile Velocity (rpm)
#    robot_final_rpm (float) 기본 150  robot_init 완료 후 유지될 Profile Velocity (rpm)
#                                     ※ 이 값이 정책 보행 중에도 그대로 유지되므로,
#                                       너무 낮으면 정책의 큰 액션 변화를 모터가
#                                       못 따라간다. 보행 가능한 최소치를 잡을 것.
#    robot_init_tol_rad     (float) 기본 0.05 robot_init 목표 도달 허용 오차 (rad)
#    robot_init_settle_count (int) 기본 10   허용 오차 내 안정 카운트
#    robot_init_timeout_sec (float) 기본 5.0  robot_init 최대 대기 시간 (초)
#
#  실행
#    ros2 run humanoid_serial_bridge button_input_node
#    (GPIO 접근 권한이 필요하면 사용자 계정을 gpio 그룹에 추가하거나
#     systemd 서비스를 root 로 띄운다.)
# =============================================================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from std_msgs.msg import Bool, Empty

from humanoid_behavior_interfaces.srv import TorqueEnable, RobotInit

import os
import contextlib

# Jetson.GPIO 는 NVIDIA 공식 라이브러리. RPi.GPIO 와 API 동일.
# import 실패 시 노드 생성 단계에서 친절히 알리고 종료한다.
#
# [로그 정리] Jetson.GPIO 는 import 시점(gpio_pin_data.get_data)에 보드 모델을
#   판별하는데, reComputer J4012 처럼 NVIDIA 공식 개발자 키트가 아닌 캐리어에서는
#   "Carrier board is not from a Jetson Developer Kit ..." 경고 3줄을 print() 로
#   찍는다. 이건 라이브러리 내부 print 라서 warnings 필터로는 못 막는다.
#   이 보드에서 40핀 헤더(11/13)는 정상 동작하는 것을 확인했고, 저 경고는
#   "핀맵이 공식 검증된 건 아니다"라는 정보성 메시지일 뿐이라 로그만 더럽힌다.
#   → import 하는 동안만 stdout/stderr 를 /dev/null 로 잠시 돌려 그 메시지를 삼킨다.
#   (import 실패 시 예외 객체는 그대로 _GPIO_IMPORT_ERROR 에 보관하므로,
#    진짜 import 에러는 노드 생성 단계에서 우리가 직접 로그로 출력한다)
try:
    with open(os.devnull, 'w') as _devnull, \
            contextlib.redirect_stdout(_devnull), \
            contextlib.redirect_stderr(_devnull):
        import Jetson.GPIO as GPIO
    _GPIO_IMPORT_ERROR = None
except Exception as e:  # ImportError 외에 권한/환경 문제도 잡는다
    GPIO = None
    _GPIO_IMPORT_ERROR = e


class ButtonInputNode(Node):
    def __init__(self):
        super().__init__('button_input_node')

        # ---------------------------------------------------------------------
        #  파라미터 선언
        # ---------------------------------------------------------------------
        self.declare_parameter('stop_pin', 11)        # 정지 버튼 BOARD 핀
        self.declare_parameter('start_pin', 13)       # 시작 버튼 BOARD 핀
        self.declare_parameter('poll_hz', 200.0)      # 폴링 주파수(Hz)
        self.declare_parameter('debounce_sec', 0.05)  # 디바운스 시간(초)
        self.declare_parameter('auto_torque_off', True)  # 정지 시 토크오프 호출
        self.declare_parameter('call_robot_init_on_start', True)  # 시작 시 robot_init 호출

        # robot_init 호출 시 사용할 RPM / 도달 판정 파라미터.
        # 기본값은 fall_manager 가 쓰는 값(100/200)보다 보수적으로 낮춰뒀다.
        # 버튼 누름은 "넘어진 자세에서 임의로 호출"되는 케이스라 자세 변화가
        # 클 수 있어, fall_manager 의 "방금 일어난 직후" 케이스보다 더 천천히
        # 움직여야 안전하다.
        self.declare_parameter('robot_init_rpm', 40.0)
        self.declare_parameter('robot_final_rpm', 150.0)
        self.declare_parameter('robot_init_tol_rad', 0.05)
        self.declare_parameter('robot_init_settle_count', 10)
        self.declare_parameter('robot_init_timeout_sec', 5.0)

        self.stop_pin = int(self.get_parameter('stop_pin').value)
        self.start_pin = int(self.get_parameter('start_pin').value)
        self.poll_hz = float(self.get_parameter('poll_hz').value)
        self.debounce_sec = float(self.get_parameter('debounce_sec').value)
        self.auto_torque_off = bool(self.get_parameter('auto_torque_off').value)
        self.call_robot_init_on_start = bool(
            self.get_parameter('call_robot_init_on_start').value)
        self.robot_init_rpm = float(self.get_parameter('robot_init_rpm').value)
        self.robot_final_rpm = float(self.get_parameter('robot_final_rpm').value)
        self.robot_init_tol_rad = float(
            self.get_parameter('robot_init_tol_rad').value)
        self.robot_init_settle_count = int(
            self.get_parameter('robot_init_settle_count').value)
        self.robot_init_timeout_sec = float(
            self.get_parameter('robot_init_timeout_sec').value)

        # ---------------------------------------------------------------------
        #  GPIO 초기화
        # ---------------------------------------------------------------------
        if GPIO is None:
            # 라이브러리가 없으면 노드를 띄우는 의미가 없으므로 즉시 알리고
            # 예외를 던져 main 에서 깔끔히 종료시킨다.
            self.get_logger().fatal(
                f'Jetson.GPIO import 실패: {_GPIO_IMPORT_ERROR}. '
                f'"sudo pip3 install Jetson.GPIO" 후 다시 실행하세요.')
            raise RuntimeError('Jetson.GPIO unavailable')

        # BOARD 모드: 40핀 헤더 물리 번호(1~40)를 그대로 사용.
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        # 외부 풀업(10kΩ)을 이미 달아 active-low 가 안정적으로 동작하므로
        # 내부 풀은 필요 없다. Jetson 은 내부 풀을 핀먹스/디바이스트리에서만
        # 설정하고 setup() 의 pull_up_down 인자는 무시한다(→ "ignores setup()'s
        # pull_up_down parameter" UserWarning, setwarnings(False)로도 안 꺼짐).
        # PUD_OFF 를 명시하든 인자를 생략하든 "내부 풀 미설정"으로 동작이 동일하므로,
        # 경고가 안 뜨도록 인자를 아예 생략한다.
        GPIO.setup(self.stop_pin, GPIO.IN)
        GPIO.setup(self.start_pin, GPIO.IN)

        # ---------------------------------------------------------------------
        #  퍼블리셔
        # ---------------------------------------------------------------------
        # 정지 상태 미러링은 latched(transient_local) 로 발행해서, 나중에 뜬
        # 구독자도 마지막 정지 상태를 즉시 받게 한다. 안전 신호이므로
        # RELIABLE 로 둔다.
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pub_stop_state = self.create_publisher(
            Bool, '/buttons/stop_pressed', latched_qos)

        # 엣지 이벤트는 일반 QoS 로 1회성 펄스 발행.
        self.pub_start_event = self.create_publisher(
            Empty, '/buttons/start_event', 10)
        self.pub_stop_event = self.create_publisher(
            Empty, '/buttons/stop_event', 10)

        # ---------------------------------------------------------------------
        #  토크 오프 서비스 클라이언트 (정지 버튼용)
        # ---------------------------------------------------------------------
        # fall_manager 와 동일한 /u2d2/torque_enable 서비스를 공유한다.
        # 서비스가 아직 안 올라와 있을 수 있으므로, 호출 시점에 가용성만
        # 확인하고 비동기로 호출한다(노드 시작을 블로킹하지 않음).
        self.cli_torque = self.create_client(TorqueEnable, '/u2d2/torque_enable')

        # 시작 버튼 -> 정책 기본자세 정렬. fall_manager 와 동일한 서비스.
        self.cli_robot_init = self.create_client(RobotInit, '/u2d2/robot_init')

        # ---------------------------------------------------------------------
        #  상태 추적용 변수 (디바운스 + 엣지 검출)
        # ---------------------------------------------------------------------
        now = self.get_clock().now().nanoseconds * 1e-9  # 초 단위
        # 시작 시점의 실제 핀 값을 읽어 초기 상태로 저장한다.
        self._stop_state = GPIO.input(self.stop_pin)    # 0=눌림,1=뗌
        self._start_state = GPIO.input(self.start_pin)
        self._stop_last_change = now
        self._start_last_change = now

        # 시작 시 정지 상태를 한 번 latched 로 발행해 둔다.
        # (active-low 이므로 LOW=눌림=True)
        self._publish_stop_state()

        # ---------------------------------------------------------------------
        #  폴링 타이머
        # ---------------------------------------------------------------------
        # 인터럽트(add_event_detect) 대신 폴링을 쓰는 이유:
        #   - 외부 풀업이 갓 구성된 단계라 동작을 눈으로 확인하기 쉽고,
        #   - Jetson.GPIO 의 엣지 인터럽트가 일부 핀/JetPack 조합에서
        #     불안정하다는 보고가 있어, 진단 단계에선 폴링이 더 견고하다.
        period = 1.0 / max(self.poll_hz, 1.0)
        self.timer = self.create_timer(period, self._poll)

        self.get_logger().info(
            f'button_input_node 시작. '
            f'STOP=핀{self.stop_pin}, START=핀{self.start_pin} (BOARD, active-low), '
            f'poll={self.poll_hz:.0f}Hz, debounce={self.debounce_sec*1000:.0f}ms, '
            f'auto_torque_off={self.auto_torque_off}')

    # =========================================================================
    #  내부 헬퍼
    # =========================================================================
    def _publish_stop_state(self):
        """정지 버튼 현재 상태를 /buttons/stop_pressed 로 발행 (눌림=True)."""
        msg = Bool()
        msg.data = (self._stop_state == GPIO.LOW)  # LOW=눌림=True
        self.pub_stop_state.publish(msg)

    def _call_torque_off(self):
        """전신 토크 오프 서비스를 비동기로 호출한다 (정지 안전 조치)."""
        if not self.auto_torque_off:
            return
        if not self.cli_torque.service_is_ready():
            # 서비스가 아직 준비 안 됐으면 경고만 남기고 넘어간다.
            # (body_motor_controller 가 아직 안 떴을 수 있음)
            self.get_logger().warn(
                '/u2d2/torque_enable 서비스가 준비되지 않아 토크오프를 '
                '건너뜀. (정지 이벤트 토픽은 정상 발행됨)')
            return
        req = TorqueEnable.Request()
        req.enable = False
        # 비동기 호출. 결과는 콜백에서 로그만 남긴다(블로킹 금지).
        future = self.cli_torque.call_async(req)
        future.add_done_callback(self._on_torque_off_done)

    def _on_torque_off_done(self, future):
        try:
            resp = future.result()
            self.get_logger().info(
                f'토크 오프 완료: success={resp.success}, msg="{resp.message}"')
        except Exception as e:
            self.get_logger().error(f'토크 오프 서비스 호출 실패: {e}')

    def _call_robot_init(self):
        """정책 기본자세(robot_init) 정렬 서비스를 비동기로 호출 (시작 버튼)."""
        if not self.call_robot_init_on_start:
            return
        if not self.cli_robot_init.service_is_ready():
            # body_motor_controller 가 아직 안 떴을 수 있음.
            self.get_logger().warn(
                '/u2d2/robot_init 서비스가 준비되지 않아 호출을 건너뜀. '
                '(시작 이벤트 토픽은 정상 발행됨)')
            return
        # fall_manager 의 복구 단계와 동일한 파라미터 구조이지만, 값은
        # ROS 파라미터로 노출해 런치/CLI 에서 조절 가능하게 했다.
        #   init_rpm : 정렬 시작 시 느린 속도 (안전하게 출발)
        #   final_rpm: 정렬 완료 후 유지될 Profile Velocity (정책 보행에도 영향)
        #   pos_tolerance_rad / settle_count : 목표 도달 판정
        #   timeout_sec : 최대 대기 시간
        req = RobotInit.Request()
        req.init_rpm = self.robot_init_rpm
        req.final_rpm = self.robot_final_rpm
        req.pos_tolerance_rad = self.robot_init_tol_rad
        req.settle_count = self.robot_init_settle_count
        req.timeout_sec = self.robot_init_timeout_sec
        self.get_logger().info(
            f'robot_init 호출: init_rpm={req.init_rpm:.1f}, '
            f'final_rpm={req.final_rpm:.1f}, tol={req.pos_tolerance_rad:.3f}rad')
        future = self.cli_robot_init.call_async(req)
        future.add_done_callback(self._on_robot_init_done)

    def _on_robot_init_done(self, future):
        try:
            resp = future.result()
            self.get_logger().info(
                f'robot_init 완료: success={resp.success}, msg="{resp.message}"')
        except Exception as e:
            self.get_logger().error(f'robot_init 서비스 호출 실패: {e}')

    # =========================================================================
    #  메인 폴링 콜백
    # =========================================================================
    def _poll(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        # ---- 정지 버튼 ----
        v_stop = GPIO.input(self.stop_pin)
        if v_stop != self._stop_state:
            # 디바운스: 마지막 확정 변화 이후 debounce_sec 경과 시에만 인정
            if (now - self._stop_last_change) >= self.debounce_sec:
                self._stop_state = v_stop
                self._stop_last_change = now
                # 상태 미러링 발행(눌림/뗌 모두)
                self._publish_stop_state()
                if v_stop == GPIO.LOW:  # 눌리는 순간(하강 에지)
                    self.get_logger().info('[STOP] 정지 버튼 눌림 -> 토크오프 시도')
                    self.pub_stop_event.publish(Empty())
                    self._call_torque_off()
                else:
                    self.get_logger().info('[STOP] 정지 버튼 뗌')

        # ---- 시작 버튼 ----
        v_start = GPIO.input(self.start_pin)
        if v_start != self._start_state:
            if (now - self._start_last_change) >= self.debounce_sec:
                self._start_state = v_start
                self._start_last_change = now
                if v_start == GPIO.LOW:  # 눌리는 순간만 이벤트 발행
                    self.get_logger().info(
                        '[START] 시작 버튼 눌림 -> robot_init 시도 + start_event 발행')
                    self.pub_start_event.publish(Empty())
                    self._call_robot_init()

    # =========================================================================
    #  정리
    # =========================================================================
    def destroy_node(self):
        # 노드 종료 시 GPIO 리소스 정리. (다른 노드가 같은 칩을 안 쓰는 전제)
        try:
            if GPIO is not None:
                GPIO.cleanup()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ButtonInputNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # GPIO import 실패 등으로 생성 단계에서 던진 예외. 메시지는 이미 출력됨.
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
    
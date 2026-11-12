#!/usr/bin/env python3
# =============================================================================
#  fall_manager_node.py
# -----------------------------------------------------------------------------
#  역할
#    KHUBER 의 "낙상 안전 계층(safety layer)". 보행/킥 등 어떤 동작 중이든
#    항상 IMU 를 감시하다가, 로봇이 넘어지는 것을 감지하면 최우선으로 끼어들어
#    복구 절차를 자동으로 수행한다.
#
#  전체 흐름 (FSM)
#    NORMAL ──기울기 45° 초과──▶ PROTECT (보호자세)
#      ▲                           │ 각속도 안정 후 판정
#      │              tilt 작음 → NORMAL 복귀 / tilt 큼 → FALLEN
#      │                           ▼
#      │                        FALLEN  (재시도 카운터 0 초기화)
#      │                           │  classify_and_dispatch
#      │                           ▼
#      │                      STANDING_UP ◀───────────────┐
#      │   standup 키프레임 재생 종료(=behavior_status 의   │ 여전히 누움 →
#      │   active True→False 엣지) + 각속도 안정 후,        │ 같은 gx 방향으로
#      │   그때의 자세로 다시 classify_and_dispatch:        │ standup 재시도
#      │     • tilt < recover     → RECOVERING (성공) ──┐  │
#      │     • gx ≥ 0 & tilt 큼   → forward  standup ───┼──┘
#      │     • gx < 0 & tilt 큼   → backward standup ───┘
#      │                           │  (재시도 max 초과 시 → STANDUP_FAILED)
#      │                           ▼
#      │                      RECOVERING ──robot_init 정렬 완료──▶ NORMAL
#      └───────────────────────────┘
#
#    STANDUP_FAILED : 반복 실패 시 안전 정지. 보호자세로 토크 홀드하며 사람이
#                     세워주길 대기. 똑바로 서지면(tilt<recover & 안정)
#                     RECOVERING 으로 자동 복귀한다.
#
#  핵심 설계: "결과 판정 시점 = standup 키프레임 재생이 끝난 순간".
#    standup 서비스는 호출 즉시 리턴하고 재생은 body_motor_controller 에서
#    비동기로 돈다. 그래서 시간 추정이 아니라 /u2d2/behavior_status 토픽의
#    'standup active True→False' 엣지를 재생 종료 신호로 삼는다(실제 이벤트).
#
#  다른 노드와의 관계
#    이 노드는 현재 상태를 /robot_state (std_msgs/String) 토픽으로 계속
#    발행한다. 보행 정책(actor_onnx_runner)이나 행동 FSM(behavior_fsm)은
#    이 토픽을 구독하여, 상태가 "NORMAL" 일 때만 동작하고 그 외에는
#    출력을 멈춰야 한다. 즉 fall_manager 가 항상 우선권을 가진다.
#
#  넘어짐 판정 원리 (projected gravity 기반)
#    /imu/projected_gravity 는 로봇 몸 기준으로 본 중력 벡터다.
#      - 정자세: (0, 0, -1) 근처
#      - 기울어질수록 gx/gy 가 커지고 gz 는 0 쪽으로 올라간다
#    전체 기울기 각도(수직으로부터):
#        tilt = acos( -gz / |g| )       (0도=정자세, 90도=완전히 누움)
#    앞/뒤 방향:
#        gx > 0  -> 앞으로 엎어짐(forward),  gx < 0 -> 뒤로 자빠짐(backward)
#        (실측: 앞으로 누움 gx=+0.90, 뒤로 누움 gx=-0.96)
#
#  구독
#    /imu/projected_gravity  (geometry_msgs/Vector3Stamped)
#    /imu/data               (sensor_msgs/Imu)  ← 각속도(안정화 판정)용
#    /u2d2/behavior_status   (std_msgs/String, JSON) ← standup 재생 종료 감지용
#
#  발행
#    /robot_state            (std_msgs/String)  ← 현재 FSM 상태
#    /joint_goals            (sensor_msgs/JointState) ← 보호 자세 직접 명령
#
#  서비스 클라이언트
#    /u2d2/standup           (humanoid_behavior_interfaces/StandUp)
#    /u2d2/robot_init        (humanoid_behavior_interfaces/RobotInit)
#    /u2d2/torque_enable     (humanoid_behavior_interfaces/TorqueEnable)
#
#  실행
#    ros2 run humanoid_serial_bridge fall_manager_node
# =============================================================================

import json
import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3Stamped

# 정지 상태(/buttons/stop_pressed)는 button_input_node 가 latched(transient_local)
# 로 발행하므로, 같은 durability 로 구독해야 '나중에 떠도' 현재 정지 상태를 받는다.
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, \
    QoSHistoryPolicy

from humanoid_behavior_interfaces.srv import StandUp, RobotInit, TorqueEnable

# 보호 자세 키프레임을 RobotInitBehavior 에서 가져온다.
# (robot_init_behavior.py 의 self.protective_q 와 동일한 값을 사용)
from humanoid_serial_bridge.behaviors.robot_init_behavior import RobotInitBehavior
# [2026-08 CPU 최적화] projected gravity 파생 계산 헬퍼.
#   /imu/projected_gravity(200Hz) 구독을 없애고 cb_imu 안에서 orientation 으로
#   부터 직접 만든다. 원본과 수식·연산순서가 같아 값이 비트 단위로 동일하다.
from humanoid_serial_bridge.imu_math import projected_gravity_from_quat


# =============================================================================
#  FSM 상태 정의
#    문자열 상수로 관리한다. 그대로 /robot_state 토픽으로 발행된다.
# =============================================================================
STATE_NORMAL      = "NORMAL"        # 평상시 - 보행/킥 정책이 제어
STATE_PROTECT     = "PROTECT"       # 넘어지는 중 - 보호 자세로 전환
STATE_FALLEN      = "FALLEN"        # 완전히 누움 - 방향 판단 후 standup 호출
STATE_STANDING_UP = "STANDING_UP"   # standup 시퀀스 진행 중 + 재생 종료 후 결과 판정
STATE_RECOVERING  = "RECOVERING"    # robot_init 으로 정책 기본자세 정렬 중
STATE_STANDUP_FAILED = "STANDUP_FAILED"  # standup 반복 실패 -> 안전 정지(사람 픽업 대기)
STATE_STOPPED     = "STOPPED"       # 정지 버튼으로 의도적 토크오프 - 넘어짐 감지/standup 억제
STATE_MOTOR_FAULT = "MOTOR_FAULT"   # 모터 이상 감지 - standup 안 하고 가만히 대기(사람 개입)


class FallManagerNode(Node):
    def __init__(self):
        super().__init__("fall_manager_node")

        # =====================================================================
        #  파라미터 선언
        # =====================================================================
        # ── 넘어짐 감지 임계값 (히스테리시스) ───────────────────────
        # fall_tilt_deg : 이 각도를 넘으면 "넘어진다"고 판단 -> PROTECT 진입.
        #                 기본 45도.
        # 단일 임계값만 쓰면 경계(44~46도)에서 상태가 떨릴 수 있으므로
        # 복귀 판정에는 별도의 더 낮은 각도(recover_tilt_deg)를 쓴다.
        self.declare_parameter("fall_tilt_deg", 45.0)

        # recover_tilt_deg : 보호 자세 후 각속도가 안정됐을 때, 기울기가
        #                    이 각도보다 작으면 "다시 섰다"(NORMAL 복귀),
        #                    크면 "누웠다"(FALLEN -> standup)고 판단한다.
        #                    기본 20도.
        #
        # 주의: 별도의 "완전히 누움" 각도 문턱(예: 70도)은 두지 않는다.
        #   45도를 넘어 PROTECT 에 들어왔다는 것은 이미 넘어지는 중이라는
        #   뜻이고, 그 후의 판정 시점은 "움직임이 멈췄을 때(각속도 안정)"
        #   이다. 안정된 시점의 기울기가 작으면 다시 선 것, 크면 누운 것.
        #   임의의 각도 문턱을 두면 로봇이 완전히 누워도 그 각도에
        #   도달하지 못해 무한 대기에 빠질 수 있다.
        self.declare_parameter("recover_tilt_deg", 20.0)

        # ── 안정화(정착) 판정 ───────────────────────────────────────
        # settle_gyro_thresh : 각속도(rad/s) 크기가 이 값보다 작으면
        #                      "더 이상 구르지 않고 정착했다"고 본다.
        self.declare_parameter("settle_gyro_thresh", 0.5)
        # settle_duration_sec : 위 조건이 이 시간(초) 동안 유지되어야
        #                       완전히 누운 것으로 확정한다.
        self.declare_parameter("settle_duration_sec", 1.0)

        # ── 타이밍 ─────────────────────────────────────────────────
        # protect_min_sec : PROTECT 상태 최소 유지 시간. 보호 자세 명령을
        #                   보낸 뒤 모터가 자세를 잡을 시간을 준다.
        self.declare_parameter("protect_min_sec", 0.4)
        # standup_wait_sec : standup 서비스 호출 후, 완료를 기다리는 최대
        #                    시간. 이 시간이 지나거나 자세가 회복되면
        #                    STANDING_UP -> RECOVERING 으로 넘어간다.
        self.declare_parameter("standup_wait_sec", 10.0)
        # robot_init_wait_sec : robot_init 호출 후 정책 기본자세 정렬을
        #                       기다리는 시간.
        self.declare_parameter("robot_init_wait_sec", 5.0)

        # ── 보호 자세 명령 ─────────────────────────────────────────
        # protect_profile_rpm : 보호 자세로 전환할 때의 모터 속도(rpm).
        #                       빠르게 자세를 잡아야 하므로 비교적 크게.
        self.declare_parameter("protect_profile_rpm", 50.0)

        # ── standup 결과 판정 / 재시도 ─────────────────────────────
        # max_standup_retries : 일어서기 실패 시 자동 재시도 최대 횟수.
        #   이 횟수를 넘기면 STANDUP_FAILED 로 빠져 모터를 무한정 휘젓지
        #   않고 사람 픽업을 대기한다(골대 끼임/모터 약화 등 물리적으로
        #   불가능한 상황 보호).
        self.declare_parameter("max_standup_retries", 3)
        # post_standup_settle_sec : standup 재생이 끝난 뒤, 결과를 판정하기
        #   전에 각속도 안정(is_settled)을 기다리는 상한 시간.
        #   재생 종료 순간 아직 구르는 중이면 gx 가 순간값이라 방향을
        #   오분류할 수 있으므로 안정될 때까지 기다린 뒤 분류한다. 다만
        #   미세 진동으로 안정이 영영 안 잡힐 수 있어, 재생 종료 후 이 시간이
        #   지나면 안정 여부와 무관하게 현재 자세로 판정한다.
        self.declare_parameter("post_standup_settle_sec", 2.0)

        # ── FSM 갱신 주기 ─────────────────────────────────────────
        self.declare_parameter("update_rate_hz", 50.0)
        self.declare_parameter("torque_state_topic", "/u2d2/torque_enabled")

        # 파라미터 값 읽기
        self.fall_tilt_deg     = float(self.get_parameter("fall_tilt_deg").value)
        self.recover_tilt_deg  = float(self.get_parameter("recover_tilt_deg").value)
        self.settle_gyro       = float(self.get_parameter("settle_gyro_thresh").value)
        self.settle_duration   = float(self.get_parameter("settle_duration_sec").value)
        self.protect_min_sec   = float(self.get_parameter("protect_min_sec").value)
        self.standup_wait_sec  = float(self.get_parameter("standup_wait_sec").value)
        self.robot_init_wait   = float(self.get_parameter("robot_init_wait_sec").value)
        self.protect_rpm       = float(self.get_parameter("protect_profile_rpm").value)
        self.max_standup_retries = int(self.get_parameter("max_standup_retries").value)
        self.post_standup_settle = float(self.get_parameter("post_standup_settle_sec").value)
        update_hz              = float(self.get_parameter("update_rate_hz").value)
        self.torque_state_topic = str(
            self.get_parameter("torque_state_topic").value)

        # =====================================================================
        #  보호 자세 키프레임 로드
        #    robot_init_behavior.py 의 protective_q 를 그대로 사용한다.
        #    hw=None 으로 인스턴스를 만들어 자세 dict 만 꺼낸다.
        # =====================================================================
        try:
            rib = RobotInitBehavior(hw=None)
            self.protective_q = dict(rib.protective_q)
            self.get_logger().info(
                f"보호 자세 키프레임 로드 완료: {len(self.protective_q)} joints"
            )
        except Exception as e:
            # protective_q 가 아직 없거나 import 실패 시: 빈 dict 로 두고 경고.
            self.protective_q = {}
            self.get_logger().error(
                f"보호 자세 키프레임 로드 실패: {e} "
                f"(robot_init_behavior.py 에 protective_q 가 있는지 확인)"
            )

        # =====================================================================
        #  센서 상태 변수
        # =====================================================================
        # projected gravity (정자세에서 (0,0,-1))
        self.gx = 0.0
        self.gy = 0.0
        self.gz = -1.0
        self.got_gravity = False

        # IMU 각속도 (rad/s)
        self.wx = 0.0
        self.wy = 0.0
        self.wz = 0.0
        self.got_imu = False

        # =====================================================================
        #  FSM 상태 변수
        # =====================================================================
        self.state = STATE_NORMAL
        # 현재 상태에 진입한 시각 (경과 시간 측정용)
        self.state_enter_time = self.now()
        # "안정화 조건"이 연속으로 만족되기 시작한 시각. None 이면 미충족.
        self.settle_start_time = None
        # FALLEN 단계에서 판단한 낙하 방향 ("forward"/"backward")
        self.fall_direction = "unknown"
        # standup / robot_init 서비스를 이미 호출했는지 플래그
        self.standup_called = False
        self.robot_init_called = False
        # robot_init 서비스 응답을 받았는지 플래그.
        # RECOVERING 상태가 고정 시간을 기다리는 대신, 이 플래그가
        # 서면 즉시 NORMAL 로 전이하기 위함이다.
        self.robot_init_done = False

        # ── standup 결과 판정용 상태 ───────────────────────────────
        # standup_retry : 현재 낙상 사이클에서의 누적 재시도 횟수.
        #   FALLEN(새 낙상) 진입 시와 NORMAL 복귀 시 0 으로 초기화한다.
        #   transition() 에서는 일부러 건드리지 않는다(재시도 중 보존).
        self.standup_retry = 0
        # behavior_status(/u2d2/behavior_status) 로 받은 standup 실행 상태.
        #   standup_seen_active : 이번에 명령한 standup 이 한 번이라도
        #                         active=True 로 관측됐는가 (시작 확인).
        #   standup_ended       : 그 standup 이 active True→False 로 끝났는가
        #                         (재생 종료 엣지). dispatch_standup 에서 둘 다
        #                         재무장(False)한다.
        self.standup_seen_active = False
        self.standup_ended = False

        # =====================================================================
        #  ROS 구독 / 발행
        # =====================================================================
        # ── [2026-08 CPU 최적화] IMU 구독 2개(200Hz) → 1개(50Hz) ─────────────
        #   기존:
        #     /imu/projected_gravity (Vector3Stamped, 200Hz) → cb_gravity
        #     /imu/data              (Imu,            200Hz) → cb_imu
        #     = 초당 400회 깨우기. 그런데 FSM(update_fsm)은 50Hz 로 돈다.
        #       두 콜백 모두 값을 '저장만' 하므로 4개 중 3개는 버려졌다.
        #       rclpy 는 깨어날 때마다 wait set 을 Python 에서 재구축하므로
        #       콜백이 대입 몇 줄이어도 깨우기 비용은 그대로 다 낸다.
        #       (실측: 이 노드 CPU 의 상당 부분)
        #
        #   변경:
        #     /imu/data_50hz (Imu, 50Hz) 하나만 구독하고, 중력벡터는
        #     imu_math.projected_gravity_from_quat 로 파생 계산한다.
        #     /imu/projected_gravity 는 orientation 만의 순수 함수라 중복
        #     토픽이었고, 파생값은 원본과 비트 단위로 동일하다.
        #     → 깨우기 400 → 50 (7/8 감소)
        #
        #   지연 영향: 값이 최대 15ms 더 오래된 것이 된다. 낙상은 ~300ms 에
        #   걸쳐 일어나므로 보호자세 진입 판정에 의미 있는 영향이 없다.
        #   ★ 그래도 실기 낙상 테스트로 확인할 것 (전방/후방/측면 각 3회).
        #
        #   되돌리려면 launch 에서 imu_topic:=/imu/data 로 지정하면 200Hz 로
        #   돌아간다(중력은 여전히 파생 계산 — 값이 같으므로 무해).
        self.declare_parameter("imu_topic", "/imu/data_50hz")
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self.create_subscription(Imu, self._imu_topic, self.cb_imu, 20)
        # body_motor_controller 가 발행하는 behavior 실행 상태(JSON).
        # standup 키프레임 재생 종료 엣지를 잡기 위해 구독한다.
        self.create_subscription(
            String, "/u2d2/behavior_status", self.cb_behavior_status, 20
        )

        # ── 정지 버튼 상태 구독 ────────────────────────────────────────
        # 정지 버튼으로 '의도적으로' 토크오프해 로봇을 눕힌 경우, 이 노드가
        # 그걸 '넘어짐'으로 오인해 standup 을 걸면 안 된다. button_input_node 가
        # /buttons/stop_pressed(Bool, 눌림=True)를 latched(transient_local)로
        # 발행하므로, 같은 QoS 로 구독해 정지 상태를 항상 알 수 있게 한다.
        self._stop_button_pressed = False
        self._torque_enabled = True
        self._stopped = False   # True = 정지(의도적 토크오프) → fall FSM 억제
        stop_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Bool, "/buttons/stop_pressed", self.cb_stop_pressed, stop_qos
        )
        self.create_subscription(
            Bool, self.torque_state_topic, self.cb_torque_enabled, stop_qos
        )

        # ── 모터 진단 구독 (standup 전 모터 이상 확인용) ──────────────
        # body_motor_controller 가 /motor_diagnostics(JSON String)로 발행하는
        # 모터별 hw_error(addr 70) 를 받아, 일어나기 직전 죽은 모터가 있으면
        # standup 을 하지 않고 MOTOR_FAULT 로 대기한다.
        #   diag_stale_sec            : 이 시간 넘게 진단이 안 오면 '오래됨'으로 간주
        #   block_standup_on_stale_diag: 진단이 오래됐을 때 standup 을 막을지.
        #     기본 False = 진단 없어도 standup 진행(진단 파이프라인 다운으로
        #     로봇이 영영 못 일어나는 것 방지). True = 안전 우선(못 확인하면 대기).
        self.declare_parameter("diag_stale_sec", 2.0)
        self.declare_parameter("block_standup_on_stale_diag", False)
        self.diag_stale_sec = float(self.get_parameter("diag_stale_sec").value)
        self.block_standup_on_stale_diag = bool(
            self.get_parameter("block_standup_on_stale_diag").value)

        self._diag_stamp = None        # 마지막 진단 수신 시각(ROS 초). None=아직 없음
        self._motor_error_ids = []      # hw_error != 0 인 모터 id 목록
        self._motor_error_map = {}      # {id: "name(에러설명)"} 로그용
        self.create_subscription(
            String, "/motor_diagnostics", self.cb_motor_diag, 10
        )

        # 현재 FSM 상태를 발행 (다른 노드가 구독)
        self.state_pub = self.create_publisher(String, "/robot_state", 10)
        # 보호 자세를 직접 명령하기 위한 publisher
        self.goal_pub = self.create_publisher(JointState, "/joint_goals", 10)

        # =====================================================================
        #  서비스 클라이언트
        # =====================================================================
        self.cli_standup    = self.create_client(StandUp, "/u2d2/standup")
        self.cli_robot_init = self.create_client(RobotInit, "/u2d2/robot_init")
        self.cli_torque     = self.create_client(TorqueEnable, "/u2d2/torque_enable")

        # =====================================================================
        #  타이머: FSM 주기 갱신
        # =====================================================================
        self.create_timer(1.0 / max(1.0, update_hz), self.update_fsm)
        # 상태 발행은 조금 더 느린 주기로 (10Hz) 충분
        self.create_timer(0.1, self.publish_state)

        self.get_logger().info(
            f"FallManager 시작: fall={self.fall_tilt_deg}deg, "
            f"recover={self.recover_tilt_deg}deg, "
            f"settle={self.settle_gyro}rad/s for {self.settle_duration}s"
        )

    # =========================================================================
    #  유틸리티
    # =========================================================================
    def now(self) -> float:
        """현재 시각(초). ROS 클록 기준."""
        return self.get_clock().now().nanoseconds * 1e-9

    def cb_gravity(self, msg: Vector3Stamped):
        """[2026-08 이후 미사용] projected gravity 전용 콜백.

        /imu/projected_gravity 구독을 없애면서 호출 경로가 사라졌다.
        중력벡터는 cb_imu 안에서 orientation 으로부터 파생 계산한다.
        외부 도구가 이 노드를 재사용할 때를 대비해 함수 자체는 남겨둔다
        (수동으로 create_subscription 해서 붙이면 그대로 동작한다).
        """
        self.gx = float(msg.vector.x)
        self.gy = float(msg.vector.y)
        self.gz = float(msg.vector.z)
        self.got_gravity = True

    def cb_imu(self, msg: Imu):
        """IMU 콜백: 각속도 저장 + 중력벡터 파생 계산.

        [2026-08 CPU 최적화]
          과거에는 /imu/data 와 /imu/projected_gravity 를 각각 200Hz 로 구독해
          두 콜백이 따로 깨어났다. 후자는 전자의 orientation 만으로 만들어지는
          중복 토픽이라, 여기서 한 번에 처리하도록 합쳤다.
          값은 imu_parser 가 보내주던 것과 비트 단위로 동일하다.
        """
        self.wx = float(msg.angular_velocity.x)
        self.wy = float(msg.angular_velocity.y)
        self.wz = float(msg.angular_velocity.z)
        self.got_imu = True

        # projected gravity 파생 (구 cb_gravity 역할 흡수)
        self.gx, self.gy, self.gz = projected_gravity_from_quat(msg.orientation)
        self.got_gravity = True

    def cb_behavior_status(self, msg: String):
        """
        /u2d2/behavior_status (JSON) 콜백.
        body_motor_controller 가 현재 실행 중인 blocking behavior 상태를
        주기적으로 발행한다. 우리는 standup 의 '재생 종료 엣지'만 관심 있다.

        엣지 검출(소비자 측):
          - dispatch_standup() 가 standup 을 명령하면서 두 플래그를
            (seen_active=False, ended=False) 로 무장(arm)해 둔다.
          - 이후 standup 이 active=True 로 한 번이라도 보이면 seen_active=True.
          - seen_active 가 된 뒤 standup 이 더 이상 active 가 아니면
            (behavior!=standup 또는 active=False) → ended=True.
        이렇게 하면 1회성 메시지를 놓쳐도(연속 발행이므로) 정상 동작하며,
        직전(stale) standup 의 종료를 이번 standup 종료로 오인하지 않는다.
        """
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        is_standup_active = (data.get("behavior") == "standup"
                             and bool(data.get("active", False)))
        if is_standup_active:
            self.standup_seen_active = True
        elif self.standup_seen_active:
            # 시작을 본 뒤 더 이상 active 가 아님 → 재생 종료
            self.standup_ended = True

    def tilt_deg(self) -> float:
        """
        수직(정자세)으로부터의 전체 기울기 각도(도)를 계산한다.
        앞/뒤/옆 어느 방향이든 통합한 "얼마나 누웠나" 값.

          tilt = acos( -gz / |g| )
            - 정자세    : gz=-1 -> tilt = acos(1)   = 0도
            - 90도 누움 : gz=0  -> tilt = acos(0)   = 90도
        """
        norm = math.sqrt(self.gx ** 2 + self.gy ** 2 + self.gz ** 2)
        if norm < 1e-6:
            return 0.0
        # -gz/norm 을 [-1, 1] 로 클램프 (수치 오차로 범위를 벗어날 수 있음)
        cos_t = max(-1.0, min(1.0, -self.gz / norm))
        return math.degrees(math.acos(cos_t))

    def gyro_magnitude(self) -> float:
        """각속도 벡터의 크기(rad/s). 작을수록 정착(안정)된 상태."""
        return math.sqrt(self.wx ** 2 + self.wy ** 2 + self.wz ** 2)

    def is_settled(self) -> bool:
        """
        로봇이 "정착"했는지(더 이상 구르지 않는지) 판정한다.
        각속도 크기가 임계값 아래로 settle_duration 초 동안 연속
        유지되어야 True 를 반환한다.
        """
        if self.gyro_magnitude() < self.settle_gyro:
            # 안정 조건 만족 중. 시작 시각을 아직 안 찍었으면 지금 찍는다.
            if self.settle_start_time is None:
                self.settle_start_time = self.now()
            # 충분히 오래 유지됐는지 확인
            return (self.now() - self.settle_start_time) >= self.settle_duration
        else:
            # 흔들림 -> 안정 타이머 리셋
            self.settle_start_time = None
            return False

    def time_in_state(self) -> float:
        """현재 상태에 머문 시간(초)."""
        return self.now() - self.state_enter_time

    def transition(self, new_state: str):
        """
        FSM 상태를 전이한다.
        진입 시각을 갱신하고, 상태별 일회성 플래그를 초기화한다.
        """
        if new_state == self.state:
            return
        self.get_logger().info(
            f"[FSM] {self.state} -> {new_state}  "
            f"(tilt={self.tilt_deg():.1f}deg, gx={self.gx:+.2f})"
        )
        self.state = new_state
        self.state_enter_time = self.now()
        self.settle_start_time = None
        # 새 상태에서 서비스 호출 플래그 초기화
        self.standup_called = False
        self.robot_init_called = False
        self.robot_init_done = False
        # NORMAL 로 돌아오면(정상 복귀) 재시도 카운터를 리셋한다.
        # 그 외 전이(특히 STANDING_UP 내부 재시도)에서는 standup_retry 를
        # 보존해야 하므로 여기서 일괄 초기화하지 않는다.
        if new_state == STATE_NORMAL:
            self.standup_retry = 0

    # =========================================================================
    #  보호 자세 명령
    # =========================================================================
    def send_protective_pose(self):
        """
        보호 자세 키프레임을 /joint_goals 로 발행한다.
        joint_bridge 가 이를 받아 모터에 목표 각도로 전달한다.
        """
        if not self.protective_q:
            self.get_logger().warn("보호 자세 키프레임이 비어 있음 - 전송 생략")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.protective_q.keys())
        msg.position = [float(v) for v in self.protective_q.values()]
        self.goal_pub.publish(msg)

    # =========================================================================
    #  서비스 호출 (비동기)
    # =========================================================================
    def call_standup(self, direction: str):
        """
        standup 서비스를 비동기로 호출한다.
        direction : "forward" 또는 "backward" (이미 판단된 낙하 방향)
        """
        if self._stopped:
            self.get_logger().warn("fall_manager stopped: standup 호출 생략")
            return
        if not self.cli_standup.service_is_ready():
            self.get_logger().warn("standup 서비스가 아직 준비되지 않음")
            return
        req = StandUp.Request()
        req.command = "start"
        req.fall_direction = direction   # 방향을 명시 -> standup 내부 auto 판정 생략
        req.final_rpm = 200.0
        req.pos_tolerance_rad = 0.05
        req.settle_count = 5
        req.timeout_sec = self.standup_wait_sec
        future = self.cli_standup.call_async(req)
        future.add_done_callback(self._on_standup_response)
        self.get_logger().info(f"standup 서비스 호출: direction={direction}")

    def _on_standup_response(self, future):
        """standup 서비스 응답 콜백 (로깅 용도)."""
        try:
            res = future.result()
            self.get_logger().info(
                f"standup 응답: success={res.success}, msg='{res.message}'"
            )
        except Exception as e:
            self.get_logger().error(f"standup 서비스 호출 실패: {e}")

    def call_standup_stop(self):
        """진행 중(또는 직전)의 standup 을 정리한다.
        재시도 전에 호출해, behavior 가 'already running' 상태로 남아 새 start
        가 거부되는 것을 막는다. 이미 끝난 경우엔 무해하게 무시된다."""
        if not self.cli_standup.service_is_ready():
            return
        req = StandUp.Request()
        req.command = "stop"
        fut = self.cli_standup.call_async(req)
        fut.add_done_callback(lambda f: None)

    def dispatch_standup(self, direction: str):
        """
        지정 방향으로 standup 을 시작하고, 재생 종료 엣지 검출을 재무장(arm)한다.
        STANDING_UP 의 첫 시도와 재시도 모두 이 함수로 통일한다.
        """
        # 재생 종료 엣지 검출 재무장: 이번 standup 의 시작/종료만 보도록 초기화
        self.standup_seen_active = False
        self.standup_ended = False
        # 하드 타임아웃(standup_wait_sec)·안정 타이머 기준 시각도 다시 시작
        self.state_enter_time = self.now()
        self.settle_start_time = None
        self.fall_direction = direction
        self.call_standup(direction)

    def classify_and_dispatch(self):
        """
        현재 자세(tilt + gx)로 다음 행동을 3분기로 결정한다.
          - tilt < recover_tilt_deg          → 똑바로 섬 : RECOVERING(성공)
          - tilt 큼 & gx ≥ 0 (앞으로 엎어짐) → forward  standup
          - tilt 큼 & gx < 0 (뒤로 자빠짐)   → backward standup

        첫 낙상(FALLEN)과 standup 재생 후 재판정에서 동일하게 호출된다.
        forward/backward 의 경우 STANDING_UP 으로 (재)진입해 결과를 기다린다.

        주의: 옆으로 누운 경우(gx≈0)는 부호만으로 앞/뒤를 가르므로 애매할 수
        있지만, 재시도 때 매번 현재 gx 로 다시 분류하므로 굴러서 방향이 잡히면
        자연히 올바른 시퀀스를 고르게 된다.
        """
        tilt = self.tilt_deg()
        if tilt < self.recover_tilt_deg:
            # 일어서기 성공
            self.get_logger().info(
                f"일어서기 성공 (tilt={tilt:.1f}deg) → RECOVERING")
            self.transition(STATE_RECOVERING)
            return

        # 아직 누워있음 → 방향 분류 후 standup
        direction = "forward" if self.gx >= 0.0 else "backward"
        self.get_logger().info(
            f"여전히 누움 (tilt={tilt:.1f}deg, gx={self.gx:+.3f}) → "
            f"{direction} standup "
            f"(재시도 {self.standup_retry}/{self.max_standup_retries})"
        )
        # STANDING_UP 이 아니면 진입(첫 시도). 이미 STANDING_UP 이면 상태를
        # 유지한 채 재시도(dispatch_standup 가 타이머/엣지 플래그를 리셋한다).
        if self.state != STATE_STANDING_UP:
            self.transition(STATE_STANDING_UP)
        self.dispatch_standup(direction)

    def _evaluate_standup_result(self, tilt: float):
        """
        standup 재생이 끝난 시점에서 결과를 판정한다.
          - tilt < recover_tilt_deg : 성공 → RECOVERING
          - 여전히 누움 : 실패 → 재시도 카운터 증가 후
              * 한도 이내면 standup 정리(stop) 후 classify_and_dispatch 로
                현재 gx 방향에 맞춰 재시도
              * 한도 초과면 STANDUP_FAILED 로 안전 정지(사람 픽업 대기)
        """
        if tilt < self.recover_tilt_deg:
            self.get_logger().info(
                f"일어서기 성공 (tilt={tilt:.1f}deg) → RECOVERING")
            self.transition(STATE_RECOVERING)
            return

        # 실패 경로
        self.standup_retry += 1
        # 진행/직전 standup 을 명시적으로 정리(하드 타임아웃 경로 대비).
        self.call_standup_stop()
        if self.standup_retry > self.max_standup_retries:
            self.get_logger().error(
                f"standup {self.max_standup_retries}회 실패 "
                f"(tilt={tilt:.1f}deg) → STANDUP_FAILED (사람 픽업 대기)"
            )
            self.transition(STATE_STANDUP_FAILED)
            return
        # 한도 이내 → 현재 자세 기준으로 다시 분류해 재시도.
        # (classify_and_dispatch 가 STANDING_UP 유지한 채 dispatch_standup 수행)
        self.classify_and_dispatch()

    def call_robot_init(self):
        """robot_init 서비스를 비동기로 호출 (정책 기본자세 정렬)."""
        if self._stopped:
            self.get_logger().warn("fall_manager stopped: robot_init 호출 생략")
            return
        if not self.cli_robot_init.service_is_ready():
            self.get_logger().warn("robot_init 서비스가 아직 준비되지 않음")
            return
        req = RobotInit.Request()
        req.init_rpm = 100.0
        req.final_rpm = 200.0
        req.pos_tolerance_rad = 0.05
        req.settle_count = 10
        req.timeout_sec = self.robot_init_wait
        future = self.cli_robot_init.call_async(req)
        future.add_done_callback(self._on_robot_init_response)
        self.get_logger().info("robot_init 서비스 호출 (정책 기본자세 정렬)")

    def _on_robot_init_response(self, future):
        """robot_init 서비스 응답 콜백.
        응답을 받으면 robot_init_done 플래그를 세워, RECOVERING 상태가
        고정 시간을 다 기다리지 않고 즉시 NORMAL 로 넘어갈 수 있게 한다."""
        try:
            res = future.result()
            self.get_logger().info(
                f"robot_init 응답: success={res.success}, msg='{res.message}'"
            )
            self.robot_init_done = True
        except Exception as e:
            self.get_logger().error(f"robot_init 서비스 호출 실패: {e}")
            # 실패해도 무한정 RECOVERING 에 머물지 않도록 완료로 처리한다.
            # (시간 초과 안전장치는 update_fsm 에 별도로 있다)
            self.robot_init_done = True

    def cb_stop_pressed(self, msg: Bool):
        """정지 버튼 상태(latched). True=정지(의도적 토크오프).

        - 정지 진입(False→True) 엣지: 진행 중이던 standup 을 즉시 정리한다.
          (다시 세우지 않기 위함. 토크는 건드리지 않는다 = 계속 꺼진 채로 둠.)
        - 실제 fall FSM 억제는 update_fsm() 상단 게이트에서 처리한다.
        """
        was = self._stopped
        self._stop_button_pressed = bool(msg.data)
        self._stopped = bool(self._stop_button_pressed or not self._torque_enabled)
        if self._stopped and not was:
            self.get_logger().info(
                "정지 버튼 감지 → fall_manager 억제(STOPPED). 넘어짐 감지/standup 중단")
            # 진행 중이던 standup 키프레임이 있으면 정리(재생 중단).
            self.call_standup_stop()
        elif was and not self._stopped:
            self.get_logger().info("정지 해제 → fall_manager 재개(NORMAL)")

    def cb_torque_enabled(self, msg: Bool):
        """전체 토크 상태. False면 정지 버튼과 동일하게 fall FSM을 억제한다."""
        was = self._stopped
        self._torque_enabled = bool(msg.data)
        self._stopped = bool(self._stop_button_pressed or not self._torque_enabled)
        if self._stopped and not was:
            self.get_logger().info(
                "토크 OFF 감지 -> fall_manager 억제(STOPPED). 낙상감지/standup 중단")
            self.call_standup_stop()
        elif was and not self._stopped:
            self.get_logger().info("토크 ON 감지 -> fall_manager 재개(NORMAL)")

    def cb_motor_diag(self, msg: String):
        """/motor_diagnostics(JSON) 수신 → hw_error 있는 모터 목록 갱신.

        body_motor_controller 가 발행. 각 모터의 'hw_error'(addr 70) != 0 이면
        그 모터에 이상(과부하/과열 등)이 있는 것. standup 직전 판정에 쓴다.
        """
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        motors = data.get("motors", []) or []
        err_ids, err_map = [], {}
        for m in motors:
            code = int(m.get("hw_error", 0) or 0)
            if code != 0:
                mid = m.get("id")
                name = m.get("name", f"id{mid}")
                err_ids.append(mid)
                err_map[mid] = f"{name}(0x{code:02X})"
        self._motor_error_ids = err_ids
        self._motor_error_map = err_map
        self._diag_stamp = self.now()

    def _diag_fresh(self) -> bool:
        """진단이 최근(diag_stale_sec 이내)에 들어왔는가."""
        if self._diag_stamp is None:
            return False
        return (self.now() - self._diag_stamp) <= self.diag_stale_sec

    def _any_motor_error(self) -> bool:
        """신선한 진단 기준으로 이상 모터가 있는가. (진단이 오래됐으면 False=모름)"""
        return self._diag_fresh() and bool(self._motor_error_ids)

    def _should_block_standup(self) -> bool:
        """standup 을 막아야 하는가(=일어서지 말고 대기)를 판정.

        - 신선한 진단에 이상 모터가 있으면 → 막는다(True).
        - 진단이 오래됐으면 → block_standup_on_stale_diag 파라미터에 따름.
        - 정상(신선 + 이상 없음)이면 → 막지 않는다(False).
        """
        if self._any_motor_error():
            names = ", ".join(
                self._motor_error_map.get(i, f"id{i}") for i in self._motor_error_ids)
            self.get_logger().error(
                f"모터 이상 감지({names}) → standup 중단, 가만히 대기(MOTOR_FAULT)")
            return True
        if not self._diag_fresh():
            if self.block_standup_on_stale_diag:
                self.get_logger().warn(
                    "모터 진단 없음/오래됨 → 안전상 standup 보류(MOTOR_FAULT)")
                return True
            self.get_logger().warn(
                "모터 진단 없음/오래됨 → 상태 확인 못 함, standup 진행")
        return False

    # =========================================================================
    #  FSM 메인 갱신 루프
    #    update_rate_hz 주기로 호출된다.
    # =========================================================================
    def update_fsm(self):
        # ── 정지(stop) 게이트 ────────────────────────────────────────
        # 정지 버튼으로 의도적으로 눕힌 상태에서는 fall FSM 전체를 멈춘다.
        # standup 을 걸지 않으므로 "정지 → 넘어짐 오인 → 다시 일어남"이 사라진다.
        # 사람이 시작 버튼을 누르면(stop 해제) NORMAL 로 복귀하고,
        # 그때의 recovery(robot_init/standup)는 시작 버튼 흐름이 담당한다.
        if self._stopped:
            if self.state != STATE_STOPPED:
                # 진행 중이던 standup/protect 를 정리하고 STOPPED 로 진입.
                self.call_standup_stop()
                self.transition(STATE_STOPPED)
            return
        else:
            # 정지가 풀렸는데 아직 STOPPED 에 있으면 정상 감시로 복귀.
            if self.state == STATE_STOPPED:
                self.transition(STATE_NORMAL)

        # 센서 데이터가 아직 안 들어왔으면 아무것도 하지 않는다.
        if not self.got_gravity:
            return

        tilt = self.tilt_deg()

        # ── 상태별 처리 ─────────────────────────────────────────────
        if self.state == STATE_NORMAL:
            # 평상시. 기울기가 fall 임계값을 넘으면 넘어지는 것으로 판단.
            if tilt > self.fall_tilt_deg:
                self.get_logger().warn(
                    f"넘어짐 감지! tilt={tilt:.1f}deg > {self.fall_tilt_deg}deg"
                )
                self.transition(STATE_PROTECT)

        elif self.state == STATE_PROTECT:
            # 보호 자세로 전환 중. 매 tick 보호 자세를 계속 명령한다.
            # (한 번만 보내도 되지만, 반복 발행하면 중간에 다른 노드가
            #  목표를 덮어써도 보호 자세를 유지할 수 있다.)
            self.send_protective_pose()

            # ── 판정 시점: "움직임이 멈췄을 때" ──────────────────
            # 45도를 넘어 PROTECT 에 들어왔다는 것은 이미 넘어지는
            # 중이라는 뜻이다. 그 후의 결말은 둘 중 하나다:
            #   (a) 계속 굴러 결국 바닥에 누워서 멈춤
            #   (b) 보호 자세 덕에 다시 균형을 잡아 멈춤
            # 어느 쪽이든 "각속도가 안정됨(움직임이 멈춤)"이 판정
            # 시점이다. 별도의 각도 문턱을 기다리지 않는다.
            #
            # 안정되면 그 순간의 기울기로 결말을 가른다:
            #   tilt < recover_tilt_deg  -> 다시 섰음   -> NORMAL
            #   tilt >= recover_tilt_deg -> 누웠음      -> FALLEN
            if (self.time_in_state() >= self.protect_min_sec
                    and self.is_settled()):
                if tilt < self.recover_tilt_deg:
                    self.get_logger().info(
                        f"보호 자세 후 자세 회복 (tilt={tilt:.1f}deg) "
                        f"- NORMAL 복귀"
                    )
                    self.transition(STATE_NORMAL)
                else:
                    self.get_logger().info(
                        f"넘어짐 확정 (tilt={tilt:.1f}deg, 각속도 안정) "
                        f"- FALLEN 으로 진행"
                    )
                    self.transition(STATE_FALLEN)

        elif self.state == STATE_FALLEN:
            # 완전히 누운 상태(새 낙상). 일어서기 직전에 모터 상태를 확인한다.
            #   - 이상 모터가 있으면(과부하/과열 등) standup 을 하지 않는다.
            #     죽은 관절로는 못 일어나고, 스톨된 모터를 반복 명령하면 더 상한다.
            #     → MOTOR_FAULT 로 가서 가만히 대기(사람 개입/모터 리셋 후 자동 재시도).
            #   - 정상이면 재시도 카운터를 0 으로 초기화하고 classify_and_dispatch.
            if self._should_block_standup():
                self.transition(STATE_MOTOR_FAULT)
            else:
                self.standup_retry = 0
                self.classify_and_dispatch()

        elif self.state == STATE_STANDING_UP:
            # standup 키프레임 재생이 끝나면(=behavior_status 의 active T→F 엣지),
            # 그때의 자세로 결과를 판정한다(_evaluate_standup_result):
            #   - 똑바로 섬   → RECOVERING (성공)
            #   - 여전히 누움 → 같은 gx 방향으로 standup 재시도
            # 시간 추정이 아니라 '재생 종료' 라는 실제 이벤트에 판정을 건다.
            #
            # 하드 타임아웃(standup_wait_sec): behavior_status 를 못 받는
            # 비정상 상황(예: body_motor_controller 가 status 미발행)에서도
            # 멈추지 않도록 하는 안전장치. 정상 경로에서는 거의 쓰이지 않는다.
            ended   = self.standup_ended
            settled = self.is_settled()
            hard_timeout = self.time_in_state() >= self.standup_wait_sec

            if ended and settled:
                # 정상 판정: 재생 끝 + 안정 → 결과 분기
                self._evaluate_standup_result(tilt)
            elif ended and self.time_in_state() >= self.post_standup_settle:
                # 재생은 끝났는데 안정이 계속 안 잡힘(미세 진동 등) → 그래도 판정
                self.get_logger().warn(
                    "standup 재생 종료 후 안정 대기 초과 - 현재 자세로 판정"
                )
                self._evaluate_standup_result(tilt)
            elif hard_timeout:
                # behavior_status 미수신 등으로 ended 가 영영 안 옴 → 안전 진행
                self.get_logger().warn(
                    "standup 하드 타임아웃 - behavior_status 미수신 의심, "
                    "현재 자세로 판정"
                )
                self._evaluate_standup_result(tilt)

        elif self.state == STATE_STANDUP_FAILED:
            # 반복 실패로 안전 정지. 보호자세를 유지(토크 홀드)하며
            # 사람이 로봇을 세워주길 기다린다.
            # (토크를 풀어 늘어지게 하고 싶으면 여기서 torque_enable(False)
            #  서비스를 호출하도록 바꾸면 된다. RoboCup 픽업 상황에선 보호자세
            #  유지가 운반에 더 안전하다고 보고 기본은 홀드로 둔다.)
            self.send_protective_pose()
            # 사람이 세워주면 자동 복귀: 똑바로 섰고 안정되면 RECOVERING 으로.
            if tilt < self.recover_tilt_deg and self.is_settled():
                self.get_logger().info(
                    "STANDUP_FAILED: 자세 회복 감지(사람이 세워줌?) "
                    "→ RECOVERING 으로 복귀"
                )
                self.standup_retry = 0
                self.transition(STATE_RECOVERING)

        elif self.state == STATE_MOTOR_FAULT:
            # 모터 이상으로 일어서지 않고 가만히 대기하는 상태.
            #   - standup 을 시도하지 않는다(죽은 관절 혹사 방지).
            #   - 이상이 해소되면(모터 리셋/복구로 hw_error 가 사라지면) 다시
            #     FALLEN 으로 보내 standup 을 재시도한다. → 확인은 신선한 진단에
            #     이상이 없을 때만(오래된 진단으로 섣불리 재시도하지 않음).
            #   - 여기서는 토크/자세를 능동적으로 바꾸지 않는다. 이미 누운 상태이고,
            #     죽은 모터는 자체 래치로 토크오프돼 있을 수 있다. "가만히"가 목적.
            if self._diag_fresh() and not self._motor_error_ids:
                self.get_logger().info(
                    "모터 이상 해소 감지 → standup 재시도(FALLEN)")
                self.transition(STATE_FALLEN)
            # 아니면 그대로 대기(사람이 정지/리셋하거나 모터가 복구될 때까지).

        elif self.state == STATE_RECOVERING:
            # robot_init 으로 정책 기본자세에 정렬한다.
            # 이 단계가 중요한 이유: standup 이 끝난 직후 로봇은 standup
            # 마지막 자세에 있는데, 보행 정책은 leg_init_q 자세에서
            # 시작한다고 가정한다. 이 불일치를 robot_init 으로 맞춰야
            # 정책 재개 시 급격한 동작/넘어짐을 막을 수 있다.
            if not self.robot_init_called:
                self.call_robot_init()
                self.robot_init_called = True

            # 완료 판정: robot_init 서비스 응답을 받으면(robot_init_done)
            # 즉시 NORMAL 로 복귀한다. 고정 시간을 다 기다리지 않으므로
            # robot_init 이 빨리 끝나면 그만큼 빨리 보행을 재개한다.
            # robot_init_wait 는 응답이 오지 않을 때를 대비한 안전장치
            # (시간 초과)로만 쓴다.
            if self.robot_init_done:
                self.get_logger().info(
                    "robot_init 완료 응답 수신 - NORMAL 복귀"
                )
                self.transition(STATE_NORMAL)
            elif self.time_in_state() >= self.robot_init_wait:
                self.get_logger().warn(
                    "robot_init 응답 대기 시간 초과 - NORMAL 강제 복귀"
                )
                self.transition(STATE_NORMAL)

    # =========================================================================
    #  상태 발행
    # =========================================================================
    def publish_state(self):
        """현재 FSM 상태를 /robot_state 토픽으로 발행한다."""
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FallManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

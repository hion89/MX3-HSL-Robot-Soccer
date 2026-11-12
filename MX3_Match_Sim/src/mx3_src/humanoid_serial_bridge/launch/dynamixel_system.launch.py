# dynamixel_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import socket
import yaml
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


# ==========================================================================
# robot_params.yaml 로더
# --------------------------------------------------------------------------
# humanoid_bringup/config/robot_params.yaml 의 <로봇> 섹션에서 하드웨어
# 캘리브레이션 값(IMU 마운팅 보정)을 읽어 온다.
#
# [왜 여기서 읽나]
#   khuber_imu_calib 도구가 "robot_params.yaml 에 붙여넣으세요" 라고 안내하는데,
#   정작 dynamixel_system.launch 가 imu_parser 에 아무것도 전달하지 않아서
#   값이 파일에만 있고 반영되지 않는 상태였다(2026-08 발견).
#   → 로봇별로 다른 값이고 재부팅 후에도 유지돼야 하므로 런치가 읽어서 넘긴다.
#
# [package.xml 에 humanoid_bringup 의존을 넣지 않는 이유]
#   humanoid_bringup 이 이미 humanoid_serial_bridge 를 exec_depend 하고 있어서,
#   반대 방향을 선언하면 순환 의존이 되어 colcon build 가 깨진다.
#   humanoid_strategy / humanoid_game_controller 도 같은 이유로 선언 없이
#   런타임에만 읽는다. 이 파일도 그 관례를 따른다.
#   → 파일이 없거나 섹션이 없어도 반드시 기본값(0.0)으로 살아남아야 한다.
# ==========================================================================
def _resolve_robot_name(name: str = "") -> str:
    """섹션 이름 결정. 비면 hostname 앞부분 사용 (robo02-desktop → robo02)."""
    name = (name or "").strip()
    if name:
        return name
    host = socket.gethostname()
    return host.split("-")[0].split(".")[0]


def _load_merged_params(robot: str) -> dict:
    """robot_params.yaml 에서 common + <robot> 병합본 반환 (robot 우선).

    strategy.launch.py / game_controller.launch.py 의 로더와 동일한 규칙.
    파일이 없거나 파싱에 실패해도 예외를 던지지 않는다 — IMU 보정값 하나
    때문에 전체 하드웨어 브링업이 못 뜨는 상황을 만들면 안 된다.
    """
    try:
        cfg = os.path.join(
            get_package_share_directory("humanoid_bringup"),
            "config", "robot_params.yaml")
    except Exception:
        return {}
    if not os.path.isfile(cfg):
        return {}
    try:
        with open(cfg, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    merged = dict(data.get("common", {}) or {})
    merged.update(data.get(robot, {}) or {})
    return merged


def generate_launch_description():

    # ---- 로봇별 하드웨어 캘리브레이션 값 로딩 ----
    # generate_launch_description() 시점(= 파이썬 실행 시점)에 읽어서
    # DeclareLaunchArgument 의 기본값으로 넣는다. 이렇게 하면
    #   · 평소엔 robot_params.yaml 값이 자동 적용되고
    #   · 필요할 땐 launch arg 로 그 자리에서 덮어쓸 수 있다
    #       ros2 launch ... imu_mount_roll_deg:=1.20
    _robot = _resolve_robot_name()
    _params = _load_merged_params(_robot)

    # ---- MCU(IMU)용 포트 ----
    port_arg = DeclareLaunchArgument('port', default_value='/dev/ttyACM0')
    baud_arg = DeclareLaunchArgument('baud', default_value='115200')

    # ---- IMU 마운팅 정렬(extrinsic) 보정 ----
    # khuber_imu_calib 로 '수평 기준자세'에서 측정한 값.
    # robot_params.yaml 의 <로봇> 섹션에 없으면 0.0(보정 없음)으로 동작한다.
    imu_roll_arg = DeclareLaunchArgument(
        'imu_mount_roll_deg',
        default_value=str(float(_params.get('imu_mount_roll_deg', 0.0))),
        description=f'IMU 장착 roll 보정 (deg). 기본값은 robot_params.yaml '
                    f'[{_robot}] 섹션에서 자동 로딩.')
    imu_pitch_arg = DeclareLaunchArgument(
        'imu_mount_pitch_deg',
        default_value=str(float(_params.get('imu_mount_pitch_deg', 0.0))),
        description=f'IMU 장착 pitch 보정 (deg). 기본값은 robot_params.yaml '
                    f'[{_robot}] 섹션에서 자동 로딩.')

    # ---- U2D2(모터)용 포트 ----
    u2d2_port_arg = DeclareLaunchArgument('u2d2_port', default_value='/dev/ttyUSB0')
    u2d2_baud_arg = DeclareLaunchArgument('u2d2_baud', default_value='3000000')

    # [2026-08] U2D2 반이중 경합 제거 (TX gap).
    #   SyncWrite 직후 바이트가 선로로 빠져나갈 시간만큼 대기한다.
    #   벤치 실측(u2d2_txrx_bench.py, 20모터 @3Mbps, 토크 무관):
    #       gap   0   → SyncRead 왕복 9.45ms
    #       gap 200us → 3.03ms
    #       gap 400us → 2.86ms   ← 기본값 (SyncWrite 114B@3Mbps = 0.38ms)
    #   액션 지연(GoalDelay)은 늘지 않는다 — goal 바이트는 이미 나간 뒤에
    #   기다리는 것이기 때문(실측 p50 8.2 → 6.6ms 로 오히려 개선).
    #   ★ 초판은 tcdrain 방식이었으나 write 가 8.5ms 로 폭증해 총합이
    #     악화됐다. 경위는 u2d2_hardware.tx_gap() 주석 참조.
    #   롤백: tx_gap_us:=0.0
    tx_gap_arg = DeclareLaunchArgument(
        'tx_gap_us', default_value='400.0',
        description='SyncWrite 후 대기(us). SyncRead 경합 제거. 0=비활성')

    # ---- 목 모터 각도 안전 한계 ----
    # 부호: YAW + = 왼쪽 (REP-103 yaw), PITCH + = 위
    yaw_min_arg = DeclareLaunchArgument(
        'yaw_deg_min', default_value='-90.0',
        description='YAW 오른쪽 한계 (deg, - = 오른쪽)')
    yaw_max_arg = DeclareLaunchArgument(
        'yaw_deg_max', default_value='90.0',
        description='YAW 왼쪽 한계 (deg, + = 왼쪽)')
    pitch_min_arg = DeclareLaunchArgument(
        'pitch_deg_min', default_value='-60.0',   # 2026-07: -70 → -60 으로 아래 가동범위 축소
                                                  # (이력: -45 → -70(2026-06) → -60(2026-07))
        description='PITCH 아래 한계 (deg, - = 아래)')
    pitch_max_arg = DeclareLaunchArgument(
        'pitch_deg_max', default_value='-25.0',
        description='PITCH 위 한계 (deg). 정면(0)에서 멈춤.')

    # ---- 목 스캔 노드 파라미터 ----
    # 스캔 범위는 neck_control 의 안전 한계(±90°) 보다 안쪽으로 마진을 둔다.
    # (기구 hard-stop 충돌 방지 + AX-18A 가 끝단에서 stall 하는 걸 회피)
    # 필요 시 launch arg 로 덮어쓰기 가능:
    #   ros2 launch humanoid_serial_bridge dynamixel_system.launch.py \
    #        scan_yaw_deg_min:=-60.0 scan_yaw_deg_max:=60.0 \
    #        scan_sweep_period_sec:=4.0 scan_speed_raw:=120
    scan_yaw_min_arg = DeclareLaunchArgument(
        'scan_yaw_deg_min', default_value='-60.0',
        description='스캔 오른쪽 끝 (deg, - = 오른쪽). neck_control 한계 안쪽 추천.')
    scan_yaw_max_arg = DeclareLaunchArgument(
        'scan_yaw_deg_max', default_value='60.0',
        description='스캔 왼쪽 끝 (deg, + = 왼쪽). neck_control 한계 안쪽 추천.')
    scan_period_arg = DeclareLaunchArgument(
        'scan_sweep_period_sec', default_value='3.0',
        description='한쪽 끝→반대쪽 끝 1회 이동 추정 시간 (초). '
                    '실제 모터 속도(scan_speed_raw) 와 스윙 폭에 맞춰 튜닝. '
                    '짧으면 명령이 끊겨 모터가 덜덜거리고, 길면 끝점에서 머무는 시간↑.')
    scan_speed_arg = DeclareLaunchArgument(
        'scan_speed_raw', default_value='90',
        description='스캔 시 목 속도 (구 AX MOVING_SPEED raw, 0.111 rpm/unit). '
                    '90 ≈ 10 rpm. 목이 MX-64 로 바뀐 뒤에도 neck_scan_node 는 '
                    '이 단위로 발행하고, neck_control_node 가 rpm 으로 환산한다. '
                    '(하위호환 경로 — 새 코드는 /neck/speed_rpm 을 쓸 것)')

    # ---- 목 모터 기본 속도 (Profile Velocity, rpm) ----
    # 목이 U2D2/MX-64 로 넘어오면서 몸통과 같은 레지스터(addr 112,
    # 0.229 rpm/LSB)를 쓴다. 다만 servo_map 의 profile_velocity_exclude 에
    # 걸려 있어 전신 일괄 속도(45rpm)의 영향을 받지 않고, 이 값이 유지된다.
    #   10 rpm ≈ 60 °/s. 더 올리면 추적은 빨라지지만 카메라 흔들림 ↑.
    neck_speed_arg = DeclareLaunchArgument(
        'neck_speed_rpm', default_value='10.0',
        description='목 Profile Velocity 기본값 (rpm). 0 은 "무제한(최대속도)" 라 금지.')

    standup_yaml_arg = DeclareLaunchArgument(
        'standup_yaml',
        default_value=os.path.join(
            get_package_share_directory('humanoid_serial_bridge'),
            'config', 'standup_keyframes.yaml'
        )
    )
    kick_yaml_arg = DeclareLaunchArgument(
        'kick_yaml',
        default_value=os.path.join(
            get_package_share_directory('humanoid_serial_bridge'),
            'config', 'kick_keyframes.yaml'
        )
    )

    map_yaml = os.path.join(
        get_package_share_directory('humanoid_serial_bridge'),
        'config', 'servo_map.yaml'
    )

    # PID 게인 프로파일.
    # 게인은 RAM 영역이라 전원 OFF / 모터 REBOOT 시 EEPROM 기본값으로
    # 돌아간다. body_motor_controller 가 기동 시·리부트 후·토크 ON 전에
    # 이 파일을 자동으로 다시 쓴다.
    #
    # 튜닝 중에는 src 의 원본을 직접 가리키면 colcon build 없이
    # 수정 -> reload_gains 만으로 반복할 수 있다:
    #   gain_file:=/home/robo02/ros2_ws/src/humanoid_serial_bridge/config/motor_gains.yaml
    gain_yaml_default = os.path.join(
        get_package_share_directory('humanoid_serial_bridge'),
        'config', 'motor_gains.yaml'
    )
    gain_arg = DeclareLaunchArgument(
        'gain_file', default_value=gain_yaml_default,
        description='PID 게인 YAML 경로. 빈 문자열이면 게인을 건드리지 않음'
    )

    return LaunchDescription([
        port_arg, baud_arg, gain_arg,
        imu_roll_arg, imu_pitch_arg,
        u2d2_port_arg, u2d2_baud_arg, standup_yaml_arg, kick_yaml_arg,
        # [2026-08] ★ 선언(DeclareLaunchArgument)만 하고 여기 등록을 빠뜨리면
        #   launch 인자가 조용히 무시된다. 선언과 등록은 항상 한 쌍이다.
        tx_gap_arg,
        yaw_min_arg, yaw_max_arg, pitch_min_arg, pitch_max_arg, neck_speed_arg,
        # 스캔 노드 파라미터 (런타임 launch arg 로 덮어쓰기 가능)
        scan_yaw_min_arg, scan_yaw_max_arg, scan_period_arg, scan_speed_arg,

        # ────── MCU 통신 노드 (USB CDC ↔ ROS topics) ──────
        Node(
            package='humanoid_serial_bridge',
            executable='mcu_bridge_node',
            name='mcu_bridge',
            parameters=[{
                'port': LaunchConfiguration('port'),
                'baud': LaunchConfiguration('baud')
            }],
            output='screen'
        ),

        # ────── IMU 파서 노드 ──────
        Node(
            package='humanoid_serial_bridge',
            executable='imu_parser_node',
            name='imu_parser',
            parameters=[{
                # 로봇별 IMU 장착 오차 보정. robot_params.yaml 에서 자동 로딩되며,
                # 값이 0 이 아니면 기동 로그에 "IMU 마운팅 보정 적용: ..." 이 뜬다.
                # 반영 여부는 수평 기준자세에서 /imu/projected_gravity 가
                # (0, 0, -1) 에 가까워지는지로 확인한다.
                'imu_mount_roll_deg': LaunchConfiguration('imu_mount_roll_deg'),
                'imu_mount_pitch_deg': LaunchConfiguration('imu_mount_pitch_deg'),
            }],
            output='screen'
        ),

        # ────── (삭제됨) neck_state_parser_node ──────
        # 2026-08 목 이관: 목 Present Position 은 이제 body_motor_controller 의
        # SyncRead 에 함께 실려 /joint_states 로 나간다(50Hz). MCU 프레임을
        # 파싱하던 이 노드는 더 이상 필요 없다.
        # deg 단위 상태(/neck/yaw_deg_state 등)는 neck_control_node 가
        # /joint_states 를 되받아 재발행한다.

        # ────── DIP 스위치 상태 수신 (PING 응답 필터) ──────
        Node(
            package='humanoid_serial_bridge',
            executable='dip_status_node',
            name='dip_status',
            output='screen'
        ),

        # ────── 물리 버튼 입력 (정지/시작, GPIO) ──────
        # 런치와 함께 항상 떠서 11/13핀을 폴링하며 /buttons/* 토픽을 발행한다.
        # 버튼은 런치를 띄우고 내리는 게 아니라 "행동 on/off" 신호만 준다.
        #   정지 누름 -> /buttons/stop_event 발행 + 토크오프 자동 호출
        #   시작 누름 -> /buttons/start_event 발행 + robot_init 자동 호출
        Node(
            package='humanoid_serial_bridge',
            executable='button_input_node',
            name='button_input',
            output='screen',
            parameters=[{
                'stop_pin': 11,                    # 정지 버튼 BOARD 핀
                'start_pin': 13,                   # 시작 버튼 BOARD 핀
                'poll_hz': 50.0,                   # 폴링 주파수 (200→50: Jetson.GPIO 부하 ↓, 50ms 디바운스엔 충분)
                'debounce_sec': 0.05,              # 디바운스 50ms
                'auto_torque_off': True,           # 정지 시 토크오프 자동 호출
                'call_robot_init_on_start': True,  # 시작 시 robot_init 자동 호출
                # robot_init 동작 속도(fall_manager 의 100/200 보다 보수적으로 낮춤).
                # final_rpm 은 init 후 정책 보행 중에도 유지되니 너무 낮추지 말 것.
                'robot_init_rpm': 20.0,            # 초기화 중 모터 속도 (rpm)
                'robot_final_rpm': 45.0,          # 초기화 완료 후 유지 속도 (rpm). 모터 한계 45.2 → 45.0 = 사실상 최대
                'robot_init_tol_rad': 0.05,        # 목표 도달 허용 오차 (rad)
                'robot_init_settle_count': 10,     # 허용 오차 내 안정 카운트
                'robot_init_timeout_sec': 5.0,     # 최대 대기 시간 (초)
            }],
        ),

        # ────── 목 제어 (deg API ↔ U2D2 관절 rad 어댑터) ──────
        # 2026-08 목 이관: MCU 프레임 생성기가 아니라 단위 변환 어댑터가 됐다.
        #   /neck/*_deg  → /joint_goals (rad)      → body_motor_controller
        #   /joint_states (rad) → /neck/*_deg_state (deg)  ← 구 parser 역할 흡수
        Node(
            package='humanoid_serial_bridge',
            executable='neck_control_node',
            name='neck_control',
            parameters=[{
                'yaw_deg_min':   LaunchConfiguration('yaw_deg_min'),
                'yaw_deg_max':   LaunchConfiguration('yaw_deg_max'),
                'pitch_deg_min': LaunchConfiguration('pitch_deg_min'),
                'pitch_deg_max': LaunchConfiguration('pitch_deg_max'),
                # URDF/servo_map 과 반드시 일치해야 하는 관절명
                'yaw_joint_name':   'neck_yaw_joint',
                'pitch_joint_name': 'neck_pitch_joint',
                # API deg ↔ URDF rad 부호. 기존 AX-18A 시절 규약 유지.
                #   pitch 는 API(+ = 위) 와 URDF 가 반대라 -1.
                'yaw_sign':    1.0,
                'pitch_sign': -1.0,
                # 목 Profile Velocity 기본값. 카메라 흔들림과의 절충값.
                # (servo_map 의 profile_velocity_exclude 때문에 전신 일괄
                #  속도 변경의 영향을 받지 않는다)
                'default_speed_rpm': LaunchConfiguration('neck_speed_rpm'),
                'speed_rpm_max': 30.0,
            }],
            output='screen'
        ),

        # ────── 목 스캔 (좌우 핑퐁 탐색) ──────
        # 항상 떠 있고 /neck/scan_enable (Bool) 토픽으로 켜고 끈다.
        #   scan-on  : ros2 topic pub --once /neck/scan_enable std_msgs/Bool "{data: true}"
        #   scan-off : ros2 topic pub --once /neck/scan_enable std_msgs/Bool "{data: false}"
        # 시작 시 자동 동작하지 않으므로 idle 상태에서는 토픽/모터에
        # 영향 없음 (구독자 1개 추가될 뿐).
        Node(
            package='humanoid_serial_bridge',
            executable='neck_scan_node',
            name='neck_scan',
            parameters=[{
                'yaw_min_deg':      LaunchConfiguration('scan_yaw_deg_min'),
                'yaw_max_deg':      LaunchConfiguration('scan_yaw_deg_max'),
                'sweep_period_sec': LaunchConfiguration('scan_sweep_period_sec'),
                'moving_speed_raw': LaunchConfiguration('scan_speed_raw'),
            }],
            output='screen'
        ),

        # ────── 전신 모터 제어 (U2D2) ──────
        # 2026-08 이관 후 목(ID 19/20, MX-64)까지 이 노드가 소유한다.
        # servo_map.yaml 의 joint_map 이 20개인지, 기동 로그의 "Used IDs" 에
        # 19/20 이 보이는지 반드시 확인할 것.
        Node(
            package='humanoid_serial_bridge',
            executable='body_motor_controller_node',
            name='body_motor_controller',
            output='screen',
            parameters=[{
                'map_file': map_yaml,
                'device_name': LaunchConfiguration('u2d2_port'),
                'baudrate': LaunchConfiguration('u2d2_baud'),
                'read_rate_hz': 50.0,
                # [2026-08] SyncWrite→SyncRead 경합 제거. 위 tx_gap_arg 주석 참조.
                'tx_gap_us': LaunchConfiguration('tx_gap_us'),
                'standup_keyframe_yaml': LaunchConfiguration('standup_yaml'),
                'kick_keyframe_yaml': LaunchConfiguration('kick_yaml'),
                # 모터 Velocity Limit(45.2rpm)에 맞춘 Profile Velocity 상한.
                # standup/kick _finish() 의 final_rpm(200)이 이 값으로 clamp 되어
                # 씹히지 않고 보행 가능 속도로 남는다. 45.2 미만이라 0.2 여유로 45.0.
                'profile_velocity_limit_rpm': 45.0,
                # PID 게인 (RAM). 빈 문자열이면 미적용 = 기존 동작 유지.
                'gain_file': LaunchConfiguration('gain_file'),
                'gain_verify': True,
            }],
        ),

        # ────── 보드 LED 제어 (LED1~3) ──────
        # /led/set [led_id, mode] 구독, /led/status 발행. mcu_bridge 통해 USB.
        Node(
            package='humanoid_serial_bridge',
            executable='led_control_node',
            name='led_control',
            output='screen',
        ),

    ])
# v1s_joypad_test.launch.py
#
# v1s(stop-aware gait) 정책 + 조이패드 텔레옵 벤치 테스트용 launch.
#
# 구성:
#   joy_node                  → /joy
#   joypad_teleop_node    → /cmd_vel, /gait_frequency, /gait_mode, /kick_trigger
#   actor_onnx_walking_runner     → /joint_goals, /gait/phase, /gait/state
#
# [먼저 띄워야 하는 것]
#   ros2 launch humanoid_serial_bridge dynamixel_system.launch.py
#     → /joint_states, /imu/data, /imu/projected_gravity 를 발행하고
#       /joint_goals 를 받는다. 이게 없으면 러너가 tick 자체를 못 돈다.
#   (선택) ros2 run humanoid_serial_bridge fall_manager_node
#     → 바닥에 내려놓고 테스트할 때만. 매달아놓는 첫 테스트에는 불필요.
#
# [띄우면 안 되는 것]
#   humanoid_bringup 의 control.launch.py
#     → v0 계열 actor 를 같이 올려 /joint_goals 가 충돌한다.
#
# 주의: 이 launch 는 twist_mux 를 거치지 않고 /cmd_vel 로 직결한다.
#       벤치에서 정책 단독 검증용이다. 경기 파이프라인에 넣을 때는
#       cmd_vel_topic 을 cmd_vel_teleop 로 remap 해 twist_mux 를 태울 것.
#
# 실행:
#   # 패키지 안의 models/policy_v1s.onnx 를 쓰는 경우
#   ros2 launch humanoid_policy_runner v1s_joypad_test.launch.py
#
#   # 임의 경로의 모델을 쓰는 경우 (튜닝 중에는 이쪽이 편하다.
#   #  data_files 는 --symlink-install 로도 복사되므로, 패키지 안에 두면
#   #  모델을 바꿀 때마다 colcon build 를 다시 돌려야 한다)
#   ros2 launch humanoid_policy_runner v1s_joypad_test.launch.py \
#       actor_path:=/home/robo02/models/policy_v1s.onnx
#
#   # v1 정책(관측 235)을 테스트하는 경우
#   ros2 launch humanoid_policy_runner v1s_joypad_test.launch.py \
#       policy_variant:=v1 \
#       actor_path:=/home/robo02/models/policy_v1.onnx
#   → v1 은 gait_mode 개념이 없다. 속도 0 이면 자동으로 정지(위상 동결)라
#     D-pad 모드 전환은 무시되고 경고만 뜬다. 주파수도 2.0 고정이므로
#     D-pad 상하 역시 의미가 없다.
#
# 첫 테스트는 반드시 로봇을 매단 상태에서:
#   1) gait_mode:=stop 으로 시작 → 축/버튼 매핑 확인
#   2) D-pad 우 → march 로 전환 → 제자리 걷기 확인
#   3) D-pad 우 → stop → 두 발이 모이며 멈추는지 확인
#   4) D-pad 우 → auto → 왼쪽 스틱 살짝 밀어 전진

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    policy_variant = LaunchConfiguration('policy_variant')
    actor_path = LaunchConfiguration('actor_path')
    joy_dev = LaunchConfiguration('joy_dev')
    gait_mode = LaunchConfiguration('gait_mode')
    gait_freq = LaunchConfiguration('gait_freq')
    enable_button = LaunchConfiguration('enable_button')
    grav_offset_x = LaunchConfiguration('grav_offset_x')
    grav_offset_y = LaunchConfiguration('grav_offset_y')

    # 패키지 share 안의 기본 모델 경로.
    # 빈 문자열을 default 로 두면 러너에서 expanduser("") 가 홈 디렉터리로
    # 해석되어 "actor_path not found: /home/user" 로 죽는다.
    default_actor = os.path.join(
        get_package_share_directory('humanoid_policy_runner'),
        'models',
        'policy_v1s.onnx',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            # v0_1 : 관측 235. v0 + 위상만. 게이팅 없음, 정지 시 위상 0 수렴
            # v1   : 관측 235. 게이팅 있음, 정지 시 위상 동결
            # v1s  : 관측 240. gait_mode 로 보행/제자리걷기/정지 전환
            #
            # ★ v0_1 과 v1 은 차원이 같아 자동 판별이 안 된다 ★
            #   잘못 지정하면 조용히 오동작하므로 반드시 정확히 줄 것.
            'policy_variant', default_value='v1s',
            description='정책 변형 (v0_1 | v1 | v1s). ONNX 차원과 반드시 일치',
        ),
        DeclareLaunchArgument(
            'actor_path',
            default_value=default_actor,
            description='v1s ONNX 경로 (관측 240 차원). 기본값은 패키지 share/models',
        ),
        DeclareLaunchArgument(
            'joy_dev', default_value='/dev/input/js0',
            description='조이스틱 디바이스. ls /dev/input/js* 로 확인',
        ),
        DeclareLaunchArgument(
            'gait_mode', default_value='stop',
            description='초기 보행 모드 (auto|march|stop). 벤치는 stop 권장',
        ),
        DeclareLaunchArgument(
            'gait_freq', default_value='2.0',
            description='초기 보행 주파수 [Hz]. 학습 분포는 1.8~2.2',
        ),
        DeclareLaunchArgument(
            # 가상 중력 오프셋. 정책이 보는 자세를 트림한다.
            #   -x = 정책이 "더 앞으로 숙었다" 고 인지 -> 뒤로 보정 -> 실제로 세워짐
            #   (코드는 g_b = 실측 - offset 이므로 음수를 주면 관측값이 커진다)
            # IMU 장착 오차 보정(imu_mount_pitch_deg)과는 별개다.
            #
            # robo02 실측 (v0_2, cmd_vx 0.25, 게인 P800/D500):
            #   offset  0.0   딱딱 +3.90°  잔디 +5.34°
            #   offset -0.1   딱딱 -1.85°  잔디 +0.42°   (딱딱에서 과함)
            #   offset -0.07  두 지면 절충값. sim(+1.15°)에 가장 근접
            #
            # 실시간 튜닝:
            #   ros2 param set /actor_onnx_walking_node grav_offset_x -- -0.07
            'grav_offset_x', default_value='-0.07',
            description='중력 오프셋 x. 앞으로 쏠리면 -값 (-0.07 ~= 4.0deg)',
        ),
        DeclareLaunchArgument(
            'grav_offset_y', default_value='0.0',
            description='중력 오프셋 y. 좌우 쏠림 보정',
        ),
        DeclareLaunchArgument(
            # 데드맨 스위치. 기본 비활성(-1) — 조작성 때문에 껐다.
            # 즉시 정지는 B 버튼(btn1)이 담당한다: gait_mode -> stop.
            # 바닥 테스트에서 데드맨을 다시 쓰려면 enable_button:=5 (RB).
            'enable_button', default_value='-1',
            description='데드맨 버튼 인덱스. -1=비활성(기본). RB=5',
        ),

        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'device_name': '',
                'dev': joy_dev,
                # 스틱을 가만히 두어도 주기적으로 발행되도록 한다.
                # 0 이면 값 변화가 있을 때만 발행되어 D-pad 를 누르고 있어도
                # 주파수 반복 증감이 동작하지 않는다.
                'autorepeat_rate': 20.0,
                # raw 데드존은 작게. 실제 데드존은 텔레옵 노드가 축별로 처리한다.
                # (control_joypad.launch.py 와 동일하게 0.05)
                'deadzone': 0.05,
            }],
            output='screen',
        ),

        Node(
            package='humanoid_policy_runner',
            executable='joypad_teleop_node',
            name='joypad_teleop_node',
            parameters=[{
                'cmd_vel_topic': 'cmd_vel',
                'enable_button': enable_button,
                'gait_mode_default': gait_mode,
                'gait_freq_default': gait_freq,
                # 학습 분포와 일치시킬 것 (velocity_v1s_env_cfg.py 의
                # StoppableGaitPhaseCommandCfg.frequency_range 참조)
                'gait_freq_min': 1.8,
                'gait_freq_max': 2.2,
                # 1.8~2.2 를 0.1 씩 → 1.8/1.9/2.0/2.1/2.2 다섯 단계
                'gait_freq_step': 0.1,
                # D-pad 매핑. `ros2 topic echo /joy` 로 확인 후 조정.
                # buttons 로 오는 패드라면 axis_dpad_* 를 -1 로 두고
                # button_freq_up/down, button_mode_next/prev 를 지정한다.
                # 축/버튼은 humanoid_bringup/launch/control_joypad.launch.py
                # 의 실측값과 동일. 8BitDo Ultimate 2 (XInput) 기준.
                #   axes    0=좌스틱LR 1=좌스틱UD 2=우스틱LR 3=우스틱UD
                #           6/7=D-pad
                #   buttons A=0 B=1 X=3 Y=4
                'axis_vx': 1,
                'axis_vy': 0,
                'axis_wz': 2,    # 우스틱 좌우. 3으로 두면 앞뒤로 회전한다
                'invert_vx': False,
                'invert_vy': False,
                'invert_wz': False,
                'deadzone_vy': 0.35,      # 대각선 새어듦 방지
                'axis_dpad_ud': 7,
                'axis_dpad_lr': 6,
                'button_kick': 0,         # A
                'button_estop': 1,        # B: 즉시 stop 모드
                'button_freq_reset': 4,   # Y
            }],
            output='screen',
        ),

        Node(
            package='humanoid_policy_runner',
            executable='actor_onnx_walking_runner',
            name='actor_onnx_walking_node',
            parameters=[{
                'policy_variant': policy_variant,
                'actor_path': actor_path,
                'policy_rate_hz': 50.0,
                'action_scale': 0.25,
                # ── 학습 cfg 와 반드시 일치시킬 것 ──
                # velocity_v1s_env_cfg.py 의 StoppableGaitPhaseCommandCfg:
                #   frequency_range=(1.8, 2.2), frequency_scale=2.0,
                #   stop_rate=2.0, standing_threshold=0.1
                'gait_frequency': gait_freq,
                'gait_frequency_scale': 2.0,
                'gait_stop_rate': 2.0,
                'standing_threshold': 0.1,
                'gait_mode': gait_mode,
                'stop_delay_s': 1.0,
                'grav_offset_x': grav_offset_x,
                'grav_offset_y': grav_offset_y,
                'startup_hold_seconds': 1.0,
                'resume_hold_seconds': 1.0,
                'enable_csv_log': True,
            }],
            output='screen',
        ),
    ])

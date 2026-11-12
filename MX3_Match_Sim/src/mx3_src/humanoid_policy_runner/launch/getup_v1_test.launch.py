# getup_v1_test.launch.py
#
# ============================================================================
#  get-up (누운 상태에서 일어나기) 정책 실기 테스트 launch
# ============================================================================
#  학습 태스크 : Unitree-MX3-Getup-V1-2  (getup_v1_2_env_cfg.py, 18 DOF)
#                ※ Unitree-MX3-Getup-V1 모델도 그대로 돌아간다.
#                  v1_2 는 v1 최고 판의 복사본에서 출발했고 관측·액션
#                  규격이 동일하다 (프레임 60 x 히스토리 5 = 300, 액션 18,
#                  scale 0.25, target_limits 동일). 바뀐 것은 전부 학습
#                  쪽(리셋 분포·리워드·마찰 DR)이다.
#                ※ Getup-V2 모델은 관측 330 이라 호환되지 않는다.
#                  getup_v2_test.launch.py 를 쓸 것.
#  구성        : actor_onnx_getup_runner 단 하나. 텔레옵도 twist_mux 도 없다.
#                기상 정책은 속도 명령을 받지 않는다(commands = None).
#
# ── ★ 2026-09-15 — 시작 자세 제한이 기본 해제됐다 ★ ──
#   v1_2 의 리셋 분포가 바뀌었다:
#       prone_ratio        = 0.5   env 절반이 **엎드린 자세**로 스폰
#       fall_angle_min_deg = 0.0   각도를 0~90 도 밴드에서 균일 추출
#       yaw_range          = ±π
#   즉 거의 선 상태부터 완전히 누운 상태까지, 앞뒤·전 방향이 모두 학습
#   분포 안이다. v1 이 걸던 게이트(tilt>=60 이고 gx<=-0.5)는 이제 학습
#   범위의 절반 이상을 막으므로 기본값을 none 으로 바꿨다.
#
#   그래서 아래 명령이 **현재 자세와 무관하게** 동작한다:
#       ros2 launch humanoid_policy_runner getup_v1_test.launch.py
#
#   앙와위 전용 정책(v1 원본)을 돌릴 때만 되살린다:
#       ... start_pose_check:=supine
#
#   ⚠ 게이트가 없으므로 오트리거가 곧 동작이다. 보행 중에 /getup_trigger
#     가 실수로 들어오면 로봇이 그 자리에서 웅크린다.
#   ⚠ 거의 선 상태에서 트리거하면 PREP(순수 위치 제어, 균형 제어 없음)
#     동안 팔이 벌어지며 넘어질 수 있다. 러너가 경고를 띄운다.
#     prep_duration_s 를 3~4 s 로 늘리거나 잡아 줄 것.
#
# ── 먼저 띄워야 하는 것 ──
#   ros2 launch humanoid_serial_bridge dynamixel_system.launch.py
#     → /joint_states(20 관절), /imu/data 를 발행하고 /joint_goals 를 받는다.
#       이게 없으면 러너가 tick 자체를 못 돈다.
#
# ── ★ 절대 같이 띄우면 안 되는 것 ★ ──
#   1) fall_manager_node
#      로봇이 누워 있으면 FALLEN 으로 판정해 standup 키프레임을 재생하고
#      /joint_goals 에 보호자세를 직접 쏜다. 이 노드와 매 틱 서로를 덮어써서
#      관절이 진동한다. 반드시 종료할 것.
#   2) actor_onnx_walking_runner / actor_onnx_walkkick_v1_runner
#      /joint_goals 발행자가 둘이 되면 즉시 발산한다.
#   3) humanoid_bringup 의 control*.launch.py (위 둘을 함께 올린다)
#
# ── 실행 ──
#   ros2 launch humanoid_policy_runner getup_v1_test.launch.py \
#       actor_path:=/home/robo02/models/getup_v1.onnx
#
#   # 첫 시험은 dry_run 을 권장한다. 관절을 움직이지 않고 관측/액션만
#   # CSV 로 남기므로, 관측이 정상 범위인지(중력벡터 gx≈-1 등) 먼저 확인할 수 있다.
#   ros2 launch humanoid_policy_runner getup_v1_test.launch.py \
#       actor_path:=... dry_run:=true
#
# ── 시험 절차 ──
#   1) 로봇을 매트 위에 둔다. **자세는 상관없다** — 등(supine), 엎드림
#      (prone), 옆으로 누움 모두 v1_2 학습 범위 안이다. 팔은 접힌 상태
#      그대로 두면 된다(PREP 이 ±1.5 로 맞춘다).
#   2) 토크가 켜져 있는지 확인한다(이 노드가 시작 시 자동으로 켠다).
#   3) 트리거:
#        ros2 service call /actor_onnx_getup_node/start std_srvs/srv/Trigger
#      → PREP 2 s (팔꿈치 3.04 → 1.5 로 이동) → POLICY 3 s → 종료
#      시작 시 터미널에 "시작 자세: SUPINE|PRONE|SIDE|UPRIGHT" 이 찍힌다.
#   4) 언제든 중단:
#        ros2 service call /actor_onnx_getup_node/abort std_srvs/srv/Trigger
#      → 그 순간 자세로 굳는다(관절이 더 움직이지 않는다).
#
# ── 확인용 토픽 ──
#   /getup_test/state          (String)        WAIT_READY|IDLE|PREP|POLICY|FINISH|ABORT
#   /getup_test/elapsed        (Float32)       현재 구간 경과 [s]
#   /getup_test/tilt_deg       (Float32)       수직 대비 기울기. 90→0 이면 일어난 것
#   /getup_test/debug/markers  (MarkerArray)   RViz/Foxglove 용 중력벡터 + 상태 텍스트
#   /getup_done                (Bool)          정책 구간 종료 시 True
#
#   ros2 topic echo /getup_test/tilt_deg
#     → PREP 동안 시작 자세를 유지, POLICY 동안 감소하면 일어나는 중이다.
#
#   CSV 의 start_pose 컬럼으로 그룹핑하면 prone 과 supine 의 성공률을
#   따로 볼 수 있다. 두 자세는 난이도가 다르므로 섞어서 보면 안 된다.
#
# ── 팔 관련 주의 (기존 러너와 가장 다른 점) ──
#   기상 정책은 팔을 실제로 구동한다. motor_gains.yaml 의 arms 그룹에
#   goal_pwm: 50 (약 5.6%) 이 걸려 있어 그대로는 팔이 안 움직인다.
#   → 이 노드가 시작 시 팔 6 관절을 100% 로 승격하고, 끝나면
#     /body_motor_controller/reload_gains 로 되돌린다.
#   → 노드가 비정상 종료됐다면 반드시 수동으로 복원할 것:
#       ros2 service call /body_motor_controller/reload_gains std_srvs/srv/Trigger
#
#   팔이 떨리면 motor_gains.yaml 의 arms 에 position_d_gain 을 넣어 볼 것
#   (다리는 P=800/D=500, 팔은 P=800/D=0 이다). 보행에도 영향이 가는 값이라
#   이 노드가 자동으로 바꾸지는 않는다.
# ============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    actor_path = LaunchConfiguration('actor_path')
    prep_duration_s = LaunchConfiguration('prep_duration_s')
    policy_duration_s = LaunchConfiguration('policy_duration_s')
    end_action = LaunchConfiguration('end_action')
    dry_run = LaunchConfiguration('dry_run')
    start_pose_check = LaunchConfiguration('start_pose_check')
    restore_goal_pwm = LaunchConfiguration('restore_goal_pwm_on_finish')
    max_target_delta = LaunchConfiguration('max_target_delta_from_current_rad')
    policy_profile_velocity_rpm = LaunchConfiguration('policy_profile_velocity_rpm')

    # 패키지 share 안의 기본 모델 경로.
    #   빈 문자열을 default 로 두면 러너에서 expanduser("") 가 홈 디렉터리로
    #   해석되어 "actor_path not found: /home/robo02" 로 죽는다.
    default_actor = os.path.join(
        get_package_share_directory('humanoid_policy_runner'),
        'models',
        'getup_v1_2_policy.onnx',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'actor_path', default_value=default_actor,
            description=(
                '기상 정책 ONNX 경로 (관측 300 = 프레임 60 x 히스토리 5, 액션 18). '
                'v1 / v1_2 공용이다 — 두 태스크의 관측·액션 규격이 같다. '
                'V2 모델(관측 330)은 여기 물리면 러너가 차원 검사에서 죽는다. '
                '튜닝 중에는 패키지 밖 경로를 주는 편이 편하다 — models/ 에 두면 '
                '모델을 바꿀 때마다 colcon build 를 다시 돌려야 한다'),
        ),
        DeclareLaunchArgument(
            'prep_duration_s', default_value='2.0',
            description=(
                '초기자세(학습 액션 원점)까지 이동하는 시간 [s]. '
                '팔꿈치가 약 3.04 → 1.5 로 크게 움직이므로 너무 짧으면 '
                '반동으로 몸통이 굴러간다'),
        ),
        DeclareLaunchArgument(
            'policy_duration_s', default_value='10.0',
            description=(
                '정책 추론 구간 [s]. ★안전장치★ 이 시간이 지나면 성공 여부와 '
                '무관하게 끊는다. sim 실측 기립 완료는 1.0~1.2 s'),
        ),
        DeclareLaunchArgument(
            'end_action', default_value='hold',
            description=(
                '정책 종료 후 동작. hold=마지막 목표 유지(기본, 가장 안전) / '
                'walk_ready=보행 시작 자세로 전이(팔꿈치 복귀) / none=발행 중단'),
        ),
        DeclareLaunchArgument(
            'dry_run', default_value='false',
            description=(
                'true 면 /joint_goals 를 발행하지 않고 관측·액션만 CSV 에 남긴다. '
                '첫 시험에서 관측이 정상 범위인지 확인할 때 쓴다'),
        ),
        DeclareLaunchArgument(
            'start_pose_check', default_value='none',
            description=(
                '시작 자세 게이트. ★기본 none = 현재 자세와 무관하게 시작★ '
                'v1_2 는 prone_ratio=0.5 / fall_angle_min_deg=0.0 이라 '
                '0~90도 · 앞뒤 양방향이 전부 학습 분포 안이므로 거를 근거가 없다. '
                'lying=누워 있을 때만(앞뒤 안 가림) / '
                'supine=등을 댄 자세만(v1 원본 앙와위 전용 정책에만 쓸 것)'),
        ),
        DeclareLaunchArgument(
            'max_target_delta_from_current_rad', default_value='10.0',
            description=(
                '현재 각도 기준 목표 변화량 상한 [rad]. 기본값은 사실상 비활성. '
                '첫 시험에서 팔이 튀는 것이 무서우면 0.5 로 조여서 확인만 할 것'),
        ),
        DeclareLaunchArgument(
            'policy_profile_velocity_rpm', default_value='-1.0',
            description=(
                'Profile Velocity [rpm]. -1=건드리지 않음(robot_init 의 45 유지). '
                'sim 실측 기상 관절속도 4.57 rad/s = 43.6 rpm 이라 45 는 거의 '
                '한계선이다. 여유가 필요하면 50 을 시도할 것. ★0 은 무제한 '
                '특수값이므로 절대 넣지 말 것★'),
        ),

        DeclareLaunchArgument(
            'restore_goal_pwm_on_finish', default_value='true',
            description=(
                '종료 시 reload_gains 로 팔 Goal PWM 상한(50)을 복원한다. '
                'false 로 두면 승격된 100% 가 유지되므로 레지스터 확인용으로 '
                '쓸 수 있으나, ★반드시 수동 복원할 것★ — 안 내리면 다음 '
                '낙상에서 감속기 보호가 사라진다. '
                'ros2 service call /body_motor_controller/reload_gains '
                'std_srvs/srv/Trigger'),
        ),

        Node(
            package='humanoid_policy_runner',
            executable='actor_onnx_getup_runner',
            name='actor_onnx_getup_node',
            parameters=[{
                'actor_path': actor_path,
                # ── 시퀀스 타이밍 ──
                #   ParameterValue 로 감싸지 않으면 LaunchConfiguration 이
                #   문자열로 전달되어 double 파라미터에서 타입 에러가 난다.
                'prep_duration_s': ParameterValue(
                    prep_duration_s, value_type=float),
                'prep_settle_s': 0.0,
                'policy_duration_s': ParameterValue(
                    policy_duration_s, value_type=float),
                # ── 학습 cfg 와 반드시 일치 ──
                #   getup_v1_2_env_cfg.py (v1 과 동일한 값):
                #     GetupJointPositionActionCfg(scale=0.25)
                #     base_ang_vel(scale=0.2) / joint_vel_rel(scale=0.05)
                'action_scale': 0.25,
                'clip_actions': 10.0,
                'base_ang_vel_scale': 0.2,
                'joint_vel_scale': 0.05,
                'policy_rate_hz': 50.0,
                # ── 안전 ──
                #   ★ 기본 none — 자세를 보지 않고 시작한다 ★
                #     근거는 이 파일 상단 주석과 러너 헤더 참조.
                'start_pose_check': start_pose_check,
                # lying/supine 모드에서만 쓰는 기울기 하한 [deg]
                'lying_tilt_min_deg': 60.0,
                # supine 모드에서만 쓴다.
                #   실측: 앞으로 엎어짐 gx=+0.90 / 뒤로 누움 gx=-0.96
                'supine_gx_max': -0.5,
                # 자세 분류(로그·CSV) 문턱. 이보다 덜 기울면 UPRIGHT.
                'upright_tilt_max_deg': 30.0,
                'max_target_delta_from_current_rad': ParameterValue(
                    max_target_delta, value_type=float),
                'dry_run': ParameterValue(dry_run, value_type=bool),
                # 자동 시작 비활성. 로봇을 눕히기 전에 팔이 움직이면 위험하다.
                'auto_start_delay_s': -1.0,
                # ── 마무리 ──
                'end_action': end_action,
                'end_transition_s': 1.0,
                # ── 하드웨어 준비/복원 ──
                'enable_torque_on_start': True,
                'raise_arm_goal_pwm': True,
                'restore_goal_pwm_on_finish': ParameterValue(
                    restore_goal_pwm, value_type=bool),
                'policy_profile_velocity_rpm': ParameterValue(
                    policy_profile_velocity_rpm, value_type=float),
                'restore_profile_velocity_rpm': 45.0,
                # ── 디버그 ──
                'debug_print_every': 25,
                'debug_frame': 'base_link',
                'publish_markers': True,
                'enable_csv_log': True,
                'csv_output_dir': '~/policy_logs',
            }],
            output='screen',
        ),
    ])

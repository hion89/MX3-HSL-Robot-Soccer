# getup_v2_test.launch.py
#
# ============================================================================
#  supine get-up V2 (보행 통합형 기상) 정책 실기 테스트 launch
# ============================================================================
#  학습 태스크 : Unitree-MX3-Getup-V2  (getup_v2_env_cfg.py, 18 DOF)
#  구성        : actor_onnx_getup_v2_runner 단 하나.
#
#  ★ V1 launch(getup_v1_test.launch.py) 와 모델 호환이 없다 ★
#    V1 : 관측 300 (프레임 60 x 5). velocity_commands / gait 관측 없음.
#    V2 : 관측 330 (프레임 66 x 5). velocity_commands(3) + gait_phase(3) 추가.
#    잘못 물리면 러너가 ONNX 로드 직후 차원 검사에서 죽는다(조용히 오동작
#    하지는 않는다).
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
#   3) actor_onnx_getup_runner (V1)
#   4) humanoid_bringup 의 control*.launch.py (위 노드들을 함께 올린다)
#
# ── 실행 ──
#   ros2 launch humanoid_policy_runner getup_v2_test.launch.py \
#       actor_path:=/home/robo02/models/getup_v2.onnx
#
#   # 첫 시험은 dry_run 을 권장한다. 관절을 움직이지 않고 관측/액션만
#   # CSV 로 남기므로, 관측이 정상 범위인지 먼저 확인할 수 있다.
#   #   - 중력벡터 gx ≈ -1 (등을 댄 자세)
#   #   - gait_obs_cos/sin/freq 가 의도한 모드대로 나오는지  ← ★V2 신규★
#   #   - cmd_used_* 가 0 인지
#   ros2 launch humanoid_policy_runner getup_v2_test.launch.py \
#       actor_path:=... dry_run:=true
#
# ── 시험 절차 ──
#   1) 로봇을 등을 대고(supine) 매트 위에 눕힌다. 팔은 접힌 상태 그대로 둔다.
#   2) 토크가 켜져 있는지 확인한다(이 노드가 시작 시 자동으로 켠다).
#   3) 트리거:
#        ros2 service call /actor_onnx_getup_v2_node/start std_srvs/srv/Trigger
#      → PREP 2 s (팔꿈치 3.04 → 1.5 로 이동) → POLICY 10 s → 종료
#   4) 언제든 중단:
#        ros2 service call /actor_onnx_getup_v2_node/abort std_srvs/srv/Trigger
#      → 그 순간 자세로 굳는다(관절이 더 움직이지 않는다).
#   5) /joint_goals 점유 해제 (노드는 살려 둔 채):
#        ros2 service call /actor_onnx_getup_v2_node/reset std_srvs/srv/Trigger
#
# ── ★ V2 신규: gait 위상 ★ ──
#   V2 관측에는 보행 위상 (cos, sin, freq/2.0) 3 차원이 들어간다. 학습
#   RobotPlayEnvCfg 조건(전 env 속도 0)에서 gait 는 두 갈래로 나뉜다:
#       완전 정지  60 %  (rel_stop_when_standing=0.6)  → gait_mode:=stop
#       제자리걷기 40 %                                → gait_mode:=march
#   둘 다 학습 분포 안이다. 기본값은 stop 이다 — 관측이 (1,0,0) 상수라
#   가장 조용하고, 기상 동작만 분리해서 보기 좋다.
#
#   실행 중 전환:
#     ros2 topic pub --once /getup_v2/gait_mode std_msgs/String "{data: march}"
#     ros2 topic pub --once /getup_v2/gait_frequency std_msgs/Float32 "{data: 2.0}"
#
#   ※ 학습 frequency_range 는 (1.7, 2.4) Hz 다. 이 밖의 값을 주면 러너가
#     경고를 띄운다(막지는 않는다).
#
# ── ★ V2 신규: 속도 명령 ★ ──
#   기본은 cmd_vel_source:=zero — 항상 (0,0,0). 학습 Play 조건과 동일하다.
#   일어선 뒤 같은 정책으로 보행까지 이어서 보려면:
#     ros2 launch humanoid_policy_runner getup_v2_test.launch.py \
#         actor_path:=... cmd_vel_source:=subscribe gait_mode:=auto
#   이때 tilt 가 cmd_vel_require_upright_deg(기본 25도) 아래로 내려가기
#   전까지는 속도 명령이 0 으로 강제된다(누운 채 전진 명령 방지).
#
# ── 확인용 토픽 ──
#   /getup_v2/state                     (String)      WAIT_READY|IDLE|PREP|
#                                                     POLICY|FINISH|ABORT
#   /getup_v2/elapsed                   (Float32)     현재 구간 경과 [s]
#   /getup_v2/tilt_deg                  (Float32)     수직 대비 기울기.
#                                                     90→0 이면 일어난 것
#   /getup_v2/gait/phase                (Float32)     위상 0~1  ★V2 신규★
#   /getup_v2/gait/effective_frequency  (Float32)     실제 적용 주파수 [Hz]
#   /getup_v2/gait/state                (String)      WALKING|MARCHING|
#                                                     STOPPING|STOPPED
#   /getup_v2/debug/markers             (MarkerArray) RViz/Foxglove 용
#                                                     중력벡터 + 상태 텍스트
#                                                     + gait 위상 다이얼
#   /getup_done                         (Bool)        정책 구간 종료 시 True
#
#   ros2 topic echo /getup_v2/tilt_deg
#     → PREP 동안 90 근처 유지, POLICY 동안 감소하면 일어나고 있는 것이다.
#   ros2 topic echo /getup_v2/gait/phase
#     → stop 모드면 0 에 붙어 있고, march 모드면 톱니파를 그린다.
#
# ── RViz 확인 ──
#   Fixed Frame 을 base_link 로 두고 MarkerArray 로
#   /getup_v2/debug/markers 를 추가하면
#     - 중력 화살표: 몸이 어느 쪽으로 누워 있는지
#     - 상태 텍스트: 구간/경과/tilt/gait/cmd/추론시간
#     - 위상 다이얼: 원 위를 도는 구. 주황=보행, 노랑=정지 수렴, 회색=정지
#   가 한 화면에 나온다. gait 가 V2 신규 관측이므로 **여기부터 확인**할 것.
#
# ── 팔 관련 주의 (V1 과 동일) ──
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
    require_supine = LaunchConfiguration('require_supine')
    restore_goal_pwm = LaunchConfiguration('restore_goal_pwm_on_finish')
    max_target_delta = LaunchConfiguration('max_target_delta_from_current_rad')
    policy_profile_velocity_rpm = LaunchConfiguration('policy_profile_velocity_rpm')
    gait_mode = LaunchConfiguration('gait_mode')
    gait_frequency = LaunchConfiguration('gait_frequency')
    cmd_vel_source = LaunchConfiguration('cmd_vel_source')

    # 패키지 share 안의 기본 모델 경로.
    #   빈 문자열을 default 로 두면 러너에서 expanduser("") 가 홈 디렉터리로
    #   해석되어 "actor_path not found: /home/robo02" 로 죽는다.
    default_actor = os.path.join(
        get_package_share_directory('humanoid_policy_runner'),
        'models',
        'getup_v2_policy.onnx',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'actor_path', default_value=default_actor,
            description=(
                '기상 V2 정책 ONNX 경로 (관측 330 = 프레임 66 x 히스토리 5, 액션 18). '
                '튜닝 중에는 패키지 밖 경로를 주는 편이 편하다 — models/ 에 두면 '
                '모델을 바꿀 때마다 colcon build 를 다시 돌려야 한다'),
        ),
        DeclareLaunchArgument(
            'prep_duration_s', default_value='2.0',
            description=(
                '초기자세(학습 액션 원점)까지 이동하는 시간 [s]. '
                '팔꿈치가 약 3.04 → 1.5 로 크게 움직이므로 너무 짧으면 '
                '반동으로 몸통이 굴러간다. 이 구간은 학습 reset_hold 에 '
                '대응하므로 속도 명령 0 / gait 주파수 0 이 유지된다'),
        ),
        DeclareLaunchArgument(
            'policy_duration_s', default_value='10.0',
            description=(
                '정책 추론 구간 [s]. ★안전장치★ 이 시간이 지나면 성공 여부와 '
                '무관하게 끊는다. sim 실측 90도 기립 완료는 약 1 s 이고, V2 는 '
                '기립 후 서 있기까지 같은 정책이 담당하므로 10 s 를 기본값으로 '
                '둔다. 0 이하면 무제한(★안전장치 없음★)'),
        ),
        DeclareLaunchArgument(
            'gait_mode', default_value='stop',
            description=(
                '★V2 신규★ gait 위상 모드. '
                'stop=완전 정지(기본, 관측 (1,0,0) 상수. 학습 Play 조건의 60%) / '
                'march=제자리 걷기(주파수 유지. 학습 Play 조건의 40%) / '
                'walk=항상 주파수 유지 / '
                'auto=속도 0 이 gait_stop_delay_s 지속되면 정지'),
        ),
        DeclareLaunchArgument(
            'gait_frequency', default_value='2.0',
            description=(
                '★V2 신규★ gait 주파수 [Hz]. 학습은 env 별 랜덤 (1.7~2.4) 이라 '
                '실기에서는 하나를 골라야 한다. 범위 밖 값은 분포 외이므로 '
                '러너가 경고를 띄운다. gait_mode=stop 이면 쓰이지 않는다'),
        ),
        DeclareLaunchArgument(
            'cmd_vel_source', default_value='zero',
            description=(
                '★V2 신규★ velocity_commands 관측의 출처. '
                'zero=항상 (0,0,0) (기본. 학습 RobotPlayEnvCfg 와 동일 조건) / '
                'subscribe=/cmd_vel 구독. 기립 후 보행까지 이어서 볼 때만 쓴다'),
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
                '첫 시험에서 관측(특히 V2 신규 gait 컬럼)이 정상인지 확인할 때 쓴다'),
        ),
        DeclareLaunchArgument(
            'require_supine', default_value='true',
            description=(
                '누운 자세일 때만 시작을 허용한다(tilt>60deg 이고 gx<-0.5). '
                'V2 는 서 있는 상태도 학습 분포 안이지만(보행 통합), 이 노드는 '
                '기상 시험용이라 기본은 막는다. 보행만 확인하려면 false'),
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
            executable='actor_onnx_getup_v2_runner',
            name='actor_onnx_getup_v2_node',
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
                #   getup_v2_env_cfg.py:
                #     GetupJointPositionActionCfg(scale=0.25)
                #     base_ang_vel(scale=0.2) / joint_vel_rel(scale=0.05)
                'action_scale': 0.25,
                'clip_actions': 10.0,
                'base_ang_vel_scale': 0.2,
                'joint_vel_scale': 0.05,
                # ★ 표시용이 아니다 ★ gait 위상 적분 dt 로 쓰인다.
                #   학습 env.step_dt = decimation(4) * sim.dt(0.005) = 0.02 s.
                'policy_rate_hz': 50.0,
                # ── ★V2 신규★ gait (htwk_gait_v01.PhaseOnlyGaitCommandCfg) ──
                'gait_mode': gait_mode,
                'gait_frequency': ParameterValue(
                    gait_frequency, value_type=float),
                'gait_frequency_scale': 2.0,     # 학습 frequency_scale
                'gait_stop_rate': 2.0,           # 학습 stop_rate [phase/s]
                'gait_standing_threshold': 0.1,  # 학습 standing_threshold
                'gait_stop_delay_s': 0.5,        # auto 모드 전용(실기 값)
                # 학습 freeze_during_reset_hold=True 와 대응
                'gait_freeze_during_prep': True,
                # ── ★V2 신규★ velocity_commands ──
                'cmd_vel_source': cmd_vel_source,
                'cmd_vel_topic': '/cmd_vel',
                'cmd_vel_timeout_s': 0.5,
                # 학습 limit_ranges: x(-0.5,0.7) y(-0.5,0.5) wz(-1.5,1.5)
                'cmd_vel_limits': [-0.5, 0.7, -0.5, 0.5, -1.5, 1.5],
                # 이 각도보다 기울어져 있으면 속도 명령을 0 으로 강제
                'cmd_vel_require_upright_deg': 25.0,
                # ── 안전 ──
                'require_supine': ParameterValue(
                    require_supine, value_type=bool),
                'supine_tilt_min_deg': 60.0,
                # 실측: 앞으로 엎어짐 gx=+0.90 / 뒤로 누움 gx=-0.96
                'supine_gx_max': -0.5,
                'max_target_delta_from_current_rad': ParameterValue(
                    max_target_delta, value_type=float),
                'enable_target_clamp': True,
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

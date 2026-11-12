from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'humanoid_policy_runner'

# ============================================================================
#  humanoid_policy_runner — 정책 러너 / 텔레옵 노드 패키지
# ----------------------------------------------------------------------------
#  [2026-08-18 정리] 구버전 러너 8개 삭제, 사용 중인 5개만 유지.
#
#   삭제 근거: 워크스페이스 전체를 훑어 launch 의 executable= 참조와
#   모듈 간 import 를 교차 확인한 결과, 8개 모두 실제 참조가 0 이었다
#   (다른 파일에 이름이 나오는 것은 전부 주석이었다).
#
#     삭제됨                                     대체
#     ─────────────────────────────────────────  ──────────────────────────────
#     actor_onnx_runner                          actor_onnx_walking_runner
#     actor_onnx_residual_runner                 actor_onnx_walking_runner
#     actor_onnx_walkkick_runner                 actor_onnx_walkkick_v1_runner
#     actor_onnx_kick_runner                     actor_onnx_walkkick_v1_runner
#     actor_onnx_residual_kick_power_runner      actor_onnx_walkkick_v1_runner
#     actor_onnx_walk_residual_kick_runner       walking + walkkick_v1 핸드오프
#     joypad_teleop_node (구버전)                joypad_teleop_node (신규, 개명됨)
#     trajectory_player_node                     (사용처 없음)
#
#  [같은 날 개명]
#     actor_onnx_v1s_runner → actor_onnx_walking_runner
#     actor_onnx_v1s_node   → actor_onnx_walking_node
#     이 러너는 v0_1/v0_2/v1/v1s 를 policy_variant 로 모두 지원하는데
#     이름이 특정 변형(v1s)을 가리켜 오해를 샀다. control.launch.py 의
#     policy_variant 기본값은 실제로 v0_2 다. 역할(보행)로 이름을 바꿨다.
#
#     joypad_teleop_v1s_node → joypad_teleop_node
#     구 joypad_teleop_node 와 공존시키려 붙였던 v1s 접미사가, 구 노드를
#     삭제하면서 무의미해졌다. keyboard_teleop_node 와 대칭도 맞췄다.
#
#  [명명 규칙]
#     _runner : ONNX 정책을 로드해 추론하는 노드 (2개)
#     _node   : 그 외 일반 노드 (3개)
#
#  ※ 러너를 추가·삭제할 때는 아래 세 곳을 함께 확인할 것
#     1) 이 파일의 console_scripts
#     2) 해당 .py 파일
#     3) 그 러너를 실행/참조하는 launch 파일의 executable= 와 *_actor_path 인자
#        (러너만 지우고 launch 인자를 남기면 존재하지 않는 모델 경로를
#         기본값으로 들고 있게 된다)
# ============================================================================

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 파일
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*')),
        # ONNX 정책 파일
        #   ※ --symlink-install 로 빌드하면 러너가 src 의 models/ 를 직접
        #     읽으므로, 모델 파일만 교체할 때는 재빌드가 필요 없다.
        (os.path.join('share', package_name, 'models'),
         glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robo1',
    maintainer_email='robo1@todo.todo',
    description='RL policy runner nodes (ONNX) for KHUBER humanoid',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ── 정책 러너 ────────────────────────────────────────────────
            # [보행] policy_variant 로 v0_1/v0_2/v1/v1s 전환. 관측 240.
            #   노드명 : actor_onnx_walking_node
            #   사용처 : control.launch.py, v1s_joypad_test.launch.py
            'actor_onnx_walking_runner = humanoid_policy_runner.actor_onnx_walking_runner:main',  # noqa: E501

            # [보행+킥] 접근 보행까지 포함한 독립 정책. 관측 265 (53 x 5).
            #   노드명 : actor_onnx_walkkick_v1_node
            #   사용처 : control_joypad.launch.py (A 버튼 킥)
            #   ※ walking 러너와 /joint_goals 를 공유하므로, 반드시
            #     /kick_trigger·/kick_done 으로 소유권을 주고받아야 한다.
            #     두 노드가 동시에 발행하면 매 틱 서로를 덮어써 발산한다.
            'actor_onnx_walkkick_v1_runner = humanoid_policy_runner.actor_onnx_walkkick_v1_runner:main',  # noqa: E501

            # [기상] supine get-up 실기 테스트 러너. 관측 300 (60 x 5), 액션 18.
            #   노드명 : actor_onnx_getup_node
            #   사용처 : getup_v1_test.launch.py
            #   ※ 이 러너만 **팔 6 관절을 실제로 구동**한다. 다른 러너는 다리
            #     12 개만 명령한다. 그래서 시작 시 팔 Goal PWM 을 100% 로
            #     승격했다가 끝나면 reload_gains 로 되돌린다
            #     (motor_gains.yaml 의 arms.goal_pwm 이 50 = 약 5.6% 라
            #      그대로는 팔이 목표까지 이동하지 못한다).
            #   ※ 정해진 시간(PREP 2s + POLICY 3s)만 돌고 스스로 멈추는
            #     벤치 테스트 노드다. 상시 노드가 아니다.
            #   ※ fall_manager_node / 보행 러너와 동시에 띄우면 /joint_goals
            #     를 두고 싸운다. 반드시 단독으로 실행할 것.
            'actor_onnx_getup_runner = humanoid_policy_runner.actor_onnx_getup_runner:main',  # noqa: E501

            # [기상 V2] 보행 통합형 기상. 관측 330 (66 x 5), 액션 18.
            #   노드명 : actor_onnx_getup_v2_node
            #   사용처 : getup_v2_test.launch.py
            #   ※ V1 과 모델 호환이 없다. V2 는 보행 env(velocity_v0_6)를
            #     베이스로 기상을 얹은 통합 환경이라, 관측에
            #     velocity_commands(3) + gait_phase(3) 가 추가됐다.
            #       V1 프레임 60 -> 관측 300
            #       V2 프레임 66 -> 관측 330
            #     잘못 물리면 러너가 ONNX 로드 직후 차원 검사에서 죽는다.
            #   ※ gait 위상은 실기에서 러너가 직접 적분한다
            #     (htwk_gait_v01.PhaseOnlyGaitCommand 이식).
            #     gait_mode: stop(기본) / march / walk / auto
            #   ※ V1 러너와 마찬가지로 **팔 6 관절을 실제로 구동**하므로
            #     시작 시 팔 Goal PWM 을 100% 로 승격했다가 끝나면
            #     reload_gains 로 되돌린다.
            #   ※ fall_manager_node / 다른 러너와 동시에 띄우면 /joint_goals
            #     를 두고 싸운다. 반드시 단독으로 실행할 것.
            'actor_onnx_getup_v2_runner = humanoid_policy_runner.actor_onnx_getup_v2_runner:main',  # noqa: E501

            # ── 명령 정형 ────────────────────────────────────────────────
            # /cmd_vel_raw -> (vy bias 보정) -> /cmd_vel
            #   사용처 : control.launch.py
            'lateral_drift_correction_node = humanoid_policy_runner.lateral_drift_correction_node:main',  # noqa: E501

            # ── 텔레옵 ───────────────────────────────────────────────────
            # [키보드] 사용처 : control.launch.py(use_teleop), control_teleop.launch.py
            'keyboard_teleop_node = humanoid_policy_runner.keyboard_teleop_node:main',

            # [조이패드] gait 모드/주파수 제어 포함.
            #   사용처 : control_joypad.launch.py, v1s_joypad_test.launch.py
            'joypad_teleop_node = humanoid_policy_runner.joypad_teleop_node:main',
        ],
    },
)

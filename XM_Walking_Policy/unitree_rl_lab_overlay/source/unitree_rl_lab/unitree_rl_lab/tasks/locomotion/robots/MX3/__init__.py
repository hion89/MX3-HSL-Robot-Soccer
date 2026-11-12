# unitree_rl_lab/tasks/locomotion/robots/MX3/__init__.py
# 원본에서 이 패키지의 태스크 등록 블록만 그대로 발췌했다 (원본 510~562 행).
import gymnasium as gym

# ✅ v0_6 (v0_5 + 가변 보행 주파수 1.7~2.4 Hz)
#    v0_5 대비 변경점은 gait.frequency_range=(1.7,2.4) 한 줄뿐이다.
#    (v0_5 는 None = 2.0 Hz 고정)
#    속도와 결합하지 않는 독립 샘플이라 주파수가 별도 제어 입력이 된다.
#    gait_phase 가 이미 3 차원(cos,sin,freq)이라 관측 240/255 불변 →
#    실기 러너 수정 불필요, v0_5 체크포인트 resume 가능, _MIRROR_SPEC 불변.
#    상세는 velocity_v0_6_env_cfg.py 헤더.
gym.register(
    id="Unitree-MX3-Velocity-V0-6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.robots.MX3.velocity_v0_6_env_cfg:RobotEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.robots.MX3.velocity_v0_6_env_cfg:RobotPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_sym_cfg:SymmetryPPOMirror3RunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-MX3-Velocity-V0-6-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.robots.MX3.velocity_v0_6_env_cfg:RobotPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_sym_cfg:SymmetryPPOMirror3RunnerCfg"
        ),
    },
)

# ✅ v0_6 + 대칭 "지표만" 기록 (학습 반영 없음)
gym.register(
    id="Unitree-MX3-Velocity-V0-6-Metric",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.robots.MX3.velocity_v0_6_env_cfg:RobotEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.robots.MX3.velocity_v0_6_env_cfg:RobotPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents."
            "rsl_rl_ppo_sym_cfg:SymmetryMetricOnlyPPORunnerCfg"
        ),
    },
)

# unitree_rl_lab/tasks/locomotion/robots/MX3/__init__.py
# 원본에서 이 패키지의 태스크 등록 블록만 그대로 발췌했다 (원본 1209~1261 행).
# V1-5 는 V1-5B 의 부모 설정 (킥 세기 없음) 이라 일부 스크립트 기본 태스크로 남겨 둔다.
import gymnasium as gym


# ── V1-5: 경기 넘겨받기 조건 (2026-10-31) ─────────────────────────────
# W90B70 + convexhull·팔 내림·Profile Velocity·넘겨받기 상태 시작·공 믿음 오차·구르는 공.
# experiment 자동 유도 → unitree_mx3_walkkick_v1_5. W90B70 model_18200 에서 resume.
gym.register(
    id="Unitree-MX3-WalkKick-V1-5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.robots.MX3.walk_kick_v1_5_env:RobotV15EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.robots.MX3.walk_kick_v1_5_env:RobotV15PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_sym_cfg:WalkKickPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-MX3-WalkKick-V1-5-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.robots.MX3.walk_kick_v1_5_env:RobotV15PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_sym_cfg:WalkKickPPORunnerCfg"
        ),
    },
)


# ── V1-5B: v1_5 + 킥 세기 (2026-11-01). experiment → unitree_mx3_walkkick_v1_5b ──
gym.register(
    id="Unitree-MX3-WalkKick-V1-5B",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.MX3.walk_kick_v1_5_env:RobotV15BEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.MX3.walk_kick_v1_5_env:RobotV15BPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_sym_cfg:WalkKickPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-MX3-WalkKick-V1-5B-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.MX3.walk_kick_v1_5_env:RobotV15BPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_sym_cfg:WalkKickPPORunnerCfg",
    },
)

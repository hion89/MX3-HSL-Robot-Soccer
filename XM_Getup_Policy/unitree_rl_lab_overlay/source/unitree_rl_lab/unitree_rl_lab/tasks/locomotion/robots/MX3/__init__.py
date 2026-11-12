# unitree_rl_lab/tasks/locomotion/robots/MX3/__init__.py
# 원본에서 이 패키지의 태스크 등록 블록만 그대로 발췌했다 (원본 143~162 행).
import gymnasium as gym

# ── Getup-V1-2 : v1 최고 판(OK_16-27-54_smooth_stage3)에서 출발한 수정본 ──
#   V3 중지 후 이쪽으로 돌아왔다. 상세는 getup_v1_2_env_cfg.py 헤더 참조.
gym.register(
    id="Unitree-MX3-Getup-V1-2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.MX3.getup_v1_2_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.MX3.getup_v1_2_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GetupV12PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-MX3-Getup-V1-2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.MX3.getup_v1_2_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GetupV12PPORunnerCfg",
    },
)

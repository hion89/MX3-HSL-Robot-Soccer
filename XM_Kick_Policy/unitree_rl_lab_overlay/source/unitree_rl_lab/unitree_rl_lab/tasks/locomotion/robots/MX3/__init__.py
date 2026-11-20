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


# ── V1-6: 디딤발 강건화 (2026-11-16). v1_5b 22300 에서 resume. 설명: walk_kick_v1_6_env.py 머리말 ──
#   A = 구동기 (DC 곡선·전압·β 커리큘럼) / B = A + 관측 지연 / C = 전체 (+ 바닥·러너 계약·디딤발 yaw 벌점)
_V16 = "unitree_rl_lab.tasks.locomotion.robots.MX3.walk_kick_v1_6_env"
_V16_AGENT = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_sym_cfg:WalkKickPPORunnerCfg"
for _sfx, _train in (("A", "RobotV16AEnvCfg"), ("B", "RobotV16BEnvCfg"), ("C", "RobotV16CEnvCfg")):
    gym.register(
        id=f"Unitree-MX3-WalkKick-V1-6{_sfx}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{_V16}:{_train}",
            "play_env_cfg_entry_point": f"{_V16}:RobotV16PlayEnvCfg",
            "rsl_rl_cfg_entry_point": _V16_AGENT,
        },
    )
for _sfx, _play in (("", "RobotV16PlayEnvCfg"), ("-Turf", "RobotV16PlayTurfEnvCfg"),
                    ("-IMU20", "RobotV16PlayIMU20EnvCfg"), ("-Volt106", "RobotV16PlayVolt106EnvCfg"),
                    ("-RealLike", "RobotV16PlayRealLikeEnvCfg"),
                    ("-Diag-ActFlat", "RobotV16DiagActFlatEnvCfg"), ("-Diag-Act", "RobotV16DiagActEnvCfg"),
                    ("-Diag-Obs", "RobotV16DiagObsEnvCfg"), ("-Diag-Surf", "RobotV16DiagSurfEnvCfg"),
                    ("-Diag-Ctr", "RobotV16DiagCtrEnvCfg")):
    gym.register(
        id=f"Unitree-MX3-WalkKick-V1-6-Play{_sfx}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{_V16}:{_play}", "rsl_rl_cfg_entry_point": _V16_AGENT},
    )

# V1-6D (2026-11-19): C 에서 Explicit DC 모터를 뺀 학습 태스크 (22300 과 같은 ImplicitActuator)
gym.register(
    id="Unitree-MX3-WalkKick-V1-6D",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_V16}:RobotV16DEnvCfg",
        "play_env_cfg_entry_point": f"{_V16}:RobotV16DPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _V16_AGENT,
    },
)
for _sfx, _play in (("", "RobotV16DPlayEnvCfg"), ("-RealLike", "RobotV16DPlayRealLikeEnvCfg")):
    gym.register(
        id=f"Unitree-MX3-WalkKick-V1-6D-Play{_sfx}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{_V16}:{_play}", "rsl_rl_cfg_entry_point": _V16_AGENT},
    )

# V1-6E (2026-11-19): D + 안 차기 벌점 (−8, 헛발질·다툼 제외) + 디딤발 yaw −0.02. 재생·평가는 V1-6D-Play / -RealLike 그대로 쓴다.
gym.register(
    id="Unitree-MX3-WalkKick-V1-6E",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_V16}:RobotV16EEnvCfg",
        "play_env_cfg_entry_point": f"{_V16}:RobotV16EPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _V16_AGENT,
    },
)

# V1-6F (2026-11-19): 보상은 V1-5B 와 같고 환경만 (관측 지연 + 바닥 + hold·필터). 재생·평가는 V1-6D-Play / -RealLike.
gym.register(
    id="Unitree-MX3-WalkKick-V1-6F",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_V16}:RobotV16FEnvCfg",
        "play_env_cfg_entry_point": f"{_V16}:RobotV16DPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _V16_AGENT,
    },
)

# V1-6G (2026-11-19): F + 안 차고 끝남 (데드라인·넘어짐 무관) −4. F model_23500 에서 이어 학습. 재생·평가는 V1-6D-Play / -RealLike.
gym.register(
    id="Unitree-MX3-WalkKick-V1-6G",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_V16}:RobotV16GEnvCfg",
        "play_env_cfg_entry_point": f"{_V16}:RobotV16DPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _V16_AGENT,
    },
)

# V1-6H (2026-11-20): F + 차기 전 플러스 보상 게이트 (H: α1·데드라인, H2: α1·1.5 s, H3: α0.5·1.5 s). 재생·평가는 V1-6D-Play / -RealLike.
for _h in ("H", "H2", "H3"):
    gym.register(
        id=f"Unitree-MX3-WalkKick-V1-6{_h}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{_V16}:RobotV16{_h}EnvCfg",
            "play_env_cfg_entry_point": f"{_V16}:RobotV16DPlayEnvCfg",
            "rsl_rl_cfg_entry_point": _V16_AGENT,
        },
    )

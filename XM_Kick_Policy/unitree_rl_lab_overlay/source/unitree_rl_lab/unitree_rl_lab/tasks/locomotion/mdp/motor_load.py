# motor_load.py
# ============================================================================
# 모터 부하 한도 벌점 + 지형 기준 높이 종료 (rs_bot RS01 V0-2 도입, 로봇 무관)
#
# 배경 (2026-10-24, RS01 V0-1 model_24000 진단 — diag_ankle.py / diag_phase.py)
#   V0-1 은 무게중심을 발목보다 5~7 cm 앞에 두고 발끝으로 받치며 걷는다(발 들기 2→4~5 cm).
#   걸음새는 사람에 가깝고 안정적이지만, 디딤 중 발목 토크가 평균 3.6~5.6 N·m 로
#   RS00 정격 5 N·m 를 넘었다 (RMS 4.5~6.1, p95 10~13.5 / 피크 14).
#   보상에 관절 토크 항이 없어 정책이 RS00(정격 5) 과 RS06(정격 11) 을 구분하지 못한 탓이다.
#
# 설계 — 걸음새를 지우지 않고 "한도를 넘는 부분만" 벌한다
#   motor_torque_excess      순간 |τ| 가 peak_soft 를 넘은 양의 제곱합  (피크 포화 방지)
#   motor_torque_rms_excess  √EMA(τ²) (시정수 ema_tau_s) 가 rms_soft 를 넘은 양의 제곱합  (발열 방지)
#   한도 안에서는 0 이라, 발목을 쓰는 걸음새 자체는 그대로 둔다.
#   모터 그룹(RS00 / RS06)마다 항을 따로 두고 한도를 정격 기준으로 준다.
#   ※ EMA 상태는 env 에 joint 그룹 이름으로 저장하고, 에피소드 첫 스텝에 0 으로 리셋한다.
#
# root_height_below_minimum_terrain
#   터레인이 있으면 절대 높이 종료(root_height_below_minimum)가 경사 꼭대기·구덩이에서 틀어진다.
#   몸통 아래 레이캐스트(height_scanner) 평균 지면 높이 기준 상대 높이로 판정한다.
# ============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["motor_torque_excess", "motor_torque_rms_excess", "root_height_below_minimum_terrain",
           "base_height_above_terrain"]


def motor_torque_excess(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, peak_soft: float) -> torch.Tensor:
    """Σ relu(|τ| − peak_soft)²  [N·m²]. 음수 weight 로 쓴다."""
    tau = env.scene[asset_cfg.name].data.applied_torque[:, asset_cfg.joint_ids]
    return torch.relu(tau.abs() - peak_soft).pow(2).sum(dim=1)


def motor_torque_rms_excess(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, rms_soft: float,
                            ema_tau_s: float = 3.0, group: str = "default") -> torch.Tensor:
    """Σ relu(√EMA(τ²) − rms_soft)²  [N·m²]. EMA 시정수 ema_tau_s 초. 음수 weight 로 쓴다."""
    tau = env.scene[asset_cfg.name].data.applied_torque[:, asset_cfg.joint_ids]
    key = f"_mtl_ema_{group}"
    ema = getattr(env, key, None)
    if ema is None or ema.shape != tau.shape:
        ema = torch.zeros_like(tau)
        setattr(env, key, ema)
    alpha = min(env.step_dt / ema_tau_s, 1.0)
    ema.mul_(1.0 - alpha).add_(alpha * tau.pow(2))
    ema[env.episode_length_buf <= 1] = 0.0
    return torch.relu(ema.sqrt() - rms_soft).pow(2).sum(dim=1)


def _terrain_height(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w[..., 2]
    hits = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0)
    return hits.mean(dim=1)


def base_height_above_terrain(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """몸통 루트의 지형 기준 높이 [m] (진단·평가용)."""
    return env.scene[asset_cfg.name].data.root_pos_w[:, 2] - _terrain_height(env, sensor_cfg)


def root_height_below_minimum_terrain(env: ManagerBasedRLEnv, minimum_height: float, sensor_cfg: SceneEntityCfg,
                                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """지형 기준 몸통 높이가 minimum_height 미만이면 종료."""
    return base_height_above_terrain(env, sensor_cfg, asset_cfg) < minimum_height


# ============================================================================
# 경로 길이 기준 터레인 커리큘럼 (RS01 V0-2 2 차 도입)
#
#   IsaacLab terrain_levels_vel 은 에피소드 끝에 "출발점에서의 직선거리" 를 명령속도×에피소드길이×0.5 와
#   비교한다. 회전 명령(±1.5 rad/s)이 무작위로 섞이고 10 s 마다 명령이 바뀌는 환경에서는 잘 걸어도
#   원을 돌거나 되돌아와 직선거리가 짧아 강등만 된다 (V0-2 1 차: 완주 99 % 인데 level 2.5 → 0.8).
#
#   여기서는 실제로 걸은 경로 길이 ∫|v_xy| dt 를 명령 경로 길이 ∫|cmd_xy| dt 와 비교한다.
#     승급: 넘어지지 않음 & 명령 경로 ≥ min_cmd_path & 걸은 경로 ≥ up_ratio × 명령 경로
#     강등: 넘어짐(시간초과가 아닌 종료)  또는  (명령 경로 ≥ min_cmd_path & 걸은 경로 < down_ratio × 명령 경로)
#   정지 명령 env 는 명령 경로가 짧아 넘어지지 않는 한 그대로다.
#
#   walk_path_tracker (interval, 매 스텝 전역) 가 누적하고, terrain_levels_path (curriculum, 리셋 시) 가
#   판정 후 해당 env 의 누적값을 0 으로 돌린다. (_reset_idx 에서 커리큘럼이 씬 리셋보다 먼저 불린다)
# ============================================================================
__all__ += ["walk_path_tracker", "terrain_levels_path"]


def _path_buffers(env):
    if not hasattr(env, "_walk_path"):
        env._walk_path = torch.zeros(env.num_envs, device=env.device)
        env._cmd_path = torch.zeros(env.num_envs, device=env.device)
    return env._walk_path, env._cmd_path


def walk_path_tracker(env: ManagerBasedRLEnv, env_ids: torch.Tensor,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "base_velocity"):
    """매 스텝 걸은 경로와 명령 경로를 누적 (interval 전역 이벤트로 매 스텝 호출)."""
    walk, cmdp = _path_buffers(env)
    v = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2].norm(dim=1)
    c = env.command_manager.get_command(command_name)[:, :2].norm(dim=1)
    walk.add_(v * env.step_dt)
    cmdp.add_(c * env.step_dt)


def terrain_levels_path(env: ManagerBasedRLEnv, env_ids, up_ratio: float = 0.8, down_ratio: float = 0.4,
                        min_cmd_path: float = 2.0) -> torch.Tensor:
    """경로 길이 기준 터레인 승급/강등. 반환: 전체 평균 terrain level."""
    terrain = env.scene.terrain
    walk, cmdp = _path_buffers(env)
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    w, c = walk[env_ids], cmdp[env_ids]
    fell = env.termination_manager.terminated[env_ids]
    enough = c >= min_cmd_path
    move_up = (~fell) & enough & (w >= up_ratio * c)
    move_down = fell | (enough & (w < down_ratio * c))
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    walk[env_ids] = 0.0
    cmdp[env_ids] = 0.0
    return torch.mean(terrain.terrain_levels.float())

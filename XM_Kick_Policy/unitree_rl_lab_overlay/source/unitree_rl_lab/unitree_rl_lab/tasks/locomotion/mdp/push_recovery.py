# push_recovery.py
# ============================================================================
# 밀림 회복 학습용 외란 + 정지 보상 게이팅 (rs_bot RS01 V0-1 에서 도입, 로봇 무관)
#
# 배경 (2026-10-23, RS01 V0 model_13500 실측)
#   학습 외란이 "순간 속도 임펄스 ±0.4 m/s" 뿐이었다. 그 결과
#     · 툭 치는 충격(≥0.6 m/s)에는 스텝을 한다
#     · 그러나 손으로 서서히 미는 지속 외력에는 앞뒤로 17~19° 기울 때까지 발을 안 뗀다
#       (eval_push_ramp.py, 실기 증상 "10~20° 기울여도 가만히 있음" 과 일치)
#   지속 외력을 한 번도 겪지 않았기 때문이다. 여기서 그 경험을 넣는다.
#
# 구성
#   sustained_push_force     (interval, 매 스텝 전역)  사다리꼴 프로파일의 수평 지속 외력
#   reset_sustained_push     (reset)                   리셋 env 의 외력 즉시 해제 + 다음 밀기 예약
#   push_by_setting_velocity_marked (interval)         기존 임펄스 + "밀린 시각" 기록
#   stop_posture_v01_push_gated / stop_default_posture_v01_push_gated
#                            밀린 직후 hold_s 동안(그리고 지속 외력이 걸려 있는 동안) 정지 보상을 끈다
#   metric_push_recovering   (weight 0) 게이트가 꺼져 있는 비율 — 텐서보드 확인용
#
# 게이트 설계 — 정책이 조작할 수 없는 양으로 건다
#   "기울기 > X" 처럼 상태로 게이팅하면 정책이 그 상태를 만들어 벌점을 피할 수 있다
#   (MX3 기상: 높이로 다리 벌점을 게이팅했더니 낮게 서서 회피). 여기서는 외란을 건 **환경 타이머**
#   만 쓴다. 정책은 언제 밀릴지 고를 수 없다.
#
# 외력 모델
#   방향  : 월드 수평, 균등 랜덤 (밀기 동안 고정 = 손으로 미는 방향)
#   크기  : 0 → peak (rise 초) → 유지 → 0 (fall 초). 전체 duration 초
#   작용점: 몸통 COM 위 height [m]. COM 힘 + 모멘트 (0,0,h) × F 로 준다
#   ⚠ IsaacLab 의 set_external_force_and_torque 는 버퍼에 남아 매 서브스텝 적용된다.
#     매 스텝 다시 쓰고, 끝나면 0 을 써서 뗀다. 리셋 시 반드시 0 (reset_sustained_push).
# ============================================================================

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg

from .htwk_gait_v01 import stop_default_posture_v01, stop_posture_v01

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "sustained_push_force", "reset_sustained_push", "push_by_setting_velocity_marked",
    "stop_posture_v01_push_gated", "stop_default_posture_v01_push_gated",
    "metric_push_recovering", "metric_push_force", "base_tilt_excess", "push_feet_planted", "knee_bend_floor",
    "stand_tilt_excess", "inplace_tilt_excess", "walk_backtilt_excess",
    "push_lean_with_force", "push_lean_target", "stop_tilt_excess",
]

_NEVER = -10**9


def _u(lo_hi, n, device):
    lo, hi = lo_hi
    return torch.rand(n, device=device) * (hi - lo) + lo


def _init_state(env, interval_range_s):
    if hasattr(env, "_sp_t"):
        return
    n, dev = env.num_envs, env.device
    env._sp_t = torch.full((n,), -1.0, device=dev)            # 현재 밀기 경과 [s], -1 = 대기
    env._sp_wait = _u(interval_range_s, n, dev)               # 다음 밀기까지 [s]
    env._sp_dur = torch.zeros(n, device=dev)
    env._sp_rise = torch.zeros(n, device=dev)
    env._sp_fall = torch.zeros(n, device=dev)
    env._sp_peak = torch.zeros(n, device=dev)
    env._sp_h = torch.zeros(n, device=dev)
    env._sp_dir = torch.zeros(n, 3, device=dev)
    if not hasattr(env, "_push_last_step"):
        env._push_last_step = torch.full((n,), _NEVER, dtype=torch.long, device=dev)


def sustained_push_force(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    duration_range: tuple[float, float],
    rise_range: tuple[float, float],
    fall_range: tuple[float, float],
    height_range: tuple[float, float],
    interval_range_s: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis"),
    front_bias: float = 0.0,
    force_range_moving: tuple[float, float] | None = None,
    gait_command_name: str = "gait",
    rise_range_moving: tuple[float, float] | None = None,
    duration_range_moving: tuple[float, float] | None = None,
):
    """매 스텝(전역 interval) 호출. env 별로 대기 → 밀기 → 대기를 돌린다.

    front_bias (RS01 V0-4 추가, 기본 0 = 예전과 같음): 이 비율의 밀기는 '몸 앞쪽에서 뒤로' 민다
      (힘 방향 = 밀기 시작 순간 몸통 요 기준 뒤쪽 ±45°). 나머지는 월드 수평 균등 랜덤.
    force_range_moving (RS01 V0-4 4 차 추가, 기본 None = force_range 하나): 밀기 시작 순간 완전 정지가 아니면(걷는 중·멈추는 중)
      이 범위에서 힘을 뽑는다. 서 있을 때만 센 외력(깊이 기대다 놓이는 경험)을 주고, 걷는 중엔 보행이 보수화되지 않게.
    rise_range_moving / duration_range_moving (RS01 V0-6 2 차 추가, 기본 None = rise_range / duration_range 하나):
      걷는 중 시작한 밀기의 상승·지속 시간. 서 있을 때만 '천천히 세지는 긴 밀기' (사람 손) 를 넣고 걷는 중 노출은 그대로 두려고.
    """
    _init_state(env, interval_range_s)
    asset = env.scene[asset_cfg.name]
    dt = env.step_dt
    dev = env.device

    idle = env._sp_t < 0
    env._sp_wait = torch.where(idle, env._sp_wait - dt, env._sp_wait)
    start = idle & (env._sp_wait <= 0)
    if start.any():
        ids = start.nonzero(as_tuple=False).squeeze(-1)
        k = len(ids)
        env._sp_t[ids] = 0.0
        dur = _u(duration_range, k, dev)
        rise = _u(rise_range, k, dev)
        if rise_range_moving is not None or duration_range_moving is not None:
            gt_ = env.command_manager.get_term(gait_command_name)
            mv = (gt_.is_walking | gt_.is_stopping)[ids]
            if duration_range_moving is not None:
                dur = torch.where(mv, _u(duration_range_moving, k, dev), dur)
            if rise_range_moving is not None:
                rise = torch.where(mv, _u(rise_range_moving, k, dev), rise)
        env._sp_dur[ids] = dur
        env._sp_rise[ids] = torch.minimum(rise, 0.5 * env._sp_dur[ids])
        env._sp_fall[ids] = torch.minimum(_u(fall_range, k, dev), 0.5 * env._sp_dur[ids])
        env._sp_peak[ids] = _u(force_range, k, dev)
        if force_range_moving is not None:
            gt = env.command_manager.get_term(gait_command_name)
            moving = (gt.is_walking | gt.is_stopping)[ids]
            env._sp_peak[ids] = torch.where(moving, _u(force_range_moving, k, dev), env._sp_peak[ids])
        env._sp_h[ids] = _u(height_range, k, dev)
        ang = _u((-math.pi, math.pi), k, dev)
        if front_bias > 0.0:
            q = asset.data.root_quat_w[ids]
            yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            front = torch.rand(k, device=dev) < front_bias
            ang = torch.where(front, yaw + math.pi + _u((-math.pi / 4, math.pi / 4), k, dev), ang)
        env._sp_dir[ids] = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], dim=-1)
        env._push_last_step[ids] = env.common_step_counter

    active = env._sp_t >= 0
    t = env._sp_t
    up = torch.clamp(t / env._sp_rise.clamp(min=1e-3), max=1.0)
    down = torch.clamp((env._sp_dur - t) / env._sp_fall.clamp(min=1e-3), max=1.0)
    mag = env._sp_peak * torch.clamp(torch.minimum(up, down), min=0.0) * active.float()
    force = env._sp_dir * mag.unsqueeze(-1)
    arm = torch.zeros_like(force)
    arm[:, 2] = env._sp_h
    torque = torch.cross(arm, force, dim=-1)
    asset.set_external_force_and_torque(force.unsqueeze(1), torque.unsqueeze(1),
                                        body_ids=asset_cfg.body_ids, is_global=True)
    # 밀기 진행 / 종료
    env._sp_t = torch.where(active, env._sp_t + dt, env._sp_t)
    done = active & (env._sp_t >= env._sp_dur)
    if done.any():
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        env._sp_t[ids] = -1.0
        env._sp_wait[ids] = _u(interval_range_s, len(ids), dev)
        # 끝난 순간도 "밀린 시각" 으로 갱신 → 외력이 빠진 뒤 hold_s 동안 회복 시간을 준다
        env._push_last_step[ids] = env.common_step_counter


def reset_sustained_push(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    interval_range_s: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis"),
):
    """리셋된 env: 외력 즉시 0, 대기 상태로, 다음 밀기 예약, 밀린 시각 초기화."""
    _init_state(env, interval_range_s)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    env._sp_t[env_ids] = -1.0
    env._sp_wait[env_ids] = _u(interval_range_s, len(env_ids), env.device)
    env._push_last_step[env_ids] = _NEVER
    zeros = torch.zeros(len(env_ids), 1, 3, device=env.device)
    asset.set_external_force_and_torque(zeros, zeros, body_ids=asset_cfg.body_ids, env_ids=env_ids)


def push_by_setting_velocity_marked(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """IsaacLab push_by_setting_velocity + 밀린 시각 기록 (게이트용)."""
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    if not hasattr(env, "_push_last_step"):
        env._push_last_step = torch.full((env.num_envs,), _NEVER, dtype=torch.long, device=env.device)
    env._push_last_step[env_ids] = env.common_step_counter


def _recovering(env, hold_s: float) -> torch.Tensor:
    """외란 직후 hold_s 이내이거나 지속 외력이 걸려 있으면 True."""
    n = env.num_envs
    if not hasattr(env, "_push_last_step"):
        return torch.zeros(n, dtype=torch.bool, device=env.device)
    since = (env.common_step_counter - env._push_last_step).float() * env.step_dt
    rec = since < hold_s
    if hasattr(env, "_sp_t"):
        rec = rec | (env._sp_t >= 0)
    return rec


def stop_posture_v01_push_gated(env, sensor_cfg, gait_command_name: str = "gait", hold_s: float = 1.5):
    """stop_posture_v01 (양발 접지 보상) — 외란 회복 중에는 0."""
    r = stop_posture_v01(env, sensor_cfg=sensor_cfg, gait_command_name=gait_command_name)
    return r * (~_recovering(env, hold_s)).float()


def stop_default_posture_v01_push_gated(env, asset_cfg, gait_command_name: str = "gait",
                                        include_stopping: bool = False, hold_s: float = 1.5):
    """stop_default_posture_v01 (기본자세 복귀 벌점) — 외란 회복 중에는 0."""
    r = stop_default_posture_v01(env, asset_cfg=asset_cfg, gait_command_name=gait_command_name,
                                 include_stopping=include_stopping)
    return r * (~_recovering(env, hold_s)).float()


def metric_push_recovering(env, hold_s: float = 1.5):
    """진단용 (weight 0): 정지 보상 게이트가 꺼진(회복 중) env 비율."""
    return _recovering(env, hold_s).float()


def metric_push_force(env):
    """진단용 (weight 0): 현재 지속 외력 크기 [N]."""
    if not hasattr(env, "_sp_t"):
        return torch.zeros(env.num_envs, device=env.device)
    t = env._sp_t
    active = t >= 0
    up = torch.clamp(t / env._sp_rise.clamp(min=1e-3), max=1.0)
    down = torch.clamp((env._sp_dur - t) / env._sp_fall.clamp(min=1e-3), max=1.0)
    return env._sp_peak * torch.clamp(torch.minimum(up, down), min=0.0) * active.float()


# ============================================================================
# 기울기 초과 벌점 (RS01 V0-3 도입)
#
# 배경 (2026-10-24, RS01 V0-2 model_34500 — eval_push_ramp / tilt_dist)
#   정지 지속 외력 램프에서 앞 방향 첫 스텝이 22.2° · 18.8 N (V0-1 12.7° · 9.7 N, 옆 3~4°).
#   18.8 N 은 학습 외력 상한(18 N) 바로 위 — 학습 범위의 앞 방향 밀림은 전부 기울여서 버텼다.
#   몸통 벌점이 roll −5 / pitch −1 이라 앞뒤로 기우는 게 싸다(옆은 3~4° 에서 스텝).
#   V0-2 의 발목 토크 한도 벌점이 "발목으로 버티기" 를 비싸게 만들어 기울이기로 더 기운 것으로 본다.
#   학습 분포 실측 (지형+외란): 평소 보행 p95 6.1° / p99 9.8°, 외란 중 p95 20~22° / p99 27~29°.
#
# 설계
#   relu(tilt − deadband)²  [rad²], tilt = 중력 방향과 몸통 z 축 사이 각 (roll·pitch 합)
#   데드밴드 6° — 평소 보행엔 거의 0 (V0-2 평균 0.0002), 외란 중에만 큼 (평균 0.01).
#   상시 적용, 게이트 없음 — 기울기를 줄이는 정당한 방법은 발을 옮겨 지지면을 넓히는 것뿐이다
#   (발목 토크는 한도 벌점, 엉덩이로 버티면 몸통이 기운다).
# ============================================================================
def base_tilt_excess(env, deadband_deg: float = 6.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """relu(tilt − deadband)² [rad²]. 음수 weight 로 쓴다."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    return torch.relu(tilt - math.radians(deadband_deg)).pow(2)


# ============================================================================
# 지속 외력 중 발 붙이기 (RS01 V0-4 도입)
#
# 배경 (2026-10-26, V0-2 실기 + lean_response / lean_torque 시뮬)
#   V0-2 는 뒤에서 밀면 힘 쪽으로 기대어(pitch −16~−23°) 17~19 N 까지 발을 안 떼지만,
#   앞에서 밀면 몸이 힘에 밀려 뒤로 젖혀지고(5 N 에 −3°) 7.4 N 에서 뒤로 발을 뺀다. 실기에서 뒤로 빼는 발이 불안정.
#   V0-1 은 앞에서 밀 때도 앞으로 기댔다(+16~+24°, 발목 토크 최대 2.8 N·m — 모터 한도 탓 아님).
#   V0-1 부터 밀리는 동안 정지 보상(양발 접지)을 꺼 둬서 '기대기' 와 '발 빼기' 가 보상상 거의 같다 → 방향마다 우연히 갈렸다.
#
# 설계: 서 있다가(완전 정지) 지속 외력이 걸려 있는 동안 떨어진 발 수를 벌한다. 음수 weight.
#   버틸 수 있는 힘에서는 기대어 버티게, 넘어질 만큼 세면(종료 비용이 훨씬 큼) 여전히 발을 뗀다.
#   순간 충격(속도 임펄스)과 외력이 풀린 뒤는 대상 아님 — 회복 스텝은 막지 않는다.
#   게이트 = 환경 타이머(_sp_t) + 명령 상태(완전 정지). 정책이 조작할 수 있는 상태량이 아니다.
# ============================================================================
def push_feet_planted(env, sensor_cfg: SceneEntityCfg, gait_command_name: str = "gait",
                      max_force: float = 1.0e9) -> torch.Tensor:
    """완전 정지 중 지속 외력이 걸려 있고 그 순간 힘이 max_force [N] 이하이면 바닥에서 떨어진 발 수 (0~2).

    ★ V0-4 2 차: max_force 추가. 1 차(모든 힘에서 발 떼기 벌점)는 기대는 각도에 상한이 없어 +34 → +40° 로 계속 커졌고
      밀다 놓으면 앞으로 넘어졌다 (15 N 12 %, 20 N 55 %). '작은 힘은 기대어 버티고, 큰 힘은 발을 떼라' 로 목표를 분명히 한다.
      힘 크기는 환경이 정한 값(외란 프로파일)이라 정책이 조작할 수 없다.
    """
    from isaaclab.sensors import ContactSensor

    if not hasattr(env, "_sp_t"):
        return torch.zeros(env.num_envs, device=env.device)
    term = env.command_manager.get_term(gait_command_name)
    full_stop = (~term.is_walking) & (~term.is_stopping)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0
    gate = (env._sp_t >= 0) & full_stop & (metric_push_force(env) <= max_force)
    return in_air.float().sum(dim=1) * gate.float()


# ============================================================================
# 무릎 최소 굽힘 보호 장치 (RS01 V0-4 2 차 도입)
#
# 배경: '밀려도 버텨라' 압력을 주면 두 번 모두 다리를 펴 기둥처럼 버티는 쪽으로 갔고 그 자세가 걷기로 번졌다.
#   V0-3 (기울기 벌점): 디딤 무릎 절대 3~4°, 무릎 < 15° 인 시간 63 %
#   V0-4 1 차 (발 붙이기 벌점) model_36000: < 15° 6.2 % (V0-2 2.9 %), 디딤 무릎 −3 → −8°, 스윙 무릎 최대 17 → 11° 로 진행 중
# 설계: Σ relu(min_bend − |q_knee|)² [rad²]. |q| = 무릎 굽힘 크기 (왼무릎 −140~0, 오른무릎 0~140). 음수 weight.
#   V0-2 걸음새는 평균 0.00027 (거의 0), V0-3 걸음새는 0.023 (85 배). 정상 걸음은 안 건드리고 펴지는 쪽만 막는다.
# ============================================================================
def knee_bend_floor(env, asset_cfg: SceneEntityCfg, min_bend_deg: float = 15.0) -> torch.Tensor:
    """Σ relu(min_bend − |q_knee|)² [rad²]."""
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids].abs()
    return torch.relu(math.radians(min_bend_deg) - q).pow(2).sum(dim=1)


# ============================================================================
# 외란 없이 서 있을 때 몸통 세우기 (RS01 V0-4 3 차 도입)
#
# 배경 (2026-10-27): V0-4 2 차는 밀림 목표를 달성했지만 서 있을 때 몸통이 체크포인트마다 +1.7~+6.2° (평균 +2.5~3°) 앞으로 숙였다
#   (V0-2 +0.8°). 무게중심은 그대로(2.6~3.8 cm, V0-2 2.8)라 엉덩이를 접어 상체만 숙인 자세. 지속 외력의 절반이 앞에서
#   오니 미리 숙여 두는 게 약간 이득이고, 세우는 힘은 몸통 pitch 벌점(−1, 3° 에 초당 0.003)뿐이라 0 으로 돌아오지 않는다.
# 설계: relu(tilt − deadband)² [rad²], tilt = 중력-몸통 z 축 각. 완전 정지 & 외란 회복 구간 아님(_recovering: 지속 외력 중 +
#   어떤 밀기든 끝난 뒤 hold_s 이내 제외) 일 때만. 밀리는 동안의 기대기는 건드리지 않는다. 게이트 = 명령 상태 + 환경 타이머.
#   V0-3 의 상시 기울기 벌점은 편 다리를 만들었다 — 이 항은 서 있을 때만이고 무릎 보호 장치(knee_bend_floor)가 같이 있다.
# ============================================================================
def stand_tilt_excess(env, gait_command_name: str = "gait", hold_s: float = 1.5, deadband_deg: float = 1.0,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """완전 정지 & 외란 회복 구간 아님일 때 relu(tilt − deadband)² [rad²]. 음수 weight."""
    term = env.command_manager.get_term(gait_command_name)
    full_stop = (~term.is_walking) & (~term.is_stopping)
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    pen = torch.relu(tilt - math.radians(deadband_deg)).pow(2)
    return pen * (full_stop & ~_recovering(env, hold_s)).float()


# ============================================================================
# 제자리걸음 몸통 앞뒤 기울기 초과 벌점 (V0-5 2 차)
# 배경 (2026-10-28): V0-5 1 차(마찰 0.3~1.2) 47000 실기에서 과회전은 잡혔지만 제자리걸음 때 몸통이 뒤로 5~8° 젖혀졌다
#   (시뮬도 −7°, V0-4 는 +2~3°). stand_tilt_excess 는 '완전 정지'(위상 시계 멈춤) 때만 걸려 제자리걸음은 보호받지 못했다.
#   체크포인트 스캔: 45000 부터 −5~−10°, 43500 은 시뮬 −0.6° 였지만 실기 −6.3° — 체크포인트 선택으로는 안 잡히는 약한 자세.
# 설계: relu(|pitch| − deadband)² [rad²], pitch = atan2(g_x, −g_z) (앞 +). 걸음 중 좌우 흔들림(roll)은 정상이라 pitch 만.
#   게이트 = 속도 명령 정지 & 위상 시계가 도는 중(제자리걸음) & 외란 회복 구간 아님 — 명령 상태 + 환경 타이머라 정책이 못 바꾼다.
#   걸음 중엔 몸통이 출렁이므로 deadband 를 서 있을 때(1°)보다 넓게 (2.5°).
# ============================================================================
def inplace_tilt_excess(env, gait_command_name: str = "gait", hold_s: float = 1.5, deadband_deg: float = 2.5,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """제자리걸음(명령 0 & 위상 회전) & 외란 회복 구간 아님일 때 relu(|pitch| − deadband)² [rad²]. 음수 weight."""
    term = env.command_manager.get_term(gait_command_name)
    inplace = term.is_walking & term._standing_mask()
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    pen = torch.relu(pitch.abs() - math.radians(deadband_deg)).pow(2)
    return pen * (inplace & ~_recovering(env, hold_s)).float()


# ============================================================================
# 걷는 중 몸통 뒤로 젖힘 초과 벌점 (V0-5 3 차)
# 배경 (2026-10-28): 2 차에서 제자리걸음에만 벌점(inplace_tilt_excess)을 걸었더니 뒤로 젖히는 성향이 전진으로 옮겨갔다
#   (전진 0.5 pitch 47000 −3.5 → 2 차 끝 48900 −12.3°). 막힌 곳 대신 벌점 없는 곳으로 새는 패턴.
# 설계: relu(−pitch − deadband)² [rad²] — 뒤로 젖힘만. 앞으로 숙이는 건 빠른 걸음에서 자연스럽고 실기는 시뮬보다 4~9° 더 숙이는 편이라 막지 않는다.
#   게이트 = 이동 명령 중(속도 명령 ≥ 정지 문턱) & 위상 회전 & 외란 회복 아님. 제자리걸음은 inplace_tilt_excess 가 맡는다.
#   V0-3 의 상시 기울기 벌점(양방향 6°)은 편 다리를 만들었다 — knee_bend_floor 가 같이 있고, 이 항은 한 방향·걷는 중만.
# ============================================================================
def walk_backtilt_excess(env, gait_command_name: str = "gait", hold_s: float = 1.5, deadband_deg: float = 3.0,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """이동 명령 & 위상 회전 & 외란 회복 아님일 때 relu(−pitch − deadband)² [rad²] (뒤로 젖힘만). 음수 weight."""
    term = env.command_manager.get_term(gait_command_name)
    moving = term.is_walking & ~term._standing_mask()
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    pen = torch.relu(-pitch - math.radians(deadband_deg)).pow(2)
    return pen * (moving & ~_recovering(env, hold_s)).float()


# ============================================================================
# 지속 외력 방향으로 숙임 벌점 (V0-5 5 차)
# 배경 (2026-10-29): 실기 "49500 은 자세가 안정적인데 뒤에서 밀 때 버티기는 47000 이 좋다 — 전진하다 멈출 때도
#   한 번에 서느냐, 반동 + 발구르기로 균형을 잡느냐의 차이". 시뮬 명목 밀기 시험(실기 순위와 유일하게 맞은 지표):
#   47000 뒤 밀기 10 · 15 N 에서 몸통 −1.5 · 0° (똑바로), 첫 스텝 24 N / 49500 +12 · +20° (미는 방향으로 숙임), 22 N.
#   push_feet_planted (발 떼기 벌점) 만 있어서 '발 붙인 채 힘에 밀려 숙이기' 가 공짜였다. 3 차 이후 체크포인트마다 숙임 +12~+32°.
# 설계: 몸통 z 축의 수평 성분을 외력 방향(월드 수평 단위벡터)에 투영한 기울기 = '힘 쪽으로 숙인 각'.
#   relu(그 각 − deadband)² [rad²], 힘 반대쪽으로 기대는 건 0 (앞에서 밀 때 앞으로 기대는 V0-4 레시피는 그대로).
#   게이트 = push_feet_planted 와 같음: 완전 정지 & 지속 외력 중 & 그 순간 힘 ≤ max_force (환경 타이머·명령·외란 프로파일 — 정책이 못 바꿈).
#   발을 떼서 피하지 않도록 push_feet_planted 와 짝으로 쓴다 (둘 다 0 인 해 = 발 붙이고 똑바로 버티기, 47000 이 실제로 한 것).
# ============================================================================
def push_lean_with_force(env, gait_command_name: str = "gait", max_force: float = 20.0, deadband_deg: float = 5.0,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """완전 정지 & 지속 외력(≤ max_force) 중 relu(외력 방향 숙임각 − deadband)² [rad²]. 음수 weight."""
    if not hasattr(env, "_sp_t"):
        return torch.zeros(env.num_envs, device=env.device)
    term = env.command_manager.get_term(gait_command_name)
    full_stop = (~term.is_walking) & (~term.is_stopping)
    gate = (env._sp_t >= 0) & full_stop & (metric_push_force(env) <= max_force)
    lean = _lean_along_push(env, asset_cfg)
    pen = torch.relu(lean - math.radians(deadband_deg)).pow(2)
    return pen * gate.float()


def _lean_along_push(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """몸통 z 축 수평 성분을 지속 외력 방향(env._sp_dir, 월드 수평 단위벡터)에 투영한 기울기 [rad]. + = 힘 쪽으로 숙임."""
    q = env.scene[asset_cfg.name].data.root_quat_w            # (w, x, y, z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # 몸통 z 축의 월드 성분 (회전행렬 3 번째 열)
    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    zz = 1.0 - 2.0 * (x * x + y * y)
    along = zx * env._sp_dir[:, 0] + zy * env._sp_dir[:, 1]
    return torch.atan2(along, zz.clamp(min=1e-3))


# ============================================================================
# 멈춘 직후 몸통 앞뒤 기울기 벌점 (V0-5 5 차)
# 배경: 위와 같은 실기 피드백 (멈출 때 앞으로 쏠렸다 반동 + 발구르기). 시뮬 정지 시험: 49500 은 똑바로 걷다 멈추면 +5.5° 쏠렸다 돌아옴.
#   서 있을 때·제자리걸음 기울기 벌점(stand/inplace_tilt_excess)이 이미 있지만, 학습 중 '걷다가 멈추기' 가 env 당 ~75 s 에 한 번뿐이라
#   거의 배우지 않는 상황이고, 멈추는 중(위상이 0 으로 가는 0.25 s)은 어느 벌점도 안 걸렸다.
# 설계: 속도 명령이 이동 → 정지로 바뀐 뒤 window_s 동안 relu(|pitch| − deadband)² [rad²]. 앞쏠림과 뒤 반동 둘 다.
#   게이트 = 명령 전환 시각(환경 타이머) & 지금도 정지 명령 & 외란 회복 아님. 정지 사건 빈도는 cfg 에서 명령 재샘플링으로 늘린다.
#   상태: env._stopw_prev (직전 스텝 정지 명령 여부), env._stopw_t (전환 후 경과 s). 리셋 직후 첫 스텝은 전환으로 안 친다.
# ============================================================================
def stop_tilt_excess(env, gait_command_name: str = "gait", window_s: float = 1.5, deadband_deg: float = 2.0,
                     hold_s: float = 1.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """이동 → 정지 명령 전환 후 window_s 이내 & 외란 회복 아님일 때 relu(|pitch| − deadband)² [rad²]. 음수 weight."""
    term = env.command_manager.get_term(gait_command_name)
    standing = term._standing_mask()
    fresh = env.episode_length_buf <= 1
    if not hasattr(env, "_stopw_t"):
        env._stopw_prev = standing.clone()
        env._stopw_t = torch.full((env.num_envs,), 1.0e9, device=env.device)
    newly = standing & ~env._stopw_prev & ~fresh
    env._stopw_t = torch.where(newly, torch.zeros_like(env._stopw_t), env._stopw_t + env.step_dt)
    env._stopw_t = torch.where(fresh, torch.full_like(env._stopw_t, 1.0e9), env._stopw_t)
    env._stopw_prev = standing.clone()
    gate = standing & (env._stopw_t < window_s) & ~_recovering(env, hold_s)
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    pen = torch.relu(pitch.abs() - math.radians(deadband_deg)).pow(2)
    return pen * gate.float()


# ============================================================================
# 지속 외력에 '힘 반대쪽으로 기대기' 각도 목표 (V0-5 5 차 수정판)
# 배경 (2026-10-29): 사용자 "v0_4 (38500) 가 내가 원하는 밀기 동작" — 앞·뒤 모두 발 붙인 채 힘 반대쪽으로 몸을 꺾어 기대며 버팀.
#   38500 시뮬 (lean_torque2): 뒤 10·15 N 몸통 −18·−25°, 앞 +20·+25°, 첫 스텝 18·20 N, 놓기 넘어짐 0 %.
#   5 차 push_lean_with_force (힘 방향 숙임 > 5° 만 벌) 는 숙임을 +6~8° 까지 줄였지만 deadband 에서 멈춤 — '기대라' 는 신호가 없었다.
# 설계: 기대기 각 c = −(외력 방향 숙임각) (+ = 힘 반대쪽). 목표 c* = min(deg_per_n × F, max_deg) — 38500 실측 (≈1.7°/N, 25° 에서 포화).
#   relu(|c − c*| − tol)² [rad²]. 덜 기대도(똑바로·힘 방향 숙임) 더 기대도(앞 밀기 +34° 같은) 벌한다. 상한이 있어 V0-4 1 차처럼
#   기대는 각이 끝없이 커져 놓으면 넘어지는 일을 막는다. 게이트 = push_feet_planted 와 같음 (완전 정지 & 지속 외력 & F ≤ max_force).
#   push_feet_planted 와 짝: 둘 다 0 인 해 = 발 붙이고 목표만큼 기대기 = 38500 의 동작.
# ============================================================================
def push_lean_target(env, gait_command_name: str = "gait", max_force: float = 20.0, deg_per_n: float = 1.7,
                     max_deg: float = 25.0, tol_deg: float = 5.0,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """완전 정지 & 지속 외력(≤ max_force) 중 relu(|힘 반대 기대기각 − min(k·F, max)| − tol)² [rad²]. 음수 weight."""
    if not hasattr(env, "_sp_t"):
        return torch.zeros(env.num_envs, device=env.device)
    term = env.command_manager.get_term(gait_command_name)
    full_stop = (~term.is_walking) & (~term.is_stopping)
    f = metric_push_force(env)
    gate = (env._sp_t >= 0) & full_stop & (f <= max_force)
    counter = -_lean_along_push(env, asset_cfg)
    target = torch.clamp(f * math.radians(deg_per_n), max=math.radians(max_deg))
    pen = torch.relu((counter - target).abs() - math.radians(tol_deg)).pow(2)
    return pen * gate.float()

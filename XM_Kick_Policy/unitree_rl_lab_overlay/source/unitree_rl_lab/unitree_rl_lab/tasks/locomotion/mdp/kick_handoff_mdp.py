# kick_handoff_mdp.py
# ============================================================================
# walk_kick v1_5 — "경기에서 넘겨받는 순간" 을 학습에 넣는 MDP 항 (2026-10-31)
# ----------------------------------------------------------------------------
# 근거: scripts/mujoco_sim/kick_match_cond.py (경기 넘겨받기 615 건 × 조건, 브리지 물리)
#   W90B70 18200 — 서서 차기 0.2 % / 걷다가 4~6 % / 공이 추정보다 +3·+4 cm 멂 6 · 19 % /
#   0.3 m/s 로 굴러 멀어짐 19 % / 헛발질 3 %. 넘어진 판의 95 % 가 오른발·앞으로.
#   학습은 (1) 서 있다가 시작 (2) 공 관측 오차가 매 스텝 i.i.d. 라 5 프레임 평균이면 정확
#   (3) 공이 못 구름 (4) 다리 목표 즉시 적용 — 실기·경기와 네 군데가 다르다.
#
#   1. ProfileVelocityJointPositionAction : 다리 목표를 Dynamixel Profile Velocity (실기 45 rpm)
#      처럼 물리 스텝마다 rate 만큼만 따라가게 한다 (sim_bridge.py pre_step 과 같은 식).
#   2. ResetFromHandoffBank               : 최종 보행 정책 (v0_6 13500) 이 넘겨받기 명령으로 걷다가
#      명령 0 을 0.04 s 받은 직후의 상태 (gen_handoff_bank.py) 에서 에피소드를 시작한다.
#   3. perturb_ball_belief                : 정책이 아는 공 (frozen 관측) 과 실제 공을 어긋나게 한다
#      (고정 오차 / 굴러가는 중 / 헛발질 / 다툼으로 튕겨 나감).
#   4. duel_knock_ball                    : 다툼 에피소드에서 다가가는 도중 공을 튕겨 낸다.
#   5. ball_rolling_resistance            : 공 회전 상한을 풀면 PhysX 공은 감쇠 0.2 만으로 6~17 m
#      구른다 → 구름 저항 (평면 속도 일정 감속) 을 매 스텝 속도에 직접 준다.
#   6. fall_by_type                       : 넘어짐 종료를 에피소드 종류별로 나눠 TB 에서 따로 본다.
# ============================================================================

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
from .kick_clock import kick_clock_s, kick_clock_steps  # 통합 태스크의 킥 시작 기준 시각 (기존 태스크는 에피소드 시각 그대로)

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from . import kick_mdp
from .reset_hold import DelayedJointPositionAction, DelayedJointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 에피소드 종류 (env.kick_episode_type). 시작 상태 (서서/걷다가) 는 env.kick_from_bank 로 따로 둔다.
EP_NOMINAL, EP_BIAS, EP_ROLL, EP_WHIFF, EP_DUEL = 0, 1, 2, 3, 4
EP_NAMES = {"nominal": EP_NOMINAL, "bias": EP_BIAS, "roll": EP_ROLL, "whiff": EP_WHIFF, "duel": EP_DUEL}

_RPM = 2.0 * math.pi / 60.0


# ============================================================================
# 1. Profile Velocity 행동
# ============================================================================
class ProfileVelocityJointPositionAction(DelayedJointPositionAction):
    """DelayedJointPositionAction (홀드·지연·클램프) + 목표 속도 제한.

    실기 body_motor_controller 는 다리 Profile Velocity 를 45 rpm (= 4.71 rad/s, 모터 Velocity
    Limit 45.2 rpm 바로 아래) 으로 둔다. 서보는 새 목표를 받으면 그 속도로 이동하는 궤적을 만든다.
    학습은 목표를 즉시 PD 에 넣었다 → 같은 정책이라도 킥 스윙이 실기에선 다르게 나온다.
    (MuJoCo 에서 같은 615 건: 목표 즉시 14 % vs 45 rpm 따라가기 0.2 % 넘어짐 — 무시할 차이가 아니다.)

    setpoint 는 물리 스텝마다 rate·dt 만큼 목표로 다가간다. 리셋 직후 첫 스텝에 현재 관절각에서 시작.
    rate 는 env 마다 profile_rpm_range 에서 뽑는다 (실기 펌웨어 궤적과 완전히 같다는 보장이 없어서).
    """

    cfg: "ProfileVelocityJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = env.num_envs
        self._sp = torch.zeros_like(self._processed_actions)
        self._sp_valid = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._rate = torch.zeros(n, 1, device=env.device)
        self._sample_rate(torch.arange(n, device=env.device))

    def _sample_rate(self, env_ids):
        lo, hi = self.cfg.profile_rpm_range
        rpm = torch.empty(len(env_ids), device=self._env.device).uniform_(lo, hi)
        self._rate[env_ids, 0] = rpm * _RPM * self._env.physics_dt

    def apply_actions(self):
        if not torch.all(self._sp_valid):
            ids = (~self._sp_valid).nonzero(as_tuple=False).squeeze(-1)
            self._sp[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
            self._sp_valid[ids] = True
        self._sp += torch.clamp(self._processed_actions - self._sp, -self._rate, self._rate)
        self._asset.set_joint_position_target(self._sp, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self._env.device)
        if len(env_ids) == 0:
            return
        self._sp_valid[env_ids] = False
        self._sample_rate(env_ids)


@configclass
class ProfileVelocityJointPositionActionCfg(DelayedJointPositionActionCfg):
    class_type = ProfileVelocityJointPositionAction

    profile_rpm_range: tuple[float, float] = (40.0, 50.0)
    """다리 Profile Velocity [rpm]. 실기 45 rpm (dynamixel_system.launch.py profile_velocity_limit_rpm)."""


# ============================================================================
# 2. 넘겨받기 상태 은행에서 시작
# ============================================================================
class ResetFromHandoffBank(ManagerTermBase):
    """reset_robot_joints 자리 (reset_base 바로 뒤) 에 넣는다.

    은행 (torch.save dict): joint_names, joint_pos (M, J), joint_vel (M, J), root_z (M,),
    root_roll / root_pitch (M,), lin_vel_h / ang_vel_h (M, 3) — 로봇 heading (yaw) 기준 프레임.
    reset_base 가 정한 xy 위치·yaw 는 그대로 두고, 높이·roll·pitch·속도·관절을 은행 표본으로 바꾼다.
    standing_prob 비율은 예전처럼 서서 시작 (reset_joints_by_scale) — 기존 기술을 잊지 않게.
    """

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        path = os.path.expanduser(p["bank_path"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[kick_handoff] 넘겨받기 상태 은행이 없습니다: {path} "
                                    f"(scripts/kick_v1_5/gen_handoff_bank.py 로 만든다)")
        bank = torch.load(path, map_location="cpu")
        robot: Articulation = env.scene[p.get("asset_cfg", SceneEntityCfg("robot")).name]
        names = list(bank["joint_names"])
        missing = [j for j in robot.joint_names if j not in names]
        if missing:
            raise KeyError(f"[kick_handoff] 은행에 없는 관절: {missing}")
        idx = torch.tensor([names.index(j) for j in robot.joint_names])
        dev = env.device
        self.jp = bank["joint_pos"][:, idx].to(dev)
        self.jv = bank["joint_vel"][:, idx].to(dev)
        self.z = bank["root_z"].to(dev)
        self.roll = bank["root_roll"].to(dev)
        self.pitch = bank["root_pitch"].to(dev)
        self.lin = bank["lin_vel_h"].to(dev)
        self.ang = bank["ang_vel_h"].to(dev)
        self.m = self.jp.shape[0]
        if not hasattr(env, "kick_from_bank"):
            env.kick_from_bank = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        print(f"[kick_handoff] 넘겨받기 상태 {self.m} 개 ({path})")

    def __call__(self, env: "ManagerBasedRLEnv", env_ids: torch.Tensor, bank_path: str,
                 standing_prob: float, position_range: tuple[float, float],
                 velocity_range: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        from isaaclab.envs import mdp as il_mdp

        robot: Articulation = env.scene[asset_cfg.name]
        stand = torch.rand(len(env_ids), device=env.device) < standing_prob
        env.kick_from_bank[env_ids] = ~stand
        ids_s = env_ids[stand]
        if len(ids_s):
            il_mdp.reset_joints_by_scale(env, ids_s, position_range, velocity_range, asset_cfg)
        ids = env_ids[~stand]
        if len(ids) == 0:
            return
        k = torch.randint(0, self.m, (len(ids),), device=env.device)
        root = robot.data.root_state_w[ids].clone()
        _, _, yaw = euler_xyz_from_quat(root[:, 3:7])
        root[:, 2] = env.scene.env_origins[ids, 2] + self.z[k]
        root[:, 3:7] = quat_from_euler_xyz(self.roll[k], self.pitch[k], yaw)
        c, s = torch.cos(yaw), torch.sin(yaw)

        def rot(v):     # heading 프레임 → 월드
            return torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], dim=-1)

        root[:, 7:10] = rot(self.lin[k])
        root[:, 10:13] = rot(self.ang[k])
        robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
        robot.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)
        robot.write_joint_state_to_sim(self.jp[k], self.jv[k], env_ids=ids)


# ============================================================================
# 3. 공 믿음 오차 (reset 이벤트 — reset_kick_state 가 frozen 관측을 찍은 뒤에 와야 한다)
# ============================================================================
def perturb_ball_belief(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    probs: dict,
    bias_x_range: tuple[float, float],
    bias_y_range: tuple[float, float],
    roll_speed_range: tuple[float, float],
    roll_away_prob: float,
    whiff_distance: float,
    duel_time_range: tuple[float, float],
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """frozen 관측 (env.ball_pos_b_frozen) 은 그대로 두고 실제 공만 옮긴다 → 실제 = 추정 + 오차.

    probs: {"nominal", "bias", "roll", "whiff", "duel"} 비율 (합 1).
      bias  : 몸통 heading 기준 (dx, dy) 고정 오차. +x = 실제 공이 더 멀다 (가장 위험한 쪽이라 넓게)
      roll  : 미끄럼 없는 구름으로 굴러가는 중. roll_away_prob 비율은 로봇→공 방향 ±45° (멀어짐)
      whiff : 실제 공을 로봇 뒤 whiff_distance 로 치움 (킥 보상이 나올 수 없다 → 헛발질 후 버티기만 배움)
      duel  : duel_time_range 사이 한 시각에 duel_knock_ball 이 공을 튕겨 낸다
    """
    kick_mdp._ensure_kick_state(env)
    dev = env.device
    n = len(env_ids)
    if not hasattr(env, "kick_episode_type"):
        env.kick_episode_type = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        env.duel_knock_t = torch.full((env.num_envs,), -1.0, device=dev)
    p = torch.tensor([probs[k] for k in ("nominal", "bias", "roll", "whiff", "duel")], device=dev)
    kind = torch.multinomial(p / p.sum(), n, replacement=True)
    env.kick_episode_type[env_ids] = kind
    env.duel_knock_t[env_ids] = -1.0

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_ids])
    c, s = torch.cos(yaw), torch.sin(yaw)
    pose = ball.data.root_pose_w[env_ids].clone()
    vel = torch.zeros(n, 6, device=dev)
    r_ball = float(pose[0, 2]) if n else 0.065

    m = kind == EP_BIAS
    if m.any():
        dx = torch.empty(int(m.sum()), device=dev).uniform_(*bias_x_range)
        dy = torch.empty(int(m.sum()), device=dev).uniform_(*bias_y_range)
        pose[m, 0] += c[m] * dx - s[m] * dy
        pose[m, 1] += s[m] * dx + c[m] * dy

    m = kind == EP_ROLL
    if m.any():
        k = int(m.sum())
        sp = torch.empty(k, device=dev).uniform_(*roll_speed_range)
        rel = pose[m, :2] - robot.data.root_pos_w[env_ids][m, :2]
        away = torch.atan2(rel[:, 1], rel[:, 0]) + torch.empty(k, device=dev).uniform_(-math.pi / 4, math.pi / 4)
        anyd = torch.empty(k, device=dev).uniform_(-math.pi, math.pi)
        ang = torch.where(torch.rand(k, device=dev) < roll_away_prob, away, anyd)
        vx, vy = sp * torch.cos(ang), sp * torch.sin(ang)
        vel[m, 0], vel[m, 1] = vx, vy
        vel[m, 3], vel[m, 4] = -vy / r_ball, vx / r_ball        # ω = z × v / r

    m = kind == EP_WHIFF
    if m.any():
        pose[m, 0] = robot.data.root_pos_w[env_ids][m, 0] - c[m] * whiff_distance
        pose[m, 1] = robot.data.root_pos_w[env_ids][m, 1] - s[m] * whiff_distance

    m = kind == EP_DUEL
    if m.any():
        env.duel_knock_t[env_ids[m]] = torch.empty(int(m.sum()), device=dev).uniform_(*duel_time_range)

    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)


# ============================================================================
# 4. 다툼: 다가가는 도중 공을 튕겨 냄 (interval 이벤트, 매 스텝)
# ============================================================================
def duel_knock_ball(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    speed_range: tuple[float, float],
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """duel_knock_t (에피소드 시각 [s]) 를 막 지난 env 의 공을 무작위 방향으로 한 번 튕긴다."""
    if not hasattr(env, "duel_knock_t"):
        return
    t = kick_clock_s(env)[env_ids]
    knock_t = env.duel_knock_t[env_ids]
    hit = (knock_t >= 0) & (t >= knock_t)
    if not hit.any():
        return
    ids = env_ids[hit]
    env.duel_knock_t[ids] = -1.0
    ball: RigidObject = env.scene[ball_cfg.name]
    k = len(ids)
    sp = torch.empty(k, device=env.device).uniform_(*speed_range)
    ang = torch.empty(k, device=env.device).uniform_(-math.pi, math.pi)
    vel = torch.zeros(k, 6, device=env.device)
    vel[:, 0], vel[:, 1] = sp * torch.cos(ang), sp * torch.sin(ang)
    ball.write_root_velocity_to_sim(vel, env_ids=ids)


# ============================================================================
# 5. 공 구름 저항 (interval 이벤트, 매 스텝 — 외력은 다음 갱신까지 유지된다)
# ============================================================================
def ball_rolling_resistance(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    decel: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """매 스텝 공 평면 속도를 decel·dt 만큼 깎고 (0 에서 멈춤), 수평 회전도 같은 비율로 줄인다 (구름 유지).

    외력 (set_external_force_and_torque) 으로는 안 된다: IsaacLab 2.3 wrench composer 는 힘을 몸체 좌표로
    저장하는데 공이 60 rad/s 로 돌면 물리 4 스텝 사이에 힘 방향이 같이 돌아 대부분 상쇄된다
    (보정 시험에서 decel 1.6 → 3.2 로 올려도 굴림 거리 17.2 → 16.7 m).
    decel 은 MuJoCo 경기 공 (convexhull, rolling 0.030 + 감쇠 0.2) 굴림 거리에 맞춘 값
    (scripts/kick_v1_5/calib_ball_roll.py). 공이 떠 있으면 (z > 반지름 + 1 cm) 건드리지 않는다.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    vel = ball.data.root_vel_w[env_ids].clone()          # (lin 3, ang 3)
    sp = torch.norm(vel[:, :2], dim=-1)
    on_ground = (ball.data.root_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]) < 0.075
    moving = on_ground & (sp > 1e-4)
    if not moving.any():
        return
    ratio = torch.clamp(sp - decel * env.step_dt, min=0.0) / torch.clamp(sp, min=1e-6)
    ratio = torch.where(moving, ratio, torch.ones_like(ratio)).unsqueeze(-1)
    vel[:, :2] *= ratio
    vel[:, 3:5] *= ratio
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)


# ============================================================================
# 5b. 팔 토크 한계 무작위 (startup — PhysX 관절 최대 힘을 env 마다 한 번)
# ============================================================================
def randomize_joint_effort_limit(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    effort_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """실기 팔은 goal PWM 5.6 % 로 묶여 토크가 작고 개체·전압마다 다르다 → 관절 최대 힘을 범위에서 뽑는다."""
    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:                     # startup 모드는 env_ids=None 으로 부른다
        env_ids = torch.arange(env.num_envs, device=env.device)
    jids = asset_cfg.joint_ids
    n_j = robot.num_joints if isinstance(jids, slice) else len(jids)
    lim = torch.empty(len(env_ids), n_j, device=env.device).uniform_(*effort_range)
    robot.write_joint_effort_limit_to_sim(lim, joint_ids=jids, env_ids=env_ids)


# ============================================================================
# 5c. 몸통 yaw 각속도 상한 벌점
# ============================================================================
def base_yaw_rate_over_limit(
    env: "ManagerBasedRLEnv",
    limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(|ω_z| − limit)₊² — 음수 weight. v1_4 kick_ang_vel 은 roll/pitch 만 본다.

    v1_5 19200 은 킥할 때 몸통을 6.3 rad/s 로 비튼다 (W90B70 4.4). Isaac 은 발바닥 패치 마찰이 비틀림을
    꽉 잡아 문제없지만, 디딤발이 도는 바닥 (MuJoCo 비틀림 0.2, 실제 잔디도 그쪽일 수 있다) 에서는 공이 없을 때
    (헛발질) 회전을 못 멈추고 뒤로 넘어진다: MuJoCo 헛발질 넘어짐 73 % (비틀림 1.0 이면 10 %, W90B70 1.5 %).
    보행 회전은 1 rad/s 이하라 킥 순간에만 걸린다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(torch.clamp(robot.data.root_ang_vel_b[:, 2].abs() - limit, min=0.0))


# ============================================================================
# 6. 종류별 넘어짐 종료
# ============================================================================
def fall_by_type(
    env: "ManagerBasedRLEnv",
    kind: str,
    limit_angle: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """bad_orientation | root_height_below_minimum 을 종류로 나눈 것 (합치면 원래 두 항과 같다).

    kind: "stand" = 서서 시작한 에피소드 전부, 그 밖 = 넘겨받기 상태에서 시작 + 그 공 믿음 종류.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)) > limit_angle
    low = robot.data.root_pos_w[:, 2] < minimum_height
    fell = tilt | low
    from_bank = getattr(env, "kick_from_bank", None)
    if from_bank is None:
        from_bank = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if kind == "stand":
        return fell & ~from_bank
    ep = getattr(env, "kick_episode_type", None)
    if ep is None:
        ep = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return fell & from_bank & (ep == EP_NAMES[kind])


# ============================================================================
# 7. 킥 세기 (v1_5b, 2026-11-01) — 관측 kick_strength 자리 (이전엔 ones_obs) 에 s ∈ [0, 1]
# ----------------------------------------------------------------------------
# 예전 시도 (kick_mdp_v0.kick_power_match) 는 세기 맞춤 항을 파워 보상에 "더했다" → 정책은 늘 세게 차는
# 쪽을 고른다 (인사이드 킥 때 배운 "항을 더하면 싼 쪽으로 샌다"). std 0.6 m/s 로 폭도 넓었다.
# 여기서는 s < 1 이면 파워 보상을 목표 속도 맞춤으로 "대체" 하고 (× 인사이드 품질 그대로), 목표 초과를 벌한다.
# s = 1 은 기존 보상 그대로 (슛 유지). 목표 속도는 최대 킥에 대한 비율 — Isaac 과 MuJoCo 의 절대 공속이
# 1.8 배 다르므로 (서서 킥 2.8 vs 5.0 m/s) 실기에서 s → 거리 표를 따로 잰다.
#   frac(s) = lo + (1 − lo)·s,  v*(s) = v_full · frac(s)
# ============================================================================
def dev_of(env):
    return env.device


def strength_frac(env: "ManagerBasedRLEnv", lo: float) -> torch.Tensor:
    """frac(s) = lo + (1 − lo)·s. 커리큘럼 (sample_kick_strength_v15 의 lo_start) 을 쓰면 env.str_lo 가 lo 를 대신한다."""
    kick_mdp._ensure_kick_strength(env)
    lo = getattr(env, "str_lo", lo)
    return lo + (1.0 - lo) * env.kick_strength_cmd


def sample_kick_strength_v15(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    full_prob: float,
    lo: float,
    strike_full: float,
    eval_levels: tuple | None = None,
    lo_start: float | None = None,
    lo_step: float = 0.1,
    err_threshold: float = 0.12,
    min_samples: int = 30000,
    ema_window: float = 20000.0,
):
    """에피소드마다 s: full_prob 비율은 1.0 (최대 킥), 나머지는 U(0, 1). 강타 래치 문턱도 frac(s) 배.

    lo_start 를 주면 커리큘럼: 하한 env.str_lo 를 lo_start 에서 시작해, s < 1 이고 공을 찬 에피소드의
    상대 오차 |최고 목표방향 공속 − v*| / v* 의 EMA 가 err_threshold 아래면 lo_step 씩 내린다 (lo 까지).
    (시도 1·2: 처음부터 lo 0.35 → 목표가 지금 킥에서 너무 멀어 1000 iter 동안 거의 안 움직였다.)
    eval_levels 를 주면 env 번호 순서대로 그 값들을 돌려 쓴다 (평가용).
    """
    kick_mdp._ensure_kick_strength(env)
    dev = env.device
    n = len(env_ids)
    if not hasattr(env, "str_peak"):
        env.str_peak = torch.zeros(env.num_envs, device=dev)
    if lo_start is not None and not hasattr(env, "str_lo"):
        env.str_lo = float(lo_start)
        env.str_ema, env.str_count = 0.0, 0
    if lo_start is not None:
        # 끝난 에피소드 성적 → EMA → 하한 내리기
        s_old = env.kick_strength_cmd[env_ids]
        pk = env.str_peak[env_ids]
        ok = (s_old < 0.999) & (pk > 0.0)
        if bool(ok.any()):
            v_star = 2.8 if not hasattr(env, "str_v_full") else env.str_v_full
            tgt = v_star * (env.str_lo + (1.0 - env.str_lo) * s_old[ok])
            err = ((pk[ok] - tgt).abs() / tgt).mean().item()
            k = int(ok.sum())
            w = ema_window / (ema_window + k)
            env.str_ema = w * env.str_ema + (1.0 - w) * err if env.str_count else err
            env.str_count += k
            if env.str_count >= min_samples and env.str_ema < err_threshold and env.str_lo > lo + 1e-6:
                prev = env.str_lo
                env.str_lo = max(lo, env.str_lo - lo_step)
                print(f"[kick_strength] 커리큘럼: 하한 {prev:.2f} → {env.str_lo:.2f} (오차 EMA {env.str_ema:.3f}, "
                      f"표본 {env.str_count})", flush=True)
                env.str_count = 0
    env.str_peak[env_ids] = 0.0
    if eval_levels is not None:
        lv = torch.tensor(eval_levels, device=dev, dtype=torch.float)
        s = lv[env_ids % len(eval_levels)]
    else:
        s = torch.where(torch.rand(n, device=dev) < full_prob, torch.ones(n, device=dev),
                        torch.rand(n, device=dev))
    env.kick_strength_cmd[env_ids] = s
    cur_lo = getattr(env, "str_lo", lo)
    if not hasattr(env, "ins_min_strike_override"):
        env.ins_min_strike_override = torch.full((env.num_envs,), strike_full, device=dev)
    env.ins_min_strike_override[env_ids] = strike_full * (cur_lo + (1.0 - cur_lo) * s)


def ball_velocity_strength(
    env: "ManagerBasedRLEnv",
    v_full: float,
    lo: float,
    sigma_frac: float,
    sigma_min: float,
    lin_err: float = 0.0,
    lin_w: float = 0.0,
    soft_scale: float = 1.0,
    max_speed: float = 3.0,
    post_kick_window_s: float = 0.4,
    quality_floor: float = 0.3,
    min_strike_speed: float = 1.8,
    window_steps: int = 20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """s = 1: kick_inside_mdp.ball_velocity_inside_gated 그대로. s < 1: max_speed² · 맞춤 × 품질.

    맞춤 = (1 − lin_w)·exp(−((v − v*)/σ)²) + lin_w·max(0, 1 − |v − v*|/lin_err).
    시도 1 (lin_w 0, σ 0.15) 은 가우시안이 좁아 목표에서 1 m/s 떨어지면 보상이 0 → 방향 신호가 초과 벌점뿐이었다
    (500 iter 에 s 0 → 2.15 m/s, 목표 0.98). 직선 항이 멀리서도 기울기를 준다.

    크기를 max_speed² 로 맞춰 약한 패스도 강한 킥과 같은 보상 규모 (약하게 차면 손해라서 안 차는 일이 없게).
    σ = max(sigma_frac · v*, sigma_min). 창 게이팅·품질 배율은 원 항과 같다.
    """
    from . import kick_inside_mdp as kim

    full = kim.ball_velocity_inside_gated(env, max_speed, post_kick_window_s, quality_floor, min_strike_speed,
                                          window_steps, asset_cfg, ball_cfg)
    ball: RigidObject = env.scene[ball_cfg.name]
    proj = (ball.data.root_lin_vel_w[:, :2] * kim._target_dir_xy(env)).sum(dim=-1)
    env.str_v_full = v_full
    if not hasattr(env, "str_peak"):
        env.str_peak = torch.zeros(env.num_envs, device=dev_of(env))
    if hasattr(env, "kick_time_s"):      # 에피소드 최고 목표방향 공속 (킥 후 0.5 s 안) — 맞춤·커리큘럼·평가 공통 기준
        el = kick_clock_s(env) - env.kick_time_s
        inwin = (env.kick_time_s >= 0) & (el <= 0.5)
        env.str_peak = torch.where(inwin, torch.maximum(env.str_peak, proj), env.str_peak)
    v_star = v_full * strength_frac(env, lo)
    sigma = torch.clamp(sigma_frac * v_star, min=sigma_min)
    # 시도 4~: 매 스텝 공속이 아니라 "지금까지 최고 공속" 으로 맞춘다. 공은 창 0.4 s 동안 ~0.8 m/s 느려져서,
    # 매 스텝 기준이면 정책이 최고 공속을 목표보다 ~0.4 m/s 높게 찼다 (시도 3: s 0 → 2.61, 목표 2.24).
    err = env.str_peak - v_star
    match = torch.exp(-torch.square(err / sigma))
    if lin_w > 0.0:
        match = (1.0 - lin_w) * match + lin_w * torch.clamp(1.0 - err.abs() / lin_err, min=0.0)
    match = soft_scale * (max_speed ** 2) * match       # 시도 5~: 0.5 — 정밀 맞춤 압력이 안정성을 밀어내지 않게
    mult = quality_floor + (1.0 - quality_floor) * env.ins_quality
    soft = match * mult
    if hasattr(env, "kick_time_s"):
        elapsed = kick_clock_s(env) - env.kick_time_s
        past = (env.kick_time_s >= 0) & (elapsed > post_kick_window_s)
        soft = torch.where(past, torch.zeros_like(soft), soft)
        soft = torch.where(env.kick_time_s < 0, torch.zeros_like(soft), soft)   # 킥 전 (공 정지) 엔 0
    return torch.where(env.kick_strength_cmd >= 0.999, full, soft)


def ball_overspeed_strength(
    env: "ManagerBasedRLEnv",
    target_full: float,
    v_full: float,
    lo: float,
    margin: float,
    post_kick_window_s: float = 0.4,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """(v_proj − 목표)₊² — s = 1 은 기존 target_full (3.0), s < 1 은 v* + margin. 음수 weight."""
    kick_mdp._ensure_target_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    proj = (ball.data.root_lin_vel_w[:, :2] * env.kick_target_dir_w[:, :2]).sum(dim=-1)
    tgt = torch.where(env.kick_strength_cmd >= 0.999, torch.full_like(proj, target_full),
                      v_full * strength_frac(env, lo) + margin)
    pen = torch.square(torch.clamp(proj - tgt, min=0.0))
    if hasattr(env, "kick_time_s"):
        elapsed = kick_clock_s(env) - env.kick_time_s
        active = (env.kick_time_s >= 0) & (elapsed <= post_kick_window_s)
        pen = torch.where(active, pen, torch.zeros_like(pen))
    return pen


def fall_after_kick(
    env: "ManagerBasedRLEnv",
    window_s: float,
    limit_angle: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """공을 찬 뒤 window_s 안에 넘어지면 (종료 조건과 같은 판정) 그 스텝에 1.0 — 큰 음수 weight (보상엔 dt 0.02 가 곱해진다).

    킥 보상 (~360 / 킥) 은 킥 순간에 받고, 그 뒤 넘어져 잃는 건 alive·성공 보너스 정도라 작다 → 세기 맞춤처럼
    킥에 압력을 더하면 정책이 안정성을 판다 (시도 4: Isaac 넘어짐 6~7 → 9~10 %, MuJoCo 공 +4 cm 0.5 → 8.8 %).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)) > limit_angle
    fell = tilt | (robot.data.root_pos_w[:, 2] < minimum_height)
    if not hasattr(env, "kick_time_s"):
        return torch.zeros(env.num_envs, device=env.device)
    el = kick_clock_s(env) - env.kick_time_s
    return (fell & (env.kick_time_s >= 0) & (el <= window_s)).float()

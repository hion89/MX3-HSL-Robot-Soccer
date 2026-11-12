# /home/roboser/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/commands/velocity_command.py
#
# 이 파일은 기존 UniformLevelVelocityCommandCfg는 그대로 두고,
# DeadbandVelocityCommand / DeadbandVelocityCommandCfg를 추가한다.

from __future__ import annotations

import torch
from dataclasses import MISSING

# IsaacLab 본체 import
# Cfg는 isaaclab.envs.mdp에서 직접 가져올 수 있음 (기존 패턴 유지)
from isaaclab.envs.mdp import UniformVelocityCommandCfg
# UniformVelocityCommand 클래스 자체는 직접 import 필요
# 경로: isaaclab/envs/mdp/commands/velocity_command.py
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse


# =====================================================================
# 기존 UniformLevelVelocityCommandCfg — 그대로 유지
# =====================================================================
@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """UniformVelocityCommandCfg + curriculum이 사용하는 limit_ranges.
    
    class_type을 지정하지 않으므로 부모의 UniformVelocityCommand가 그대로 사용된다.
    """
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


# =====================================================================
# DeadbandVelocityCommand — 새로 추가
# =====================================================================
# IsaacLab UniformVelocityCommand 소스 분석 결과 (velocity_command.py):
#
#   1) self.vel_command_b: torch.Tensor, shape [num_envs, 3] = [vx, vy, wz]
#      - public attribute (line 86)
#      - command property가 이것을 그대로 반환 → reward/obs 모두 이 값을 봄
#
#   2) _resample_command(env_ids)에서:
#      - ranges에서 vx, vy, wz를 균등 샘플링하여 vel_command_b에 기록
#      - is_standing_env 마스크를 갱신하지만, vel_command_b를 0으로 만들지는 않음
#        (standing 강제는 _update_command에서 수행)
#
#   3) _update_command()는 매 step 호출되며:
#      - heading_command=True면 wz를 heading error로 재계산
#      - is_standing_env가 True인 env의 vel_command_b를 0으로 강제
#
# Deadband 적용 전략:
#   KHUBER config는 heading_command=False이고 resampling_time_range=(10, 10)이므로
#   _resample_command 시점에 deadband를 한 번 적용하면 다음 resample까지 유지된다.
#   _update_command에서 standing envs의 cmd를 0으로 만드는 것은 deadband와 충돌하지
#   않는다 (둘 다 cmd를 작거나 0으로 만드는 방향이므로).
#
#   단, 안전을 위해 _update_command도 오버라이드하여 매 step 한 번 더 deadband를
#   적용한다. 이는 비용이 거의 없고(벡터화 연산), heading_command=True 사용
#   가능성이나 향후 IsaacLab 변경에 대해 robust하게 동작한다.
# =====================================================================
class DeadbandVelocityCommand(UniformVelocityCommand):
    """샘플링된 velocity command 각 성분에 deadband를 적용하는 wrapper.

    배경:
        ranges에서 균등 샘플링된 cmd 성분 중 절댓값이 너무 작은 값
        (예: vx=0.03 m/s)은 KHUBER 스케일(다리 30cm, 보행 주기 ~0.5s,
        stride 5cm 기준 최저 의미 속도 ≈ 0.10 m/s)에서
        물리적으로 추적이 불가능하다.

        이런 어중간한 명령이 학습 데이터에 일부 비율로 들어가면
        정책이 "걸을지 설지"를 명확히 학습하지 못해 cmd가 작은
        영역에서 떨림(jitter)이 발생한다.

    동작:
        cmd 성분(vx, vy, wz)의 절댓값이 deadband 미만이면 0으로 강제.
        cmd_norm 기준이 아니라 성분별로 자르는 이유: 
        예) (vx=0.02, vy=0.02, wz=0.5)는 cmd_norm=0.5지만 
        vx/vy는 추적 불가능한 noise이므로 따로 잘라야 한다.

    적용 시점:
        - _resample_command 직후: 새로 샘플링된 cmd에 deadband 적용
        - _update_command 끝에: 매 step 한 번 더 safety net으로 적용
          (heading_command=True인 경우 wz가 재계산되므로 필요)
    """

    cfg: "DeadbandVelocityCommandCfg"

    def _apply_deadband(self, env_ids=None):
        """벡터화된 deadband 적용.

        Args:
            env_ids: 적용할 env 인덱스. None이면 모든 env에 적용.
        """
        # cfg.deadband가 (0,0,0)이면 아무 효과 없으므로 빠르게 종료
        db = self.cfg.deadband
        if db == (0.0, 0.0, 0.0):
            return

        # device/dtype을 vel_command_b와 동일하게 맞춘 deadband 텐서
        # shape: [3] — broadcasting으로 [N, 3]과 비교 가능
        db_tensor = torch.tensor(
            db,
            device=self.vel_command_b.device,
            dtype=self.vel_command_b.dtype,
        )

        if env_ids is None:
            # 전체 env에 적용
            mask = torch.abs(self.vel_command_b) < db_tensor   # [N, 3]
            # 마스크된 위치는 0으로 (in-place로 vel_command_b 수정)
            # torch.where 대신 mask 인덱싱을 쓰는 이유: in-place 보장
            self.vel_command_b[mask] = 0.0
        else:
            # 지정된 env_ids에만 적용
            # env_ids가 list/sequence면 텐서로 변환
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, device=self.vel_command_b.device)
            sub = self.vel_command_b[env_ids]                   # [K, 3]
            mask = torch.abs(sub) < db_tensor                   # [K, 3]
            sub = torch.where(mask, torch.zeros_like(sub), sub)
            self.vel_command_b[env_ids] = sub

    def _resample_command(self, env_ids):
        # 1) 부모 클래스의 표준 균등 샘플링 수행
        #    - ranges에서 vx, vy, wz 샘플링하여 vel_command_b에 기록
        #    - is_standing_env 마스크 갱신 (vel_command_b는 아직 0으로 안 만듦)
        super()._resample_command(env_ids)

        # 2) 새로 샘플링된 환경에만 deadband 적용
        #    여기서 적용하면 다음 resample(10초 후)까지 유지됨
        self._apply_deadband(env_ids)

    def _update_command(self):
        # 1) 부모 클래스의 표준 update 수행
        #    - heading_command=True면 wz를 heading error로 재계산
        #    - is_standing_env가 True인 env의 vel_command_b를 0으로 강제
        super()._update_command()

        # 2) safety net: 매 step 모든 env에 deadband 한 번 더 적용
        #    - heading_command=True 사용 시 재계산된 wz에도 deadband 보장
        #    - 비용은 미미 (벡터화된 in-place 마스킹)
        #    - standing env의 cmd는 이미 0이므로 deadband가 추가 영향 없음
        self._apply_deadband(env_ids=None)


# =====================================================================
# DeadbandVelocityCommandCfg
# =====================================================================
@configclass
class DeadbandVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """DeadbandVelocityCommand의 config.

    UniformLevelVelocityCommandCfg를 상속받아 limit_ranges(curriculum용)는
    그대로 유지하고, deadband와 class_type만 새로 지정.

    Attributes:
        deadband: (vx_db, vy_db, wz_db) — 각 성분별 deadband 절댓값.
            성분의 절댓값이 이 값 미만이면 0으로 강제된다.

            KHUBER 권장값 근거:
                vx_db = 0.10 m/s
                    stride 5cm × 보행 주기 0.5s 기준 한 걸음 최소 이동
                vy_db = 0.08 m/s
                    횡이동은 stride가 더 작아도 됨
                wz_db = 0.20 rad/s
                    한 보행 사이클(0.5s)에 약 5.7° 회전

            보수적 시작값: (0.05, 0.05, 0.10) — 기존 분포와 큰 차이 없음
            적극적 값:    (0.10, 0.08, 0.20) — 셔플링/떨림 강하게 억제
            끈 상태:      (0.0, 0.0, 0.0) — 동작은 부모 클래스와 완전 동일
    """
    # CommandManager가 인스턴스화할 클래스 지정 — 이게 핵심
    class_type: type = DeadbandVelocityCommand

    # 성분별 deadband. (0.0, 0.0, 0.0)이면 효과 없음 (기본 동작과 동일)
    deadband: tuple[float, float, float] = (0.0, 0.0, 0.0)


# =====================================================================
# ForwardToBallSpeedCommand — walk_kick_v2 "일반 보행 접근"
# =====================================================================
# 설계: cmd_vel 은 "전진 속도(vx)만" 준다. 공까지 거리 기반으로 페이스를
# 정하고(멀면 max_speed, 근거리 kick_range 에서 0), vy·wz 는 0 으로 둔다
# (env 에서 이 둘을 "추종하지 않으므로" 정책이 알아서 몸을 틀어 공을 향함).
#   - 전진 속도를 추종(track)하면 버스트 없는 "일정한 차분한 걸음" 이 나온다
#     (기존 speed_cap 은 상한만 걸어 버스트 허용 → 성큼성큼).
#   - 근거리서 vx→0 → 발 딛고 킥. 킥 후 / reset_hold 엔 0 (공 안 쫓음).
# ※ 공 위치를 읽으므로 배포 시 공 추적 필요(= 이미 알려진 sim2real 나중 문제).
# =====================================================================
class ForwardToBallSpeedCommand(UniformVelocityCommand):
    """전진 속도(vx)만 명령: 공까지 거리 기반. vy=wz=0."""

    cfg: "ForwardToBallSpeedCommandCfg"

    def _update_command(self):
        env = self._env
        c = self.cfg
        robot = env.scene[c.asset_name]
        ball = env.scene[c.ball_name]
        d = torch.norm(
            ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
        )
        vx = ((d - c.kick_range) / c.ramp).clamp(0.0, 1.0) * c.max_speed

        # 킥 후 / reset_hold 엔 전진 0 (굴러가는 공 안 쫓음 / 리셋 안정화)
        kt = getattr(env, "kick_time_s", None)
        if kt is not None:
            vx = torch.where(kt >= 0, torch.zeros_like(vx), vx)
        hold = getattr(env, "reset_hold_steps", None)
        if hold is not None:
            vx = torch.where(hold > 0, torch.zeros_like(vx), vx)

        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = 0.0


@configclass
class ForwardToBallSpeedCommandCfg(UniformLevelVelocityCommandCfg):
    """ForwardToBallSpeedCommand 설정."""

    class_type: type = ForwardToBallSpeedCommand
    ball_name: str = "ball"
    max_speed: float = 0.5      # 먼 구간 전진 페이스
    kick_range: float = 0.4     # 이 거리 안이면 vx=0 (발 딛고 킥)
    ramp: float = 0.3           # kick_range~kick_range+ramp 에서 0→max


# =====================================================================
# WalkToBallCommand — BallFollow-V0 "공을 바라보며 걸어가기"
# =====================================================================
# ForwardToBallSpeedCommand(전진 vx 만) 에 heading 제어를 추가한다.
#   · wz = k_yaw · (공 방위각)      → 몸을 틀어 공을 바라봄
#   · vx = 거리기반 × cos(방위각)⁺  → 공을 향할 때만 전진 (먼저 틀고 걸음)
#   · vy = 0
# v2 의 "vx 만 명령 + wz 자유" 는 조향 신호가 없어 방황했다. 여기서는
# 명령 자체가 공을 가리키고, 정책은 track_lin_vel_xy + track_ang_vel_z 로
# 이 명령을 추종 → 공으로 곧장 곡선 접근, 근거리(kick_range)서 정지.
# ※ 공 위치를 읽음(배포 시 외부 공좌표 필요 = sim2real 검증 대상).
# =====================================================================
class WalkToBallCommand(ForwardToBallSpeedCommand):
    """전진 vx(거리기반) + wz(공을 바라보게) 명령."""

    cfg: "WalkToBallCommandCfg"

    def _update_command(self):
        env = self._env
        c = self.cfg
        robot = env.scene[c.asset_name]
        ball = env.scene[c.ball_name]
        rel_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
        rel_b = quat_apply_inverse(robot.data.root_quat_w, rel_w)  # 로봇 프레임
        d = torch.norm(rel_b[:, :2], dim=-1)
        heading = torch.atan2(rel_b[:, 1], rel_b[:, 0])  # 공 방위각(정면=0)

        vx = ((d - c.kick_range) / c.ramp).clamp(0.0, 1.0) * c.max_speed
        vx = vx * torch.cos(heading).clamp(min=0.0)            # 공 향할 때만 전진
        wz = (c.k_yaw * heading).clamp(-c.wz_max, c.wz_max)    # 공을 바라보게

        hold = getattr(env, "reset_hold_steps", None)
        if hold is not None:
            m = hold > 0
            vx = torch.where(m, torch.zeros_like(vx), vx)
            wz = torch.where(m, torch.zeros_like(wz), wz)

        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = wz


@configclass
class WalkToBallCommandCfg(ForwardToBallSpeedCommandCfg):
    """WalkToBallCommand 설정 (ForwardToBallSpeedCommandCfg + heading)."""

    class_type: type = WalkToBallCommand
    k_yaw: float = 1.5     # 방위각 → wz 게인
    wz_max: float = 1.0    # 최대 회전 명령

# =====================================================================
# TerrainCappedVelocityCommand — RS01 V0-3 2 차 "어려운 지형에선 느린 명령만"
# =====================================================================
# 배경 (2026-10-25, RS01 V0-3 1 차 model_37200):
#   블라인드 정책인데 명령은 지형과 무관하게 같은 분포에서 나왔다. 10 cm 계단 위에서도 1.2 m/s ·
#   1.5 rad/s 명령이 나오고, 그러면 넘어진다 (V0-2 를 V0-3 지형에 1.2 m/s → 계단 올라가기 행 9 100 %).
#   정책은 자기가 어느 지형에 있는지 모르니 "빠른 명령 = 넘어질 위험" 으로 배워
#   평지에서도 1.2 m/s 명령에 앞으로 안 가게 됐다 (0.87 → 0.36 → −0.18 m/s 로 점점 붕괴, 회전 1.36 → 0.92).
#
# 동작: 리샘플 시점에 env 의 terrain level 로 명령을 축소한다 (곱셈 = 균등 분포 모양 유지).
#   f = clamp((level − cap_levels[0]) / (cap_levels[1] − cap_levels[0]), 0, 1)
#   성분 배율 = 1 − f·(1 − cap_scale[k])   k = 전진(vx>0) / 후진(vx<0) / 옆(vy) / 회전(wz)
#   리샘플 때 한 번만 곱한다 (매 스텝 곱하면 누적된다). 다음 리샘플(10 s)까지 유지.
#   terrain_levels 는 _reset_idx 에서 커리큘럼 → 씬 리셋 → 명령 리셋 순이라 갱신된 레벨을 본다.
#   터레인이 없으면(평지 plane) 아무것도 안 한다.
# 실기 대응: 계단을 1.2 m/s 로 오르라고 할 일은 없다 — 빠른 명령은 쉬운 바닥에서만 나온다는 가정이 현실과 맞다.
# =====================================================================
class TerrainCappedVelocityCommand(UniformVelocityCommand):
    """terrain level 이 높을수록 리샘플된 속도 명령을 축소."""

    cfg: "TerrainCappedVelocityCommandCfg"

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        terrain = getattr(self._env.scene, "terrain", None)
        levels = getattr(terrain, "terrain_levels", None) if terrain is not None else None
        if levels is None:
            return
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        lo, hi = self.cfg.cap_levels
        f = ((levels[env_ids].float() - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
        s_fwd, s_back, s_y, s_wz = self.cfg.cap_scale
        c = self.vel_command_b[env_ids]
        k_x = torch.where(c[:, 0] >= 0, 1.0 - f * (1.0 - s_fwd), 1.0 - f * (1.0 - s_back))
        c[:, 0] *= k_x
        c[:, 1] *= 1.0 - f * (1.0 - s_y)
        c[:, 2] *= 1.0 - f * (1.0 - s_wz)
        self.vel_command_b[env_ids] = c


@configclass
class TerrainCappedVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """TerrainCappedVelocityCommand 설정.

    cap_levels: (축소 시작 레벨, 최대 축소 레벨). 그 사이는 선형.
    cap_scale : 최대 축소 레벨에서의 배율 (전진, 후진, 옆, 회전).
                예) ranges 가 vx −0.5~1.2, vy ±0.4, wz ±1.5 이고 (0.5, 0.6, 0.75, 0.533) 이면
                    레벨 9 에서 vx −0.3~0.6, vy ±0.3, wz ±0.8.
    """

    class_type: type = TerrainCappedVelocityCommand
    cap_levels: tuple[float, float] = (3.0, 9.0)
    cap_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

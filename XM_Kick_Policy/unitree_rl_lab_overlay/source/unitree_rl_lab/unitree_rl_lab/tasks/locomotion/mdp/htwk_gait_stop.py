# htwk_gait_stop.py
# ============================================================================
# StoppableGaitPhaseCommand — 정지/재개 전이를 명시적으로 다루는 보행 위상.
#
# htwk_gait.GaitPhaseCommand(v1) 는 그대로 두고 새 클래스를 추가한다.
# v1 환경은 영향받지 않는다.
#
# ────────────────────────────────────────────────────────────────────────
# ■ v1(GaitPhaseCommand) 의 한계 세 가지
#
# (1) 제자리 걷기가 표현 불가능
#     v1 은 속도 명령으로 주파수를 강제 게이팅한다.
#         standing = norm(vel_cmd) < 0.1
#         freq = 0 if standing else frequency
#     즉 "속도 0" 이면 무조건 완전 정지가 되어, 공을 기다리며 리듬을
#     유지하다 즉시 출발하는 동작(marching in place)을 학습할 수 없다.
#     축구에서는 완전히 멈췄다 출발하면 첫 스텝이 굼뜨다.
#
# (2) 정지 시 위상이 얼어붙음
#     v1 은 freq=0 이면 phase 가 그 자리에 멈춘다. 정지 구간에서 정책은
#     위상을 못 보므로 편한 중립 자세로 서는데, 재개 시 위상은 멈췄던
#     지점(예: 0.3, 스윙 직후)에서 이어진다. 다리 자세와 위상이 어긋난
#     채로 보행이 시작되어 첫 한두 스텝이 어색해진다.
#
# (3) 정지 과도 구간이 정책에 보이지 않음
#     freq 가 0 이 되는 순간 관측이 (0,0) 으로 게이팅되므로, 정책 입장에서는
#     돌던 위상이 예고 없이 뚝 끊긴다. 감속을 준비할 근거가 없다.
#
# ────────────────────────────────────────────────────────────────────────
# ■ 이 클래스의 설계
#
# (A) 주파수를 속도와 독립된 명령으로 만든다
#         속도 != 0, 주파수 > 0   -> 보행
#         속도 == 0, 주파수 > 0   -> 제자리 걷기   (v1 에서 불가능했던 상태)
#         속도 == 0, 주파수 == 0  -> 완전 정지
#     속도 != 0 인데 주파수 0 인 모순 조합은 _update_command 에서 차단한다.
#
# (B) 주파수는 즉시 0 으로 떨어뜨린다 (램프 없음)
#     주파수를 서서히 낮추면 "다리를 천천히 움직여라" 라는 신호가 되어
#     "빨리 멈춰라" 와 모순된다. 감속은 위상 수렴이 담당한다.
#
# (C) 위상은 0 을 향해 최단 경로로 선형 이동한다
#         phase > 0.5 -> 정방향으로 밀어 1.0(=0) 도달
#         phase < 0.5 -> 역방향으로 당겨 0.0 도달
#     최대 이동 거리가 0.5 이므로 stop_rate=2.0 이면 0.25 초 안에 멈춘다.
#     위상 0 은 offset (0.0, 0.5) / threshold 0.55 기준 양발 stance 구간이라
#     정지 목표 자세로 물리적으로 타당하다.
#
# (D) cos/sin 게이팅을 제거하고 주파수를 관측에 넣는다
#     v1 의 게이팅은 "정지 상태" 를 (0,0) 이라는 단위원 밖 값으로 표시해
#     보행 중 위상과 구분하는 장치였다. 그런데 게이팅이 걸리면 정지 과도
#     구간의 위상 수렴이 정책에 보이지 않아 (3) 문제가 생긴다.
#
#     주파수를 관측에 포함하면 구분 역할을 주파수가 대신한다.
#         보행 중 위상 0 : (cos, sin, f) = (1, 0, 1.0)
#         완전 정지      : (cos, sin, f) = (1, 0, 0.0)
#     따라서 게이팅 없이도 모호하지 않고, 과도 구간 위상도 그대로 보인다.
#
#     ※ include_frequency=False 로 두면 이 구분이 깨지므로, 그때는
#       하위 호환을 위해 v1 방식 게이팅으로 자동 폴백한다.
#       (정지 과도 설계를 쓰려면 include_frequency=True 가 필수)
#
# ────────────────────────────────────────────────────────────────────────
# ■ 실기 배포 시 반드시 동일하게 구현할 것
#     - 위상은 증분 적분 (time_now * freq 방식 금지 — 주파수 변경 시 점프)
#     - 정지 명령 시 위상을 0 으로 리셋하지 말고 최단 경로로 수렴시킬 것
#     - 관측의 주파수 성분은 frequency_scale 로 나눈 값
# ============================================================================

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StoppableGaitPhaseCommand(CommandTerm):
    """정지/재개 전이를 명시적으로 다루는 보행 위상 명령.

    command 텐서 (include_frequency=True 기준 3 차원):
        [:, 0] cos(2*pi*phase)
        [:, 1] sin(2*pi*phase)
        [:, 2] frequency / frequency_scale

    외부 참조용 public 속성:
        phase                [N] 현재 위상 (0~1)
        frequency            [N] 샘플링된 주파수 명령
        effective_frequency  [N] 모순 차단 후 실제 적용된 주파수
        is_walking           [N] bool. 주파수 > 0 (보행 또는 제자리 걷기)
        is_stopping          [N] bool. 정지 과도 구간 (위상이 0 으로 수렴 중)
        phase_active         [N] bool. is_walking | is_stopping
                                 = 위상이 움직이고 있는가
    """

    cfg: StoppableGaitPhaseCommandCfg

    def __init__(self, cfg: StoppableGaitPhaseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.frequency = torch.zeros(self.num_envs, device=self.device)
        self.effective_frequency = torch.zeros(self.num_envs, device=self.device)

        # 정지 선호 비트. 속도 명령이 0 일 때만 의미를 갖는다.
        # True  -> 완전 정지
        # False -> 제자리 걷기
        self._want_stop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.is_walking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_stopping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._num_dims = 3 if self.cfg.include_frequency else 2
        self._command = torch.zeros(self.num_envs, self._num_dims, device=self.device)

        lo, hi = self.cfg.frequency_range
        self.frequency.uniform_(lo, hi)

        self.metrics["gait_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ratio_walking"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ratio_stopping"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ratio_full_stop"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "StoppableGaitPhaseCommand:\n"
        msg += f"\tfrequency_range: {self.cfg.frequency_range} Hz\n"
        msg += f"\tstop_rate: {self.cfg.stop_rate} phase/s "
        msg += f"(최대 정지 시간 {0.5 / self.cfg.stop_rate:.3f} s)\n"
        msg += f"\trel_stop_when_standing: {self.cfg.rel_stop_when_standing}\n"
        msg += f"\tinclude_frequency: {self.cfg.include_frequency}\n"
        if not self.cfg.include_frequency:
            msg += "\t[경고] include_frequency=False -> v1 방식 게이팅으로 폴백.\n"
            msg += "\t       정지 과도 구간이 정책에 보이지 않는다.\n"
        return msg

    # ------------------------------------------------------------------
    # CommandTerm 인터페이스
    # ------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        lo, hi = self.cfg.frequency_range
        self.frequency[env_ids] = torch.empty(n, device=self.device).uniform_(lo, hi)

        # 정지 선호 비트 샘플링.
        # 속도 명령이 0 인 env 중 이 비율만큼이 완전 정지가 되고,
        # 나머지는 제자리 걷기가 된다.
        r = torch.rand(n, device=self.device)
        self._want_stop[env_ids] = r < self.cfg.rel_stop_when_standing

        # 리셋 직후에는 위상 0 에서 출발한다. 실기 기동 시에도 동일하므로
        # 학습 분포와 배포 분포가 일치한다.
        if self.cfg.reset_phase_on_resample:
            self.phase[env_ids] = 0.0

    def _update_command(self):
        dt = self._env.step_dt

        # ── 1) 속도 명령 확인 ────────────────────────────────────────
        standing = self._standing_mask()

        # ── 2) 실효 주파수 결정 ──────────────────────────────────────
        freq = self.frequency.clone()

        # 속도 != 0 인데 주파수 0 인 모순 조합 차단.
        # (그런 조합은 물리적으로 불가능하고, 학습되면 정책이 혼란스럽다)
        lo = self.cfg.frequency_range[0]
        freq = torch.where(freq < self.cfg.eps, torch.full_like(freq, lo), freq)

        # 속도 0 이면서 정지 선호인 env 만 완전 정지
        full_stop = standing & self._want_stop
        freq = torch.where(full_stop, torch.zeros_like(freq), freq)

        # reset hold 구간에서는 위상을 정지시킨다.
        # (hold 동안 액션이 마스킹되므로 위상만 흘러가면 정책이 보는
        #  위상과 실제 다리 상태가 어긋난 채로 학습이 시작된다)
        if self.cfg.freeze_during_reset_hold:
            hold = getattr(self._env, "reset_hold_steps", None)
            if hold is not None:
                freq = torch.where(hold > 0, torch.zeros_like(freq), freq)

        self.effective_frequency = freq
        self.is_walking = freq > self.cfg.eps

        # ── 3) 위상 갱신 ────────────────────────────────────────────
        # (a) 보행 중: 정방향 증분 적분
        phase_walk = torch.fmod(self.phase + dt * freq, 1.0)

        # (b) 정지 중: 0 을 향해 최단 경로로 선형 이동
        step = self.cfg.stop_rate * dt
        # 정방향 거리 = 1 - phase, 역방향 거리 = phase
        go_forward = self.phase > 0.5
        # clamp 로 목표를 넘어가지 않게 하여 정확히 0 에 착지시킨다
        d_fwd = torch.clamp(1.0 - self.phase, max=step)
        d_bwd = torch.clamp(self.phase, max=step)
        phase_fwd = torch.fmod(self.phase + d_fwd, 1.0)   # 1.0 도달 시 0.0
        phase_bwd = self.phase - d_bwd
        phase_stop = torch.where(go_forward, phase_fwd, phase_bwd)

        self.phase = torch.where(self.is_walking, phase_walk, phase_stop)

        # 정지 과도 구간 = 보행 중이 아니면서 위상이 아직 0 이 아님
        self.is_stopping = (~self.is_walking) & (self.phase > self.cfg.eps)
        self.phase_active = self.is_walking | self.is_stopping

        # ── 4) 관측 생성 ────────────────────────────────────────────
        two_pi = 2.0 * math.pi
        cos_p = torch.cos(two_pi * self.phase)
        sin_p = torch.sin(two_pi * self.phase)

        if self.cfg.include_frequency:
            # 게이팅 없음. 정지 상태는 주파수 성분 0 으로 구분된다.
            # 덕분에 정지 과도 구간의 위상 수렴이 정책에 그대로 보인다.
            self._command[:, 0] = cos_p
            self._command[:, 1] = sin_p
            self._command[:, 2] = freq / self.cfg.frequency_scale
        else:
            # 하위 호환 폴백: v1 과 동일한 게이팅.
            # 정지 과도 구간이 보이지 않으므로 권장하지 않는다.
            gate = self.is_walking.float()
            self._command[:, 0] = cos_p * gate
            self._command[:, 1] = sin_p * gate

    def _update_metrics(self):
        self.metrics["gait_frequency"] = self.effective_frequency.clone()
        self.metrics["ratio_walking"] = self.is_walking.float()
        self.metrics["ratio_stopping"] = self.is_stopping.float()
        self.metrics["ratio_full_stop"] = (
            (~self.is_walking) & (~self.is_stopping)
        ).float()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------
    def _standing_mask(self) -> torch.Tensor:
        """속도 명령 노름이 임계값 미만인지."""
        name = self.cfg.velocity_command_name
        if name is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            cmd = self._env.command_manager.get_command(name)
        except (KeyError, AttributeError):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.norm(cmd[:, :3], dim=1) < self.cfg.standing_threshold


@configclass
class StoppableGaitPhaseCommandCfg(CommandTermCfg):
    """StoppableGaitPhaseCommand 설정."""

    class_type: type = StoppableGaitPhaseCommand

    resampling_time_range: tuple[float, float] = (10.0, 10.0)

    # 보행 주파수 범위 [Hz]. 좁게 시작할 것.
    # 처음부터 (1.5, 2.5) 로 벌리면 정책이 양 극단을 커버하려다
    # 중간이 어정쩡해질 수 있다.
    frequency_range: tuple[float, float] = (1.8, 2.2)

    # 관측에 주파수를 포함할지.
    # ★ 이 클래스의 정지 설계를 쓰려면 True 가 필수다.
    #   False 면 게이팅 폴백이 걸려 정지 과도 구간이 정책에 보이지 않는다.
    include_frequency: bool = True

    # 주파수 관측 정규화 계수. f / frequency_scale 이 O(1) 이 되도록.
    frequency_scale: float = 2.0

    # 정지 시 위상이 0 으로 수렴하는 속도 [phase/s].
    # 최대 이동 거리가 0.5 이므로 최대 정지 시간 = 0.5 / stop_rate.
    #   2.0 -> 0.25 s   (보행 주파수와 비슷한 속도, 권장)
    #   4.0 -> 0.125 s  (급정지. 너무 빠르면 다리가 튀어 넘어질 수 있음)
    stop_rate: float = 2.0

    # 정지 판정에 쓸 velocity command term 이름.
    velocity_command_name: str | None = "base_velocity"

    # 속도 명령 노름이 이 값 미만이면 "속도 0" 으로 본다.
    standing_threshold: float = 0.1

    # 속도 0 인 env 중 완전 정지가 되는 비율. 나머지는 제자리 걷기.
    # velocity term 의 rel_standing_envs=0.15 와 곱해져
    #   완전 정지  ≈ 0.15 * 0.6 = 9 %
    #   제자리걷기 ≈ 0.15 * 0.4 = 6 %
    # 가 된다.
    rel_stop_when_standing: float = 0.6

    # reset 시 위상을 0 으로 되돌릴지.
    reset_phase_on_resample: bool = True

    # reset hold 구간에서 위상을 얼릴지.
    freeze_during_reset_hold: bool = True

    # 부동소수 비교 임계값
    eps: float = 1.0e-6

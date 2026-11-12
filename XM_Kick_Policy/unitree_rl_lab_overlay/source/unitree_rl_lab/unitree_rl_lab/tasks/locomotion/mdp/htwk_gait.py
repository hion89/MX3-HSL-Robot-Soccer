# htwk_gait.py
# ============================================================================
# 보행 위상(gait phase)을 "상태"로 유지하는 CommandTerm.
#
# ── 왜 필요한가 ──
# 기존 mdp.feet_gait / observations.gait_phase 는 위상을
#     phase = (episode_length_buf * step_dt) % period / period
# 로 매 스텝 재계산한다. 이 방식의 문제:
#
#   1) 주기(period)가 상수로 고정 → 보행 주파수를 명령으로 줄 수 없다.
#   2) 관측에 넣더라도 리워드가 참조하는 위상과 별도로 계산되어
#      두 곳의 정의가 어긋날 위험이 있다.
#   3) 실기 배포 시 "절대 시각 × 주파수" 로 계산하면 주파수를 바꾸는 순간
#      위상이 점프한다. (HTWK 배포 코드 policy_thomas.py 의 실제 버그)
#      학습은 증분 적분이므로 배포도 반드시 증분 적분이어야 한다.
#
# 이 CommandTerm 은 위상을 env 별 버퍼로 들고 매 스텝 "증분 적분" 한다.
#     phase <- fmod(phase + step_dt * frequency, 1.0)
# 리워드와 관측이 동일한 term 을 참조하므로 정의가 갈릴 일이 없고,
# 실기 배포 코드도 이 한 줄만 그대로 옮기면 된다.
#
# ── 정지(standing) 처리 ──
# HTWK 와 동일하게, 속도 명령이 거의 0 이면 주파수를 0 으로 만들어
#   - 위상 적분이 멈추고 (phase 가 그 자리에 얼어붙음)
#   - 관측의 cos/sin 이 0 으로 게이팅된다
# "정지 = 위상 없음" 이라는 신호를 정책에 명확히 준다.
#
# ── 관측 차원 ──
#   include_frequency=False → 2 차원 [cos, sin]
#   include_frequency=True  → 3 차원 [cos, sin, f_norm]
# ============================================================================

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GaitPhaseCommand(CommandTerm):
    """보행 위상 + (선택적) 보행 주파수 명령.

    command 텐서 구성:
        [:, 0] = cos(2*pi*phase) * gate
        [:, 1] = sin(2*pi*phase) * gate
        [:, 2] = frequency / frequency_scale   (include_frequency=True 일 때만)

    gate 는 (실효 주파수 > 0) 여부. 정지 상태에서 0 이 된다.

    외부에서 참조 가능한 public 속성:
        self.phase                 [N]  현재 위상 (0~1)
        self.frequency             [N]  샘플링된 주파수 (정지 게이팅 전)
        self.effective_frequency   [N]  정지 게이팅 후 실제 적분에 쓰인 주파수
    """

    cfg: GaitPhaseCommandCfg

    def __init__(self, cfg: GaitPhaseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 위상 / 주파수 버퍼
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.frequency = torch.zeros(self.num_envs, device=self.device)
        self.effective_frequency = torch.zeros(self.num_envs, device=self.device)

        # 출력 차원: cos, sin (+ frequency)
        self._num_dims = 3 if self.cfg.include_frequency else 2
        self._command = torch.zeros(self.num_envs, self._num_dims, device=self.device)

        # 초기 주파수 샘플링 (reset 전에도 유효한 값이 들어있도록)
        lo, hi = self.cfg.frequency_range
        self.frequency.uniform_(lo, hi)

        # 텐서보드 지표 — 학습 중 주파수/정지비율 추적용
        self.metrics["gait_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gait_standing_ratio"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "GaitPhaseCommand:\n"
        msg += f"\tfrequency_range: {self.cfg.frequency_range} Hz\n"
        msg += f"\tinclude_frequency: {self.cfg.include_frequency}\n"
        msg += f"\tcommand dims: {self._num_dims}\n"
        return msg

    # ------------------------------------------------------------------
    # CommandTerm 인터페이스
    # ------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids):
        """주파수 재샘플링. frequency_range 의 min==max 이면 사실상 고정값."""
        if len(env_ids) == 0:
            return
        lo, hi = self.cfg.frequency_range
        n = len(env_ids)
        self.frequency[env_ids] = torch.empty(n, device=self.device).uniform_(lo, hi)

        # 리셋된 env 는 위상을 0 에서 다시 시작 (에피소드 간 위상 연속성 제거).
        # 실기에서도 보행 시작 시 위상 0 부터 출발하므로 분포가 일치한다.
        if self.cfg.reset_phase_on_resample:
            self.phase[env_ids] = 0.0

    def _update_command(self):
        """매 스텝 호출. 위상을 증분 적분하고 관측 텐서를 갱신한다.

        주의: CommandManager 는 cfg 에 선언된 순서대로 term 을 갱신한다.
              velocity_command_name 이 가리키는 term(base_velocity)이
              이 term 보다 "먼저" 선언되어 있어야 정지 판정이 같은 스텝의
              값을 본다. env_cfg 의 CommandsCfg 선언 순서를 지킬 것.
        """
        standing = self._standing_mask()

        # 정지 env 는 주파수 0 → 위상 정지
        freq = torch.where(standing, torch.zeros_like(self.frequency), self.frequency)

        # reset hold 구간에서도 위상을 정지시킨다.
        # (hold 동안 액션이 마스킹되므로 위상만 흘러가면 정책이 보는
        #  위상과 실제 다리 상태가 어긋난 채로 학습이 시작된다)
        if self.cfg.freeze_during_reset_hold:
            hold = getattr(self._env, "reset_hold_steps", None)
            if hold is not None:
                freq = torch.where(hold > 0, torch.zeros_like(freq), freq)

        self.effective_frequency = freq

        # ── 증분 적분 (실기 배포와 반드시 동일한 식) ──
        self.phase = torch.fmod(self.phase + self._env.step_dt * freq, 1.0)

        gate = (freq > 1.0e-8).float()
        two_pi = 2.0 * math.pi
        self._command[:, 0] = torch.cos(two_pi * self.phase) * gate
        self._command[:, 1] = torch.sin(two_pi * self.phase) * gate
        if self.cfg.include_frequency:
            self._command[:, 2] = freq / self.cfg.frequency_scale

    def _update_metrics(self):
        self.metrics["gait_frequency"] = self.effective_frequency.clone()
        self.metrics["gait_standing_ratio"] = (self.effective_frequency < 1.0e-8).float()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------
    def _standing_mask(self) -> torch.Tensor:
        """속도 명령 노름이 임계값 미만이면 정지로 판정."""
        name = self.cfg.velocity_command_name
        if name is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            cmd = self._env.command_manager.get_command(name)
        except (KeyError, AttributeError):
            # velocity term 이 없으면 정지 판정을 하지 않는다 (항상 보행)
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.norm(cmd[:, :3], dim=1) < self.cfg.standing_threshold


@configclass
class GaitPhaseCommandCfg(CommandTermCfg):
    """GaitPhaseCommand 설정."""

    class_type: type = GaitPhaseCommand

    # 주파수 재샘플링 주기 [s]. 속도 명령과 맞추는 것을 권장.
    resampling_time_range: tuple[float, float] = (10.0, 10.0)

    # 보행 주파수 범위 [Hz]. (2.0, 2.0) 이면 고정 2 Hz = 주기 0.5 s.
    # 1 단계에서는 고정으로 시작하고, 안정되면 (1.6, 2.6) 등으로 넓힌다.
    frequency_range: tuple[float, float] = (2.0, 2.0)

    # True 면 command 에 정규화된 주파수를 3 번째 성분으로 포함한다.
    # 주파수를 고정으로 쓸 때는 False (상수를 관측에 넣어봐야 무의미).
    include_frequency: bool = False

    # 주파수 관측 정규화 계수. f / frequency_scale 이 대략 O(1) 이 되도록.
    frequency_scale: float = 2.0

    # 정지 판정에 쓸 velocity command term 이름. None 이면 정지 판정 없음.
    velocity_command_name: str | None = "base_velocity"

    # 속도 명령 노름이 이 값 미만이면 정지로 본다 [m/s 및 rad/s 혼합 노름].
    standing_threshold: float = 0.1

    # reset 시 위상을 0 으로 되돌릴지 여부.
    reset_phase_on_resample: bool = True

    # reset hold 구간에서 위상을 얼릴지 여부.
    freeze_during_reset_hold: bool = True

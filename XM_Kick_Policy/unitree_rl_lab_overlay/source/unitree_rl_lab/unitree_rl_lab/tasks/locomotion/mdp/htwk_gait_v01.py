# htwk_gait_v01.py
# ============================================================================
# PhaseOnlyGaitCommand — v0 에 "보행 위상만" 얹기 위한 최소 CommandTerm.
#
# htwk_gait.py(v1) / htwk_gait_stop.py(v1s) 는 건드리지 않는다.
# 세 파일이 독립적으로 공존하며 각각의 환경에서만 쓰인다.
#
# ────────────────────────────────────────────────────────────────────────
# ■ 왜 또 만드는가
#
# v1 은 v0 대비 8~9 가지를 한꺼번에 바꿨다(위상, DR 확대, EMA 추종,
# 스탠스 폭, 태스크공간 비대칭, feet_swing, 크리틱 DR 관측, actor 축소).
# 정지 상태에서 5.45Hz yaw 발진이 생겼는데, v0 는 멀쩡하므로 원인이
# 이 변경분 어딘가에 있다. 그런데 한꺼번에 바뀌어 분리가 불가능하다.
#
# 이 term 은 "위상만" 격리해서 v0 에 얹기 위한 것이다. v0_1 이 안정적이면
# 위상은 무죄이고 나머지 변경분에 원인이 있다. v0_1 도 떨면 위상이
# 방아쇠다. 어느 쪽이든 결론이 명확해진다.
#
# ────────────────────────────────────────────────────────────────────────
# ■ 설계 (v1 / v1s 와의 차이)
#
#                   v1            v1s              v0_1 (이 파일)
#   관측 차원        2             3                 2
#   정지 시 게이팅    있음(0,0)      없음              없음
#   정지 시 위상      동결          0 으로 수렴        0 으로 수렴
#   주파수           2.0 고정       1.8~2.2 명령       2.0 고정
#   주파수 관측       없음          있음              없음
#   제자리 걷기       불가          가능              불가
#
# ■ 게이팅 없이 정지를 어떻게 구분하나
#
#   v1s 는 주파수 관측이 "정지 = freq 0" 을 알려줬다. 여기는 주파수를
#   관측에 넣지 않으므로 그 수단이 없다. 대신 히스토리가 대신한다.
#
#       완전 정지 : [(1,0), (1,0), (1,0), (1,0), (1,0)]   ← 위상 정지
#       보행 중   : [(0.50,-0.87), (0.71,-0.71), ...]     ← 매 프레임 변화
#
#   위상이 얼어붙었는지 흘러가는지가 5 프레임 히스토리에 그대로 보인다.
#   ObsGroup 의 history_length=5 를 전제로 하는 설계이므로, 히스토리를
#   1 로 줄이면 정지/보행 구분이 불가능해진다. 주의할 것.
#
# ■ 왜 게이팅을 빼는가
#
#   게이팅(정지 시 cos/sin 을 0 으로)은 구분을 명확히 해주지만, 정지
#   과도 구간(위상이 0 으로 수렴하는 0.25 초)에 위상이 정책에 보이지
#   않게 만든다. 그러면 정책은 "돌던 위상이 예고 없이 끊기는" 것만
#   보게 되어 감속 동작을 학습할 근거가 없다.
#
# ■ 왜 위상 0 으로 수렴시키는가 (동결이 아니라)
#
#   동결하면 재개 시 멈췄던 지점(예: 0.3, 스윙 직후)부터 이어진다.
#   정지 중 정책은 편한 중립 자세로 서 있으므로, 다리 자세와 위상이
#   어긋난 채 보행이 시작되어 첫 한두 스텝이 어색해진다.
#   위상 0 은 offset (0.0, 0.5) / threshold 0.55 기준 양발 stance 구간이라
#   정지 목표 자세로도 타당하다.
#
# ────────────────────────────────────────────────────────────────────────
# ■ 실기 배포 시 반드시 동일하게 구현할 것
#     - 증분 적분 (phase = fmod(phase + dt*freq, 1.0))
#       절대 시각 기반은 주파수 변경 시 위상이 점프한다
#     - 정지 시 위상을 0 으로 즉시 리셋하지 말 것. 최단 경로로 수렴시킬 것
#     - cos/sin 게이팅 없음
# ============================================================================

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PhaseOnlyGaitCommand(CommandTerm):
    """보행 위상 2 차원 (cos, sin). 게이팅 없음, 정지 시 0 으로 수렴.

    command 텐서:
        [:, 0] = cos(2*pi*phase)
        [:, 1] = sin(2*pi*phase)

    외부 참조용 public 속성:
        phase                [N] 현재 위상 (0~1)
        effective_frequency  [N] 실제 적용된 주파수 (정지면 0)
        is_walking           [N] bool
        is_stopping          [N] bool  위상이 0 으로 수렴 중
        phase_active         [N] bool  is_walking | is_stopping
    """

    cfg: PhaseOnlyGaitCommandCfg

    def __init__(self, cfg: PhaseOnlyGaitCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.effective_frequency = torch.zeros(self.num_envs, device=self.device)
        # env 별 기준 주파수. frequency_range 가 None 이면 cfg.frequency 로 고정된
        # 채 한 번도 안 바뀌므로, 기존 환경의 동작은 완전히 동일하다.
        self.base_frequency = torch.full(
            (self.num_envs,), float(self.cfg.frequency), device=self.device
        )
        self.is_walking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_stopping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # finish_cycle_on_stop: 정지 요청 뒤 이번 주기를 마저 도는 중 · 그걸 쓰는 env (첫 갱신 때 정함)
        self._finishing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fin_mask = None

        # 정지 선호 비트. 속도 명령이 0 일 때만 의미를 갖는다.
        #   True  -> 완전 정지 (주파수 0, 위상 0 으로 수렴)
        #   False -> 제자리 걷기 (주파수 유지, 위상 계속 회전)
        # rel_stop_when_standing=1.0 (기본) 이면 항상 True 라서
        # v0_1 과 동일하게 "속도 0 = 완전 정지" 가 된다.
        self._want_stop = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device)

        # 주파수 전용 리샘플 타이머 (frequency_resampling_time_range 가 있을 때만 씀)
        self._freq_timer = torch.full((self.num_envs,), 1.0e9, device=self.device)

        self._num_dims = 3 if self.cfg.include_frequency else 2
        self._command = torch.zeros(self.num_envs, self._num_dims, device=self.device)

        self.metrics["gait_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ratio_walking"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ratio_stopping"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PhaseOnlyGaitCommand:\n"
        if self.cfg.frequency_range is None:
            msg += f"\tfrequency: {self.cfg.frequency} Hz (고정)\n"
        else:
            lo, hi = self.cfg.frequency_range
            msg += f"\tfrequency: {lo}~{hi} Hz (env 별 랜덤, 리샘플마다 갱신)\n"
        msg += f"\tstop_rate: {self.cfg.stop_rate} phase/s "
        msg += f"(최대 정지 시간 {0.5 / self.cfg.stop_rate:.3f} s)\n"
        msg += "\tgating: 없음 (히스토리로 정지/보행 구분)\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _sample_frequency(self, env_ids):
        """frequency_range 에서 균등 샘플. frequency_low_vs_speed 가 있으면 하한을 지금 속도 명령에 따라 올린다.
        주파수 전용 타이머도 다시 건다."""
        lo, hi = self.cfg.frequency_range
        n = len(env_ids)
        lo_t = torch.full((n,), float(lo), device=self.device)
        if self.cfg.frequency_low_vs_speed is not None and self.cfg.velocity_command_name is not None:
            v0, f0, v1, f1 = self.cfg.frequency_low_vs_speed
            try:
                cmd = self._env.command_manager.get_command(self.cfg.velocity_command_name)
                spd = torch.norm(cmd[env_ids, :2], dim=1)
                a = ((spd - v0) / max(v1 - v0, 1e-6)).clamp(0.0, 1.0)
                lo_t = f0 + a * (f1 - f0)
            except (KeyError, AttributeError):
                pass
        self.base_frequency[env_ids] = lo_t + torch.rand(n, device=self.device) * (hi - lo_t)
        if self.cfg.frequency_resampling_time_range is not None:
            a, b = self.cfg.frequency_resampling_time_range
            self._freq_timer[env_ids] = torch.rand(n, device=self.device) * (b - a) + a

    def _resample_command(self, env_ids):
        # 리셋된 env 만 위상을 0 에서 다시 시작시킨다.
        if len(env_ids) == 0:
            return

        # ── 가변 주파수 (frequency_range 가 있을 때만) ──────────────────
        # env 별로 독립 샘플한다. 속도 명령과 결합하지 않으므로 주파수가
        # 독립적인 제어 입력이 된다 — 실기에서 "빠른 잔걸음 / 느린 성큼걸음"
        # 을 골라 명령할 수 있다는 뜻이다.
        # 관측(gait_phase 셋째 성분 = freq/frequency_scale)에 이미 들어가
        # 있으므로 정책은 이 값을 볼 수 있다. 차원도 안 바뀐다.
        if self.cfg.frequency_range is not None:
            self._sample_frequency(env_ids)

        # 정지 선호 비트 재샘플링.
        # 속도 0 인 env 중 이 비율만큼이 완전 정지가 되고 나머지는
        # 제자리 걷기가 된다. 1.0 이면 전부 완전 정지 (= v0_1 동작).
        r = torch.rand(len(env_ids), device=self.device)
        self._want_stop[env_ids] = r < self.cfg.rel_stop_when_standing

        if self.cfg.reset_phase_on_resample:
            self.phase[env_ids] = 0.0
            # 위상이 0 으로 뛰었으니 '주기 마저 돌기' 를 이어 가지 않음 (finish_cycle_on_stop 일 때만, 기존 동작은 그대로)
            if self.cfg.finish_cycle_on_stop != "off":
                self._finishing[env_ids] = False
                self.is_walking[env_ids] = False

    def _finish_cycle_mask(self) -> torch.Tensor:
        """finish_cycle_on_stop 을 쓰는 env. "tap" = 속도 명령 term 의 탭 env (TapVelocityCommand._tap, 없으면 없음)."""
        if self._fin_mask is None:
            if self.cfg.finish_cycle_on_stop == "all":
                self._fin_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                term = self._env.command_manager.get_term(self.cfg.velocity_command_name)
                tap = getattr(term, "_tap", None)
                self._fin_mask = tap if tap is not None else torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device)
        return self._fin_mask

    def _update_command(self):
        """매 스텝 호출.

        주의: CommandManager 는 cfg 선언 순서대로 term 을 갱신한다.
              velocity_command_name 이 가리키는 term(base_velocity)이
              이 term 보다 먼저 선언되어 있어야 정지 판정이 같은 스텝의
              값을 본다. CommandsCfg 선언 순서를 지킬 것.
        """
        dt = self._env.step_dt

        # 주파수만 따로 리샘플 (위상·정지 선호는 그대로 — 걷는 도중 주파수가 바뀌는 상황)
        if self.cfg.frequency_resampling_time_range is not None and self.cfg.frequency_range is not None:
            self._freq_timer -= dt
            ids = (self._freq_timer <= 0.0).nonzero(as_tuple=False).flatten()
            if len(ids) > 0:
                self._sample_frequency(ids)

        standing = self._standing_mask()
        # env 별 기준 주파수. 고정 모드면 전부 cfg.frequency 와 같다.
        freq = self.base_frequency.clone()

        # 속도 0 이면서 "정지 선호" 인 env 만 주파수 0.
        # 속도 0 인데 want_stop=False 면 주파수를 유지해 제자리 걷기가 된다.
        full_stop = standing & self._want_stop
        freq = torch.where(full_stop, torch.zeros_like(freq), freq)

        # reset hold 구간에서도 위상을 정지시킨다.
        # (hold 동안 액션이 마스킹되므로 위상만 흘러가면 정책이 보는
        #  위상과 실제 다리 상태가 어긋난 채로 학습이 시작된다)
        hold_mask = None
        if self.cfg.freeze_during_reset_hold:
            hold = getattr(self._env, "reset_hold_steps", None)
            if hold is not None:
                hold_mask = hold > 0
                freq = torch.where(hold_mask, torch.zeros_like(freq), freq)

        # ── 노드식 정지 (finish_cycle_on_stop, 11-12 RS02 V0-3 5 차) ──
        # 걷던 중 정지 요청이면 위상이 1 (= 0) 에 닿을 때까지 지금 주파수로 마저 돈다 (명령 0 = 제자리걸음),
        # 닿는 스텝에 위상 0 · 주파수 0 (되감기 없음). 리셋 홀드는 바로 멈춤.
        # 노드 policy_core.GaitClock(finish_cycle=True) 와 같은 규칙.
        fin_end = None
        if self.cfg.finish_cycle_on_stop != "off":
            rot = self.is_walking | self._finishing      # 직전 스텝까지 위상이 돌고 있었음
            fin = self._finish_cycle_mask() & full_stop & rot
            if hold_mask is not None:
                fin = fin & ~hold_mask
            nxt = self.phase + dt * self.base_frequency
            fin_go = fin & (nxt < 1.0)
            fin_end = fin & (nxt >= 1.0)
            freq = torch.where(fin_go, self.base_frequency, freq)
            self._finishing = fin_go

        self.effective_frequency = freq
        self.is_walking = freq > self.cfg.eps

        # ── 보행: 정방향 증분 적분 ──
        phase_walk = torch.fmod(self.phase + dt * freq, 1.0)

        # ── 정지: 0 을 향해 최단 경로로 선형 이동 ──
        # phase > 0.5 면 정방향(1.0 도달 시 0 으로 랩), 아니면 역방향.
        # clamp 로 목표를 넘어가지 않게 해 정확히 0 에 착지시킨다.
        step = self.cfg.stop_rate * dt
        go_forward = self.phase > 0.5
        d_fwd = torch.clamp(1.0 - self.phase, max=step)
        d_bwd = torch.clamp(self.phase, max=step)
        phase_fwd = torch.fmod(self.phase + d_fwd, 1.0)
        phase_bwd = self.phase - d_bwd
        phase_stop = torch.where(go_forward, phase_fwd, phase_bwd)

        self.phase = torch.where(self.is_walking, phase_walk, phase_stop)
        if fin_end is not None:
            self.phase = torch.where(fin_end, torch.zeros_like(self.phase), self.phase)

        self.is_stopping = (~self.is_walking) & (self.phase > self.cfg.eps)
        self.phase_active = self.is_walking | self.is_stopping

        # ── 관측 (게이팅 없음) ──
        two_pi = 2.0 * math.pi
        self._command[:, 0] = torch.cos(two_pi * self.phase)
        self._command[:, 1] = torch.sin(two_pi * self.phase)
        if self.cfg.include_frequency:
            # 제자리 걷기와 완전 정지를 구분하는 신호.
            # 주파수가 고정(2.0)이므로 실질적으로 이진값이다.
            #   보행 / 제자리걷기 : freq/scale = 1.0
            #   완전 정지         : 0.0
            # 위상 cos/sin 만으로는 "속도 0 + 발을 드는 중"과
            # "속도 0 + 가만히 서 있음"을 구분할 근거가 약하다.
            self._command[:, 2] = freq / self.cfg.frequency_scale

    def _update_metrics(self):
        self.metrics["gait_frequency"] = self.effective_frequency.clone()
        self.metrics["ratio_walking"] = self.is_walking.float()
        self.metrics["ratio_stopping"] = self.is_stopping.float()

    def _standing_mask(self) -> torch.Tensor:
        """속도 명령 노름이 임계값 미만이면 정지로 판정."""
        name = self.cfg.velocity_command_name
        if name is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            cmd = self._env.command_manager.get_command(name)
        except (KeyError, AttributeError):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.norm(cmd[:, :3], dim=1) < self.cfg.standing_threshold


@configclass
class PhaseOnlyGaitCommandCfg(CommandTermCfg):
    """PhaseOnlyGaitCommand 설정."""

    class_type: type = PhaseOnlyGaitCommand

    # 주파수가 고정이라 리샘플링은 위상 리셋 용도로만 쓰인다.
    resampling_time_range: tuple[float, float] = (10.0, 10.0)

    # 보행 주파수 [Hz]. v0 의 feet_gait period=0.5 s 와 동일하게 2.0.
    # frequency_range 가 None 일 때만 쓰인다(그때는 전 env 가 이 값으로 고정).
    frequency: float = 2.0

    # ── 가변 주파수 [Hz] ──────────────────────────────────────────────
    #   None (기본) : 전 env 가 frequency 로 고정 — v0_1~v0_5 / walk_kick 동작
    #   (lo, hi)    : 리샘플마다 env 별로 균등 샘플
    #
    # 속도 명령과 결합하지 않는다(독립 샘플). 그래야 주파수가 속도의 함수가
    # 아니라 진짜 제어 입력이 된다. 대신 "고속 + 저주파" 같은 보폭이 큰
    # 조합도 학습에 섞이므로 범위를 너무 넓히면 안 된다.
    #   0.7 m/s 기준 한 걸음 보폭 = v / (2f)
    #       1.6 Hz -> 0.219 m   2.0 Hz -> 0.175 m   2.6 Hz -> 0.135 m
    #
    # ※ 관측은 gait_phase 셋째 성분(freq / frequency_scale)에 이미 있다.
    #   include_frequency=True 이면 차원이 안 바뀌므로 체크포인트 resume 과
    #   symmetry_mx3._MIRROR_SPEC 모두 수정 불필요하다.
    #   (미러 스펙의 gait_phase 부호 [-1,-1,+1] — 주파수는 좌우 반전에 불변)
    frequency_range: tuple[float, float] | None = None

    # 주파수 전용 리샘플 간격 [s] (V0-5 2 차~). None (기본) = 주파수는 resampling_time_range 때만 바뀜 (기존 동작).
    #   (a, b) : env 별 a~b 초마다 주파수만 다시 뽑는다. 위상·정지 선호(want_stop)는 그대로 → 걷는 도중 주파수 변화.
    frequency_resampling_time_range: tuple[float, float] | None = None

    # 속도에 따른 주파수 하한 (V0-5 2 차~). None (기본) = 하한은 frequency_range[0] 고정 (기존 동작).
    #   (v0, f0, v1, f1) : 평면 속도 명령 |v| ≤ v0 이면 하한 f0, ≥ v1 이면 f1, 사이는 선형. 상한은 frequency_range[1].
    #   "저주파 + 고속" (보폭이 다리 길이에 가까운 조합) 을 빼면서 느린 걸음에는 낮은 주파수를 허용.
    frequency_low_vs_speed: tuple[float, float, float, float] | None = None

    # 정지 시 위상이 0 으로 수렴하는 속도 [phase/s].
    # 최대 이동 거리가 0.5 이므로 최대 정지 시간 = 0.5 / stop_rate.
    #   2.0 -> 0.25 s (보행 주기의 절반. 권장)
    stop_rate: float = 2.0

    # 멈추는 방식 (11-12 RS02 V0-3 5 차~). 실기 노드는 walk.yaml finish_cycle_on_stop: true —
    # 걷던 중 정지 요청이면 이번 주기를 지금 주파수로 마저 돌고 (명령 0 = 제자리걸음) 위상 0 에서 멈춤.
    # 0.08 s 짜리 조이패드 탭도 노드에선 한 주기 (~0.54 s) 를 다 걷는데, 위 되감기 정지만 배우면 이 상황이 학습에 없다.
    #   "off" (기본) : 위 stop_rate 되감기 (기존 동작)
    #   "tap"        : 속도 명령 term 의 탭 env (TapVelocityCommand._tap) 만 노드식, 나머지는 되감기
    #   "all"        : 전 env 노드식
    finish_cycle_on_stop: str = "off"

    # 관측에 주파수를 포함할지.
    #   False (기본) : 2 차원 (cos, sin)        — v0_1
    #   True         : 3 차원 (cos, sin, freq)  — v0_2 (제자리 걷기 지원)
    include_frequency: bool = False

    # 주파수 관측 정규화 계수. freq / frequency_scale 이 O(1) 이 되도록.
    frequency_scale: float = 2.0

    # 속도 0 인 env 중 "완전 정지" 가 되는 비율. 나머지는 제자리 걷기.
    #   1.0 (기본) : 전부 완전 정지 = v0_1 동작
    #   0.6        : velocity term 의 rel_standing_envs=0.15 와 곱해져
    #                완전 정지 9%, 제자리 걷기 6%
    rel_stop_when_standing: float = 1.0

    velocity_command_name: str | None = "base_velocity"
    standing_threshold: float = 0.1

    reset_phase_on_resample: bool = True
    freeze_during_reset_hold: bool = True

    eps: float = 1.0e-6



def gait_phase_v01(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
) -> torch.Tensor:
    """PhaseOnlyGaitCommand 의 (cos, sin) 을 관측으로 노출.

    generated_commands 로도 같은 결과를 얻지만, 이름이 명시적이라
    env_cfg 가독성이 좋아 별도 래퍼를 둔다.
    """
    return env.command_manager.get_command(gait_command_name)


def feet_gait_v01(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    gait_command_name: str = "gait",
    offset: tuple[float, ...] = (0.0, 0.5),
    threshold: float = 0.55,
) -> torch.Tensor:
    """stance/swing 일치 보상. v0 의 feet_gait 를 term 위상 참조로 재작성.

    ── v0 feet_gait 와의 차이 ──
    v0 는 위상을 episode_length_buf 로 매번 재계산했다. 그 위상은 관측에
    들어가지 않았으므로, 정책이 볼 수 없는 시계로 채점하는 셈이었다.
    이 버전은 관측에 들어가는 것과 "같은" 위상을 참조한다. 단일 진실 원천.

    정지 과도 구간(위상이 0 으로 수렴 중)에도 작동한다. 위상 0 근처에서는
    두 다리 모두 stance 로 판정되므로(leg0 phase 0 < 0.55, leg1 phase 0.5
    < 0.55), 이 항이 켜져 있으면 자연스럽게 양발 접지를 유도한다.
    완전 정지(위상 0 도달) 후에는 꺼지고, 그 상태의 접지 유지는 v0 의
    feet_contact_without_cmd 가 담당한다.
    """
    from isaaclab.sensors import ContactSensor

    term = env.command_manager.get_term(gait_command_name)
    phase = term.phase
    active = term.phase_active.float()

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        leg_phase = torch.fmod(phase + offset[i], 1.0)
        is_stance = leg_phase < threshold
        reward += (~(is_stance ^ is_contact[:, i])).float()
    return reward * active


def metric_phase_v01(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
) -> torch.Tensor:
    """현재 위상 (0~1). 진단용 (weight=0)."""
    return env.command_manager.get_term(gait_command_name).phase


def stop_posture_v01(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    gait_command_name: str = "gait",
) -> torch.Tensor:
    """완전 정지 상태에서 양발이 접지해 있으면 보상.

    ── v0 의 feet_contact_without_cmd 를 대체하는 이유 ──
    기존 항은 "속도 명령이 0 인가" 로 판정한다. 그런데 제자리 걷기는
    속도 0 이면서 발을 들어야 하는 상태다. 속도 기준 판정을 그대로 쓰면
    "발을 들어라"(feet_gait) 와 "양발을 붙여라"(feet_contact_without_cmd)
    가 정면으로 충돌해 제자리 걷기가 학습되지 않는다.

    이 함수는 주파수 기준으로 판정한다.
        완전 정지 (freq=0, 위상 0)  -> 보상 대상
        제자리 걷기 (freq>0)        -> 대상 아님
    """
    from isaaclab.sensors import ContactSensor

    term = env.command_manager.get_term(gait_command_name)
    full_stop = (~term.is_walking) & (~term.is_stopping)

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    both_contact = torch.all(is_contact, dim=1)
    return (both_contact & full_stop).float()



def stop_default_posture_v01(
    env: ManagerBasedRLEnv,
    asset_cfg,
    gait_command_name: str = "gait",
    include_stopping: bool = False,
) -> torch.Tensor:
    """완전 정지 시 다리 관절을 기본자세(default_joint_pos)로 당기는 페널티.

    ── 왜 필요한가 ──
    정지 시 위상이 0 으로 수렴하는데, 위상 0 은 offset (0.0, 0.5) 기준
    "양발 stance" 일 뿐 보행 주기 중간의 staggered 자세(한 발 앞/한 발 뒤)다.
    stop_posture_v01 은 "양발 접지" 만 보므로 staggered 도 만족 -> 발을
    모을 유인이 없어 정책이 한 발 앞으로 낸 자세를 유지한다.
    이 항이 관절을 default(발 모음 중립)로 직접 당겨 그 문제를 교정한다.

    include_stopping=False (기본): 완전정지 구간에서만 작동. 정지 과도
    구간(is_stopping)에는 feet_gait_v01 이 위상 0 stance 를 유도하므로,
    거기서 default 로 당기면 서로 충돌한다. 그래서 전환은 feet_gait_v01 에
    맡기고, 완전정지 도달 후 자세만 이 항이 다듬는다.

    반환은 penalty(양수) -> env_cfg 에서 음수 weight.
    """
    term = env.command_manager.get_term(gait_command_name)
    active = ~term.is_walking
    if not include_stopping:
        active = active & (~term.is_stopping)  # 완전정지만

    asset = env.scene[asset_cfg.name]
    dev = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(dev), dim=1) * active.float()



def metric_is_marching_v01(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
    velocity_command_name: str = "base_velocity",
    standing_threshold: float = 0.1,
) -> torch.Tensor:
    """제자리 걷기 비율 (속도 0 인데 주파수 > 0). 진단용 (weight=0)."""
    term = env.command_manager.get_term(gait_command_name)
    cmd = env.command_manager.get_command(velocity_command_name)
    standing = torch.norm(cmd[:, :3], dim=1) < standing_threshold
    return (standing & term.is_walking).float()


def metric_is_full_stop_v01(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
) -> torch.Tensor:
    """완전 정지 비율. 진단용 (weight=0)."""
    term = env.command_manager.get_term(gait_command_name)
    return ((~term.is_walking) & (~term.is_stopping)).float()

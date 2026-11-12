# standup_behavior.py
# 넘어진 후 일어나는 동작 - 다단계 키프레임 시퀀스
#
# 두 가지 모드:
#   step mode: 서비스 호출로 한 phase씩 진행 (튜닝용)
#   auto mode: tick()에서 도착 판정 후 자동 진행 (실전용)

import os
import yaml


def _raise_goal_pwm(hw):
    """Goal PWM 상한을 100% 로 올린다 (구버전 컨트롤러 호환).

    hw 에 메서드가 없으면(구버전 body_motor_controller) 조용히 건너뛴다.
    그 경우 PWM 상한 기능 자체가 없으므로 기존 동작과 동일하다.
    """
    fn = getattr(hw, "set_goal_pwm_full", None)
    if fn is not None:
        fn()


def _restore_goal_pwm(hw):
    """Goal PWM 상한을 게인 프로파일의 운용값으로 되돌린다."""
    fn = getattr(hw, "restore_goal_pwm", None)
    if fn is not None:
        fn()


class StandUpBehavior:

    def __init__(self, hw, keyframe_yaml: str = ""):
        self.hw = hw
        self._active = False
        self._done = False
        self._auto_mode = False     # True=자동진행, False=수동(step)

        # phase tracking
        self._sequence = []
        self._phase_idx = -1
        self._phase_name = ""
        self._targets = {}
        self._hold_sec = 0.0
        self._hold_start = None
        self._phase_start_time = 0.0

        # arrive detection
        self.tol = 0.05
        self.settle_need = 5
        self.settle_now = 0

        # global
        self.timeout_sec = 30.0
        self.t_start = 0.0
        self.final_rpm = 200.0
        self._fall_direction = "unknown"

        # 키프레임 로드
        self._fwd_seq = []
        self._bwd_seq = []
        self._yaml_path = keyframe_yaml or ""
        if keyframe_yaml and os.path.isfile(keyframe_yaml):
            self._load_yaml(keyframe_yaml)
            hw.log_info(f"standup keyframes: {keyframe_yaml}")
        else:
            self._fwd_seq = self._default_forward()
            self._bwd_seq = self._default_backward()
            hw.log_info("standup using built-in default keyframes")

    # ──────────────────────────────────────
    # YAML
    # ──────────────────────────────────────
    def _load_yaml(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._fwd_seq = self._parse_seq(data.get("forward_sequence", []))
        self._bwd_seq = self._parse_seq(data.get("backward_sequence", []))

    @staticmethod
    def _parse_seq(seq_list: list) -> list:
        result = []
        for e in seq_list:
            name = str(e.get("name", f"phase_{len(result)}"))
            rpm = float(e.get("rpm", 60.0))
            hold = float(e.get("hold_sec", 0.5))
            joints = {str(k): float(v) for k, v in e.get("joints", {}).items()}
            result.append((name, rpm, hold, joints))
        return result

    def reload_yaml(self, path: str):
        if self._active:
            self.hw.log_warn("cannot reload while active")
            return
        self._load_yaml(path)
        self.hw.log_info(f"keyframes reloaded: {path}")

    # ──────────────────────────────────────
    # 외부 인터페이스
    # ──────────────────────────────────────
    def is_active(self) -> bool:
        return bool(self._active)

    def get_fall_direction(self) -> str:
        return self._fall_direction

    def get_phase_name(self) -> str:
        return self._phase_name

    def get_phase_idx(self) -> int:
        return self._phase_idx

    def get_total_phases(self) -> int:
        return len(self._sequence)

    # ──── start: 시퀀스 시작 + 첫 phase 실행 ────
    def start(
        self,
        fall_direction: str = "auto",
        gravity_z: float = 0.0,
        auto_mode: bool = False,
        final_rpm: float = 200.0,
        tol: float = 0.05,
        settle_count: int = 5,
        timeout_sec: float = 30.0,
    ) -> tuple:
        if self._active:
            return False, "standup already running"

        # 낙하 방향
        if fall_direction == "auto":
            self._fall_direction = "forward" if gravity_z > -0.3 else "backward"
        elif fall_direction in ("forward", "backward"):
            self._fall_direction = fall_direction
        else:
            return False, f"invalid fall_direction: {fall_direction}"

        # 시퀀스 선택
        if self._fall_direction == "forward":
            self._sequence = list(self._fwd_seq)
        else:
            self._sequence = list(self._bwd_seq)

        if not self._sequence:
            return False, "keyframe sequence is empty"

        self._auto_mode = auto_mode
        self.tol = float(tol)
        self.settle_need = int(settle_count)
        self.timeout_sec = float(timeout_sec)
        self.final_rpm = float(final_rpm)

        self._active = True
        self._done = False
        self._phase_idx = -1
        self.settle_now = 0
        self._hold_start = None
        self.t_start = self.hw.now_sec()

        self.hw.set_torque_all(True)
        # ★ 팔로 몸을 밀어 올려야 하므로 Goal PWM 상한을 100% 로 올린다.
        #   낮은 상한이 걸린 채로는 팔이 몸무게를 못 이겨 일어서지 못한다.
        _raise_goal_pwm(self.hw)

        mode_str = "auto" if auto_mode else "step"
        self.hw.log_info(
            f"standup start: dir={self._fall_direction}, "
            f"mode={mode_str}, phases={len(self._sequence)}"
        )

        # 첫 phase 실행
        return self._go_phase(0)

    # ──── next: 다음 phase ────
    def next_phase(self) -> tuple:
        if not self._active:
            return False, "standup not running. call start first"

        next_idx = self._phase_idx + 1
        if next_idx >= len(self._sequence):
            self._finish()
            return True, "standup complete (all phases done)"

        return self._go_phase(next_idx)

    # ──── back: 이전 phase ────
    def back_phase(self) -> tuple:
        if not self._active:
            return False, "standup not running"

        if self._phase_idx <= 0:
            return False, "already at first phase"

        return self._go_phase(self._phase_idx - 1)

    # ──── reset: 첫 phase로 ────
    def reset_phase(self) -> tuple:
        if not self._active:
            return False, "standup not running"

        return self._go_phase(0)

    # ──── stop: 중단 ────
    def stop(self) -> tuple:
        if not self._active:
            return False, "standup not running"

        self.hw.log_info("standup stopped by user")
        self._finish()
        return True, "standup stopped"

    # ──── goto: 이름으로 특정 phase만 실행 ────
    def goto_phase(self, phase_name: str, fall_direction: str = "forward") -> tuple:
        """
        특정 phase를 이름으로 바로 실행한다.
        start 없이 단독으로 동작하므로 튜닝에 편리하다.
        """
        # 시퀀스가 아직 없으면 방향에 맞게 로드
        if not self._sequence:
            if fall_direction == "backward":
                self._sequence = list(self._bwd_seq)
                self._fall_direction = "backward"
            else:
                self._sequence = list(self._fwd_seq)
                self._fall_direction = "forward"

        if not self._sequence:
            return False, "keyframe sequence is empty"

        # 이름으로 찾기
        found_idx = -1
        for i, (name, rpm, hold, joints) in enumerate(self._sequence):
            if name == phase_name:
                found_idx = i
                break

        # 숫자 인덱스도 허용
        if found_idx < 0:
            try:
                idx = int(phase_name)
                if 0 <= idx < len(self._sequence):
                    found_idx = idx
            except ValueError:
                pass

        if found_idx < 0:
            names = [s[0] for s in self._sequence]
            return False, f"'{phase_name}' not found. available: {names}"

        # active 상태로 전환
        self._active = True
        self._done = False
        self._auto_mode = False
        self.t_start = self.hw.now_sec()

        self.hw.set_torque_all(True)
        _raise_goal_pwm(self.hw)
        return self._go_phase(found_idx)

    # ──── reload: YAML 다시 읽기 (노드 재시작 없이) ────
    def do_reload(self, path: str = "") -> tuple:
        """YAML을 다시 읽고 시퀀스를 갱신한다."""
        if not path:
            path = self._yaml_path
        if not path:
            return False, "no yaml path (use start with yaml first)"

        if not os.path.isfile(path):
            return False, f"file not found: {path}"

        # 실행 중이면 멈추고 리로드
        if self._active:
            self._finish()

        self._load_yaml(path)
        self._sequence = []  # 다음 start/goto에서 다시 로드되도록
        self.hw.log_info(f"keyframes reloaded: {path}")
        return True, f"reloaded {path} (fwd={len(self._fwd_seq)}, bwd={len(self._bwd_seq)} phases)"

    # ──── tick: 매 사이클 호출 (auto mode에서만 자동 진행) ────
    def tick(self, last_js):
        if not self._active:
            return

        now = self.hw.now_sec()

        # ── [진단] tick 호출 간격 추적 ──
        # read_and_publish 가 50Hz 라면 dt 가 ~20ms 여야 한다.
        # 만약 dt 가 가끔 100ms 처럼 크게 튄다면 read 사이클이 밀린 것 →
        # phase 전환 판단 자체가 늦어진다 (가능성 C).
        # _last_tick_ts 가 없는 첫 호출은 건너뜀.
        last_tick = getattr(self, "_last_tick_ts", None)
        if last_tick is not None and self._auto_mode:
            tick_dt_ms = (now - last_tick) * 1000.0
            # 25ms 초과 (50Hz=20ms 기준 25% 이상 늦음) 인 경우만 경고.
            # 모든 tick 을 찍으면 로그가 너무 시끄러움.
            if tick_dt_ms > 25.0:
                self.hw.log_warn(
                    f"[DIAG] tick gap {tick_dt_ms:.1f}ms (expected ~20ms) "
                    f"at phase {self._phase_idx} ({self._phase_name})"
                )
        self._last_tick_ts = now

        # 글로벌 타임아웃
        if self._auto_mode and now - self.t_start > self.timeout_sec:
            self.hw.log_warn(
                f"standup timeout at phase {self._phase_idx} ({self._phase_name})"
            )
            self._finish()
            return

        # auto mode: 시간 기반 전환 (hold_sec 경과하면 다음 phase)
        if self._auto_mode:
            elapsed = now - self._phase_start_time
            if elapsed >= self._hold_sec:
                # ── [진단] 만료 시점과 실제 전환 시점 차이 ──
                # 이상적으로는 elapsed == hold_sec 일 때 즉시 넘어가지만,
                # tick 이 hold_sec 만료 직후가 아니라 다음 사이클에 잡으면
                # over_ms 만큼 늦게 전환된다 (가능성 A).
                over_ms = (elapsed - self._hold_sec) * 1000.0
                if over_ms > 5.0:
                    self.hw.log_info(
                        f"[DIAG] phase transition over by {over_ms:.1f}ms "
                        f"(target={self._hold_sec*1000:.0f}ms, "
                        f"actual={elapsed*1000:.0f}ms)"
                    )

                next_idx = self._phase_idx + 1
                if next_idx < len(self._sequence):
                    self._go_phase(next_idx)
                else:
                    self.hw.log_info("standup complete!")
                    self._finish()

    # ──────────────────────────────────────
    # 내부
    # ──────────────────────────────────────
    def _go_phase(self, idx: int) -> tuple:
        # ── [진단] 이전 phase 시작 시점부터 지금까지의 실제 경과 시간 ──
        # 첫 phase 가 아니라면 이전 _phase_start_time 이 있다.
        # YAML 의 hold_sec 와 비교해서 얼마나 어긋났는지 본다.
        now_enter = self.hw.now_sec()
        if self._phase_idx >= 0:
            # 이전 phase 의 의도된 hold_sec 와 실제 elapsed 차이
            prev_target  = self._hold_sec
            prev_elapsed = now_enter - self._phase_start_time
            drift_ms     = (prev_elapsed - prev_target) * 1000.0
            self.hw.log_info(
                f"[DIAG] phase {self._phase_idx} ({self._phase_name}) "
                f"target={prev_target*1000:.0f}ms, actual={prev_elapsed*1000:.0f}ms, "
                f"drift={drift_ms:+.1f}ms"
            )

        phase_name, rpm, hold_sec, joint_targets = self._sequence[idx]

        self._phase_idx = idx
        self._phase_name = phase_name
        self._hold_sec = hold_sec
        self._phase_start_time = now_enter  # phase 시작 시각 (now_enter 재사용)
        self._hold_start = None
        self.settle_now = 0

        targets = {}
        for name, rad in joint_targets.items():
            if self.hw.has_joint(name):
                targets[name] = float(rad)

        self._targets = targets

        # ── [진단] set_profile_velocity + send_goals 소요 시간 ──
        # 이 두 호출이 SyncWrite 를 통해 U2D2 에 나가는데, 모터 수 (18~22)
        # × 모터당 ~수 ms 라 phase 명령 자체가 수십~수백 ms 걸릴 수 있다.
        # 만약 이게 크면, "phase 명령을 보내는 동안" hold_sec 시계가 이미
        # 흐르고 있어서 다음 phase 까지의 실제 간격이 hold_sec 보다 짧아
        # 보이거나, 반대로 명령이 늦게 도착해서 동작 시작이 늦어진다.
        t0 = self.hw.now_sec()
        self.hw.set_profile_velocity_all(rpm)
        t1 = self.hw.now_sec()
        self.hw.send_goals(targets)
        t2 = self.hw.now_sec()
        write_ms = (t2 - t0) * 1000.0
        prof_ms  = (t1 - t0) * 1000.0
        send_ms  = (t2 - t1) * 1000.0

        total = len(self._sequence)
        self.hw.log_info(
            f"phase {idx}/{total-1}: {phase_name} "
            f"(rpm={rpm:.0f}, hold={hold_sec:.1f}s, {len(targets)} joints) "
            f"| [DIAG] write={write_ms:.1f}ms (prof={prof_ms:.1f}, send={send_ms:.1f})"
        )

        msg = f"phase {idx}/{total-1}: {phase_name} (rpm={rpm:.0f}, hold={hold_sec:.1f}s)"
        return True, msg

    def _finish(self):
        self.hw.set_profile_velocity_all(self.final_rpm)
        # 시작 때 100% 로 올린 Goal PWM 상한을 운용값으로 되돌린다.
        # 안 내리면 팔이 100% 인 채로 보행에 들어가 낙상 시 감속기 보호가 없다.
        _restore_goal_pwm(self.hw)
        self._active = False
        self._done = True

    # ──────────────────────────────────────
    # 기본 키프레임
    # ──────────────────────────────────────
    _STAND = {
        "r_pelvis_pitch_joint": 0.35,  "r_pelvis_roll_joint": 0.0,
        "r_pelvis_yaw_joint": 0.0,     "r_knee_pitch_joint": 0.52,
        "r_ankle_pitch_joint": 0.35,   "r_ankle_roll_joint": 0.0,
        "l_pelvis_pitch_joint": -0.35, "l_pelvis_roll_joint": 0.0,
        "l_pelvis_yaw_joint": 0.0,     "l_knee_pitch_joint": -0.52,
        "l_ankle_pitch_joint": -0.35,  "l_ankle_roll_joint": 0.0,
        "r_shoulder_pitch_joint": -0.3, "r_shoulder_roll_joint": 0.0,
        "r_elbow_pitch_joint": 2.0,
        "l_shoulder_pitch_joint": 0.3,  "l_shoulder_roll_joint": 0.0,
        "l_elbow_pitch_joint": -2.0,
    }

    @classmethod
    def _default_forward(cls):
        return [
            ("tuck", 60.0, 0.5, {
                "r_pelvis_pitch_joint": 1.2, "r_pelvis_roll_joint": 0.0,
                "r_pelvis_yaw_joint": 0.0, "r_knee_pitch_joint": 2.0,
                "r_ankle_pitch_joint": 0.8, "r_ankle_roll_joint": 0.0,
                "l_pelvis_pitch_joint": -1.2, "l_pelvis_roll_joint": 0.0,
                "l_pelvis_yaw_joint": 0.0, "l_knee_pitch_joint": -2.0,
                "l_ankle_pitch_joint": -0.8, "l_ankle_roll_joint": 0.0,
                "r_shoulder_pitch_joint": -1.2, "r_shoulder_roll_joint": 0.3,
                "r_elbow_pitch_joint": 0.5,
                "l_shoulder_pitch_joint": 1.2, "l_shoulder_roll_joint": -0.3,
                "l_elbow_pitch_joint": -0.5,
            }),
            ("pushup", 50.0, 0.8, {
                "r_pelvis_pitch_joint": 0.8, "r_pelvis_roll_joint": 0.0,
                "r_pelvis_yaw_joint": 0.0, "r_knee_pitch_joint": 2.0,
                "r_ankle_pitch_joint": 0.5, "r_ankle_roll_joint": 0.0,
                "l_pelvis_pitch_joint": -0.8, "l_pelvis_roll_joint": 0.0,
                "l_pelvis_yaw_joint": 0.0, "l_knee_pitch_joint": -2.0,
                "l_ankle_pitch_joint": -0.5, "l_ankle_roll_joint": 0.0,
                "r_shoulder_pitch_joint": -0.8, "r_shoulder_roll_joint": 0.2,
                "r_elbow_pitch_joint": 1.5,
                "l_shoulder_pitch_joint": 0.8, "l_shoulder_roll_joint": -0.2,
                "l_elbow_pitch_joint": -1.5,
            }),
            ("kneel", 50.0, 0.5, {
                "r_pelvis_pitch_joint": 0.6, "r_pelvis_roll_joint": 0.0,
                "r_pelvis_yaw_joint": 0.0, "r_knee_pitch_joint": 1.5,
                "r_ankle_pitch_joint": 0.5, "r_ankle_roll_joint": 0.0,
                "l_pelvis_pitch_joint": -0.6, "l_pelvis_roll_joint": 0.0,
                "l_pelvis_yaw_joint": 0.0, "l_knee_pitch_joint": -1.5,
                "l_ankle_pitch_joint": -0.5, "l_ankle_roll_joint": 0.0,
                "r_shoulder_pitch_joint": -0.3, "r_shoulder_roll_joint": 0.0,
                "r_elbow_pitch_joint": 2.0,
                "l_shoulder_pitch_joint": 0.3, "l_shoulder_roll_joint": 0.0,
                "l_elbow_pitch_joint": -2.0,
            }),
            ("stand", 60.0, 0.5, cls._STAND),
            ("stabilize", 100.0, 0.3, cls._STAND),
        ]

    @classmethod
    def _default_backward(cls):
        return [
            ("curl", 60.0, 0.5, {
                "r_pelvis_pitch_joint": 1.0, "r_pelvis_roll_joint": 0.0,
                "r_pelvis_yaw_joint": 0.0, "r_knee_pitch_joint": 2.0,
                "r_ankle_pitch_joint": 0.5, "r_ankle_roll_joint": 0.0,
                "l_pelvis_pitch_joint": -1.0, "l_pelvis_roll_joint": 0.0,
                "l_pelvis_yaw_joint": 0.0, "l_knee_pitch_joint": -2.0,
                "l_ankle_pitch_joint": -0.5, "l_ankle_roll_joint": 0.0,
                "r_shoulder_pitch_joint": 0.5, "r_shoulder_roll_joint": 0.8,
                "r_elbow_pitch_joint": 0.3,
                "l_shoulder_pitch_joint": -0.5, "l_shoulder_roll_joint": -0.8,
                "l_elbow_pitch_joint": -0.3,
            }),
            ("roll", 50.0, 0.8, {
                "r_pelvis_pitch_joint": 0.8, "r_pelvis_roll_joint": 0.3,
                "r_pelvis_yaw_joint": 0.0, "r_knee_pitch_joint": 2.0,
                "r_ankle_pitch_joint": 0.5, "r_ankle_roll_joint": 0.0,
                "l_pelvis_pitch_joint": -0.8, "l_pelvis_roll_joint": 0.3,
                "l_pelvis_yaw_joint": 0.0, "l_knee_pitch_joint": -2.0,
                "l_ankle_pitch_joint": -0.5, "l_ankle_roll_joint": 0.0,
                "r_shoulder_pitch_joint": -0.5, "r_shoulder_roll_joint": 0.5,
                "r_elbow_pitch_joint": 1.0,
                "l_shoulder_pitch_joint": 0.5, "l_shoulder_roll_joint": -0.5,
                "l_elbow_pitch_joint": -1.0,
            }),
            ("kneel", 50.0, 0.5, {
                "r_pelvis_pitch_joint": 0.6, "r_pelvis_roll_joint": 0.0,
                "r_pelvis_yaw_joint": 0.0, "r_knee_pitch_joint": 1.5,
                "r_ankle_pitch_joint": 0.5, "r_ankle_roll_joint": 0.0,
                "l_pelvis_pitch_joint": -0.6, "l_pelvis_roll_joint": 0.0,
                "l_pelvis_yaw_joint": 0.0, "l_knee_pitch_joint": -1.5,
                "l_ankle_pitch_joint": -0.5, "l_ankle_roll_joint": 0.0,
                "r_shoulder_pitch_joint": -0.3, "r_shoulder_roll_joint": 0.0,
                "r_elbow_pitch_joint": 2.0,
                "l_shoulder_pitch_joint": 0.3, "l_shoulder_roll_joint": 0.0,
                "l_elbow_pitch_joint": -2.0,
            }),
            ("stand", 60.0, 0.5, cls._STAND),
            ("stabilize", 100.0, 0.3, cls._STAND),
        ]
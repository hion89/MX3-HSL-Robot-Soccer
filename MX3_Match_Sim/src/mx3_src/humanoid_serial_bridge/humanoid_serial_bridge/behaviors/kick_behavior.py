#!/usr/bin/env python3
"""Keyframe kick behavior for the U2D2 bridge."""

import os
import yaml


JOINT_LIMITS = {
    "r_pelvis_pitch_joint": (-1.5707963267949, 1.5707963267949),
    "r_pelvis_roll_joint": (-0.523598775598299, 1.5707963267949),
    "r_pelvis_yaw_joint": (-1.5707963267949, 1.5707963267949),
    "r_knee_pitch_joint": (0.0, 2.44346095279206),
    "r_ankle_pitch_joint": (-0.872664625997165, 1.91986217719376),
    "r_ankle_roll_joint": (-0.523598775598299, 0.523598775598299),
    "l_pelvis_pitch_joint": (-1.5707963267949, 1.5707963267949),
    "l_pelvis_roll_joint": (-1.5707963267949, 0.523598775598299),
    "l_pelvis_yaw_joint": (-1.5707963267949, 1.5707963267949),
    "l_knee_pitch_joint": (-2.44346095279206, 0.0),
    "l_ankle_pitch_joint": (-1.91986217719376, 0.872664625997165),
    "l_ankle_roll_joint": (-0.523598775598299, 0.523598775598299),
    "r_shoulder_pitch_joint": (-2.61799387799149, 2.61799387799149),
    "r_shoulder_roll_joint": (-0.261799387799149, 2.87979326579064),
    "r_elbow_pitch_joint": (0.0, 2.96705972839036),
    "l_shoulder_pitch_joint": (-2.61799387799149, 2.61799387799149),
    "l_shoulder_roll_joint": (-2.87979326579064, 0.261799387799149),
    "l_elbow_pitch_joint": (-2.96705972839036, 0.0),
}


class KickBehavior:
    def __init__(self, hw, keyframe_yaml: str = ""):
        self.hw = hw
        self._active = False
        self._done = False
        self._auto_mode = False

        self._right_seq = []
        self._left_seq = []
        self._sequence = []
        self._phase_idx = -1
        self._phase_name = ""
        self._hold_sec = 0.0
        self._phase_start_time = 0.0
        self._yaml_path = keyframe_yaml or ""
        self._active_leg = "right"

        self.timeout_sec = 10.0
        self.t_start = 0.0
        self.final_rpm = 200.0

        if keyframe_yaml and os.path.isfile(keyframe_yaml):
            self._load_yaml(keyframe_yaml)
            hw.log_info(f"kick keyframes: {keyframe_yaml}")
        else:
            hw.log_warn("kick keyframe yaml not found; kick behavior has no phases")

    def _load_yaml(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._right_seq = self._parse_seq(data.get("right_kick_sequence", []))
        self._left_seq = self._parse_seq(data.get("left_kick_sequence", []))
        self._yaml_path = path

    @staticmethod
    def _parse_seq(seq_list: list) -> list:
        result = []
        for entry in seq_list:
            name = str(entry.get("name", f"phase_{len(result)}"))
            rpm = float(entry.get("rpm", 50.0))
            hold = float(entry.get("hold_sec", 0.5))
            joints = {str(k): float(v) for k, v in entry.get("joints", {}).items()}
            result.append((name, rpm, hold, joints))
        return result

    def is_active(self) -> bool:
        return bool(self._active)

    def get_active_leg(self) -> str:
        return self._active_leg

    def get_phase_name(self) -> str:
        return self._phase_name

    def get_phase_idx(self) -> int:
        return self._phase_idx

    def get_total_phases(self) -> int:
        return len(self._sequence)

    def start(
        self,
        leg: str = "right",
        auto_mode: bool = False,
        final_rpm: float = 200.0,
        timeout_sec: float = 10.0,
    ) -> tuple:
        if self._active:
            return False, "kick already running"

        leg = (leg or "right").lower()
        if leg not in ("right", "left"):
            return False, f"invalid leg: {leg}"

        self._sequence = list(self._right_seq if leg == "right" else self._left_seq)
        if not self._sequence:
            return False, f"{leg}_kick_sequence is empty"

        self._active_leg = leg
        self._auto_mode = auto_mode
        self.final_rpm = float(final_rpm)
        self.timeout_sec = float(timeout_sec)
        self._active = True
        self._done = False
        self._phase_idx = -1
        self.t_start = self.hw.now_sec()

        self.hw.set_torque_all(True)
        self.hw.log_info(
            f"kick start: leg={self._active_leg}, "
            f"mode={'auto' if auto_mode else 'step'}, phases={len(self._sequence)}"
        )
        return self._go_phase(0)

    def goto_phase(self, phase_name: str, leg: str = "right") -> tuple:
        if not self._sequence or self._active_leg != (leg or "right").lower():
            leg = (leg or "right").lower()
            self._sequence = list(self._right_seq if leg == "right" else self._left_seq)
            self._active_leg = leg

        if not self._sequence:
            return False, f"{self._active_leg}_kick_sequence is empty"

        found_idx = -1
        for i, (name, _rpm, _hold, _joints) in enumerate(self._sequence):
            if name == phase_name:
                found_idx = i
                break

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

        self._active = True
        self._done = False
        self._auto_mode = False
        self.t_start = self.hw.now_sec()
        self.hw.set_torque_all(True)
        return self._go_phase(found_idx)

    def next_phase(self) -> tuple:
        if not self._active:
            return False, "kick not running. call start first"
        next_idx = self._phase_idx + 1
        if next_idx >= len(self._sequence):
            self._finish()
            return True, "kick complete (all phases done)"
        return self._go_phase(next_idx)

    def back_phase(self) -> tuple:
        if not self._active:
            return False, "kick not running"
        if self._phase_idx <= 0:
            return False, "already at first phase"
        return self._go_phase(self._phase_idx - 1)

    def reset_phase(self) -> tuple:
        if not self._active:
            return False, "kick not running"
        return self._go_phase(0)

    def stop(self) -> tuple:
        if not self._active:
            return False, "kick not running"
        self.hw.log_info("kick stopped by user")
        self._finish()
        return True, "kick stopped"

    def do_reload(self, path: str = "") -> tuple:
        if not path:
            path = self._yaml_path
        if not path:
            return False, "no yaml path"
        if not os.path.isfile(path):
            return False, f"file not found: {path}"
        if self._active:
            self._finish()
        self._load_yaml(path)
        self._sequence = []
        self.hw.log_info(f"kick keyframes reloaded: {path}")
        return True, f"reloaded {path} (right={len(self._right_seq)}, left={len(self._left_seq)} phases)"

    def tick(self, _last_js):
        if not self._active:
            return

        now = self.hw.now_sec()
        if self._auto_mode and now - self.t_start > self.timeout_sec:
            self.hw.log_warn(f"kick timeout at phase {self._phase_idx} ({self._phase_name})")
            self._finish()
            return

        if self._auto_mode:
            elapsed = now - self._phase_start_time
            if elapsed >= self._hold_sec:
                next_idx = self._phase_idx + 1
                if next_idx < len(self._sequence):
                    self._go_phase(next_idx)
                else:
                    self.hw.log_info("kick complete!")
                    self._finish()

    def _go_phase(self, idx: int) -> tuple:
        phase_name, rpm, hold_sec, joint_targets = self._sequence[idx]
        self._phase_idx = idx
        self._phase_name = phase_name
        self._hold_sec = hold_sec
        self._phase_start_time = self.hw.now_sec()

        targets = {}
        for name, rad in joint_targets.items():
            if self.hw.has_joint(name):
                targets[name] = self._clamp_joint(name, float(rad))

        self.hw.set_profile_velocity_all(rpm)
        self.hw.send_goals(targets)

        total = len(self._sequence)
        self.hw.log_info(
            f"kick phase {idx}/{total-1}: {phase_name} "
            f"(rpm={rpm:.0f}, hold={hold_sec:.2f}s, {len(targets)} joints)"
        )
        return True, f"phase {idx}/{total-1}: {phase_name} (rpm={rpm:.0f}, hold={hold_sec:.2f}s)"

    def _clamp_joint(self, name: str, rad: float) -> float:
        if name not in JOINT_LIMITS:
            return rad
        lo, hi = JOINT_LIMITS[name]
        if rad < lo:
            self.hw.log_warn(f"kick target clipped: {name} {rad:.3f} -> {lo:.3f}")
            return lo
        if rad > hi:
            self.hw.log_warn(f"kick target clipped: {name} {rad:.3f} -> {hi:.3f}")
            return hi
        return rad

    def _finish(self):
        self.hw.set_profile_velocity_all(self.final_rpm)
        self._active = False
        self._done = True

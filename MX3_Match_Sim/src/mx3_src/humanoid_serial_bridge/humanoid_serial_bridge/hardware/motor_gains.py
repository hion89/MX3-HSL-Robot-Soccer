# motor_gains.py
# ============================================================================
# Dynamixel PID 게인 프로파일 로더 + SyncWrite 적용
#
# ── 왜 별도 모듈인가 ──
# PID 게인(addr 80/82/84)은 EEPROM 이 아니라 RAM 이다. 이 차이가 전부다.
#
#   (1) 토크 ON 상태에서도 쓸 수 있다
#       → 로봇이 서 있는 채로, 보행 중에도 게인을 바꿔가며 비교할 수 있다.
#
#   (2) ★ 전원 OFF 또는 모터 REBOOT 시 EEPROM 기본값으로 돌아간다 ★
#       → 부팅 시뿐 아니라 "리부트 직후에도" 반드시 다시 써야 한다.
#         이걸 놓치면 넘어졌다 일어난 뒤 거동이 달라지는데 원인을
#         찾을 수 없는 상황이 된다. (에러 복구 경로가 reboot 를 쓴다)
#
# ── XM540 기본값 ──
#   Position P = 800 (또는 850)   Position I = 0   Position D = 0
#   D 가 0 이라 감쇠가 전혀 없다. P 만 올리면 발진한다.
#
# ── 내부 스케일 (e-manual) ──
#   PWM 기여분 = P/128 * err + I/65536 * ∫err + D/16 * derr
#   절대값이 커 보여도 실효 이득은 이 스케일로 나눈 값이다.
#
# ── 쓰기 순서 ──
#   주소가 80(D) / 82(I) / 84(P) 로 연속이므로 6 바이트를 한 번에
#   SyncWrite 한다. 개별로 쓰면 P 가 먼저 들어가는 순간 D 가 아직
#   0 인 상태로 동작해 로봇이 튈 수 있는데, 한 패킷이면 그 문제가
#   원천적으로 사라진다.
# ============================================================================

from __future__ import annotations

import os
from typing import Callable, Optional

import yaml

# ─────────────────────────────────────────────────────────────────────────
# 컨트롤 테이블 (XM540, Protocol 2.0). 전부 RAM 영역.
# ─────────────────────────────────────────────────────────────────────────
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84

# Goal PWM(100, 2B, RAM). 위치 제어 모드에서 PID 가 계산한 PWM 출력의 '상한'.
#   낮추면 토크와 속도가 함께 낮아진다 → 일정 힘 이상에서 관절이 항복하는
#   토크 리미터로 쓸 수 있다. 위치 제어는 그대로 살아 있다.
#   ★ 단위: PWM Limit(36) 이 100% 다. 공장 기본 885. 1 LSB ≈ 0.113%
#   ★ 게인과 똑같이 RAM 이라 전원 OFF / REBOOT 시 PWM Limit 으로 돌아간다.
#     그래서 게인과 같은 경로(_apply_gains)에서 함께 기입한다.
ADDR_GOAL_PWM = 100
LEN_GOAL_PWM = 2

# D, I, P 가 연속 주소이므로 한 번에 쓴다
ADDR_PID_BLOCK = ADDR_POSITION_D_GAIN
LEN_PID_BLOCK = 6

GAIN_KEYS = ("position_d_gain", "position_i_gain", "position_p_gain")

# goal_pwm 은 게인과 스케일도 의미도 달라서 GAIN_KEYS 와 분리해 다룬다.
#   · 없으면(None) 그 관절의 Goal PWM 을 건드리지 않는다 = 기존 동작 유지
#   · 있으면 그 값으로 상한을 건다
GOAL_PWM_KEY = "goal_pwm"
GOAL_PWM_MAX = 885   # 공장 기본 PWM Limit. 실제 상한은 모터에서 읽어 clamp 한다

# XM540 공장 기본값
DEFAULT_GAINS = {
    "position_d_gain": 0,
    "position_i_gain": 0,
    "position_p_gain": 800,
}

GAIN_MAX = 16383  # 2^14 - 1

# 그룹 판정에 쓰는 이름 부분일치 기본값.
# YAML 에서 groups.<name>.match 로 덮어쓸 수 있다.
DEFAULT_GROUP_MATCH = {
    "legs": ["pelvis", "knee", "ankle"],
    "arms": ["shoulder", "elbow"],
}


def _u16_le(v: int) -> bytes:
    v = int(v) & 0xFFFF
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def pid_block_bytes(gains: dict) -> bytes:
    """D, I, P 를 6 바이트 리틀엔디언으로 직렬화 (주소 80 부터)."""
    return (_u16_le(gains["position_d_gain"])
            + _u16_le(gains["position_i_gain"])
            + _u16_le(gains["position_p_gain"]))


class GainProfile:
    """motor_gains.yaml 을 읽어 관절별 게인을 확정한다.

    적용 우선순위: default -> groups -> joints (뒤가 우선)

    YAML 예시는 config/motor_gains.yaml 참조.
    파일이 없거나 비어 있으면 XM540 기본값을 그대로 쓴다
    (= 게인 기능을 안 쓰는 것과 동일. 기존 동작 유지).
    """

    def __init__(self,
                 path: str = "",
                 log_info: Callable[[str], None] = print,
                 log_warn: Callable[[str], None] = print):
        self.path = path or ""
        self._log_info = log_info
        self._log_warn = log_warn

        self.default = dict(DEFAULT_GAINS)
        self.groups: dict = {}
        self.joints: dict = {}
        self.loaded = False

        if self.path:
            self.load(self.path)

    # ------------------------------------------------------------------
    def load(self, path: str) -> bool:
        """YAML 을 읽는다. 실패하면 기본값 유지하고 False 반환.

        튜닝 중 YAML 을 고치고 재적용할 때도 이 메서드를 다시 부른다.
        """
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self._log_warn(f"[gains] 파일 없음: {path} → XM540 기본값 사용")
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            self._log_warn(f"[gains] YAML 파싱 실패: {e} → 기본값 유지")
            return False

        self.path = path
        self.default = dict(DEFAULT_GAINS)
        self.default.update(self._sanitize(data.get("default", {}), "default"))

        self.groups = {}
        for gname, gval in (data.get("groups") or {}).items():
            gval = gval or {}
            match = gval.get("match") or DEFAULT_GROUP_MATCH.get(gname, [])
            self.groups[gname] = {
                "match": [str(m).lower() for m in match],
                "gains": self._sanitize(gval, f"groups.{gname}"),
            }

        self.joints = {}
        for jname, jval in (data.get("joints") or {}).items():
            self.joints[jname] = self._sanitize(jval or {}, f"joints.{jname}")

        self.loaded = True
        return True

    # ------------------------------------------------------------------
    def _sanitize(self, d: dict, where: str) -> dict:
        """게인 키만 뽑고 범위를 검사한다. 잘못된 값은 버리고 경고."""
        out = {}
        for k in GAIN_KEYS:
            if k not in d:
                continue
            try:
                v = int(d[k])
            except (TypeError, ValueError):
                self._log_warn(f"[gains] {where}.{k} 값이 정수가 아님 → 무시")
                continue
            if not (0 <= v <= GAIN_MAX):
                self._log_warn(
                    f"[gains] {where}.{k}={v} 범위 초과(0~{GAIN_MAX}) → 무시")
                continue
            out[k] = v

        # goal_pwm (선택). 범위는 0 < v <= GOAL_PWM_MAX.
        #   0 은 "출력 0" 이라 관절이 완전히 풀려버린다. 의도치 않은 0 을
        #   막기 위해 하한을 1 로 둔다(정말 풀고 싶으면 토크를 끄는 게 맞다).
        if GOAL_PWM_KEY in d:
            try:
                v = int(d[GOAL_PWM_KEY])
            except (TypeError, ValueError):
                self._log_warn(f"[gains] {where}.{GOAL_PWM_KEY} 값이 정수가 아님 → 무시")
            else:
                if not (1 <= v <= GOAL_PWM_MAX):
                    self._log_warn(
                        f"[gains] {where}.{GOAL_PWM_KEY}={v} 범위 초과"
                        f"(1~{GOAL_PWM_MAX}) → 무시")
                else:
                    out[GOAL_PWM_KEY] = v
        return out

    # ------------------------------------------------------------------
    def group_of(self, joint_name: str) -> Optional[str]:
        """관절 이름으로 그룹을 판정한다. 먼저 일치하는 그룹이 이긴다."""
        low = joint_name.lower()
        for gname, g in self.groups.items():
            if any(m in low for m in g["match"]):
                return gname
        return None

    def goal_pwm_for(self, joint_name: str):
        """한 관절의 Goal PWM 상한. 설정이 없으면 None(= 건드리지 않음).

        게인과 같은 우선순위 규칙(default -> group -> joint)을 따른다.
        """
        return self.gains_for(joint_name).get(GOAL_PWM_KEY)

    def build_goal_pwm_map(self, joint_to_id: dict) -> dict:
        """{joint_name: dxl_id} → {dxl_id: goal_pwm}. 설정 없는 관절은 제외."""
        out = {}
        for jname, jid in joint_to_id.items():
            v = self.goal_pwm_for(jname)
            if v is not None:
                out[int(jid)] = int(v)
        return out

    def gains_for(self, joint_name: str) -> dict:
        """한 관절의 최종 게인. default -> group -> joint 순으로 덮어쓴다."""
        g = dict(self.default)
        gname = self.group_of(joint_name)
        if gname:
            g.update(self.groups[gname]["gains"])
        g.update(self.joints.get(joint_name, {}))
        return g

    # ------------------------------------------------------------------
    def build_id_map(self, joint_to_id: dict) -> dict:
        """{joint_name: dxl_id} → {dxl_id: gains} 로 변환."""
        return {int(jid): self.gains_for(jname)
                for jname, jid in joint_to_id.items()}

    def safety_warnings(self, joint_to_id: dict) -> list:
        """위험한 조합을 찾아 경고 문자열 목록으로 반환.

        P 를 기본값보다 크게 올리면서 D 가 0 이면 감쇠가 없어 발진한다.
        게인 튜닝에서 가장 흔한 실수라 명시적으로 잡는다.
        """
        warns = []
        for jname in joint_to_id:
            g = self.gains_for(jname)
            if g["position_p_gain"] > DEFAULT_GAINS["position_p_gain"] + 100 \
                    and g["position_d_gain"] == 0:
                warns.append(
                    f"{jname}: P={g['position_p_gain']} 인데 D=0 → 감쇠 없음. "
                    f"발진 위험")
        return warns

    def summary(self, joint_to_id: dict) -> str:
        """로그용 한 줄 요약. 그룹별로 묶어서 보여준다."""
        buckets = {}
        for jname in joint_to_id:
            g = self.gains_for(jname)
            key = (g["position_p_gain"], g["position_i_gain"], g["position_d_gain"])
            buckets.setdefault(key, []).append(jname)

        parts = []
        for (p, i, d), names in sorted(buckets.items(), key=lambda x: -len(x[1])):
            gname = self.group_of(names[0]) or "etc"
            parts.append(f"{gname}({len(names)}) P={p} I={i} D={d}")
        src = os.path.basename(self.path) if self.loaded else "기본값"
        return f"[{src}] " + ", ".join(parts)

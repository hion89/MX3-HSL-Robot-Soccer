# -*- coding: utf-8 -*-
# =============================================================================
#  cmd_shaping.py   (공용 명령 정형 — 데드밴드 / floor / clamp)
#
#  [무엇인가]
#  --------------------------------------------------------------------------
#  RL 보행정책(actor_onnx_*)은 cmd_vel 을 "명령값"으로 받는데, 작은 명령은
#  데드존이라 안 움직이고(전진 0.20 미만, 회전 0.50 미만 등), 큰 명령은 학습
#  범위를 벗어난다. 그래서 제어 노드가 만든 연속적인 속도 요청을 그대로 내보내면
#  로봇이 "움직였다 멈췄다"를 반복하거나(채터), 학습 범위 밖으로 발산한다.
#
#  visual_servo_kick_node 가 안정적으로 걷는 핵심 이유가 바로 이 "floor"다:
#    · 미세 명령은 0 으로(불필요한 떨림 제거)
#    · 실제로 내보낼 명령은 [min, max] 구간으로 끌어올림/잘라냄(데드존 회피 + 범위 보호)
#  이 동작을 pose_goal_controller_node 등 다른 노드도 "똑같이" 써야 거동이 일치하므로,
#  노드마다 복붙해 두 벌이 어긋나는 것을 막기 위해 순수 함수로 한 곳에 모았다.
#
#  [중요 — 값의 출처]
#  --------------------------------------------------------------------------
#  여기 기본 데드밴드(vx 1e-3, vy 0.02, wz 1.5°)와 호출부가 넘기는 min/max 는
#  keyboard_teleop_node / 정책 학습범위 / 실측 데드존에서 나온 값이다. 즉 이 모듈은
#  "정책의 명령 특성"을 인코딩한다. 정책을 바꾸면 호출부 파라미터(vx_min 등)를 바꾼다.
#  (이 모듈 자체는 정책 비의존 — 한계값을 인자로만 받는다.)
#
#  [config 중앙화가 아님]
#  --------------------------------------------------------------------------
#  각 노드는 여전히 자기 파라미터(vx_min/max ...)를 자기 패키지에서 소유한다.
#  여기 모인 것은 "설정"이 아니라 "동일 수식"뿐이다(순수 함수, 상태/파라미터 없음).
# =============================================================================

import math

# 데드밴드 기본값 — visual_servo_kick_node 의 _floor_* 와 동일하게 맞춤.
#   이 미만의 명령은 "사실상 0"으로 보고 버린다(불필요한 미세 떨림 차단).
DEFAULT_VX_DEADBAND = 1e-3            # 전진: 거의 0이면 정지
DEFAULT_VY_DEADBAND = 0.02           # 횡이동: 2cm/s 명령 미만은 무시
DEFAULT_WZ_DEADBAND_RAD = math.radians(1.5)  # 회전: 1.5° 미만 요청은 이미 정렬로 보고 0


def clamp(v: float, lo: float, hi: float) -> float:
    """v 를 [lo, hi] 구간으로 자른다."""
    return lo if v < lo else (hi if v > hi else v)


def wrap_angle(a: float) -> float:
    """각도를 (−π, π] 로 정규화."""
    return math.atan2(math.sin(a), math.cos(a))


def floor_vx(v: float, vx_min: float, vx_max: float,
             deadband: float = DEFAULT_VX_DEADBAND) -> float:
    """전진속도 floor.

    · |v| 가 deadband 미만 → 0 (정지. 전진은 항상 양수만 의미 있으므로 음수도 0 처리)
    · 그 외 → max(v, vx_min) 으로 데드존 위로 끌어올린 뒤 [0, vx_max] 로 clamp.
      (전진은 단독 동작 가능 — 보행 게이트 불필요. 후진은 이 함수가 다루지 않는다.)
    """
    if v <= deadband:
        return 0.0
    return clamp(max(v, vx_min), 0.0, vx_max)


def floor_wz(w: float, wz_min: float, wz_max: float,
             deadband: float = DEFAULT_WZ_DEADBAND_RAD) -> float:
    """회전속도 floor.

    · |w| 가 deadband(기본 1.5°) 미만 → 0 (이미 충분히 정렬됨)
    · 그 외 → 부호 유지하며 |w| 를 wz_min 이상으로 올린 뒤 [-wz_max, wz_max] clamp.
      (정책상 |명령wz| 가 wz_min(보통 0.50) 미만이면 로봇이 실제로 안 돈다.)
    """
    if abs(w) < deadband:
        return 0.0
    return clamp(math.copysign(max(abs(w), wz_min), w), -wz_max, wz_max)


def floor_vy(v: float, vy_min: float, vy_max: float,
             deadband: float = DEFAULT_VY_DEADBAND) -> float:
    """횡이동속도 floor.

    · |v| 가 deadband(기본 0.02) 미만 → 0
    · 그 외 → 부호 유지하며 |v| 를 vy_min 이상으로 올린 뒤 [-vy_max, vy_max] clamp.

    ★주의: vy 는 "보행 중(vx 또는 wz 활성)"일 때만 정책이 실제로 듣는다.
      그 보행 게이트(walking gate)는 이 함수가 아니라 '호출부'에서 처리한다.
      (호출부가 vx/wz 상태를 보고 vy 적용 여부를 결정해야 하므로.)
    """
    if abs(v) < deadband:
        return 0.0
    return clamp(math.copysign(max(abs(v), vy_min), v), -vy_max, vy_max)

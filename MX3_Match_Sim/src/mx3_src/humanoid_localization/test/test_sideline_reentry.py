# -*- coding: utf-8 -*-
# =============================================================================
#  test_sideline_reentry.py
#
#  sideline_reentry_pose() 단위 검증 (ROS 없이 standalone).
#       python3 -m pytest humanoid_localization/test/test_sideline_reentry.py
#   또는:
#       python3 humanoid_localization/test/test_sideline_reentry.py
#
#  [검증 목적]
#   1) SW1 False/True → 왼쪽(+y)/오른쪽(-y) 사이드라인 바깥
#   2) x 는 우리 골(-x) 근처, |y| = half_width + margin
#   3) 헤딩: 왼쪽 -90°, 오른쪽 +90° (필드 안쪽, Y축 평행)
#   4) 연습장↔대회 전환 시 좌표가 자동 스케일
#   5) start_yaw_deg override, sw1_false_is_left 반전, x_inset_m 동작
# =============================================================================

import math
import os
import sys

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from humanoid_localization.vision.field_geometry import (
    competition_field_geometry,
    practice_field_geometry,
    sideline_reentry_pose,
)


def _ang_close(a, b, tol=1e-9):
    d = math.atan2(math.sin(a - b), math.cos(a - b))
    return abs(d) < tol


def test_left_right_and_heading():
    g = competition_field_geometry()       # 9 x 6 → half (4.5, 3.0)
    margin = 0.4

    # SW1=False → 왼쪽(+y)
    x, y, yaw = sideline_reentry_pose(g, False, outside_margin_m=margin)
    assert abs(x - (-g.half_length_m)) < 1e-9          # 우리 골라인(-x)
    assert abs(y - (g.half_width_m + margin)) < 1e-9   # 왼쪽 사이드라인 바깥(+y)
    assert _ang_close(yaw, -math.pi / 2)               # 필드 안쪽(-y) 향함
    assert y > g.half_width_m                           # 필드 "바깥"

    # SW1=True → 오른쪽(-y)
    x2, y2, yaw2 = sideline_reentry_pose(g, True, outside_margin_m=margin)
    assert abs(x2 - (-g.half_length_m)) < 1e-9
    assert abs(y2 - (-(g.half_width_m + margin))) < 1e-9
    assert _ang_close(yaw2, math.pi / 2)               # 필드 안쪽(+y) 향함
    assert y2 < -g.half_width_m

    # 좌/우는 y 부호만 다르고 x 는 같다(둘 다 우리 골 쪽)
    assert abs(x - x2) < 1e-9 and abs(y + y2) < 1e-9
    print("[OK] 좌/우 + 헤딩(필드 안쪽) 규칙 통과")


def test_scales_with_profile():
    margin = 0.4
    gc = competition_field_geometry()
    gp = practice_field_geometry()

    _, yc, _ = sideline_reentry_pose(gc, False, outside_margin_m=margin)
    _, ypc, _ = sideline_reentry_pose(gp, False, outside_margin_m=margin)
    xc, _, _ = sideline_reentry_pose(gc, False)
    xp, _, _ = sideline_reentry_pose(gp, False)

    # 대회가 연습장보다 더 바깥/더 멀다(필드가 크므로)
    assert yc > ypc and abs(xc) > abs(xp)
    # margin 을 뺀 순수 폭 부분은 정확히 half_width 비율로 스케일
    assert abs((yc - margin) - gc.half_width_m) < 1e-9
    assert abs((ypc - margin) - gp.half_width_m) < 1e-9
    print(f"[OK] 프로파일 스케일: 대회 y={yc:.3f} > 연습 y={ypc:.3f}, "
          f"대회 x={xc:.3f} vs 연습 x={xp:.3f}")


def test_overrides():
    g = competition_field_geometry()

    # start_yaw_deg override → 그 각도 그대로
    _, _, yaw = sideline_reentry_pose(g, False, start_yaw_deg=0.0)
    assert _ang_close(yaw, 0.0)
    _, _, yaw2 = sideline_reentry_pose(g, True, start_yaw_deg=180.0)
    assert _ang_close(yaw2, math.pi)

    # sw1_false_is_left=False → False 가 오른쪽이 된다(반전)
    _, yL, _ = sideline_reentry_pose(g, False, sw1_false_is_left=True)
    _, yR, _ = sideline_reentry_pose(g, False, sw1_false_is_left=False)
    assert yL > 0 and yR < 0

    # x_inset_m → 골라인에서 안쪽(+x 방향)으로 당겨짐(절댓값 감소)
    x0, _, _ = sideline_reentry_pose(g, False, x_inset_m=0.0)
    x1, _, _ = sideline_reentry_pose(g, False, x_inset_m=0.5)
    assert abs(x1) < abs(x0) and abs((abs(x0) - abs(x1)) - 0.5) < 1e-9
    print("[OK] override(yaw) / sw1_false_is_left 반전 / x_inset 통과")


if __name__ == "__main__":
    test_left_right_and_heading()
    test_scales_with_profile()
    test_overrides()
    print("\n=== sideline_reentry 단위 테스트 통과 ===")

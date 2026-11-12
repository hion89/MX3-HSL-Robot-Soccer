# -*- coding: utf-8 -*-
# =============================================================================
#  test_vision_solver.py
#
#  vision 서브패키지 코어(geometry/landmarks/solver)의 단위 검증.
#  ROS 없이 standalone 으로 돌아간다:
#       python3 -m pytest humanoid_localization/test/test_vision_solver.py
#   또는 그냥:
#       python3 humanoid_localization/test/test_vision_solver.py
#
#  [검증 목적]
#   1) 랜드마크가 치수에서 올바르게 생성되는지(개수/대표 좌표)
#   2) 연습장↔대회 전환 시 좌표가 스케일되는지
#   3) solver 가 시드 오차를 받아 참 pose 를 복원하는지(round-trip)
#   4) 오검출(엉뚱한 관측)이 게이트로 걸러져 solver 가 망가지지 않는지
# =============================================================================

import math
import os
import sys

# 패키지 루트(setup.py 가 있는 humanoid_localization/)를 import 경로에 추가.
# 이 파일은 humanoid_localization/test/ 안에 있으므로 두 단계 위가 패키지 루트.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from humanoid_localization.vision.field_geometry import (
    competition_field_geometry,
    s_field_geometry,
    practice_field_geometry,
    field_geometry_for_profile,
)
from humanoid_localization.vision.field_landmarks import (
    generate_field_landmarks,
    count_landmarks,
    CLASS_GOALPOST, CLASS_X, CLASS_T, CLASS_L,
)
from humanoid_localization.vision.landmark_pose_solver import (
    Observation, LandmarkSolverConfig, LandmarkPoseSolver, normalize_angle,
)


def _has_point(landmarks, class_id, x, y, tol=1e-6):
    """class_id 안에 (x, y) 좌표를 가진 랜드마크가 있는지(부동소수 허용오차)."""
    for lm in landmarks.get(class_id, []):
        if abs(lm.x - x) < tol and abs(lm.y - y) < tol:
            return True
    return False


def _make_observation(true_pose, lm):
    """참 pose 에서 랜드마크 lm 을 봤을 때의 (range, bearing) 관측 생성.

    camera offset 0 가정(관측을 base_link 기준으로 넣는 설계와 동일).
    이는 solver.predict_measurement 의 역방향 계산이다.
    """
    x, y, yaw = true_pose
    dx = lm.x - x
    dy = lm.y - y
    rng = math.hypot(dx, dy)
    brg = normalize_angle(math.atan2(dy, dx) - yaw)
    return Observation(class_id=lm.class_id, range_m=rng, bearing_rad=brg)


def test_landmark_counts_and_competition_coords():
    """대회 규격(HSL 2026 S-Field)에서 개수와 대표 좌표가 맞는지.

    ※ 옛 KidSize 대비 바뀐 값 검증 포인트:
        - GP(골포스트) y = goal_width/2 = 1.8/2 = 0.9   (옛 1.3 아님)
        - 페널티에어리어 T/L y = penalty_area_width/2 = 4.0/2 = 2.0 (옛 2.5 아님)
    """
    lms = generate_field_landmarks(s_field_geometry())

    # 개수: GP 4, X 5, T 10, L 12 → 총 31 (생성 규칙은 동일, 치수만 S-Field)
    assert len(lms[CLASS_GOALPOST]) == 4
    assert len(lms[CLASS_X]) == 5
    assert len(lms[CLASS_T]) == 10
    assert len(lms[CLASS_L]) == 12
    assert count_landmarks(lms) == 31

    # HSL 2026 S-Field 대표 좌표가 그대로 생성되는지(샘플 검증)
    assert _has_point(lms, CLASS_GOALPOST, 4.5, 0.9)    # GP 골 기둥 (goal_width 1.8 → ±0.9)
    assert _has_point(lms, CLASS_X, 0.0, 0.0)           # 센터 마크
    assert _has_point(lms, CLASS_X, 3.0, 0.0)           # 페널티 마크 (4.5 - 1.5)
    assert _has_point(lms, CLASS_X, 0.0, 0.75)          # 센터서클 상단 (지름 1.5 → r 0.75)
    assert _has_point(lms, CLASS_T, 4.5, 1.5)           # 골에어리어×골라인 T (goal_area_w 3.0)
    assert _has_point(lms, CLASS_T, 4.5, 2.0)           # 페널티에어리어×골라인 T (pen_w 4.0 → ±2.0)
    assert _has_point(lms, CLASS_T, 0.0, 3.0)           # 하프웨이×사이드라인 T (field_w 6.0)
    assert _has_point(lms, CLASS_L, 4.5, 3.0)           # 필드 코너 L
    assert _has_point(lms, CLASS_L, 2.5, 2.0)           # 페널티에어리어 안쪽 코너 L (x 2.5, y 2.0)
    assert _has_point(lms, CLASS_L, 3.5, 1.5)           # 골에어리어 안쪽 코너 L
    print("[OK] S-Field 랜드마크 개수/좌표 검증 통과 (총 31점, GP ±0.9 / 페널티 ±2.0)")


def test_practice_scales_down():
    """연습장 프로파일에서 좌표가 0.433 스케일로 줄어드는지(대표점)."""
    lms = generate_field_landmarks(practice_field_geometry())
    # 개수는 동일(생성 규칙은 같고 치수만 다름)
    assert count_landmarks(lms) == 31
    # 필드 코너: (1.95, 1.30) = (3.9/2, 2.6/2)
    assert _has_point(lms, CLASS_L, 1.95, 1.30)
    # 골 기둥 y = goal_width/2 = 1.13/2 = 0.565
    assert _has_point(lms, CLASS_GOALPOST, 1.95, 0.565)
    # 페널티 마크 x = hl - 0.65 = 1.95 - 0.65 = 1.30
    assert _has_point(lms, CLASS_X, 1.30, 0.0)
    print("[OK] 연습장 스케일 검증 통과 (코너 1.95×1.30)")


def test_profile_selector():
    """프로파일 이름 선택기가 동작하고, 잘못된 이름은 에러를 내는지.

    s_field(정식) 과 competition(별칭) 이 같은 S-Field 값을 주는지,
    그리고 옛 KidSize 대비 바뀐 치수(goal_width 1.8, penalty_area_width 4.0)가
    실제로 반영됐는지 함께 확인한다.
    """
    assert field_geometry_for_profile("practice").length_m == 3.9
    # 외곽은 9 x 6 그대로
    assert field_geometry_for_profile("s_field").length_m == 9.0
    assert field_geometry_for_profile("competition").length_m == 9.0
    assert field_geometry_for_profile("hsl2026_small").length_m == 9.0

    # s_field == competition(별칭) — 같은 값을 반환해야 한다
    sf = field_geometry_for_profile("s_field")
    comp = field_geometry_for_profile("competition")
    assert sf == comp, "competition 별칭이 s_field 와 다른 값을 반환함"
    # 직접 함수 별칭도 동일해야 한다(competition_field_geometry == s_field_geometry)
    assert competition_field_geometry() == s_field_geometry()

    # ★ 바뀐 값이 실제로 반영됐는지(그대로 쓰면 안 되는 부분)
    assert sf.goal_width_m == 1.8, "goal_width 가 S-Field(1.8) 가 아님"
    assert sf.penalty_area_width_m == 4.0, "penalty_area_width 가 S-Field(4.0) 가 아님"
    # 그대로인 값(회귀 방지)
    assert sf.center_circle_radius_m == 0.75
    assert sf.penalty_area_depth_m == 2.0

    try:
        field_geometry_for_profile("bogus")
        assert False, "잘못된 프로파일인데 에러가 안 났음"
    except ValueError:
        pass
    print("[OK] 프로파일 선택기 검증 통과 (practice/s_field/competition 별칭 + 에러 처리)")


def test_solver_recovers_true_pose():
    """시드에 (현실적인) 오차를 줘도 solver 가 참 pose 를 복원하는지(round-trip).

    [수렴 basin — 실측 특성]
      이 solver 는 "시드 근처에서 정밀화하는 지역 보정기"다. sparse 한 관측
      (class 당 1~2개) + 기본 게이트에서 실측한 수렴 basin 은 약 ±0.12m / ±5°.
      그 밖이면 가까운 다른 랜드마크로 데이터 연관이 어긋나 실패할 수 있다.
      → 실제 시스템에선 EKF(map→base_link)가 20Hz 로 시드를 주므로 정상 동작
        중 시드 오차는 cm 수준이라 basin 안이다. 시작/재위치 시에는 /initialpose
        를 ±0.12m/±5° 안에 두면 된다. (관측 수가 많거나 게이트를 좁히면 basin 은 넓어진다)
    """
    geom = practice_field_geometry()
    landmarks = generate_field_landmarks(geom)
    cfg = LandmarkSolverConfig()
    solver = LandmarkPoseSolver(landmarks, cfg)

    # 참 pose (연습장 안의 임의 위치/방향)
    true_pose = (0.30, -0.20, 0.40)  # x, y, yaw[rad]

    # 시야에 들어올 법한 랜드마크 몇 개를 골라 관측 생성(서로 다른 class 로 형상 확보)
    chosen = [
        landmarks[CLASS_X][0],   # 센터 마크
        landmarks[CLASS_X][1],   # 페널티 마크
        landmarks[CLASS_T][0],   # T 하나
        landmarks[CLASS_L][0],   # L 코너 하나
        landmarks[CLASS_GOALPOST][0],  # 골 기둥 하나
    ]
    observations = [_make_observation(true_pose, lm) for lm in chosen]

    # 시드: 참값에서 (0.10m, 0.10m, 0.07rad≈4°) 어긋난 곳 — basin 안(EKF 시드 수준)
    seed = (true_pose[0] + 0.10, true_pose[1] + 0.10, true_pose[2] + 0.07)
    est, used = solver.solve_pose_from_seed(seed[0], seed[1], seed[2], observations)

    assert used >= cfg.min_observations, f"매칭 수 부족: {used}"
    ex = abs(est[0] - true_pose[0])
    ey = abs(est[1] - true_pose[1])
    eyaw = abs(normalize_angle(est[2] - true_pose[2]))
    assert ex < 0.02 and ey < 0.02 and eyaw < 0.02, (
        f"복원 오차 큼: dx={ex:.4f} dy={ey:.4f} dyaw={eyaw:.4f}"
    )
    print(f"[OK] solver 복원 통과: 매칭 {used}개, 오차 "
          f"dx={ex*1000:.1f}mm dy={ey*1000:.1f}mm dyaw={math.degrees(eyaw):.3f}deg")


def test_solver_rejects_outlier():
    """엉뚱한(오검출) 관측이 게이트로 걸러지고 복원이 유지되는지."""
    geom = practice_field_geometry()
    landmarks = generate_field_landmarks(geom)
    cfg = LandmarkSolverConfig()
    solver = LandmarkPoseSolver(landmarks, cfg)

    true_pose = (0.10, 0.15, -0.30)
    good = [
        landmarks[CLASS_X][0],
        landmarks[CLASS_T][2],
        landmarks[CLASS_L][1],
        landmarks[CLASS_GOALPOST][1],
    ]
    observations = [_make_observation(true_pose, lm) for lm in good]
    # 오검출: 실제와 동떨어진 거리/방위의 X 관측 한 개 추가(게이트로 걸려야 함)
    observations.append(Observation(class_id=CLASS_X, range_m=5.0, bearing_rad=2.5))

    seed = (true_pose[0] - 0.08, true_pose[1] + 0.08, true_pose[2] - 0.06)
    est, used = solver.solve_pose_from_seed(seed[0], seed[1], seed[2], observations)

    ex = abs(est[0] - true_pose[0])
    ey = abs(est[1] - true_pose[1])
    eyaw = abs(normalize_angle(est[2] - true_pose[2]))
    assert ex < 0.03 and ey < 0.03 and eyaw < 0.03, (
        f"오검출에 끌려감: dx={ex:.4f} dy={ey:.4f} dyaw={eyaw:.4f}"
    )
    print(f"[OK] 오검출 거부 통과: 오차 "
          f"dx={ex*1000:.1f}mm dy={ey*1000:.1f}mm dyaw={math.degrees(eyaw):.3f}deg")


if __name__ == "__main__":
    # pytest 없이 직접 실행할 수 있게 한다.
    test_landmark_counts_and_competition_coords()
    test_practice_scales_down()
    test_profile_selector()
    test_solver_recovers_true_pose()
    test_solver_rejects_outlier()
    print("\n=== 모든 vision 코어 단위 테스트 통과 ===")

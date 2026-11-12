# -*- coding: utf-8 -*-
# =============================================================================
#  field_landmarks.py
#
#  경기장 치수(FieldGeometry)로부터 점 랜드마크(goalpost/X/T/L)를 "계산"해
#  solver 가 바로 쓰는 {class_id: [Landmark, ...]} 형태로 만든다.
#
#  [왜 하드코딩 YAML 대신 생성하는가 — 근본 원인 접근]
#  --------------------------------------------------------------------------
#   랜드마크 좌표는 결국 경기장 치수의 함수다. 손으로 쓴 landmarks.yaml 을
#   연습장/대회용으로 따로 유지하면, 치수가 바뀔 때마다 두 파일을 수동 동기화해야
#   하고 그게 오차/불일치의 원인이 된다. 치수에서 "생성"하면:
#       프로파일만 바꾸면 → 랜드마크 지도 전체가 같은 스케일로 자동 갱신.
#   즉 연습장 ↔ 대회 전환이 파라미터 한 개로 끝난다.
#
#  [클래스 ID 규약 — 검출기와 반드시 일치]
#  --------------------------------------------------------------------------
#   humanoid_perception/yolo_detector_3d_node.py 의 class_names 인덱스와 일치:
#       0:ball 1:goalpost 2:robot 3:field_edge 4:line
#       5:X-Intersection 6:T-Intersection 7:L-Intersection 8:obstacle
#   이 중 "정적인 점 특징"만 랜드마크로 쓴다:
#       goalpost(1), X(5), T(6), L(7)
#   ball/robot/obstacle(0/2/8)은 움직이므로 앵커에서 제외,
#   line/field_edge(3/4)는 점이 아니라 선이므로 이 단순 solver 에선 제외.
#   ※ goalpost 는 검출 신뢰도가 높지만 "기둥(높이)"이라 3D 점의 지면투영을
#     넣어야 한다(노드 쪽 책임). 높이 편향이 우려되면 landmark_classes 에서
#     1 을 빼 바닥 마커(X/T/L)만으로도 풀 수 있게 분리해 둔다.
#
#  [좌표계] khuber 규약(중앙 원점, +x 길이=골대 방향, +y 폭=왼쪽), 단위 m.
# =============================================================================

from typing import Dict, List, Optional, Sequence, Tuple

from .field_geometry import FieldGeometry, default_field_geometry
from .landmark_pose_solver import Landmark


# --- 검출기 클래스 ID 상수 (yolo_detector_3d_node.class_names 인덱스와 동일) ---
CLASS_GOALPOST = 1   # 'goalpost'
CLASS_X = 5          # 'X-Intersection'
CLASS_T = 6          # 'T-Intersection'
CLASS_L = 7          # 'L-Intersection'

# 기본으로 로컬라이제이션에 쓸 랜드마크 클래스(정적 점 특징만)
DEFAULT_LANDMARK_CLASSES: Tuple[int, ...] = (CLASS_GOALPOST, CLASS_X, CLASS_T, CLASS_L)


def _quad(x: float, y: float) -> List[Tuple[float, float]]:
    """(x, y) 의 4분면 대칭점 [(±x, ±y)] 를 만든다.

    경기장은 가로/세로 양 축에 대해 대칭이라, 한 점만 정의하면 부호 조합으로
    네 점이 자동으로 나온다. (x 또는 y 가 0 이면 중복은 호출부에서 처리)
    """
    return [(x, y), (x, -y), (-x, y), (-x, -y)]


def generate_field_landmarks(
    geometry: Optional[FieldGeometry] = None,
    landmark_classes: Sequence[int] = DEFAULT_LANDMARK_CLASSES,
    flip_axes: bool = False,
) -> Dict[int, List[Landmark]]:
    """경기장 치수로부터 {class_id: [Landmark, ...]} 랜드마크 지도를 생성한다.

    geometry        : 대상 경기장 치수. None 이면 기본 프로파일(연습장).
    landmark_classes: 생성/포함할 클래스들. 예) goalpost 를 빼고 싶으면 (5,6,7).
    flip_axes       : turtle X1/X3/X2/X4 규약이 필요할 때만 True(기본 khuber 규약).

    생성 규칙(모두 치수에서 파생 — turtle landmarks.yaml 의 배치와 동일):
      GP(goalpost): 골라인 위 골 기둥           (±hl, ±goal_half_width)
      X           : 센터 + 양 페널티 마크 + 센터서클 상/하
      T           : 골라인×골에어리어폭, 골라인×페널티에어리어폭, 하프웨이×사이드라인
      L           : 필드 코너, 페널티에어리어 안쪽 코너, 골에어리어 안쪽 코너
    """
    geom = geometry or default_field_geometry()
    want = set(int(c) for c in landmark_classes)

    hl = geom.half_length_m
    hw = geom.half_width_m
    pwh = geom.penalty_area_half_width_m
    pxi = geom.penalty_area_inner_x_m
    gwh = geom.goal_area_half_width_m
    gxi = geom.goal_area_inner_x_m
    pmx = geom.penalty_mark_x_m
    cr = geom.center_circle_radius_m
    ghw = geom.goal_half_width_m

    # 각 class 별 "원시 점 목록" (아직 부호 대칭 전개 전, 의미 단위로 모음)
    raw: Dict[int, List[Tuple[str, float, float]]] = {}

    # --- goalpost(1): 골라인 위 두 기둥 × 양쪽 골대 = 4점 ---
    if CLASS_GOALPOST in want:
        pts = []
        for i, (x, y) in enumerate(_quad(hl, ghw)):
            pts.append((f"GP{i+1}", x, y))
        raw[CLASS_GOALPOST] = pts

    # --- X(5): 센터(1) + 페널티 마크(2) + 센터서클 상/하(2) = 5점 ---
    if CLASS_X in want:
        pts = [("X_center", 0.0, 0.0)]
        pts.append(("X_penalty_pos", pmx, 0.0))    # +x 쪽 페널티 마크
        pts.append(("X_penalty_neg", -pmx, 0.0))   # -x 쪽 페널티 마크
        pts.append(("X_circle_top", 0.0, cr))      # 센터서클 상단(라인 교차)
        pts.append(("X_circle_bot", 0.0, -cr))     # 센터서클 하단
        raw[CLASS_X] = pts

    # --- T(6): 골라인×골에어리어폭(4) + 골라인×페널티에어리어폭(4)
    #            + 하프웨이라인×사이드라인(2) = 10점 ---
    if CLASS_T in want:
        pts = []
        idx = 1
        for x, y in _quad(hl, gwh):                 # 골라인에서 골에어리어 폭이 만나는 T
            pts.append((f"T_goalarea{idx}", x, y)); idx += 1
        for x, y in _quad(hl, pwh):                 # 골라인에서 페널티에어리어 폭이 만나는 T
            pts.append((f"T_penarea{idx}", x, y)); idx += 1
        pts.append(("T_half_pos", 0.0, hw))         # 하프웨이라인이 +y 사이드라인과 만나는 T
        pts.append(("T_half_neg", 0.0, -hw))        # -y 쪽
        raw[CLASS_T] = pts

    # --- L(7): 필드 코너(4) + 페널티에어리어 안쪽 코너(4)
    #           + 골에어리어 안쪽 코너(4) = 12점 ---
    if CLASS_L in want:
        pts = []
        idx = 1
        for x, y in _quad(hl, hw):                  # 필드 4 코너
            pts.append((f"L_corner{idx}", x, y)); idx += 1
        for x, y in _quad(pxi, pwh):                # 페널티에어리어 안쪽 코너
            pts.append((f"L_penarea{idx}", x, y)); idx += 1
        for x, y in _quad(gxi, gwh):                # 골에어리어 안쪽 코너
            pts.append((f"L_goalarea{idx}", x, y)); idx += 1
        raw[CLASS_L] = pts

    # --- 좌표 규약 적용(flip_axes) 후 Landmark 객체로 변환 ---
    out: Dict[int, List[Landmark]] = {}
    for class_id, items in raw.items():
        lms: List[Landmark] = []
        for lm_id, x, y in items:
            mx, my = (-x, -y) if flip_axes else (x, y)
            lms.append(Landmark(lm_id=lm_id, class_id=class_id, x=float(mx), y=float(my)))
        out[class_id] = lms
    return out


def count_landmarks(landmarks: Dict[int, List[Landmark]]) -> int:
    """전체 랜드마크 개수(디버그/검증용)."""
    return sum(len(v) for v in landmarks.values())

# -*- coding: utf-8 -*-
# =============================================================================
#  field_geometry.py
#
#  경기장(축구 필드) 치수를 한 곳에서 정의한다.
#  로컬라이제이션에 쓰는 모든 점 랜드마크(GP/X/T/L) 좌표는 여기서 정의한
#  치수로부터 "계산"되므로(field_landmarks.py 참조), 이 파일의 치수만 바꾸면
#  랜드마크 지도 전체가 같은 스케일로 자동 갱신된다.
#
#  [연습장 ↔ 대회 규격 전환]
#  --------------------------------------------------------------------------
#  지금은 연습장(3.9 x 2.6, 인조 잔디 5x3m 안의 라인)으로 진행하지만,
#  대회(HSL 2026 small division, 9 x 6m)로 바꿀 여지를 남긴다.
#  전환 방법은 단 하나 — 프로파일 이름만 바꾼다:
#       geom = field_geometry_for_profile("practice")   # 지금(연습장)
#       geom = field_geometry_for_profile("s_field")    # 대회(= competition 별칭)
#  ※ "competition" 은 하위호환 별칭이라 그대로 써도 S-Field 값이 나온다.
#  (또는 YAML 로 임의 치수를 덮어쓰려면 load_field_geometry(path) 사용)
#
#  [좌표계 — khuber 규약과 일치]
#  --------------------------------------------------------------------------
#  field_lines_node 와 동일하게:
#     - 원점        : 필드 정중앙
#     - +x          : 길이 방향(골대 쪽)
#     - +y          : 폭 방향(왼쪽)
#     - 단위        : meter, 각도 rad
#  turtle 코드에 있던 X1/X2/X3/X4 축 뒤집기(use_x4_axis_frame)는 기본적으로
#  쓰지 않는다(혼동 방지). 필요 시 flip_axes 인자로만 명시적으로 켠다.
#  ※ 프레임 혼동은 로컬라이제이션 실패의 단골 원인이므로, 한 가지 규약으로
#    통일하고 변환은 경계(외부 인터페이스)에서만 한다.
# =============================================================================

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml


@dataclass(frozen=True)
class FieldGeometry:
    """경기장 치수 묶음(단위 m). frozen=True 라 생성 후 불변.

    여기에 들어가는 값은 "실제 라인 도면" 기준의 물리 치수다.
    half_*, *_inner_x, penalty_mark_x 등 파생값은 property 로 계산한다.
    """

    length_m: float                                # 필드 라인 외곽 길이(골라인~골라인)
    width_m: float                                 # 필드 라인 외곽 폭(사이드라인~사이드라인)
    center_circle_radius_m: float                  # 센터 서클 반지름
    penalty_mark_distance_from_goal_line_m: float  # 골라인에서 페널티 마크까지 거리
    penalty_area_depth_m: float                    # 페널티 에어리어 깊이(골라인 기준 안쪽으로)
    penalty_area_width_m: float                    # 페널티 에어리어 폭
    goal_area_depth_m: float                       # 골 에어리어 깊이
    goal_area_width_m: float                       # 골 에어리어 폭
    goal_width_m: float                            # 골대 두 기둥 사이 간격
    spot_cross_half_size_m: float = 0.09           # 페널티/센터 마크 십자 반쪽 길이(시각화용)

    # --- 파생 치수(자주 쓰는 좌표 계산 단축) ---
    @property
    def half_length_m(self) -> float:
        """필드 절반 길이. 골라인 x 좌표 = ±half_length_m."""
        return 0.5 * self.length_m

    @property
    def half_width_m(self) -> float:
        """필드 절반 폭. 사이드라인 y 좌표 = ±half_width_m."""
        return 0.5 * self.width_m

    @property
    def penalty_area_inner_x_m(self) -> float:
        """페널티 에어리어 안쪽 라인의 x(골라인에서 depth 만큼 안쪽)."""
        return self.half_length_m - self.penalty_area_depth_m

    @property
    def penalty_area_half_width_m(self) -> float:
        """페널티 에어리어 폭의 절반(±y)."""
        return 0.5 * self.penalty_area_width_m

    @property
    def goal_area_inner_x_m(self) -> float:
        """골 에어리어 안쪽 라인의 x."""
        return self.half_length_m - self.goal_area_depth_m

    @property
    def goal_area_half_width_m(self) -> float:
        """골 에어리어 폭의 절반(±y)."""
        return 0.5 * self.goal_area_width_m

    @property
    def penalty_mark_x_m(self) -> float:
        """페널티 마크 x(골라인에서 안쪽으로 penalty_mark_distance)."""
        return self.half_length_m - self.penalty_mark_distance_from_goal_line_m

    @property
    def goal_half_width_m(self) -> float:
        """골 기둥 y 좌표 = ±goal_half_width_m."""
        return 0.5 * self.goal_width_m


# =============================================================================
#  프로파일 — 연습장 / 대회 규격
# =============================================================================
def s_field_geometry() -> FieldGeometry:
    """RoboCup HSL 2026 'S-Field' (small division) 대회 규격.

    [출처] HSL-Rules 2026, environment.tex Table tab:field_dim 'S-Field' 열.
      헤더에 'S-Field (former KidSize and SPL)' 로 명시 — 옛 KidSize 와 SPL 을
      통합한 필드다. small division 은 max 110cm / 15kg, foundation 4명/팀이라
      KHUBER(600mm, 4대 fleet)가 뛰는 바로 그 division 이다.

    [★ 옛 KidSize(9x6) 대비 바뀐 값 — 그대로 쓰면 안 되는 부분]
      - goal_width        2.6 → 1.8   (룰북 범위 1.8~1.9; 다이어그램 기본 1.8 채택)
                          → goal_half_width 1.3→0.9. 골포스트(GP) 랜드마크가
                            y로 0.4m 안쪽 이동. 측위 앵커·킥 조준점에 직접 영향.
      - penalty_area_width 5.0 → 4.0
                          → penalty_area_half_width 2.5→2.0. T_penarea / L_penarea
                            랜드마크가 y로 0.5m 안쪽 이동.
      - mark size(half)   0.09 → 0.05 (룰북 전체 0.10) — 시각화상 영향만.

    [그대로인 값 — 옛 규격과 동일하므로 손대지 않음]
      length 9.0, width 6.0, center circle radius 0.75(지름 1.5),
      penalty mark distance 1.5, penalty area depth 2.0, goal area 1.0 x 3.0.

    [참고: 이 2D geometry 에 없는 small division 신규값(현장 반영 대상)]
      골 깊이 0.5~1.0 / 골 높이 1.2~1.3 / 골포스트 지름 7~12cm(시각화 viz 파라미터),
      보더 스트립 1.0m(재진입 outside_margin 재검토용),
      공 = SPL 볼 10cm 또는 FIFA mini size1 14cm(perception 튜닝),
      잔디 8~12mm(다른 division 20~30mm보다 짧음 — 킥 grass-lean 완화 가능).
      ※ 골대 폭은 룰북상 1.8~1.9 범위이므로, 정밀이 필요하면 현장 실측값으로
        이 함수의 goal_width_m 만 교체하면 GP 랜드마크가 자동으로 따라간다.
    """
    return FieldGeometry(
        length_m=9.0,
        width_m=6.0,
        center_circle_radius_m=0.75,                   # 지름 1.5 / 2
        penalty_mark_distance_from_goal_line_m=1.5,
        penalty_area_depth_m=2.0,                       # 룰북 G(length)
        penalty_area_width_m=4.0,                       # 룰북 H — 5.0 에서 변경
        goal_area_depth_m=1.0,                          # 룰북 E
        goal_area_width_m=3.0,                          # 룰북 F
        goal_width_m=1.8,                               # 룰북 D(1.8~1.9) — 2.6 에서 변경
        spot_cross_half_size_m=0.05,                    # 룰북 전체 0.10 의 절반
    )


def competition_field_geometry() -> FieldGeometry:
    """하위호환 별칭 — 기존 import / field_profile='competition' 을 위해 유지.

    'competition' 이라는 이름은 의미가 모호해(어느 division?) s_field 로 정식화했다.
    기존 코드/yaml/launch 가 'competition' 을 그대로 쓰더라도 깨지지 않도록,
    이 함수는 s_field_geometry() 를 그대로 반환한다(= 올바른 S-Field 값).
    """
    return s_field_geometry()


def practice_field_geometry() -> FieldGeometry:
    """연습장 규격(3.9 x 2.6 m).

    인조 잔디 5x3m 안에 그린 라인으로, 대회 규격을 균일 축척 0.433(=2.6/6)
    으로 줄인 값이다. 수치는 humanoid_localization/config/field.yaml
    (field_lines_node 시각화용)과 일치시켜, 시각화 라인과 로컬라이제이션
    랜드마크가 같은 좌표계를 쓰도록 한다.

    ★ 로컬라이제이션에 쓰는 것은 점 랜드마크의 (x, y) 뿐이므로, 도면상의
      라인 좌표(균일 축척)만 정확하면 된다. (골대 "높이"는 슈팅 연습용으로
      축소하지 않지만, 그 값은 2D 위치추정과 무관하므로 여기서 다루지 않는다.)
    """
    return FieldGeometry(
        length_m=3.9,                                 # 9.0 × 0.433
        width_m=2.6,                                  # 6.0 × 0.433
        center_circle_radius_m=0.35,                  # 도면 실측값(0.325→0.35 상향)
        penalty_mark_distance_from_goal_line_m=0.65,  # 1.5 × 0.433
        penalty_area_depth_m=0.87,                    # 2.0 × 0.433
        penalty_area_width_m=2.17,                    # 5.0 × 0.433
        goal_area_depth_m=0.43,                       # 1.0 × 0.433
        goal_area_width_m=1.3,                        # 3.0 × 0.433
        goal_width_m=1.13,                            # 2.6 × 0.433
        spot_cross_half_size_m=0.08,
    )


# 프로파일 이름 → 생성 함수. 새 프로파일을 추가하려면 여기에 한 줄 등록한다.
#   - "practice"            : 연습장 3.9 x 2.6
#   - "s_field"/"hsl2026_small" : HSL 2026 대회 small division(9 x 6, KHUBER 필드) — 정식 이름
#   - "competition"         : s_field 의 하위호환 별칭(기존 yaml/launch 가 그대로 동작)
_PROFILES = {
    "practice": practice_field_geometry,
    "s_field": s_field_geometry,
    "hsl2026_small": s_field_geometry,
    "competition": s_field_geometry,   # 하위호환 별칭 → 올바른 S-Field 값
}

# 기본 프로파일(현재 진행: 연습장). 대회 때 "competition" 으로 바꾸면 끝.
DEFAULT_PROFILE = "practice"


def field_geometry_for_profile(profile: str = DEFAULT_PROFILE) -> FieldGeometry:
    """프로파일 이름으로 FieldGeometry 를 얻는다.

    profile: "practice"(연습장, 기본) 또는 "competition"(대회 규격).
    잘못된 이름이면 사용 가능한 목록과 함께 명확히 에러를 낸다(조용한 오동작 금지).
    """
    key = (profile or DEFAULT_PROFILE).strip().lower()
    if key not in _PROFILES:
        raise ValueError(
            f"알 수 없는 field_profile='{profile}'. "
            f"사용 가능: {sorted(_PROFILES.keys())}"
        )
    return _PROFILES[key]()


def default_field_geometry() -> FieldGeometry:
    """현재 기본 프로파일(연습장)의 FieldGeometry. 호환용 단축 함수."""
    return field_geometry_for_profile(DEFAULT_PROFILE)


def load_field_geometry(path: Union[str, Path, None]) -> FieldGeometry:
    """YAML 파일에서 임의 치수를 읽어 FieldGeometry 를 만든다(전체 override).

    프로파일(연습장/대회)로 부족하고 특정 치수를 직접 지정하고 싶을 때만 사용.
    path 가 None/없는 파일이면 기본 프로파일을 반환한다(안전한 폴백).

    YAML 형식 예:
        field:
          length_m: 3.9
          width_m: 2.6
          center_circle_radius_m: 0.35
          penalty_mark_distance_from_goal_line_m: 0.65
          penalty_area: { depth_m: 0.87, width_m: 2.17 }
          goal_area:    { depth_m: 0.43, width_m: 1.3 }
          goal:         { width_m: 1.13 }
    """
    if not path:
        return default_field_geometry()

    geom_path = Path(path)
    if not geom_path.exists():
        return default_field_geometry()

    with geom_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    field = data.get("field", {})
    if not isinstance(field, dict):
        return default_field_geometry()

    base = default_field_geometry()

    def _sub(key: str) -> dict:
        # 중첩 dict(penalty_area/goal_area/goal) 안전 추출
        val = field.get(key)
        return val if isinstance(val, dict) else {}

    return FieldGeometry(
        length_m=float(field.get("length_m", base.length_m)),
        width_m=float(field.get("width_m", base.width_m)),
        center_circle_radius_m=float(
            field.get("center_circle_radius_m", base.center_circle_radius_m)
        ),
        penalty_mark_distance_from_goal_line_m=float(
            field.get(
                "penalty_mark_distance_from_goal_line_m",
                base.penalty_mark_distance_from_goal_line_m,
            )
        ),
        penalty_area_depth_m=float(
            _sub("penalty_area").get("depth_m", base.penalty_area_depth_m)
        ),
        penalty_area_width_m=float(
            _sub("penalty_area").get("width_m", base.penalty_area_width_m)
        ),
        goal_area_depth_m=float(
            _sub("goal_area").get("depth_m", base.goal_area_depth_m)
        ),
        goal_area_width_m=float(
            _sub("goal_area").get("width_m", base.goal_area_width_m)
        ),
        goal_width_m=float(_sub("goal").get("width_m", base.goal_width_m)),
        spot_cross_half_size_m=float(
            field.get("spot_cross_half_size_m", base.spot_cross_half_size_m)
        ),
    )


# =============================================================================
#  재진입 시작 위치 — SW1(좌/우) 기반 사이드라인 바깥 시드 pose
# =============================================================================
def sideline_reentry_pose(
    geom: FieldGeometry,
    sw1: bool,
    *,
    own_goal_sign: int = -1,
    sw1_false_is_left: bool = True,
    outside_margin_m: float = 0.4,
    x_inset_m: float = 0.0,
    start_yaw_deg: Optional[float] = None,
) -> Tuple[float, float, float]:
    """SW1(좌/우 토글)에 따른 재진입 시작 pose (x, y, yaw) 를 계산한다.

    퇴장/고장 후 재진입 시, 우리 골대 쪽 사이드라인 "바깥"의 정해진 자리에
    로봇을 세우고 실행버튼을 누르면 이 pose 로 로컬라이저를 시드한다.
    (기존 RViz '2D Pose Estimate' 마우스 클릭을 하드웨어 입력으로 자동화)

    좌표/각도 규약은 이 파일 상단과 동일:
      원점=필드 중앙, +x=길이(골대 쪽), +y=폭(왼쪽), 단위 m/rad.

    규칙:
      - 우리 진영은 항상 -x (own_goal_sign=-1 고정).
      - sw1_false_is_left=True 면  SW1=False → 왼쪽(+y),  SW1=True → 오른쪽(-y).
      - x : 우리 골라인 근처(own_goal_sign*half_length, x_inset_m 만큼 안쪽으로 당김 가능).
      - y : 사이드라인 바깥(|y| = half_width + outside_margin_m).
      - yaw: 필드 안쪽을 향함(Y축 평행). 왼쪽(+y)에 서면 -y(-90°), 오른쪽(-y)이면 +y(+90°).
             start_yaw_deg 가 주어지면 그 값으로 강제(물리적으로 다르게 세우는 경우만).

    이 함수는 순수 함수다(ROS 의존 없음). 프로파일이 바뀌면 geom 만 바꿔 호출하면
    같은 규칙으로 좌표가 자동 스케일된다.

    반환: (x, y, yaw)  — map(필드) 프레임의 시드 pose.
    """
    is_left = (not bool(sw1)) if sw1_false_is_left else bool(sw1)   # 왼쪽 = +y
    y_sign = 1.0 if is_left else -1.0

    x = float(own_goal_sign) * (geom.half_length_m - float(x_inset_m))
    y = y_sign * (geom.half_width_m + float(outside_margin_m))
    if start_yaw_deg is not None:
        yaw = math.radians(float(start_yaw_deg))
    else:
        yaw = -0.5 * math.pi if is_left else 0.5 * math.pi          # 필드 안쪽

    return float(x), float(y), float(yaw)


# =============================================================================
#  (선택) 라인 세그먼트 / 센터 서클 점 생성 — 주로 RViz 시각화·샘플링용
#  로컬라이제이션 코어는 점 랜드마크만 쓰므로 필수는 아니지만, 디버그/표시에
#  유용해 함께 둔다. khuber 규약(중앙 원점, +x 길이, +y 폭)을 그대로 따른다.
# =============================================================================
def _maybe_flip(x: float, y: float, flip_axes: bool) -> Tuple[float, float]:
    """flip_axes=True 면 turtle 의 X1/X3/X2/X4 규약(x,y 부호 반전)으로 변환.

    기본(False)은 khuber 규약 그대로. 호환이 필요한 외부 도구가 있을 때만 켠다.
    """
    if not flip_axes:
        return float(x), float(y)
    return float(-x), float(-y)


def generate_field_segments(
    geometry: Optional[FieldGeometry] = None,
    flip_axes: bool = False,
) -> List[Tuple[float, float, float, float]]:
    """필드 라인을 (x0, y0, x1, y1) 선분 리스트로 생성한다(시각화/라인매칭용)."""
    geom = geometry or default_field_geometry()
    segs: List[Tuple[float, float, float, float]] = []
    hl = geom.half_length_m
    hw = geom.half_width_m
    pxi = geom.penalty_area_inner_x_m
    pwh = geom.penalty_area_half_width_m
    gxi = geom.goal_area_inner_x_m
    gwh = geom.goal_area_half_width_m
    pmx = geom.penalty_mark_x_m
    cross = geom.spot_cross_half_size_m

    # 외곽 사각형 + 하프웨이 라인
    segs += [(-hl, hw, hl, hw), (-hl, -hw, hl, -hw)]
    segs += [(-hl, -hw, -hl, hw), (hl, -hw, hl, hw)]
    segs += [(0.0, -hw, 0.0, hw)]

    # 페널티 에어리어(양쪽)
    segs += [(pxi, -pwh, pxi, pwh), (pxi, pwh, hl, pwh), (pxi, -pwh, hl, -pwh)]
    segs += [(-pxi, -pwh, -pxi, pwh), (-hl, pwh, -pxi, pwh), (-hl, -pwh, -pxi, -pwh)]

    # 골 에어리어(양쪽)
    segs += [(gxi, -gwh, gxi, gwh), (gxi, gwh, hl, gwh), (gxi, -gwh, hl, -gwh)]
    segs += [(-gxi, -gwh, -gxi, gwh), (-hl, gwh, -gxi, gwh), (-hl, -gwh, -gxi, -gwh)]

    # 센터 마크 + 페널티 마크(십자)
    segs += [(-cross, 0.0, cross, 0.0), (0.0, -cross, 0.0, cross)]
    segs += [(pmx - cross, 0.0, pmx + cross, 0.0), (pmx, -cross, pmx, cross)]
    segs += [(-pmx - cross, 0.0, -pmx + cross, 0.0), (-pmx, -cross, -pmx, cross)]

    out = []
    for x0, y0, x1, y1 in segs:
        mx0, my0 = _maybe_flip(x0, y0, flip_axes)
        mx1, my1 = _maybe_flip(x1, y1, flip_axes)
        out.append((mx0, my0, mx1, my1))
    return out


def generate_center_circle_points(
    geometry: Optional[FieldGeometry] = None,
    flip_axes: bool = False,
    num_points: int = 72,
) -> List[Tuple[float, float]]:
    """센터 서클을 점 리스트로 근사 생성한다(시각화용)."""
    geom = geometry or default_field_geometry()
    r = geom.center_circle_radius_m
    pts = []
    for i in range(num_points):
        th = 2.0 * math.pi * i / float(num_points)
        pts.append(_maybe_flip(r * math.cos(th), r * math.sin(th), flip_axes))
    return pts

# -*- coding: utf-8 -*-
# =============================================================================
#  landmark_pose_solver.py
#
#  "점 랜드마크에 대한 (거리, 방위) 관측"으로 로봇의 2D pose(x, y, yaw)를
#  추정하는 Gauss-Newton solver 코어.
#
#  [무엇을 받고 무엇을 내는가]
#  --------------------------------------------------------------------------
#   입력:
#     - landmarks    : 지도상 알려진 랜드마크 위치 {class_id: [Landmark(x, y), ...]}
#                      (field_landmarks.py 가 경기장 치수로부터 생성)
#     - observations : 현재 카메라로 본 랜드마크들의 [Observation(class_id, range, bearing)]
#                      (range/bearing 은 "로봇 base_link 기준"으로 변환해 넣는다)
#     - seed pose    : 반복 시작점 (x0, y0, yaw0). 보통 직전 추정/EKF 피드백.
#   출력:
#     - (x, y, yaw) 추정 + 실제로 매칭에 쓰인 관측 수
#
#  [핵심 원리]
#  --------------------------------------------------------------------------
#   각 랜드마크에 대해 "현재 추정 pose 라면 이렇게 보일 것"(예측 range/bearing)을
#   계산하고, 실제 관측과의 차이(residual)를 최소화하도록 pose 를 조금씩 보정한다.
#       residual = [관측거리 - 예측거리, normalize(관측방위 - 예측방위)]
#   이를 모든 매칭 관측에 대해 모아 정규방정식(H·Δ = -g)을 풀어 Δ(pose 보정량)를
#   구하고, 스텝을 제한(clip)하며 max_iterations 만큼 반복한다.
#
#  [과거 실패와의 차이 — 의도적으로 뺀 것]
#  --------------------------------------------------------------------------
#   - turtle 원본에 있던 "grid 후보 disambiguation"(여러 셀 후보를 세우고 점수화)을
#     통째로 제거했다. 시드(seed)는 EKF/odom 피드백이 항상 제공하므로 필요 없고,
#     그 로직이 튜닝을 어렵게 만든 복잡도의 핵심이었다.
#   - 즉 이 코어는 "시드 근처에서 관측에 맞게 pose 를 정밀화"하는 한 가지 일만 한다.
#     전역 재위치(global relocalization)는 EKF + initialpose 가 담당한다.
#
#  [중요]
#   - 이 모듈은 ROS 의존성이 없다(numpy/math 만). standalone 으로 단위 검증 가능.
#   - 관측을 base_link 기준으로 넣으므로 camera offset 은 0 이 기본값이다.
#     (카메라→base_link 변환은 노드 쪽에서 TF 로 처리)
# =============================================================================

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Landmark:
    """지도상 알려진 랜드마크 한 개.

    lm_id    : 사람이 읽기 위한 식별 문자열(예: "GP1", "L3_corner"). 로직엔 영향 없음.
    class_id : 검출기 클래스 ID(goalpost=1, X=5, T=6, L=7). 같은 class 끼리만 매칭.
    x, y     : 지도(map) 좌표 [m]. khuber 규약(중앙 원점, +x 길이, +y 폭).
    """

    lm_id: str
    class_id: int
    x: float
    y: float


@dataclass
class Observation:
    """현재 프레임에서 본 랜드마크 한 개의 관측값(로봇 base_link 기준).

    class_id    : 검출기 클래스 ID. landmarks 의 키와 같은 class 끼리 매칭한다.
    range_m     : 로봇에서 그 랜드마크까지의 평면 거리 [m].
    bearing_rad : 로봇 정면(+x) 기준 방위각 [rad]. 좌(+y)가 +, 우(-y)가 -.
    """

    class_id: int
    range_m: float
    bearing_rad: float


@dataclass
class LandmarkSolverConfig:
    """solver 동작 파라미터. 모두 런치/파라미터에서 덮어쓸 수 있게 노출 권장."""

    # 매칭이 이 개수 미만이면 그 반복(또는 전체)을 신뢰하지 않는다.
    # 2D(x,y,yaw) 를 풀려면 관측이 최소 2개는 있어야 형상이 잡힌다.
    min_observations: int = 2

    # Gauss-Newton 최대 반복 횟수. 보통 몇 회면 수렴한다.
    max_iterations: int = 6

    # 데이터 연관(association) 게이트.
    #   예측과 관측의 거리/방위 차가 이 값을 넘으면 "다른 랜드마크"로 보고 버린다.
    #   (오검출/오연관이 solver 를 망가뜨리지 않게 하는 안전장치)
    range_gate_m: float = 1.5
    bearing_gate_rad: float = 0.9

    # 관측 잡음 표준편차. 작을수록 그 관측을 더 신뢰(가중치 ∝ 1/sigma^2).
    sigma_range_m: float = 0.25
    sigma_bearing_rad: float = 0.20

    # 거리 의존 range σ 팽창: 먼 관측의 "거리" 측정을 덜 신뢰한다.
    #   스테레오 depth 오차는 거리²에 비례하므로, σ_range(d) = sigma_range_m × (1 + (d/falloff)²).
    #   falloff[m] 에서 σ 가 2배(가중치 1/4), 2×falloff 에서 5배(1/25)가 된다.
    #   0(또는 음수) = 끔(거리 무관 동일 가중치). 작을수록 먼 마커를 더 강하게 깎는다.
    #   방위(bearing)는 거리와 무관하게 정확하므로 건드리지 않는다(먼 마커의 각도는 살림).
    range_sigma_falloff_m: float = 0.0

    # 한 반복에서 허용하는 최대 보정 스텝(발산 방지용 clip).
    max_step_xy_m: float = 0.30
    max_step_yaw_rad: float = 0.25

    # 카메라 장착 오프셋(로봇 base_link 기준).
    #   이 코어는 관측을 "base_link 기준"으로 받는 것을 전제로 하므로 기본 0.
    #   (카메라→base_link 변환을 노드에서 TF 로 끝내고 넣기 때문)
    camera_x_m: float = 0.0
    camera_y_m: float = 0.0


def normalize_angle(angle: float) -> float:
    """각도를 (-pi, pi] 범위로 정규화한다. 방위 차 계산의 연속성 보장용."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class LandmarkPoseSolver:
    """재사용 가능한 2D 랜드마크 pose solver(ROS 무관).

    ROS 구독/발행/TF/진단 상태가 전혀 없다. 순수 연산만 하므로
    노드(Phase 2)나 단위 테스트에서 그대로 가져다 쓸 수 있다.
    """

    def __init__(self, landmarks: Dict[int, List[Landmark]], config: LandmarkSolverConfig) -> None:
        # landmarks: class_id -> 그 class 의 모든 Landmark 리스트
        self.landmarks = landmarks
        self.config = config

    # =========================================================================
    #  메인: 시드에서 출발해 관측에 맞게 pose 를 정밀화 (Gauss-Newton)
    # =========================================================================
    def range_sigma(self, range_m: float) -> float:
        """거리에 따라 커지는 range 측정 표준편차 σ_range(d).

        스테레오 depth 오차는 거리²에 비례하므로 먼 관측의 "거리"를 덜 신뢰한다.
          σ(d) = sigma_range_m × (1 + (d / falloff)²)
        falloff <= 0 이면 거리 무관(기존 동작, σ = sigma_range_m).
        """
        f = self.config.range_sigma_falloff_m
        if f <= 0.0:
            return self.config.sigma_range_m
        return self.config.sigma_range_m * (1.0 + (range_m / f) ** 2)

    def solve_pose_from_seed(
        self,
        x0: float,
        y0: float,
        yaw0: float,
        observations: List[Observation],
    ) -> Tuple[np.ndarray, int]:
        """시드 pose 에서 출발해 관측을 가장 잘 설명하는 pose 를 반환한다.

        반환: (state[x, y, yaw], used_matches)
          - used_matches 가 min_observations 미만이면 호출 측에서 신뢰하지 말 것.
        """
        state = np.array([x0, y0, yaw0], dtype=np.float64)
        used_matches = 0

        for _ in range(self.config.max_iterations):
            # 1) 현재 추정으로 관측↔랜드마크 데이터 연관(게이트 통과분만)
            matches = self.associate_observations(state, observations)
            used_matches = len(matches)
            if used_matches < self.config.min_observations:
                # 매칭이 부족하면 더 풀어도 의미 없음 → 중단
                break

            # 2) 정규방정식 누적: H = Σ Jᵀ W J,  g = Σ Jᵀ W r
            hessian = np.zeros((3, 3), dtype=np.float64)
            gradient = np.zeros((3,), dtype=np.float64)

            for obs, lm in matches:
                residual = self.residual(state, obs, lm)            # [dr, db]
                jacobian = self.numeric_jacobian(state, obs, lm)    # 2x3
                # 가중치 W = diag(1/σ_range(d)², 1/σ_bearing²).
                #   range σ 는 거리에 따라 커진다(먼 관측의 거리 측정을 덜 신뢰).
                #   bearing σ 는 거리 무관(각도는 멀어도 정확) → 먼 마커의 방위는 살림.
                sr = self.range_sigma(obs.range_m)
                weight = np.diag([
                    1.0 / (sr ** 2),
                    1.0 / (self.config.sigma_bearing_rad ** 2),
                ])
                hessian += jacobian.T @ weight @ jacobian
                gradient += jacobian.T @ weight @ residual

            # 3) Δ = -(H + λI)⁻¹ g  (λ=1e-6: 특이행렬 방지용 미세 정칙화)
            try:
                delta = -np.linalg.solve(hessian + 1e-6 * np.eye(3), gradient)
            except np.linalg.LinAlgError:
                # 수치적으로 못 풀면 더 진행하지 않고 현재 state 유지
                break

            # 4) 스텝 제한(clip) 후 적용 — 한 번에 너무 크게 튀어 발산하는 것 방지
            dx = float(np.clip(delta[0], -self.config.max_step_xy_m, self.config.max_step_xy_m))
            dy = float(np.clip(delta[1], -self.config.max_step_xy_m, self.config.max_step_xy_m))
            dyaw = float(np.clip(delta[2], -self.config.max_step_yaw_rad, self.config.max_step_yaw_rad))
            state[0] += dx
            state[1] += dy
            state[2] = normalize_angle(state[2] + dyaw)

            # 5) 보정량이 충분히 작아지면 수렴으로 보고 종료
            if abs(dx) < 1e-4 and abs(dy) < 1e-4 and abs(dyaw) < 1e-4:
                break

        return state, used_matches

    # =========================================================================
    #  데이터 연관: 각 관측을 같은 class 의 "가장 그럴듯한" 랜드마크에 붙인다
    # =========================================================================
    def associate_observations(
        self,
        state: np.ndarray,
        observations: List[Observation],
    ) -> List[Tuple[Observation, Landmark]]:
        """현재 추정 pose 기준으로 관측↔랜드마크 매칭을 만든다.

        - 같은 class_id 후보들 중 (거리오차/σ + 방위오차/σ) 비용이 최소인 것을 고른다.
        - 단, 최소 비용이라도 게이트(range_gate/bearing_gate)를 넘으면 버린다.
          → 오검출이나 처음 보는 위치의 관측이 solver 를 끌고가지 못하게 막는 핵심.
        """
        matches: List[Tuple[Observation, Landmark]] = []
        for obs in observations:
            candidates = self.landmarks.get(obs.class_id, [])
            if not candidates:
                continue

            best = None
            best_cost = 1e9
            best_dr = 0.0
            best_db = 0.0

            for landmark in candidates:
                pred_range, pred_bearing = self.predict_measurement(state, landmark)
                dr = obs.range_m - pred_range
                db = normalize_angle(obs.bearing_rad - pred_bearing)
                # 비용: 거리/방위 오차를 각자의 σ 로 무차원화해 더한다
                cost = abs(dr) / self.config.sigma_range_m + abs(db) / self.config.sigma_bearing_rad
                if cost < best_cost:
                    best_cost = cost
                    best = landmark
                    best_dr = dr
                    best_db = db

            if best is None:
                continue
            # 게이트: 가장 가까운 후보라도 너무 멀면(오연관 위험) 버린다
            if abs(best_dr) > self.config.range_gate_m:
                continue
            if abs(best_db) > self.config.bearing_gate_rad:
                continue
            matches.append((obs, best))
        return matches

    # =========================================================================
    #  잔차(residual) / 야코비안 / 측정 예측
    # =========================================================================
    def residual(self, state: np.ndarray, obs: Observation, landmark: Landmark) -> np.ndarray:
        """관측 - 예측 = [거리오차, 방위오차(정규화)]."""
        pred_range, pred_bearing = self.predict_measurement(state, landmark)
        dr = obs.range_m - pred_range
        db = normalize_angle(obs.bearing_rad - pred_bearing)
        return np.array([dr, db], dtype=np.float64)

    def numeric_jacobian(self, state: np.ndarray, obs: Observation, landmark: Landmark) -> np.ndarray:
        """잔차의 (x, y, yaw)에 대한 야코비안(2x3)을 수치 미분으로 계산한다.

        해석적 미분 대신 중심차분을 쓴 이유: 식이 단순하고 반복 수가 적어
        계산비용이 무시할 만하며, 부호/도함수 실수 위험을 없애 안정적이다.
        방위(bearing) 성분은 wrap 때문에 차분 결과도 normalize 해 준다.
        """
        jacobian = np.zeros((2, 3), dtype=np.float64)
        eps = np.array([1e-4, 1e-4, 1e-4], dtype=np.float64)
        for idx in range(3):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[idx] += eps[idx]
            state_minus[idx] -= eps[idx]
            residual_plus = self.residual(state_plus, obs, landmark)
            residual_minus = self.residual(state_minus, obs, landmark)
            # 거리 성분: 일반 중심차분
            jacobian[0, idx] = (residual_plus[0] - residual_minus[0]) / (2.0 * eps[idx])
            # 방위 성분: 차이를 정규화한 뒤 중심차분 (±pi 경계 불연속 방지)
            jacobian[1, idx] = normalize_angle(residual_plus[1] - residual_minus[1]) / (2.0 * eps[idx])
        return jacobian

    def predict_measurement(self, state: np.ndarray, landmark: Landmark) -> Tuple[float, float]:
        """추정 pose 에서 그 랜드마크가 어떻게 보일지(예측 range, bearing) 계산.

        - 카메라 위치 = base_link 위치 + yaw 회전한 카메라 오프셋.
          (config.camera_x_m/y_m=0 이면 base_link 원점에서 보는 것과 같음)
        - range  = 카메라→랜드마크 평면 거리
        - bearing= 카메라→랜드마크 절대 방위 - 로봇 yaw (로봇 정면 기준 상대각)
        """
        x, y, yaw = float(state[0]), float(state[1]), float(state[2])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 카메라(관측 원점)의 월드 좌표
        cam_wx = x + cos_yaw * self.config.camera_x_m - sin_yaw * self.config.camera_y_m
        cam_wy = y + sin_yaw * self.config.camera_x_m + cos_yaw * self.config.camera_y_m

        dx = landmark.x - cam_wx
        dy = landmark.y - cam_wy

        pred_range = max(1e-6, math.hypot(dx, dy))  # 0 division 방지
        pred_bearing = normalize_angle(math.atan2(dy, dx) - yaw)
        return pred_range, pred_bearing
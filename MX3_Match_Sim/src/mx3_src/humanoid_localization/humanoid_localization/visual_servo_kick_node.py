# -*- coding: utf-8 -*-
# =============================================================================
#  visual_servo_kick_node.py   (Phase 3: 비전 서보 정렬 + 킥)
#
#  Phase 2(ball_follow_nav_node, Nav2)가 공 근처까지 "러프 접근"을 끝낸 뒤,
#  이 노드가 카메라 측정으로 cmd_vel 을 직접 닫는 폐루프를 돌려
#    "로봇–공–골대 일직선(공 뒤에서 골대 정면)"  까지 정밀 정렬한 다음 킥을 쏜다.
#
#  [왜 비전 서보인가 — Phase 2 와의 차이]
#  --------------------------------------------------------------------------
#   Phase 2 는 map 프레임 + Nav2(플래너/goal tolerance 0.3m, 갱신 1Hz)라
#   "로컬라이제이션 오차 + 플래너 지연" 만큼 부정확하다. 킥 정렬엔 부족.
#   Phase 3 는 공을 "base_link 기준 vision 측정값"으로 매 프레임 신선하게 받아
#   제어 주기(20Hz)로 직접 cmd_vel 을 만든다 → 공에 대해선 로컬라이제이션 무관.
#
#  [입출력 — 실제 스택 인터페이스에 맞춤]
#  --------------------------------------------------------------------------
#   입력:
#     - /perception/detections_3d (vision_msgs/Detection3DArray)
#         · yolo_detector_3d_node 발행. frame=camera_color_optical_frame.
#         · class_id 문자열: 'ball','goalpost',... / position=(X右,Y下,Z前)[m].
#         · 매 프레임 TF(optical→base_link, 동적 목이라 '최신 TF' 폴백)로 변환.
#     - /kick_done (std_msgs/Bool) : 잔차킥 정책 완료 통지.
#     - TF: optical→base_link (필수), map→base_link (골대 localization 폴백용).
#   출력:
#     - cmd_vel_behavior (geometry_msgs/Twist) : twist_mux priority 10 슬롯.
#         → /cmd_vel → RL 보행정책(actor_onnx_walk_residual_kick)이 추종.
#         ※ Phase 2 의 cmd_vel_nav(priority 20)가 살아있으면 그쪽이 이긴다.
#           통합 시 상위 오케스트레이터가 Phase 2 goal 을 취소(=nav 침묵)시킨 뒤
#           이 노드를 enable 해야 한다. 단독 테스트 땐 Nav2 없이 바로 동작.
#     - /kick_trigger (std_msgs/Bool true) : 정렬 완료 시 1회 발행 → 잔차킥 발동.
#     - (옵션) 잔차킥 노드의 ball_pos_b_x/y 파라미터를 "실측 공위치"로 set
#         (push_ball_pos_to_kicker=True). 킥 정확도↑. 트리거 직전 1회.
#     - /visual_servo/state (std_msgs/String) : FSM 상태.
#     - /visual_servo/markers (visualization_msgs/MarkerArray) : 공/골대/타깃/
#         정렬선/상태텍스트(RViz, base_link). 정렬 여부를 눈으로 즉시 검증.
#
#  [정렬 타깃 = 잔차킥 학습 스위트스폿]
#  --------------------------------------------------------------------------
#   잔차킥 정책은 트리거 순간 base_link 공위치를 freeze 하고, 학습 기본값이
#   (x=0.30, y=0.05)[m] 이다(actor_onnx_walk_residual_kick_runner 의 ball_pos_b_*).
#   그래서 "공 중앙"이 아니라 이 (0.30, 0.05) 지점에 공을 가져다 놓고 킥한다.
#   target_ball_x/y 로 노출하니 킥 노드 값과 반드시 일치시킬 것.
#
#  [제어법 — 운동학 시뮬로 부호/수렴 검증 완료]
#  --------------------------------------------------------------------------
#   매 주기 base_link 에서:
#     θ_b = 공 방위, r_b = 공 거리, φ_g = 골대 방위
#     δ   = wrap(θ_b − φ_g)   ← 정렬오차. δ=0 이면 공과 골대가 같은 방위
#                               = 로봇–공–골대 일직선(로봇이 공 뒤).
#   상태:
#     APPROACH : 공이 멀다(r_b>orbit_r+band) → 공 쪽으로 전진 + 공 중앙유지.
#     ORBIT    : 가까운데 정렬오차 큼(|δ|>tol) → 반경 유지하며 공 주위를 돈다.
#                vy = +Ko·δ  (★ 시뮬 검증: +부호라야 모든 시작자세에서 수렴.
#                             −부호는 발산. 공을 차버리지 않고 공 뒤로 돌아감)
#     ADVANCE  : 정렬됨 → 공을 (target_x,target_y) 스위트스폿으로 끌어와 골대 정면.
#     KICK_ARM : 킥조건 충족 → (옵션)ball_pos_b set 후 arm_delay 대기.
#     KICK_WAIT: /kick_trigger 발행 후 정지하고 /kick_done 대기.
#     DONE     : 킥 완료. (continuous_mode 면 잠시 뒤 SEARCH 재진입)
#     SEARCH   : 공은 보이나 골대 방위 미정 → 공을 시야 중앙에 두고 대기(공 추종).
#                ※enable 직후/DONE 직후의 초기 상태로도 잠깐 거쳐감(다음 tick 에 재평가).
#                  이 상태는 '공이 보이는' 상황이라 분실(LOST)과 구분된다.
#     LOST     : 공 분실(ball_timeout 초과) → ★제자리 정지 후 status 로 보고만 한다.
#                회수(탐색/후퇴)는 여기서 하지 않는다. 분실 처리 정책을 상위 strategy
#                한 곳으로 모으기 위함(ball_follow 의 LOST 보고와 대칭 구조).
#                즉 visual_servo 는 더 이상 자체 제자리 회전탐색을 하지 않는다.
#                strategy 가 /visual_servo/state == 'LOST' 를 보고 다음 행동을 결정한다.
#
#  [한계 — 솔직하게]
#   - 골대 방위 φ_g: 골포스트 2개가 보이면 vision(드리프트 무관), 안 보이면
#     map 골대좌표−로봇pose(=localization, Phase 2 와 동급 오차)로 폴백한다.
#     공 추적 중엔 목을 숙여 골대가 화면 밖일 때가 잦으므로 'auto' 가 현실적.
#   - 정밀도는 결국 vision 공거리 정확도(D405 size 기반 추정)와 보행 추종성에
#     달려 있다. 이 노드는 그 위에서 폐루프로 줄여줄 뿐.
# =============================================================================

import math
import csv
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from rclpy.duration import Duration
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)
from rcl_interfaces.msg import SetParametersResult, Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters

from geometry_msgs.msg import Twist, PointStamped, Point
from std_msgs.msg import Bool, String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection3DArray

from tf2_ros import Buffer, TransformListener, TransformException
# tf2_geometry_msgs 를 import 하면 PointStamped 의 do_transform 변환이 등록된다.
from tf2_geometry_msgs import do_transform_point


def wrap_angle(a: float) -> float:
    """각도를 (−π, π] 로 정규화."""
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class VisualServoKickNode(Node):
    """비전 측정으로 로봇–공–골대를 정렬하고 킥을 쏘는 Phase 3 폐루프 노드."""

    def __init__(self) -> None:
        super().__init__('visual_servo_kick_node')

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 토픽/프레임
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('detections_topic', '/perception/detections_3d')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel_behavior')
        self.declare_parameter('kick_trigger_topic', '/kick_trigger')
        self.declare_parameter('kick_done_topic', '/kick_done')
        self.declare_parameter('state_topic', '/visual_servo/state')
        self.declare_parameter('marker_topic', '/visual_servo/markers')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')

        # 검출 class_id 문자열(yolo_detector_3d 의 class_names 와 일치해야 함).
        self.declare_parameter('ball_class', 'ball')
        self.declare_parameter('goal_class', 'goalpost')

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 골대 방향 소스
        # ──────────────────────────────────────────────────────────
        # 'auto'         : 골포스트 2개 이상 신선하게 보이면 vision, 아니면 localization
        # 'vision'       : 항상 vision(골포스트 2개 필요. 없으면 골대미정→정렬 보류)
        # 'localization' : 항상 map 골대좌표−로봇pose
        self.declare_parameter('goal_source', 'auto')
        # localization 폴백용 공격 골대 중심(map). 팀 상대 좌표라 상대 골=+x 고정
        # (연습장 프로파일 기본 1.95, S-Field 4.5). 사이드 배정에 따라 뒤집지 않음.
        self.declare_parameter('target_goal_x', 1.95)
        self.declare_parameter('target_goal_y', 0.0)
        # vision 골대를 인정하려면 골포스트가 최소 몇 개 보여야 하는지(중심=두 기둥 중점).
        self.declare_parameter('min_goalposts', 2)

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 기하/제어(운동학 시뮬로 검증한 값)
        # ──────────────────────────────────────────────────────────
        # ── 킥 스위트스폿(base_link). 잔차킥 ball_pos_b_x/y 와 반드시 일치 ──
        self.declare_parameter('target_ball_x', 0.20)
        self.declare_parameter('target_ball_y', 0.05)
        self.declare_parameter('kick_pos_tol_x', 0.05)   # 전방 허용오차
        self.declare_parameter('kick_pos_tol_y', 0.06)   # 측면 허용오차
        self.declare_parameter('kick_yaw_tol_deg', 6.0)  # 킥: 골대 정면 허용오차
        self.declare_parameter('bx_min', 0.20)           # 공을 이보다 더 끌어당기지 않음(학습범위 하한)

        # ── 명령 범위(keyboard_teleop_node 정의값) ──
        # 전진/회전은 단독 가능, 횡이동(vy)은 '보행 중'(vx 또는 wz 활성)일 때만 듣는다.
        # 그래서 ★전진+횡+회전을 동시에 미세하게 섞어★ 공을 시야에 둔 채 한 번에 정렬한다
        # (공 주위를 크게 도는 회전은 공을 시야 밖으로 보내 분실 위험 → 사용 안 함).
        #   전진 0.15~0.50(실측 최소 0.20) / 회전 0.50~1.50 / 횡 0.10~0.30
        self.declare_parameter('vx_min', 0.20)
        self.declare_parameter('vx_max', 0.50)
        self.declare_parameter('vy_min', 0.10)
        self.declare_parameter('vy_max', 0.30)
        self.declare_parameter('wz_min', 0.50)
        self.declare_parameter('wz_max', 1.50)

        # ── 복합 제어 게인(ALIGN: 전진+횡+회전 동시) ──
        self.declare_parameter('k_fwd', 0.8)     # 전진←공 전방오차(bx−target)
        self.declare_parameter('k_lat', 1.5)     # 횡이동←공 측면오차(by−target)
        self.declare_parameter('k_face', 1.3)    # 회전←골대 정면(φ_g)
        # ── 보조 GETBEHIND 게인(공을 바라보며 횡이동으로 뒤로 복귀; 큰 회전 회피) ──
        self.declare_parameter('k_center', 1.5)  # 회전←공 방위(시야 중앙 유지)
        self.declare_parameter('behind_tol_deg', 35.0)   # GETBEHIND→ALIGN(공 뒤 진입)
        self.declare_parameter('behind_exit_deg', 45.0)  # ALIGN→GETBEHIND(공 뒤 크게 벗어남)
        # ── ALIGN 커밋 반경 (이름과 달리 '상태 전이 게이트') ──
        #   ALIGN→GETBEHIND 조건에 (r_b > r_min) 이 들어간다. 공이 이
        #   반경 안이면 정렬이 크게 틀어져도 GETBEHIND 로 되돌아가지 않고
        #   ALIGN 미세정렬로 끝까지 간다.
        #   ★ 불변식 : target_ball_x + kick_pos_tol_x  <  r_min_m ★
        #     ALIGN 수렴 밴드가 이 반경을 '관통'하면 비전 거리 노이즈로
        #     전이 게이트가 깜빡여 ALIGN <-> GETBEHIND 를 왕복한다. 두 상태는
        #     회전 목표가 서로 달라(골대 vs 공) 명령이 매 프레임 뒤집힌다.
        #   ※ 궤도 반경은 orbit_r_m 으로 분리됐다(아래).
        self.declare_parameter('r_min_m', 0.45)
        # ── GETBEHIND 궤도 반경 (2026-08-18 분리) ──
        #  [왜 분리했나]
        #    r_min_m 이 두 역할을 겸하고 있었다.
        #      (1) GETBEHIND 전진 정지선 → 공 주위를 도는 '궤도 반경'
        #      (2) ALIGN→GETBEHIND 전이 게이트 → '커밋 반경'
        #    (1)은 작아야 공에 가까이 붙어 돌고, (2)는 커야 ALIGN 수렴 밴드
        #    (target_ball_x ± kick_pos_tol_x)를 통째로 감싼다. 서로 반대
        #    방향이라 한 값으로는 둘 다 만족할 수 없었다.
        #    → 궤도 반경만 여기로 뗐다. r_min_m 은 커밋 반경 전용이 된다.
        #  [값 잡는 법]
        #    target_ball_x 보다 '약간 크게' 두는 것이 안전하다. 궤도에서
        #    ALIGN 으로 넘어갈 때 전진으로 스위트스폿에 들어가게 되어,
        #    데드존이 있는 후진을 쓰지 않아도 된다.
        #    너무 작으면(0.25 미만) 궤도를 도는 중에 공을 밀 위험이 커진다.
        self.declare_parameter('orbit_r_m', 0.32)          # 이보다 가까우면 전진 정지(공 보호)
        # ── GETBEHIND 접근 방식 (2026-08-18 추가) ──
        #   'orbit'  : 기존. 고정 반경 원운동으로 공 뒤까지 돌아간다.
        #              공을 밀 위험이 낮지만 호가 길어 '옆으로만 돈다'는 느낌.
        #   'direct' : 공 뒤 standoff 지점을 base 프레임에서 직접 계산해
        #              대각선으로 질러간다(ball_follow 와 같은 발상).
        #              단 orbit_r_m 안쪽으로 파고드는 반경 성분은 잘라내
        #              공 보호는 유지한다 — 필요할 때만 돌아간다.
        #   런타임 전환 가능: ros2 param set ... getbehind_mode orbit
        self.declare_parameter('getbehind_mode', 'direct')
        # direct 모드에서 노릴 '공 뒤' 거리[m]. target_ball_x 근처가 자연스럽다.
        self.declare_parameter('approach_standoff_m', 0.32)
        # direct 모드 접근 게인. 크면 빠르지만 오버슈트가 는다.
        self.declare_parameter('k_approach', 1.2)
        # ── 킥 접근 자세 (2026-08-19 추가) ─────────────────────────────────
        #  'straight'      : 기존. 로봇-공-골대를 일직선으로 세우고(delta→0)
        #                    골대를 정면으로 본 채 찬다. 정렬 목표가 1개뿐이라
        #                    최악의 경우 180도를 돌아야 한다.
        #  'perpendicular' : 공을 정면으로 보고(theta_b→0) 골대를 옆으로 두어
        #                    (|phi_g|≈perp_target_deg) 인사이드 킥으로 꺾어 찬다.
        #                    [이점 1] 목표가 +-2개라 가까운 쪽을 고를 수 있어
        #                             최악 회전량이 180도→90도로 절반이 된다.
        #                    [이점 2] 몸통을 비틀지 않아 낙상 안정도가 높다.
        #                    ※ 양발 인사이드 킥을 학습했기에 +-가 대칭이다.
        #                      한 발만 쓰는 정책이면 한쪽만 유효해 이점 1이 사라진다.
        #
        #  ★ 앞서 실패한 face_ball 과의 차이 ★
        #    face_ball 은 공을 정면으로 보게 하면서 |phi_g|<=30 을 요구해,
        #    두 조건이 서로를 밀어내 게이트 통과율이 7.4% 였다. perpendicular
        #    는 phi_g 목표 자체를 +-80 으로 옮겨 충돌이 없다.
        self.declare_parameter('align_kick_mode', 'straight')
        # perpendicular 목표 각도[deg]. 정책 게이트가 +-90 이므로 경계에 붙지
        # 않도록 80 을 기본으로 한다(측정 노이즈로 90 을 넘으면 정책이 거부).
        self.declare_parameter('perp_target_deg', 80.0)
        self.declare_parameter('perp_gate_min_deg', 70.0)
        self.declare_parameter('perp_gate_max_deg', 88.0)
        # 공 방위 허용오차[deg] — 측면 정렬 게이트. 0 대칭이다.
        #   구 kick_pos_tol_y(직교좌표 띠)를 대체한다. 잔차킥 시절에는 공을
        #   특정 발 앞에 정확히 놓아야 해 target_ball_y=0.04 로 치우친 띠를
        #   썼지만, 지금 정책은 방위 +-90 을 받으므로 그럴 이유가 없다.
        self.declare_parameter('align_bearing_tol_deg', 15.0)
        # ── 진단용 CSV 로그 (2026-08-18 추가) ──────────────────────────────
        #  제어 주기마다 '왜 그 명령이 나왔는지'를 통째로 남긴다.
        #  정렬이 수렴하지 않는 문제는 추측으로 파라미터를 만지면 뭐가
        #  효과였는지 알 수 없으므로, 수치로 원인을 특정하기 위한 것이다.
        #  특히 floor 전/후 값을 같이 남긴다 — 데드밴드가 명령을 먹었는지,
        #  애초에 게인이 작았는지를 구분하는 핵심 단서다.
        self.declare_parameter('enable_csv_log', False)
        self.declare_parameter('csv_log_dir', '~/servo_logs')

        # ── 후진 속도 범위(공이 target_bx 보다 가까워졌을 때 물러나 복귀) ──
        # 보행정책 실측: vx 후진은 −0.20 ~ −0.30 만 추종, 0 ~ −0.20 은 데드존(안 움직임).
        # 그래서 후진 명령은 절댓값을 [vx_back_min, vx_back_max] 로 강제해야 실제로 듣는다.
        # (overshoot 로 공이 0.20m 안쪽까지 붙어버린 데드락을 푸는 핵심 수단)
        self.declare_parameter('vx_back_min', 0.20)      # 최소 후진속도(데드존 상한 = 이 값부터 움직임)
        self.declare_parameter('vx_back_max', 0.30)      # 최대 후진속도
        # ── gait-keeper: vy(횡스텝)를 듣게 하는 최소 보행 vx ──
        # 보행정책은 vy 단독(제자리 횡이동) 불가 — vx 가 이 값 이상으로 '걷고
        # 있어야' vy 가 실제로 듣는다. vx_min(0.20, 전진 데드밴드/유효전진 하한)과는
        # 별개의 '더 낮은 보행 엔진 임계'(실측 0.10). 측면 정렬만 남았을 때 이 값을
        # vx 에 깔아 gait 를 살리는 용도(아래 ALIGN gait-keeper 참고).
        self.declare_parameter('vx_gait_min', 0.10)

        # SEARCH 제자리 회전 속도(데드밴드 이상이어야 실제 회전).
        self.declare_parameter('search_wz', 0.6)

        # ──────────────────────────────────────────────────────────
        # 파라미터 — 타이밍/동작
        # ──────────────────────────────────────────────────────────
        self.declare_parameter('control_rate_hz', 20.0)
        # 공 수신 끊김→LOST. ★ ball_follow_nav_node / strategy_fsm_node 의
        #   ball_lost_timeout_s(둘 다 1.0)와 같은 값이어야 한다 ★
        #   예전 기본값은 0.5 였다. yaml 이 1.0 으로 덮어써서 경기에서는
        #   문제가 없었지만, yaml 없이 이 노드만 띄우면 이 노드만 먼저
        #   LOST 를 선언해 단독 디버깅에서 재현되지 않는 현상이 생겼다.
        self.declare_parameter('ball_timeout_s', 1.0)
        self.declare_parameter('goal_timeout_s', 0.8)    # vision 골대 신선도
        self.declare_parameter('kick_arm_delay_s', 0.15)  # 파라미터 set 반영 대기
        self.declare_parameter('kick_wait_timeout_s', 6.0)
        self.declare_parameter('auto_start', False)      # True면 켜자마자 동작
        self.declare_parameter('continuous_mode', False)  # True면 킥 후 재탐색 반복
        # DONE -> SEARCH 재진입 대기 [s] (continuous_mode 일 때만 의미).
        #   ★ 예전에는 1.0 초가 코드에 하드코딩되어 있었다 ★
        #     그 1초 동안 cmd_vel=0 을 내보내는데, _enter_done 에서 gait_mode 를
        #     'auto' 로 되돌리기까지 해서 보행 위상이 아예 멈췄다. 그래서 킥
        #     직후 '한 번 멈췄다가' 다음 사이클로 넘어가는 끊김이 생겼다.
        #   0.0 이면 대기 없이 즉시 재탐색한다.
        self.declare_parameter('done_hold_s', 0.0)
        # ── 킥 쿨다운 (2026-08-20 추가) ────────────────────────────────────
        #  [왜 필요한가]
        #    킥 직후 공은 굴러가 시야에서 사라지지만, ball_timeout_s(1.0s)
        #    동안은 '마지막으로 본 위치'가 유효한 것으로 남는다. 그 사이에
        #    ALIGN 이 그 옛 위치로 재정렬해 조건을 만족시키면, 공이 없는데도
        #    한 번 더 차버린다. 예전에는 보행 러너의 1.0초 hold 가 이 구간을
        #    가려주고 있었는데, hold 를 없애면서 증상이 드러났다.
        #
        #  [FSM 과의 관계 — 충돌 없음]
        #    strategy_fsm 에도 같은 목적의 kick_cooldown_s(1.0)/post_kick_settle_s
        #    (3.0)가 있다. 다만 그건 FSM 이 있을 때만 동작하고, 서보 단독
        #    테스트에서는 방어가 통째로 빠진다. 이 쿨다운은 '트리거를 막기만'
        #    할 뿐 강제로 차게 만들지 않으므로, 둘이 함께 있어도 더 긴 쪽이
        #    이길 뿐 서로 간섭하지 않는다.
        #
        #  ※ 쿨다운 중에도 ALIGN 은 계속 돈다(정렬 유지). 발사만 막는다.
        #    0 이하면 비활성.
        self.declare_parameter('kick_cooldown_s', 1.5)
        self.declare_parameter('game_stop_lock_topic', 'game_stop_lock')

        # 잔차킥 노드에 실측 공위치를 ball_pos_b 로 push 할지(킥 정확도↑).
        self.declare_parameter('push_ball_pos_to_kicker', True)
        # 킥 방향(골대 방위)도 함께 push 할지.
        #   True  : 정책이 조준 → 이 노드는 정렬만. kick_yaw_tol_deg 를
        #           크게 열 수 있다(정렬 시간 단축, 교착 감소).
        #   False : 예전 동작(정면 킥). 로봇이 골대를 정면으로 봐야 하므로
        #           kick_yaw_tol_deg 를 6deg 수준으로 되돌릴 것.
        self.declare_parameter('push_kick_direction', True)
        # ── gait 모드 제어 (2026-08-18 추가) ──
        #  [왜 필요한가]
        #    보행 정책(auto 모드)은 명령 노름이 standing_threshold 아래면
        #    위상을 멈춘다. 그런데 ALIGN 막바지에는 전방 스위트스폿에 들어와
        #    vx=0, 골대도 이미 정면이라 wz≈0 이 되어, 측면 정렬(vy)만 남는다.
        #    이때 보행이 멈춰 vy 가 죽고 그 자리서 교착됐다.
        #    (예전 해법 = gait_keeper: ex 방향으로 최소 전진을 억지로 깔기)
        #  [지금 해법]
        #    서보 구간에만 gait_mode 를 'march'(제자리 걷기)로 바꾼다. 위상이
        #    항상 돌므로 전진 없이도 횡이동이 듣는다. 억지 전진이 사라져
        #    공을 밀어버리는 위험도 없어진다.
        #  [반드시 되돌릴 것]
        #    끝나고 'auto' 로 복원하지 않으면 로봇이 계속 제자리걸음을 한다.
        #    DONE / disable / 종료 시 복원한다.
        #  ※ 조이패드(joypad_teleop_node)도 /gait_mode 를 발행한다. 둘을 동시에
        #    쓰면 서로 덮어쓰므로, 조이패드 운용 중에는 false 로 둘 것.
        self.declare_parameter('gait_mode_control', True)
        self.declare_parameter('gait_mode_active', 'march')
        self.declare_parameter('gait_mode_idle', 'auto')
        self.declare_parameter('gait_mode_topic', '/gait_mode')
        # gait-keeper(억지 전진) 사용 여부. march 모드를 쓰면 불필요하다.
        #   True 로 되돌리면 예전 동작(vx_gait_min 만큼 전진을 깔아 보행 유지).
        self.declare_parameter('gait_keeper_enable', False)
        self.declare_parameter('kick_node_name', 'actor_onnx_walk_residual_kick_node')

        # ──────────────────────────────────────────────────────────
        # 파라미터 캐싱
        # ──────────────────────────────────────────────────────────
        g = self.get_parameter
        det_topic = str(g('detections_topic').value)
        self._cmd_topic = str(g('cmd_vel_topic').value)
        kick_trig_topic = str(g('kick_trigger_topic').value)
        kick_done_topic = str(g('kick_done_topic').value)
        state_topic = str(g('state_topic').value)
        marker_topic = str(g('marker_topic').value)
        self._base_frame = str(g('base_frame').value)
        self._map_frame = str(g('map_frame').value)
        self._ball_class = str(g('ball_class').value)
        self._goal_class = str(g('goal_class').value)

        self._goal_source = str(g('goal_source').value)
        self._goal_x = float(g('target_goal_x').value)
        self._goal_y = float(g('target_goal_y').value)
        self._min_posts = int(g('min_goalposts').value)

        self._tgt_bx = float(g('target_ball_x').value)
        self._tgt_by = float(g('target_ball_y').value)
        self._kick_tol_x = float(g('kick_pos_tol_x').value)
        self._kick_tol_y = float(g('kick_pos_tol_y').value)
        self._kick_yaw_tol = math.radians(float(g('kick_yaw_tol_deg').value))
        self._bx_min = float(g('bx_min').value)

        self._vx_min = float(g('vx_min').value)
        self._vx_max = float(g('vx_max').value)
        self._vy_min = float(g('vy_min').value)
        self._vy_max = float(g('vy_max').value)
        self._wz_min = float(g('wz_min').value)
        self._wz_max = float(g('wz_max').value)

        self._k_fwd = float(g('k_fwd').value)
        self._k_lat = float(g('k_lat').value)
        self._k_face = float(g('k_face').value)
        self._k_center = float(g('k_center').value)
        self._behind_tol = math.radians(float(g('behind_tol_deg').value))
        self._behind_exit = math.radians(float(g('behind_exit_deg').value))
        self._r_min = float(g('r_min_m').value)
        self._orbit_r = float(g('orbit_r_m').value)
        self._getbehind_mode = str(g('getbehind_mode').value).strip().lower()
        self._approach_standoff = float(g('approach_standoff_m').value)
        self._k_approach = float(g('k_approach').value)
        self._kick_mode = str(g('align_kick_mode').value).strip().lower()
        self._perp_target = math.radians(float(g('perp_target_deg').value))
        self._perp_gmin = math.radians(float(g('perp_gate_min_deg').value))
        self._perp_gmax = math.radians(float(g('perp_gate_max_deg').value))
        self._bearing_tol = math.radians(
            float(g('align_bearing_tol_deg').value))
        self._csv_enable = bool(g('enable_csv_log').value)
        self._csv_dir = str(g('csv_log_dir').value)
        self._csv_f = None
        self._csv_w = None
        self._t0 = time.time()
        # 제어법 각 분기가 채우는 진단 버퍼. 매 틱 CSV 한 줄로 떨어진다.
        self._dbg = {}
        self._vx_back_min = float(g('vx_back_min').value)
        self._vx_back_max = float(g('vx_back_max').value)
        self._vx_gait_min = float(g('vx_gait_min').value)
        self._search_wz = float(g('search_wz').value)

        self._rate = float(g('control_rate_hz').value)
        self._ball_timeout = float(g('ball_timeout_s').value)
        self._goal_timeout = float(g('goal_timeout_s').value)
        self._arm_delay = float(g('kick_arm_delay_s').value)
        self._kick_wait_timeout = float(g('kick_wait_timeout_s').value)
        self._continuous = bool(g('continuous_mode').value)
        self._done_hold = max(0.0, float(g('done_hold_s').value))
        self._kick_cooldown = max(0.0, float(g('kick_cooldown_s').value))
        self._kick_done_time = None   # 마지막 /kick_done 수신 시각(쿨다운 기준)
        game_stop_lock_topic = str(g('game_stop_lock_topic').value)

        self._push_ball = bool(g('push_ball_pos_to_kicker').value)
        self._push_kick_dir = bool(g('push_kick_direction').value)
        self._gait_ctrl = bool(g('gait_mode_control').value)
        self._gait_active = str(g('gait_mode_active').value)
        self._gait_idle = str(g('gait_mode_idle').value)
        self._gait_topic = str(g('gait_mode_topic').value)
        self._gait_keeper = bool(g('gait_keeper_enable').value)
        self._gait_mode_sent = None      # 마지막으로 발행한 모드(중복 발행 방지)
        kick_node = str(g('kick_node_name').value)

        # 런타임 튜닝(게인/허용오차/소스/타깃 등 즉시 반영).
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ──────────────────────────────────────────────────────────
        # 내부 상태
        # ──────────────────────────────────────────────────────────
        self._enabled = bool(g('auto_start').value)
        self._state = 'IDLE'
        self._game_stop_locked = False
        self._ball_b = None          # 최신 공 base_link (x,y) | None
        self._ball_stamp = None      # 공 수신 시각
        self._goal_dir = None        # 최신 골대 방위 φ_g[rad] | None
        self._goal_pt_b = None       # 골대 방향 마커용 base 점 (x,y) | None
        self._goal_is_vision = False  # 현재 골대 방위가 vision 인지
        self._goal_stamp = None      # vision 골대 수신 시각
        self._last_ball_side = 1.0   # SEARCH 회전 방향(마지막 본 공의 좌/우 부호)
        self._kick_done = False      # /kick_done 수신 플래그
        self._arm_t0 = None          # KICK_ARM 진입 시각
        self._kick_wait_t0 = None    # KICK_WAIT 진입 시각
        self._done_t0 = None         # DONE 진입 시각

        # ──────────────────────────────────────────────────────────
        # TF / Pub / Sub / Srv
        # ──────────────────────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._cmd_pub = self.create_publisher(Twist, self._cmd_topic, 10)
        self._trig_pub = self.create_publisher(Bool, kick_trig_topic, 10)
        self._state_pub = self.create_publisher(String, state_topic, 10)
        self._marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        # gait_mode 는 latched(TRANSIENT_LOCAL)로 발행한다. 보행 러너가 나중에
        # 떠도 마지막 모드를 받도록 — 조이패드 노드와 같은 규약이다.
        _gait_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._gait_pub = self.create_publisher(String, self._gait_topic, _gait_qos)

        self.create_subscription(Detection3DArray, det_topic, self._on_detections, 10)
        self.create_subscription(Bool, kick_done_topic, self._on_kick_done, 10)
        self.create_subscription(
            Bool, game_stop_lock_topic, self._on_game_stop_lock, 10)

        # enable/disable 서비스(오케스트레이터/사람이 Phase 3 on/off).
        self.create_service(SetBool, '~/enable', self._on_enable)

        # (옵션) 잔차킥 노드 파라미터 set 클라이언트.
        self._kick_param_cli = None
        if self._push_ball:
            self._kick_param_cli = self.create_client(
                SetParameters, f'/{kick_node}/set_parameters')

        # 제어 타이머.
        self._timer = self.create_timer(1.0 / max(self._rate, 1.0), self._on_timer)

        self.get_logger().info(
            f"visual_servo_kick 시작 — det='{det_topic}', cmd='{self._cmd_topic}', "
            f"goal_source='{self._goal_source}', r_min={self._r_min}m, "
            f"target_ball=({self._tgt_bx},{self._tgt_by}), "
            f"vx_back=[{self._vx_back_min},{self._vx_back_max}], "
            f"auto_start={self._enabled}")
        # ── 불변식 검사 ──
        #   깨지면 ALIGN 수렴 밴드가 커밋 반경을 관통해 상태 왕복이 생긴다.
        #   조용히 넘기면 '정렬이 안 끝난다'는 증상으로만 드러나 원인 찾기가
        #   어려우므로 기동 시점에 경고한다.
        _band_hi = self._tgt_bx + self._kick_tol_x
        if _band_hi >= self._r_min:
            self.get_logger().error(
                f'★불변식 위반★ target_ball_x({self._tgt_bx:.2f}) + '
                f'kick_pos_tol_x({self._kick_tol_x:.2f}) = {_band_hi:.2f} '
                f'>= r_min_m({self._r_min:.2f}). ALIGN 밴드가 커밋 반경을 '
                f'관통해 ALIGN<->GETBEHIND 왕복이 생깁니다. '
                f'r_min_m 을 {_band_hi + 0.05:.2f} 이상으로 올리세요.')
        if self._orbit_r < 0.25:
            self.get_logger().warn(
                f'orbit_r_m({self._orbit_r:.2f})이 작습니다. 궤도를 도는 중에 '
                '공을 밀 수 있습니다(0.25 이상 권장).')
        self._init_csv()

    # =========================================================================
    #  런타임 파라미터 변경
    # =========================================================================
    def _on_set_parameters(self, params):
        m = {
            'goal_source': ('_goal_source', str),
            'target_goal_x': ('_goal_x', float),
            'target_goal_y': ('_goal_y', float),
            'min_goalposts': ('_min_posts', int),
            'target_ball_x': ('_tgt_bx', float),
            'target_ball_y': ('_tgt_by', float),
            'kick_pos_tol_x': ('_kick_tol_x', float),
            'kick_pos_tol_y': ('_kick_tol_y', float),
            'bx_min': ('_bx_min', float),
            'orbit_r_m': ('_orbit_r', float),
            'getbehind_mode': ('_getbehind_mode', str),
            'approach_standoff_m': ('_approach_standoff', float),
            'k_approach': ('_k_approach', float),
            'align_kick_mode': ('_kick_mode', str),
            'done_hold_s': ('_done_hold', float),
            'kick_cooldown_s': ('_kick_cooldown', float),
            'vx_min': ('_vx_min', float),
            'vx_max': ('_vx_max', float),
            'vx_back_min': ('_vx_back_min', float),
            'vx_back_max': ('_vx_back_max', float),
            'vx_gait_min': ('_vx_gait_min', float),
            'vy_min': ('_vy_min', float),
            'vy_max': ('_vy_max', float),
            'wz_min': ('_wz_min', float),
            'wz_max': ('_wz_max', float),
            'k_fwd': ('_k_fwd', float),
            'k_lat': ('_k_lat', float),
            'k_face': ('_k_face', float),
            'k_center': ('_k_center', float),
            'r_min_m': ('_r_min', float),
            'search_wz': ('_search_wz', float),
            'ball_timeout_s': ('_ball_timeout', float),
            'goal_timeout_s': ('_goal_timeout', float),
        }
        for p in params:
            if p.name in m:
                attr, cast = m[p.name]
                setattr(self, attr, cast(p.value))
            elif p.name == 'behind_tol_deg':
                self._behind_tol = math.radians(float(p.value))
            elif p.name == 'behind_exit_deg':
                self._behind_exit = math.radians(float(p.value))
            elif p.name == 'kick_yaw_tol_deg':
                self._kick_yaw_tol = math.radians(float(p.value))
            elif p.name == 'continuous_mode':
                self._continuous = bool(p.value)
                # DONE 에 머문 채 continuous 를 끄면 march 가 그대로 남아
                # 로봇이 계속 제자리걸음을 한다. 그 자리에서 원복한다.
                if not self._continuous and self._state == 'DONE':
                    self._set_gait_mode(self._gait_idle)
        return SetParametersResult(successful=True)

    # =========================================================================
    #  콜백
    # =========================================================================
    def _on_enable(self, req: SetBool.Request, res: SetBool.Response):
        if bool(req.data) and self._game_stop_locked:
            res.success = False
            res.message = 'game_stop_lock active'
            return res
        self._enabled = bool(req.data)
        if not self._enabled:
            self._publish_cmd(0.0, 0.0, 0.0)   # 끄면 즉시 정지
            # ★ 반드시 auto 로 복원 ★ march 로 남기면 계속 제자리걸음을 한다.
            self._set_gait_mode(self._gait_idle)
            self._set_state('IDLE')
        else:
            # 서보 구간은 march — 전진 없이도 횡이동이 듣게 한다.
            self._set_gait_mode(self._gait_active)
            self._set_state('SEARCH')
        res.success = True
        res.message = f'visual_servo {"ENABLED" if self._enabled else "DISABLED"}'
        self.get_logger().info(res.message)
        return res

    def _on_game_stop_lock(self, msg: Bool) -> None:
        locked = bool(msg.data)
        if locked and not self._game_stop_locked:
            self._abort_for_game_stop_lock()
        self._game_stop_locked = locked

    def _abort_for_game_stop_lock(self) -> None:
        """Drop stale visual servo / kick state when game_stop_lock turns on."""
        self._enabled = False
        self._publish_cmd(0.0, 0.0, 0.0)
        self._ball_b = None
        self._ball_stamp = None
        self._goal_dir = None
        self._goal_pt_b = None
        self._goal_is_vision = False
        self._goal_stamp = None
        self._last_ball_side = 1.0
        self._kick_done = False
        self._arm_t0 = None
        self._kick_wait_t0 = None
        self._done_t0 = None
        self._set_state('IDLE')
        self.get_logger().warn(
            'game_stop_lock ON -> visual_servo align/kick state cleared')

    def _on_kick_done(self, msg: Bool) -> None:
        if msg.data:
            self._kick_done = True
            # 쿨다운 기준 시각. 이후 kick_cooldown_s 동안 KICK_ARM 진입을 막아
            # '이미 차서 굴러간 공'을 옛 위치로 한 번 더 차는 것을 방지한다.
            self._kick_done_time = self.get_clock().now()

    def _on_detections(self, msg: Detection3DArray) -> None:
        """검출(optical 프레임) → base_link 로 변환해 공/골대 갱신."""
        src_frame = msg.header.frame_id or 'camera_color_optical_frame'
        # 동적 목 TF: exact-time lookup 은 거의 매 프레임 실패 → 최신 TF 폴백.
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, src_frame, RclpyTime())
        except TransformException as e:
            self.get_logger().warn(
                f'{src_frame}→{self._base_frame} TF 없음({e}) — TF 트리/목 상태 확인',
                throttle_duration_sec=2.0)
            return

        best_ball = None     # (score, x, y)
        posts = []           # [(score, x, y), ...]
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            cls = hyp.hypothesis.class_id
            score = float(hyp.hypothesis.score)
            # 검출 점 → base_link 변환.
            ps = PointStamped()
            ps.header.frame_id = src_frame
            ps.point = det.bbox.center.position
            try:
                pb = do_transform_point(ps, tf).point
            except Exception:  # noqa: BLE001 (변환 실패 방어)
                continue
            x, y = float(pb.x), float(pb.y)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if cls == self._ball_class:
                if best_ball is None or score > best_ball[0]:
                    best_ball = (score, x, y)
            elif cls == self._goal_class:
                posts.append((score, x, y))

        now = self.get_clock().now()
        # 공 갱신.
        if best_ball is not None:
            self._ball_b = (best_ball[1], best_ball[2])
            self._ball_stamp = now
            self._last_ball_side = 1.0 if best_ball[2] >= 0.0 else -1.0
        # vision 골대 갱신(상위 min_posts 개의 중점).
        if len(posts) >= self._min_posts:
            posts.sort(key=lambda t: t[0], reverse=True)
            sel = posts[:self._min_posts]
            cx = sum(p[1] for p in sel) / len(sel)
            cy = sum(p[2] for p in sel) / len(sel)
            self._goal_pt_b = (cx, cy)
            self._goal_stamp = now

    # =========================================================================
    #  골대 방위 결정 — vision/localization/auto
    # =========================================================================
    def _resolve_goal_dir(self, now):
        """φ_g[rad] 와 마커용 base 점을 결정. 못 구하면 (None, None, False)."""
        vision_fresh = (
            self._goal_pt_b is not None and self._goal_stamp is not None
            and (now - self._goal_stamp).nanoseconds * 1e-9 <= self._goal_timeout)

        use_vision = (self._goal_source == 'vision'
                      or (self._goal_source == 'auto' and vision_fresh))

        if use_vision and vision_fresh:
            gx, gy = self._goal_pt_b
            return math.atan2(gy, gx), (gx, gy), True

        if self._goal_source == 'vision':
            # vision 강제인데 골대 안 보임 → 미정.
            return None, None, False

        # localization 폴백: map 골대좌표 → base 변환.
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._map_frame, RclpyTime())
        except TransformException:
            return None, None, False
        ps = PointStamped()
        ps.header.frame_id = self._map_frame
        ps.point = Point(x=self._goal_x, y=self._goal_y, z=0.0)
        try:
            pb = do_transform_point(ps, tf).point
        except Exception:  # noqa: BLE001
            return None, None, False
        return math.atan2(pb.y, pb.x), (pb.x, pb.y), False

    # =========================================================================
    #  제어 타이머 — FSM + 제어법
    # =========================================================================
    def _on_timer(self) -> None:
        if self._game_stop_locked:
            self._publish_cmd(0.0, 0.0, 0.0)
            return
        if not self._enabled:
            return  # IDLE: 아무것도 발행 안 함(다른 cmd 소스에 양보)

        now = self.get_clock().now()

        # KICK_ARM / KICK_WAIT 는 정렬 측정과 독립적으로 먼저 처리.
        if self._state == 'KICK_ARM':
            self._publish_cmd(0.0, 0.0, 0.0)
            if self._arm_t0 is not None and \
               (now - self._arm_t0).nanoseconds * 1e-9 >= self._arm_delay:
                self._fire_kick()
            self._publish_markers(now)
            return
        if self._state == 'KICK_WAIT':
            self._publish_cmd(0.0, 0.0, 0.0)  # 보행 정지(정책이 킥 실행)
            if self._kick_done:
                self.get_logger().info('킥 완료(/kick_done) 수신')
                self._enter_done(now)
            elif self._kick_wait_t0 is not None and \
                    (now - self._kick_wait_t0).nanoseconds * 1e-9 >= self._kick_wait_timeout:
                self.get_logger().warn('킥 완료 timeout — DONE 으로 강제 전이')
                self._enter_done(now)
            self._publish_markers(now)
            return
        if self._state == 'DONE':
            self._publish_cmd(0.0, 0.0, 0.0)
            if (self._continuous and self._done_t0 is not None
                    and (now - self._done_t0).nanoseconds * 1e-9
                    >= self._done_hold):
                # continuous 면 _enter_done 에서 march 를 유지했으므로 여기서
                # 다시 걸 필요가 없다. 방어적으로 한 번 더 호출해도 중복
                # 발행은 _set_gait_mode 가 막는다.
                self._set_gait_mode(self._gait_active)
                self._set_state('SEARCH')
            self._publish_markers(now)
            return

        # ── 공 신선도 확인 ──
        ball_fresh = (self._ball_b is not None and self._ball_stamp is not None
                      and (now - self._ball_stamp).nanoseconds * 1e-9 <= self._ball_timeout)
        if not ball_fresh:
            # ── 공 분실(ball_timeout 초과) ─────────────────────────────────
            # ★변경: 예전엔 마지막 본 방향(_last_ball_side)으로 제자리 회전하며
            #   자체 재탐색했다. 그러나 분실 회수(탐색/후퇴) 정책을 상위 strategy
            #   한 곳으로 일원화하기로 함(ball_follow 의 LOST 보고와 대칭).
            #   → 여기서는 (1) 즉시 정지 (2) status='LOST' 로 보고만 한다.
            #   strategy 가 /visual_servo/state == 'LOST' 를 받아 다음 행동을 결정.
            #   ※주의: 현재 strategy 의 ALIGN_KICK 하드 래치는 /kick_done 으로만
            #     풀린다. 이 LOST 신호로 래치를 빠져나가게 하는 연결(BALL_SEARCH)은
            #     strategy 쪽 후속 작업에서 추가한다. 그 전까지는 ALIGN_KICK 중
            #     분실 시 '정지 상태로 대기 → strategy 의 공 타임아웃(1.0s) 후
            #     mode 가 CHASE 를 벗어나며 home 복귀' 흐름을 탄다.
            self._set_state('LOST')
            self.get_logger().info(
                'TELEM [LOST] 공 없음(검출/ball TF 확인) — 정지, strategy 로 보고',
                throttle_duration_sec=1.0)
            self._publish_cmd(0.0, 0.0, 0.0)   # ★회전 제거 — 제자리 정지
            self._publish_markers(now)
            return

        bx, by = self._ball_b
        th_b = math.atan2(by, bx)
        r_b = math.hypot(bx, by)

        # ── 골대 방위 결정 ──
        phi_g, goal_pt, is_vis = self._resolve_goal_dir(now)
        self._goal_dir = phi_g
        self._goal_pt_b = goal_pt if goal_pt is not None else self._goal_pt_b
        self._goal_is_vision = is_vis
        if phi_g is None:
            # 골대 방위 미정(vision 강제+골대 안 보임 등) → 공만 중앙에 두고 대기.
            self._set_state('SEARCH')
            self.get_logger().info(
                f'TELEM [SEARCH] 골대 방위 미정(source={self._goal_source}) '
                f'ball=({bx:+.2f},{by:+.2f}) r_b={r_b:.2f}', throttle_duration_sec=1.0)
            self._publish_cmd(0.0, 0.0, self._floor_wz(self._k_center * th_b))
            self._publish_markers(now)
            return

        delta = wrap_angle(th_b - phi_g)   # 정렬오차(0=로봇·공·골대 일직선, 공 뒤)
        # ── 진단 버퍼 초기화 + 측정값 기록 ──
        self._dbg = {
            'state': self._state,
            'mode': self._getbehind_mode,
            'r_b': r_b, 'bx': bx, 'by': by,
            'th_b_deg': math.degrees(th_b),
            'phi_g_deg': math.degrees(phi_g),
            'delta_deg': math.degrees(delta),
            'goal_src': 'vision' if self._goal_is_vision else 'loc',
        }
        if self._ball_stamp is not None:
            self._dbg['ball_age_s'] = (
                (now - self._ball_stamp).nanoseconds * 1e-9)

        # ── 상태 ── 복합 미세정렬(ALIGN, 주) + 공 바라보며 복귀(GETBEHIND, 보조).
        # 공 주위를 크게 도는 회전을 안 써, 공이 시야에서 벗어나지 않게 한다.
        # ALIGN: 전진+횡이동+회전을 동시에 섞어 공을 스위트스폿 자세로 한 번에 정렬.
        # GETBEHIND: 공 뒤를 크게 벗어났을 때만, 공을 바라본 채 횡이동으로 복귀.
        if self._state not in ('ALIGN', 'GETBEHIND'):
            self._set_state('ALIGN')              # SEARCH/IDLE 등에서 복귀 시 진입
        # 전이 판정은 '목표 대비 오차'로 한다. straight 는 목표 0 이라 예전과
        # 동일하고, perpendicular 는 목표가 +-perp 라 그 근처가 정렬 완료다.
        _d_err_state = wrap_angle(delta - self._delta_target(delta))
        if self._state == 'ALIGN':
            if abs(_d_err_state) > self._behind_exit and r_b > self._r_min:
                self._set_state('GETBEHIND')
        elif self._state == 'GETBEHIND':
            if abs(_d_err_state) <= self._behind_tol:
                self._set_state('ALIGN')

        if self._state == 'GETBEHIND':
            # 공은 항상 시야 중앙에 둔다(두 모드 공통).
            self._dbg['wz_raw'] = self._k_center * th_b
            wz = self._floor_wz(self._k_center * th_b)

            if self._getbehind_mode == 'direct':
                # ── [direct] 공 뒤 standoff 지점으로 '직선' 접근 ──────────
                #  ball_follow 와 같은 발상이지만 map 좌표가 아니라 base 프레임
                #  상대 측정값만 쓴다 → 오도메트리 드리프트에 여전히 면역.
                #
                #  골대까지 거리는 모르지만 공(0.3~0.6m)보다 훨씬 멀므로
                #  '공→골대 방향 ≈ 골대 방위 단위벡터' 근사가 성립한다.
                #      P = 공 − standoff × (cos φ_g, sin φ_g)
                #  P 를 향해 홀로노믹(vx·vy 동시)으로 간다. 원호가 아니라
                #  대각선으로 질러가므로 '옆으로만 돈다'는 문제가 사라진다.
                gux, guy = math.cos(phi_g), math.sin(phi_g)
                px = bx - self._approach_standoff * gux
                py = by - self._approach_standoff * guy
                vx_des = self._k_approach * px
                vy_des = self._k_approach * py

                # ── 공 보호: 궤도 반경 안쪽으로는 '파고들지' 않는다 ─────────
                #  로봇이 골대와 공 사이(잘못된 쪽)에 있으면 P 로 가는 직선이
                #  공을 관통한다. 그래서 속도를 공 기준 반경/접선 성분으로
                #  분해해, orbit_r 안쪽에서는 '공에 가까워지는' 반경 성분만
                #  잘라낸다. 접선 성분은 남으므로 공 주위를 돌아 빠져나간다.
                #  (= 필요할 때만 원운동, 그 외엔 직선)
                if r_b > 1e-3:
                    ux, uy = bx / r_b, by / r_b          # 로봇→공 단위벡터
                    v_rad = vx_des * ux + vy_des * uy    # +면 공에 접근
                    if r_b <= self._orbit_r and v_rad > 0.0:
                        vx_des -= v_rad * ux
                        vy_des -= v_rad * uy
                self._dbg['vx_raw'] = vx_des
                self._dbg['vy_raw'] = vy_des
                vx = self._floor_vx(vx_des)
                vy_cmd = self._floor_vy(vy_des)
            else:
                # ── [orbit] 기존 동작: 고정 반경 원운동 ──────────────────
                #  궤도 반경 안쪽이면 전진 정지 → 그 거리를 유지한 채 횡이동만
                #  하게 되어, 공 중심 orbit_r_m 반경으로 돌게 된다.
                self._dbg['vx_raw'] = (self._vx_min if r_b > self._orbit_r
                                       else 0.0)
                self._dbg['vy_raw'] = self._k_lat * delta
                vy_cmd = self._floor_vy(self._k_lat * delta)  # δ→횡이동(공 뒤로)
                vx = self._floor_vx(self._vx_min) if r_b > self._orbit_r else 0.0
            # walking 판정은 gait 엔진 임계(vx_gait_min)로 통일(ALIGN 과 동일 기준).
            # march 모드에서는 위상이 항상 돌므로 vy 를 게이팅할 이유가 없다.
            # auto 모드(gait_keeper 사용)일 때만 예전 판정을 유지한다.
            if self._gait_ctrl and self._gait_active == 'march':
                walking = True
            else:
                walking = ((abs(vx) >= self._vx_gait_min)
                           or (abs(wz) >= self._wz_min))
            self._dbg['walking'] = int(walking)
            vy = vy_cmd if walking else 0.0                   # vy 는 보행 중에만 적용
        else:  # ALIGN — 킥 스위트스폿으로 미세정렬
            ex = bx - self._tgt_bx            # 전방 오차(+:공이 멀다 / -:너무 가깝다)
            ey = by - self._tgt_by            # 측면 오차(직교좌표)
            phi_tgt = self._phi_target(phi_g)          # 이번 킥의 골대방위 목표
            phi_err = wrap_angle(phi_g - phi_tgt)      # 그 목표 대비 오차
            self._dbg['ex'] = ex
            self._dbg['ey'] = ey

            if self._kick_mode == 'perpendicular':
                # ══ [perpendicular] 공을 정면으로, 골대는 옆으로 ═════════════
                #  목표 자세: theta_b→0 (공 정면), |phi_g|→perp_target(80deg)
                #  인사이드 킥으로 옆으로 꺾어 차므로 몸통을 비틀 필요가 없다.
                #
                #  [축 배분] 로봇이 잘하는 축에 큰 일을 맡긴다.
                #    wz : 공을 정면으로            (회전 — 강함)
                #    vx : 공까지 거리              (전진/후진 — 강함)
                #    vy : 공 주위 각도 미세 조정   (횡이동 — 약함, 잔여분만)
                #
                #  [vy 부호] 실측/수치검증 결과 로봇이 자기 기준 +y 로 이동하면
                #    phi_g 가 '증가'한다(약 +180deg/m, theta_b 를 0 으로 유지할 때).
                #    따라서 phi_g 를 줄이려면 -y 로 가야 하므로 부호가 반대다.
                #    호 길이 환산을 위해 r_b 를 곱한다(각오차 x 반경 = 이동거리).
                self._dbg['wz_raw'] = self._k_center * th_b
                self._dbg['vx_raw'] = (self._k_fwd * ex
                                       if abs(ex) > self._kick_tol_x else 0.0)
                self._dbg['vy_raw'] = -self._k_lat * phi_err * r_b
                wz = self._floor_wz(self._k_center * th_b)
                vx = (self._floor_vx(self._k_fwd * ex)
                      if abs(ex) > self._kick_tol_x else 0.0)
                vy_cmd = self._floor_vy(-self._k_lat * phi_err * r_b)
                gate_x = abs(ex) < self._kick_tol_x
                gate_y = abs(th_b) <= self._bearing_tol
                gate_yaw = (self._perp_gmin <= abs(phi_g) <= self._perp_gmax)
            else:
                # ══ [straight] 로봇-공-골대 일직선, 골대를 정면으로 ═══════════
                #   ex > +tol : 공이 멀다  -> 전진
                #   ex < -tol : 공이 가깝다 -> 후진(overshoot 데드락 해소)
                #
                # ★ 회전 목표는 '골대'다 — 공이 아니다 ★ (2026-08-19 실험)
                #   공을 정면으로 보게 회전시키는 face_ball 을 시도했다가
                #   되돌렸다. 몸통 방향은 하나뿐인데 게이트가 공 방위와 골대
                #   방위를 '동시에' 요구하므로, 공을 향해 돌면 골대 방위가
                #   밀려나 gate_yaw 를 영영 만족할 수 없다(실측 통과율 7.4%).
                #   perpendicular 모드는 phi_g 목표를 +-80 으로 옮겨 이 충돌을
                #   구조적으로 피한 것이다.
                self._dbg['vx_raw'] = (self._k_fwd * ex
                                       if abs(ex) > self._kick_tol_x else 0.0)
                self._dbg['wz_raw'] = self._k_face * phi_g
                self._dbg['vy_raw'] = self._k_lat * ey
                if abs(ex) > self._kick_tol_x:
                    vx = self._floor_vx(self._k_fwd * ex)
                else:
                    vx = 0.0
                wz = self._floor_wz(self._k_face * phi_g)   # 골대 정면 회전
                vy_cmd = self._floor_vy(self._k_lat * ey)   # 횡이동 측면 정렬
                gate_x = abs(ex) < self._kick_tol_x
                gate_y = abs(th_b) <= self._bearing_tol
                gate_yaw = abs(phi_g) <= self._kick_yaw_tol

            self._dbg['gate_x'] = int(gate_x)
            self._dbg['gate_y'] = int(gate_y)
            self._dbg['gate_yaw'] = int(gate_yaw)

            # ── gait-keeper: '순수 횡이동 불가' 교착 해소(보험) ─────────────
            #   보행정책은 vy 를 단독으로 못 듣고, vx 가 vx_gait_min 이상으로
            #   '걷고 있어야' vy 가 듣는다. gait_mode='march' 가 걸리면 위상이
            #   항상 돌아 불필요하므로 기본은 꺼져 있다(gait_keeper_enable).
            need_lateral = abs(vy_cmd) > 1e-6
            if (self._gait_keeper and vx == 0.0
                    and abs(wz) < self._wz_min and need_lateral):
                vx = self._vx_gait_min if ex >= 0.0 else -self._vx_back_min

            if self._gait_ctrl and self._gait_active == 'march':
                walking = True
            else:
                walking = ((abs(vx) >= self._vx_gait_min)
                           or (abs(wz) >= self._wz_min))
            self._dbg['walking'] = int(walking)
            vy = vy_cmd if walking else 0.0

            if gate_x and gate_y and gate_yaw:
                if self._in_kick_cooldown(now):
                    # 정렬은 계속하되 발사만 보류한다.
                    self.get_logger().info(
                        '킥 조건 충족했으나 쿨다운 중 — 보류 '
                        f'({self._kick_cooldown:.1f}s)',
                        throttle_duration_sec=1.0)
                else:
                    self._enter_kick_arm(now, bx, by, phi_g)
                    self._publish_markers(now)
                    return

        # ── 텔레메트리(0.5s throttle) — 정렬/주행 진단용 ──
        self.get_logger().info(
            f'TELEM [{self._state}] r_b={r_b:.2f}m ball=({bx:+.2f},{by:+.2f}) '
            f'phi_g={math.degrees(phi_g):+5.0f} delta={math.degrees(delta):+5.0f} '
            f'src={"vision" if self._goal_is_vision else "loc"} '
            f'cmd=(vx{vx:+.2f},vy{vy:+.2f},wz{wz:+.2f}) '
            f'raw=(vx{self._dbg.get("vx_raw", 0.0):+.3f},'
            f'vy{self._dbg.get("vy_raw", 0.0):+.3f},'
            f'wz{self._dbg.get("wz_raw", 0.0):+.3f}) '
            f'gate=({self._dbg.get("gate_x", "-")}'
            f'{self._dbg.get("gate_y", "-")}'
            f'{self._dbg.get("gate_yaw", "-")})',
            throttle_duration_sec=0.5)

        self._dbg['state'] = self._state   # 전이가 있었을 수 있으니 갱신
        self._dbg['vx'] = vx
        self._dbg['vy'] = vy
        self._dbg['wz'] = wz
        self._log_row()
        self._publish_cmd(vx, vy, wz)
        self._publish_markers(now)

    # =========================================================================
    #  킥 시퀀스
    # =========================================================================
    def _enter_kick_arm(self, now, bx, by, phi_g=0.0) -> None:
        self._set_state('KICK_ARM')
        self._arm_t0 = now
        self._kick_done = False
        # (옵션) 실측 공위치(+킥 방향)를 킥 노드로 push → 다음 트리거에 freeze.
        if self._push_ball and self._kick_param_cli is not None:
            self._push_ball_pos(bx, by, phi_g)
        self._publish_cmd(0.0, 0.0, 0.0)

    def _push_ball_pos(self, bx: float, by: float, phi_g: float = 0.0) -> None:
        """킥 노드에 실측값을 비동기 set 한다(fire-and-forget).

        [무엇을 보내나]
          ball_pos_b_x/y             : base_link 기준 실측 공위치
          kick_target_yaw_offset_rad : 킥 방향(= 골대 방위 phi_g)

        ★ 킥 방향 push 가 이 노드의 정렬 부담을 크게 덜어준다 ★
          예전 잔차킥은 CSV 레퍼런스 재생이라 '정면'만 찰 수 있었고, 그래서
          로봇 몸 자체가 조준 장치였다(kick_yaw_tol 6deg 같은 빡빡한 정렬).
          walkkick_v1 은 target_dir_b 를 관측으로 받아 +-60deg 범위 안에서
          스스로 조준한다. 골대 방위를 넘겨주면 로봇이 골대를 정면으로 볼
          필요가 없어져 정렬 허용오차를 크게 열 수 있다.

        [부호 규약]
          phi_g 는 base_link 기준 골대 방위(atan2, CCW +).
          킥 노드의 target_dir_b = (cos(offset - dyaw), sin(offset - dyaw)) 로
          같은 규약이므로 변환 없이 그대로 넘긴다.

        push_kick_direction=False 면 공위치만 보낸다(예전 동작 = 정면 킥).
        그때는 로봇이 골대를 정면으로 봐야 하므로 kick_yaw_tol_deg 를
        6deg 수준으로 되돌려야 한다.
        """
        if not self._kick_param_cli.service_is_ready():
            self.get_logger().warn(
                'kick 파라미터 서비스 미준비 — 킥 노드 기본 ball_pos_b 사용',
                throttle_duration_sec=2.0)
            return
        items = [('ball_pos_b_x', bx), ('ball_pos_b_y', by)]
        if self._push_kick_dir:
            items.append(('kick_target_yaw_offset_rad', phi_g))
        req = SetParameters.Request()
        for name, val in items:
            p = Parameter()
            p.name = name
            p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                     double_value=float(val))
            req.parameters.append(p)
        self._kick_param_cli.call_async(req)
        if self._push_kick_dir:
            self.get_logger().info(
                f'킥 노드 set → ball_pos_b=({bx:.3f}, {by:.3f}), '
                f'target_yaw={math.degrees(phi_g):+.1f}deg')
        else:
            self.get_logger().info(
                f'킥 노드 set → ball_pos_b=({bx:.3f}, {by:.3f})')

    def _fire_kick(self) -> None:
        self._set_state('KICK_WAIT')
        self._kick_wait_t0 = self.get_clock().now()
        self._kick_done = False
        self._trig_pub.publish(Bool(data=True))
        self.get_logger().info('■ /kick_trigger 발행 — 잔차킥 발동, 완료 대기')

    # =========================================================================
    #  진단 CSV 로거
    # =========================================================================
    def _init_csv(self) -> None:
        """제어 주기마다 한 줄씩 남기는 CSV 를 연다.

        [왜 필요한가]
          "정렬이 안 끝난다"는 증상은 원인 후보가 많다 — 게인 부족, 데드밴드,
          상태 왕복, 게이트 조건 불충족, 측정 노이즈. 로그 없이 파라미터를
          만지면 무엇이 효과였는지 구분되지 않는다.

        [핵심 컬럼]
          *_raw : floor(데드밴드) 적용 '전' 명령. 이게 작으면 게인 문제,
                  이게 큰데 최종이 0 이면 데드밴드/게이팅 문제다.
          gate_*: 킥 게이트 세 조건이 각각 통과했는지(0/1). 어느 조건이
                  마지막까지 안 맞는지 바로 보인다.
        """
        if not self._csv_enable:
            return
        try:
            d = os.path.expanduser(self._csv_dir)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(
                d, f"servo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self._csv_f = open(path, 'w', newline='')
            self._csv_w = csv.writer(self._csv_f)
            self._csv_w.writerow([
                't', 'state', 'mode',
                'r_b', 'th_b_deg', 'phi_g_deg', 'delta_deg',
                'bx', 'by', 'ex', 'ey',
                'vx_raw', 'vy_raw', 'wz_raw',
                'vx', 'vy', 'wz',
                'walking', 'gate_x', 'gate_y', 'gate_yaw',
                'goal_src', 'ball_age_s',
            ])
            self.get_logger().info(f'CSV 로그: {path}')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'CSV 로그 열기 실패({exc}) — 로깅 없이 진행')
            self._csv_f = None
            self._csv_w = None

    def _log_row(self) -> None:
        """_dbg 에 모인 진단값을 한 줄로 기록한다."""
        if self._csv_w is None:
            return
        d = self._dbg

        def f(key, nd=4):
            v = d.get(key)
            return '' if v is None else round(float(v), nd)

        try:
            self._csv_w.writerow([
                round(time.time() - self._t0, 3),
                d.get('state', ''), d.get('mode', ''),
                f('r_b'), f('th_b_deg', 2), f('phi_g_deg', 2), f('delta_deg', 2),
                f('bx'), f('by'), f('ex'), f('ey'),
                f('vx_raw'), f('vy_raw'), f('wz_raw'),
                f('vx'), f('vy'), f('wz'),
                d.get('walking', ''), d.get('gate_x', ''),
                d.get('gate_y', ''), d.get('gate_yaw', ''),
                d.get('goal_src', ''), f('ball_age_s', 3),
            ])
            self._csv_f.flush()   # 중간에 죽어도 남도록 매 줄 flush
        except Exception:  # noqa: BLE001
            pass

    def _close_csv(self) -> None:
        if self._csv_f is not None:
            try:
                self._csv_f.close()
            except Exception:  # noqa: BLE001
                pass
            self._csv_f = None
            self._csv_w = None

    # =========================================================================
    #  킥 접근 자세 목표
    # =========================================================================
    def _in_kick_cooldown(self, now) -> bool:
        """직전 킥으로부터 kick_cooldown_s 가 지나지 않았으면 True.

        킥 직후 공은 굴러가 사라지지만 ball_timeout_s 동안은 마지막 위치가
        유효하게 남는다. 그 옛 위치로 재정렬해 한 번 더 차는 것을 막는다.
        ALIGN 자체는 계속 돌기 때문에, 쿨다운이 끝나는 즉시 정상 발사된다.
        """
        if self._kick_cooldown <= 0.0 or self._kick_done_time is None:
            return False
        el = (now - self._kick_done_time).nanoseconds * 1e-9
        return el < self._kick_cooldown

    def _phi_target(self, phi_g: float) -> float:
        """이번 킥에서 노릴 골대 방위(phi_g) 목표값을 고른다.

        straight      : 0 (골대를 정면으로)
        perpendicular : +-perp_target 중 '현재 phi_g 에 가까운 쪽'
            양발 인사이드 킥이라 좌우가 대칭이므로 둘 중 아무 쪽이나 유효하다.
            가까운 쪽을 고르면 공 주위를 도는 각도가 절반으로 줄어든다.
        """
        if self._kick_mode != 'perpendicular':
            return 0.0
        cands = (self._perp_target, -self._perp_target)
        return min(cands, key=lambda t: abs(wrap_angle(phi_g - t)))

    def _delta_target(self, delta: float) -> float:
        """GETBEHIND 가 노릴 delta 목표값.

        delta = theta_b - phi_g 이고, ALIGN 에서 공을 정면으로 보므로
        (theta_b→0) phi_g = -delta 가 된다. 따라서 phi_g 목표가 +-perp 면
        delta 목표는 그 부호 반대다.
        """
        if self._kick_mode != 'perpendicular':
            return 0.0
        cands = (self._perp_target, -self._perp_target)
        return min(cands, key=lambda t: abs(wrap_angle(delta - t)))

    def _set_gait_mode(self, mode: str) -> None:
        """보행 러너의 gait_mode 를 바꾼다(중복 발행 억제).

        서보 구간(SEARCH~KICK_WAIT)에는 'march' 로 두어 위상이 항상 돌게 한다.
        그래야 전진 없이도 횡이동(vy)이 듣는다. 끝나면 반드시 'auto' 로
        되돌려야 한다 — 안 그러면 로봇이 계속 제자리걸음을 한다.
        """
        if not self._gait_ctrl or mode == self._gait_mode_sent:
            return
        self._gait_pub.publish(String(data=mode))
        self._gait_mode_sent = mode
        self.get_logger().info(f'gait_mode → {mode}')

    def _enter_done(self, now) -> None:
        self._set_state('DONE')
        self._done_t0 = now
        # ── gait_mode 원복 여부 ──
        #  continuous_mode 면 곧바로 SEARCH 로 돌아가므로 march 를 유지한다.
        #  여기서 'auto' 로 되돌리면 cmd_vel=0 과 겹쳐 보행 위상이 멈추고,
        #  재진입 때 다시 march 로 거는 과정에서 '한 번 멈췄다 가는' 끊김이
        #  생긴다(2026-08-19 실기 증상).
        #  단발(continuous=False)이면 반드시 원복해야 한다 — 안 그러면
        #  로봇이 영영 제자리걸음을 한다.
        if not self._continuous:
            self._set_gait_mode(self._gait_idle)

    # =========================================================================
    #  발행 헬퍼
    # =========================================================================
    def _floor_vx(self, v: float) -> float:
        """전·후진 속도 데드밴드 floor(부호 보존, 비대칭).
        전진(v>0): 미세값은 0, 그 외 [vx_min, vx_max].
        후진(v<0): 미세값은 0, 그 외 절댓값을 [vx_back_min, vx_back_max] 로.
        ★보행정책상 후진은 −0.20~−0.30 만 듣고 0~−0.20 은 데드존이라, 후진은
          반드시 최소 vx_back_min(0.20) 이상으로 밀어 올려야 실제로 물러난다."""
        if v > 1e-3:                                          # 전진
            return clamp(max(v, self._vx_min), 0.0, self._vx_max)
        if v < -1e-3:                                         # 후진(절댓값 클램프 후 부호 복원)
            return -clamp(max(-v, self._vx_back_min), 0.0, self._vx_back_max)
        return 0.0

    def _floor_wz(self, w: float) -> float:
        """회전속도 데드밴드 floor. 1.5° 미만은 0(이미 충분히 정렬), 그 외엔 부호
        유지하며 [wz_min, wz_max]. 회전은 wz_min(0.5) 미만이면 로봇이 안 돈다."""
        if abs(w) < math.radians(1.5):
            return 0.0
        return clamp(math.copysign(max(abs(w), self._wz_min), w),
                     -self._wz_max, self._wz_max)

    def _floor_vy(self, v: float) -> float:
        """횡이동속도 데드밴드 floor. 미세값은 0, 그 외엔 부호 유지하며
        [vy_min, vy_max]. ★vy 는 보행 중(vx 또는 wz 활성)일 때만 실제로 듣는다 —
        그 게이팅은 호출부에서 처리한다."""
        if abs(v) < 0.02:
            return 0.0
        return clamp(math.copysign(max(abs(v), self._vy_min), v),
                     -self._vy_max, self._vy_max)

    def _publish_cmd(self, vx: float, vy: float, wz: float) -> None:
        t = Twist()
        t.linear.x = float(vx)
        t.linear.y = float(vy)
        t.angular.z = float(wz)
        self._cmd_pub.publish(t)

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self.get_logger().info(f'상태: {self._state} → {state}')
            self._state = state
        self._state_pub.publish(String(data=state))

    def _publish_markers(self, now) -> None:
        """공/골대방향/타깃/정렬선/상태텍스트를 base_link 프레임에 발행(RViz)."""
        arr = MarkerArray()
        stamp = now.to_msg()
        life = Duration(seconds=0.5).to_msg()

        def base_marker(mid, mtype, ns='visual_servo'):
            m = Marker()
            m.header.frame_id = self._base_frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.lifetime = life
            return m

        # (1) 공(주황 구).
        if self._ball_b is not None:
            bx, by = self._ball_b
            s = base_marker(1, Marker.SPHERE)
            s.pose.position.x, s.pose.position.y, s.pose.position.z = bx, by, 0.03
            s.scale.x = s.scale.y = s.scale.z = 0.07
            s.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)
            arr.markers.append(s)

        # (2) 골대 방향(노랑=vision / 청록=localization 화살표, 원점→골대점).
        if self._goal_pt_b is not None:
            gx, gy = self._goal_pt_b
            ln = base_marker(2, Marker.LINE_STRIP)
            ln.scale.x = 0.02
            ln.color = (ColorRGBA(r=1.0, g=0.85, b=0.1, a=0.9) if self._goal_is_vision
                        else ColorRGBA(r=0.1, g=0.8, b=0.8, a=0.9))
            ln.points = [Point(x=0.0, y=0.0, z=0.01),
                         Point(x=float(gx), y=float(gy), z=0.01)]
            arr.markers.append(ln)

        # (3) 정렬선: 골대점 → 공 → 원점(로봇). 정렬되면 곧게 보임.
        if self._ball_b is not None and self._goal_pt_b is not None:
            bx, by = self._ball_b
            gx, gy = self._goal_pt_b
            al = base_marker(3, Marker.LINE_STRIP)
            al.scale.x = 0.015
            al.color = ColorRGBA(r=1.0, g=0.2, b=0.9, a=0.8)
            al.points = [Point(x=float(gx), y=float(gy), z=0.005),
                         Point(x=float(bx), y=float(by), z=0.005),
                         Point(x=0.0, y=0.0, z=0.005)]
            arr.markers.append(al)

        # (4) 킥 스위트스폿(초록 구) — 공이 여기로 와야 킥.
        sp = base_marker(4, Marker.SPHERE)
        sp.pose.position.x, sp.pose.position.y, sp.pose.position.z = \
            self._tgt_bx, self._tgt_by, 0.02
        sp.scale.x = sp.scale.y = sp.scale.z = 2.0 * self._kick_tol_x
        sp.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.5)
        arr.markers.append(sp)

        # (5) 상태 텍스트.
        tx = base_marker(5, Marker.TEXT_VIEW_FACING)
        tx.pose.position.x, tx.pose.position.y, tx.pose.position.z = 0.0, 0.0, 0.4
        tx.scale.z = 0.08
        tx.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        tx.text = self._state
        arr.markers.append(tx)

        self._marker_pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualServoKickNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # ★ 종료 시 gait_mode 를 원복한다 ★
        #   march 로 남긴 채 죽으면 보행 러너가 계속 제자리걸음을 하고,
        #   latched 발행이라 나중에 뜨는 노드도 march 를 받는다.
        try:
            node._set_gait_mode(node._gait_idle)
        except Exception:  # noqa: BLE001
            pass
        try:
            node._close_csv()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

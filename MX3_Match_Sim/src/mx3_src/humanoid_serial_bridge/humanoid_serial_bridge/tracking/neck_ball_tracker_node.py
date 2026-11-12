#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neck_ball_tracker_node.py
=========================

목(neck)을 움직여 공을 화면 중앙에 유지하는 시선(gaze) 추적 노드.

[ 동작 한 줄 요약 ]
  /perception/detections_3d 에서 'ball' 검출의 카메라 광학프레임 3D 좌표를 받아
  광축 대비 수평/수직 각오차를 구하고, 그 오차가 0이 되도록 /neck/yaw_deg,
  /neck/pitch_deg 위치 명령을 천천히(rate-limit) 보내 공을 따라본다.
  공이 일정 시간 안 보이면 → 중립(ready) 자세로 복귀한다.
  (중립 복귀: 나중에 팀 통신으로 다른 로봇이 찾은 방향을 받으면 그쪽으로
   재포착하기 위한 깨끗한 기준 자세.)

[ 좌표/부호 규약 — 반드시 이해하고 수정할 것 ]
  · 검출 3D 좌표는 camera_color_optical_frame (REP-103 광학):
        x = 오른쪽(+),  y = 아래(+),  z = 전방(+)
  · 공의 광축 대비 각오차:
        err_h = atan2(x, z)   → +면 공이 화면 오른쪽
        err_v = atan2(y, z)   → +면 공이 화면 아래
  · 목 명령(neck_control_node 규약):
        /neck/yaw_deg   : 반시계(+) = 왼쪽,   [-90 .. +90]
        /neck/pitch_deg : 0=정면, 아래로 갈수록 음수, [-60 .. 0] (위로는 안 감)
  · 따라서 공을 따라가려면 (오른쪽=yaw 감소, 아래=pitch 감소):
        목표 yaw   = 현재yaw   - Kp_yaw   * deg(err_h)
        목표 pitch = 현재pitch - Kp_pitch * deg(err_v)
    → 두 축 모두 "현재각 - Kp*deg(atan2(·, z))" 로 깔끔히 통일된다.
    (카메라가 목 끝에 살짝 기울어 장착돼 있어도, 루프가 '이미지' 오차를 0으로
     몰기 때문에 그 마운트 오프셋은 자동 흡수된다.)

[ 전제 (이 노드가 켜기 전에 떠 있어야 함) ]
  · neck_control_node       : /neck/yaw_deg, /neck/pitch_deg 를 rad 로 바꿔
                              /joint_goals 로 보냄 (→ body_motor_controller → U2D2)
  · neck_control_node       : /neck/yaw_deg_state, /neck/pitch_deg_state (현재각) 발행
                              (2026-08 목 이관 후 구 neck_state_parser_node 역할 흡수)
  · 목 토크 ON (/neck/torque_enable True) — 이 노드는 토크를 켜지 않는다.
  · yolo_detector_3d_node   : /perception/detections_3d (Detection3DArray) 발행
                              (검출 frame 은 camera_color_optical_frame 이어야 함)

[ 구독 토픽 ]
  /perception/detections_3d  (vision_msgs/Detection3DArray)  공 검출(3D)
  /neck/yaw_deg_state        (std_msgs/Float32)              현재 yaw [deg]
  /neck/pitch_deg_state      (std_msgs/Float32)              현재 pitch [deg]

[ 발행 토픽 ]
  /neck/yaw_deg              (std_msgs/Float32)              yaw 위치 명령 [deg]
  /neck/pitch_deg            (std_msgs/Float32)              pitch 위치 명령 [deg]
  /neck/tracking_state       (std_msgs/String)              디버그: 현재 상태
  /neck/track_error_deg      (std_msgs/Float32MultiArray)   디버그: [err_h, err_v] (deg)

[ 제어 구조 ]
  · 검출 콜백은 "최신 공 각오차"만 EMA로 평활화해 저장(가벼움).
  · 고정 주기 타이머(control_rate_hz)가 실제 제어를 수행:
      - 공이 신선하면(TRACKING): 측정된 현재 목각 + 이미지 오차로 '절대 목표각'을
        매 사이클 다시 계산 → 적분 와인드업/오버슈트 없음.
      - 공이 오래 안 보이면(RETURN/NEUTRAL): 목표각을 중립으로.
    목표각으로 명령을 rate-limit(사이클당 최대 Δdeg)하며 부드럽게 접근시킨다.
  · rate-limit(=슬루 속도) 은 로컬라이저의 '목 각속도 게이트'와 직접 연동되는
    손잡이다. 평상시 미세 추적은 슬루가 작아 게이트가 안 걸리고(비전 계속 보정),
    공이 화면을 가로지르는 큰 재포착/중립복귀 때만 빠른 슬루로 게이트가 잠깐
    걸린다(이때는 비전을 끄는 게 맞음).
"""

import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Float32MultiArray, String
from vision_msgs.msg import Detection3DArray


def _clamp(v: float, lo: float, hi: float) -> float:
    """v 를 [lo, hi] 범위로 자른다."""
    return lo if v < lo else (hi if v > hi else v)


class NeckBallTrackerNode(Node):
    """공을 목으로 추적하고, 분실 시 중립으로 복귀하는 노드."""

    def __init__(self):
        super().__init__('neck_ball_tracker_node')

        # =====================================================================
        #  파라미터 선언
        # =====================================================================
        # --- 토픽 ---
        self.declare_parameter('detections_topic', '/perception/detections_3d')
        self.declare_parameter('yaw_cmd_topic', '/neck/yaw_deg')
        self.declare_parameter('pitch_cmd_topic', '/neck/pitch_deg')
        self.declare_parameter('yaw_state_topic', '/neck/yaw_deg_state')
        self.declare_parameter('pitch_state_topic', '/neck/pitch_deg_state')
        self.declare_parameter('state_topic', '/neck/tracking_state')
        self.declare_parameter('error_topic', '/neck/track_error_deg')

        # --- 검출 필터 ---
        # 검출 노드는 class_id 에 '이름 문자열'을 넣는다('ball', 'goalpost'...).
        self.declare_parameter('ball_class_name', 'ball')
        self.declare_parameter('min_score', 0.40)          # 신뢰도 하한
        self.declare_parameter('min_valid_z_m', 0.05)      # 깊이 무효(z<=이값) 제거

        # --- 제어 게인/평활 ---
        self.declare_parameter('control_rate_hz', 30.0)    # 제어 타이머 주기
        self.declare_parameter('kp_yaw', 0.5)              # yaw 비례게인(0~1; 1=한번에 정조준)
        self.declare_parameter('kp_pitch', 0.5)            # pitch 비례게인
        self.declare_parameter('deadzone_deg', 1.5)        # 이 각오차 안이면 정지(떨림 방지)
        self.declare_parameter('ema_alpha', 0.5)           # 공 각오차 EMA(1=평활없음, 작을수록 부드럽고 느림)

        # --- 슬루(속도) 제한: 사이클당 최대 변화량 ---
        #   control_rate_hz 와 곱하면 최대 각속도[deg/s] 가 된다.
        #   예) 2.5deg * 30Hz = 75 deg/s. 로컬라이저 목 게이트 임계보다 약간 위로 두면
        #   미세추적은 게이트 통과, 큰 슬루만 게이트가 잡는다.
        self.declare_parameter('max_step_yaw_deg', 2.5)
        self.declare_parameter('max_step_pitch_deg', 2.0)

        # --- 목 가동 범위(neck_control_node 와 동일하게) ---
        self.declare_parameter('yaw_min_deg', -90.0)
        self.declare_parameter('yaw_max_deg', 90.0)
        self.declare_parameter('pitch_min_deg', -60.0)  # 2026-07: -70 → -60 (yaml 미지정 시 fallback)
        self.declare_parameter('pitch_max_deg', 0.0)
        # ★공이 바닥에 있어 목을 시작각(중립) 위로 들면 공을 놓친다. True면 추적 중
        #   pitch 위쪽 한계를 '시작각(neutral_pitch_deg)'에 묶어, 그 위로 안 든다.
        #   (아래로는 pitch_min_deg 까지 자유롭게 내려 가까운 공도 추적)
        self.declare_parameter('cap_pitch_to_start', True)
        # ★pitch 가 이 각도 이하(깊이 아래를 봄)로 내려가면 yaw 를 0°(정면)으로 고정한다.
        #   공을 발밑/정면 바닥에서 찾을 때 좌우로 안 흔들리고 정면만 스캔하게 한다.
        #   끄려면 pitch_min 보다 더 낮은 값(예 -71) 으로 설정.
        self.declare_parameter('yaw_lock_pitch_deg', -50.0)

        # --- 분실/중립 복귀 ---
        self.declare_parameter('lost_timeout_sec', 0.5)    # 이 시간 공 미검출이면 분실로 판단
        # ★분실 후 바로 중립 복귀하지 않고, 이 시간 동안 마지막 목 위치를 유지(마진).
        #   잠깐의 검출 끊김에 목이 휙 중립으로 돌아가 공을 더 못 보는 걸 막는다.
        #   유지 후에도 공이 안 보이면 그제서야 중립 복귀.
        self.declare_parameter('lost_hold_sec', 2.0)
        self.declare_parameter('neutral_yaw_deg', 0.0)     # 중립 yaw(정면)
        self.declare_parameter('neutral_pitch_deg', -15.0) # 중립 pitch(약간 아래로 필드를 봄)
        self.declare_parameter('neutral_settle_deg', 0.5)  # 중립 근처 도달 판정 여유

        # --- 명령 발행 최적화 ---
        # 명령 변화가 이값 미만이면 발행 생략(데드존/중립 정착 시 버스 트래픽 절감).
        self.declare_parameter('publish_epsilon_deg', 0.05)
        self.declare_parameter('publish_debug', True)      # 상태/오차 디버그 발행

        # =====================================================================
        #  파라미터 읽기
        # =====================================================================
        self._ball_name = str(self.get_parameter('ball_class_name').value)
        self._min_score = float(self.get_parameter('min_score').value)
        self._min_valid_z = float(self.get_parameter('min_valid_z_m').value)

        self._rate = float(self.get_parameter('control_rate_hz').value)
        self._kp_yaw = float(self.get_parameter('kp_yaw').value)
        self._kp_pitch = float(self.get_parameter('kp_pitch').value)
        self._deadzone = float(self.get_parameter('deadzone_deg').value)
        self._ema = float(self.get_parameter('ema_alpha').value)

        self._max_step_yaw = float(self.get_parameter('max_step_yaw_deg').value)
        self._max_step_pitch = float(self.get_parameter('max_step_pitch_deg').value)

        self._yaw_min = float(self.get_parameter('yaw_min_deg').value)
        self._yaw_max = float(self.get_parameter('yaw_max_deg').value)
        self._pitch_min = float(self.get_parameter('pitch_min_deg').value)
        self._pitch_max = float(self.get_parameter('pitch_max_deg').value)

        self._lost_timeout = float(self.get_parameter('lost_timeout_sec').value)
        self._neutral_yaw = float(self.get_parameter('neutral_yaw_deg').value)
        self._neutral_pitch = float(self.get_parameter('neutral_pitch_deg').value)
        self._neutral_settle = float(self.get_parameter('neutral_settle_deg').value)

        # 추적 시 pitch 위쪽 유효 한계. cap_pitch_to_start 면 시작각(중립)으로 묶는다
        # → 목이 시작각 위로 절대 안 올라가 바닥의 공을 시야에서 놓치지 않는다.
        self._cap_pitch_to_start = bool(self.get_parameter('cap_pitch_to_start').value)
        self._pitch_track_max = (min(self._pitch_max, self._neutral_pitch)
                                 if self._cap_pitch_to_start else self._pitch_max)
        # pitch 이하면 yaw 0 고정 임계 / 분실 후 위치 유지 시간
        self._yaw_lock_pitch = float(self.get_parameter('yaw_lock_pitch_deg').value)
        self._lost_hold = float(self.get_parameter('lost_hold_sec').value)

        self._pub_eps = float(self.get_parameter('publish_epsilon_deg').value)
        self._debug = bool(self.get_parameter('publish_debug').value)

        # =====================================================================
        #  내부 상태
        # =====================================================================
        # 측정된 현재 목각(neck_control_node 피드백). None = 아직 못 받음.
        self._cur_yaw = None
        self._cur_pitch = None
        self._state_ready = False        # 현재각을 한 번이라도 받아 cmd 를 초기화했는가

        # 마지막으로 '보낸' 명령(= 명령 궤적 상의 현재 위치). rate-limit 기준점.
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0
        # 마지막으로 '실제 발행한' 값(발행 생략 판정용).
        self._last_pub_yaw = None
        self._last_pub_pitch = None

        # 최신 공 각오차(EMA 평활). None = 아직 공 본 적 없음.
        self._ball_h_deg = None
        self._ball_v_deg = None
        self._ball_stamp = None          # 마지막 공 검출 시각(rclpy Time)

        self._warned_frame = False       # 광학프레임 경고 1회용

        # =====================================================================
        #  퍼블리셔 / 서브스크라이버 / 타이머
        # =====================================================================
        yaw_cmd_topic = self.get_parameter('yaw_cmd_topic').value
        pitch_cmd_topic = self.get_parameter('pitch_cmd_topic').value
        self.pub_yaw = self.create_publisher(Float32, yaw_cmd_topic, 10)
        self.pub_pitch = self.create_publisher(Float32, pitch_cmd_topic, 10)

        if self._debug:
            self.pub_state = self.create_publisher(
                String, self.get_parameter('state_topic').value, 10)
            self.pub_err = self.create_publisher(
                Float32MultiArray, self.get_parameter('error_topic').value, 10)

        self.create_subscription(
            Detection3DArray, self.get_parameter('detections_topic').value,
            self._on_detections, 10)
        self.create_subscription(
            Float32, self.get_parameter('yaw_state_topic').value,
            self._on_yaw_state, 10)
        self.create_subscription(
            Float32, self.get_parameter('pitch_state_topic').value,
            self._on_pitch_state, 10)
        self._timer = self.create_timer(1.0 / self._rate, self._control_loop)

        self.get_logger().info(
            f"neck_ball_tracker 시작 — ball='{self._ball_name}' "
            f"Kp=({self._kp_yaw},{self._kp_pitch}) deadzone={self._deadzone}deg "
            f"max_step=({self._max_step_yaw},{self._max_step_pitch})deg@ {self._rate}Hz "
            f"neutral=({self._neutral_yaw},{self._neutral_pitch})deg "
            f"pitch=[{self._pitch_min},{self._pitch_track_max}]deg"
            f"{'(시작각 cap)' if self._cap_pitch_to_start else ''} "
            f"yaw_lock<=pitch{self._yaw_lock_pitch}deg "
            f"lost_timeout={self._lost_timeout}s hold={self._lost_hold}s")

    # =========================================================================
    #  현재 목각 콜백 — cmd 초기화(첫 수신 시 점프 방지)
    # =========================================================================
    def _on_yaw_state(self, msg: Float32):
        v = float(msg.data)
        if not math.isfinite(v):
            # NaN/inf 피드백은 무시 → 직전 정상값 유지. 브리지가 잠깐 결손돼도
            # 옛 각도로 버틸 뿐, NaN 이 제어/명령으로 새지 않는다.
            self.get_logger().warn(
                'yaw 현재각 비정상(NaN/inf) — 무시하고 직전값 유지',
                throttle_duration_sec=2.0)
            return
        self._cur_yaw = v
        self._try_init_cmd()

    def _on_pitch_state(self, msg: Float32):
        v = float(msg.data)
        if not math.isfinite(v):
            self.get_logger().warn(
                'pitch 현재각 비정상(NaN/inf) — 무시하고 직전값 유지',
                throttle_duration_sec=2.0)
            return
        self._cur_pitch = v
        self._try_init_cmd()

    def _try_init_cmd(self):
        """yaw/pitch 현재각을 모두 받으면 cmd 를 거기서 시작(첫 명령 점프 방지)."""
        if (not self._state_ready
                and self._cur_yaw is not None
                and self._cur_pitch is not None):
            self._cmd_yaw = self._cur_yaw
            self._cmd_pitch = self._cur_pitch
            self._state_ready = True
            self.get_logger().info(
                f"현재 목각 수신 — cmd 초기화 yaw={self._cmd_yaw:.1f} "
                f"pitch={self._cmd_pitch:.1f} deg")

    # =========================================================================
    #  검출 콜백 — 'ball' 중 최고 신뢰도 1개의 각오차를 EMA 로 저장
    # =========================================================================
    def _on_detections(self, msg: Detection3DArray):
        # 검출 frame 이 광학프레임이 아니면 부호/각도 가정이 깨진다 — 1회 경고.
        frame = msg.header.frame_id or ''
        if (not self._warned_frame) and ('optical' not in frame):
            self.get_logger().warn(
                f"검출 frame_id='{frame}' 가 광학프레임이 아닐 수 있음. "
                f"이 노드는 camera_color_optical_frame(x오른쪽 y아래 z전방)을 가정함.")
            self._warned_frame = True

        # --- 'ball' 중 최고 점수 1개 선택 ---
        best_p = None
        best_score = -1.0
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            # 검출 노드가 class_id 에 이름 문자열을 넣음('ball' 등).
            if str(hyp.hypothesis.class_id) != self._ball_name:
                continue
            score = float(hyp.hypothesis.score)
            if score < self._min_score:
                continue
            p = det.bbox.center.position   # 광학프레임 3D 점
            # NaN/inf 좌표(브리지 결손/깊이 이상)는 즉시 버린다.
            #   주의: NaN 은 모든 비교가 False 라 단순 'z < thr' 만으론 걸러지지
            #   않고 통과해버린다 → atan2(NaN) → NaN 명령으로 새므로 isfinite 먼저.
            if not (math.isfinite(p.x) and math.isfinite(p.y)
                    and math.isfinite(p.z)):
                continue
            if float(p.z) < self._min_valid_z:
                continue                   # 깊이 무효(투영 불가) → 버림
            if score > best_score:
                best_score = score
                best_p = p

        if best_p is None:
            return  # 이번 프레임엔 공 없음. 분실 판정은 타이머가 stamp 나이로 처리.

        # --- 광축 대비 각오차(deg) ---
        h_deg = math.degrees(math.atan2(float(best_p.x), float(best_p.z)))  # +오른쪽
        v_deg = math.degrees(math.atan2(float(best_p.y), float(best_p.z)))  # +아래

        now = self.get_clock().now()

        # 직전 공이 '신선'했는지(연속 추적 중인지). 분실 후 재등장이면 EMA 리셋.
        cont = (self._ball_stamp is not None
                and (now - self._ball_stamp).nanoseconds * 1e-9 <= self._lost_timeout
                and self._ball_h_deg is not None)
        if cont:
            a = self._ema
            self._ball_h_deg = a * h_deg + (1.0 - a) * self._ball_h_deg
            self._ball_v_deg = a * v_deg + (1.0 - a) * self._ball_v_deg
        else:
            # 재포착 첫 프레임: 평활 없이 즉시 반영(지연 방지)
            self._ball_h_deg = h_deg
            self._ball_v_deg = v_deg

        self._ball_stamp = now

    # =========================================================================
    #  제어 루프(고정 주기) — 목표각 계산 → rate-limit → 발행
    # =========================================================================
    def _control_loop(self):
        # 현재 목각을 아직 못 받았으면 명령 금지(점프 방지).
        if not self._state_ready:
            return

        now = self.get_clock().now()
        fresh = (self._ball_stamp is not None
                 and (now - self._ball_stamp).nanoseconds * 1e-9 <= self._lost_timeout)
        if fresh:
            # ── TRACKING: 측정 현재각 + 이미지 오차로 '절대 목표각' 재계산 ──
            #    (매 사이클 측정에서 새로 계산하므로 적분 와인드업이 없다)
            state = 'TRACKING'
            err_h = self._ball_h_deg
            err_v = self._ball_v_deg

            # 데드존: 오차가 작으면 그 축은 현재 cmd 유지(미세 떨림 방지)
            if abs(err_h) <= self._deadzone:
                tgt_yaw = self._cmd_yaw
            else:
                tgt_yaw = _clamp(self._cur_yaw - self._kp_yaw * err_h,
                                 self._yaw_min, self._yaw_max)

            if abs(err_v) <= self._deadzone:
                tgt_pitch = self._cmd_pitch
            else:
                tgt_pitch = _clamp(self._cur_pitch - self._kp_pitch * err_v,
                                   self._pitch_min, self._pitch_track_max)
        else:
            # ── 분실 ── lost_timeout 경과 후 lost_hold 동안은 마지막 위치 유지,
            #            그 뒤에도 공이 없으면 중립(ready)으로 복귀.
            held = (self._ball_stamp is not None
                    and (now - self._ball_stamp).nanoseconds * 1e-9
                        <= self._lost_timeout + self._lost_hold)
            if held:
                state = 'HOLD'                    # 마지막 목 위치 유지(중립 복귀 마진)
                tgt_yaw = self._cmd_yaw
                tgt_pitch = self._cmd_pitch
            else:
                # 마진까지 지나도 공 없음 → 중립 복귀
                tgt_yaw = self._neutral_yaw
                tgt_pitch = self._neutral_pitch
                if (abs(self._cmd_yaw - self._neutral_yaw) <= self._neutral_settle
                        and abs(self._cmd_pitch - self._neutral_pitch) <= self._neutral_settle):
                    state = 'NEUTRAL'
                else:
                    state = 'RETURN'

        # ── pitch 가 임계 이하(깊이 아래를 봄)면 yaw 0°(정면) 고정 — 정면 바닥 스캔 ──
        #    (어느 상태든 적용. cmd_pitch=현재 목 명령 기준)
        if self._cmd_pitch <= self._yaw_lock_pitch:
            tgt_yaw = 0.0

        # ── 목표각으로 명령을 rate-limit 하며 한 스텝 접근 ──
        self._cmd_yaw += _clamp(tgt_yaw - self._cmd_yaw,
                                -self._max_step_yaw, self._max_step_yaw)
        self._cmd_pitch += _clamp(tgt_pitch - self._cmd_pitch,
                                  -self._max_step_pitch, self._max_step_pitch)
        # 안전상 최종 범위 clamp (pitch 위쪽은 시작각 cap 적용)
        self._cmd_yaw = _clamp(self._cmd_yaw, self._yaw_min, self._yaw_max)
        self._cmd_pitch = _clamp(self._cmd_pitch, self._pitch_min, self._pitch_track_max)

        # ── 변화가 의미 있을 때만 발행(트래픽 절감) ──
        self._publish_if_changed()

        # ── 디버그 발행 ──
        if self._debug:
            sm = String()
            sm.data = state
            self.pub_state.publish(sm)
            em = Float32MultiArray()
            if fresh:
                em.data = [float(self._ball_h_deg), float(self._ball_v_deg)]
            else:
                em.data = [0.0, 0.0]
            self.pub_err.publish(em)

    def _publish_if_changed(self):
        """명령이 직전 발행값과 epsilon 이상 다를 때만 발행한다.
        최종 안전장치로 NaN/inf 명령은 절대 발행하지 않는다 — neck_control_node 의
        int(round(NaN)) 크래시/오동작을 원천 차단한다(여기까지 NaN 이 올 일은
        위 가드로 거의 없지만, 명령 토픽은 마지막 방어선이라 한 번 더 확인)."""
        if math.isfinite(self._cmd_yaw) and (
                self._last_pub_yaw is None
                or abs(self._cmd_yaw - self._last_pub_yaw) > self._pub_eps):
            self.pub_yaw.publish(Float32(data=float(self._cmd_yaw)))
            self._last_pub_yaw = self._cmd_yaw

        if math.isfinite(self._cmd_pitch) and (
                self._last_pub_pitch is None
                or abs(self._cmd_pitch - self._last_pub_pitch) > self._pub_eps):
            self.pub_pitch.publish(Float32(data=float(self._cmd_pitch)))
            self._last_pub_pitch = self._cmd_pitch


def main(args=None):
    rclpy.init(args=args)
    node = NeckBallTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

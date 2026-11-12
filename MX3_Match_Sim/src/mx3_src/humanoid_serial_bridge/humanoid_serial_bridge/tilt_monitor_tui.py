#!/usr/bin/env python3
# =============================================================================
#  tilt_monitor_tui.py
# -----------------------------------------------------------------------------
#  목적
#    로봇의 앞/뒤(pitch), 좌/우(roll) 기울기를 터미널에 막대그래프로 실시간
#    표시하고, "넘어짐 방향" 판정 임계값(threshold)을 키보드로 그 자리에서
#    조정하면서 판정 결과가 어떻게 바뀌는지 눈으로 확인하기 위한 도구.
#
#  배경 / 왜 이 노드가 필요한가
#    현재 넘어짐 방향 판정은 standup_behavior.py 에서 단 한 줄로 처리된다:
#
#        self._fall_direction = "forward" if gravity_z > -0.3 else "backward"
#
#    즉 IMU projected gravity 벡터의 z 성분(gz) 하나만 보고, 경계값 -0.3 을
#    기준으로 앞/뒤를 가른다. 그런데 이 -0.3 이라는 값이 실제 로봇이 앞으로
#    엎어졌을 때 / 뒤로 자빠졌을 때 gz 가 실제로 어떤 값을 갖는지 측정한 적이
#    없어서 근거가 불분명하다.
#
#    이 노드는 로봇을 직접 앞으로/뒤로 눕혀 보면서 그때그때의 gz 와 pitch
#    값을 화면에서 관찰하게 해 준다. 세션 동안 관측된 최소/최대값도 계속
#    기록하므로, "앞으로 누우면 gz 가 대략 얼마, 뒤로 누우면 얼마"가
#    자연스럽게 화면에 남고, 그 두 값의 가운데쯤을 임계값으로 잡으면 된다.
#
#  구독 토픽
#    /imu/projected_gravity   (geometry_msgs/Vector3Stamped)
#        imu_parser_node.py 가 발행하는 body-frame 중력 벡터.
#        정자세에서 대략 (0, 0, -1).
#    /imu/data                (sensor_msgs/Imu)   ※ 보조 (각속도 표시용)
#
#  pitch / roll 계산식
#    standup_node.py 와 동일한 식을 그대로 사용하여 일관성을 유지한다:
#        pitch = atan2(gx, -gz)   → 앞뒤 기울기
#        roll  = atan2(gy, -gz)   → 좌우 기울기
#
#  키 조작 (curses 화면 안에서)
#    ↑ / ↓        : 데드존 임계값을 +0.01 / -0.01 조정 (0.0~1.0)
#    → / ←        : 데드존 임계값을 +0.05 / -0.05 (큰 폭) 조정
#    r            : 관측 기록 및 임계값 초기화
#    f            : 현재 gx 값을 "앞으로 누운 상태"로 기록 (마커)
#    b            : 현재 gx 값을 "뒤로 누운 상태"로 기록 (마커)
#    q / ESC      : 종료
#
#  실행 방법
#    1) imu_parser_node 가 먼저 떠 있어야 한다 (IMU 데이터 발행원).
#       ros2 run humanoid_serial_bridge imu_parser_node
#    2) 별도 터미널에서:
#       ros2 run humanoid_serial_bridge tilt_monitor_tui
#
#  주의
#    curses TUI 이므로 반드시 실제 터미널에서 실행할 것.
#    NoMachine 원격 터미널에서도 정상 동작한다.
# =============================================================================

import math
import curses
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped


# =============================================================================
#  ROS 노드 부분
#    - ROS 콜백으로 최신 IMU / 중력 데이터를 받아 보관만 한다.
#    - 화면 그리기는 별도 스레드(curses 루프)에서 수행한다.
#    - 콜백 스레드와 화면 스레드가 같은 변수를 건드리므로 Lock 으로 보호한다.
# =============================================================================
class TiltMonitorNode(Node):
    def __init__(self):
        super().__init__('tilt_monitor_tui')

        # ── 최신 센서값 저장용 변수 ─────────────────────────────
        # projected gravity 의 3개 성분. 초기값은 정자세(0,0,-1)로 둔다.
        self.gx = 0.0
        self.gy = 0.0
        self.gz = -1.0

        # IMU 각속도(rad/s). 화면에 보조 정보로 표시한다.
        self.wx = 0.0
        self.wy = 0.0
        self.wz = 0.0

        # 데이터가 한 번이라도 들어왔는지 여부 (안 들어오면 화면에 경고 표시)
        self.got_gravity = False
        self.got_imu = False

        # ── 콜백/화면 스레드 간 동시 접근 보호용 Lock ──────────
        self.lock = threading.Lock()

        # ── 토픽 구독 ───────────────────────────────────────────
        # imu_parser_node 가 발행하는 body-frame projected gravity
        self.create_subscription(
            Vector3Stamped, '/imu/projected_gravity', self.cb_gravity, 20
        )
        # raw IMU (각속도 표시에만 사용)
        self.create_subscription(
            Imu, '/imu/data', self.cb_imu, 20
        )

    def cb_gravity(self, msg: Vector3Stamped):
        """projected gravity 콜백: 최신 중력 벡터를 저장."""
        with self.lock:
            self.gx = msg.vector.x
            self.gy = msg.vector.y
            self.gz = msg.vector.z
            self.got_gravity = True

    def cb_imu(self, msg: Imu):
        """IMU 콜백: 각속도만 저장 (보조 정보)."""
        with self.lock:
            self.wx = msg.angular_velocity.x
            self.wy = msg.angular_velocity.y
            self.wz = msg.angular_velocity.z
            self.got_imu = True

    def snapshot(self):
        """
        화면 그리기 스레드가 호출.
        Lock 으로 보호된 상태에서 현재 센서값 묶음을 복사해 돌려준다.
        (콜백이 중간에 값을 바꿔서 화면이 깨지는 것을 방지)
        """
        with self.lock:
            return {
                'gx': self.gx, 'gy': self.gy, 'gz': self.gz,
                'wx': self.wx, 'wy': self.wy, 'wz': self.wz,
                'got_gravity': self.got_gravity,
                'got_imu': self.got_imu,
            }


# =============================================================================
#  화면 그리기 헬퍼
# =============================================================================
def make_bar(value, vmin, vmax, width):
    """
    값(value)을 [vmin, vmax] 범위에 대응시켜 가운데가 0인 막대그래프 문자열을
    만든다. 중앙 기준으로 왼쪽/오른쪽으로 막대가 자라난다.

    예) pitch 막대:
        왼쪽 끝 = 뒤로 최대로 누움, 오른쪽 끝 = 앞으로 최대로 누움
        가운데 '|' = 0도 (정자세)

    인자
        value : 표시할 값
        vmin  : 막대 왼쪽 끝에 해당하는 값
        vmax  : 막대 오른쪽 끝에 해당하는 값
        width : 막대 전체 칸 수 (홀수로 주면 중앙이 정확히 잡힘)

    반환
        (bar_str, marker_pos)
        bar_str    : 막대 문자열
        marker_pos : 막대 안에서 현재 값의 위치 인덱스 (색칠 등에 사용)
    """
    # 전체 범위에서 value 가 차지하는 비율 (0.0 ~ 1.0)
    span = vmax - vmin
    if span <= 0:
        ratio = 0.5
    else:
        ratio = (value - vmin) / span
    # 범위를 벗어나면 끝에 고정
    ratio = max(0.0, min(1.0, ratio))

    # 비율을 막대 칸 인덱스로 변환
    pos = int(round(ratio * (width - 1)))

    center = (width - 1) // 2  # 0에 해당하는 칸

    # 막대 문자열 생성: 중앙(center)부터 현재 위치(pos)까지 '#' 로 채움
    chars = []
    for i in range(width):
        if i == center:
            chars.append('|')          # 중앙(0도) 표시
        elif center < i <= pos:
            chars.append('#')          # 중앙보다 오른쪽(앞으로 기울어짐)
        elif pos <= i < center:
            chars.append('#')          # 중앙보다 왼쪽(뒤로 기울어짐)
        else:
            chars.append('.')          # 빈 칸
    return ''.join(chars), pos


# =============================================================================
#  curses TUI 메인 루프
#    - 약 20Hz 로 화면을 다시 그린다.
#    - 키 입력을 받아 임계값을 조정한다.
# =============================================================================
def tui_loop(stdscr, node: 'TiltMonitorNode'):
    # ── curses 초기 설정 ────────────────────────────────────────
    curses.curs_set(0)          # 커서 숨김
    stdscr.nodelay(True)        # getch() 가 입력 없으면 즉시 -1 반환 (논블로킹)
    stdscr.timeout(50)          # getch() 대기시간 50ms → 약 20Hz 갱신

    # 색상 사용 가능하면 색상 페어 정의
    use_color = curses.has_colors()
    if use_color:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # 정자세 / 안전
        curses.init_pair(2, curses.COLOR_YELLOW, -1)  # 주의
        curses.init_pair(3, curses.COLOR_RED, -1)     # 넘어짐 판정
        curses.init_pair(4, curses.COLOR_CYAN, -1)    # 라벨/제목

    def color(n):
        """색상 페어를 attribute 로 변환. 색상 미지원이면 0(기본)."""
        return curses.color_pair(n) if use_color else 0

    # ── 조정 대상 상태값 ────────────────────────────────────────
    # 넘어짐 방향 판정 임계값(데드존).
    #
    #   판정 기준은 gz 가 아니라 projected gravity 의 gx 성분이다.
    #   - gz 는 "얼마나 누웠는지"만 알려줄 뿐 앞/뒤를 구분하지 못한다.
    #     (실측: 앞으로 누움 gz=-0.44, 뒤로 누움 gz=-0.28 으로 거의 같음)
    #   - gx 는 앞/뒤에 따라 부호가 명확히 갈린다.
    #     (실측: 앞으로 누움 gx=+0.90, 뒤로 누움 gx=-0.96)
    #
    #   판정 규칙:
    #     gx >  threshold   -> FORWARD  (앞으로 넘어짐)
    #     gx < -threshold   -> BACKWARD (뒤로 넘어짐)
    #     그 사이(데드존)   -> 판정 보류 (거의 정자세이거나 옆으로 누움)
    #
    #   threshold 는 0~1 사이의 양수. 0 에 가까울수록 민감하게,
    #   1 에 가까울수록 확실히 누웠을 때만 판정한다.
    threshold = 0.3

    # 세션 동안 관측된 최소/최대 추적용. 처음엔 None.
    pitch_min = None
    pitch_max = None
    gx_min = None       # gx 최소 (뒤로 누울수록 -1 쪽)
    gx_max = None       # gx 최대 (앞으로 누울수록 +1 쪽)

    # f / b 키로 찍는 마커 (앞으로/뒤로 누운 상태의 gx 값을 수동 기록)
    marker_fwd = None   # "앞으로 누움" 으로 기록한 gx
    marker_bwd = None   # "뒤로 누움"  으로 기록한 gx

    while rclpy.ok():
        # ── 최신 센서값 스냅샷 ──────────────────────────────────
        s = node.snapshot()
        gx, gy, gz = s['gx'], s['gy'], s['gz']
        wx, wy, wz = s['wx'], s['wy'], s['wz']

        # ── pitch / roll 계산 (standup_node.py 와 동일한 식) ────
        # pitch: 앞뒤 기울기. 앞으로 숙이면 부호가 한쪽으로 커진다.
        # roll : 좌우 기울기.
        pitch_deg = math.degrees(math.atan2(gx, -gz))
        roll_deg = math.degrees(math.atan2(gy, -gz))

        # ── 관측 min/max 갱신 ──────────────────────────────────
        if pitch_min is None or pitch_deg < pitch_min:
            pitch_min = pitch_deg
        if pitch_max is None or pitch_deg > pitch_max:
            pitch_max = pitch_deg
        if gx_min is None or gx < gx_min:
            gx_min = gx
        if gx_max is None or gx > gx_max:
            gx_max = gx

        # ── 현재 임계값으로 넘어짐 방향 판정 ───────────────────
        # standup_behavior.py 와 동일하게 gx 부호 + 데드존으로 판정한다.
        if gx > threshold:
            verdict = "FORWARD (앞으로 넘어짐)"
            verdict_color = color(3)
        elif gx < -threshold:
            verdict = "BACKWARD (뒤로 넘어짐)"
            verdict_color = color(3)
        else:
            # 데드존: 앞/뒤 어느 쪽도 아님.
            # 거의 서 있거나(gz 가 -1 근처), 옆으로 누운 경우다.
            if gz < -0.85:
                verdict = "STANDING (정자세에 가까움)"
                verdict_color = color(1)
            else:
                verdict = "UNCLEAR (앞/뒤 불명 - 옆으로 누움?)"
                verdict_color = color(2)

        # ── 화면 그리기 ─────────────────────────────────────────
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        bar_w = max(21, min(61, w - 24))  # 막대 폭 (터미널 크기에 맞춤, 홀수 유지)
        if bar_w % 2 == 0:
            bar_w -= 1

        row = 0

        def put(text, attr=0):
            """한 줄 출력 헬퍼. 화면 범위를 벗어나면 무시한다."""
            nonlocal row
            if 0 <= row < h:
                try:
                    stdscr.addnstr(row, 0, text, w - 1, attr)
                except curses.error:
                    pass
            row += 1

        # 제목
        put("=" * (w - 1), color(4))
        put("  KHUBER  TILT MONITOR  -  넘어짐 임계값 튜닝 도구", color(4))
        put("=" * (w - 1), color(4))
        put("")

        # 센서 수신 상태 경고
        if not s['got_gravity']:
            put("  [!] /imu/projected_gravity 수신 없음 - imu_parser_node 실행 확인",
                color(3))
            put("")

        # ── pitch 막대 (앞뒤) ──────────────────────────────────
        # 막대 범위는 -90도 ~ +90도. 넘어지면 보통 이 범위 안에 들어온다.
        pitch_bar, _ = make_bar(pitch_deg, -90.0, 90.0, bar_w)
        put("  PITCH (앞뒤 기울기)", color(4))
        put("  뒤로 <--{}--> 앞으로".format(" " * (bar_w - 16)))
        put("  [{}]  {:+7.2f} deg".format(pitch_bar, pitch_deg))
        put("")

        # ── roll 막대 (좌우) ───────────────────────────────────
        roll_bar, _ = make_bar(roll_deg, -90.0, 90.0, bar_w)
        put("  ROLL (좌우 기울기)", color(4))
        put("  왼쪽 <--{}--> 오른쪽".format(" " * (bar_w - 16)))
        put("  [{}]  {:+7.2f} deg".format(roll_bar, roll_deg))
        put("")

        # ── projected gravity 원시값 ───────────────────────────
        put("  PROJECTED GRAVITY (body frame)", color(4))
        put("    gx={:+.3f}   gy={:+.3f}   gz={:+.3f}".format(gx, gy, gz))
        put("")

        # ── gx 막대 (앞/뒤 판정의 핵심 지표) ───────────────────
        # gx 범위 -1.0 ~ +1.0. 막대에 데드존 경계(threshold)를 함께 표시.
        gx_bar, _ = make_bar(gx, -1.0, 1.0, bar_w)
        put("  GX  <-- 넘어짐 방향 판정 기준 -->", color(4))
        put("  뒤로(-){}앞으로(+)".format(" " * (bar_w - 14)))
        put("  [{}]  gx={:+.3f}".format(gx_bar, gx))
        # 데드존 경계를 ^ 표시로 나타낸다.
        center = (bar_w - 1) // 2
        # threshold(양수)를 막대 인덱스로 변환: gx=+threshold 위치, gx=-threshold 위치
        pos_hi = int(round(((threshold + 1.0) / 2.0) * (bar_w - 1)))
        pos_lo = int(round(((-threshold + 1.0) / 2.0) * (bar_w - 1)))
        marker_line = [' '] * bar_w
        if 0 <= pos_lo < bar_w:
            marker_line[pos_lo] = '^'
        if 0 <= pos_hi < bar_w:
            marker_line[pos_hi] = '^'
        put("   {}".format(''.join(marker_line)))
        put("   ^표시 사이 = 데드존(판정 보류)")
        put("")

        # ── gz 막대 (누운 정도 - 참고용) ───────────────────────
        # gz 막대: 정자세 -1.0, 완전히 누우면 0 근처 또는 양수
        gz_bar, _ = make_bar(gz, -1.0, 1.0, bar_w)
        put("  GZ  (누운 정도 - 참고용, 앞/뒤 구분 불가)", color(4))
        put("    gz [{}]".format(gz_bar))
        put("       -1.0 (정자세){}+1.0".format(" " * (bar_w - 18)))
        put("")

        # ── 각속도 (보조) ──────────────────────────────────────
        put("  ANGULAR VELOCITY  wx={:+.2f}  wy={:+.2f}  wz={:+.2f} (rad/s)"
            .format(wx, wy, wz))
        put("")

        # ── 임계값 및 판정 결과 ────────────────────────────────
        put("-" * (w - 1), color(4))
        put("  넘어짐 판정 데드존 임계값 (gx) : +-{:.3f}".format(threshold),
            color(2))
        put("    규칙:  gx > +thr -> FORWARD,   gx < -thr -> BACKWARD,"
            "   그 사이 -> 보류")
        put("  현재 판정 : {}".format(verdict), verdict_color)
        put("")

        # ── 세션 관측 min/max ──────────────────────────────────
        put("  === 세션 관측 범위 (임계값 결정 근거) ===", color(4))
        if pitch_min is not None:
            put("    pitch :  최소 {:+7.2f}    최대 {:+7.2f}  deg"
                .format(pitch_min, pitch_max))
            put("    gx    :  최소 {:+7.3f}    최대 {:+7.3f}"
                .format(gx_min, gx_max))
        # f / b 로 찍은 마커 출력
        if marker_fwd is not None:
            put("    [f] 앞으로 누움 기록 : gx = {:+7.3f}".format(marker_fwd),
                color(1))
        if marker_bwd is not None:
            put("    [b] 뒤로  누움 기록 : gx = {:+7.3f}".format(marker_bwd),
                color(1))
        # 두 마커가 다 있으면 추천 임계값 힌트 제공
        if marker_fwd is not None and marker_bwd is not None:
            # 앞(양수)/뒤(음수) gx 의 절댓값 중 작은 쪽의 절반을
            # 데드존 임계값으로 추천한다. (양쪽 다 확실히 판정 가능하도록)
            safe = min(abs(marker_fwd), abs(marker_bwd))
            put("    >> 추천 데드존 임계값 : +-{:.3f}".format(safe * 0.5),
                color(2))
        if gx_min is not None and gx_max is not None:
            # 관측된 gx 범위가 양/음 양쪽으로 충분히 벌어졌는지 안내
            if gx_max > 0.5 and gx_min < -0.5:
                put("    (앞/뒤 모두 측정됨 - 데드존 임계값 결정 가능)",
                    color(1))
            else:
                put("    (앞/뒤 양쪽으로 눕혀 gx 범위를 확보하세요)",
                    color(2))
        put("")

        # ── 키 안내 ────────────────────────────────────────────
        put("-" * (w - 1), color(4))
        put("  [up/down] threshold +-0.01   [left/right] +-0.05", color(4))
        put("  [f] 앞으로누움 기록   [b] 뒤로누움 기록   [r] 리셋   [q] 종료",
            color(4))

        stdscr.refresh()

        # ── 키 입력 처리 ────────────────────────────────────────
        try:
            key = stdscr.getch()
        except curses.error:
            key = -1

        if key in (ord('q'), ord('Q'), 27):       # q 또는 ESC → 종료
            break
        elif key == curses.KEY_UP:                 # 임계값 미세 +
            # 데드존 임계값은 0.0 ~ 1.0 범위의 양수
            threshold = min(1.0, threshold + 0.01)
        elif key == curses.KEY_DOWN:               # 임계값 미세 -
            threshold = max(0.0, threshold - 0.01)
        elif key == curses.KEY_RIGHT:              # 임계값 크게 +
            threshold = min(1.0, threshold + 0.05)
        elif key == curses.KEY_LEFT:               # 임계값 크게 -
            threshold = max(0.0, threshold - 0.05)
        elif key in (ord('r'), ord('R')):          # 리셋
            pitch_min = pitch_max = None
            gx_min = gx_max = None
            marker_fwd = marker_bwd = None
            threshold = 0.3
        elif key in (ord('f'), ord('F')):          # 현재 자세를 "앞으로 누움" 기록
            marker_fwd = gx                        # gx 값을 기록 (판정 기준)
        elif key in (ord('b'), ord('B')):          # 현재 자세를 "뒤로 누움" 기록
            marker_bwd = gx

    # 루프 종료 후, 마지막으로 정한 임계값을 터미널에 남겨 준다.
    # (curses 종료 후 stdout 으로 출력되도록 node 에 보관)
    node._final_threshold = threshold


# =============================================================================
#  main
# =============================================================================
def main():
    rclpy.init()
    node = TiltMonitorNode()
    node._final_threshold = None

    # ROS 콜백 처리를 별도 스레드에서 돌린다.
    # (메인 스레드는 curses 화면 루프를 담당)
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    try:
        # curses.wrapper: 화면 초기화/복원을 자동 처리.
        # 예외가 나도 터미널이 깨지지 않도록 보장한다.
        curses.wrapper(tui_loop, node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    # curses 종료 후 정상 stdout 으로 결과 출력
    if node._final_threshold is not None:
        thr = node._final_threshold
        print("\n[tilt_monitor] 최종 설정한 데드존 임계값: "
              "+-{:.3f}".format(thr))
        print("[tilt_monitor] standup_behavior.py 의 start() 판정식을 "
              "다음과 같이 작성하세요:")
        print("    FALL_DIR_THRESHOLD = {:.3f}".format(thr))
        print("    if gravity_x > FALL_DIR_THRESHOLD:")
        print('        self._fall_direction = "forward"')
        print("    elif gravity_x < -FALL_DIR_THRESHOLD:")
        print('        self._fall_direction = "backward"')
        print("    else:")
        print('        return False, "fall direction unclear"')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keyboard_teleop_node.py

KHUBER 보행정책용 키보드 텔레오퍼레이터.
/cmd_vel (geometry_msgs/Twist) 토픽으로 cmd_vx, cmd_vy, cmd_wz를 발행한다.

[축별 속도 범위 — 비대칭 + 최소 시작값(데드존)]
    축이 0(정지)일 때 방향키를 누르면 "최소 크기"로 바로 점프하고,
    이후 같은 방향키는 step 만큼 증가(최대에서 멈춤), 반대 방향키는 step 만큼
    감소하다가 최소 크기 밑으로 내려가면 0(정지)으로 떨어진다.

      전진(vx +):  +0.15 ~ +0.50,  step 0.05   (0에서 W → +0.15부터)
      후진(vx -):  -0.10 ~ -0.30,  step 0.05   (0에서 S → -0.10부터)
      회전(wz ±):  ±0.50 ~ ±1.50,  step 0.10   (0에서 A/D → ±0.50부터)
      횡이동(vy ±): ±0.10 ~ ±0.30,  step 0.05   (0에서 Q/E → ±0.10부터)

    예) wz=+0.5 에서 D(반대방향) → 0.5-0.1=0.4 < 0.5(최소) → 0.0(정지),
        다시 D → 0에서 반대방향 시작이므로 -0.5.

조작법:
    W : 전진 (vx +)        S : 후진 (vx -)
    A : 좌회전 (wz +)      D : 우회전 (wz -)
    Q : 좌횡이동 (vy +)    E : 우횡이동 (vy -)
    F : 킥 (/kick_trigger 발행)
    SPACE : 급정지 (모두 0)
    , : 측정 구간 시작      . : 측정 구간 종료
    X : 측정 구간 취소
    Ctrl+C : 종료

[수정 이력]
    2026-08-28  오도메트리 실험용 구간 마커 추가.
        배경: 5m 구간을 여러 속도로 주행하며 소요 시간을 재려면, 구간의
              시작/끝 시각이 관절 데이터와 같은 시간축에 정확히 남아야
              한다. 스톱워치는 사람 반응지연(편도 0.2~0.3s)이 17초짜리
              측정에서 2% 오차라 측정 대상과 크기가 비슷해 못 쓴다.
        조치: · ',' → 구간 시작, '.' → 구간 종료, 'x' → 취소
              · /experiment/marker (std_msgs/String) 로 발행
              · 페이로드에 구간번호와 그 순간의 cmd 값을 함께 실어,
                분석 스크립트가 별도 대조 없이 구간을 자를 수 있게 한다
              · 종료 즉시 터미널에 소요 시간을 띄운다. 이상한 값이 나오면
                그 자리에서 다시 하기 위함이다.
        ★ 키 선택 이유 ★
              처음에 '[' ']' 를 검토했으나 '[' 는 방향키 이스케이프
              시퀀스(ESC [ A)의 두 번째 바이트다. 방향키 오입력이 그대로
              구간 마커가 되어 데이터를 오염시킨다. ',' '.' 는 물리적으로
              나란히 있어 손가락 이동도 없다.

    2026-08-28  방향키 오입력 버그 수정.
        증상: 방향키를 누르면 의도치 않게 좌회전(wz+)이 걸린다.
        원인: 방향키는 ESC '[' 'A' 3바이트로 오는데, get_key() 가 1바이트씩
              읽고 process_key() 가 key.lower() 를 하므로 마지막 'A' 가
              'a'(좌회전) 로 매칭된다. '[' 도 별개 키로 한 번 더 들어온다.
        조치: get_key() 에서 ESC(0x1b) 를 만나면 뒤따르는 시퀀스를 모두
              흘려보내고 '' 를 반환한다.
"""

import sys
import termios
import time
import tty
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


# ── 키 매핑 ──────────────────────────────────────────
# 값은 (vx_dir, vy_dir, wz_dir) — 각 축에 줄 "방향"(+1/-1/0)만 의미한다.
# 실제 증감 크기는 축별 범위 함수가 결정한다(여기서는 방향만 전달).
KEY_BINDINGS = {
    'w': ( 1,  0,  0),   # 전진
    's': (-1,  0,  0),   # 후진
    'a': ( 0,  0,  1),   # 좌회전
    'd': ( 0,  0, -1),   # 우회전
    'q': ( 0,  1,  0),   # 좌횡이동
    'e': ( 0, -1,  0),   # 우횡이동
}

HELP_TEXT = """
╔══════════════════════════════════════════════════╗
║         KHUBER Humanoid Keyboard Teleop          ║
╠══════════════════════════════════════════════════╣
║   W : 전진  (+0.15 ~ +0.50, 0.05)                ║
║   S : 후진  (-0.10 ~ -0.30, 0.05)                ║
║   A : 좌회전 (+0.50 ~ +1.50, 0.10)               ║
║   D : 우회전 (-0.50 ~ -1.50, 0.10)               ║
║   Q : 좌횡이동 (+0.10 ~ +0.30, 0.05)             ║
║   E : 우횡이동 (-0.10 ~ -0.30, 0.05)             ║
║   F : 킥 (/kick_trigger 발행)                     ║
║   SPACE : 급정지 (모두 0)                         ║
╠══════════════════════════════════════════════════╣
║   , : 측정 구간 시작    . : 측정 구간 종료         ║
║   X : 측정 구간 취소 (잘못 눌렀을 때)              ║
╠══════════════════════════════════════════════════╣
║   Ctrl+C : 종료                                  ║
╚══════════════════════════════════════════════════╝
"""


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__('keyboard_teleop_node')

        # ── 파라미터 (축별 최소/최대 크기와 step) ──
        # vx 는 전후 비대칭이라 pos(전진)/neg(후진)를 따로 둔다. 크기(양수)로 표현.
        self.declare_parameter('vx_pos_min', 0.15)   # 전진 최소
        self.declare_parameter('vx_pos_max', 0.50)   # 전진 최대
        self.declare_parameter('vx_neg_min', 0.10)   # 후진 최소(크기)
        self.declare_parameter('vx_neg_max', 0.30)   # 후진 최대(크기)
        self.declare_parameter('vx_step', 0.05)

        # vy 는 좌우 대칭.
        self.declare_parameter('vy_min', 0.10)
        self.declare_parameter('vy_max', 0.30)
        self.declare_parameter('vy_step', 0.05)

        # wz 는 좌우 대칭.
        self.declare_parameter('wz_min', 0.50)
        self.declare_parameter('wz_max', 1.50)
        self.declare_parameter('wz_step', 0.10)

        self.declare_parameter('publish_rate_hz', 10.0)

        self.vx_pos_min = float(self.get_parameter('vx_pos_min').value)
        self.vx_pos_max = float(self.get_parameter('vx_pos_max').value)
        self.vx_neg_min = float(self.get_parameter('vx_neg_min').value)
        self.vx_neg_max = float(self.get_parameter('vx_neg_max').value)
        self.vx_step = float(self.get_parameter('vx_step').value)

        self.vy_min = float(self.get_parameter('vy_min').value)
        self.vy_max = float(self.get_parameter('vy_max').value)
        self.vy_step = float(self.get_parameter('vy_step').value)

        self.wz_min = float(self.get_parameter('wz_min').value)
        self.wz_max = float(self.get_parameter('wz_max').value)
        self.wz_step = float(self.get_parameter('wz_step').value)

        pub_hz = float(self.get_parameter('publish_rate_hz').value)

        # 킥 트리거 토픽(절대경로 → cmd_vel remap 영향 없음).
        self.declare_parameter('kick_topic', '/kick_trigger')
        kick_topic = str(self.get_parameter('kick_topic').value)

        # ── 실험용 구간 마커 ──
        # 절대경로로 둔다. cmd_vel 을 remap 해도 마커까지 끌려가면 안 된다.
        self.declare_parameter('marker_topic', '/experiment/marker')
        marker_topic = str(self.get_parameter('marker_topic').value)
        # 구간 번호 시작값. 세션을 나눠 찍을 때 이어붙이려면 올려서 실행한다.
        # (파일이 달라도 번호가 겹치지 않아야 나중에 합칠 수 있다)
        self.declare_parameter('segment_id_start', 1)

        # ── 상태 ──
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._event = ''   # 상태줄에 잠깐 띄울 이벤트 메시지(예: KICK)

        # ── 구간 마커 상태 ──
        # _seg_id      : 다음에 부여할 구간 번호
        # _seg_active  : 현재 구간이 열려 있는지
        # _seg_t0      : 구간 시작 시각(ROS 시계, 초)
        # _seg_cmd     : 구간 시작 시점의 명령값. 종료 표시에 같이 쓴다.
        #                ★ 시작 시점 값을 쓰는 게 중요하다 ★ 구간 도중
        #                명령을 바꾸면 그 구간은 무효인데, 종료 시점 값을
        #                쓰면 그 사실이 로그에서 사라진다.
        self._seg_id = int(self.get_parameter('segment_id_start').value)
        self._seg_active = False
        self._seg_t0 = 0.0
        self._seg_cmd = (0.0, 0.0, 0.0)
        # 마지막 구간 결과. 상태줄 아래에 계속 띄워 둔다.
        self._last_result = ''

        # ── 퍼블리셔 ──
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.kick_pub = self.create_publisher(Bool, kick_topic, 10)
        # 마커는 유실되면 그 구간을 통째로 못 쓰므로 큐를 넉넉히 잡는다.
        self.marker_pub = self.create_publisher(String, marker_topic, 20)

        # 키를 안 눌러도 마지막 값을 주기적으로 유지 발행.
        self.timer = self.create_timer(1.0 / pub_hz, self.publish_cmd)

    def get_key(self, timeout: float = 0.05) -> str:
        """stdin에서 키 하나를 non-blocking으로 읽음. 없으면 ''.

        ★ 이스케이프 시퀀스는 통째로 버린다 (2026-08-28) ★
        방향키·기능키는 ESC(0x1b) 로 시작하는 여러 바이트로 들어온다.
        예) ↑ = ESC '[' 'A'
        1 바이트씩 그대로 흘리면 '[' 가 별개 키가 되고, 'A' 는
        process_key() 의 key.lower() 를 거쳐 'a'(좌회전)로 매칭된다.
        즉 방향키를 누르면 로봇이 돈다. 그래서 ESC 를 만나면 뒤따르는
        바이트를 짧은 타임아웃으로 전부 읽어 버리고 '' 를 반환한다.

        ESC 단독 입력(사용자가 ESC 키만 누른 경우)도 '' 가 되는데,
        ESC 에 할당된 기능이 없으므로 문제되지 않는다.
        """
        if not select.select([sys.stdin], [], [], timeout)[0]:
            return ''
        ch = sys.stdin.read(1)
        if ch != '\x1b':
            return ch

        # ESC 뒤에 남은 시퀀스를 흘려보낸다. 타임아웃을 아주 짧게 두어
        # (터미널이 시퀀스를 한 번에 보내므로) 사용자 입력을 삼키지 않는다.
        while select.select([sys.stdin], [], [], 0.002)[0]:
            sys.stdin.read(1)
        return ''

    @staticmethod
    def _step_axis(cur: float, direction: int,
                   pos_min: float, pos_max: float,
                   neg_min: float, neg_max: float,
                   step: float) -> float:
        """
        하나의 축 값을 방향키 입력에 따라 한 단계 갱신한다.

        규칙:
          - cur == 0 에서 누르면 해당 방향의 "최소 크기"로 점프.
          - 같은 방향으로 더 누르면 step 만큼 증가(최대 크기에서 멈춤).
          - 반대 방향으로 누르면 step 만큼 감소하다, 최소 크기 밑으로
            내려가면 0(정지)으로 떨어진다.

        파라미터:
          cur       : 현재 축 값(부호 포함)
          direction : 입력 방향(+1 / -1)
          pos_min/max : + 방향의 최소/최대 크기(양수)
          neg_min/max : - 방향의 최소/최대 크기(양수)
          step      : 증감 단위
        대칭 축(vy, wz)은 pos_*, neg_* 에 같은 값을 넣어 호출하면 된다.
        """
        eps = 1e-9

        # 정지 상태에서 출발 → 해당 방향 최소 크기로 점프
        if abs(cur) < eps:
            if direction > 0:
                return round(pos_min, 3)
            else:
                return round(-neg_min, 3)

        if cur > 0:
            if direction > 0:
                # 전진 방향 증가 (최대에서 clamp)
                return round(min(cur + step, pos_max), 3)
            else:
                # 전진 중 반대키 → 감소, 최소 밑이면 정지
                new = cur - step
                return 0.0 if new < pos_min - eps else round(new, 3)
        else:  # cur < 0
            if direction < 0:
                # 후진/역방향 크기 증가 (최대에서 clamp)
                mag = min(abs(cur) + step, neg_max)
                return round(-mag, 3)
            else:
                # 후진 중 반대키 → 크기 감소, 최소 밑이면 정지
                new_mag = abs(cur) - step
                return 0.0 if new_mag < neg_min - eps else round(-new_mag, 3)

    def process_key(self, key: str):
        """키 입력에 따라 vx, vy, wz 업데이트."""
        if not key:
            return

        k = key.lower()

        # 스페이스 → 급정지
        if k == ' ':
            self.vx = 0.0
            self.vy = 0.0
            self.wz = 0.0
            return

        # F → 킥 트리거 발행 (보행 정지→킥→보행 재개는 actor 노드가 처리)
        if k == 'f':
            self.kick_pub.publish(Bool(data=True))
            self._event = 'KICK!'
            return

        # ── 실험용 구간 마커 ──
        # 텔레옵 키보다 먼저 처리한다. ',' '.' 'x' 는 KEY_BINDINGS 에
        # 없으므로 순서가 결과를 바꾸지는 않지만, 의도를 분명히 한다.
        if k == ',':
            self._seg_start()
            return
        if k == '.':
            self._seg_end()
            return
        if k == 'x':
            self._seg_abort()
            return

        if k in KEY_BINDINGS:
            dvx, dvy, dwz = KEY_BINDINGS[k]
            if dvx != 0:
                self.vx = self._step_axis(
                    self.vx, dvx,
                    self.vx_pos_min, self.vx_pos_max,
                    self.vx_neg_min, self.vx_neg_max,
                    self.vx_step)
            if dvy != 0:
                # vy 대칭 → pos/neg 동일값
                self.vy = self._step_axis(
                    self.vy, dvy,
                    self.vy_min, self.vy_max,
                    self.vy_min, self.vy_max,
                    self.vy_step)
            if dwz != 0:
                # wz 대칭 → pos/neg 동일값
                self.wz = self._step_axis(
                    self.wz, dwz,
                    self.wz_min, self.wz_max,
                    self.wz_min, self.wz_max,
                    self.wz_step)

    # ── 실험용 구간 마커 ──────────────────────────────────────
    def _now_sec(self) -> float:
        """ROS 시계 기준 현재 시각[s].

        ★ time.time() 이 아니라 ROS 시계를 쓴다 ★
        로거와 /joint_states 가 전부 ROS 시각을 쓰므로, 여기서 벽시계를
        쓰면 두 시간축이 어긋나 구간을 정확히 자를 수 없다.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_marker(self, event: str, seg_id: int,
                        cmd: tuple, stamp: float):
        """구간 마커를 발행한다.

        페이로드는 콤마 구분 문자열이다. 전용 msg 타입을 만들지 않은 것은
        인터페이스 패키지를 건드리면 전체 재빌드가 필요한데, 실험 전용
        기능이라 그만한 값이 없기 때문이다. 파싱은 분석 스크립트가 한다.

            SEG,<START|END|ABORT>,<seg_id>,<vx>,<vy>,<wz>,<stamp_sec>

        cmd 는 '구간 시작 시점' 의 명령값이다. START 와 END 에 같은 값이
        실리므로, 분석 시 둘이 다르면 구간 도중 명령이 바뀐 것이 아니라
        마커 페어링이 깨진 것이다(진단에 쓸 수 있다).
        """
        msg = String()
        msg.data = (f'SEG,{event},{seg_id},'
                    f'{cmd[0]:.4f},{cmd[1]:.4f},{cmd[2]:.4f},'
                    f'{stamp:.6f}')
        self.marker_pub.publish(msg)

    def _seg_start(self):
        """측정 구간 시작(',')."""
        if self._seg_active:
            # 이미 열려 있는데 또 시작을 눌렀다. 앞 구간은 종료 마커가
            # 없어 무효이므로 ABORT 로 닫고 새로 연다. 조용히 덮어쓰면
            # 분석 때 START 가 두 번 연속 나와 파싱이 깨진다.
            self._publish_marker('ABORT', self._seg_id, self._seg_cmd,
                                 self._now_sec())
            self._last_result = (f'경고: #{self._seg_id} 종료 없이 재시작 '
                                 f'→ 취소 처리')
            self._seg_id += 1

        self._seg_t0 = self._now_sec()
        self._seg_cmd = (self.vx, self.vy, self.wz)
        self._seg_active = True
        self._publish_marker('START', self._seg_id, self._seg_cmd,
                             self._seg_t0)
        self._event = f'SEG#{self._seg_id} START'

    def _seg_end(self):
        """측정 구간 종료('.')."""
        if not self._seg_active:
            self._last_result = "경고: 시작(',') 없이 종료를 눌렀습니다"
            self._event = 'SEG ERR'
            return

        t1 = self._now_sec()
        dt = t1 - self._seg_t0
        self._publish_marker('END', self._seg_id, self._seg_cmd, t1)

        vx, vy, wz = self._seg_cmd
        # 종료 즉시 소요 시간을 띄운다. 이상한 값이면 그 자리에서 다시
        # 하기 위한 것이고, 이게 실험 효율을 크게 좌우한다.
        self._last_result = (f'#{self._seg_id}  cmd(vx={vx:+.2f} '
                             f'vy={vy:+.2f} wz={wz:+.2f})  '
                             f'{dt:.2f}s')
        # 구간이 비정상적으로 짧으면 오입력일 가능성이 높다.
        if dt < 1.0:
            self._last_result += '  ← 너무 짧음(오입력?)'

        self._seg_active = False
        self._seg_id += 1
        self._event = f'SEG END {dt:.2f}s'

    def _seg_abort(self):
        """측정 구간 취소('x'). 잘못 시작했을 때 되돌린다."""
        if not self._seg_active:
            self._last_result = '취소할 구간이 없습니다'
            return
        self._publish_marker('ABORT', self._seg_id, self._seg_cmd,
                             self._now_sec())
        self._last_result = f'#{self._seg_id} 취소됨'
        self._seg_active = False
        self._seg_id += 1
        self._event = 'SEG ABORT'

    def publish_cmd(self):
        """현재 속도를 /cmd_vel로 발행."""
        msg = Twist()
        msg.linear.x = self.vx
        msg.linear.y = self.vy
        msg.angular.z = self.wz
        self.pub.publish(msg)

    def print_status(self):
        """현재 속도를 한 줄로 출력 (in-place)."""
        ev = f'  [{self._event}]' if self._event else ''

        # 구간이 열려 있으면 경과 시간을 실시간으로 보여준다.
        # 5m 를 0.30 으로 가면 약 17초라, 화면을 보며 도착 시점을 가늠할
        # 수 있어야 종료 키 타이밍이 안정된다.
        if self._seg_active:
            el = self._now_sec() - self._seg_t0
            seg = f'  REC#{self._seg_id} {el:5.1f}s'
        else:
            seg = '  ' + ' ' * 14

        # 2 줄 출력: 위는 실시간 상태, 아래는 마지막 구간 결과.
        # ANSI 이스케이프로 커서를 올렸다 내리며 제자리 갱신한다.
        # (\r 만으로는 두 줄을 갱신할 수 없다)
        line1 = (f'  vx={self.vx:+.2f}  vy={self.vy:+.2f}  '
                 f'wz={self.wz:+.2f}{seg}{ev}')
        line2 = f'  {self._last_result}' if self._last_result else ''

        # 이전 줄 잔상 제거를 위해 각 줄 끝에 '지우기'(\x1b[K) 를 붙인다.
        sys.stdout.write(f'\r{line1}\x1b[K\n{line2}\x1b[K\x1b[1A')
        sys.stdout.flush()
        self._event = ''   # 한 번 보여주고 지움(플래시)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()

    print(HELP_TEXT)

    # 터미널을 raw 모드로 설정
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)

            key = node.get_key(timeout=0.05)
            node.process_key(key)
            node.print_status()

    except KeyboardInterrupt:
        pass
    finally:
        # 터미널 복원을 가장 먼저!
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # 열려 있는 구간이 있으면 ABORT 로 닫는다. 안 닫으면 로거의
        # 구간 CSV 에 START 만 있고 END 가 없는 행이 남아, 분석 스크립트가
        # 마지막 구간을 처리하지 못한다.
        if node._seg_active:
            node._seg_abort()
            # 발행이 나갈 시간을 잠깐 준다(즉시 destroy 하면 유실된다).
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)

        # 종료 시 0 속도 전송
        node.vx = 0.0
        node.vy = 0.0
        node.wz = 0.0
        node.publish_cmd()
        print('\n\n정지 명령 전송 완료. 종료합니다.')

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

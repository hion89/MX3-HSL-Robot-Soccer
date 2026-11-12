#!/usr/bin/env python3
# ==========================================================================
# game_controller_return_node.py
# --------------------------------------------------------------------------
# 각 로봇이 자기 상태(자세/공/넘어짐)를 GameController 로 되돌려 보내는 노드.
# GameController 가 "이 로봇 살아있음(active)"으로 인식하게 하고, TCM(Team
# Communication Monitor) 같은 도구가 우리 로봇의 위치/공 관측을 시각화할 수 있게 한다.
#
# [왜 필요한가]
#   GameController 는 0.5~2Hz 로 status 메시지를 받기를 "요구"한다(HSL 규칙).
#   이 메시지를 안 보내면 GUI 에서 로봇이 회색(inactive)으로 뜨고, 심판이
#   픽업/패널티 판정을 할 때 우리 로봇을 추적하지 못한다.
#
# [패킷 — RoboCupGameControlReturnData (RoboCup-HumanoidSoccerLeague/GameController)]
#   헤더 "RGrt", VERSION 4, 총 32바이트, little-endian, UDP unicast → GC:3939
#   ※ 수신부(game_controller_receiver_node)가 쓰는 컨트롤 패킷("RGme", v20)과
#     return 패킷은 별개 구조/버전이다. 헤더 파일 정의를 그대로 옮긴다.
#
#   offset 0  : char    header[4]   "RGrt"
#   offset 4  : uint8   version     (= 4)
#   offset 5  : uint8   playerNum   1부터 시작
#   offset 6  : uint8   teamNum     우리 팀 번호
#   offset 7  : uint8   fallen      1=넘어짐(플레이 불가), 0=정상
#   offset 8  : float   pose[0] x   [mm]  필드중앙 원점
#   offset 12 : float   pose[1] y   [mm]  +x=공격 골 방향, +y=+x에서 CCW 90°
#   offset 16 : float   pose[2] th  [rad] +x축 기준, CCW 증가
#   offset 20 : float   ballAge     [s]   마지막 공 관측 후 경과(-1=한번도 못봄)
#   offset 24 : float   ball[0] x   [mm]  로봇 기준 상대, +x=로봇 전방
#   offset 28 : float   ball[1] y   [mm]  +y=로봇 기준 CCW 90°
#   → 4 + 4(uint8 4개) + 6*4(float 6개) = 32바이트
#
# [수집 토픽]
#   /localization/vision_pose  (PoseWithCovarianceStamped, frame=map, m)  → pose
#   /ball/map_point            (PointStamped,              frame=map, m)  → ball, ballAge
#   /robot_state               (std_msgs/String)                          → fallen
#     · NORMAL                 → fallen=0 (서서 플레이 가능)
#     · 그 외(PROTECT/FALLEN/STANDING_UP/RECOVERING/STANDUP_FAILED) → fallen=1
#
# [좌표 변환]
#   pose : map(m) → mm. map 프레임이 GC 필드 좌표와 같은 컨벤션(중앙원점, +x 공격,
#          CCW yaw)이라고 가정. 진영(코트)이 반대면 mirror_field=true 로 점대칭
#          (x,y,θ)→(-x,-y,θ+π) 적용. ball 은 로봇 기준이라 mirror 영향 없음.
#   ball : map 공좌표 - map 로봇좌표 → 로봇 yaw 로 역회전 → 로봇 기준 상대좌표(mm).
#            rel_x =  cos(yaw)*dx + sin(yaw)*dy
#            rel_y = -sin(yaw)*dx + cos(yaw)*dy
#
# [GC 주소 결정 — HSL return 은 unicast 라 GC IP 가 필요]
#   gc_address 파라미터가 비면("") 3838 브로드캐스트(컨트롤 패킷 "RGme")를
#   보조로 수신해 송신지 IP 를 GC 로 자동 학습한다. 명시하면 그 IP 로 바로 unicast.
#   (보조 소켓은 receiver 노드와 SO_REUSEADDR 로 같은 3838 을 공유 — 브로드캐스트라
#    두 소켓 모두 사본을 받는다.)
#
# [파라미터]
#   team_number     (int,  기본 35)
#   player_number   (int,  기본 1)
#   gc_address      (str,  기본 "")        GC IP. 비면 자동 학습.
#   return_port     (int,  기본 3939)
#   send_rate_hz    (float,기본 2.0)        0.5~2.0 권장(HSL 규격 상한 2Hz)
#   pose_topic      (str,  기본 /localization/vision_pose)
#   ball_topic      (str,  기본 /ball/map_point)
#   robot_state_topic (str,기본 /robot_state)
#   ball_lost_sec   (float,기본 0.8)        이 시간 넘으면 공 분실로 보고 ballAge 계속 증가
#   mirror_field    (bool, 기본 false)      진영 반대일 때 pose 점대칭
#   debug_topic     (str,  기본 /gc_return/debug)  송신 요약 String(검증용)
#
# [검증]
#   1) GameController GUI 에서 이 로봇이 active(컬러)로 뜨는지 확인.
#   2) TCM 으로 pose/ball 이 실제 위치와 맞게 찍히는지(진영 반대면 mirror_field).
#   3) ros2 topic echo /gc_return/debug 로 송신 값 실시간 확인.
# ==========================================================================
import math
import socket
import struct
import threading

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped

# ── return 패킷 상수 ──
GC_RETURN_HEADER = b'RGrt'
GC_RETURN_VERSION = 4          # RoboCup-HumanoidSoccerLeague/GameController
# struct 포맷: 헤더(4s) + uint8 4개(version,player,team,fallen) + float 6개(pose3,ballAge,ball2)
_RETURN_FMT = '<4s4B6f'
_RETURN_SIZE = struct.calcsize(_RETURN_FMT)    # = 32

# ── 컨트롤 패킷(자동학습용) 상수 ──
GC_CONTROL_HEADER = b'RGme'    # 3838 브로드캐스트 헤더 — 이걸로 GC 송신지 학습
GC_CONTROL_PORT = 3838

# ── fallen 매핑: 이 상태들만 "정상(서서 플레이 가능)" ──
_UPRIGHT_STATES = {'NORMAL'}


class GameControllerReturn(Node):
    def __init__(self):
        super().__init__('game_controller_return')

        # ── 파라미터 ──
        self.team_number = int(self.declare_parameter('team_number', 35).value)
        self.player_number = int(self.declare_parameter('player_number', 1).value)
        self.gc_address = str(self.declare_parameter('gc_address', '').value).strip()
        self.return_port = int(self.declare_parameter('return_port', 3939).value)
        self.send_rate_hz = float(self.declare_parameter('send_rate_hz', 2.0).value)
        self.pose_topic = str(
            self.declare_parameter('pose_topic', '/localization/vision_pose').value)
        self.ball_topic = str(
            self.declare_parameter('ball_topic', '/ball/map_point').value)
        self.robot_state_topic = str(
            self.declare_parameter('robot_state_topic', '/robot_state').value)
        self.ball_lost_sec = float(self.declare_parameter('ball_lost_sec', 0.8).value)
        self.mirror_field = bool(self.declare_parameter('mirror_field', False).value)
        self.debug_topic = str(
            self.declare_parameter('debug_topic', '/gc_return/debug').value)

        # ── 최신값 캐시 (콜백이 채우고, 송신 타이머가 읽는다) ──
        # 자세: (x_m, y_m, yaw_rad). None=아직 수신 전.
        self._pose = None
        # 공: (x_m, y_m) map 좌표 + 마지막 수신시각(rclpy.Time). None=아직 수신 전.
        self._ball_map = None
        self._ball_stamp = None
        # fall 상태 문자열. None=아직 수신 전(=정상 가정).
        self._robot_state = None
        # 여러 콜백/타이머가 캐시를 동시에 만지므로 락으로 보호.
        self._lock = threading.Lock()

        # ── 구독자 ──
        self.create_subscription(
            PoseWithCovarianceStamped, self.pose_topic, self._cb_pose, 10)
        self.create_subscription(
            PointStamped, self.ball_topic, self._cb_ball, 10)
        self.create_subscription(
            String, self.robot_state_topic, self._cb_state, 10)

        # ── 디버그 발행자(송신 요약 — ros2 topic echo 로 검증) ──
        self._debug_pub = self.create_publisher(String, self.debug_topic, 10)

        # ── 송신 소켓 (UDP unicast) ──
        self._tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        # ── GC 주소 자동학습 (gc_address 가 비었을 때만 3838 보조 수신) ──
        self._learn_sock = None
        self._learn_thread = None
        if not self.gc_address:
            self._start_gc_learner()
            self.get_logger().info(
                "gc_address 미지정 → 3838 브로드캐스트로 GC IP 자동 학습 대기")
        else:
            self.get_logger().info(f"gc_address 고정 = {self.gc_address}")

        # ── 송신 타이머 ──
        rate = self.send_rate_hz if self.send_rate_hz > 0.0 else 2.0
        self._timer = self.create_timer(1.0 / rate, self._on_send)

        self.get_logger().info(
            f"GameController return 시작 — team={self.team_number} "
            f"player={self.player_number} → :{self.return_port} @ {rate:.1f}Hz "
            f"mirror_field={self.mirror_field}")

    # ----------------------------------------------------------------------
    #  GC 주소 자동학습 (보조 3838 수신 스레드)
    # ----------------------------------------------------------------------
    def _start_gc_learner(self):
        """3838 브로드캐스트('RGme')를 받아 송신지 IP 를 GC 로 학습한다.

        receiver 노드와 같은 3838 을 SO_REUSEADDR 로 공유한다(브로드캐스트라
        두 소켓 모두 사본 수신). 학습이 끝나도 GC IP 변경 추적을 위해 계속 듣는다.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            s.bind(('0.0.0.0', GC_CONTROL_PORT))
        except OSError as e:
            # 바인드 실패(드묾) — 자동학습 불가. gc_address 수동 지정 안내.
            self.get_logger().error(
                f"3838 보조 바인드 실패({e}) → gc_address 파라미터로 GC IP 를 직접 지정하세요")
            s.close()
            return
        s.settimeout(0.5)
        self._learn_sock = s
        self._learn_running = True
        self._learn_thread = threading.Thread(target=self._learn_loop, daemon=True)
        self._learn_thread.start()

    def _learn_loop(self):
        while rclpy.ok() and getattr(self, '_learn_running', False):
            try:
                data, addr = self._learn_sock.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception:   # noqa: BLE001
                continue
            # 'RGme' 헤더인 패킷의 송신지만 GC 로 인정(다른 브로드캐스트 노이즈 배제).
            if len(data) >= 4 and data[0:4] == GC_CONTROL_HEADER:
                if self.gc_address != addr[0]:
                    self.gc_address = addr[0]
                    self.get_logger().info(f"GC IP 학습됨 → {self.gc_address}")

    # ----------------------------------------------------------------------
    #  구독 콜백 (캐시 갱신)
    # ----------------------------------------------------------------------
    def _cb_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self._yaw_from_quat(q.x, q.y, q.z, q.w)
        with self._lock:
            self._pose = (p.x, p.y, yaw)

    def _cb_ball(self, msg: PointStamped):
        with self._lock:
            self._ball_map = (msg.point.x, msg.point.y)
            # 분실 판정/ballAge 기준은 "노드 클럭 기준 수신시각"으로 통일한다
            # (메시지 stamp 가 다른 클럭일 수 있어 now() 로 일관성 확보).
            self._ball_stamp = self.get_clock().now()

    def _cb_state(self, msg: String):
        with self._lock:
            self._robot_state = msg.data

    # ----------------------------------------------------------------------
    #  송신 타이머
    # ----------------------------------------------------------------------
    def _on_send(self):
        # GC 주소를 아직 모르면 송신 스킵(자동학습 대기).
        if not self.gc_address:
            self.get_logger().warn(
                "GC IP 미확보 — 송신 대기(컨트롤 패킷 수신 또는 gc_address 지정 필요)",
                throttle_duration_sec=5.0)
            return

        # ── 캐시 스냅샷 (락 최소 구간) ──
        with self._lock:
            pose = self._pose
            ball_map = self._ball_map
            ball_stamp = self._ball_stamp
            state = self._robot_state

        now = self.get_clock().now()

        # ── pose 계산 (map m → 필드 mm/rad) ──
        if pose is not None:
            rx, ry, ryaw = pose
            if self.mirror_field:
                # 진영(코트) 반대 → 점대칭: (x,y,θ)→(-x,-y,θ+π)
                px_mm = -rx * 1000.0
                py_mm = -ry * 1000.0
                ptheta = self._wrap_pi(ryaw + math.pi)
            else:
                px_mm = rx * 1000.0
                py_mm = ry * 1000.0
                ptheta = self._wrap_pi(ryaw)
        else:
            # 자세 미수신 — 0 으로 보내되(패킷은 유효), 경고.
            rx = ry = ryaw = 0.0
            px_mm = py_mm = ptheta = 0.0
            self.get_logger().warn(
                f"'{self.pose_topic}' 미수신 — pose=0 으로 송신",
                throttle_duration_sec=5.0)

        # ── ball 계산 (map 절대 → 로봇 기준 상대 mm, ballAge) ──
        ball_age = -1.0
        ball_x_mm = 0.0
        ball_y_mm = 0.0
        if ball_map is not None and ball_stamp is not None:
            age = (now - ball_stamp).nanoseconds * 1e-9
            ball_age = float(age)   # 분실이어도 마지막 본 후 경과시간을 그대로 보고
            # 상대좌표는 자세가 있어야 계산 가능. 자세 없으면 ball 0(=미지) 처리.
            if pose is not None:
                dx = ball_map[0] - rx        # map 기준 로봇→공 벡터(m)
                dy = ball_map[1] - ry
                # 로봇 yaw 로 역회전 → 로봇 기준 상대좌표
                cos_y = math.cos(ryaw)
                sin_y = math.sin(ryaw)
                rel_x = cos_y * dx + sin_y * dy
                rel_y = -sin_y * dx + cos_y * dy
                ball_x_mm = rel_x * 1000.0
                ball_y_mm = rel_y * 1000.0

        # ── fallen 계산 ──
        if state is None:
            fallen = 0   # robot_state 미수신 → 정상 가정(fall_manager 가 곧 채움)
        else:
            fallen = 0 if state in _UPRIGHT_STATES else 1

        # ── 패킷 구성 & 송신 ──
        packet = struct.pack(
            _RETURN_FMT,
            GC_RETURN_HEADER,
            GC_RETURN_VERSION,
            self.player_number & 0xFF,
            self.team_number & 0xFF,
            fallen & 0xFF,
            float(px_mm), float(py_mm), float(ptheta),
            float(ball_age),
            float(ball_x_mm), float(ball_y_mm),
        )
        try:
            self._tx_sock.sendto(packet, (self.gc_address, self.return_port))
        except OSError as e:
            self.get_logger().error(
                f"송신 실패({self.gc_address}:{self.return_port}) — {e}",
                throttle_duration_sec=2.0)
            return

        # ── 디버그 요약 발행(검증용) ──
        lost = (ball_age < 0.0) or (ball_age > self.ball_lost_sec)
        ball_txt = 'NONE' if ball_age < 0.0 else (
            f"({ball_x_mm:.0f},{ball_y_mm:.0f})mm age={ball_age:.2f}s"
            f"{' LOST' if lost else ''}")
        dbg = String()
        dbg.data = (
            f"p{self.player_number}@t{self.team_number} "
            f"pose=({px_mm:.0f},{py_mm:.0f})mm,{math.degrees(ptheta):.0f}deg "
            f"fallen={fallen} ball={ball_txt} → {self.gc_address}:{self.return_port}")
        self._debug_pub.publish(dbg)
        self.get_logger().info(dbg.data, throttle_duration_sec=2.0)

    # ----------------------------------------------------------------------
    #  수학 헬퍼
    # ----------------------------------------------------------------------
    @staticmethod
    def _yaw_from_quat(x, y, z, w):
        """쿼터니언 → yaw(rad). tf 의존성 없이 직접 계산."""
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny, cosy)

    @staticmethod
    def _wrap_pi(a):
        """각도를 [-π, π) 로 정규화."""
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    # ----------------------------------------------------------------------
    def destroy_node(self):
        self._learn_running = False
        for s in (self._learn_sock, self._tx_sock):
            try:
                if s is not None:
                    s.close()
            except Exception:   # noqa: BLE001
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GameControllerReturn()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

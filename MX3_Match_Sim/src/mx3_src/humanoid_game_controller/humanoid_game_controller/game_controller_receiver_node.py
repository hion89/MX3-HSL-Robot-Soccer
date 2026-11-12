#!/usr/bin/env python3
# ==========================================================================
# game_controller_receiver_node.py
# --------------------------------------------------------------------------
# RoboCup HSL GameController(UDP 3838, 헤더 "RGme")에서 오는 158바이트
# RoboCupGameControlData 패킷을 *구조체 레이아웃 그대로* 파싱해서
# humanoid_behavior_interfaces/GameState 메시지로 발행한다.
#
# [왜 struct 기반인가]
#   control_ws 의 기존 디코더는 data[9], data[98] 처럼 손으로 센 오프셋을
#   썼는데, 우리가 쓰는 GameController 는 STRUCT_VERSION 20(HSL 2026)이라
#   예전 SPL 레이아웃과 필드가 다르다(competitionType / goalkeeper / cautions /
#   messageBudget 추가, players[20]). 그래서 헤더 파일의 실제 정의대로
#   struct.unpack 하지 않으면 값이 어긋난다. 오프셋을 외워서 박지 말고,
#   레이아웃을 그대로 코드로 옮긴다.
#
# [패킷 레이아웃 — RoboCupGameControlData.h (STRUCT_VERSION 20), little-endian]
#   offset 0  : char     header[4]        "RGme"
#   offset 4  : uint8    version          (= 20)
#   offset 5  : uint8    packetNumber
#   offset 6  : uint8    playersPerTeam
#   offset 7  : uint8    competitionType  0=SMALL 1=MIDDLE 2=LARGE
#   offset 8  : uint8    stopped          1=플레이 정지
#   offset 9  : uint8    gamePhase
#   offset 10 : uint8    state
#   offset 11 : uint8    setPlay
#   offset 12 : uint8    firstHalf
#   offset 13 : uint8    kickingTeam      다음 재개 주도 팀 번호(255=none)
#   offset 14 : int16    secsRemaining
#   offset 16 : int16    secondaryTime
#   offset 18 : TeamInfo teams[2]         (각 70바이트)
#   → 총 18 + 70*2 = 158 바이트
#
#   TeamInfo(70B): uint8 6개 + uint16 2개(=10B) + RobotInfo players[20](20*3=60B)
#     uint8  teamNumber, fieldPlayerColour, goalkeeperColour, goalkeeper,
#            score, penaltyShot
#     uint16 singleShots, messageBudget
#   RobotInfo(3B): uint8 penalty, secsTillUnpenalised, cautions
#
#   ※ 송신부(game_controller_msgs, Rust)가 put_u16_le 사용 → little-endian('<').
#
# [발행]
#   /game_state  (humanoid_behavior_interfaces/GameState)
#
# [파라미터]
#   team_number      (int,   기본 35)     우리 팀 번호. teams[] 중 이 번호를 우리로 본다.
#   listen_port      (int,   기본 3838)
#   bind_address     (str,   기본 "0.0.0.0")
#   log_throttle_sec (float, 기본 2.0)    사람이 보기 위한 요약 로그 주기
#
# [검증]
#   ros2 topic echo /game_state
#   → GameController GUI 에서 state 를 SET/PLAYING 등으로 바꿔보며 값이 맞는지 확인.
# ==========================================================================
import socket
import struct
import threading

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from humanoid_behavior_interfaces.msg import GameState

# ── 패킷 상수 ──
GC_HEADER = b'RGme'          # 데이터 패킷 식별자
GC_STRUCT_VERSION = 20       # HSL 2026 GameController
GC_DATA_SIZE = 158           # RoboCupGameControlData 총 바이트
MAX_NUM_PLAYERS = 20         # 헤더의 MAX_NUM_PLAYERS

# ── struct 포맷 (모두 little-endian '<', C 자연정렬과 동일하게 패딩 없음) ──
# 제어 블록 18B: 헤더(4s) + uint8 10개 + int16 2개
_CONTROL_FMT = '<4s10Bhh'
_CONTROL_SIZE = struct.calcsize(_CONTROL_FMT)          # = 18
# TeamInfo 앞부분 10B: uint8 6개 + uint16 2개
_TEAM_HEAD_FMT = '<6B2H'
_TEAM_HEAD_SIZE = struct.calcsize(_TEAM_HEAD_FMT)      # = 10
# RobotInfo 3B: uint8 3개
_ROBOT_FMT = '<3B'
_ROBOT_SIZE = struct.calcsize(_ROBOT_FMT)              # = 3
_TEAM_SIZE = _TEAM_HEAD_SIZE + MAX_NUM_PLAYERS * _ROBOT_SIZE   # = 70

# 사람이 읽기 위한 이름 (로그 전용)
_STATE_NAMES = {0: 'INITIAL', 1: 'READY', 2: 'SET', 3: 'PLAYING', 4: 'FINISHED'}
_PHASE_NAMES = {0: 'NORMAL', 1: 'PENALTY_SHOOTOUT', 2: 'EXTRA_TIME', 3: 'TIMEOUT'}
_SETPLAY_NAMES = {
    0: 'NONE', 1: 'DIRECT_FK', 2: 'INDIRECT_FK', 3: 'PENALTY_KICK',
    4: 'THROW_IN', 5: 'GOAL_KICK', 6: 'CORNER_KICK',
}


class GameControllerReceiver(Node):
    def __init__(self):
        super().__init__('game_controller_receiver')

        # ── 파라미터 ──
        self.team_number = int(self.declare_parameter('team_number', 35).value)
        self.listen_port = int(self.declare_parameter('listen_port', 3838).value)
        self.bind_address = str(self.declare_parameter('bind_address', '0.0.0.0').value)
        self.log_throttle_sec = float(
            self.declare_parameter('log_throttle_sec', 2.0).value)

        # ── 발행자 ──
        self.pub = self.create_publisher(GameState, '/game_state', 10)

        # ── UDP 소켓 (브로드캐스트 수신) ──
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind((self.bind_address, self.listen_port))
        # 0.5s timeout 으로 두어, 종료 시 스레드가 깔끔히 빠져나오게 한다.
        self.sock.settimeout(0.5)

        # ── 수신 스레드 (recvfrom 은 블로킹이므로 spin 과 분리) ──
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

        self.get_logger().info(
            f"GameController 수신 시작 — {self.bind_address}:{self.listen_port}, "
            f"team_number={self.team_number}")

    # ----------------------------------------------------------------------
    def _listen_loop(self):
        """UDP 패킷을 블로킹 수신하고, 유효하면 파싱·발행한다."""
        while rclpy.ok() and self.running:
            try:
                data, _addr = self.sock.recvfrom(512)
            except socket.timeout:
                continue   # 0.5s 동안 패킷 없으면 루프만 돈다(종료 체크용)
            except OSError:
                break      # 소켓이 닫힘 → 종료
            except Exception as e:   # noqa: BLE001
                if self.running:
                    self.get_logger().error(f"수신 에러: {e}")
                continue

            # ── 유효성 검사 ──
            if len(data) != GC_DATA_SIZE:
                continue
            if data[0:4] != GC_HEADER:
                continue
            if data[4] != GC_STRUCT_VERSION:
                # 버전이 다르면 레이아웃이 달라 오파싱 위험 → 스킵하고 경고
                self.get_logger().warn(
                    f"STRUCT_VERSION 불일치: 수신={data[4]} 기대={GC_STRUCT_VERSION} "
                    f"(GameController 버전 확인 필요)",
                    throttle_duration_sec=self.log_throttle_sec)
                continue

            try:
                msg = self._parse(data)
            except struct.error as e:
                self.get_logger().error(f"파싱 실패: {e}")
                continue

            self.pub.publish(msg)
            self._log_summary(msg)

    # ----------------------------------------------------------------------
    def _parse(self, data: bytes) -> GameState:
        """158바이트 패킷을 GameState 메시지로 변환한다."""
        # ── 제어 블록 ──
        (_hdr, version, packet_number, players_per_team, competition_type,
         stopped, game_phase, state, set_play, first_half, kicking_team,
         secs_remaining, secondary_time) = struct.unpack_from(_CONTROL_FMT, data, 0)

        # ── 양 팀 파싱 ──
        off = _CONTROL_SIZE
        teams = []
        for _ in range(2):
            (team_number, _fpc, _gkc, goalkeeper, score, _penalty_shot,
             _single_shots, _msg_budget) = struct.unpack_from(_TEAM_HEAD_FMT, data, off)
            players = []
            poff = off + _TEAM_HEAD_SIZE
            for p in range(MAX_NUM_PLAYERS):
                penalty, secs_till, cautions = struct.unpack_from(
                    _ROBOT_FMT, data, poff + p * _ROBOT_SIZE)
                players.append((penalty, secs_till, cautions))
            teams.append({
                'team_number': team_number,
                'goalkeeper': goalkeeper,
                'score': score,
                'players': players,
            })
            off += _TEAM_SIZE

        # ── 우리 팀 / 상대 팀 분리 ──
        our = next((t for t in teams if t['team_number'] == self.team_number), None)
        opp = next((t for t in teams if t is not our), None)

        # ── 메시지 채우기 ──
        msg = GameState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''

        msg.version = version
        msg.packet_number = packet_number
        msg.players_per_team = players_per_team
        msg.competition_type = competition_type
        msg.stopped = bool(stopped)
        msg.game_phase = game_phase
        msg.state = state
        msg.set_play = set_play
        msg.first_half = bool(first_half)
        msg.secs_remaining = secs_remaining
        msg.secondary_time = secondary_time
        msg.our_team_number = self.team_number

        if our is not None:
            msg.team_found = True
            # kickingTeam 은 "팀 번호". 우리 번호와 같으면 다음 재개 주도권이 우리.
            msg.we_are_kicking = (kicking_team == self.team_number)
            msg.our_score = our['score']
            msg.opponent_score = opp['score'] if opp is not None else 0
            msg.our_goalkeeper = our['goalkeeper']
            # 패널티 배열은 players_per_team 길이만 잘라서 담는다(나머지는 미사용 슬롯).
            n = players_per_team if 0 < players_per_team <= MAX_NUM_PLAYERS else MAX_NUM_PLAYERS
            msg.penalties = [our['players'][i][0] for i in range(n)]
            msg.secs_till_unpenalised = [our['players'][i][1] for i in range(n)]
            msg.cautions = [our['players'][i][2] for i in range(n)]
        else:
            # 패킷 안에 우리 팀 번호가 없음(팀 번호 오설정 가능성) → 글로벌 정보만 유효.
            msg.team_found = False
            msg.we_are_kicking = False
            msg.our_score = 0
            msg.opponent_score = 0
            msg.our_goalkeeper = 0
            msg.penalties = []
            msg.secs_till_unpenalised = []
            msg.cautions = []
            self.get_logger().warn(
                f"패킷에서 우리 팀(team_number={self.team_number})을 못 찾음 "
                f"— 수신 팀들: {[t['team_number'] for t in teams]}",
                throttle_duration_sec=self.log_throttle_sec)

        return msg

    # ----------------------------------------------------------------------
    def _log_summary(self, msg: GameState):
        """사람이 눈으로 확인하기 위한 한 줄 요약 (throttle)."""
        kick = 'US' if msg.we_are_kicking else 'THEM/None'
        self.get_logger().info(
            f"[GC] state={_STATE_NAMES.get(msg.state, msg.state)} "
            f"phase={_PHASE_NAMES.get(msg.game_phase, msg.game_phase)} "
            f"setplay={_SETPLAY_NAMES.get(msg.set_play, msg.set_play)} "
            f"stopped={msg.stopped} "
            f"score {msg.our_score}-{msg.opponent_score} kick={kick} "
            f"found={msg.team_found}",
            throttle_duration_sec=self.log_throttle_sec)

    # ----------------------------------------------------------------------
    def destroy_node(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:   # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GameControllerReceiver()
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

#!/usr/bin/env python3
# ============================================================================
#  khuber_arm_pwm_test.py
# ----------------------------------------------------------------------------
#  팔 관절의 Goal PWM(100) 상한을 실기에서 찾는 벤치 도구.
#
#  ── 왜 필요한가 ──
#  낙상 시 팔이 잠긴 채 충격을 받으면 152.3:1 감속기가 그 충격을 그대로
#  받고, 모터는 버티다 Overload Error(addr 70, bit 5)를 래치한다.
#  래치는 REBOOT 로만 지워지는데 REBOOT 는 RAM 을 날려 PID 게인과
#  Profile Velocity 가 EEPROM 기본값으로 돌아간다.
#  → "넘어졌다 일어난 뒤부터 거동이 달라지는데 원인을 못 찾는" 상황.
#
#  Goal PWM 을 낮추면 PID 가 계산한 PWM 출력에 상한이 걸려, 일정 힘
#  이상에서 관절이 항복한다(= 토크 리미터). 위치 제어는 그대로 살아
#  있으므로 보호 자세도 유지된다.
#
#  ── 단위 (반드시 확인) ──
#    PWM Limit(36)  : EEPROM. 기본 885. 이게 100% 다.
#    Goal PWM(100)  : RAM.    0 ~ PWM Limit. 1 LSB ≈ 0.113%
#    ★ "PWM 100" 은 100% 가 아니라 100/885 = 11.3% 다. 헷갈리지 말 것.
#
#  ── RAM 이라는 뜻 ──
#    Goal PWM 도 게인과 똑같이 REBOOT/전원 OFF 시 기본값(=PWM Limit)으로
#    돌아간다. 실제 운용에 넣을 때는 게인과 같은 경로에서 재기입해야 한다.
#
#  ── 이 도구가 답해야 하는 세 가지 ──
#    Q1. Goal PWM 이 실제로 위치 제어 모드에서 출력을 제한하는가?
#        → Present PWM(124) 이 설정값에서 포화하는지로 확인.
#    Q2. 팔이 중력을 이기고 보호 자세까지 올라가는 최소 PWM 은 얼마인가?
#        → 이 값 아래로는 못 내려간다(하한).
#    Q3. 사람이 손으로 밀었을 때 항복하는가?
#        → 이 값 위로는 의미가 없다(상한).
#    Q2 < Q3 인 구간이 존재해야 이 접근이 성립한다. 없으면 폐기.
#
#  ── 사용 ──
#    # 반드시 로봇을 매달거나 눕혀서, 팔이 자유롭게 움직일 수 있게 둘 것
#    ros2 run humanoid_utils khuber_arm_pwm_test --id 17
#    ros2 run humanoid_utils khuber_arm_pwm_test --id 17 --port /dev/ttyUSB0
#    ros2 run humanoid_utils khuber_arm_pwm_test --id 17 --sweep 885,600,400,300,200,150,100
#
#  ── 주의 ──
#    · body_motor_controller 를 반드시 먼저 내릴 것. 같은 포트를 두 프로세스가
#      열면 SyncRead 가 깨진다.
#    · 팔 하나만 대상으로 한다. 전신 토크는 건드리지 않는다.
# ============================================================================

import argparse
import glob
import os
import sys
import time

try:
    from dynamixel_sdk import PortHandler, PacketHandler
except ImportError:
    print("dynamixel_sdk 가 없습니다: pip install dynamixel-sdk")
    sys.exit(1)


# 단일 패킷 TxRx 재시도 횟수.
#   DXL SDK 의 read*/write*TxRx 는 SyncRead 와 달리 재시도가 없다. 3Mbps 에서
#   USB 지터나 미세한 버스 간섭 하나에 통째로 실패한다. 벤치 도구는 한 번
#   실패했다고 죽으면 안 되므로 여기서 감싼다.
COMM_RETRY = 3
COMM_RETRY_SLEEP = 0.01


def who_holds_port(dev_path: str) -> list:
    """이 시리얼 장치를 열고 있는 다른 프로세스를 찾는다.

    ★ 이 도구에서 가장 흔한 실패 원인 ★
      리눅스는 시리얼 포트를 기본적으로 배타 잠금하지 않는다. 그래서
      body_motor_controller 가 50Hz 로 SyncRead 를 돌리는 중에 이 도구가
      같은 포트를 열면, 두 프로세스의 패킷이 버스에서 섞여
      "Incorrect status packet"(COMM_RX_CORRUPT) 이 간헐적으로 터진다.

    root 없이도 같은 사용자 소유 프로세스는 /proc 로 찾을 수 있다.
    반환: [(pid, cmdline), ...]
    """
    holders = []
    try:
        real = os.path.realpath(dev_path)
    except OSError:
        return holders
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.path.realpath(os.path.join(fd_dir, fd)) != real:
                        continue
                except OSError:
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmd = f.read().replace(b"\0", b" ").decode(
                            errors="replace").strip()
                except OSError:
                    cmd = "(cmdline 읽기 실패)"
                holders.append((pid, cmd or "(빈 cmdline)"))
                break
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return holders


def check_latency_timer(dev_path: str):
    """FTDI latency_timer 를 확인한다. 1ms 가 아니면 경고 문자열 반환.

    재부팅 후 16ms 로 되돌아가는 문제가 알려져 있다(udev 로 고정 필요).
    16ms 면 왕복이 느려져 타임아웃/손상 확률이 올라간다.
    """
    name = os.path.basename(os.path.realpath(dev_path))
    for path in glob.glob(
            f"/sys/bus/usb-serial/devices/{name}/latency_timer"):
        try:
            with open(path, "r") as f:
                val = int(f.read().strip())
        except (OSError, ValueError):
            return None
        if val != 1:
            return (f"latency_timer = {val}ms  ★ 1ms 여야 합니다\n"
                    f"    echo 1 | sudo tee {path}")
        return None
    return None


# ── 컨트롤 테이블 (XM540 / MX-64(2.0), Protocol 2.0) ──────────────────
ADDR_OPERATING_MODE = 11    # 1B, EEPROM. 3 = Position Control
ADDR_PWM_LIMIT = 36    # 2B, EEPROM. 기본 885 (= 100%)
ADDR_VELOCITY_LIMIT = 44    # 4B, EEPROM. Profile Velocity 상한
ADDR_MAX_POS_LIMIT = 48    # 4B, EEPROM. Goal Position 상한
ADDR_MIN_POS_LIMIT = 52    # 4B, EEPROM. Goal Position 하한
ADDR_TORQUE_ENABLE = 64    # 1B, RAM
ADDR_HW_ERROR = 70    # 1B, RAM (읽기전용)
ADDR_GOAL_PWM = 100   # 2B, RAM. 0 ~ PWM Limit
ADDR_PROFILE_VELOCITY = 112   # 4B, RAM
ADDR_GOAL_POSITION = 116   # 4B, RAM
ADDR_PRESENT_PWM = 124   # 2B, RAM (signed)
ADDR_PRESENT_CURRENT = 126   # 2B, RAM (signed)
ADDR_PRESENT_POSITION = 132   # 4B, RAM (signed)

RAD_PER_TICK = 0.0015339807878856412   # 2π / 4096
CENTER_TICK = 2048
PWM_PCT_PER_LSB = 0.113                 # e-manual: 1 LSB ≈ 0.113 %


def _s16(v: int) -> int:
    """부호 없는 16비트 → 부호 있는 정수."""
    return v - 65536 if v > 32767 else v


def _s32(v: int) -> int:
    """부호 없는 32비트 → 부호 있는 정수."""
    return v - 4294967296 if v > 2147483647 else v


class Dxl:
    """단일 모터 read/write 래퍼.

    모든 접근을 COMM_RETRY 회까지 재시도한다. 3Mbps 단일 패킷 TxRx 는
    지터 하나에도 깨지는데, 벤치 도구가 그 한 번에 죽으면 안 된다.
    재시도 통계를 누적해 버스 건전성 판단에 쓴다.
    """

    def __init__(self, port_name: str, baud: int, dxl_id: int):
        self.id = dxl_id
        self.port_name = port_name
        self.port = PortHandler(port_name)
        self.packet = PacketHandler(2.0)
        # 통신 통계: (시도 횟수, 재시도로 살린 횟수, 최종 실패 횟수)
        self.n_try = 0
        self.n_retry = 0
        self.n_fail = 0
        if not self.port.openPort():
            raise RuntimeError(f"포트 열기 실패: {port_name}")
        if not self.port.setBaudRate(baud):
            raise RuntimeError(f"보레이트 설정 실패: {baud}")

    def close(self):
        self.port.closePort()

    def _call(self, fn, what, *args, quiet=False):
        """읽기/쓰기 공통 재시도 래퍼.

        read*TxRx  → (value, comm, err) 3-튜플
        write*TxRx → (comm, err)        2-튜플
        """
        last = ""
        for attempt in range(COMM_RETRY):
            self.n_try += 1
            res = fn(self.port, self.id, *args)
            if len(res) == 3:
                value, comm, err = res
            else:
                value, (comm, err) = None, res
            if comm == 0 and err == 0:
                if attempt > 0:
                    self.n_retry += 1
                return value
            if comm != 0:
                last = self.packet.getTxRxResult(comm)
            else:
                last = self.packet.getRxPacketError(err)
            time.sleep(COMM_RETRY_SLEEP)
        self.n_fail += 1
        if quiet:
            return None
        raise RuntimeError(f"{what}: {COMM_RETRY}회 재시도 후 실패 — {last}")

    def r1(self, addr, quiet=False):
        return self._call(self.packet.read1ByteTxRx, f"read1({addr})",
                          addr, quiet=quiet)

    def r2(self, addr, quiet=False):
        return self._call(self.packet.read2ByteTxRx, f"read2({addr})",
                          addr, quiet=quiet)

    def r4(self, addr, quiet=False):
        return self._call(self.packet.read4ByteTxRx, f"read4({addr})",
                          addr, quiet=quiet)

    def w1(self, addr, val):
        self._call(self.packet.write1ByteTxRx, f"write1({addr})",
                   addr, int(val))

    def w2(self, addr, val):
        self._call(self.packet.write2ByteTxRx, f"write2({addr})",
                   addr, int(val))

    def w4(self, addr, val):
        self._call(self.packet.write4ByteTxRx, f"write4({addr})",
                   addr, int(val))

    def bus_health(self, n: int = 40) -> float:
        """Present Position 을 n 회 읽어 1회 시도 성공률[%] 을 낸다.

        재시도로 가려지기 전의 '생' 성공률을 봐야 버스 오염을 판단할 수 있다.
        """
        ok = 0
        for _ in range(n):
            _, comm, err = self.packet.read4ByteTxRx(
                self.port, self.id, ADDR_PRESENT_POSITION)
            if comm == 0 and err == 0:
                ok += 1
            time.sleep(0.005)
        return 100.0 * ok / max(1, n)

    # ── 편의 ──
    def pos_tick(self):
        return _s32(self.r4(ADDR_PRESENT_POSITION))

    def pos_rad(self):
        return (self.pos_tick() - CENTER_TICK) * RAD_PER_TICK

    def pwm(self):
        return _s16(self.r2(ADDR_PRESENT_PWM))

    def cur_ma(self):
        # XM540 = 2.69 mA/LSB, MX-64 = 3.36 mA/LSB.
        # 팔(ID 13~18)은 XM540 이므로 2.69 로 표시한다.
        return _s16(self.r2(ADDR_PRESENT_CURRENT)) * 2.69


def sample(dxl: Dxl, dur_s: float, hz: float = 50.0):
    """dur_s 동안 상태를 샘플링해 요약값을 돌려준다.

    개별 샘플이 실패해도 시험 전체를 중단하지 않는다(quiet=True).
    실패가 많으면 마지막 요약의 n 값이 작아지므로 그걸로 판단한다.
    """
    n = max(1, int(dur_s * hz))
    pwms, curs, poss = [], [], []
    for _ in range(n):
        pwm = dxl.r2(ADDR_PRESENT_PWM, quiet=True)
        cur = dxl.r2(ADDR_PRESENT_CURRENT, quiet=True)
        pos = dxl.r4(ADDR_PRESENT_POSITION, quiet=True)
        if pwm is not None and cur is not None and pos is not None:
            pwms.append(abs(_s16(pwm)))
            curs.append(abs(_s16(cur) * 2.69))
            poss.append((_s32(pos) - CENTER_TICK) * RAD_PER_TICK)
        time.sleep(1.0 / hz)
    if not pwms:
        return None
    return {
        "n": len(pwms),
        "n_want": n,
        "pwm_max": max(pwms),
        "pwm_avg": sum(pwms) / len(pwms),
        "cur_max": max(curs),
        "pos_last": poss[-1],
        "pos_min": min(poss),
        "pos_max": max(poss),
    }


def main():
    ap = argparse.ArgumentParser(
        description="팔 관절 Goal PWM 상한 벤치 테스트")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=3000000)
    ap.add_argument("--id", type=int, required=True,
                    help="대상 모터 ID (팔: 13~18)")
    ap.add_argument("--sweep", default="885,600,400,300,200,150,100",
                    help="시험할 Goal PWM 값들 (콤마 구분, 큰 값부터)")
    ap.add_argument("--target-rad", type=float, default=None,
                    help="Q2 시험에서 명령할 목표 각도[rad]. "
                         "생략하면 현재 위치에서 +1.0 rad")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="각 단계에서 기다리는 시간[s]")
    ap.add_argument("--push-sec", type=float, default=4.0,
                    help="[Q3] 손으로 미는 동안의 측정 창 길이[s]")
    ap.add_argument("--force", action="store_true",
                    help="포트 점유/버스 불량이 감지돼도 강행 (결과 신뢰 불가)")
    args = ap.parse_args()

    sweep = []
    for tok in args.sweep.split(","):
        tok = tok.strip()
        if tok:
            sweep.append(int(tok))
    if not sweep:
        print("sweep 값이 비었습니다")
        return 1

    # ── -1) 버스 독점 여부 확인 (포트를 열기 전에) ──────────────
    holders = who_holds_port(args.port)
    if holders:
        print("=" * 68)
        print(f"  ★ {args.port} 를 이미 열고 있는 프로세스가 있습니다")
        print("=" * 68)
        for pid, cmd in holders:
            print(f"    pid {pid}: {cmd[:110]}")
        print()
        print("  리눅스는 시리얼 포트를 배타 잠금하지 않아 두 프로세스가")
        print("  동시에 열 수 있고, 패킷이 버스에서 섞여")
        print("  'Incorrect status packet' 이 간헐적으로 발생합니다.")
        print()
        print("  → dynamixel_system.launch.py (body_motor_controller) 를")
        print("    먼저 내린 뒤 다시 실행하세요.")
        if not args.force:
            print("\n  그래도 강행하려면 --force")
            return 1
        print("\n  --force 지정됨 — 결과를 신뢰하지 마세요.\n")

    lat_warn = check_latency_timer(args.port)

    dxl = Dxl(args.port, args.baud, args.id)
    try:
        # ── 0) 사전 확인 ────────────────────────────────────────────
        mode = dxl.r1(ADDR_OPERATING_MODE)
        pwm_limit = dxl.r2(ADDR_PWM_LIMIT)
        hw_err = dxl.r1(ADDR_HW_ERROR)
        # ── EEPROM 가동 한계 ──
        # ★ 이걸 안 읽고 명령하면 모터가 Range Error 로 거부한다.
        #   "The data value exceeds the limit value!" 가 그 증상이고,
        #   통신 문제가 아니므로 재시도해도 똑같이 거부당한다.
        pos_min_tick = _s32(dxl.r4(ADDR_MIN_POS_LIMIT))
        pos_max_tick = _s32(dxl.r4(ADDR_MAX_POS_LIMIT))
        vel_limit = _s32(dxl.r4(ADDR_VELOCITY_LIMIT))
        pos_min_rad = (pos_min_tick - CENTER_TICK) * RAD_PER_TICK
        pos_max_rad = (pos_max_tick - CENTER_TICK) * RAD_PER_TICK

        print("=" * 68)
        print(f"  ID {args.id} @ {args.port} {args.baud}bps")
        print(f"  Operating Mode(11) : {mode}  "
              f"{'(Position Control OK)' if mode == 3 else '★ 3 이 아님!'}")
        print(f"  PWM Limit(36)      : {pwm_limit}  (= 100%)")
        print(f"  Position Limit     : {pos_min_tick} ~ {pos_max_tick} tick "
              f"= {pos_min_rad:+.3f} ~ {pos_max_rad:+.3f} rad")
        print(f"  Velocity Limit(44) : {vel_limit} "
              f"(= {vel_limit * 0.229:.1f} rpm)")
        print(f"  HW Error(70)       : 0x{hw_err:02X}"
              f"{'  ★ 에러 래치됨 — REBOOT 필요' if hw_err else ''}")
        if lat_warn:
            print(f"  {lat_warn}")
        print("=" * 68)

        if mode != 3:
            print("위치 제어 모드가 아니면 이 시험은 의미가 없습니다. 중단합니다.")
            return 1
        if hw_err:
            print("하드웨어 에러가 래치된 모터입니다. 먼저 리셋하세요. 중단합니다.")
            return 1

        # ── 0b) 버스 건전성 측정 ────────────────────────────────────
        # 재시도로 가려지기 전의 '생' 성공률을 본다. 여기가 100% 가 아니면
        # 이후 측정값(Present PWM/전류)도 신뢰할 수 없다.
        print("\n버스 건전성 측정 중 (40회 단발 읽기)...")
        health = dxl.bus_health(40)
        print(f"  단발 read 성공률: {health:.0f}%")
        if health < 100.0:
            print("  ★ 100% 가 아닙니다. 아래를 순서대로 확인하세요:")
            print("    1) body_motor_controller 등 다른 프로세스가 포트를 쓰는가")
            print("       ros2 node list | grep body_motor_controller")
            print("    2) FTDI latency_timer 가 1ms 인가 (재부팅 후 16ms 로 복귀)")
            print("    3) U2D2 USB 케이블 / 버스 종단")
            if health < 90.0 and not args.force:
                print("\n  성공률이 너무 낮아 측정이 무의미합니다. 중단합니다.")
                print("  (강행하려면 --force)")
                return 1
        else:
            print("  정상 — 측정 진행")

        over = [v for v in sweep if v > pwm_limit]
        if over:
            print(f"★ PWM Limit({pwm_limit}) 을 넘는 값 {over} 은 모터가 거부합니다. "
                  f"제외하고 진행합니다.")
            sweep = [v for v in sweep if v <= pwm_limit]

        input("\n팔이 자유롭게 움직일 수 있는 상태인지 확인 후 Enter: ")

        # 천천히 움직이게 해서 시험 중 팔이 튀지 않게 한다.
        # Profile Velocity 도 Velocity Limit(44) 을 넘으면 거부되므로 clamp.
        prof_vel = min(30, max(1, vel_limit))   # 30 tick ≈ 6.9 rpm
        dxl.w1(ADDR_TORQUE_ENABLE, 0)
        dxl.w4(ADDR_PROFILE_VELOCITY, prof_vel)
        dxl.w1(ADDR_TORQUE_ENABLE, 1)

        start_rad = dxl.pos_rad()

        # ── 목표 위치 결정 ─────────────────────────────────────────
        # 반드시 EEPROM Position Limit 안이어야 한다. 밖이면 모터가
        # Range Error 로 거부하고, 재시도해도 결과가 같다.
        #
        # 자동 선택 규칙(--target-rad 미지정 시):
        #   현재 위치에서 +1.0 rad 쪽을 먼저 보고, 한계에 막혀 이동 여유가
        #   0.3 rad 미만이면 반대(-1.0 rad) 쪽을 쓴다. 양쪽 다 좁으면 중단.
        margin = 0.02   # 한계에 딱 붙이지 않는다(반올림 오차 여유)
        lo = pos_min_rad + margin
        hi = pos_max_rad - margin

        if args.target_rad is not None:
            target = float(args.target_rad)
            if not (lo <= target <= hi):
                print("\n" + "=" * 68)
                print(f"  ★ 지정한 목표 {target:+.3f} rad 는 이 모터의 가동 한계 밖입니다")
                print("=" * 68)
                print(f"    허용 범위 : {pos_min_rad:+.3f} ~ {pos_max_rad:+.3f} rad")
                print(f"    현재 위치 : {start_rad:+.3f} rad")
                print()
                print("    ID 와 부호를 확인하세요. servo_map.yaml 기준:")
                print("      ID 17 = l_elbow_pitch_joint  → 초기자세 -3.04 rad")
                print("      ID 18 = r_elbow_pitch_joint  → 초기자세 +3.04 rad")
                print("      ID 13 = l_shoulder_pitch     → -0.2")
                print("      ID 14 = r_shoulder_pitch     → +0.2")
                print("\n    --target-rad 를 생략하면 한계 안에서 자동 선택합니다.")
                return 1
        else:
            cand_up = min(start_rad + 1.0, hi)
            cand_dn = max(start_rad - 1.0, lo)
            if abs(cand_up - start_rad) >= 0.3:
                target = cand_up
            elif abs(cand_dn - start_rad) >= 0.3:
                target = cand_dn
            else:
                print("\n★ 현재 위치가 양쪽 한계에 가까워 이동 여유가 없습니다.")
                print(f"  현재 {start_rad:+.3f}, 허용 "
                      f"{pos_min_rad:+.3f} ~ {pos_max_rad:+.3f} rad")
                print("  팔을 가동범위 가운데로 옮긴 뒤 다시 실행하세요.")
                return 1

        target_tick = int(round(target / RAD_PER_TICK)) + CENTER_TICK
        # tick 단계에서도 한 번 더 막는다(반올림으로 1 tick 넘어가는 경우).
        target_tick = max(pos_min_tick, min(pos_max_tick, target_tick))
        travel = abs(target - start_rad)
        print(f"\n시작 위치 {start_rad:+.3f} rad → 목표 {target:+.3f} rad "
              f"(이동량 {travel:.3f} rad)")
        print("  ★ 이 방향이 '중력을 이겨야 하는' 방향인지 눈으로 확인하세요.")
        print("    내려가는 방향이면 PWM 을 낮춰도 그냥 떨어져서 하한 판정이 무의미합니다.")
        if travel < 0.3:
            print("  ★ 이동량이 작아 PID 출력이 작습니다. Q1 포화가 안 잡힐 수 있습니다.")

        # 원위치 복귀에 쓸 tick 도 미리 한계 안으로 고정해 둔다.
        start_tick = max(pos_min_tick, min(
            pos_max_tick, int(round(start_rad / RAD_PER_TICK)) + CENTER_TICK))

        results = []

        # ── Q1 + Q2) PWM 별로 목표 추종이 되는지 ────────────────────
        print("\n" + "=" * 68)
        print("  [Q1/Q2] Goal PWM 별 목표 추종 + Present PWM 포화 확인")
        print("=" * 68)
        print(f"{'GoalPWM':>8} {'%':>6} {'PWM최대':>8} {'전류mA':>8} "
              f"{'도달오차rad':>12} {'판정':>8}")
        print("-" * 68)

        for gp in sweep:
            # 일단 원위치로 되돌린다 (충분한 힘으로)
            dxl.w2(ADDR_GOAL_PWM, pwm_limit)
            dxl.w4(ADDR_GOAL_POSITION, start_tick)
            time.sleep(args.settle)

            # 시험 PWM 으로 낮추고 목표를 명령
            dxl.w2(ADDR_GOAL_PWM, gp)
            dxl.w4(ADDR_GOAL_POSITION, target_tick)
            s = sample(dxl, args.settle)
            if s is None:
                print(f"{gp:>8} {'--':>6} {'샘플 실패':>8}")
                continue

            err = abs(target - s["pos_last"])
            reached = err < 0.10          # 0.1 rad ≈ 5.7°
            saturated = s["pwm_max"] >= gp * 0.95
            # 샘플이 많이 유실됐으면 그 행의 수치는 믿을 수 없다.
            lossy = s["n"] < s["n_want"] * 0.8
            results.append({
                "goal_pwm": gp, "reached": reached,
                "saturated": saturated, "err": err, "s": s,
            })
            mark = "도달" if reached else "미달"
            if lossy:
                mark += "?"
            print(f"{gp:>8} {gp * PWM_PCT_PER_LSB:>5.1f}% "
                  f"{s['pwm_max']:>8} {s['cur_max']:>8.0f} "
                  f"{err:>12.3f} {mark:>8}"
                  + (f"   ← 샘플 {s['n']}/{s['n_want']} 유실" if lossy else ""))

        dxl.w2(ADDR_GOAL_PWM, pwm_limit)
        dxl.w4(ADDR_GOAL_POSITION, start_tick)
        time.sleep(args.settle)

        # ── Q1 판정 ────────────────────────────────────────────────
        print("\n" + "=" * 68)
        print("  [Q1] Goal PWM 이 실제로 출력을 제한하는가")
        print("=" * 68)
        sat = [r for r in results if r["saturated"]]
        if sat:
            print(f"  Present PWM 이 설정값 근처에서 포화한 단계: "
                  f"{[r['goal_pwm'] for r in sat]}")
            print("  → Goal PWM 이 위치 제어 모드에서 상한으로 동작함. 접근 성립.")
        else:
            print("  ★ 어느 단계에서도 포화가 관측되지 않았습니다.")
            print("    목표까지 거리가 짧아 PID 출력 자체가 작았을 수 있습니다.")
            print("    --target-rad 로 더 먼(중력을 크게 이겨야 하는) 목표를 주고")
            print("    다시 시험하세요. 그래도 포화가 없으면 이 접근은 폐기합니다.")

        # ── Q2 판정: 중력을 이기는 최소 PWM ─────────────────────────
        print("\n" + "=" * 68)
        print("  [Q2] 보호 자세까지 올라가는 최소 Goal PWM (하한)")
        print("=" * 68)
        ok = [r["goal_pwm"] for r in results if r["reached"]]
        if ok:
            floor = min(ok)
            print(f"  목표 도달에 성공한 최저값: {floor} "
                  f"({floor * PWM_PCT_PER_LSB:.1f}%)")
            print(f"  → 운용값은 이보다 여유 있게. 권장 시작점 {int(floor * 1.5)}")
        else:
            floor = None
            print("  ★ 어느 값에서도 목표에 도달하지 못했습니다.")
            print("    sweep 에 더 큰 값을 넣거나, 목표가 기구 가동범위 밖인지 확인하세요.")

        # ── Q2b) 자세를 '유지'만 하는 데 필요한 최소 PWM ────────────
        # ★ 이게 실제 운용값이다.
        #   Q2 의 하한은 "정지 상태에서 1.0 rad 를 새로 이동하는" 데 드는 값이라
        #   가속 + 마찰 극복분이 다 들어가 있다. 그런데 보행/대기/낙상 중에
        #   팔이 하는 일은 이미 도달한 자세를 '버티는' 것뿐이다.
        #   이동은 robot_init / standup 때만 필요하고 그때는 885 를 쓸 것이므로,
        #   운용값은 이 유지 하한 기준으로 잡아야 한다. 훨씬 낮게 나온다.
        print("\n" + "=" * 68)
        print("  [Q2b] 자세 '유지'에 필요한 최소 Goal PWM ← 실제 운용값")
        print("=" * 68)
        print("  885 로 목표에 도달시킨 뒤, PWM 을 단계적으로 낮추며")
        print("  자세가 흘러내리는 지점을 찾습니다.")
        print(f"{'GoalPWM':>8} {'%':>6} {'PWM최대':>8} {'전류mA':>8} "
              f"{'처짐rad':>10} {'판정':>8}")
        print("-" * 68)

        dxl.w2(ADDR_GOAL_PWM, pwm_limit)
        dxl.w4(ADDR_GOAL_POSITION, target_tick)
        time.sleep(args.settle * 2)
        hold_ref = dxl.pos_rad()
        hold_floor = None

        for gp in sorted(set(sweep + [80, 60, 40, 30, 20, 10]), reverse=True):
            if gp > pwm_limit:
                continue
            dxl.w2(ADDR_GOAL_PWM, gp)
            s = sample(dxl, args.settle)
            if s is None:
                continue
            sag = abs(s["pos_last"] - hold_ref)
            held = sag < 0.05          # 0.05 rad ≈ 2.9°
            print(f"{gp:>8} {gp * PWM_PCT_PER_LSB:>5.1f}% "
                  f"{s['pwm_max']:>8} {s['cur_max']:>8.0f} "
                  f"{sag:>10.3f} {'유지' if held else '처짐':>8}")
            if held:
                hold_floor = gp
            else:
                # 한 번 흘러내리기 시작하면 더 낮춰볼 필요가 없다.
                break

        dxl.w2(ADDR_GOAL_PWM, pwm_limit)
        time.sleep(0.5)

        if hold_floor is not None:
            print(f"\n  유지 하한: {hold_floor} "
                  f"({hold_floor * PWM_PCT_PER_LSB:.1f}%)")
            print(f"  → 운용 권장 시작점 {int(hold_floor * 2)} "
                  f"(보행 진동/외란 여유 2배)")
        else:
            print("\n  ★ 어느 값에서도 자세를 유지하지 못했습니다.")

        # ── Q3) 손으로 밀어 항복하는지 (대화형) ─────────────────────
        print("\n" + "=" * 68)
        print("  [Q3] 손으로 밀었을 때 항복하는가 (상한)")
        print("=" * 68)
        print(f"  각 단계에서 Enter 를 누르면 {args.push_sec:.0f}초 동안 측정합니다.")
        print("  ★ Enter 를 누른 '뒤에' 그 시간 동안 계속 미세요.")
        print("    (누르기 전에 밀면 측정 창 밖이라 아무것도 안 잡힙니다)")
        print()
        print("  읽는 법:")
        print("   · 변위가 크고 PWM 이 상한에 붙음  → 항복함 (좋음)")
        print("   · 변위가 작고 PWM 이 상한에 붙음  → 버팀. 이 값은 너무 높음")
        print("   · PWM 이 상한 근처도 안 감        → 덜 민 것. 더 세게 다시")
        cand = [r["goal_pwm"] for r in results if r["reached"]] or sweep
        push_rows = []
        for gp in sorted(set(cand), reverse=True):
            dxl.w2(ADDR_GOAL_PWM, gp)
            time.sleep(0.3)
            base = dxl.pos_rad()
            input(f"\n  Goal PWM={gp} ({gp * PWM_PCT_PER_LSB:.1f}%) 준비됨. "
                  f"Enter 누르고 {args.push_sec:.0f}초간 미세요: ")
            s = sample(dxl, args.push_sec)
            if not s:
                print("    샘플 실패")
                continue
            # 미는 동안 기준 위치에서 얼마나 벗어났는가(항복량)
            dev = max(abs(s["pos_max"] - base), abs(s["pos_min"] - base))
            at_cap = s["pwm_max"] >= gp * 0.9
            if not at_cap:
                verdict = "덜 밈 — 다시"
            elif dev >= 0.05:
                verdict = "항복"
            else:
                verdict = "버팀"
            push_rows.append({"gp": gp, "dev": dev, "at_cap": at_cap,
                              "verdict": verdict, "s": s})
            print(f"    항복량 {dev:.3f} rad | PWM 최대 {s['pwm_max']}/{gp}"
                  f"{' (상한 도달)' if at_cap else ' (상한 미달)'}"
                  f" | 전류 최대 {s['cur_max']:.0f} mA → {verdict}")
            e = dxl.r1(ADDR_HW_ERROR)
            if e:
                print(f"    ★ HW Error 0x{e:02X} 발생 — 이 값에서는 래치가 납니다")

        yielded = [r["gp"] for r in push_rows if r["verdict"] == "항복"]
        push_ceiling = max(yielded) if yielded else None

        print("\n" + "=" * 68)
        print("  요약")
        print("=" * 68)
        if floor:
            print(f"  이동 하한 (Q2)   : {floor}"
                  f"  ← robot_init / standup 에서 필요한 값")
        if hold_floor:
            print(f"  유지 하한 (Q2b)  : {hold_floor}"
                  f"  ← 보행/대기/낙상 중 운용값의 기준")
        if push_ceiling:
            print(f"  항복 상한 (Q3)   : {push_ceiling}"
                  f"  ← 이 값 이하에서 손힘에 항복함")
        else:
            print("  항복 상한 (Q3)   : 관측 안 됨"
                  " (시험한 전 구간에서 버팀 또는 덜 밈)")
        print()
        if hold_floor and push_ceiling and hold_floor * 2 <= push_ceiling:
            lo_v, hi_v = hold_floor * 2, push_ceiling
            print(f"  → 운용 구간이 존재합니다: {lo_v} ~ {hi_v}")
            print(f"    motor_gains.yaml 의 arms.goal_pwm 에 "
                  f"{int((lo_v + hi_v) / 2)} 부터 시도")
        elif hold_floor and push_ceiling:
            print("  ★ 유지 하한(×2) 이 항복 상한보다 큽니다 — 구간이 없습니다.")
            print("    PWM 상한만으로는 '자세 유지'와 '충격 항복'을 동시에")
            print("    만족시킬 수 없다는 뜻입니다. 접근을 재검토하세요.")
        print("  → robot_init / standup 진입 시에는 885 로 올려야 합니다.")
        print("=" * 68)

    finally:
        # 무슨 일이 있어도 PWM 상한은 원복하고 토크는 끈다.
        try:
            dxl.w2(ADDR_GOAL_PWM, 885)
            dxl.w1(ADDR_TORQUE_ENABLE, 0)
            print("\n정리 완료: Goal PWM 885 복원, 토크 OFF")
        except RuntimeError as e:
            print(f"\n★ 정리 실패: {e}")
            print("  Goal PWM 이 낮은 채로 남았을 수 있습니다. RAM 이므로")
            print("  전원 재투입 또는 모터 REBOOT 로 885 로 돌아갑니다.")
        print(f"통신 통계: 시도 {dxl.n_try}회, 재시도로 복구 {dxl.n_retry}회, "
              f"최종 실패 {dxl.n_fail}회")
        dxl.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

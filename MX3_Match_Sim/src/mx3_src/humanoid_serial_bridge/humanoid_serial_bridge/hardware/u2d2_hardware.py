#!/usr/bin/env python3
"""
u2d2_hardware.py
================

Dynamixel SDK (U2D2 / Protocol 2.0) low-level wrapper.

이 클래스는 **순수 하드웨어 IO 레이어**다. ROS 의존성이 없고, 로깅은
주입된 logger 콜백으로 처리한다. 그래서 unit test 시 MockHardware 로
바꿔치우거나, 다른 노드/툴에서 재사용하기 좋다.

책임 범위:
  - 시리얼 포트 / DXL SDK 핸들 생성/소멸
  - servo_map.yaml 파싱 (joint_name ↔ dxl_id 매핑, 방향, 변환계수)
  - SyncRead/SyncWrite 그룹 셋업
  - Torque / Goal Position 쓰기
  - Profile Velocity 쓰기 (전체 일괄 / 관절별 개별 두 가지)
  - Present Velocity + Position 읽기 + 단위 변환
  - Hardware Error Status 읽기

비책임 (Node 가 담당):
  - ROS 서비스/토픽/콜백
  - rate 통계, diagnostics
  - 상위 동작 시퀀스 (standup, kick 등)
"""
import math
import time
import yaml
from dataclasses import dataclass, field
from typing import Callable, Optional

from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupSyncRead,
    GroupSyncWrite,
)

from .motor_gains import (
    ADDR_PID_BLOCK,
    LEN_PID_BLOCK,
    ADDR_GOAL_PWM,
    LEN_GOAL_PWM,
    GOAL_PWM_KEY,
    GainProfile,
    pid_block_bytes,
)

# PWM Limit(36, 2B, EEPROM). Goal PWM 의 절대 상한이자 100% 기준값.
# 공장 기본 885. 이 값을 넘는 Goal PWM 은 Range Error 로 거부된다.
ADDR_PWM_LIMIT = 36


# ────────────────────────────────────────────────────────────────────────
# Bit helpers
# ────────────────────────────────────────────────────────────────────────
def _clip_int(v: int, vmin: int, vmax: int) -> int:
    return max(vmin, min(vmax, v))


def _int32_from_u32(u: int) -> int:
    """Convert unsigned 32-bit to signed int32."""
    u &= 0xFFFFFFFF
    return u - 0x100000000 if (u & 0x80000000) else u


def _u16_le_bytes(u: int) -> bytes:
    """부호 없는 16비트 → 리틀엔디언 2바이트 (Goal PWM 등 2B 레지스터용)."""
    u = int(u) & 0xFFFF
    return bytes([u & 0xFF, (u >> 8) & 0xFF])


def _u32_to_le_bytes(u: int) -> list[int]:
    u &= 0xFFFFFFFF
    return [u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF, (u >> 24) & 0xFF]


# ────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ────────────────────────────────────────────────────────────────────────
@dataclass
class ControlTable:
    """Protocol 2.0 / MX-64(2.0) / X-series 기본값. 모델 다르면 노드에서 override."""
    torque_enable:    int = 64
    goal_position:    int = 116
    present_velocity: int = 128
    present_position: int = 132
    profile_velocity: int = 112
    hardware_error:   int = 70

    # EEPROM 영역의 Position Limit (MX-64AR / XM540 동일).
    # Min/Max 가 raw tick 기준 (0~4095) 으로 저장됨.
    # GUI 슬라이더 범위 자동 설정 등에 사용.
    max_position_limit: int = 48
    min_position_limit: int = 52

    len_goal_position:    int = 4
    len_present_velocity: int = 4
    len_present_position: int = 4
    len_profile_velocity: int = 4
    len_position_limit:   int = 4

    bulk_read_start_addr: int = 128   # vel(128) ~ pos(132) 까지 8바이트 한 번에
    bulk_read_length:     int = 8


@dataclass
class U2D2Config:
    device_name: str = "/dev/ttyUSB0"
    baudrate: int = 1000000
    map_file: str = ""                # required
    control_table: ControlTable = field(default_factory=ControlTable)
    rpm_per_tick_profile: float = 0.229   # profile velocity tick → rpm
    # 모터 Velocity Limit(EEPROM addr 44)에 대응하는 rpm 상한.
    # Profile Velocity(addr 112) 쓰기가 이 값을 넘으면 펌웨어가 "조용히 거부"한다
    # (SyncWrite 라 응답 패킷이 없어 실패를 알 수도 없음). 거부되면 레지스터는
    # 직전 값으로 남기 때문에, 예컨대 standup _finish() 에서 200 을 쓰려다 씹히면
    # 마지막 phase 속도(20~30rpm)가 최종 속도로 굳어 이후 보행이 불가능해진다.
    # 따라서 set_profile_velocity_all() 에서 이 값으로 상한 clamp 하여
    # "거부" 대신 "최대치(=보행 가능 속도)"로 들어가게 만든다.
    # 실제 모터에 설정한 Velocity Limit 과 동일하게 맞출 것(현재 하드웨어: 45.2rpm).
    # 45.2 미만이면 씹히지 않으므로 0.2rpm 여유를 두고 45.0 으로 둔다.
    # launch 의 robot_final_rpm(45.0) 과 일관되게 기본 45.0 으로 둔다.
    profile_velocity_limit_rpm: float = 45.0


@dataclass
class JointReading:
    """단일 모터 read 결과."""
    name: str
    rad: float
    radps: float


# ────────────────────────────────────────────────────────────────────────
# Main hardware class
# ────────────────────────────────────────────────────────────────────────
class U2D2Hardware:
    """
    DXL SDK 래퍼. 사용 패턴:

        hw = U2D2Hardware(cfg, log_info=node.get_logger().info,
                                log_warn=node.get_logger().warn)
        hw.open()
        hw.set_torque_all(True)
        ...
        hw.close()
    """

    def __init__(self,
                 cfg: U2D2Config,
                 log_info: Callable[[str], None] = print,
                 log_warn: Callable[[str], None] = print):
        self.cfg = cfg
        self._log_info = log_info
        self._log_warn = log_warn

        # YAML 매핑/한계 로드
        if not cfg.map_file:
            raise RuntimeError("map_file is empty. servo_map.yaml 경로를 지정하세요.")

        with open(cfg.map_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.jid: dict = data["joint_map"]                         # {joint_name: dxl_id}
        self.rev_jid: dict = {int(v): k for k, v in self.jid.items()}

        limit = data.get("limits", {})
        self.rad_per_tick = float(limit.get("rad_per_tick", 0.005123))
        self.center_tick  = int(limit.get("center_tick", 2048))
        self.rpm_per_tick = float(limit.get("rpm_per_tick", 0.114))
        self.dir_control  = limit.get("direction_control", {})
        self.dir_display  = limit.get("direction_display", {})
        self.pos_min      = int(limit.get("position_min", 0))
        self.pos_max      = int(limit.get("position_max", 4095))

        self.used_ids = sorted(set(int(v) for v in self.jid.values()))
        self.num_motors = len(self.used_ids)

        # ── Profile Velocity 일괄 쓰기 제외 목록 ──
        # set_profile_velocity_all() 이 건드리면 안 되는 관절(현재는 목).
        # 목은 보행 속도(45rpm ≈ 270°/s)로 돌면 카메라가 흔들려 인식이
        # 무너지므로, 일괄 쓰기에서 빼고 set_profile_velocity_joints() 로만
        # 개별 지정한다. servo_map.yaml 의 limits.profile_velocity_exclude 참조.
        # (없으면 빈 리스트 → 기존 동작과 100% 동일)
        self.pv_exclude_names = [str(n) for n in
                                 (limit.get("profile_velocity_exclude") or [])]
        self.pv_exclude_ids = set(
            int(self.jid[n]) for n in self.pv_exclude_names if n in self.jid
        )

        # DXL handles (open() 에서 실제 생성)
        self.port: Optional[PortHandler] = None
        self.packet: Optional[PacketHandler] = None
        self.group_read: Optional[GroupSyncRead] = None
        self.group_write: Optional[GroupSyncWrite] = None
        self.group_write_profile: Optional[GroupSyncWrite] = None

        # PWM Limit(36) 캐시. open() 에서 read_pwm_limits() 로 채운다.
        self.pwm_limit: dict = {}

        # read 통계 (단순 카운터; rate 통계는 노드 책임)
        self.read_total = 0
        self.read_fail  = 0
        self.id_ok   = {i: 0 for i in self.used_ids}
        self.id_miss = {i: 0 for i in self.used_ids}

    # ───────────────────────────────────────────
    # Lifecycle
    # ───────────────────────────────────────────
    def open(self):
        """포트 열고 DXL SDK 객체 생성. 실패 시 RuntimeError."""
        self.port = PortHandler(self.cfg.device_name)
        self.packet = PacketHandler(2.0)

        if not self.port.openPort():
            raise RuntimeError(f"포트 오픈 실패: {self.cfg.device_name}")
        if not self.port.setBaudRate(self.cfg.baudrate):
            raise RuntimeError(f"보레이트 설정 실패: {self.cfg.baudrate}")

        ct = self.cfg.control_table

        # SyncRead (vel + pos 한 번에)
        self.group_read = GroupSyncRead(
            self.port, self.packet,
            ct.bulk_read_start_addr, ct.bulk_read_length,
        )
        for dxl_id in self.used_ids:
            if not self.group_read.addParam(dxl_id):
                self._log_warn(f"GroupSyncRead addParam failed: id={dxl_id}")

        # SyncWrite (goal position)
        self.group_write = GroupSyncWrite(
            self.port, self.packet,
            ct.goal_position, ct.len_goal_position,
        )

        # SyncWrite (profile velocity)
        self.group_write_profile = GroupSyncWrite(
            self.port, self.packet,
            ct.profile_velocity, ct.len_profile_velocity,
        )

        # Goal PWM 쓰기를 clamp 하려면 각 모터의 PWM Limit 을 알아야 한다.
        # (기동 시 1회. N회 단발 read 지만 이후 SyncWrite 거부를 원천 차단)
        self.read_pwm_limits()

        self._log_info(f"U2D2 open OK: {self.cfg.device_name}, baud={self.cfg.baudrate}")
        self._log_info(f"Used IDs: {self.used_ids}")
        if self.pv_exclude_ids:
            # 제외 목록이 실수로 비면 목이 보행 속도로 튀므로 기동 로그에 남긴다.
            self._log_info(
                f"ProfileVelocity 일괄 쓰기 제외: {self.pv_exclude_names} "
                f"(ids={sorted(self.pv_exclude_ids)})")

    def close(self):
        if self.port is not None:
            try:
                self.port.closePort()
            except Exception:
                pass
            self.port = None

    # ───────────────────────────────────────────
    # Mapping helpers
    # ───────────────────────────────────────────
    def has_joint(self, joint_name: str) -> bool:
        return joint_name in self.jid

    def id_to_joint_name(self, dxl_id: int):
        return self.rev_jid.get(int(dxl_id), None)

    # ───────────────────────────────────────────
    # Torque
    # ───────────────────────────────────────────
    def set_torque_all(self, enable: bool):
        val = 1 if enable else 0
        addr = self.cfg.control_table.torque_enable
        for dxl_id in self.used_ids:
            comm, err = self.packet.write1ByteTxRx(self.port, dxl_id, addr, val)
            if comm != 0:
                self._log_warn(f"Torque write comm fail id={dxl_id}: {self.packet.getTxRxResult(comm)}")
            elif err != 0:
                self._log_warn(f"Torque write error id={dxl_id}: {self.packet.getRxPacketError(err)}")
        self._log_info(f"Torque {'EN' if enable else 'DIS'} for all motors")

    # ───────────────────────────────────────────
    # Goal Position (SyncWrite)
    # ───────────────────────────────────────────
    def send_goals(self, goal_positions: dict):
        """
        goal_positions: {joint_name: rad}
        rad → tick 변환(중심 기준, direction_control 반영) 후 SyncWrite.
        알 수 없는 joint_name 은 무시.
        """
        self.group_write.clearParam()

        any_added = False
        for name, rad in goal_positions.items():
            if name not in self.jid:
                continue
            dxl_id = int(self.jid[name])
            ticks = int(self.center_tick
                        + self.dir_control.get(name, 1) * (rad / self.rad_per_tick))
            ticks = _clip_int(ticks, self.pos_min, self.pos_max)

            param = _u32_to_le_bytes(ticks)
            if not self.group_write.addParam(dxl_id, bytes(param)):
                self._log_warn(f"GroupSyncWrite addParam failed: {name} (id={dxl_id})")
                continue
            any_added = True

        if not any_added:
            return

        comm = self.group_write.txPacket()
        if comm != 0:
            self._log_warn(f"SyncWrite failed: {self.packet.getTxRxResult(comm)}")

    # ──────────────────────────────────────────────────────────────────
    def tx_gap(self, gap_us: float) -> None:
        """SyncWrite 바이트가 선로로 빠져나갈 시간만큼 대기한다.

        ────────────────────────────────────────────────────────────────
        [2026-08] SyncWrite → SyncRead 반이중 경합 제거
        ────────────────────────────────────────────────────────────────
        ■ 왜 필요한가

          U2D2 는 반이중(half-duplex) 이다. send_goals() 의 txPacket() 은
          FTDI 드라이버에 바이트를 넘기고 '즉시' 반환한다. 즉 함수가 돌아온
          시점에도 바이트는 아직 선로 위에 있다. 그 상태로 곧바로
          read_state() 가 SyncRead 를 밀어넣으면 충돌한다.

        ■ 실측 (u2d2_txrx_bench.py, 20모터 @3Mbps, 토크 ON/OFF 무관)

              SyncRead 만                       txrx 2.94 ms
              SyncWrite → 즉시 SyncRead         txrx 9.45 ms   (+6.5)
              SyncWrite → 200us 대기 → SyncRead txrx 3.03 ms   ← 회복
              SyncWrite → 400us 대기 → SyncRead txrx 2.86 ms   ← 최선

          경합분이 모터 수에 비례하지 않고 '대기 한 번' 으로 사라진다.
          → 원인은 모터별 처리 지연이 아니라 버스 경합이다.

        ■ ★ tcdrain 을 쓰지 않는 이유 (초판의 실패)

          초판은 ser.flush()(termios tcdrain)로 구현했다. 벤치에서 txrx 가
          3.13ms 로 회복돼 성공처럼 보였지만, 실기 투입 후 이렇게 나왔다.

              적용 전 : write 0.86 + txrx 10.90 = CycleTime 12.00 ms
              적용 후 : write 9.40 + txrx  3.25 = CycleTime 12.93 ms

          txrx 는 예측대로 −7.7ms 줄었는데 그만큼 write 로 옮겨갔을 뿐이고
          총합은 오히려 나빠졌다.
            원인: ftdi_sio 의 tcdrain 은 USB 너머 FTDI 칩 내부 FIFO 를 커널이
                  볼 수 없어 보수적으로 오래 블록한다. 실제 선로시간
                  (114 byte @3Mbps = 0.38ms)과 무관하게 8.5ms 를 잡아먹었다.
          → 필요한 만큼만 기다리는 고정 간격으로 교체했다.

        ■ time.sleep() 이 아니라 바쁜대기인 이유

          리눅스 sleep 은 0.4ms 급 해상도가 부정확해 1ms 이상 튄다. 그러면
          다시 예산을 낭비한다. 바쁜대기 비용은 50Hz × 0.4ms = 코어 2%
          (8코어 기준 0.25%) 이고, 되찾는 제어 예산 8.5ms(20ms 의 42%)에
          비하면 싸다.

        ■ 액션 지연에 미치는 영향: 없음

          goal 바이트는 이미 선로로 나가고 있는 상태에서 기다리는 것이라
          모터 도달 시각이 늦어지지 않는다. 실측으로도 GoalDelay p50 이
          8.2 → 6.6ms 로 오히려 개선됐다.

        Args:
            gap_us: 대기 시간(마이크로초). 0 이하면 즉시 반환(기능 비활성).
        """
        if gap_us <= 0.0:
            return
        end = time.perf_counter_ns() + int(gap_us * 1000.0)
        while time.perf_counter_ns() < end:
            pass

        self.group_write.clearParam()

    # ───────────────────────────────────────────
    # Profile Velocity (SyncWrite)
    # ───────────────────────────────────────────
    def set_profile_velocity_all(self, rpm: float):
        # ── Velocity Limit 상한 clamp ──
        # Profile Velocity 쓰기가 모터 Velocity Limit 을 넘으면 펌웨어가 조용히 거부하고
        # (SyncWrite 응답 없음) 레지스터는 직전 값으로 남는다. 그러면 standup/kick
        # _finish() 에서 200 을 쓰려다 씹혀, 직전 phase 의 느린 속도(20~30rpm)가
        # 그대로 최종 속도가 되어 이후 보행이 안 된다.
        # → 넘는 값은 거부당하게 두지 말고 limit(=보행 가능 최대치)로 깎아서 쓴다.
        #
        # 주의: rpm == 0 은 Dynamixel 에서 "무제한(=Velocity Limit 속도 사용)" 을 뜻하는
        #       특수값이라 clamp 대상이 아니다. 0 은 그대로 통과시킨다.
        #       (motor_zero 의 start_zero(final_rpm=0.0) 등이 이 의미로 0 을 쓴다.)
        req_rpm = float(rpm)
        vel_limit = float(getattr(self.cfg, "profile_velocity_limit_rpm", 0.0) or 0.0)
        if vel_limit > 0.0 and req_rpm > vel_limit:
            self._log_warn(
                f"Profile Velocity {req_rpm:.1f}rpm 이 Velocity Limit "
                f"{vel_limit:.1f}rpm 초과 → {vel_limit:.1f}rpm 으로 clamp "
                f"(원래 그냥 쓰면 펌웨어가 씹어서 직전 속도로 굳어버림)"
            )
            req_rpm = vel_limit

        tick = int(round(req_rpm / max(1e-9, self.cfg.rpm_per_tick_profile)))
        if tick < 0:
            tick = 0
        param = _u32_to_le_bytes(tick)

        self.group_write_profile.clearParam()
        any_added = False
        for dxl_id in self.used_ids:
            # ── 제외 관절(목) 건너뛰기 ──
            # 여기서 빼지 않으면 robot_init/standup/kick 의 마무리 45rpm 이
            # 목에도 그대로 들어가 카메라가 흔들린다.
            if dxl_id in self.pv_exclude_ids:
                continue
            if not self.group_write_profile.addParam(dxl_id, bytes(param)):
                self._log_warn(f"ProfileVel addParam failed id={dxl_id}")
                continue
            any_added = True

        if any_added:
            comm = self.group_write_profile.txPacket()
            if comm != 0:
                self._log_warn(f"SyncWrite(ProfileVel) failed: {self.packet.getTxRxResult(comm)}")

        self.group_write_profile.clearParam()
        # clamp 후 실제로 들어간 값(req_rpm)을 찍는다. 요청값과 다르면 위 warn 에 남음.
        self._log_info(
            f"Profile Velocity set: {req_rpm:.1f} rpm (tick={tick})"
            + (f" [제외: {self.pv_exclude_names}]" if self.pv_exclude_ids else ""))

    def set_profile_velocity_joints(self, rpm_by_joint: dict) -> list:
        """지정한 관절들에만 Profile Velocity 를 쓴다 (관절별로 다른 값 가능).

        set_profile_velocity_all() 이 "전체 같은 값" 이라면 이건 "선택 개별 값".
        목처럼 몸통과 전혀 다른 속도로 움직여야 하는 관절을 위해 만들었다.
        제외 목록(pv_exclude_ids)에 있어도 이 API 로는 정상적으로 쓰인다
        (제외는 어디까지나 '일괄 쓰기' 에서만 빠진다는 뜻).

        Args:
            rpm_by_joint: {joint_name: rpm}
                          rpm == 0 은 Dynamixel 에서 "무제한(Velocity Limit 속도)"
                          을 뜻하는 특수값이라 clamp 하지 않고 그대로 통과시킨다.
                          ★ 목에 0 을 쓰면 최대속도로 튄다. 호출 측에서 막을 것.

        Returns:
            실제로 기입된 joint_name 리스트. 알 수 없는 이름은 무시된다.
        """
        if self.group_write_profile is None:
            self._log_warn("[pv] 포트가 열려 있지 않음 → 건너뜀")
            return []

        vel_limit = float(getattr(self.cfg, "profile_velocity_limit_rpm", 0.0) or 0.0)

        self.group_write_profile.clearParam()
        applied = []
        for name, rpm in rpm_by_joint.items():
            if name not in self.jid:
                # servo_map 에 없는 이름(오타 등)은 조용히 무시하지 않고 알린다.
                self._log_warn(f"[pv] 알 수 없는 joint: {name}")
                continue
            dxl_id = int(self.jid[name])

            req_rpm = float(rpm)
            # 0(=무제한)은 특수값이라 clamp 대상이 아니다. all() 쪽과 같은 규칙.
            if vel_limit > 0.0 and req_rpm > vel_limit:
                self._log_warn(
                    f"[pv] {name} {req_rpm:.1f}rpm > Velocity Limit {vel_limit:.1f}rpm "
                    f"→ clamp (그냥 쓰면 펌웨어가 씹어서 직전 값으로 굳는다)")
                req_rpm = vel_limit

            tick = int(round(req_rpm / max(1e-9, self.cfg.rpm_per_tick_profile)))
            if tick < 0:
                tick = 0
            if not self.group_write_profile.addParam(dxl_id, bytes(_u32_to_le_bytes(tick))):
                self._log_warn(f"[pv] addParam 실패 {name}(id={dxl_id})")
                continue
            applied.append(name)

        if applied:
            comm = self.group_write_profile.txPacket()
            if comm != 0:
                self._log_warn(
                    f"[pv] SyncWrite 실패: {self.packet.getTxRxResult(comm)}")
                applied = []

        self.group_write_profile.clearParam()
        return applied

    # ───────────────────────────────────────────
    # PID Gains (SyncWrite, RAM 영역)
    # ───────────────────────────────────────────
    def set_pid_gains(self, profile: "GainProfile") -> dict:
        """관절별 Position PID 게인을 SyncWrite 로 일괄 기입한다.

        ── EEPROM 항목들과 다른 점 ──
        게인(addr 80/82/84)은 RAM 이라 토크를 끄지 않아도 쓸 수 있다.
        오히려 끄면 로봇이 주저앉아 튜닝 반복이 어려우므로 끄지 않는다.
        대신 전원 OFF / 모터 REBOOT 시 값이 날아가므로 그때마다 다시
        불러야 한다. (노드가 기동 시와 리부트 후에 호출한다)

        ── 왜 한 패킷으로 쓰는가 ──
        D(80) / I(82) / P(84) 가 연속 주소라 6 바이트를 한 번에 쓴다.
        개별로 쓰면 P 가 먼저 들어가는 순간 D 가 아직 0 이라 감쇠 없이
        높은 강성으로 동작해 로봇이 튈 수 있다. 한 패킷이면 그 위험이
        원천적으로 사라진다.

        Returns:
            {"written": int, "failed": [dxl_id, ...]}
        """
        if self.port is None or self.packet is None:
            self._log_warn("[gains] 포트가 열려 있지 않음 → 건너뜀")
            return {"written": 0, "failed": list(self.used_ids)}

        gw = GroupSyncWrite(self.port, self.packet, ADDR_PID_BLOCK, LEN_PID_BLOCK)
        id_gains = profile.build_id_map(self.jid)

        failed = []
        added = 0
        for dxl_id in self.used_ids:
            g = id_gains.get(dxl_id)
            if g is None:
                # servo_map 에 있지만 프로파일에서 못 찾은 경우.
                # 실제로는 build_id_map 이 전부 채우므로 방어용이다.
                continue
            if not gw.addParam(dxl_id, pid_block_bytes(g)):
                self._log_warn(f"[gains] addParam 실패 id={dxl_id}")
                failed.append(dxl_id)
                continue
            added += 1

        if added:
            comm = gw.txPacket()
            if comm != 0:
                self._log_warn(
                    f"[gains] SyncWrite 실패: {self.packet.getTxRxResult(comm)}")
                failed = list(self.used_ids)
                added = 0
        gw.clearParam()

        return {"written": added, "failed": failed}

    def read_pwm_limits(self) -> dict:
        """모든 모터의 PWM Limit(36, EEPROM) 을 읽어 캐시한다.

        Goal PWM(100) 은 PWM Limit 을 넘는 값을 쓰면 Range Error 로 거부된다
        ("The data value exceeds the limit value!"). 기동 시 1회 읽어두고
        이후 모든 쓰기를 이 값으로 clamp 한다.
        읽기에 실패한 모터는 공장 기본값 885 로 가정한다.
        """
        limits = {}
        for dxl_id in self.used_ids:
            v, comm, err = self.packet.read2ByteTxRx(
                self.port, dxl_id, ADDR_PWM_LIMIT)
            if comm != 0 or err != 0:
                self._log_warn(
                    f"[pwm] id={dxl_id} PWM Limit 읽기 실패 → 885 로 가정")
                limits[dxl_id] = 885
            else:
                limits[dxl_id] = int(v)
        self.pwm_limit = limits
        return limits

    def set_goal_pwm(self, pwm_by_id: dict) -> dict:
        """관절별 Goal PWM(100) 을 SyncWrite 로 기입한다.

        Args:
            pwm_by_id: {dxl_id: goal_pwm}. 비어 있으면 아무것도 하지 않는다.

        각 값은 그 모터의 PWM Limit 으로 clamp 된다. clamp 없이 넘기면
        모터가 Range Error 로 거부하는데, SyncWrite 는 응답이 없어서
        거부됐다는 사실조차 알 수 없다(Profile Velocity 와 같은 함정).

        Returns:
            {"written": int, "clamped": [(id, 요청값, 적용값), ...]}
        """
        if not pwm_by_id:
            return {"written": 0, "clamped": []}
        if self.port is None or self.packet is None:
            self._log_warn("[pwm] 포트가 열려 있지 않음 → 건너뜀")
            return {"written": 0, "clamped": []}

        gw = GroupSyncWrite(self.port, self.packet,
                            ADDR_GOAL_PWM, LEN_GOAL_PWM)
        clamped = []
        added = 0
        for dxl_id, val in pwm_by_id.items():
            dxl_id = int(dxl_id)
            lim = self.pwm_limit.get(dxl_id, 885)
            v = _clip_int(int(val), 1, lim)
            if v != int(val):
                clamped.append((dxl_id, int(val), v))
            if not gw.addParam(dxl_id, list(_u16_le_bytes(v))):
                self._log_warn(f"[pwm] addParam 실패 id={dxl_id}")
                continue
            added += 1

        if added:
            comm = gw.txPacket()
            if comm != 0:
                self._log_warn(
                    f"[pwm] SyncWrite 실패: {self.packet.getTxRxResult(comm)}")
                added = 0
        gw.clearParam()

        for dxl_id, req, got in clamped:
            self._log_warn(
                f"[pwm] id={dxl_id} goal_pwm {req} → {got} 로 clamp "
                f"(PWM Limit={self.pwm_limit.get(dxl_id, 885)})")
        return {"written": added, "clamped": clamped}

    def set_goal_pwm_full(self, joint_names=None) -> dict:
        """지정 관절(또는 전체)의 Goal PWM 을 각자의 PWM Limit(=100%)으로 올린다.

        robot_init / standup 처럼 팔로 실제 힘을 써야 하는 behavior 진입 시
        호출한다. 상한이 낮게 걸려 있으면 팔이 목표까지 이동하지 못해
        도달 판정(tol/settle)이 timeout 으로 끝난다.
        """
        if joint_names is None:
            ids = list(self.used_ids)
        else:
            ids = [int(self.jid[n]) for n in joint_names if n in self.jid]
        return self.set_goal_pwm(
            {i: self.pwm_limit.get(i, 885) for i in ids})

    def apply_goal_pwm_profile(self, profile: "GainProfile") -> dict:
        """게인 프로파일의 goal_pwm 설정을 적용한다.

        YAML 에 goal_pwm 이 없는 관절은 손대지 않는다. 다만 behavior 가
        set_goal_pwm_full() 로 올려둔 상태일 수 있으므로, 프로파일에
        goal_pwm 이 지정된 관절은 반드시 그 값으로 되돌린다.
        """
        return self.set_goal_pwm(profile.build_goal_pwm_map(self.jid))

    def read_pid_gains(self, dxl_id: int) -> Optional[dict]:
        """단일 모터의 현재 게인을 읽는다. 검증/진단용.

        SyncRead 를 새로 만들지 않고 단발 read 를 쓴다. 검증은 기동 시
        1 회 또는 서비스 호출 시에만 하므로 성능이 문제되지 않는다.
        """
        if self.port is None or self.packet is None:
            return None
        out = {}
        for key, addr in (("position_d_gain", 80),
                          ("position_i_gain", 82),
                          ("position_p_gain", 84),
                          # Goal PWM(100) 도 같이 읽는다. 게인과 똑같이 RAM 이라
                          # REBOOT 로 날아가므로 검증 대상이 되어야 한다.
                          (GOAL_PWM_KEY, ADDR_GOAL_PWM)):
            v, comm, err = self.packet.read2ByteTxRx(self.port, dxl_id, addr)
            if comm != 0 or err != 0:
                return None
            out[key] = int(v)
        return out

    def verify_pid_gains(self, profile: "GainProfile") -> list:
        """기입값과 실제 레지스터를 대조한다. 불일치 목록을 반환.

        게인이 안 들어갔는데 모르고 튜닝하면 결과 해석이 전부 어긋난다.
        기동 시 한 번은 반드시 확인할 것.
        """
        bad = []
        for jname, dxl_id in self.jid.items():
            want = profile.gains_for(jname)
            got = self.read_pid_gains(int(dxl_id))
            if got is None:
                bad.append(f"{jname}(ID {dxl_id}): 읽기 실패")
                continue
            # want 에는 goal_pwm 이 있을 수도 없을 수도 있다(YAML 선택 항목).
            # got 에 없는 키를 비교하면 KeyError 가 나므로 교집합만 본다.
            # goal_pwm 은 PWM Limit 으로 clamp 되어 기입되므로, 기대값도
            # 같은 clamp 를 적용한 뒤에 대조해야 한다.
            lim = self.pwm_limit.get(int(dxl_id), 885)
            diff = []
            for k in want:
                if k not in got:
                    continue
                exp = want[k]
                if k == GOAL_PWM_KEY:
                    exp = _clip_int(int(exp), 1, lim)
                if got[k] != exp:
                    diff.append(f"{k} {got[k]}!={exp}")
            if diff:
                bad.append(f"{jname}(ID {dxl_id}): " + ", ".join(diff))
        return bad

    # ───────────────────────────────────────────
    # Present Velocity + Position (SyncRead)
    # ───────────────────────────────────────────
    def read_state(self) -> tuple[bool, list[JointReading]]:
        """
        한 번의 SyncRead 사이클 실행 + 단위 변환.

        반환: (ok, readings)
          - ok=False : SyncRead txRxPacket 자체가 실패. readings 는 빈 리스트.
          - ok=True  : 응답한 모터들의 [JointReading] 리스트. 응답 없는 모터는 누락.

        내부 통계(read_total, read_fail, id_ok, id_miss)도 같이 업데이트한다.
        """
        self.read_total += 1
        comm = self.group_read.txRxPacket()
        if comm != 0:
            self.read_fail += 1
            self._log_warn(f"SyncRead failed: {self.packet.getTxRxResult(comm)}")
            return False, []

        ct = self.cfg.control_table
        out: list[JointReading] = []

        for dxl_id in self.used_ids:
            if not self.group_read.isAvailable(dxl_id, ct.bulk_read_start_addr, ct.bulk_read_length):
                self.id_miss[dxl_id] += 1
                continue
            self.id_ok[dxl_id] += 1

            vel_u32 = self.group_read.getData(dxl_id, ct.present_velocity, ct.len_present_velocity)
            pos_u32 = self.group_read.getData(dxl_id, ct.present_position, ct.len_present_position)

            vel_i32 = _int32_from_u32(vel_u32)
            pos = int(pos_u32)

            name = self.rev_jid.get(dxl_id, f"dxl_{dxl_id}")
            # 부호 뒤집기(direction_display)는 위치·속도에 '똑같이' 적용해야 한다.
            #   속도는 위치의 시간미분이므로, 위치에만 -1 을 곱하고 속도를 그대로
            #   두면 q 와 q̇ 의 부호가 서로 반대가 된다.
            #   현재 servo_map 에서 -1 인 관절은 neck_pitch_joint 뿐이고 목 속도를
            #   쓰는 소비자가 없어 증상이 안 보였을 뿐이다. 다리/팔 관절 중 하나라도
            #   -1 로 바꾸는 순간 정책 러너의 dof_vel 관측이 그 관절만 부호가
            #   뒤집혀 들어가 보행이 깨진다(원인 추적도 매우 어렵다).
            sgn = self.dir_display.get(name, 1)
            rad = (pos - self.center_tick) * self.rad_per_tick * sgn
            rpm = vel_i32 * self.rpm_per_tick * sgn
            radps = rpm * (2.0 * math.pi / 60.0)

            out.append(JointReading(name=name, rad=rad, radps=radps))

        return True, out

    # ───────────────────────────────────────────
    # Hardware Error Status
    # ───────────────────────────────────────────
    def read_hardware_errors(self) -> dict:
        """
        모든 모터의 Hardware Error Status (addr 70, 1byte) 읽기.
        반환: {dxl_id: error_byte}

        MX-64(2.0) 에러 비트:
          bit 0: Input Voltage Error
          bit 2: Overheating Error
          bit 3: Motor Encoder Error
          bit 5: Overload Error
        """
        errors = {}
        addr = self.cfg.control_table.hardware_error
        for dxl_id in self.used_ids:
            val, comm, err = self.packet.read1ByteTxRx(self.port, dxl_id, addr)
            if comm != 0:
                self._log_warn(
                    f"HW error read fail id={dxl_id}: "
                    f"{self.packet.getTxRxResult(comm)}"
                )
                continue
            errors[dxl_id] = int(val)
        return errors

    # ───────────────────────────────────────────
    # Raw single-byte access (for motor reset 등)
    # ───────────────────────────────────────────
    def write1byte(self, dxl_id: int, addr: int, value: int):
        return self.packet.write1ByteTxRx(self.port, dxl_id, addr, int(value) & 0xFF)

    def read1byte(self, dxl_id: int, addr: int):
        return self.packet.read1ByteTxRx(self.port, dxl_id, addr)

    # ───────────────────────────────────────────
    # Reboot (Protocol 2.0)
    # ───────────────────────────────────────────
    def reboot(self, dxl_id: int):
        """단일 모터 리부트. (comm_result, dxl_error) 반환."""
        return self.packet.reboot(self.port, dxl_id)

    def torque_enable(self, dxl_id: int, enable: bool):
        """단일 모터 토크 ON/OFF (motor reset 후 재활성화 등에 사용)."""
        return self.write1byte(dxl_id, self.cfg.control_table.torque_enable,
                               1 if enable else 0)

    # ───────────────────────────────────────────
    # 토크오프 + 죽은(에러 래치) 모터 리셋
    # ───────────────────────────────────────────
    def set_torque_off_with_reset(self, settle_sec: float = 0.3) -> dict:
        """
        토크오프 경로 전용: HW 에러 래치된(=죽은) 모터를 REBOOT 로 지우고
        전체 토크오프한다.

        순서:
          ① HW Error(addr 70) 읽어 에러 래치된 모터 식별
          ② 전체 토크오프 (즉시 안전 확보 — 로봇 힘 빠짐)
          ③ 에러난 모터만 REBOOT
             - 오버로드/과열 등 래치는 reboot 로만 지워진다.
             - reboot 는 RAM 을 초기화하므로 Torque Enable 도 0 이 된다
               (= 리부트한 모터도 결과적으로 토크오프 상태).
          ④ 리부트 모터 복귀 대기 후 에러 재확인 → 잔존 에러 report

        토크를 끄는 상황이라 reboot 로 인한 급발진 위험이 없다(3턴의 토크-ON
        복구와 달리 present→goal 이 필요 없음).

        반환 report:
          {
            "errored":  {id: err_byte},  # 처음 읽은 에러(0 아닌 것만)
            "rebooted": [id, ...],       # 리부트한 모터
            "residual": {id: err_byte},  # reboot 후에도 남은 에러(과열/저전압 지속 등)
          }
        """
        report = {"errored": {}, "rebooted": [], "residual": {}}

        # ① 에러 읽기
        errors = self.read_hardware_errors()          # {id: err_byte}
        errored = {i: e for i, e in errors.items() if e != 0}
        report["errored"] = dict(errored)

        # ② 전체 토크오프 (즉시)
        self.set_torque_all(False)

        # 에러 없으면 일반 토크오프와 동일 — 여기서 끝(빠른 경로)
        if not errored:
            return report

        # ③ 에러난 모터 REBOOT (래치 clear). reboot 후 Torque=0.
        for dxl_id in errored:
            self._log_warn(
                f"HW error id={dxl_id} status=0x{errored[dxl_id]:02X}"
                f"({self._decode_hw_error(errored[dxl_id])}) → REBOOT (토크오프 유지)")
            self.reboot(dxl_id)
            report["rebooted"].append(dxl_id)

        # ④ 리부트 복귀 대기(~200ms+) 후 에러 재확인
        time.sleep(max(0.0, settle_sec))
        after = self.read_hardware_errors()
        for dxl_id in report["rebooted"]:
            e2 = after.get(dxl_id, None)
            if e2 is None:
                # 아직 응답 안 함(복귀 중). 토크오프 상황이라 급하지 않으므로 넘어감.
                self._log_warn(f"id={dxl_id} reboot 후 아직 응답 없음(복귀 중)")
                continue
            if e2 != 0:
                report["residual"][dxl_id] = e2
                self._log_warn(
                    f"id={dxl_id} reboot 후에도 에러 0x{e2:02X}"
                    f"({self._decode_hw_error(e2)}) 잔존 — 과열/저전압 등 원인 지속")

        # 리부트로 복귀한 모터는 Torque=0 이지만, 혹시 몰라 한 번 더 전체 off(무해).
        self.set_torque_all(False)
        return report

    @staticmethod
    def _decode_hw_error(err: int) -> str:
        """addr 70 비트 → 사람이 읽는 문자열."""
        bits = []
        if err & 0x01:
            bits.append("InputVoltage")
        if err & 0x04:
            bits.append("Overheating")
        if err & 0x08:
            bits.append("Encoder")
        if err & 0x10:
            bits.append("ElectricalShock")
        if err & 0x20:
            bits.append("Overload")
        return ",".join(bits) if bits else "none"

    # ───────────────────────────────────────────
    # Position Limits (EEPROM)
    # ───────────────────────────────────────────
    def read_position_limits_rad(self) -> dict:
        """
        모든 등록된 모터의 EEPROM Min/Max Position Limit 를 읽어서
        joint_name 기준 rad 단위로 반환.

        반환 형식:
          {
            joint_name: {"lower": float_rad, "upper": float_rad, "ok": True},
            ...
          }
          읽기 실패한 모터는 ok=False 로 표시 (lower/upper 는 0.0).

        주의:
          - EEPROM 의 Min/Max Position Limit 는 raw tick 단위 (0~4095).
          - tick → rad 변환은 read_state 와 동일한 공식 사용:
              rad = (tick - center_tick) * rad_per_tick * dir_display
          - direction_display 가 -1 인 모터는 변환 후 min/max 가 뒤집히므로
            min(a,b), max(a,b) 로 재정렬해서 lower/upper 보장.
          - EEPROM 읽기는 SyncRead 가 아니라 개별 read 라 좀 느림 (모터 수×~3ms).
            18 모터면 ~50ms 정도. GUI 시작 시 1회만 호출하므로 부담 없음.
        """
        if self.port is None or self.packet is None:
            raise RuntimeError("hw.open() before read_position_limits_rad().")

        ct = self.cfg.control_table
        out = {}
        for dxl_id in self.used_ids:
            name = self.rev_jid.get(dxl_id, f"dxl_{dxl_id}")
            # Min/Max Position Limit 둘 다 4바이트 unsigned (raw tick 0~4095)
            min_tick, comm1, err1 = self.packet.read4ByteTxRx(
                self.port, dxl_id, ct.min_position_limit)
            max_tick, comm2, err2 = self.packet.read4ByteTxRx(
                self.port, dxl_id, ct.max_position_limit)

            if comm1 != 0 or err1 != 0 or comm2 != 0 or err2 != 0:
                # 어느 한 쪽이라도 실패하면 ok=False
                self._log_warn(
                    f"position limit read fail id={dxl_id} ({name}): "
                    f"comm=({comm1},{comm2}), err=({err1},{err2})"
                )
                out[name] = {"lower": 0.0, "upper": 0.0, "ok": False}
                continue

            # tick → rad 변환. dir_display 부호가 -1 이면 결과의 부호도
            # 반전되므로 min/max 가 뒤집힐 수 있다. 안전하게 정렬.
            d = self.dir_display.get(name, 1)
            rad_a = (int(min_tick) - self.center_tick) * self.rad_per_tick * d
            rad_b = (int(max_tick) - self.center_tick) * self.rad_per_tick * d
            lower = min(rad_a, rad_b)
            upper = max(rad_a, rad_b)

            out[name] = {"lower": float(lower), "upper": float(upper), "ok": True}

        return out

    # ───────────────────────────────────────────
    # Auxiliary bulk reader (diagnostics 등)
    # ───────────────────────────────────────────
    def make_bulk_reader(self, start_addr: int, length: int) -> GroupSyncRead:
        """
        SyncRead 그룹을 새로 만들고 모든 used_id 를 등록해서 돌려준다.
        호출자가 직접 txRxPacket(), isAvailable(), getData() 를 사용한다.

        주 SyncRead(group_read) 와 다른 주소 블록을 별도로 폴링해야 할 때 사용.
        예: diagnostics 가 PWM/Current/Voltage/Temp 묶음을 따로 읽는 경우.
        """
        if self.port is None or self.packet is None:
            raise RuntimeError("hw.open() before make_bulk_reader().")
        reader = GroupSyncRead(self.port, self.packet, start_addr, length)
        for dxl_id in self.used_ids:
            if not reader.addParam(dxl_id):
                self._log_warn(f"bulk reader addParam failed: id={dxl_id}")
        return reader
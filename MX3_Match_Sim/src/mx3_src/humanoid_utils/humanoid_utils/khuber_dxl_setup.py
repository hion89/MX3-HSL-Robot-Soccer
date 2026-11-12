#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khuber_dxl_setup.py
===================
KHUBER 휴머노이드의 18개 다이나믹셀(XM540-W150-R) 모터에
  1) 관절별 Min/Max Position Limit
  2) 공통설정 (Return Delay Time / Max·Min Voltage Limit / Velocity Limit)
을 U2D2(Protocol 2.0)로 한 번에 자동 기입하는 도구.

------------------------------------------------------------------------------
[모터 / 컨트롤 테이블]  XM540-W150-R (X-series, Protocol 2.0)  — e-manual 검증값
  Addr  Item                  Bytes  Unit              비고
   9    Return Delay Time      1     2 us/LSB          0~254 (기본 250)
  32    Max Voltage Limit      2     0.1 V/LSB
  34    Min Voltage Limit      2     0.1 V/LSB
  44    Velocity Limit         4     0.229 rev/min/LSB
  48    Max Position Limit     4     1 pulse (0~4095)  ★raw 엔코더 기준
  52    Min Position Limit     4     1 pulse (0~4095)
  64    Torque Enable          1     0/1               ★EEPROM 쓰기엔 반드시 0

[중요] 위 항목은 전부 EEPROM 영역이다.
  - EEPROM은 Torque Enable=0 일 때만 쓰기가 가능 → 스크립트가 시작 시 토크를 끈다.
  - 모터 전원 OFF 후에도 값이 유지된다(비휘발성).
  - Position Limit는 Operating Mode(11)=3 (Position Control Mode)에서만 동작한다.
    본 스크립트는 Operating Mode를 변경하지 않고, 현재 모드를 읽어 경고만 표시한다.

[Min/Max Position Limit 기입 순서 주의]
  펌웨어는 "Max >= Min" 제약을 위반하는 쓰기를 거부할 수 있다.
  따라서 항상 (1)Max=4095 로 천장을 먼저 연 뒤 → (2)Min=목표 → (3)Max=목표
  순서로 기입하여, 중간 단계에서도 Max>=Min 이 항상 성립하도록 한다.

[안전]
  - 작업 전/후 모두 토크 OFF 상태를 유지한다(예기치 않은 움직임 방지).
  - 실제 하드웨어 없이 설정만 점검하려면 --dry-run 사용(SDK/포트 불필요).
------------------------------------------------------------------------------
사용 예:
  # 1) 하드웨어 없이 적용 계획만 확인 (가장 먼저 권장)
  python3 khuber_dxl_setup.py --config khuber_motor_config.xlsx --dry-run

  # 2) 실제 적용 (기본 포트/보레이트)
  python3 khuber_dxl_setup.py --config khuber_motor_config.xlsx

  # 3) CSV 두 개로 적용
  python3 khuber_dxl_setup.py \
      --joints-csv khuber_joint_limits.csv \
      --common-csv khuber_common_settings.csv

  # 4) 특정 ID만 / 공통설정 건너뛰기 / 적용 후 리부트
  python3 khuber_dxl_setup.py --config khuber_motor_config.xlsx --ids 1,2,3 --no-common --reboot
"""

import argparse
import csv
import os
import sys

# ─────────────────────────────────────────────────────────────────────────
# XM540-W150-R 컨트롤 테이블 주소/길이 (Protocol 2.0)
# ─────────────────────────────────────────────────────────────────────────
ADDR_RETURN_DELAY_TIME = 9
ADDR_OPERATING_MODE    = 11   # 3 = Position Control Mode
ADDR_MAX_VOLTAGE_LIMIT = 32
ADDR_MIN_VOLTAGE_LIMIT = 34
ADDR_VELOCITY_LIMIT    = 44
ADDR_MAX_POSITION_LIMIT = 48
ADDR_MIN_POSITION_LIMIT = 52
ADDR_TORQUE_ENABLE     = 64

LEN_1, LEN_2, LEN_4 = 1, 2, 4

POSITION_TICK_MIN = 0
POSITION_TICK_MAX = 4095
OP_MODE_POSITION  = 3

# 단위 환산(표시용 only)
UNIT_RETURN_DELAY_US = 2.0     # us per LSB
UNIT_VOLTAGE_V       = 0.1     # V  per LSB
UNIT_VELOCITY_RPM    = 0.229   # rev/min per LSB

# 공통설정 키 → (주소, 길이, 단위환산계수, 단위문자열)
COMMON_SPEC = {
    "return_delay_time": (ADDR_RETURN_DELAY_TIME, LEN_1, UNIT_RETURN_DELAY_US, "us"),
    "max_voltage_limit": (ADDR_MAX_VOLTAGE_LIMIT, LEN_2, UNIT_VOLTAGE_V,       "V"),
    "min_voltage_limit": (ADDR_MIN_VOLTAGE_LIMIT, LEN_2, UNIT_VOLTAGE_V,       "V"),
    "velocity_limit":    (ADDR_VELOCITY_LIMIT,    LEN_4, UNIT_VELOCITY_RPM,    "rev/min"),
}

# 콘솔 색상(가독성용). 터미널이 지원 안 하면 무시됨.
class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[96m"; D = "\033[2m"; X = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────
# 단위/각도 변환 — 원본 엑셀(로봇관절정보_U2D2.xlsx)의 수식 규약을 그대로 재현
#   MAX_A = 180 - lower_deg ,  MIN_A = 180 - upper_deg          (servo 절대각, 0~360)
#   max_position = ROUND(MAX_A * 4096/360, 0) + 1
#   min_position = ROUND(MIN_A * 4096/360, 0) - 1
#   (180-deg 반전이라 lower_deg→max_position, upper_deg→min_position 에 대응)
#   ※ 18개 관절 원본 tick과 1:1 일치 검증 완료.
# ─────────────────────────────────────────────────────────────────────────
import math


def excel_round(x):
    """Excel ROUND과 동일(0.5는 0에서 먼 쪽으로 올림). Python 기본 round의 은행가식 반올림과 다름."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def deg_to_ticks(lower_deg, upper_deg):
    """관절 각도(deg) → (min_position, max_position) tick. 엑셀 수식과 동일."""
    max_pos = excel_round((180.0 - lower_deg) * 4096.0 / 360.0) + 1
    min_pos = excel_round((180.0 - upper_deg) * 4096.0 / 360.0) - 1
    return min_pos, max_pos


def eng_to_raw(key, value):
    """공통설정 공학단위값(예: 13.0V, 50us, 44.88rev/min) → 레지스터 raw값."""
    scale = COMMON_SPEC[key][2]            # (addr, len, scale, unit)
    return excel_round(float(value) / scale)


# ═════════════════════════════════════════════════════════════════════════
# 1) 설정 로딩
# ═════════════════════════════════════════════════════════════════════════
def _resolve_joint(name, jid, lower_deg, upper_deg, cached_min, cached_max):
    """
    한 관절의 min/max tick을 확정한다.
    우선순위:
      1) lower_deg / upper_deg 가 있으면 → 수식으로 직접 계산 (정본)
         (엑셀 계산엔진/캐시에 의존하지 않으므로 가장 신뢰도 높음)
      2) 각도가 없으면 → 파일의 min_position/max_position(tick) 값을 사용
    또한 (1)인데 캐시된 tick이 함께 있고 다르면 경고를 띄운다(수식이 깨졌는지 점검용).
    """
    if lower_deg is not None and upper_deg is not None:
        mn, mx = deg_to_ticks(lower_deg, upper_deg)
        if cached_min is not None and cached_max is not None:
            if int(cached_min) != mn or int(cached_max) != mx:
                print(f"  {C.Y}! {name}(ID {jid}) 파일의 tick({cached_min},{cached_max})와 "
                      f"각도 계산값({mn},{mx}) 불일치 → 각도 계산값 사용{C.X}")
        return mn, mx
    if cached_min is None or cached_max is None:
        sys.exit(f"{C.R}[오류]{C.X} {name}(ID {jid}): 각도(lower/upper_deg)도 "
                 f"tick(min/max_position)도 없습니다.")
    return int(cached_min), int(cached_max)


def load_joints_csv(path):
    """관절 CSV 로드 → [{name,id,min,max,lower_deg,upper_deg}, ...]"""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("joint_name"):
                continue
            lo = _opt_float(r.get("lower_deg"))
            up = _opt_float(r.get("upper_deg"))
            cmin = _opt_int(r.get("min_position"))
            cmax = _opt_int(r.get("max_position"))
            name = r["joint_name"].strip()
            jid = int(r["dxl_id"])
            mn, mx = _resolve_joint(name, jid, lo, up, cmin, cmax)
            rows.append({"name": name, "id": jid, "min": mn, "max": mx,
                         "lower_deg": lo, "upper_deg": up})
    return rows


def load_common_csv(path):
    """
    공통설정 CSV 로드 → {key: raw_value}
    'value' 열은 공학단위(V/us/rev-min)로 보고 raw로 환산한다.
    (예전처럼 'value'에 raw를 직접 넣는 경우와 구분하기 위해 'unit' 열 유무로 판단)
    """
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        has_unit = reader.fieldnames is not None and "unit" in reader.fieldnames
        for r in reader:
            k = (r.get("key") or "").strip()
            if k not in COMMON_SPEC:
                continue
            out[k] = eng_to_raw(k, r["value"]) if has_unit else int(float(r["value"]))
    return out


def _opt_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def find_default_config():
    """
    --config 미지정 시 기본 설정 파일 경로를 찾는다.
    우선순위: (1) ROS 설치 share 디렉토리  (2) 현재 폴더
    ROS 환경이 아니어도(standalone) 동작하도록 ament import는 try로 감싼다.
    """
    # (1) ros2 run 으로 실행된 경우: 설치된 share/humanoid_utils/config
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("humanoid_utils")
        for fn in ("khuber_motor_config.xlsx", "khuber_joint_limits.csv"):
            p = os.path.join(share, "config", fn)
            if os.path.exists(p):
                return p
    except Exception:
        pass
    # (2) 현재 폴더(standalone 실행)
    for fn in ("khuber_motor_config.xlsx", "khuber_joint_limits.csv"):
        if os.path.exists(fn):
            return fn
    return None


def load_from_xlsx(path):
    """xlsx 통합본(joints/common 시트) 로드 → (joints, common). openpyxl 필요."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        sys.exit(f"{C.R}[오류]{C.X} xlsx를 읽으려면 openpyxl 필요: pip install openpyxl\n"
                 f"      또는 CSV 모드 사용: --joints-csv ... --common-csv ...")
    wb = load_workbook(path, data_only=True)

    # ── joints 시트 ──
    ws = wb["joints"]
    header = [str(c.value).strip() if c.value is not None else "" for c in next(ws.iter_rows(max_row=1))]
    idx = {h: i for i, h in enumerate(header)}
    joints = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[idx["joint_name"]] is None:
            continue
        name = str(row[idx["joint_name"]]).strip()
        jid = int(row[idx["dxl_id"]])
        lo = _opt_float(row[idx["lower_deg"]]) if "lower_deg" in idx else None
        up = _opt_float(row[idx["upper_deg"]]) if "upper_deg" in idx else None
        cmin = _opt_int(row[idx["min_position"]]) if "min_position" in idx else None
        cmax = _opt_int(row[idx["max_position"]]) if "max_position" in idx else None
        mn, mx = _resolve_joint(name, jid, lo, up, cmin, cmax)
        joints.append({"name": name, "id": jid, "min": mn, "max": mx,
                       "lower_deg": lo, "upper_deg": up})

    # ── common 시트 ──  value(공학단위) → raw 환산. 'unit' 열 있으면 공학단위로 간주.
    common = {}
    if "common" in wb.sheetnames:
        wsc = wb["common"]
        chdr = [str(c.value).strip() if c.value is not None else "" for c in next(wsc.iter_rows(max_row=1))]
        cidx = {h: i for i, h in enumerate(chdr)}
        has_unit = "unit" in cidx
        for row in wsc.iter_rows(min_row=2, values_only=True):
            if not row or row[cidx["key"]] is None:
                continue
            k = str(row[cidx["key"]]).strip()
            if k not in COMMON_SPEC:
                continue
            val = row[cidx["value"]]
            common[k] = eng_to_raw(k, val) if has_unit else int(float(val))
    return joints, common


def _opt_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ═════════════════════════════════════════════════════════════════════════
# 2) 설정 검증 (실제 쓰기 전에 값이 유효한지 확인)
# ═════════════════════════════════════════════════════════════════════════
def validate(joints, common):
    """문제 있으면 메시지 리스트 반환(비면 정상)."""
    errs = []
    seen_ids = {}
    for j in joints:
        # 포지션 범위 0~4095, min<max
        if not (POSITION_TICK_MIN <= j["min"] <= POSITION_TICK_MAX):
            errs.append(f"{j['name']}(ID {j['id']}): min_position {j['min']} 범위(0~4095) 벗어남")
        if not (POSITION_TICK_MIN <= j["max"] <= POSITION_TICK_MAX):
            errs.append(f"{j['name']}(ID {j['id']}): max_position {j['max']} 범위(0~4095) 벗어남")
        if j["min"] >= j["max"]:
            errs.append(f"{j['name']}(ID {j['id']}): min({j['min']}) >= max({j['max']}) — min<max 여야 함")
        # ID 중복 검사
        if j["id"] in seen_ids:
            errs.append(f"ID {j['id']} 중복: {seen_ids[j['id']]} / {j['name']}")
        seen_ids[j["id"]] = j["name"]

    # 공통설정 sanity check
    if "return_delay_time" in common and not (0 <= common["return_delay_time"] <= 254):
        errs.append(f"return_delay_time {common['return_delay_time']} 범위(0~254) 벗어남")
    if {"max_voltage_limit", "min_voltage_limit"} <= common.keys():
        if common["min_voltage_limit"] > common["max_voltage_limit"]:
            errs.append("min_voltage_limit > max_voltage_limit (Min<=Max 여야 함)")
    return errs


# ═════════════════════════════════════════════════════════════════════════
# 3) 적용 계획 출력 (dry-run / 실제 실행 공통)
# ═════════════════════════════════════════════════════════════════════════
def print_plan(joints, common, do_common, do_limits):
    print(f"\n{C.B}══════════════ 적용 계획 ══════════════{C.X}")
    if do_common and common:
        print(f"{C.Y}[공통설정] (모든 모터 동일 적용){C.X}")
        for k, (addr, ln, scale, unit) in COMMON_SPEC.items():
            if k in common:
                raw = common[k]
                print(f"   - {k:<20} addr {addr:<3} = {raw:<5} ({raw*scale:g} {unit})")
    if do_limits:
        print(f"{C.Y}[관절별 Position Limit] ({len(joints)}개 모터){C.X}")
        print(f"   {'ID':>3}  {'joint':<24} {'min(52)':>8} {'max(48)':>8}   {'deg(lo~up)'}")
        for j in sorted(joints, key=lambda x: x["id"]):
            deg = ""
            if j["lower_deg"] is not None and j["upper_deg"] is not None:
                deg = f"{j['lower_deg']:.0f}~{j['upper_deg']:.0f}"
            print(f"   {j['id']:>3}  {j['name']:<24} {j['min']:>8} {j['max']:>8}   {deg}")
    print(f"{C.B}════════════════════════════════════════{C.X}\n")


# ═════════════════════════════════════════════════════════════════════════
# 4) 실제 하드웨어 적용
# ═════════════════════════════════════════════════════════════════════════
def write_register(packet, port, dxl_id, addr, length, value):
    """길이에 맞는 write 호출. (comm_result, dxl_error) 반환."""
    if length == LEN_1:
        return packet.write1ByteTxRx(port, dxl_id, addr, value & 0xFF)
    elif length == LEN_2:
        return packet.write2ByteTxRx(port, dxl_id, addr, value & 0xFFFF)
    else:  # LEN_4
        return packet.write4ByteTxRx(port, dxl_id, addr, value & 0xFFFFFFFF)


def read_register(packet, port, dxl_id, addr, length):
    """길이에 맞는 read 호출. (value, comm_result, dxl_error) 반환."""
    if length == LEN_1:
        return packet.read1ByteTxRx(port, dxl_id, addr)
    elif length == LEN_2:
        return packet.read2ByteTxRx(port, dxl_id, addr)
    else:
        return packet.read4ByteTxRx(port, dxl_id, addr)


def apply_to_hardware(joints, common, args):
    """U2D2 연결 → ping → 토크OFF → 공통/리밋 기입 → 검증 → (옵션)리부트."""
    # SDK는 실제 실행 시에만 import (dry-run/설치 안 된 환경 보호)
    try:
        from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
    except ImportError:
        sys.exit(f"{C.R}[오류]{C.X} dynamixel_sdk 없음: pip install dynamixel-sdk")

    # ── 포트 / 패킷 핸들러 (Protocol 2.0) ──
    port = PortHandler(args.port)
    packet = PacketHandler(2.0)
    if not port.openPort():
        sys.exit(f"{C.R}[오류]{C.X} 포트 열기 실패: {args.port}")
    if not port.setBaudRate(args.baud):
        port.closePort()
        sys.exit(f"{C.R}[오류]{C.X} 보레이트 설정 실패: {args.baud}")
    print(f"{C.G}[연결]{C.X} {args.port} @ {args.baud}bps (Protocol 2.0)")

    results = []  # (id, name, status, detail)
    try:
        for j in sorted(joints, key=lambda x: x["id"]):
            jid, name = j["id"], j["name"]

            # ── (a) ping: 모터 존재 확인 ──
            _, comm, err = packet.ping(port, jid)
            if comm != COMM_SUCCESS or err != 0:
                msg = packet.getTxRxResult(comm) if comm != COMM_SUCCESS else packet.getRxPacketError(err)
                print(f"  {C.R}✗ ID {jid:>2} {name:<24} ping 실패: {msg}{C.X}")
                results.append((jid, name, "PING_FAIL", msg))
                continue

            # ── (b) Operating Mode 확인 (Position Control 아닌 경우 경고만) ──
            op_mode, comm, err = read_register(packet, port, jid, ADDR_OPERATING_MODE, LEN_1)
            if comm == COMM_SUCCESS and err == 0 and op_mode != OP_MODE_POSITION:
                print(f"  {C.Y}! ID {jid:>2} {name:<24} Operating Mode={op_mode} "
                      f"(Position Limit는 모드=3에서만 동작){C.X}")

            # ── (c) EEPROM 쓰기 위해 토크 OFF (필수) ──
            write_register(packet, port, jid, ADDR_TORQUE_ENABLE, LEN_1, 0)

            failed = []  # 이 모터에서 실패한 항목

            # ── (d) 공통설정 기입 ──
            if args.common_enabled:
                for k, (addr, ln, _scale, _unit) in COMMON_SPEC.items():
                    if k not in common:
                        continue
                    comm, err = write_register(packet, port, jid, addr, ln, common[k])
                    if comm != COMM_SUCCESS or err != 0:
                        failed.append(k)

            # ── (e) Position Limit 기입 (Max>=Min 항상 성립하도록 천장 먼저 개방) ──
            if args.limits_enabled:
                # (1) Max=4095 로 천장 개방
                write_register(packet, port, jid, ADDR_MAX_POSITION_LIMIT, LEN_4, POSITION_TICK_MAX)
                # (2) Min = 목표
                c1, e1 = write_register(packet, port, jid, ADDR_MIN_POSITION_LIMIT, LEN_4, j["min"])
                # (3) Max = 목표
                c2, e2 = write_register(packet, port, jid, ADDR_MAX_POSITION_LIMIT, LEN_4, j["max"])
                if c1 != COMM_SUCCESS or e1 != 0:
                    failed.append("min_position")
                if c2 != COMM_SUCCESS or e2 != 0:
                    failed.append("max_position")

            # ── (f) 읽기 검증 ──
            verify_msgs = []
            if args.verify:
                if args.limits_enabled:
                    rmin, *_ = read_register(packet, port, jid, ADDR_MIN_POSITION_LIMIT, LEN_4)
                    rmax, *_ = read_register(packet, port, jid, ADDR_MAX_POSITION_LIMIT, LEN_4)
                    if rmin != j["min"]:
                        verify_msgs.append(f"min {rmin}!={j['min']}")
                    if rmax != j["max"]:
                        verify_msgs.append(f"max {rmax}!={j['max']}")
                if args.common_enabled:
                    for k, (addr, ln, _s, _u) in COMMON_SPEC.items():
                        if k not in common:
                            continue
                        rv, *_ = read_register(packet, port, jid, addr, ln)
                        if rv != common[k]:
                            verify_msgs.append(f"{k} {rv}!={common[k]}")

            # ── (g) 결과 판정/출력 ──
            if failed or verify_msgs:
                detail = ("write실패:" + ",".join(failed) if failed else "") + \
                         (" verify:" + ",".join(verify_msgs) if verify_msgs else "")
                print(f"  {C.R}✗ ID {jid:>2} {name:<24} {detail}{C.X}")
                results.append((jid, name, "FAIL", detail))
            else:
                print(f"  {C.G}✓ ID {jid:>2} {name:<24} "
                      f"min={j['min']} max={j['max']} 적용 완료{C.X}")
                results.append((jid, name, "OK", ""))

            # ── (h) 옵션: 모터 리부트로 EEPROM 확실히 반영 ──
            if args.reboot:
                packet.reboot(port, jid)

    finally:
        port.closePort()

    # ── 최종 요약 ──
    ok = sum(1 for r in results if r[2] == "OK")
    print(f"\n{C.B}═══════════ 결과 요약 ═══════════{C.X}")
    print(f"  성공 {C.G}{ok}{C.X} / 전체 {len(results)}")
    for jid, name, st, detail in results:
        if st != "OK":
            print(f"  {C.R}- ID {jid} {name}: {st} {detail}{C.X}")
    print(f"{C.B}════════════════════════════════{C.X}")
    print(f"{C.D}* 토크는 OFF 상태로 종료했습니다. 모터 전원 OFF 후에도 EEPROM 값은 유지됩니다.{C.X}")
    return ok == len(results)


# ═════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="KHUBER 다이나믹셀(XM540-W150-R) 일괄 설정 도구 (U2D2 / Protocol 2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # 설정 소스: xlsx 한 개 또는 csv 두 개
    ap.add_argument("--config", help="통합 xlsx 경로 (시트: joints, common)")
    ap.add_argument("--joints-csv", help="관절 리밋 CSV 경로")
    ap.add_argument("--common-csv", help="공통설정 CSV 경로")
    # 연결
    ap.add_argument("--port", default="/dev/ttyUSB0", help="U2D2 시리얼 포트 (기본 /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=3000000, help="보레이트 (기본 3000000 = 3Mbps)")
    # 범위/동작 제어
    ap.add_argument("--ids", help="적용할 ID만 콤마로 (예: 1,2,3). 미지정 시 전체")
    ap.add_argument("--no-common", action="store_true", help="공통설정 기입 건너뛰기")
    ap.add_argument("--no-limits", action="store_true", help="Position Limit 기입 건너뛰기")
    ap.add_argument("--no-verify", action="store_true", help="쓰기 후 읽기 검증 건너뛰기")
    ap.add_argument("--reboot", action="store_true", help="각 모터 적용 후 reboot")
    ap.add_argument("--dry-run", action="store_true", help="하드웨어 없이 적용 계획만 출력")
    args = ap.parse_args()

    # 파생 플래그
    args.common_enabled = not args.no_common
    args.limits_enabled = not args.no_limits
    args.verify = not args.no_verify

    # ── 설정 로딩 ──
    if args.config:
        joints, common = load_from_xlsx(args.config)
    elif args.joints_csv:
        joints = load_joints_csv(args.joints_csv)
        common = load_common_csv(args.common_csv) if args.common_csv else {}
    else:
        # 인자 없으면 기본 설정 파일 자동 탐색 (ROS share → 현재 폴더)
        default = find_default_config()
        if default is None:
            sys.exit(f"{C.R}[오류]{C.X} 설정 파일을 찾을 수 없습니다. "
                     f"--config 또는 --joints-csv 로 지정하세요.")
        print(f"{C.D}[설정] 자동 선택: {default}{C.X}")
        if default.endswith(".xlsx"):
            joints, common = load_from_xlsx(default)
        else:
            joints = load_joints_csv(default)
            cpath = os.path.join(os.path.dirname(default), "khuber_common_settings.csv")
            common = load_common_csv(cpath) if os.path.exists(cpath) else {}

    # ── ID 필터 ──
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        joints = [j for j in joints if j["id"] in want]
        if not joints:
            sys.exit(f"{C.R}[오류]{C.X} --ids에 해당하는 관절이 없습니다.")

    # ── 검증 ──
    errs = validate(joints, common)
    if errs:
        print(f"{C.R}[설정 검증 실패]{C.X}")
        for e in errs:
            print(f"   - {e}")
        sys.exit(1)

    # ── 계획 출력 ──
    print_plan(joints, common, args.common_enabled, args.limits_enabled)

    if args.dry_run:
        print(f"{C.Y}[dry-run]{C.X} 하드웨어에 쓰지 않고 종료합니다.")
        return

    ok = apply_to_hardware(joints, common, args)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

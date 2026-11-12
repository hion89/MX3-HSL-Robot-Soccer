#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khuber_dxl_dump.py
==================
다이나믹셀(XM540-W150-R / U2D2 / Protocol 2.0)의 **현재 설정을 모터에서 직접 읽어**
khuber_dxl_setup 의 적용 계획과 동일한 포맷으로 보여주는 읽기 전용 도구.

읽는 항목 (모터마다):
  - Operating Mode(11)
  - Return Delay Time(9) / Max·Min Voltage Limit(32,34) / Velocity Limit(44)   ← 공통설정
  - Min·Max Position Limit(52,48)                                              ← 관절 리밋
  - Present Position(132)  ← 현재 위치(리밋 대비 어디에 있는지 확인용)
  - Hardware Error Status(70)

옵션으로 설정 파일(xlsx/csv)과 비교하여 불일치를 표시한다(기본 ON).
읽기 전용이므로 토크/EEPROM을 건드리지 않는다(안전).

사용 예:
  ros2 run humanoid_utils khuber_dxl_dump                      # 설치된 config로 ID/이름/비교
  ros2 run humanoid_utils khuber_dxl_dump --ids 1,2,3          # 특정 ID만
  ros2 run humanoid_utils khuber_dxl_dump --no-compare         # 비교 없이 값만
  ros2 run humanoid_utils khuber_dxl_dump --scan               # config 없이 1~30 핑 스캔
"""

import argparse
import os
import sys

# 같은 패키지의 setup 도구에서 상수/헬퍼 재사용 (설치/단독 실행 양쪽 대응)
try:
    from humanoid_utils.khuber_dxl_setup import (
        ADDR_RETURN_DELAY_TIME, ADDR_OPERATING_MODE, ADDR_MAX_VOLTAGE_LIMIT,
        ADDR_MIN_VOLTAGE_LIMIT, ADDR_VELOCITY_LIMIT, ADDR_MAX_POSITION_LIMIT,
        ADDR_MIN_POSITION_LIMIT, LEN_1, LEN_2, LEN_4, OP_MODE_POSITION,
        COMMON_SPEC, C, load_from_xlsx, load_joints_csv, load_common_csv,
        find_default_config,
    )
except ImportError:
    from khuber_dxl_setup import (  # noqa: F401 (단독 실행 시)
        ADDR_RETURN_DELAY_TIME, ADDR_OPERATING_MODE, ADDR_MAX_VOLTAGE_LIMIT,
        ADDR_MIN_VOLTAGE_LIMIT, ADDR_VELOCITY_LIMIT, ADDR_MAX_POSITION_LIMIT,
        ADDR_MIN_POSITION_LIMIT, LEN_1, LEN_2, LEN_4, OP_MODE_POSITION,
        COMMON_SPEC, C, load_from_xlsx, load_joints_csv, load_common_csv,
        find_default_config,
    )

ADDR_PRESENT_POSITION = 132
ADDR_HARDWARE_ERROR   = 70


def ticks_to_deg(min_pos, max_pos):
    """
    tick → 관절 각도(deg) 역산. setup 의 정변환:
        max_pos = ROUND((180-lower)*4096/360)+1
        min_pos = ROUND((180-upper)*4096/360)-1
    의 역. ±1·반올림 때문에 근사지만 정수 degree는 거의 정확히 복원된다.
    """
    lower = 180.0 - (max_pos - 1) * 360.0 / 4096.0
    upper = 180.0 - (min_pos + 1) * 360.0 / 4096.0
    return round(lower), round(upper)


# ═════════════════════════════════════════════════════════════════════════
# 렌더링 (순수 함수 — 하드웨어 없이 테스트 가능)
#   readings: [{id,name,op_mode,ret_delay,max_v,min_v,vel,min_pos,max_pos,
#               pres_pos,hw_err,alive}]
#   expect:   {id: {min,max}}  /  expect_common: {key: raw}  (비교용, 없으면 None)
# ═════════════════════════════════════════════════════════════════════════
def render(readings, expect=None, expect_common=None):
    out = []
    alive = [r for r in readings if r["alive"]]
    out.append(f"\n{C.B}══════════════ 모터 현재 설정 (읽기) ══════════════{C.X}")

    # ── 공통설정: 모터별 값이 동일해야 정상. 동일하면 1줄, 다르면 outlier 표시 ──
    out.append(f"{C.Y}[공통설정] (모터에서 읽은 값; 모든 모터 동일해야 정상){C.X}")
    field_map = {
        "return_delay_time": "ret_delay", "max_voltage_limit": "max_v",
        "min_voltage_limit": "min_v", "velocity_limit": "vel",
    }
    for key, (addr, _ln, scale, unit) in COMMON_SPEC.items():
        fld = field_map[key]
        vals = {r["id"]: r[fld] for r in alive if r[fld] is not None}
        if not vals:
            out.append(f"   - {key:<20} addr {addr:<3} = (읽기 실패)")
            continue
        # 최빈값 기준
        uniq = {}
        for v in vals.values():
            uniq[v] = uniq.get(v, 0) + 1
        common_val = max(uniq, key=uniq.get)
        outliers = [i for i, v in vals.items() if v != common_val]
        line = f"   - {key:<20} addr {addr:<3} = {common_val:<5} ({common_val*scale:g} {unit})"
        if len(uniq) == 1:
            line += f"  {C.D}[전체 동일]{C.X}"
        else:
            detail = ", ".join(f"ID{i}={vals[i]}" for i in outliers)
            line += f"  {C.R}[불일치: {detail}]{C.X}"
        # 설정과 비교
        if expect_common and key in expect_common and expect_common[key] != common_val:
            line += f"  {C.R}✗ cfg={expect_common[key]}{C.X}"
        out.append(line)

    # ── 관절별 Position Limit + 현재 위치 + 에러 ──
    out.append(f"{C.Y}[관절별 Position Limit] ({len(alive)}개 응답 / {len(readings)}개 요청){C.X}")
    out.append(f"   {'ID':>3}  {'joint':<24} {'min(52)':>8} {'max(48)':>8}   "
               f"{'deg(lo~up)':<11} {'pos(now)':>8}  {'err':>3}")
    mism = []
    for r in sorted(readings, key=lambda x: x["id"]):
        if not r["alive"]:
            out.append(f"   {r['id']:>3}  {r['name']:<24} {C.R}응답 없음{C.X}")
            continue
        lo, up = ticks_to_deg(r["min_pos"], r["max_pos"])
        deg = f"{lo}~{up}"
        err = "-" if r["hw_err"] in (0, None) else f"{r['hw_err']:#04x}"
        mark = ""
        if expect and r["id"] in expect:
            e = expect[r["id"]]
            if r["min_pos"] != e["min"] or r["max_pos"] != e["max"]:
                mark = f"  {C.R}✗ cfg {e['min']}/{e['max']}{C.X}"
                mism.append(r["id"])
        errc = C.R if (r["hw_err"] not in (0, None)) else ""
        out.append(f"   {r['id']:>3}  {r['name']:<24} {r['min_pos']:>8} {r['max_pos']:>8}   "
                   f"{deg:<11} {r['pres_pos']:>8}  {errc}{err:>3}{C.X if errc else ''}{mark}")

    out.append(f"{C.B}════════════════════════════════════════════════{C.X}")
    if expect is not None:
        ok = len(alive) - len(mism)
        if mism:
            out.append(f"  {C.R}설정과 불일치 {len(mism)}개: ID {','.join(map(str, mism))}{C.X} "
                       f"(일치 {ok}/{len(alive)})")
        else:
            out.append(f"  {C.G}설정과 전부 일치 ({ok}/{len(alive)}){C.X}")
    return "\n".join(out)


# ═════════════════════════════════════════════════════════════════════════
# 하드웨어 읽기
# ═════════════════════════════════════════════════════════════════════════
def read_all(port_name, baud, ids, names):
    try:
        from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
    except ImportError:
        sys.exit(f"{C.R}[오류]{C.X} dynamixel_sdk 없음: pip install dynamixel-sdk")

    port = PortHandler(port_name)
    packet = PacketHandler(2.0)
    if not port.openPort():
        sys.exit(f"{C.R}[오류]{C.X} 포트 열기 실패: {port_name}")
    if not port.setBaudRate(baud):
        port.closePort()
        sys.exit(f"{C.R}[오류]{C.X} 보레이트 설정 실패: {baud}")
    print(f"{C.G}[연결]{C.X} {port_name} @ {baud}bps (Protocol 2.0)")

    def rd(dxl_id, addr, length):
        if length == LEN_1:
            v, comm, err = packet.read1ByteTxRx(port, dxl_id, addr)
        elif length == LEN_2:
            v, comm, err = packet.read2ByteTxRx(port, dxl_id, addr)
        else:
            v, comm, err = packet.read4ByteTxRx(port, dxl_id, addr)
        return v if (comm == COMM_SUCCESS and err == 0) else None

    def to_signed32(u):
        if u is None:
            return None
        u &= 0xFFFFFFFF
        return u - 0x100000000 if (u & 0x80000000) else u

    readings = []
    try:
        for jid in ids:
            _, comm, err = packet.ping(port, jid)
            if comm != COMM_SUCCESS or err != 0:
                readings.append({"id": jid, "name": names.get(jid, f"id_{jid}"), "alive": False})
                continue
            readings.append({
                "id": jid,
                "name": names.get(jid, f"id_{jid}"),
                "alive": True,
                "op_mode":  rd(jid, ADDR_OPERATING_MODE, LEN_1),
                "ret_delay": rd(jid, ADDR_RETURN_DELAY_TIME, LEN_1),
                "max_v":    rd(jid, ADDR_MAX_VOLTAGE_LIMIT, LEN_2),
                "min_v":    rd(jid, ADDR_MIN_VOLTAGE_LIMIT, LEN_2),
                "vel":      rd(jid, ADDR_VELOCITY_LIMIT, LEN_4),
                "min_pos":  rd(jid, ADDR_MIN_POSITION_LIMIT, LEN_4),
                "max_pos":  rd(jid, ADDR_MAX_POSITION_LIMIT, LEN_4),
                "pres_pos": to_signed32(rd(jid, ADDR_PRESENT_POSITION, LEN_4)),
                "hw_err":   rd(jid, ADDR_HARDWARE_ERROR, LEN_1),
            })
    finally:
        port.closePort()
    return readings


# ═════════════════════════════════════════════════════════════════════════
def _load_config(args):
    """비교/이름용 설정 로드. 반환: (names{id:name}, expect{id:{min,max}}, expect_common{key:raw})"""
    path = args.config
    if path is None and not args.scan:
        path = find_default_config()
    if path is None:
        return {}, None, None
    if path.endswith(".xlsx"):
        joints, common = load_from_xlsx(path)
    else:
        joints = load_joints_csv(path)
        cpath = os.path.join(os.path.dirname(path), "khuber_common_settings.csv")
        common = load_common_csv(cpath) if os.path.exists(cpath) else {}
    print(f"{C.D}[설정] 비교 기준: {path}{C.X}")
    names = {j["id"]: j["name"] for j in joints}
    expect = {j["id"]: {"min": j["min"], "max": j["max"]} for j in joints}
    return names, expect, common


def _parse_ids(spec):
    """'1,2,3' 또는 '1-18' 또는 혼합 파싱."""
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


def main():
    ap = argparse.ArgumentParser(
        description="다이나믹셀 현재 설정 읽기/표시 (XM540-W150-R / U2D2)")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=3000000)
    ap.add_argument("--config", help="비교/이름 기준 설정 파일 (미지정 시 자동 탐색)")
    ap.add_argument("--ids", help="읽을 ID (예: 1,2,3 또는 1-18). 미지정 시 config의 ID")
    ap.add_argument("--scan", action="store_true", help="config 없이 1~30 핑 스캔")
    ap.add_argument("--no-compare", action="store_true", help="설정과 비교하지 않고 값만 표시")
    args = ap.parse_args()

    names, expect, expect_common = _load_config(args)

    # 읽을 ID 결정: --ids > config의 ID > scan(1~30)
    if args.ids:
        ids = _parse_ids(args.ids)
    elif expect:
        ids = sorted(expect.keys())
    else:
        ids = list(range(1, 31))
        print(f"{C.D}[스캔] config 없음 → ID 1~30 핑{C.X}")

    if args.no_compare:
        expect = None
        expect_common = None

    readings = read_all(args.port, args.baud, ids, names)
    print(render(readings, expect, expect_common))


if __name__ == "__main__":
    main()

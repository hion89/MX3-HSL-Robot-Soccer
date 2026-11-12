#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  test_buttons.py
#  -----------------------------------------------------------------------------
#  KHUBER 정지/시작 스위치 GPIO 연결 테스트 스크립트
#
#  목적:
#    Jetson Orin NX(reComputer J4012)의 40핀 헤더에 연결한 두 개의 물리
#    스위치(정지/시작)가 전기적으로 제대로 연결되었는지, 그리고 누름/뗌이
#    소프트웨어에서 정확히 감지되는지를 확인하기 위한 진단용 스크립트다.
#    실제 로봇 제어 로직(torque-off, FSM 전환 등)은 들어있지 않다.
#    오로지 "배선 검증"이 목적이다.
#
#  하드웨어 전제(이번 배선 기준):
#    - 정지(STOP) 스위치  -> 40핀 헤더 11번 핀
#    - 시작(START) 스위치 -> 40핀 헤더 13번 핀
#    - 두 스위치 공통 한쪽은 GND(예: 6번 또는 9번 핀)에 연결
#    - 외부 풀업 저항(예: 10kΩ)을 3.3V(1번 핀) ~ 각 GPIO 사이에 연결
#      (Jetson 내부 풀업/풀다운이 동작하지 않아 외부 풀업으로 구성했기 때문)
#
#  전기적 동작(active-low):
#    - 스위치를 누르지 않은 평상시:
#        외부 풀업이 GPIO를 3.3V로 끌어올림 -> 입력값 HIGH(1)
#    - 스위치를 누른 순간:
#        GPIO가 GND와 직접 연결됨           -> 입력값 LOW(0)
#    => 즉 "눌림 = LOW(0)", "뗌 = HIGH(1)" 이다. (active-low 로직)
#
#  핀 번호 체계:
#    jetson-gpio 라이브러리의 GPIO.BOARD 모드를 사용한다.
#    BOARD 모드는 40핀 헤더의 물리적 핀 번호(1~40)를 그대로 쓰므로,
#    BCM/Tegra 번호로 변환할 필요 없이 11, 13을 그대로 지정하면 된다.
#
#  주의:
#    - 반드시 3.3V 신호만 사용할 것. 5V를 GPIO에 직접 인가하면 Jetson 모듈이
#      영구 손상될 수 있다.
#    - 외부 풀업을 이미 달았으므로, 코드에서는 내부 풀업/풀다운을 끈
#      상태(pull_up_down=GPIO.PUD_OFF)로 입력을 설정한다. 내부/외부 풀업이
#      동시에 걸리면 전압 분배가 어긋날 수 있어서다.
#
#  실행:
#    sudo python3 test_buttons.py
#    (GPIO 접근에 권한이 필요할 수 있어 sudo 권장. 종료는 Ctrl+C)
# =============================================================================

import sys
import time

# Jetson.GPIO 라이브러리 import.
# NVIDIA 공식 라이브러리이며 RPi.GPIO와 API가 거의 동일하다.
# 설치가 안 되어 있으면 친절한 안내 메시지를 띄우고 종료한다.
try:
    import Jetson.GPIO as GPIO
except ImportError:
    print("[오류] Jetson.GPIO 라이브러리를 찾을 수 없습니다.")
    print("       다음 명령으로 설치하세요:")
    print("       sudo pip3 install Jetson.GPIO")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 설정 상수
# -----------------------------------------------------------------------------

# 40핀 헤더의 물리 핀 번호 (BOARD 모드 기준).
PIN_STOP = 11   # 정지 스위치
PIN_START = 13  # 시작 스위치

# 스위치 이름을 핀 번호로 매핑. 루프에서 두 핀을 동일하게 다루기 위함.
BUTTONS = {
    "STOP (정지)": PIN_STOP,
    "START (시작)": PIN_START,
}

# 소프트웨어 디바운스 시간(초).
# 물리 스위치는 누르는 순간 수 ms 동안 접점이 떨리는(채터링) 현상이 있다.
# 이 시간보다 짧은 간격의 연속 상태 변화는 같은 한 번의 동작으로 간주해
# 중복 카운트를 막는다.
DEBOUNCE_SEC = 0.05  # 50ms

# 폴링 주기(초). 너무 짧으면 CPU 점유가 늘고, 너무 길면 반응이 둔해진다.
POLL_SEC = 0.005  # 5ms (200Hz) -> 사람 손가락 동작 감지에 충분


def read_state_str(value):
    """
    GPIO 입력값(0/1)을 사람이 읽기 쉬운 문자열로 변환한다.

    active-low 배선이므로:
      - value == GPIO.LOW  (0) -> 스위치가 눌린 상태 "PRESSED"
      - value == GPIO.HIGH (1) -> 스위치가 떨어진 상태 "released"
    """
    if value == GPIO.LOW:
        return "PRESSED (눌림)"
    else:
        return "released (뗌)"


def main():
    # -------------------------------------------------------------------------
    # 1) GPIO 모드 설정
    # -------------------------------------------------------------------------
    # BOARD 모드: 40핀 헤더의 물리 핀 번호를 그대로 사용.
    # 이 줄이 없으면 이후 setup 호출에서 에러가 난다.
    GPIO.setmode(GPIO.BOARD)

    # 이전 실행이 비정상 종료되어 핀이 정리(cleanup)되지 않았을 경우의
    # "이미 사용 중" 경고를 끈다. 진단 스크립트라 무해하다.
    GPIO.setwarnings(False)

    # -------------------------------------------------------------------------
    # 2) 입력 핀 설정
    # -------------------------------------------------------------------------
    # 외부 풀업을 이미 달았으므로 내부 풀업/풀다운은 끈다(PUD_OFF).
    # 두 핀 모두 입력(IN)으로 설정한다.
    for name, pin in BUTTONS.items():
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_OFF)

    # -------------------------------------------------------------------------
    # 3) 시작 안내 및 초기 상태 출력
    # -------------------------------------------------------------------------
    print("=" * 60)
    print(" KHUBER 정지/시작 스위치 GPIO 연결 테스트")
    print("=" * 60)
    print(f"  정지(STOP)  : 핀 {PIN_STOP}  (BOARD)")
    print(f"  시작(START) : 핀 {PIN_START}  (BOARD)")
    print("  배선: 외부 풀업(active-low) -> 평상시 HIGH, 누르면 LOW")
    print("-" * 60)
    print("  스위치를 눌렀다 떼면서 상태 변화가 찍히는지 확인하세요.")
    print("  종료: Ctrl+C")
    print("=" * 60)

    # 각 핀의 "직전 상태"와 "마지막으로 상태가 바뀐 시각"을 기억하는 딕셔너리.
    # 디바운스 처리와 변화 감지(edge detection)에 사용한다.
    last_state = {}      # pin -> 직전에 확정된 입력값(0/1)
    last_change_t = {}   # pin -> 마지막 상태 변화가 인정된 시각(time.monotonic)
    press_count = {}     # pin -> 눌림 횟수 카운터(연결/디바운스 확인용)

    # 시작 시점의 현재 값을 읽어 초기 상태로 저장하고 화면에 보여준다.
    # 여기서 모든 핀이 HIGH(released)로 나와야 정상이다.
    # 만약 평상시인데 LOW로 찍히면 -> 배선이 항상 GND에 붙어있거나 풀업
    # 미동작일 가능성이 크다. (배선 점검 신호)
    now = time.monotonic()
    for name, pin in BUTTONS.items():
        v = GPIO.input(pin)
        last_state[pin] = v
        last_change_t[pin] = now
        press_count[pin] = 0
        print(f"  [초기] {name:14s} = {read_state_str(v)}")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # 4) 메인 폴링 루프
    # -------------------------------------------------------------------------
    # edge_trigger를 인터럽트(add_event_detect)로도 할 수 있지만, 외부 풀업이
    # 갓 구성된 단계에서는 "지금 이 핀이 실제로 무슨 값을 내는지"를 눈으로
    # 직접 확인하는 폴링 방식이 디버깅에 더 직관적이라 폴링으로 작성했다.
    try:
        while True:
            now = time.monotonic()

            for name, pin in BUTTONS.items():
                # 현재 핀 값을 읽는다.
                v = GPIO.input(pin)

                # 직전에 확정된 값과 다르면 "상태가 바뀌려는 후보"다.
                if v != last_state[pin]:
                    # 디바운스: 마지막 확정 변화 이후 DEBOUNCE_SEC가
                    # 지났을 때만 진짜 변화로 인정한다. 그 안쪽이면
                    # 채터링으로 보고 무시한다.
                    if (now - last_change_t[pin]) >= DEBOUNCE_SEC:
                        last_state[pin] = v
                        last_change_t[pin] = now

                        # 새 상태를 화면에 출력.
                        state_str = read_state_str(v)

                        # LOW(눌림)로 떨어지는 순간(하강 에지)을 한 번의
                        # 유효한 "누름"으로 카운트한다.
                        if v == GPIO.LOW:
                            press_count[pin] += 1
                            print(f"  >>> {name:14s} {state_str}"
                                  f"   (누적 누름: {press_count[pin]}회)")
                        else:
                            print(f"      {name:14s} {state_str}")

            # 다음 폴링까지 잠깐 쉰다. CPU 100% 점유 방지.
            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        # Ctrl+C로 정상 종료할 때 줄바꿈 한 번 넣어 깔끔하게 마무리.
        print("\n" + "-" * 60)
        print("  테스트 종료. 누적 누름 횟수:")
        for name, pin in BUTTONS.items():
            print(f"    {name:14s} : {press_count[pin]}회")
        print("-" * 60)

    finally:
        # 어떤 경로로 끝나든 반드시 GPIO 리소스를 정리한다.
        # 정리하지 않으면 다음 실행 때 "이미 사용 중" 경고가 뜬다.
        GPIO.cleanup()
        print("  GPIO 정리 완료.")


if __name__ == "__main__":
    main()
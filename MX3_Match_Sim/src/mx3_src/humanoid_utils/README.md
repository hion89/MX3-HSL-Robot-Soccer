# humanoid_utils

휴머노이드 공용 유틸리티 ROS 2 패키지(ament_python). 현재 포함된 도구:

- **`khuber_dxl_setup`** — 다이나믹셀(XM540-W150-R / U2D2 / Protocol 2.0)에 관절별 Position Limit + 공통설정(Return Delay / Voltage / Velocity)을 일괄 기입.

## 설계 의도 (이식성)
- **툴(파이썬)** 은 로봇 비종속 → 워크스페이스째 다른 로봇으로 옮겨도 그대로 재사용.
- **설정(xlsx/csv)** 은 로봇마다 다름 → `config/` 에 로봇별로 두고 교체.
- 따라서 새 로봇 이식 시: 워크스페이스 복사 → `config/` 의 설정값만 그 로봇에 맞게 수정.

## 빌드
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select humanoid_utils
source install/setup.bash
```

## 실행
```bash
# 설치된 기본 config(share/humanoid_utils/config) 자동 사용
ros2 run humanoid_utils khuber_dxl_setup --dry-run        # 계획만 확인
ros2 run humanoid_utils khuber_dxl_setup                  # 실제 적용

# 다른 설정/포트 지정
ros2 run humanoid_utils khuber_dxl_setup \
    --config /path/to/other_robot_config.xlsx --port /dev/ttyUSB0 --baud 3000000

# 일부 ID만 / 공통설정 제외 / 적용 후 리부트
ros2 run humanoid_utils khuber_dxl_setup --ids 1,2,3 --no-common --reboot
```
옵션: `--config` `--joints-csv` `--common-csv` `--port` `--baud` `--ids` `--no-common` `--no-limits` `--no-verify` `--reboot` `--dry-run`

> ROS 없이 단독 실행도 가능: `python3 humanoid_utils/khuber_dxl_setup.py --config config/khuber_motor_config.xlsx --dry-run`

## 현재 설정 읽기 (`khuber_dxl_dump`)
모터 EEPROM을 직접 읽어 `khuber_dxl_setup` 과 동일한 포맷으로 보여주고, config와 비교해 불일치를 표시합니다(읽기 전용, 안전).
```bash
ros2 run humanoid_utils khuber_dxl_dump                 # config 기준 ID/이름/비교
ros2 run humanoid_utils khuber_dxl_dump --ids 1-18      # ID 범위
ros2 run humanoid_utils khuber_dxl_dump --no-compare    # 비교 없이 값만
ros2 run humanoid_utils khuber_dxl_dump --scan          # config 없이 1~30 핑 스캔
```
표시: 공통설정(모터 전체 동일 여부 + outlier), 관절별 min/max tick, 역산 degree, 현재위치(pos), 하드웨어 에러, config 대비 불일치(✗).

## config 편집 규칙
`config/khuber_motor_config.xlsx` 의 **노란 입력칸만** 수정합니다.
- `joints` 시트: `lower_deg` / `upper_deg` 입력 → tick 자동 계산(잠김)
- `common` 시트: `value` 에 공학단위(V/us/rev·min) 입력 → raw 자동 계산(잠김)

스크립트는 각도에서 tick을 **직접 재계산**하므로 Excel 수식 캐시 유무와 무관하게 동작합니다.

## ⚠ 다른 로봇 이식 시 주의
degree→tick 변환식은 KHUBER의 서보 영점 규약(`servo 180° = joint 0°`, 방향 반전)을 가정합니다:
```
max_position = ROUND((180 - lower_deg) * 4096/360) + 1
min_position = ROUND((180 - upper_deg) * 4096/360) - 1
```
서보 조립 영점/방향이 다른 로봇은 이 식(또는 관절별 offset·부호)을 조정해야 합니다.
지금은 18개 관절 모두 동일 규약이라 단일 식으로 처리합니다.

## 컨트롤 테이블 (XM540-W150-R, e-manual 검증)
| Addr | 항목 | Bytes | Unit |
|---|---|---|---|
| 9 | Return Delay Time | 1 | 2 µs/LSB |
| 32 | Max Voltage Limit | 2 | 0.1 V/LSB |
| 34 | Min Voltage Limit | 2 | 0.1 V/LSB |
| 44 | Velocity Limit | 4 | 0.229 rev/min/LSB |
| 48 | Max Position Limit | 4 | 1 pulse (0~4095) |
| 52 | Min Position Limit | 4 | 1 pulse (0~4095) |
| 64 | Torque Enable | 1 | EEPROM 쓰기엔 0 필수 |

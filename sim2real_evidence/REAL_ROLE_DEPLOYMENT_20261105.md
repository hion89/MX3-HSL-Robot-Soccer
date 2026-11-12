# 최우선: robo04 한 대에서 실제 공격·수비·골키퍼 수행

정책 재검증을 반복하지 않고 제공된 가중치와 B-Human 이식 전략을 실제 제어에 연결한다.
현재 적용/빌드/실물 수행은 아직 사용자 실행 전이다.

## 유지할 창

1 하드웨어, 2 카메라+TF, 3 YOLO, 4 leg_odometry, 5 ball_tracker, 6 self_localizer는 유지한다.
기존7번 shadow행동만 Ctrl+C로 종료한다. 새7번은 실물 역할 launch로 사용한다.
8번은 빌드/역할 전환용이다.

## Windows 전송

```powershell
cd C:\Users\jes\Desktop\isaac_ws\sim2real_evidence\deploy
scp .\real_roles_20261105.tar.gz .\real_roles_20261105.tar.gz.sha256 robo04@192.168.1.161:/home/robo04/incoming/
```

## 로봇8번: 적용 및 관련4패키지 빌드

```bash
(
set -e
cd /home/robo04/incoming
sha256sum -c real_roles_20261105.tar.gz.sha256
tar -xzf real_roles_20261105.tar.gz
bash real_roles_20261105/apply_and_build.sh
)
```

기립/위치 초기화는 어제와 동일하게 한다. START는 /initialpose 입력보다 먼저 누른다.
모터를 껐거나 손으로 재배치했다면 START로 기립시킨 뒤 실제 배치의 초기 위치를 다시 넣는다.

## 로봇7번: 실제 제어 실행

```bash
bash /home/robo04/incoming/real_roles_20261105/start.sh
```

모델의 정지 관절 목표가 실제 모터에 연결된다. 처음 상태는 INITIAL/gait stop이다.
보행v0_6_13500과 킥v1_5b_22300을 사용한다. CPU provider, 속도상한 .20/.15/.50으로 시작한다.

## 로봇8번: 역할 실행/전환

```bash
bash /home/robo04/incoming/real_roles_20261105/role.sh attacker --seconds 10
bash /home/robo04/incoming/real_roles_20261105/role.sh defender --seconds 10
bash /home/robo04/incoming/real_roles_20261105/role.sh goalkeeper --seconds 15
```

처음에는 공격10초만 실행하고 실제 공 접근/정렬/킥 여부를 관찰한다.
이후 수비/골키퍼를 실행한다. 역할은 같은 로봇 번호4를 유지하며 명시적으로 전환한다.

```bash
bash /home/robo04/incoming/real_roles_20261105/role.sh stop
bash /home/robo04/incoming/real_roles_20261105/role.sh status
```

stop은 행동/보행/킥을 정지시키고 기립 토크를 유지한다.
시간 옵션 없이 실행하면 stop 명령을 줄 때까지 계속 역할을 수행한다.
정지는 역할 명령용 서비스 호출이며 물리 비상 STOP(토크OFF)을 대체하지 않는다.

## 구현 및 검증 기록

role trial은 기존 센서/월드모델을 중복 실행하지 않고 실제 제어와 원본 낙상 처리기를 추가한다.
역할은 원본 공 처리 스킬/수비 pose/골키퍼 pose를 사용한다.
수비 거리 상수는3.9m 지도 비율로 맞췄다. 가중치/게인/관절 순서는 변경하지 않았다.
입력 단절시 명령캐시가 남는 확인된 문제는0.5초 수신 시간 보호로 처리한다.
킥 정지는motion_enabled=false로 현재 킥을 종료하고 보행으로 소유권을 돌려준다.

10개 역할/시작정지 회귀 검사,5개 적용복원 검사,7개 watchdog 검사,
Python3.10/Bash 문법과 압축파일 readback을 확인했다. 실물 동작은 아직 미확인이다.

배포 SHA256: 5f876eb86737a27a4179955f2d799ec1ef48703b0c05b6b3bbf37ff518462235.
묶음은 patches/real_roles_20261105, 전달본은 deploy/real_roles_20261105.tar.gz.

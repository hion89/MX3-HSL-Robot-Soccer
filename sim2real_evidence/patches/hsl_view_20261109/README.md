# 현재 HSL 추정 위치를 MobaXterm X11 + RViz로 보기

로봇에서 RViz를 실행하고 Windows에는 창만 전달한다. ROS_DOMAIN_ID=42,
ROS_LOCALHOST_ONLY=1을 그대로 유지한다. 기존 기본 실행/보행/전략을 재시작하거나
소스를 덮어쓰거나 빌드하지 않는다. 관찰 노드는 /hsl/robot_pose와 /hsl/ball만 구독하고
/hsl/view/markers만 발행한다. 모터·속도·초기 위치·TF는 발행하지 않는다.

## 1. MobaXterm에서 관찰용 새 SSH 연결

기존 제어 터미널은 그대로 유지한다. 저장된 SSH 세션 설정의 Advanced SSH settings에서
X11-forwarding을 켜고 관찰용 새 연결을 연다. MobaXterm X server도 켜져 있어야 한다.
일반 Tunneling 목록에 ROS용 TCP 포트를 추가할 필요는 없다.

```bash
echo "$DISPLAY"
```

`localhost:10.0`처럼 값이 나와야 한다. 번호는 접속마다 다를 수 있다.
빈 값이면 해당 새 SSH 세션의 X11-forwarding 설정을 확인한다.
DISPLAY를 임의 IP나 :0으로 덮어쓰지 않는다.

## 2. 파일 전송 — Windows PowerShell

```powershell
cd C:\Users\jes\Desktop\isaac_ws\sim2real_evidence\deploy
scp .\hsl_view_20261109.tar.gz .\hsl_view_20261109.tar.gz.sha256 robo04@192.168.1.161:/home/robo04/incoming/
```

## 3. 새 MobaXterm 로봇 터미널에서 실행

```bash
(
set -e
cd /home/robo04/incoming
sha256sum -c hsl_view_20261109.tar.gz.sha256
tar -xzf hsl_view_20261109.tar.gz
bash hsl_view_20261109/start.sh
)
```

RViz가 설치되지 않았다면 스크립트가 안내하고 종료한다. 그때 한 번만:

```bash
sudo apt install ros-humble-rviz2
bash /home/robo04/incoming/hsl_view_20261109/start.sh
```

OpenGL/GLX 오류로 창이 열리지 않으면 오류 내용을 확인해야 한다.
MobaXterm은 Settings → Configuration → X11에서 OpenGL Hardware/Software 설정을 제공하지만,
RViz의 요구 OpenGL 버전과 원격 그래픽 환경에 따라 성공 여부가 달라진다.
MobaXterm 공식 문서: https://mobaxterm.mobatek.net/documentation.html

## 화면 해석

- 흰 선: 현재 HSL 추정기가 사용하는 연습장 지도(3.9x2.6m).
- 노란 점: 지도상 골포스트 위치. 왼쪽 OUR GOAL, 오른쪽 OPP GOAL.
- 파란 화살표: /hsl/robot_pose의 추정 위치와 몸통 방향.
- 청록색 선: 관찰 시작 이후 추정 위치의 궤적. 위치 보정 점프도 선에 반영된다.
- 주황색 공: /hsl/ball의 지도상 추정 위치. 관측 경과 시간도 표시한다.
- 상단: x/y[m], yaw[deg], quality, validity, 이번 주기 사용 랜드마크 수, 수신 경과 시간.
- 로봇 위치 수신이 1초 넘게 끊기면 화살표가 주황색으로 바뀌고 STALE을 표시한다.
- 공을 마지막으로 본 시점은 ball seen 값으로 판단한다. valid=true인 공도 예측 상태일 수 있다.

Frame은 map으로 고정된다. 창에서 위치를 재설정하는 도구는 넣지 않았다.
관찰 창 종료는 이 새 터미널에서 Ctrl+C로 한다. 역할 정지나 토크 OFF를 대신하지 않는다.

## 정확도 확인 순서

기존 시작 위치와 초기 좌표는 유지한다. RViz를 켰다는 이유로 init_pose.sh를 다시 실행하지 않는다.
이미 로봇이 이동했다면 현재 위치에 시작 좌표를 덮어쓰지 않는다.

1. 역할 stop으로 움직임을 멈춘 상태에서 실제 로봇과 화면의 위치/정면 방향을 대조한다.
2. 경기장 중앙 기준 몸통 중심 바닥 투영점의 실제 좌표를 측정하고 화면 x/y와 비교한다.
   +X는 상대 골대, +Y는 우리 골대에서 상대 골대를 볼 때 왼쪽이다.
   위치 오차는 sqrt((추정x-실측x)^2+(추정y-실측y)^2)이다.
3. 이후 공 접근을 관찰할 때 실제 이동 방향과 화살표·궤적이 같은지,
   라인을 가로지를 때 지도에서도 같은 라인을 지나는지 본다.

quality=2, 높은 validity, 부드러운 궤적은 추정기의 내부 상태다.
실제 위치와 맞는다는 보장은 아니므로 측정한 위치와 대조한다.

## 확인된 지도 불일치 — 추정 소스는 변경하지 않음

현재 HSL FieldModel/Field와 기존 humanoid_localization의 practice 시각화는
외곽 3.9x2.6m는 같지만 일부 내부 치수가 다르다:

| 항목 | 현재 HSL 추정 지도 | 기존 시각화 지도 |
|---|---:|---:|
| 센터서클 반지름 | 0.325m | 0.35m |
| 페널티박스 폭 | 약 1.733m | 2.17m |
| 골대 폭 | 0.78m | 1.13m |

이 뷰어는 설치된 hsl_behavior.geometry.Field를 직접 사용해 현재 HSL 지도대로 그린다.
외곽 길이/폭만으로 지도 일치를 판단하지 않는다. 화면의 박스/골포스트가 실제 라인/기둥과 다르면
실측 후 HSL 추정 지도와 전략 지도를 함께 맞춰야 한다. 이번 관찰 묶음은 지도를 변경하지 않는다.

로컬 Python3.10 문법, Bash 구문, YAML, 지도 좌표/패키지 무결성을 검사했다.
실제 RViz/X11 창의 실행은 로봇에서 확인 전이다.

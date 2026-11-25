# Phase 5B-0 오프라인 기록·분석 도구

이 도구에는 모터 publisher/service 호출/launch/원격 접속 기능이 없다. 기존 bag·CSV를 읽고 별도 결과를 만들며, 추후에만 명시적으로 선택하는 passive ROS 구독 adapter를 포함한다. **이번 단계에서 ROS 구독기를 실행하지 않았다.**

## 구성

- `phase5b0.py`: study/template/init/append/archive/import-gt/seal/validate/analyze/capture-ros CLI.
- `phase5b0_verify.py`: 원본 재추출·동일 결과 재생성·기존 회귀 테스트를 실제 실행하는 별도 검증기.
- `src/cah_mappo/measurement/schema.py`: 명시적 결측·clock·단위·프레임·GT provenance·동기 지연 계약.
- `storage.py`: exclusive 생성/잠금/append/봉인/hash·원본 사본 보관. 기존 파일·record ID 충돌을 거부한다.
- `importers.py`: SQLite 읽기 전용과 기존 rosbags를 이용한 CDR 해독, shadow CSV 변환. 원래 메시지 ID 순서를 보존한다.
- `analysis.py`: 간격·지터·긴 간격 후보, command sample-held 구간, 추정 속도 분포, 소프트웨어 상태·event 시간. 없는 GT/통신/목표 자료는 null로 보고한다.
- `validation.py`: 결측·단위·프레임·시각 역전·동기·봉인 검사와 가동 전 미확인 항목 보고.
- `collection.py`: 조건 manifest template, GT CSV/수동 record 수집, passive callback의 공통 변환. actual ROS adapter는 구독만 하며 별도 opt-in이 없으면 ROS import 전 거부한다.
- `workflow.py`: 두 실물 bag/shadow/콘솔·영상 metadata/운영 매뉴얼을 새 폴더에서 조사한다.
- `tests/test_phase5b0.py`: 합성 데이터 검증. 실측 결과가 아니다.

상세 데이터 계약은 [기록 규격](PHASE5B0_SCHEMA_KO.md), 실제 분석·시험 결과는 [결과 보고](PHASE5B0_RESULTS_KO.md), 다음 장비 준비는 [Phase5B-1 확인표](PHASE5B1_PREFLIGHT_KO.md)에 있다.

## 기존 환경과 실제 실행 명령

핵심 규격/검증/JSONL/CSV 도구는 Python 표준 라이브러리다. bag decoder만 기존 `C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/.analysis_env`의 Python3.12.14/rosbags0.10.11을 사용했다. 새 패키지는 설치하지 않았다. research 전용 `.venv`는 기존 CPU 회귀 테스트에 사용했다. 새 PC에 해당 dependency가 없다면 설치를 자동 시도하지 않고 별도 승인이 필요하다.

아래는 실제 실행한 명령이다. 같은 출력 경로에 다시 실행하면 덮어쓰지 않고 거부한다.

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim
& C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/.analysis_env/Scripts/python.exe -B research/cah_mappo/phase5b0.py study --output research/cah_mappo/phase5b0_outputs/study_v2
& C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/.analysis_env/Scripts/python.exe -B research/cah_mappo/phase5b0_verify.py --study research/cah_mappo/phase5b0_outputs/study_v2 --output research/cah_mappo/phase5b0_outputs/verification_v2 --test-python research/cah_mappo/.venv/Scripts/python.exe
```

재현할 때는 새 study/replay 경로를 지정한다. 최종 근거는 study_v2/verification_v2이며 관절 이름별 구분과 미확인 반복 번호의 명시적 null을 반영했다. v1은 개발 중간 기록으로 보존했고 기존 파일을 덮어쓰지 않았다. 통계의 원본 근거는 각 dataset의 manifest/source hash와 records의 source.locator다. 분석만 반복하려면 동일 dataset을 입력하고 새로운 output 폴더로 `analyze`한다.

```powershell
research/cah_mappo/.venv/Scripts/python.exe -B research/cah_mappo/phase5b0.py analyze --session research/cah_mappo/phase5b0_outputs/study_v2/kick_dataset --output research/cah_mappo/phase5b0_outputs/kick_reanalysis_NEW
```

회귀 테스트 명령:

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim\research\cah_mappo
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

## 추후 사용할 수집 인터페이스

현재 검증된 template은 최종 completion 출력의 `session_template.json` 또는 `template` 명령에서 새 경로로 생성한다. 실험 ID/robot/trial/반복·조건·clock·frame·안전 담당/수치 출처를 채운 뒤 `init --session NEW --manifest FILE`로 exclusive session을 만든다. template의 null 안전값을 임의 수치로 채우지 않는다.

`append --session ... --input records.jsonl`은 검증된 수동/외부 event를 추가한다. `import-gt`는 timestamp_ns,x_m,y_m,yaw_rad CSV와 clock/frame/device/calibration 선언을 요구한다. 이 도구는 자체적으로 GT 장비를 교정하거나 모든 pose를 자동 동기화하지 않는다. `archive`는 원본 파일을 hash 이름의 사본으로 보관하고 `seal`은 더 이상의 append를 막는다. `validate --output NEW.json`은 비정상 기록을 exit2로 보고한다.

`capture-ros`는 ROS2와 요청한 custom message가 설치된 환경에서만 동작하도록 준비했다. 명시적 `--allow-passive-subscriptions`, bounded duration과 기존 session이 필요하다. command/sensor/상태·TF·CameraInfo/Detection metadata를 구독하고, static TF에는 transient-local QoS를 요청한다. raw 영상·전체 CDR·kernel packet capture는 별도 원본 recorder가 필요하다. source acquire/publish/use 시각과 packet seq를 만들어 내거나 내부 hook을 자동 추가하지 않는다. 입력 header가 취득 시각인지도 자동 보증하지 않는다.

실물에서의 import/QoS/record 부하/종료 동작은 **미실행·미검증**이다. clock/GT/단위 등 오프라인 변환만 합성 테스트와 기존 자료로 검증했다. 현 단계에서는 capture 명령을 실행하지 않는다. 구독기라고 해도 P0의 장비·안전·부하 확인과 사용자 승인이 필요하다.

## 해석 제한

긴 간격 후보는2.5×해당 topic의 양의 간격 중앙값 기준이며 표본 유실의 확정이 아니다. record time이 몰리는 현상과 실제 sensor/header 주기를 구분한다. header 역전도 mixed stream/늦은 source 시각일 수 있다. event topic의2개 메시지 평균 빈도를 송신 주기로 해석하지 않는다.

과거 두 bag의 `/cmd_vel`이0이라는 사실만으로 전체 로봇에 유효한 이동 명령이 없거나 정지했다고 결론내리지 않는다. 킥 러너가 고정 내부 명령을 쓰는 경로가 있으며 zero Twist와 실제 킥 동작은 공존할 수 있다. 내부 명령·실제 운동 GT가 없으면 해당 항목은 결측이다.

unit/coordinate/sync 오류를 검출하는 도구가 물리 안전/장치 정확성을 승인하는 것은 아니다. 신뢰할 수 있는 수치 안전 기준·독립 GT·시각 연결과 실제 ROS 환경 검증이 준비되기 전에는 이동 측정을 시작할 수 없다.

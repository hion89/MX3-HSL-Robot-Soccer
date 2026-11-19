# CAH-MAPPO Phase 1 — 최소 환경·원시 로그·독립 평가

이 폴더만 신규 생성했다. 기존 ROS/모델/통신 코드를 사용하거나 변경하지 않는다. Python 3.10 이상 표준 라이브러리만 사용하며 패키지 설치가 필요 없다. MAPPO/CAH-MAPPO 학습기, PettingZoo, PyTorch, ROS, Isaac은 포함하지 않는다.

## 실행

PowerShell에서 모듈 디렉터리로 이동한다.

```powershell
cd C:\Users\jes\Desktop\MX3_Match_Sim\research\cah_mappo
python -B -m unittest discover -s tests -v
python -B run.py run --config configs/minimal_2robots.json --output outputs/my_run.jsonl
python -B run.py evaluate outputs/my_run.jsonl
python -B run.py suite --seed 0 --output outputs/my_suite
```

`python` 실행 별칭이 동작하지 않는 PC에서는 실제 Python 경로로 대체한다. 이번 검증에서는 `C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/.analysis_env/Scripts/python.exe`(Python 3.12.14)를 사용했다. 별도 환경이나 의존성은 설치하지 않았다.

출력 파일/묶음 폴더는 **새 이름**을 사용한다. 기존 출력은 덮어쓰거나 삭제하지 않고 오류로 중단한다. `-B`는 bytecode 생성을 막는다.

단일 시나리오 실행 예:

```powershell
python -B run.py run --scenario urgent --policy immediate --seed 0 --output outputs/urgent_immediate_new.jsonl
```

`--config`를 사용할 때 설정 파일은 완전히 해석된 환경 설정이다. 함께 지정한 `--policy`, `--seed`, `--scenario`만 그 필드를 덮는다. 이때 scenario 이름만 바꿔도 deadline/delay가 자동으로 바뀌는 것은 아니다. 설정 파일 없이 `--scenario`를 선택하면 내장 시나리오의 전체 설정을 사용한다. 실제 실행한 설정은 로그 header에 저장된다.

## 구성과 역할

- `src/cah_mappo/core.py`
  - `Config`: 시간·위치·속도·고장·통신 설정 검증.
  - `scenario_config()`: 다섯 내장 시나리오 설정 생성.
  - `Robot`: 기본 역할과 현재 작업, 위치·행동 가능 여부·용량.
  - `can_respond()`: 위치와 작업 용량에 따른 대응 가능성.
  - `Simulator`: 사건·통신·상태 갱신과 네 스크립트 정책 실행.
  - `write_jsonl()`: 기존 파일을 보호하는 배타적 로그 생성.
- `src/cah_mappo/metrics.py`
  - `load_jsonl()`: 원시 JSONL 읽기.
  - `evaluate()`: 시뮬레이터 인스턴스/보상/정책 준비 플래그 없이 로그의 Ground Truth와 사건에서 지표 재계산.
- `run.py`: 단일 실행, 20개 조합 suite, 로그만 읽는 evaluate 명령.
- `configs/minimal_2robots.json`: 명시적 기본 설정.
- `tests/test_phase1.py`: 해석적으로 정한 기대값 및 회귀 검사.
- `pyproject.toml`: 런타임 외부 의존성 없음. 실행을 위해 pip install을 할 필요 없다. 선택적 패키징을 실행한다면 별도 build 도구가 필요할 수 있다.

## 환경 정의

연속 2D 좌표를 일정 시간 간격으로 갱신한다. 기본 dt=1초, 전체 평가 구간 [0,16)초다. 로봇의 방향·관절·가속도·충돌·경로 장애물은 모델링하지 않는다. 예시는 x축 배치지만 이동 계산은 2D이며 대각 이동의 속도 제한도 테스트했다.

- 기본 책임 구역 중심 (0,0), 허용 거리 .25m.
- A는 (0,0)에서 기본 역할 base_guard, 현재 작업 base로 시작한다.
- B는 (-4,0)에서 기본 역할 reserve, 현재 작업 없음으로 시작한다.
- 각 로봇의 용량은 1이며 동시에 하나의 서비스에만 투입된다.
- 속도는 1m/s. 목표를 향해 직선 이동한다.
- t=2초에 동적 작업이 (4,0)에 발생한다. 작업에는 정지 상태 서비스 2초가 필요하다.
- 기본 마감은 절대시각 t=14초다. 정확히 마감에 완료하면 성공이다. 마감에 도착만 하고 서비스하지 못하면 응답으로도 계산하지 않는다.
- 새 작업을 완료하면 A는 idle 상태가 되며, 이 최소 정책은 자동 복귀까지 수행하지 않는다.

동적 작업 수행 중인 로봇은 구역과 같은 위치에 있어도 동시에 base를 커버하지 못한다. idle 또는 base에 투입된 행동 가능한 로봇만 base 대응 후보이며 실제 거리 조건도 만족해야 한다. **구역 소유자를 B라고 지정해도 B가 멀거나 고장이면 물리적 대응 가능으로 간주하지 않는다.**

정책의 자기 관측과 원시 Ground Truth는 별도 필드다. jitter와 자기 위치 관측 잡음을 설정할 수 있으며 기본값은 0이다. READY는 B의 자기 위치 관측을 이용한다. 따라서 잡음이 있으면 잘못된 READY도 가능하며 평가기는 이를 신뢰하지 않고 실제 위치로 다시 계산한다.

## 네 스크립트 정책

이들은 인프라를 검증하기 위한 통제 사례이며 Phase 2의 범용 할당 알고리즘이나 학습된 정책이 아니다.

- `immediate`: t=2 요청과 동시에 A가 base 책임을 B로 넘기고 동적 작업으로 이동한다.
- `ready`: 요청을 받은 B가 구역으로 이동하고 READY를 회신한다. A는 수신 후에만 떠난다.
- `late`: READY 수신 후에도 추가 2초 기다린다.
- `fixed`: A가 끝까지 base를 지키고 B는 대기한다. 새 작업을 수행하지 않는다.

명시적 소유자, 현재 작업, 목표 위치, 실제 대응 가능 상태를 분리했다. 인계 메시지는 REQUEST/READY만 구현했다. 요청과 회신 모두 같은 단방향 통신 지연을 받으며, 고장 난 수신자에게 도착한 메시지는 유실된다. READY 후 후임이 고장 나도 오래된 READY를 믿고 떠날 수 있다. 범용 취소·재요청·ACK·lease·작업 큐는 구현하지 않았다.

## 시나리오와 사건 순서

- `normal`: 기본 설정.
- `urgent`: 동적 작업 마감만 t=9로 단축.
- `successor_failure`: B가 t=4에 영구 고장.
- `communication_delay`: 요청·회신 각각 1초 지연.
- `temporary_failure`: B가 t=4 고장, t=7에 복구. 고장 전 이동 목표를 유지해 복구 후 이동을 재개한다.

각 경계에서 이전 서비스 완료 → 고장/복구 → 작업 생성 → 메시지 수신 → 자기 관측 READY 판정 → 즉시 수신 → 스크립트 결정 → 실제 응답/마감 판정 → 구간 로그 → 이동/서비스 순서다.

구간 [t,t+dt)의 공백은 경계 t의 실제 상태로 계산한다. 구간 중간의 도착을 보간하지 않으므로 시간 오차는 dt 규모다. 사용자가 설정한 사건 시각·서비스 길이·통신 지연은 dt의 배수여야 한다. 기본 dt=1은 해석적 검산 목적이며 실물 응답 시간의 추정치가 아니다.

## 독립 평가 지표

평가기에는 보상 계산이 없다. 모든 지표는 저장한 로그만으로 다시 계산한다.

- `coverage_gap_s`: base에 실제 대응 가능한 로봇이 없는 구간 길이의 합.
- `coverage_gap_ratio`: 위 길이 / 전체 평가시간. 이 단계는 중요도 1인 단일 구역이며 가중 다중 구역 CGR은 아직 아니다.
- `ownership_gap_ratio`: 명시적 base 소유자가 없는 시간 비율. 조기 이탈에서도 소유자를 즉시 바꾸므로 이 값은 0일 수 있다.
- `declared_owner_not_ready_ratio`: 지정된 소유자가 현장에서 대응 불가능한 시간 비율.
- `handovers[].initial_gap_s`: RELEASE 이후 후임이 처음 실제 대응 가능해질 때까지의 시간. 이미 준비됐다면 0이다. 끝까지 준비되지 않으면 관측된 하한과 right_censored=true를 기록한다.
- `handovers[].physical_gap_union_s`: RELEASE 이후 실제 구역 공백의 구간 합. READY 뒤 다시 고장 난 경우도 포착한다. 단일 인계 실험이므로 다수 인계 사건의 중첩 귀속은 아직 없다.
- `task_success_rate`: 마감 내 실제 서비스를 끝낸 작업 / 발생 작업. 현재는 한 에피소드에 작업 1개라 0 또는 1이다.
- `response_times_s`: 작업 발생부터 유효 서비스가 시작된 시각까지. 단순 배정은 응답이 아니다.
- `response_time_mean_s`: 실제 응답한 작업의 평균. 무응답이면 null이다.
- `unresponded_tasks`: 응답하지 못한 작업 수. 반드시 응답 평균과 함께 본다.
- `restricted_response_mean_s`: 무응답을 마감/종료까지 검열한 지연. 실제 응답 평균을 대체하는 성공 지표가 아니며 관측 상한으로 제한된 미응답 지연이다.
- `failure_recoveries`: 로봇 고장 후 base의 물리적 커버가 회복되는 시간. 이미 다른 로봇이 커버 중이면 0이며, 고장 로봇 자체의 수리 시간을 뜻하지 않는다. 미회복은 null/right_censored다.

고정 역할은 CGR=0이지만 작업 성공률=0과 무응답이 함께 나타난다. 단일 종합 점수로 정책을 우열 평가하지 않는다. 중복 담당자 배열은 이 단계 스키마에서 허용하지 않으며, 불필요한 중복 배정률·기본 역할 변경 횟수·여러 작업 일반화는 아직 구현하지 않았다.

## 원시 로그와 재현성

JSONL은 header, event, interval, terminal 레코드로 구성된다.

- header: 설정·seed·해석된 사건 계획·core/metrics/run 소스 SHA256.
- event: 요청/수신/준비/책임 해제/배정/작업 발생·응답·완료·만료/고장·복구의 시각과 ID.
- interval: 구간 시작·끝, 각 로봇 실제 상태와 별도 관측, 구역 소유자, 작업 진행량.
- terminal: 평가 종료의 실제 상태.

평가기는 종료 레코드가 없거나 구간이 빠지고 겹치는 로그를 거부한다. 설정·seed·코드·Python 버전이 같은 경우 JSONL 바이트가 재현된다. 로그에 벽시계 시간을 넣지 않는다. 정책이 달라도 외생 사건 계획은 동일 seed에서 같게 유지되며, 정책 분기에 따른 난수 소비에 의존하지 않는다.

기록은 정상 실행의 재계산을 위한 스키마이며 위조 로그나 모든 종류의 잘못된 JSON을 방어하는 보안 검증기는 아니다.

## 이번 실제 검증 자료

- `outputs/test_results.txt`: 최종 코드의 단위 테스트 26개 실제 실행 결과.
- `outputs/validation_record.json`: Python 버전, 실행 명령, 반환값, 검사 수.
- `outputs/phase1_final/*.jsonl`: 5개 시나리오×4개 정책, seed=0 실제 로그 20개.
- `outputs/phase1_final/summary.json`: 위 로그에서 계산한 실제 측정값.
- `RESULTS_KO.md`: 사람이 읽는 측정 결과와 해석 범위.

20개 로그 모두 별도 Python 프로세스의 evaluate 명령으로 재계산해 summary와 일치함을 확인했다. tests에는 같은 seed의 바이트 재현, 다른 seed의 변화, 정책 간 외생 사건 일치, 관측 플래그와 평가 분리, 단일 용량, 무응답·마감·고장 검열, 기존 출력 보호도 포함한다.

`outputs/phase1_validation`과 `outputs/config_example.jsonl`은 최종 추가 검사 이전 실행 자료다. 삭제하지 않고 보존했으며 완료 보고의 측정값은 **phase1_final만** 사용한다.

## 범위와 다음 단계

실물/ROS/학습/원격 작업, 패키지 설치, 기존 파일 수정, Git add/commit/push는 수행하지 않았다. Phase 2는 아직 시작하지 않는다.

후속 단계에서는 범용 작업·담당자 모델, Greedy/Hungarian 기준선, 긴급도 기반 중재, 인계 취소·재요청·메시지 노후화, 다중 시드 통계를 검토할 수 있다. 이는 현 구현의 성능 주장이나 완료 범위가 아니다.

# Phase 3A 실제 검증 결과

아래는 실제 코드 실행값이다. 학습 결과, 실물 결과, 가상의 예시 측정값이 아니다. Phase 3B는 시작하지 않았다.

## 최종 증거 파일

- `phase3a_outputs/test_results_acceptance.txt`: 단위 테스트 85개 실제 출력.
- `phase3a_outputs/validation_record_complete.json`: 최종 테스트·반복 실행·재평가·파일 보존 기록과 소스 해시.
- `phase3a_outputs/smoke_complete/*.jsonl`: 최신 환경의 32개 완료 로그.
- `phase3a_outputs/smoke_complete/summary.json`: 위 로그의 보상/논문 지표/표본 검사 수.
- `phase3a_outputs/goal_demo_complete.jsonl`, `goal_demo_complete_metrics.json`: 고정 목표 이동·서비스 완료 검증.
- `phase3a_outputs/preservation_before.json`: 시작 전 기존 파일 해시.
- `phase3a_outputs/mappo_review_sources.json`: 공식 공개 코드 검토의 경로와 blob SHA.

smoke_final/smoke_bound, goal_demo.jsonl 및 이전 test_results/validation_record는 중간 개발 자료다. 삭제하지 않고 보존했으며 이 완료 보고의 최종 근거로 사용하지 않는다.

## 단위 테스트

기존 Phase 1·2 55개 + 신규 30개 = **85개 통과, 실패 0개**.

검증 항목에는 동일 seed 초기 상태·행동 전이·JSONL 재현, Actor/Critic 분리, 숨은 작업 슬롯 누출 방지, 실제 메시지 도착/측정 나이, 관측하지 못한 작업 종료에 따른 oracle mask 방지, 패딩/고장 시 NOOP 보장, 비가용·복구, 중재 전후 원래 행동 보존, 분포/log-probability/관측 토큰 일치, 보상 유한성·단계/에피소드 한도, 종료/bootstrap 의미, 기존 평가 회귀가 포함된다.

마스크가 바뀌지 않아도 지난 프레임 또는 다른 에이전트의 행동 표본은 거부된다. 반면 중재로 작업이 거절된 표본은 원래 샘플 행동에 대한 유효한 표본으로 보존된다. 신경망을 실제 실행해서 산출한 확률을 검사한 것은 아니다.

## 학습 없는 반복 실행

8개 시나리오 × seed 0/1 × 유지/균등 무작위 행동 = **32 에피소드, 768 환경 단계**.

- 행동 표본 계약 검사 2,112건.
- 이 중 비활성 로봇 표본 92건은 policy loss mask=0.
- 최대 단계 보상 절댓값: **3.9114583333333335**.
- 유지 16회: 작업 완료 **0/28**, 팀 return 범위 **-3.8125 ~ -1.2666666666666666**.
- 균등 무작위 16회: 작업 완료 **0/28**, 팀 return 범위 **-6.133333333333333 ~ -1.975**.

임의 정책이 작업을 완료하지 못한 사실을 숨기지 않는다. 이 실행은 학습 없는 전이·마스크·수치 안정성 검사이지 성능 기준선이나 학습 가능성의 입증이 아니다.

## 고정 목표 행동의 실제 완료 검증

같은 single 환경에서 Actor가 알고 있는 로컬 작업 슬롯만 이용해 A는 task1, B는 base를 선택하는 별도 스크립트를 실행했다. 다른 로봇 GT나 미래 사건을 읽는 정책이 아니다.

- 작업 1/1 완료, 마감 실패 0.
- 실제 응답 지연 **4초**, 완료 절대 시각 **t=8초**.
- 작업 발생부터 전체 완료까지 **6초**.
- 구역 공백 **4초**, CGR **16.6667%**.
- 팀 return **1.1104166666666668**.
- 실행 중복률 0이지만 요청 중복률은 **6.6667%**다. 공통 중재가 실제 동작했음을 보여준다.

이 정책 역시 수작업 시연이다. PPO 업데이트나 MAPPO 성능을 의미하지 않는다. 테스트에서는 별도 동일 위치 통제 사례에서도 서비스 성공이 역할 유지보다 높은 보상을 받는지 확인했다. 보상 가중치가 최적이거나 모든 reward hacking을 막는다는 주장은 하지 않는다.

## 로그 재계산과 보존

- 최신 smoke 로그 32개를 각각 별도 프로세스의 평가 명령으로 재계산해 summary와 일치했다.
- 기존 Phase 1 로그 20개, Phase 2 로그 192개, 합계 **212개**를 다시 평가했다.
- Phase 1·2의 기존 파일/출력과 사용자 수정 파일을 포함한 **301개 파일의 SHA256이 시작 전과 동일**하다.
- 코드·문서·출력은 research/cah_mappo 내부에만 추가했다. 기존 ROS/모델/통신·체크포인트는 변경하지 않았다.

## 아직 실행하지 않은 것

PettingZoo parallel_api_test, Gymnasium 실제 space.contains 검사, torch 신경망 forward/optimizer, 공식 rollout buffer와의 실제 통합은 실행하지 않았다. 선택적 어댑터는 설계/구현만 했으며 설치 없는 코어 검증과 구분한다.

공식 MAPPO 연결에는 전용 runner/buffer 연결, all-inactive 업데이트 생략, 시간 제한 최종 가치 처리, 공유 critic의 마스크/가중치 정의, 확률을 동일 forward에서 얻는 검사가 남아 있다. MAPPO_CONNECTION_REVIEW_KO.md에 경로별로 정리했다.

Phase 3A의 코어 인터페이스 검증을 완료하고 여기서 종료한다. 설치·학습을 포함한 Phase 3B는 별도 승인 대상이다.

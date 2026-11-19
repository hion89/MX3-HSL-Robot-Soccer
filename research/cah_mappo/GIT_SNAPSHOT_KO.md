# 연구 코드와 결과의 Git 기록 · 2026-11-19

Phase 1부터 Phase 3B-2B까지 누적된 연구 코드를 단계별로 기록한다. 현재 완료 단계는 정상 정보 전달 조건의 Standard MAPPO 5시드 기준선 실험이다. CAH-MAPPO는 아직 구현하지 않았다. 다음 단계는 승인 전이며 `PHASE4_RULE_HANDOVER_PLAN_KO.md`는 설계 문서다.

## GitHub에 포함하는 자료

- `src/cah_mappo/`: 독립 2D 시뮬레이터, 평가기, 기준선, MARL 인터페이스, Standard MAPPO 학습기.
- `tests/`: Phase 1~3B-2B의 회귀·신규 단위 테스트 152개.
- 루트의 실행·감사·분석 스크립트, 단계별 한국어 설명·결과 문서, 의존성 목록과 MAPPO 출처·라이선스.
- `results/snapshot_20261119/`: 실제 실행에서 추출한 요약 JSON, CSV, SVG. `publication_manifest.json`이 원본 상대 경로·해시·변환 여부를 기록한다.

기존 결과 문서의 `*_outputs/` 경로는 실제 실행 당시의 **로컬 원본 경로**다. Git clone만으로 이 원시 데이터와 체크포인트가 복원되지는 않는다. 공개 스냅샷은 저장된 집계값을 보존하며 일부 JSON의 에피소드별 배열만 제거했다. `omitted_episode_records`는 제거한 배열 길이이며 새 측정값이 아니다. 원시 로그를 요구하는 전수 감사는 원본 로컬 출력 또는 새로 재현한 출력이 있어야 실행할 수 있다.

## 로컬에 보존하고 Git에서 제외하는 자료

승인된 `.venv`, `outputs/`, 모든 `*_outputs/`, 체크포인트, 원시 JSONL, 훈련 의사결정 및 rollout은 삭제하지 않고 로컬에 남긴다. 기존 최종 Holdout 400개의 데이터는 열거나 공개 스냅샷에 복사하지 않는다. 소스의 시나리오 생성 코드와 데이터 보호 규약은 포함하지만 보호 데이터 파일은 포함하지 않는다.

소스와 공개 측정 요약을 GitHub에 보관하는 것과 전체 학습 상태의 백업은 다르다. 원본 체크포인트·원시 로그는 현재 PC에만 남으므로 전체 복구가 필요하면 별도의 저장장치 백업을 마련해야 한다. 이번 작업에서 외부로 해당 데이터를 전송하지 않았다.

## 재검증

이번 Git 정리 시 기존 전용 환경에서 다시 실행한 결과: **152개 테스트 통과, 15.899초, 실패·오류·skip 0개**.

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim
& research/cah_mappo/.venv/Scripts/python.exe -B -m unittest discover -s research/cah_mappo/tests -v
```

새 PC에서 환경을 만들 때는 `PHASE3B1_README_KO.md`와 `requirements-phase3b1-resolved.txt`를 확인한다. 가상환경은 Git에 포함하지 않는다. 이후 단계의 재현 명령과 실행 예산은 각 단계 README를 따른다. 전체 학습을 재현하려면 기존 출력과 다른 새 디렉터리를 사용한다.

공개 자료를 다시 추출할 때도 덮어쓰지 않는 새 목적지를 지정한다.

```powershell
& research/cah_mappo/.venv/Scripts/python.exe -B research/cah_mappo/export_git_evidence.py --output research/cah_mappo/results/new_snapshot_name
```

이 도구는 명시적으로 허용한 결과 파일만 읽으며 Holdout 파일을 탐색하지 않는다. 기존 학습 소스·결과·체크포인트는 수정하지 않는다.

## 현재 결과를 읽는 순서

1. `PHASE3B2B_RESULTS_KO.md`: 최종 5시드 측정 결과와 한계.
2. `PHASE3B2B_PROTOCOL_KO.md`: 동결한 비교 조건과 선택 기준.
3. `results/snapshot_20261119/phase3b2b/analysis/all_metrics.csv`: 정책·시드별 지표.
4. `results/snapshot_20261119/phase3b2b/figures/`: 학습·검증 곡선과 비교 그림.
5. `PHASE3B2A_REPAIR_RESULTS_KO.md`: 이전 슬롯 편향·관측 실패 진단.

결정적 실행은 5시드 모두 작업 성공률 100%/책임 공백 0%였다. 확률적 실행의 시드 평균은 성공률 99.9222%/책임 공백 비율 0.209375%였다. 중앙 Fixed도 성공률 100%/공백 0%를 달성했으므로 MAPPO의 일방적 우위를 주장하지 않는다. 드문 큰 공백과 슬롯 편향이 남으며, 보호 Holdout이나 통신 장애 조건에서 검증한 결과가 아니다.

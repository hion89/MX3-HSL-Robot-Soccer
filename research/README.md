# 다중 로봇 책임 인계 연구

`cah_mappo/`는 실물 로봇 프로젝트와 분리된 2D 시뮬레이션·Standard MAPPO 연구 코드입니다. 실행 시 실물 ROS2나 Isaac Lab을 사용하지 않습니다.

현재 Phase 3B-2B의 Standard MAPPO 5개 독립 시드 실험까지 완료했습니다. CAH-MAPPO와 다음 단계 구현은 아직 시작하지 않았습니다.

- [코드·실험 결과·Git 보관 범위 안내](cah_mappo/GIT_SNAPSHOT_KO.md)
- [최신 측정 결과와 한계](cah_mappo/PHASE3B2B_RESULTS_KO.md)
- [동결한 학습·평가 규약](cah_mappo/PHASE3B2B_PROTOCOL_KO.md)
- [실행 방법](cah_mappo/PHASE3B2B_README_KO.md)
- [공개 측정 요약과 학습 곡선](cah_mappo/results/snapshot_20261119/)
- [초기 2D 환경 설명](cah_mappo/README_KO.md)

가상환경, 체크포인트, 원시 실험 로그와 최종 보호 Holdout 데이터는 Git에 포함하지 않습니다. Git clone은 코드와 공개 요약을 복원하며, 기존 학습 상태 전체를 복원하는 백업은 아닙니다.

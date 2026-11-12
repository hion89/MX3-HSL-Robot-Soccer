# 작업 지침

- 사용자에게 한국어로 답하고, 변경 범위와 실물 적용 여부를 구분한다.
- 이 저장소는 MX3 실물 축구 로봇과 HSL/B-Human 방식 상위 기능의 작업 기준이다.
- 사용자는 2026-11-12부터 작업마다 GitHub 반영을 요청했다. 관련 변경과 검증을 마치면 커밋하고 origin/main에 푸시한다. 인증/네트워크 실패 시 로컬 커밋 여부와 남은 푸시를 정확히 보고한다.
- 실물용 소스는 MX3_Match_Sim/src/mx3_src와 src/hsl_src다. 기존 Isaac-work-space 저장소에 무관한 변경을 함께 커밋하지 않는다.
- 별도 isaac_ws/sim2real_evidence에서 작업했으면 scripts/sync_working_copy.py로 기록과 배포물을 가져온 다음, 변경 목록을 검토하고 관련 파일만 커밋한다.
- 운영 기록은 docs/MX3_HSL_DEPLOYMENT_PROGRESS.md, 최신 상태는 docs/CURRENT_STATUS_KO.md를 확인한다. 이 파일들이 과거 매뉴얼의 시작 위치보다 우선한다.
- 현재 연습장 수동 초기 위치는 x=-1.95, y=+0.8675, yaw=0. 실제 배치와 START 후에만 사용자가 초기값을 발행한다.
- START의 임시 진입 시드는 라인 밖 0.10m로 잔디 안이다. 수동 골라인 초기 위치와 다르다.
- 저장소 푸시 권한은 로봇 동작/원격 설치 권한과 별개다. GitHub 업데이트만으로 모터를 구동하거나 initialpose를 발행하지 않는다.
- 정책 파일과 학습 체크포인트는 보존한다. 가상환경/빌드/브라우저 프로필/비밀정보/원본 영상과 ROS bag은 커밋하지 않는다.
- 긴 Markdown 표 대신 설명을 사용한다. 운영 매뉴얼 산출물은 Manual의 Word/PDF를 우선한다.

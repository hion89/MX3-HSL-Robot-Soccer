"""Generate a Korean report only from stored, measured Phase 2 results."""
import argparse
import json
from pathlib import Path

LABELS = {'single': '단일 작업', 'simultaneous': '복수 작업 동시 발생', 'priorities': '우선순위 차이',
          'slow_robot': '이동 성능 저하', 'owner_failure': '초기 구역 담당자 고장', 'urgent': '긴급 작업',
          'rising_risk': '인계 중 위험도 증가', 'communication_delay': '관측 지연'}


def number(value):
    return '해당 없음' if value is None else f'{value:.3f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('phase2_outputs'))
    args = p.parse_args()
    measured = json.loads((args.source / 'final_seed0/summary.json').read_text())
    repeated = json.loads((args.source / 'paired_5seeds/aggregates.json').read_text())
    validation = json.loads((args.source / 'validation_record_final.json').read_text())
    lines = ['# Phase 2 실제 측정 결과', '',
             '이 문서는 저장된 JSON 측정값에서 생성했다. 임의의 예시 값·실물 성능·학습 성능이 아니다.', '',
             f"전체 테스트 {validation['tests_passed']}개 통과(기존 {validation['phase1_tests']}개 유지). "
             f"신규 로그 {validation['new_logs_recomputed_in_separate_processes']}개와 Phase 1 로그 "
             f"{validation['phase1_logs_replayed']}개를 별도 프로세스에서 재계산했다.", '',
             '기본 설정은 dt=1초, horizon=24초다. 응답은 발생부터 실제 서비스 시작까지이며, '
             'makespan은 모든 작업이 성공한 경우에만 첫 발생부터 마지막 완료까지다. '
             '무응답을 평균에서 숨기지 않도록 건수를 같이 표시한다.', '',
             '## 변동 없는 seed 0: 실제 32개 실행', '']
    for name, label in LABELS.items():
        lines += [f'### {label}', '']
        for r in measured:
            if r['scenario'] != name:
                continue
            lines.append(f"- {r['policy']}: CGR {r['coverage_gap_ratio'] * 100:.2f}%, "
                         f"성공 {r['task_success_count']}/{r['task_count']}, "
                         f"응답 {number(r['response_time_mean_s'])}초, 무응답 {r['unresponded_tasks']}건, "
                         f"마감 실패 {r['deadline_miss_rate'] * 100:.2f}%, "
                         f"makespan {number(r['all_tasks_makespan_s'])}초, "
                         f"역할 전환 {r['role_switch_count']}회, 중복 {r['task_duplicate_rate'] * 100:.2f}%.")
        lines.append('')
    lines += ['## 같은 외생 계획을 사용하는 5개 시드: 실제 160개 실행', '',
              'seed 0~4에서 초기/작업 위치 변동 ±.15m를 주었다. 다음은 평균 CGR·성공률과 '
              '실제 응답 작업 평균이다. 학습 시드 실험이나 유의성 검정은 아니다. '
              '전체 지표의 min/max와 정의된 실행 수는 aggregates.json에서 확인한다.', '']
    for name, label in LABELS.items():
        lines += [f'### {label}', '']
        for r in repeated:
            if r['scenario'] == name:
                cg, success, rt = r['coverage_gap_ratio'], r['task_success_rate'], r['response_time_mean_s']
                lines.append(f"- {r['policy']}: 평균 CGR {cg['mean'] * 100:.2f}%, "
                             f"평균 성공률 {success['mean'] * 100:.2f}%, "
                             f"응답 평균 {number(rt['mean'])}초(정의된 실행 {rt['defined_runs']}/{r['runs']}).")
        lines.append('')
    lines += ['## 해석과 제한', '',
              '- Fixed는 정상 상황에서 구역을 유지하지만 긴급 작업이나 담당자 고장에 취약하다. '
              'CGR 하나만으로 우수하다고 판단하지 않는다.',
              '- Greedy는 빠르게 대응하지만 구역 공백과 역할 전환이 커질 수 있다. '
              'Exact는 전환 비용으로 배정을 유지하므로 모든 상황에서 Greedy보다 빠르거나 안전하지 않다.',
              '- Handover는 정상 인계에서 공백을 줄이는 대신 응답이 늦어진다. '
              '긴급 상황에서는 조기 해제를 허용해 성공을 얻지만 공백이 생긴다. '
              '위험도 증가 시 보류 인계를 취소하는 사례와 고장 상황에서 한계가 남는 사례를 함께 봐야 한다.',
              '- 실행 중복 0은 공통 실행 중재의 영향이다. 요청 중복률도 summary.json에 보관한다.',
              '- Exact의 최소값은 공개한 순간 할당 비용의 최소값이며 미래 일정·CGR·전체 성공률의 전역 최적값이 아니다.',
              '- 본 수치로 단일 종합점수나 정책 순위를 만들지 않았다. '
              '모델 크기·학습 알고리즘·실물 성능에 관한 결론을 내리지 않는다.', '',
              'Phase 1 파일/출력과 기존 사용자 변경은 보존했다. 설치·원격·실물·Git 커밋/푸시는 없으며 Phase 3은 아직 시작하지 않았다.', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()

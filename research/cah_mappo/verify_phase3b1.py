"""Read saved raw logs/checkpoints independently; write a new audit directory only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def verify(experiments, output):
    import torch
    from cah_mappo.metrics import load_jsonl
    from cah_mappo.phase2.metrics import evaluate
    from cah_mappo.phase3b1.learner import Learner, TrainConfig
    from cah_mappo.phase3b1.experiment import evaluate_policy
    output.mkdir(parents=True, exist_ok=False)
    saved = json.loads((ROOT / 'phase3b1_outputs/preservation_before.json').read_text())
    mismatches = [name for name, digest in saved.items()
                  if not (ROOT.parents[1] / name).is_file()
                  or hashlib.sha256((ROOT.parents[1] / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise AssertionError(f'Preexisting files changed: {mismatches}')
    reports = []
    for experiment in experiments:
        config = json.loads((experiment / 'config.json').read_text())
        initial = torch.load(experiment / 'checkpoints/initial.pt', weights_only=True, map_location='cpu')
        final = torch.load(experiment / f"checkpoints/final_step{config['total_env_steps_requested']}.pt",
                           weights_only=True, map_location='cpu')
        learner = Learner(TrainConfig(**initial['config']))
        learner.net.load_state_dict(initial['network'])
        batch = torch.load(experiment / 'first_rollout.pt', weights_only=True, map_location='cpu')
        logp_error = learner.ratio_before_update(learner.tensors(batch))
        assert logp_error < 1e-6
        parameter_changes = {part: sum(float((final['network'][k] - value).abs().sum())
                                      for k, value in initial['network'].items() if k.startswith(part + '.'))
                             for part in ('actor', 'critic')}
        assert all(value > 0 for value in parameter_changes.values())
        for value in final['network'].values():
            assert torch.isfinite(value).all()
        summaries = json.loads((experiment / 'baselines.json').read_text())
        evaluation = json.loads((experiment / 'evaluation.json').read_text())
        records = [('initial_' + name, row) for name, row in summaries.items()]
        records += list(evaluation['results'].items())
        raw_logs = 0
        replayed = 0
        for label, record in records:
            for episode in record['episodes']:
                path = experiment / label / f"seed{episode['environment_seed']}.jsonl"
                rows = load_jsonl(path)
                measured = evaluate(rows)
                assert measured == {key: episode[key] for key in measured}, str(path)
                team_return = sum(row['reward'] for row in rows if row['type'] == 'transition')
                assert abs(team_return - episode['return']) < 1e-10, str(path)
                raw_logs += 1
            if 'checkpoint' in record:
                checkpoint = torch.load(experiment / record['checkpoint'], weights_only=True, map_location='cpu')
                learner.net.load_state_dict(checkpoint['network'])
                first = record['episodes'][0]
                new = evaluate_policy(learner.net, first['mode'],
                                      [r['environment_seed'] for r in record['episodes']], first['jitter'])
                assert new == record['episodes'], f'Checkpoint replay differs: {label}'
                replayed += len(new)
        report = {'experiment': str(experiment), 'raw_evaluation_logs_recomputed': raw_logs,
                  'checkpoint_evaluation_episodes_replayed': replayed,
                  'first_rollout_preupdate_logp_max_error': logp_error,
                  'arbitrated_agent_actions_in_first_rollout': sum(sum(b['arbitration_changed']) for b in batch),
                  'initial_to_final_parameter_l1_change': parameter_changes}
        reports.append(report)
    result = {'status': 'PASS', 'preexisting_files_sha256_preserved': len(saved), 'experiments': reports}
    with (output / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.experiment, args.output)


if __name__ == '__main__':
    main()

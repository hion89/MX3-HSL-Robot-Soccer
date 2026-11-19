"""Independent post-run audit of raw logs, paired scenarios and prior-file preservation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def verify(run, output):
    import torch
    from cah_mappo.metrics import load_jsonl
    from cah_mappo.phase2.metrics import evaluate
    from cah_mappo.phase3b2a.scenarios import digest, make_env
    from cah_mappo.phase3b2a.experiment import load_net
    from cah_mappo.phase3b2a.evaluation import evaluate_many
    from cah_mappo.phase3b2a.learner import PoolLearner
    from cah_mappo.phase3b1.learner import TrainConfig
    output.mkdir(parents=True, exist_ok=False)
    before = json.loads((ROOT / 'phase3b2a_outputs/preservation_before.json').read_text())
    for path, expected in before.items():
        assert hashlib.sha256((ROOT.parents[1] / path).read_bytes()).hexdigest() == expected, path
    manifest = json.loads((run / 'scenario_manifest.json').read_text())
    plan = json.loads((run / 'predeclared_plan.json').read_text())
    freeze = json.loads((run / 'selection_freeze.json').read_text())
    assert freeze['manifest_sha256'] == digest(manifest)
    assert freeze['plan_sha256'] == digest(plan)
    specs = {s['scenario_id']: s for stage in manifest['scenarios'].values() for split in stage.values() for s in split}
    seed_sets = {split: {s['event_seed'] for s in specs.values() if s['split'] == split} for split in ('train', 'validation', 'test')}
    assert not seed_sets['train'] & seed_sets['validation']
    assert not seed_sets['train'] & seed_sets['test']
    assert not seed_sets['validation'] & seed_sets['test']
    for spec in specs.values():
        assert digest({k: v for k, v in spec.items() if k not in ('scenario_id', 'sha256')}) == spec['sha256']
    result = json.loads((run / 'test_results.json').read_text())
    records = []
    for stage, modes in result['benchmarks'].items():
        records.extend((run / 'test_logs' / stage / mode, row) for mode, row in modes.items())
    for label, modes in result['trained_policies'].items():
        records.extend((run / 'test_logs' / label / mode, row) for mode, row in modes.items())
    for stage, modes in result['transfer'].items():
        records.extend((run / 'test_logs/transfer' / stage / mode, row) for mode, row in modes.items())
    pairs = {}; logs = 0
    for folder, group in records:
        for episode in group['episodes']:
            sid = episode['scenario_id']; assert specs[sid]['split'] == 'test'
            rows = load_jsonl(folder / f'{sid}.jsonl')
            measured = evaluate(rows)
            assert measured == {key: episode[key] for key in measured}
            assert rows[-1]['time'] == 32 and rows[0]['config']['horizon'] == 32
            expected_plan = pairs.setdefault(sid, digest(rows[0]['exogenous_plan']))
            assert episode['exogenous_sha256'] == expected_plan == digest(rows[0]['exogenous_plan'])
            if episode['return'] is not None:
                assert abs(sum(r['reward'] for r in rows if r['type'] == 'transition') - episode['return']) < 1e-9
            logs += 1
    replays = 0; errors = []
    for entry in freeze['runs']:
        label = f"{entry['stage']}_weight{int(entry['coverage_weight'])}"
        config = json.loads((run / label / 'config.json').read_text())
        assert all(specs[s]['split'] == 'train' for s in config['train_ids'])
        assert all(specs[s]['split'] == 'validation' for s in config['validation_ids'])
        for name, expected in config['source_sha256'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        checkpoint = Path(entry['selected_checkpoint'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == freeze['checkpoint_sha256'][str(checkpoint)]
        network = load_net(checkpoint)
        for mode in ('mappo_greedy', 'mappo_stochastic'):
            saved = result['trained_policies'][label][f'selected_{mode}']
            replay = evaluate_many(manifest['scenarios'][entry['stage']]['test'], mode, network, entry['coverage_weight'])
            assert replay == saved
            replays += len(replay['episodes'])
        learner = PoolLearner(TrainConfig(**config['train_config']), [specs[s] for s in config['train_ids']], entry['coverage_weight'])
        initial = torch.load(run / label / 'checkpoints/initial.pt', weights_only=True, map_location='cpu')
        learner.net.load_state_dict(initial['network'])
        batch = torch.load(run / label / 'first_rollout.pt', weights_only=True, map_location='cpu')
        error = learner.ratio_before_update(learner.tensors(batch)); assert error < 1e-6
        errors.append(error)
    report = {'status': 'PASS', 'preserved_files': len(before), 'split_scenarios': {k: len(v) for k, v in seed_sets.items()},
              'test_raw_logs_recomputed': logs, 'paired_scenario_ids': len(pairs),
              'checkpoint_evaluation_replays': replays, 'first_rollout_logp_errors': errors,
              'selection_frozen_before_test': True}
    with (output / 'audit.json').open('x', encoding='utf-8') as f:json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); verify(args.run, args.output)

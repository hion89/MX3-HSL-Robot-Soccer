"""Post-hoc descriptive diagnostics only; never train or select a model."""
import argparse
import json
from pathlib import Path
import sys
from collections import Counter
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def diagnose(run, output):
    from cah_mappo.metrics import load_jsonl
    from cah_mappo.phase3b2a.experiment import load_net, save
    from cah_mappo.phase3b2a.evaluation import evaluate_many
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run / 'scenario_manifest.json').read_text())
    network = load_net(ROOT / 'phase3b1_outputs/debug_seed7_steps4096/checkpoints/final_step4096.pt')
    old_fixed_window = evaluate_many(manifest['scenarios']['A']['test'], 'mappo_stochastic', network,
                                     log_dir=output / 'old_checkpoint_randomized_A_fixed_window')
    save(output / 'posthoc_old_stochastic.json', old_fixed_window)
    diagnostics = {}
    groups = {
        'C_old_transfer': run / 'test_logs/transfer/C/fixed_3b1_checkpoint',
        'C_trained_selected': run / 'test_logs/C_weight2/selected_mappo_greedy',
        'D_A_transfer': run / 'test_logs/transfer/D/randomized_A_checkpoint',
        'D_local_fixed': run / 'test_logs/D/local_fixed',
        'A_weight8_final_greedy': run / 'test_logs/A_weight8/final_mappo_greedy',
    }
    for name, folder in groups.items():
        counts = Counter(); unseen = []; completed_jobs = Counter(); selected_slots = Counter()
        for path in folder.glob('*.jsonl'):
            rows = load_jsonl(path)
            observed = set()
            terminal_jobs = rows[-1]['truth']['jobs']
            for row in rows:
                if row.get('event') != 'policy_selection' or 'actor_observations' not in row:
                    continue
                for agent, packet in row['actor_observations'].items():
                    observed.update(jid for jid in packet['job_ids'] if jid is not None)
                    record = row['sampled_actions'][agent]
                    action = record['sampled_action']; selected_slots[str(action)] += 1
                    jid = record['requested_assignment']
                    if action >= 2:
                        assert jid == packet['job_ids'][action - 2]
                    kind = terminal_jobs[jid]['kind'] if jid is not None else 'idle'
                    counts['requested_' + kind] += 1
                    visible_tasks = any(packet['action_mask'][i + 2] and not packet['observation'][52 + i * 9]
                                        for i in range(8))
                    if visible_tasks:
                        counts['agent_ticks_with_visible_task'] += 1
                        if kind == 'base':counts['base_request_despite_visible_task'] += 1
            for jid, job in terminal_jobs.items():
                if job['kind'] != 'task':continue
                counts['tasks'] += 1
                if jid not in observed:
                    counts['never_observed_tasks'] += 1; unseen.append({'scenario_id': path.stem, 'job': jid})
                if job['status'] == 'completed':completed_jobs[jid] += 1
        diagnostics[name] = {'counts': dict(counts), 'selected_action_counts': dict(selected_slots),
                             'completed_by_job': dict(completed_jobs), 'never_observed': unseen}
    save(output / 'diagnostics.json', {'scope': 'post-hoc descriptive analysis; no tuning or training',
                                       'old_stochastic_fixed_window': old_fixed_window['summary'], 'groups': diagnostics})
    print(json.dumps({'old_stochastic_fixed_window': old_fixed_window['summary'],
                      'groups': {k: {a: b for a, b in v.items() if a != 'never_observed'} for k, v in diagnostics.items()}}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); diagnose(args.run, args.output)

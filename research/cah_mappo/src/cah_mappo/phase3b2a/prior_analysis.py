"""Descriptive audit of frozen Phase 3B-1 trajectories, with no causal overclaim."""
import json
from pathlib import Path
from statistics import mean
from ..metrics import load_jsonl


def analyze(root):
    result = {}
    for seed in (7, 11):
        p = Path(root) / f'debug_seed{seed}_steps4096'
        evaluation = json.loads((p / 'evaluation.json').read_text())
        label = 'final_learned_stochastic'
        records = evaluation['results'][label]['episodes']
        components = []; lengths = []; gap_seconds = []
        switches = []; changes = []; duplicates = []; observed_risk = []
        for record in records:
            rows = load_jsonl(p / label / f"seed{record['environment_seed']}.jsonl")
            transitions = [r for r in rows if r['type'] == 'transition']
            sums = {key: sum(t['reward_contributions'][key] for t in transitions)
                    for key in transitions[0]['reward_contributions']}
            components.append(sums)
            lengths.append(rows[0]['config']['horizon']); gap_seconds.append(record['coverage_gap_s'])
            switches.append(record['role_switch_count']); changes.append(record['assignment_change_count'])
            duplicates.append(record['requested_duplicate_rate'])
            for t in transitions:
                for packet in t['records'].values():
                    obs = packet['actor_observation']
                    for j in range(8):
                        if obs[50 + j * 9] and obs[52 + j * 9]:
                            observed_risk.append(obs[55 + j * 9] * 10)
        result[str(seed)] = {'evaluation_episodes': len(records),
                            'reward_contributions_mean': {key: mean(r[key] for r in components) for key in components[0]},
                            'observed_horizon_mean_s': mean(lengths), 'coverage_gap_seconds_mean': mean(gap_seconds),
                            'coverage_gap_ratio_mean': mean(r['coverage_gap_ratio'] for r in records),
                            'common_24s_censoring_lower_bound': mean(g / 24 for g in gap_seconds),
                            'common_24s_censoring_upper_bound': mean((g + 24 - h) / 24 for g, h in zip(gap_seconds, lengths)),
                            'role_switch_count_mean': mean(switches), 'assignment_change_count_mean': mean(changes),
                            'requested_duplicate_rate_mean': mean(duplicates),
                            'observed_base_risk_min_max': [min(observed_risk), max(observed_risk)]}
    return result

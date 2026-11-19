"""No-learning smoke episodes, schema-2 log replay and interface inspection."""
import argparse
import json
import math
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cah_mappo.phase2.scenarios import SCENARIOS, scenario
from cah_mappo.phase2.metrics import evaluate, load_jsonl
from cah_mappo.phase3a.env import ParallelCore
from cah_mappo.phase3a.contracts import Settings, SampledAction, validate_ppo_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    smoke = commands.add_parser('smoke')
    smoke.add_argument('--output', type=Path, required=True)
    smoke.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    smoke.add_argument('--modes', choices=['hold', 'uniform'], nargs='+', default=['hold', 'uniform'])
    replay = commands.add_parser('evaluate'); replay.add_argument('log', type=Path)
    commands.add_parser('inspect')
    goal = commands.add_parser('goal-demo'); goal.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'evaluate':
        print(json.dumps(evaluate(load_jsonl(args.log)), indent=2, allow_nan=False)); return
    if args.command == 'inspect':
        env = ParallelCore(); obs, _ = env.reset(0)
        print(json.dumps({'actor_size': env.observation_size, 'critic_size': env.state_size,
                          'actions': env.settings.n_actions, 'agents': env.agents,
                          'max_agents': len(env.possible_agents), 'max_jobs': env.settings.max_jobs}, indent=2)); return
    if args.command == 'goal-demo':
        env, result = goal_demo()
        env.export_log(args.output)
        print(json.dumps(result, indent=2, allow_nan=False)); return
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        raise ValueError('Repeated modes or seeds would overwrite output')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for name in SCENARIOS:
        for seed in args.seeds:
            for mode in args.modes:
                settings = Settings(radio_delay=3.0 if name == 'communication_delay' else 2.0)
                env = ParallelCore(scenario(name), settings); obs, _ = env.reset(seed)
                rng = random.Random(f'smoke-policy:{name}:{seed}:{mode}')
                checked, inactive, max_abs = 0, 0, 0.0
                while env.agents:
                    actions = {}
                    for agent in env.agents:
                        mask = obs[agent]['action_mask']; valid = [i for i, v in enumerate(mask) if v]
                        action = (1 if mask[1] else 0) if mode == 'hold' else rng.choice(valid)
                        probs = tuple(float(i == action) for i in range(len(mask))) if mode == 'hold' else tuple(1 / len(valid) if v else 0 for v in mask)
                        actions[agent] = SampledAction(action, math.log(probs[action]), tuple(mask), probs,
                                                       f'{mode}-smoke-v0', obs[agent]['observation_token'])
                    obs, rewards, _, _, infos = env.step(actions)
                    max_abs = max(max_abs, max(abs(r) for r in rewards.values()))
                    for info in infos.values():
                        active = validate_ppo_record(info['transition'])
                        checked += 1; inactive += int(not active)
                file = f'{name}__{mode}__seed{seed}.jsonl'
                env.export_log(args.output / file)
                metrics = evaluate(load_jsonl(args.output / file))
                summary.append({'scenario': name, 'seed': seed, 'mode': mode, 'log': file,
                                'steps': env.tick, 'checked_behavior_records': checked,
                                'inactive_records': inactive, 'team_return': env.episode_return,
                                'max_abs_step_reward': max_abs, **metrics})
    with (args.output / 'summary.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'episodes': len(summary), 'steps': sum(r['steps'] for r in summary),
                      'checked_behavior_records': sum(r['checked_behavior_records'] for r in summary),
                      'inactive_records': sum(r['inactive_records'] for r in summary),
                      'max_abs_step_reward': max(r['max_abs_step_reward'] for r in summary)}, indent=2))


def goal_demo():
    """Actor-cache-only deterministic service demonstration, not a learned policy."""
    env = ParallelCore(scenario('single')); obs, _ = env.reset(0)
    while env.agents:
        actions = {}
        for agent in env.agents:
            o = obs[agent]; desired = 'task1' if agent == 'A' else 'base'
            select = o['job_ids'].index(desired) + 2 if desired in o['job_ids'] else 0
            action = select if o['action_mask'][select] and select != 0 else (1 if o['action_mask'][1] else 0)
            probs = tuple(float(i == action) for i in range(len(o['action_mask'])))
            actions[agent] = SampledAction(action, 0.0, tuple(o['action_mask']), probs,
                                           'goal-demo-v0', o['observation_token'])
        obs, _, _, _, infos = env.step(actions)
        for info in infos.values():validate_ppo_record(info['transition'])
    return env, {'team_return': env.episode_return, **env.metrics()}


if __name__ == '__main__':
    main()

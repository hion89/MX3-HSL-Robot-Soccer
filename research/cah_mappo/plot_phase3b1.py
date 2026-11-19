"""Export measured learning curves as standalone SVG, with no plotting dependency."""
import argparse
import html
import json
from pathlib import Path


PANELS = (
    ('train_mean_return', 'Episode return'),
    ('train_task_success_rate', 'Task success rate'),
    ('train_response_time_mean_s', 'Response time (s; responded tasks)'),
    ('train_coverage_gap_ratio', 'Coverage gap ratio'),
    ('policy_entropy', 'Policy entropy (nats)'),
    ('actor_loss', 'Actor loss (clipped surrogate)'),
    ('critic_loss', 'Critic loss (0.5 MSE)'),
    ('approximate_kl', 'Approximate KL'),
    ('clip_fraction', 'Clip fraction'),
    ('actor_gradient_norm', 'Actor gradient norm (before clip)'),
    ('critic_gradient_norm', 'Critic gradient norm (before clip)'),
    ('preupdate_logp_max_error', 'Pre-update log-probability error'),
)


def plot(experiments, output):
    series = []
    for path in experiments:
        config = json.loads((path / 'config.json').read_text())
        series.append((config['train_config']['seed'], json.loads((path / 'learning_curve.json').read_text())))
    colors = ('#1565c0', '#c23b22', '#21814b', '#824bb3')
    width, height = 1140, 1110
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial,sans-serif" font-size="11" fill="#263238">',
           '<text x="24" y="28" font-size="19">Phase 3B-1: measured CPU MAPPO learning curves</text>',
           '<text x="24" y="49">Each point: one 256-step rollout / PPO update. Training task only; not generalization evidence.</text>']
    for index, (seed, _) in enumerate(series):
        x = 25 + index * 150
        svg += [f'<path d="M{x},68 h25" stroke="{colors[index % len(colors)]}" stroke-width="2"/>',
                f'<text x="{x + 32}" y="72">training seed {seed}</text>']
    for index, (key, title) in enumerate(PANELS):
        left, top = 65 + (index % 3) * 375, 120 + (index // 3) * 245
        w, h = 280, 162
        values = [r[key] for _, rows in series for r in rows if r.get(key) is not None]
        lo, hi = min(values), max(values)
        if key.endswith('rate') or key.endswith('ratio') or key == 'clip_fraction':
            lo = min(0, lo); hi = max(1, hi)
        elif lo == hi:
            lo -= .01 if hi == 0 else abs(hi) * .1
            hi += .01 if hi == 0 else abs(hi) * .1
        else:
            gap = (hi - lo) * .06; lo -= gap; hi += gap
        max_step = max(r['total_env_steps'] for _, rows in series for r in rows)
        svg.append(f'<text x="{left - 40}" y="{top - 16}" font-size="13">{html.escape(title)}</text>')
        for j in range(5):
            y, value = top + h * j / 4, hi - (hi - lo) * j / 4
            svg += [f'<path d="M{left},{y} h{w}" stroke="#e0e5e8"/>',
                    f'<text x="{left - 6}" y="{y + 4}" text-anchor="end">{value:.3g}</text>']
        for step in (0, max_step // 2, max_step):
            x = left + w * step / max_step
            svg.append(f'<text x="{x}" y="{top + h + 18}" text-anchor="middle">{step}</text>')
        svg.append(f'<text x="{left + w / 2}" y="{top + h + 34}" text-anchor="middle">environment steps</text>')
        for si, (_, rows) in enumerate(series):
            points = ' '.join(f"{left + w * r['total_env_steps'] / max_step:.2f},{top + h * (hi - r[key]) / (hi - lo):.2f}"
                              for r in rows if r.get(key) is not None)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[si % len(colors)]}" stroke-width="2"/>')
    svg += ['</g>', '</svg>']
    with output.open('x', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.experiment, args.output)

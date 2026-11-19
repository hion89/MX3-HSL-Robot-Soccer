"""Label an unchanged measured-curve SVG with the actual sensitivity settings."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def validation_curves(run, output):
    freeze = json.loads((run / 'selection_freeze.json').read_text())
    colors = ('#1565c0', '#c23b22', '#21814b', '#824bb3')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="420" viewBox="0 0 1100 420">',
           '<rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" font-size="12" fill="#263238">',
           '<text x="25" y="25" font-size="18">Phase 3B-2A: validation only, deterministic argmax</text>',
           '<text x="25" y="46">8 validation scenarios / point; no test-driven model selection.</text>']
    for i, entry in enumerate(freeze['runs']):
        x = 25 + 265 * i
        svg += [f'<path d="M{x},65 h22" stroke="{colors[i]}" stroke-width="2"/>',
                f'<text x="{x + 28}" y="69">{entry["stage"]}, gap weight {entry["coverage_weight"]:g}</text>']
    for panel, (key, title) in enumerate((('task_success_rate', 'Task success rate'), ('coverage_gap_ratio', 'Coverage gap ratio'))):
        left, top, w, h = 65 + panel * 540, 115, 440, 240
        svg.append(f'<text x="{left}" y="98" font-size="14">{title}</text>')
        for j in range(5):
            y = top + h * j / 4
            svg += [f'<path d="M{left},{y} h{w}" stroke="#e0e5e8"/>',
                    f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{1-j/4:g}</text>']
        for step in (0, 4096, 8192):
            x = left + w * step / 8192
            svg.append(f'<text x="{x}" y="{top + h + 20}" text-anchor="middle">{step}</text>')
        for i, entry in enumerate(freeze['runs']):
            points = ' '.join(f'{left+w*r["step"]/8192:.2f},{top+h*(1-r[key]):.2f}' for r in entry['validation_history'])
            svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="2.5"/>')
        svg.append(f'<text x="{left + 160}" y="{top + h + 42}">environment steps</text>')
    svg += ['</g></svg>']
    with output.open('x', encoding='utf-8') as f:f.write('\n'.join(svg))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path); group.add_argument('--run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.run:
        validation_curves(args.run, args.output)
        print(args.output)
        raise SystemExit(0)
    tree = ET.parse(args.input)
    legends = iter(('gap weight 2, seed 7', 'gap weight 8, seed 7'))
    for node in tree.iter('{http://www.w3.org/2000/svg}text'):
        if node.text and node.text.startswith('Phase 3B-1:'):
            node.text = 'Phase 3B-2A: randomized A / measured reward sensitivity'
        elif node.text == 'training seed 7':
            node.text = next(legends)
    with args.output.open('xb') as f:tree.write(f, encoding='utf-8', xml_declaration=True)
    print(args.output)

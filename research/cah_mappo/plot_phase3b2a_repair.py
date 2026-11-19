"""Standalone measured SVG curves and the physical-task slot intervention diagram."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.dont_write_bytecode=True


def render(run,output):
    from plot_phase3b1 import plot
    output.mkdir(parents=True,exist_ok=False)
    raw=output/'learning_curves_raw.svg'
    plot([run/'identity',run/'augmented'],raw)
    tree=ET.parse(raw);labels=iter(('identity continuation','stable slot augmentation'))
    for node in tree.iter('{http://www.w3.org/2000/svg}text'):
        if node.text and node.text.startswith('Phase 3B-1:'):
            node.text='Phase 3B-2A repair: measured Standard MAPPO CPU learning'
        elif node.text=='training seed 7':node.text=next(labels)
    with (output/'learning_curves_labeled.svg').open('xb') as f:tree.write(f,encoding='utf-8',xml_declaration=True)
    data=json.loads((run/'baseline_ladder.json').read_text())['fixed']['greedy']['episodes']
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="430" viewBox="0 0 850 430">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" fill="#263238">',
         '<text x="25" y="30" font-size="19">Slot-only intervention: unchanged physical tasks, old C checkpoint</text>',
         '<text x="25" y="53" font-size="12">All positions, IDs, spawn times, deadlines, priorities and robot states are identical.</text>',
         '<text x="25" y="78" font-size="12">Green: completed. Red: observed but never selected before deadline.</text>']
    for i,label in enumerate(('slot order','task0','task1','task2')):
        svg.append(f'<text x="{35+i*190}" y="110" font-size="14">{label}</text>')
    for row,e in enumerate(data):
        y=125+row*43
        svg.append(f'<text x="35" y="{y+24}" font-size="14">{e["order"]}</text>')
        for col,t in enumerate(e['task_timelines']):
            x=205+col*190; color='#2e7d32' if t['success'] else '#c62828'
            svg+= [f'<rect x="{x}" y="{y}" width="155" height="32" rx="4" fill="{color}"/>',
                    f'<text x="{x+10}" y="{y+21}" fill="white" font-size="12">{"completed" if t["success"] else "not selected"}</text>']
    svg+=['</g></svg>']
    with (output/'slot_only_physical_task_outcomes.svg').open('x',encoding='utf-8') as f:f.write('\n'.join(svg))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.run,a.output)

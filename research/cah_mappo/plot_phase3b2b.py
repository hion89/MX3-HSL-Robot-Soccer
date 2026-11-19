"""Measured standalone scientific SVG figures, no extra plotting dependency."""
import argparse
import html
import json
from pathlib import Path

COLORS=('#1565c0','#c23b22','#21814b','#824bb3','#a86605')
SEEDS=(101,211,307,401,503)


def learning_curves(run,path):
    panels=(('train_task_success_rate','Training task success'),('validation_greedy_task_success_rate','Validation greedy success'),
            ('validation_stochastic_task_success_rate','Validation stochastic success'),('train_episode_return','Training Episode Return'),
            ('validation_greedy_coverage_gap_ratio','Validation greedy CGR'),('validation_stochastic_coverage_gap_ratio','Validation stochastic CGR'),
            ('policy_entropy','Policy entropy'),('actor_loss','Actor loss'),('critic_loss','Critic loss'),
            ('approximate_kl','Approximate KL'),('clip_fraction','Clip fraction'),('critic_gradient_norm','Critic gradient norm before clip'))
    series=[]
    for seed in SEEDS:
        folder=run/f'seed_{seed}';curve=json.loads((folder/'learning_curve.json').read_text())
        initial=json.loads((folder/'validation_step000000.json').read_text());first={'total_env_steps':0}
        for mode in ('greedy','stochastic'):
            for metric in ('task_success_rate','coverage_gap_ratio'):first[f'validation_{mode}_{metric}']=initial[mode]['summary'][metric]
        series.append((seed,[first,*curve]))
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1110" viewBox="0 0 1200 1110">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" font-size="11" fill="#263238">',
         '<text x="24" y="28" font-size="19">Standard MAPPO: five independent from-scratch CPU learning runs</text>',
         '<text x="24" y="49">Raw measured update/validation points. Validation chooses checkpoints; benchmark is separate.</text>']
    for i,(seed,_) in enumerate(series):
        x=25+i*220;svg += [f'<path d="M{x},68 h25" stroke="{COLORS[i]}" stroke-width="2"/>',f'<text x="{x+32}" y="72">training seed {seed}</text>']
    for index,(key,title) in enumerate(panels):
        left,top=68+(index%3)*395,120+(index//3)*245;w,h=295,162
        values=[r[key] for _,rows in series for r in rows if r.get(key) is not None];lo,hi=min(values),max(values)
        if key.endswith('rate') or key.endswith('ratio') or key=='clip_fraction':lo=min(0,lo);hi=max(1,hi)
        elif lo==hi:lo-=.01;hi+=.01
        else:padding=(hi-lo)*.06;lo-=padding;hi+=padding
        maxstep=max(r['total_env_steps'] for _,rows in series for r in rows)
        svg.append(f'<text x="{left-40}" y="{top-16}" font-size="13">{html.escape(title)}</text>')
        for j in range(5):
            y=top+h*j/4;v=hi-(hi-lo)*j/4
            svg += [f'<path d="M{left},{y} h{w}" stroke="#e0e5e8"/>',f'<text x="{left-6}" y="{y+4}" text-anchor="end">{v:.3g}</text>']
        for step in (0,50000,100352):
            x=left+w*step/maxstep;svg.append(f'<text x="{x}" y="{top+h+18}" text-anchor="middle">{step}</text>')
        for si,(_,rows) in enumerate(series):
            points=' '.join(f'{left+w*r["total_env_steps"]/maxstep:.2f},{top+h*(hi-r[key])/(hi-lo):.2f}' for r in rows if r.get(key) is not None)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[si]}" stroke-width="1.6"/>')
        svg.append(f'<text x="{left+95}" y="{top+h+34}">environment steps</text>')
    svg+=['</g></svg>']
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(svg))


def comparisons(run,path):
    stats=json.loads((run/'statistics.json').read_text());rows=[]
    for mode in ('greedy','stochastic'):
        s=stats['seed_statistics']['overall'][mode]
        rows.append((f'MAPPO {mode}',s['task_success_rate']['mean'],s['task_success_rate']['t95'],s['coverage_gap_ratio']['mean'],s['coverage_gap_ratio']['t95'],'#1565c0'))
    for name,data in stats['references'].items():
        s=data['overall'];rows.append((name,s['task_success_rate'],None,s['coverage_gap_ratio'],None,'#21814b' if name.startswith('local_') else '#777777'))
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="550" viewBox="0 0 1250 550">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" font-size="12" fill="#263238">',
         '<text x="20" y="28" font-size="19">Controlled common benchmark: task success and responsibility coverage</text>',
         '<text x="20" y="50">MAPPO error bars: descriptive 95% t intervals over 5 training seeds. References are not trained replicates.</text>',
         '<text x="20" y="70">Gray central references have extra information; central_handover also has a preparation action.</text>']
    for panel,(title,idx,ciidx) in enumerate((('Task success rate',1,2),('Coverage Gap Ratio',3,4))):
        left,top,w,h=215+panel*620,125,350,340
        endpoints=[v for r in rows if r[ciidx] for v in r[ciidx]];lo=min([0,*endpoints]);hi=max([1,*endpoints])
        svg.append(f'<text x="{left}" y="105" font-size="15">{title}</text>')
        for j in range(6):
            x=left+w*j/5;v=lo+(hi-lo)*j/5
            svg += [f'<path d="M{x},{top} v{h}" stroke="#e0e5e8"/>',f'<text x="{x}" y="{top+h+23}" text-anchor="middle">{v:.3g}</text>']
        for i,r in enumerate(rows):
            y=top+i*40;x0=left+w*(0-lo)/(hi-lo);x=left+w*(r[idx]-lo)/(hi-lo)
            svg += [f'<text x="{left-8}" y="{y+20}" text-anchor="end">{html.escape(r[0])}</text>',
                    f'<rect x="{x0}" y="{y+4}" width="{max(0,x-x0)}" height="24" fill="{r[5]}"/>']
            if r[ciidx]:
                a,b=r[ciidx];xa=left+w*(a-lo)/(hi-lo);xb=left+w*(b-lo)/(hi-lo)
                svg.append(f'<path d="M{xa},{y+16} H{xb} M{xa},{y+10} V{y+22} M{xb},{y+10} V{y+22}" stroke="#111" fill="none"/>')
            svg.append(f'<text x="{min(left+w+8,x+8)}" y="{y+20}">{r[idx]:.4f}</text>')
    svg+=['</g></svg>']
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(svg))


def tradeoff(run,path):
    stats=json.loads((run/'statistics.json').read_text());points=[]
    for mode,color in (('greedy','#1565c0'),('stochastic','#c23b22')):
        s=stats['seed_statistics']['overall'][mode]
        points.append((f'MAPPO {mode}',s['coverage_gap_ratio']['mean']*100,s['response_time_mean_s']['mean'],
                       [v*100 for v in s['coverage_gap_ratio']['t95']],s['response_time_mean_s']['t95'],color))
    for name,r in stats['references'].items():
        s=r['overall'];label=name+(f' (success {s["task_success_rate"]*100:.1f}%)' if s['task_success_rate']<1 else '')
        points.append((label,s['coverage_gap_ratio']*100,s['response_time_mean_s'],None,None,
                       '#21814b' if name.startswith('local_') else '#777777'))
    left,top,w,h=85,100,780,500
    x=lambda value:left+w*value/22
    y=lambda value:top+h*(9-value)/6
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="690" viewBox="0 0 1120 690">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" font-size="12" fill="#263238">',
         '<text x="25" y="30" font-size="19">Coverage / response tradeoff: same 32-second benchmark</text>',
         '<text x="25" y="52">Error bars: descriptive t95 over 5 MAPPO training seeds. Gray central policies have extra information.</text>',
         '<text x="25" y="73">Response is conditional on a recorded response; task success is reported separately.</text>']
    for value in (0,5,10,15,20):
        xx=x(value);svg += [f'<path d="M{xx},{top} v{h}" stroke="#e0e5e8"/>',f'<text x="{xx}" y="{top+h+22}" text-anchor="middle">{value}</text>']
    for value in range(3,10):
        yy=y(value);svg += [f'<path d="M{left},{yy} h{w}" stroke="#e0e5e8"/>',f'<text x="{left-10}" y="{yy+4}" text-anchor="end">{value}</text>']
    for label,cgr,response,xci,yci,color in points:
        xx,yy=x(cgr),y(response)
        if xci:
            svg.append(f'<path d="M{x(xci[0])},{yy} H{x(xci[1])} M{xx},{y(yci[0])} V{y(yci[1])}" stroke="{color}" fill="none"/>')
        svg += [f'<circle cx="{xx}" cy="{yy}" r="5" fill="{color}"/>',
                f'<text x="{xx+10}" y="{yy-9}" fill="{color}">{html.escape(label)}</text>']
    svg += [f'<text x="{left+260}" y="{top+h+54}">Coverage Gap Ratio (%)</text>',
            '<text x="18" y="420" transform="rotate(-90 18 420)">Task Response Time (seconds)</text>','</g></svg>']
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(svg))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    learning_curves(a.run,a.output/'learning_validation_curves.svg');comparisons(a.run,a.output/'benchmark_comparison.svg')
    tradeoff(a.run,a.output/'coverage_response_tradeoff.svg')

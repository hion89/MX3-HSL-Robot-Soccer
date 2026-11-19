"""One-factor post-hoc diagnostic: lower sampling temperature, frozen B3 network.

No optimizer updates, model selection or holdout evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def probe(run,output):
    import torch
    from cah_mappo.phase3b2a.experiment import load_net,save
    from cah_mappo.phase3b2a_repair.evaluation import evaluate_one,summarize
    from cah_mappo.core import write_jsonl
    output.mkdir(parents=True,exist_ok=False)
    checkpoint=ROOT/'phase3b2a_outputs/pilot_v1/B3_weight2/checkpoints/step1024.pt'
    save(output/'probe_plan.json',{'status':'supplementary development diagnostic, not model selection',
        'checkpoint':str(checkpoint),'temperature':.5,'baseline_temperature':1.,
        'only_changed_condition':'sampling temperature','optimizer_steps':0,'holdout_evaluated':False})
    network=load_net(checkpoint)
    class Tempered:
        def distribution(self,obs,mask):
            return torch.distributions.Categorical(logits=network.distribution(obs,mask).logits/.5)
    manifest=json.loads((run/'scenario_manifest.json').read_text());episodes=[]
    logs=output/'logs';logs.mkdir()
    for spec in manifest['B3_development']:
        row,log=evaluate_one(spec,Tempered(),mode='stochastic')
        row['diagnostic_temperature']=.5;log[0]['diagnostic_temperature']=.5
        episodes.append(row);write_jsonl(log,logs/f"{spec['scenario_id']}.jsonl")
    result={'summary':summarize(episodes),'episodes':episodes,'temperature':.5,'weight_updates':0}
    save(output/'results.json',result)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();probe(args.run,args.output)

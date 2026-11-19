"""Actor-only local inference and explicitly privileged unchanged reference policies."""
from collections import Counter
import math
from pathlib import Path
import torch
from ..phase2.engine import Environment
from ..phase2.model import config_from_dict
from ..phase2.policies import make_policy
from ..phase3a.contracts import SampledAction
from ..phase3b2a.baselines import LocalPolicy
from ..phase3b2a.evaluation import actor_inputs
from ..phase3b2a.scenarios import digest
from ..phase3b2a_repair.environment import AuditedEnv
from .scenarios import native_env
from .metrics import score,aggregate
from .io import write_raw


@torch.no_grad()
def evaluate_one(spec,mode,network=None,order=(2,3,4),repeat=0,sampling_seed=7001):
    if spec['split'] not in ('train','validation','benchmark'):raise ValueError('Protected or unknown split rejected')
    if mode.startswith('central_'):
        config=config_from_dict(spec['config']);engine=Environment(config,make_policy(mode[8:],config),mode)
        rows=engine.run();result=score(rows)
        rows[0]['phase3b2b']={k:spec[k] for k in ('scenario_id','split','family','sha256','event_seed')}
        information='privileged_team_reference'
    else:
        if mode.startswith('local_') and mode not in ('local_fixed','local_eta'):raise ValueError('Only frozen local comparison policies are enabled')
        if tuple(order)==(2,3,4):env,obs=native_env(spec)
        else:env=AuditedEnv(spec,order);obs=env.obs
        policy=LocalPolicy(mode,sampling_seed) if mode.startswith('local_') else None
        if policy is None and (network is None or mode not in ('greedy','stochastic')):raise ValueError('Unknown policy')
        generator=torch.Generator(device='cpu').manual_seed(sampling_seed*100000000+spec['event_seed']+repeat*10000000)
        while env.agents:
            actions={};dist=None
            if policy is None:
                vectors,masks=actor_inputs(obs,env.settings);dist=network.distribution(vectors,masks)
                chosen=dist.probs.argmax(-1) if mode=='greedy' else torch.multinomial(dist.probs,1,generator=generator).squeeze(-1)
            for i,name in enumerate(env.settings.roster):
                packet=obs[name];action=policy.decide(name,packet) if policy else int(chosen[i])
                probs=tuple(dist.probs[i].tolist()) if mode=='stochastic' else tuple(float(j==action) for j in range(10))
                actions[name]=SampledAction(action,math.log(probs[action]),tuple(packet['action_mask']),probs,
                                           f'phase3b2b-eval-{mode}',packet['observation_token'])
            obs,*_=env.step(actions)
            if dist is not None:
                transition=next(r for r in reversed(env.engine.rows) if r['type']=='transition')
                transition['evaluation_model_probabilities']={name:dist.probs[i].tolist() for i,name in enumerate(env.settings.roster)}
        rows=env.log_rows();result=score(rows,env.episode_return);engine=env.engine
        information='actor_packets_only'
    rows[0]['evaluation']={'mode':mode,'repeat':repeat,'sampling_seed_base':sampling_seed,'slot_order':list(order),'information':information}
    result.update(scenario_id=spec['scenario_id'],family=spec['family'],split=spec['split'],mode=mode,repeat=repeat,
                  slot_order=list(order),information=information,exogenous_sha256=digest(engine.plan),event_seed=spec['event_seed'])
    return result,rows


def evaluate_many(specs,mode,network=None,orders=((2,3,4),),repeats=1,log_dir=None):
    if log_dir is not None:Path(log_dir).mkdir(parents=True,exist_ok=False)
    results=[]
    for spec in specs:
        for order in orders:
            for repeat in range(repeats):
                result,rows=evaluate_one(spec,mode,network,order,repeat)
                results.append(result)
                if log_dir is not None:
                    write_raw(Path(log_dir)/f"{spec['scenario_id']}_p{''.join(map(str,order))}_r{repeat}.jsonl.gz",rows)
    return {'summary':aggregate(results),'episodes':results}

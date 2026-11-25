"""Unchanged policies/networks; separately labeled privileged execution wrappers."""
import hashlib
import math
from pathlib import Path
import torch
from ..phase2.engine import Environment
from ..phase2.model import config_from_dict
from ..phase2.policies import make_policy
from ..phase3a.env import ParallelCore
from ..phase3a.contracts import Settings, SampledAction
from ..phase3b1.networks import ActorCritic
from ..phase3b2a.baselines import LocalPolicy
from ..phase3b2a.evaluation import actor_inputs
from ..phase3b2b.metrics import reference_rewards
from ..phase3b2a.scenarios import digest
from .scenarios import Witness
from .rules import Gate, install_gate
from .metrics import score


def load_network(path, expected_hash):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
        raise ValueError('Checkpoint provenance mismatch')
    checkpoint=torch.load(path,weights_only=True,map_location='cpu')
    net=ActorCritic();net.load_state_dict(checkpoint['network']);net.eval()
    return net


def header(spec):
    return {k:spec[k] for k in ('scenario_id','split','family','team','sha256','response_limit_s','feasibility_completion_at')}


@torch.no_grad()
def run(spec, policy, network=None, gate_mode=None, parameters=None, repeat=0, sampling_seed=4701):
    if spec['split'] not in ('validation','diagnostic'):
        raise ValueError('Protected/unknown split rejected before environment creation')
    gate=Gate(gate_mode,limits=spec['response_limit_s'],**(parameters or {})) if gate_mode else None
    c=config_from_dict(spec['config'])
    if policy.startswith('central_') or policy.startswith('witness_'):
        planner=make_policy(policy[8:],c) if policy.startswith('central_') else Witness(policy[8:])
        engine=Environment(c,planner,policy)
        if gate:install_gate(engine,gate)
        rows=engine.run();episode_return=reference_rewards(rows)
        information='privileged_current_team'
    else:
        env=ParallelCore(c,Settings(**{**spec['settings'],'roster':tuple(spec['settings']['roster'])}))
        obs,_=env.reset(spec['event_seed'],{'terminate_on_tasks_resolved':False})
        if gate:install_gate(env.engine,gate)
        local=LocalPolicy(policy,sampling_seed) if policy.startswith('local_') else None
        if not local and (network is None or policy not in ('mappo_greedy','mappo_stochastic')):
            raise ValueError('Unknown policy/network')
        rng=torch.Generator(device='cpu').manual_seed(sampling_seed*100000000+spec['event_seed']+repeat*10000000)
        while env.agents:
            dist=None;actions={}
            if not local:
                vectors,masks=actor_inputs(obs,env.settings);dist=network.distribution(vectors,masks)
                chosen=dist.probs.argmax(-1) if policy=='mappo_greedy' else torch.multinomial(dist.probs,1,generator=rng).squeeze(-1)
            for name in env.agents:
                packet=obs[name];i=env.settings.roster.index(name)
                action=local.decide(name,packet) if local else int(chosen[i])
                probs=tuple(dist.probs[i].tolist()) if not local else tuple(float(j==action) for j in range(10))
                actions[name]=SampledAction(action,math.log(probs[action]),tuple(packet['action_mask']),probs,
                                           'phase4a_frozen_policy',packet['observation_token'])
            obs,*_=env.step(actions)
        rows=env.log_rows();episode_return=env.episode_return;engine=env.engine
        information='actor_packets_only'
    rows[0]['phase4a']=header(spec)
    rows[0]['phase4a_policy']={'policy':policy,'gate':gate_mode,'parameters':parameters,'repeat':repeat,
                              'information':information,'gate_information':'privileged_current_team' if gate else None,
                              'preparation_action_extension':bool(gate),'sampling_seed':sampling_seed}
    result=score(rows)
    result.update(episode_return=episode_return,scenario_id=spec['scenario_id'],family=spec['family'],team=spec['team'],
                  policy=policy,gate=gate_mode,repeat=repeat,information=information,
                  gate_information='privileged_current_team' if gate else None,
                  exogenous_sha256=digest(engine.plan))
    return result,rows

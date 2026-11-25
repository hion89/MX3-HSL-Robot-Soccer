"""History-level information audit and explicitly privileged ablations."""
import copy
from collections import Counter
from ..phase3b2b.io import save,read_raw,write_raw
from ..phase3b2a.scenarios import digest
from ..phase4b.experiment import read,POLICIES
from .runs import execute
from .policies import BranchHintPolicy


def decision_prefix(rows,before=2.):
    return [{'time':r['time'],'public':r['public_observation'],
             'requested':r['action']['requested'],
             'metadata':r['action'].get('policy_metadata',{})}
            for r in rows if r.get('event')=='policy_selection' and r['time']<before]


def compare_histories(left,right):
    a,b=decision_prefix(left),decision_prefix(right)
    return {'identical':a==b,'decisions_each':[len(a),len(b)],'sha256_each':[digest(a),digest(b)]}


def run_information(root,output):
    import torch
    from phase4b_feedback import SensorActor
    from ..phase4a.evaluation import load_network
    torch.set_num_threads(1)
    target=output/'information';target.mkdir(exist_ok=False)
    old=root/'phase4b_outputs/study_v1';worlds=read(old/'design/diagnostic_manifest.json')
    index={(s['family'],s['team'],s['index']):s for s in worlds}
    native=[]
    for policy in POLICIES:
        for team in (2,4):
            for i in range(6):
                specs=[index[f,team,i] for f in ('A_near','D_visible')]
                histories=[read_raw(old/'comparison'/policy/'raw'/f"{s['scenario_id']}_r0.jsonl.gz") for s in specs]
                result=compare_histories(*histories)
                assert result['identical'],(policy,team,i)
                native.append({'policy':policy,'team':team,'index':i,**result})
    save(target/'native_history_audit.json',{'pairs':native,'all_identical':True,'count':len(native),
        'scope':'entire available control-input/action/metadata history before first task spawn, not only instantaneous observation'})
    frozen=read(old/'design/freeze.json');actors=[];folder=target/'actor_common_rng';folder.mkdir()
    for seed,cp in frozen['checkpoints'].items():
        network=load_network(root/cp['path'],cp['sha256'])
        for mode in ('greedy','stochastic'):
            for team in (2,4):
                for i in range(6):
                    histories=[];metrics=[]
                    for family in ('A_near','D_visible'):
                        s=index[family,team,i]
                        # Same sampling seed in both branches; future scenario seed is never an Actor input.
                        p=SensorActor(network,mode,int(seed)*10000+team*100+i)
                        m,rows=execute(s,p,feedback=True);metrics.append(m);histories.append(rows)
                        write_raw(folder/f"seed{seed}_{mode}_{s['scenario_id']}.jsonl.gz",rows)
                    c=compare_histories(*histories);assert c['identical'],(seed,mode,team,i)
                    actors.append({'seed':seed,'mode':mode,'team':team,'index':i,**c,'outcomes':metrics})
        print('history replay frozen model',seed,flush=True)
    save(target/'actor_history_audit.json',{'pairs':actors,'count':len(actors),'all_identical':True,
        'matched_sampling_rng':True,'compared':'full Perception packets, slot IDs, masks, features, observation tokens, probabilities, sampled actions and public history',
        'future_events_not_input':True})
    folder=target/'future_hint_ablation';folder.mkdir();records=[]
    for i in range(6):
        for family in ('A_near','D_visible'):
            s=index[family,2,i];task=next(j for j in s['config']['jobs'] if j['id']=='task0')
            for hint in (False,True):
                p=BranchHintPolicy(copy.deepcopy(task) if hint else None)
                m,rows=execute(s,p);m['privileged_future_announcement']=hint;records.append(m)
                write_raw(folder/f"{s['scenario_id']}_hint{int(hint)}.jsonl.gz",rows)
    save(target/'future_hint_ablation.json',{'episodes':records,
        'interpretation':'future announcement deliberately supplies absent information; privileged counterfactual, not an executable online baseline',
        'same_physics_and_action_constraints':True})
    save(target/'predictability_limits.json',{
        'native_matched_pairs':len(native),'frozen_actor_matched_pairs':len(actors),
        'balanced_branch_label_accuracy_upper_bound_given_identical_prefix':.5,
        'scope':'only these balanced paired branch labels before t=2; not a general RoboCup prediction bound',
        'observed_cues_that_distinguish_pair':[],
        'future_job_location_deadline_not_available_before_spawn':True,
        'real_ball_trajectory_cues':'possible in RoboCup, but no paired future-task-labelled ground truth dataset was found; predictive performance unmeasured',
        'causal_caution':'a hidden branch does not prove every individual policy failure is caused solely by missing information'})

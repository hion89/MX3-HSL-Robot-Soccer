"""Post-hoc development diagnostic: fault-aware release after observed threat passage.

Phase 4B's frozen READY rule kept guarding after a successor fault even after
the last observed attack had arrived. Test an explicit public-information rule
and preserve the old results. No threat/duty is removed from the evaluator.
The new rule is also checked on new development indices 6..11, not final holdout.
"""
import copy
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.phase2.model import Action,eta
from cah_mappo.phase2.policies import ExactAssignment
from cah_mappo.phase4b.policies import Native
from cah_mappo.phase4b.scenarios import scenario
from cah_mappo.phase4b.planning import OfflineSchedule
from cah_mappo.phase4b.engine import Simulation
from cah_mappo.phase4b.metrics import evaluate_new
from cah_mappo.phase4b.experiment import read,sha,result,grouped,ensure_frozen
from cah_mappo.phase3b2b.io import save,write_raw


class FaultAwareRelease:
    name='fault_aware_ready'
    information='same_current_public_team_and_visible_threats'
    def reset(self,seed):
        self.ready=Native('ready');self.ready.reset(seed);self.exact=ExactAssignment(2,50);self.exact.reset(seed);self.last={}
    def decide(self,view):
        action=self.ready.decide(view);self.last=copy.deepcopy(self.ready.last)
        obs=view.allocation
        if all(r.operational for r in obs.robots):return action
        candidate=dict(self.exact.decide(obs).assignments);allocation=dict(action.assignments);jobs={j.id:j for j in obs.jobs}
        changes=[]
        for base in (j for j in obs.jobs if j.kind=='base'):
            old=next((r for r in obs.robots if r.id==base.owner and r.operational and r.assignment==base.id),None)
            if old is None:continue
            task=jobs.get(candidate.get(old.id))
            if task is None or task.kind!='task' or allocation.get(old.id)!=base.id:continue
            if any(q['target']==base.id for q in view.threats):continue
            if obs.time+eta(old,task,obs)+task.remaining>task.deadline+1e-9:continue
            for name in allocation:
                if name!=old.id and allocation[name]==task.id:allocation[name]=None
            allocation[old.id]=task.id;changes.append({'incumbent':old.id,'base':base.id,'task':task.id})
        self.last['fault_release_without_known_active_threat']=changes
        return Action(tuple(sorted(allocation.items())),action.preparations,
                      ('persistent_responsibility_still_evaluated','public_fault_contingency'))


def main(output):
    frozen=ensure_frozen(ROOT,output);target=output/'fault_contingency';target.mkdir(exist_ok=False)
    old=[s for s in read(output/'design/diagnostic_manifest.json') if s['family'].startswith('E_')]
    fresh=[scenario('diagnostic',f,n,i) for f in ('E_incumbent','E_successor') for n in (2,4) for i in range(6,12)]
    save(target/'protocol.json',{'scope':'post-hoc mechanism diagnosis, not retuning frozen baselines',
        'old_cases_are_development':True,'fresh_development_cases':len(fresh),'new_parameters':None,
        'persistent_duty_not_terminated':True,'future_threats_unknown':True,'source_sha256':sha(Path(__file__))})
    summaries={}
    for bank,specs in (('old_development',old),('new_development',fresh)):
        for name in ('ready','lookahead','fault_aware_ready'):
            folder=target/f'{bank}_{name}';folder.mkdir();records=[]
            for s in specs:
                policy=FaultAwareRelease() if name=='fault_aware_ready' else Native(name,**frozen['parameters'])
                m,rows=result(s,policy);records.append(m);write_raw(folder/f"{s['scenario_id']}.jsonl.gz",rows)
            save(folder/'results.json',{'episodes':records,'summary':grouped(records)})
            summaries[f'{bank}_{name}']=grouped(records)
    # Oracle knows future failures; restricted schedules establish feasible fault cases only.
    oracle=target/'offline_fault';oracle.mkdir();cases=[];count=0
    for s in old:
        if s['team']!=2:continue
        folder=oracle/s['scenario_id'];folder.mkdir();joint=[];task=[]
        deadline=next(j['deadline'] for j in s['config']['jobs'] if j['id']=='task0')
        for prepare in (0,1,2):
            for depart in range(2,min(15,int(deadline)+1)):
                rows=Simulation(s,OfflineSchedule(prepare,depart)).run();m=evaluate_new(rows);count+=1
                write_raw(folder/f'p{prepare}_d{depart}.jsonl.gz',rows)
                if m['joint_task_and_defense_success']:joint.append((prepare,depart))
                if m['task_success_rate']==1:task.append((prepare,depart))
        cases.append({'scenario_id':s['scenario_id'],'family':s['family'],'has_joint_witness':bool(joint),
                      'has_task_witness':bool(task),'joint_schedule_examples':joint[:3]})
    save(target/'summary.json',summaries);save(oracle/'summary.json',{'executions':count,'cases':cases,
        'scope':'bounded future-aware witnesses; absent witness is not an impossibility proof'})
    print('fault contingency saved',len(old)+len(fresh),'worlds;',count,'oracle schedules')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'phase4b_outputs/study_v1')
    main(parser.parse_args().output)

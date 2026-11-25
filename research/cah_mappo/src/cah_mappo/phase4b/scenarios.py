"""Development banks only: inherit physical cases, never tune deadlines after seeing outcomes."""
import copy
from dataclasses import replace,asdict
from ..phase4a.scenarios import scenario as phase4a_case
from ..phase2.model import config_from_dict,JobSpec,EventSpec
from ..phase3b2a.scenarios import digest

VERSION='phase4b-validity-v1'
FAMILIES=('A_quiet','A_near','B_normal','C_reactive','D_visible','E_incumbent','E_successor',
          'R_changed','P_post_task','I_capacity')


def scenario(split,family,team,index):
    if split not in ('validation','diagnostic') or family not in FAMILIES or team not in (2,4) or not 0<=index<1000:
        raise ValueError('Unknown/protected split or scenario')
    case_index=(100 if split=='validation' else 200)+index
    source_family='A' if family in ('A_quiet','A_near','P_post_task','I_capacity') else 'B' if family in ('B_normal','E_incumbent','E_successor') else 'C'
    source=phase4a_case('diagnostic',source_family,team,case_index)
    c=config_from_dict(source['config']);events=[];jobs=list(c.jobs);worker='B' if team==2 else 'C'
    if family=='A_near':
        point=next(r.position for r in c.robots if r.id==worker)
        jobs=[replace(j,position=point,deadline=4) if j.id=='task0' else j for j in jobs]
    if family=='I_capacity':
        bases=[j for j in jobs if j.kind=='base'];point=bases[0].position
        jobs=bases+[JobSpec(f'task{i}',point,spawn=2,deadline=4,service=2,priority=2,
                           fixed_owner=worker) for i in range(team+1)]
    if family=='E_incumbent':events=[EventSpec(2,'failure','A')]
    if family=='E_successor':events=[EventSpec(3,'failure',worker)]
    if family.startswith('E_') and index%2:events.append(EventSpec(10,'recovery','A' if family=='E_incumbent' else worker))
    if family=='R_changed':events.append(EventSpec(8,'risk','base1' if team==4 else 'base0',4))
    seed=(51000000 if split=='validation' else 52000000)+team*10000+FAMILIES.index(family)*1000+index
    c=replace(c,name=f'{VERSION}-{split}-{family}-{team}-{index}',jobs=tuple(jobs),events=tuple(events),seed=seed,dt=.25)
    c.validate();base=next(j for j in c.jobs if j.id=='base0')
    threats=[]
    def add(qid,target,start,reveal,travel=4,intensity=1):
        pos=next(j.position for j in c.jobs if j.id==target)
        threats.append({'id':qid,'target':target,'origin':[pos[0]+travel,pos[1]],'start':start,
                        'reveal':reveal,'speed':1.,'intensity':float(intensity)})
    if family!='A_quiet':
        if family=='P_post_task':add('q0','base0',16,16)
        elif family=='C_reactive':add('q0','base0',0,2)
        elif family=='E_successor':add('q0','base0',0,0,travel=6)
        elif family=='B_normal':add('q0','base0',0,0,travel=8)
        else:add('q0','base0',0,0)
    if family=='R_changed':add('q1','base1' if team==4 else 'base0',8,8,intensity=2)
    data={'version':VERSION,'split':split,'family':family,'team':team,'index':index,'event_seed':seed,
          'scenario_id':c.name,'config':asdict(c),'threats':threats,'control_period':1.,
          'response_budgets':{j.id:2. for j in c.jobs if j.kind=='base'},
          'nominal_phase4a_class':source_family,'inherited_feasibility_times_dt1':source['feasibility_completion_at'],
          'assumptions':'all speeds, straight-ball proxy, budgets and occupancy durations are simulation assumptions, not measured robot/goal-risk calibration'}
    if family=='I_capacity':
        data['infeasibility_certificate']={'type':'required_service_exceeds_total_robot_time',
            'required_robot_seconds':2*(team+1),'available_robot_seconds':2*team,
            'scope':'all tasks due at t=4, released t=2; impossible even ignoring movement and defense'}
    return {**data,'sha256':digest(data)}


def bank(split,count):return [scenario(split,family,team,i) for family in FAMILIES for team in (2,4) for i in range(count)]

from dataclasses import dataclass, asdict
import itertools
import math


@dataclass(frozen=True)
class PublicView:
    allocation: object
    services: dict
    threats: tuple
    response_budgets: dict
    horizon: float

    def record(self):
        return {'allocation':asdict(self.allocation),'services':dict(self.services),
                'threats':list(self.threats),'response_budgets':dict(self.response_budgets),'horizon':self.horizon}


def arrival(threat, jobs):
    return threat['start']+math.dist(threat['origin'],jobs[threat['target']]['position'])/threat['speed']


def threat_state(threat, jobs, t):
    target=jobs[threat['target']]['position'];distance=math.dist(threat['origin'],target)
    direction=tuple((b-a)/distance for a,b in zip(threat['origin'],target))
    elapsed=max(0.,min(t-threat['start'],distance/threat['speed']))
    point=tuple(round(a+v*threat['speed']*elapsed,10) for a,v in zip(threat['origin'],direction))
    return {'id':threat['id'],'target':threat['target'],'position':point,'ball_position':point,
            'direction':direction,'speed':threat['speed'],'intensity':threat['intensity'],
            'estimated_arrival':round(t+math.dist(point,target)/threat['speed'],10),'observed_at':t}


def travel_time(robot, position, radius):
    distance=math.dist(robot['position'],position)
    if distance<=radius+1e-9:return 0.
    angle=math.atan2(position[1]-robot['position'][1],position[0]-robot['position'][0])
    rotation=abs(math.remainder(angle-robot['heading'],2*math.pi))/robot['turn_speed']
    return rotation+(distance-radius)/robot['speed']


def free(robot, state):
    job=state['jobs'].get(robot['assignment'])
    return robot['operational'] and (job is None or job['kind']=='base' or job['status']!='active')


def response_time(robot,base,state):
    """Finish an active finite commitment before responding; no simultaneous work.

    This is a deterministic feasibility estimate, not a promise the controller
    will choose the response. Unknown future faults are deliberately absent.
    """
    if not robot['operational']:return math.inf
    position=state['jobs'][base]['position'];r=dict(robot)
    job=state['jobs'].get(r['assignment']);wait=0.;now=state.get('time',0.)
    if job and job['kind']=='task' and job['status']=='active':
        wait=travel_time(r,job['position'],state['radius'])+max(0.,job['service']-job['work'])
        if now+wait>job['deadline']+1e-9:return math.inf
        d=math.dist(r['position'],job['position'])
        if d>state['radius']+1e-9:
            r['heading']=math.atan2(job['position'][1]-r['position'][1],job['position'][0]-r['position'][0])
        r['position']=job['position'];r['assignment']=None
        period=state.get('control_period',1.)
        wait+=max(0.,math.ceil((now+wait)/period-1e-9)*period-(now+wait))
    travel=travel_time(r,position,state['radius'])
    if travel>0 and wait==0 and r['assignment']!=base:wait+=state.get('decision_delay',0.)
    return wait+travel


def match(state, targets, budgets, weighted=False, at_point=False):
    """Capacity-one maximum matching; each robot covers at most one target.

    targets is a mapping of query IDs to (base ID, importance). Potential ETA
    coverage is distinct from at-point defense at the actual arrival event.
    """
    ids=sorted(targets);robots=state['robots'];options=[]
    for key in ids:
        base,_=targets[key];position=state['jobs'][base]['position']
        eligible=[]
        for name,r in sorted(robots.items()):
            if not r['operational']:continue
            reachable=(free(r,state) and math.dist(r['position'],position)<=state['radius']+1e-9 if at_point else
                       response_time(r,base,state)<=budgets[key]+1e-9)
            if reachable:eligible.append(name)
        options.append((None,*eligible))
    best_key=(-1,-1);best={}
    for vector in itertools.product(*options):
        used=[v for v in vector if v is not None]
        if len(set(used))!=len(used):continue
        count=len(used);weight=sum(targets[k][1] for k,v in zip(ids,vector) if v is not None)
        rank=(weight,count) if weighted else (count,weight)
        if rank>best_key:best_key=rank;best={k:v for k,v in zip(ids,vector) if v is not None}
    return best

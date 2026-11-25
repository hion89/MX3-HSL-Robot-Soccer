"""New runner: frozen movement/service dynamics, explicit arrivals, fixed control clock."""
import copy
from dataclasses import asdict, replace
import math
from ..phase2.engine import Environment
from ..phase2.model import config_from_dict
from .model import PublicView,arrival,threat_state,match


class Simulation(Environment):
    def __init__(self,spec,policy,physics_dt=None):
        if spec['split'] not in ('validation','diagnostic','unit'):
            raise ValueError('Protected/unknown data rejected before simulation')
        self.spec=copy.deepcopy(spec);self.control_period=spec.get('control_period',1.)
        config=config_from_dict(spec['config'])
        if physics_dt is not None:config=replace(config,dt=physics_dt)
        if abs(self.control_period/config.dt-round(self.control_period/config.dt))>1e-9:
            raise ValueError('Control clock must be a multiple of physics dt')
        super().__init__(config,policy,getattr(policy,'name',type(policy).__name__))
        self.threats=copy.deepcopy(spec['threats']);self.arrived=set();self.observed=set()
        for q in self.threats:
            if (q['target'] not in self.jobs or self.jobs[q['target']]['kind']!='base'
                    or not all(math.isfinite(q[k]) for k in ('speed','start','reveal','intensity'))
                    or q['speed']<=0 or q['intensity']<=0 or not 0<=q['start']<=q['reveal']<=arrival(q,self.jobs)<=config.horizon
                    or math.dist(q['origin'],self.jobs[q['target']]['position'])<=0):raise ValueError('Invalid threat')
            times=(q['start'],q['reveal'],arrival(q,self.jobs))
            if any(abs(t/config.dt-round(t/config.dt))>1e-8 for t in times):raise ValueError('Threat times must align with physics grid')
        if len({q['id'] for q in self.threats})!=len(self.threats):raise ValueError('Duplicate threat IDs')

    def public(self,t):
        obs=replace(self.observe(t),dt=self.control_period)
        threats=[]
        for q in self.threats:
            if q['reveal']<=t+1e-9 and q['id'] not in self.arrived:
                threats.append(threat_state(q,self.jobs,t))
                if q['id'] not in self.observed:
                    self.event(t,'threat_observed',threat=q['id']);self.observed.add(q['id'])
        return PublicView(obs,{j.id:self.jobs[j.id]['service'] for j in obs.jobs},tuple(threats),
                          dict(self.spec['response_budgets']),self.c.horizon)

    def process_threats(self,t):
        due=[q for q in self.threats if q['id'] not in self.arrived and abs(arrival(q,self.jobs)-t)<1e-8]
        targets={q['id']:(q['target'],q['intensity']) for q in due}
        defended=match(self.truth(),targets,{k:0. for k in targets},weighted=True,at_point=True)
        for q in due:
            self.arrived.add(q['id']);self.event(t,'threat_defended' if q['id'] in defended else 'threat_passed',
                threat=q['id'],base=q['target'],robot=defended.get(q['id']),intensity=q['intensity'],
                arrival_truth=self.truth())

    def run(self):
        if self.rows:raise RuntimeError('Run each simulation only once')
        self.rows.append({'type':'header','schema_version':2,'policy':self.policy_name,
            'config':asdict(self.c),'exogenous_plan':self.plan,
            'phase4b':{'version':'phase4b-coverage-v1','scenario_id':self.spec['scenario_id'],
                      'family':self.spec['family'],'split':self.spec['split'],'team':len(self.c.robots),
                      'threat_plan':self.threats,'response_budgets':self.spec['response_budgets'],
                      'control_period':self.control_period,'physics_dt':self.c.dt,
                      'information':getattr(self.policy,'information','current_public_team'),
                      'assumption':'straight dribbler/ball proxy, automatic at-point blocking; not goal probability'}})
        action=None;period_ticks=round(self.control_period/self.c.dt)
        for tick in range(round(self.c.horizon/self.c.dt)+1):
            t=round(tick*self.c.dt,10);self.boundary(t);self.update_owners(t)
            # Completion/failure boundary is processed before arrival; control reacts after arrival.
            self.process_threats(t)
            if t>=self.c.horizon-1e-9:
                self.rows.append({'type':'terminal','time':t,'truth':self.truth()});break
            decision=tick%period_ticks==0
            if decision:
                public=self.public(t);action=self.policy.decide(public)
            selection=self.apply(t,action)
            if decision:
                selection['policy_metadata']=copy.deepcopy(getattr(self.policy,'last',{}))
                self.event(t,'policy_selection',public_observation=public.record(),action=copy.deepcopy(selection))
            for job in self.jobs.values():
                if job['kind']=='task' and job['status']=='active' and job['response'] is None:
                    ready=[r for r in self.robots.values() if r['operational'] and r['assignment']==job['id']
                           and math.dist(r['position'],job['position'])<=self.c.radius+1e-9]
                    if ready:job['response']=t;self.event(t,'task_response_started',job=job['id'],robot=ready[0]['id'])
            self.rows.append({'type':'interval','tick':tick,'start':t,'end':round(t+self.c.dt,10),
                              'truth':self.truth(),'selection':selection,
                              'threat_truth':[threat_state(q,self.jobs,t) for q in self.threats
                                              if q['start']<=t<arrival(q,self.jobs)]})
            self.integrate(t)
        return self.rows

"""Versioned procedural streams, separate validation and non-final benchmark banks."""
from dataclasses import asdict
import random
from ..phase2.model import Config,RobotSpec as R,JobSpec as J,config_from_dict
from ..phase3a.contracts import Settings
from ..phase3a.env import ParallelCore
from ..phase3a.rewards import RewardWeights
from ..phase3b2a.scenarios import digest

VERSION='phase3b2b-controlled-v1'
SEEDS=(101,211,307,401,503)
STEPS=100352
INTERVAL=8192
FAMILIES=('positions','conditions')


def scenario(split,index,stream=0,family='conditions'):
    if split not in ('train','validation','benchmark'):raise ValueError('Only train/validation/non-final benchmark banks permitted')
    if not isinstance(index,int) or not 0<=index<100000 or not 0<=stream<5 or family not in FAMILIES:raise ValueError('Invalid scenario index')
    bank={'train':10000000,'validation':20000000,'benchmark':30000000}[split]
    seed=bank+stream*100000+FAMILIES.index(family)*1000000+index
    rng=random.Random(f'{VERSION}:{seed}')
    robots=[]
    for i,xy in enumerate(((0.,1.5),(0.,-1.5),(-2.5,1.5),(-2.5,-1.5))):
        scale=.05 if i<2 else .75
        robots.append(R(chr(65+i),tuple(v+rng.uniform(-scale,scale) for v in xy),
                        initial_job=f'base{i}' if i<2 else None,base_role='guard' if i<2 else 'worker',heading=rng.uniform(-.2,.2)))
    jobs=[J(f'base{i}',xy,kind='base',spawn=0,deadline=32,service=0,priority=1,fixed_owner=chr(65+i))
          for i,xy in enumerate(((0.,1.5),(0.,-1.5)))]
    count=rng.choices((1,2,3),(.2,.2,.6))[0] if split=='train' else 3
    for i,xy in enumerate(((3.,1.),(3.,0.),(3.,-1.))[:count]):
        jobs.append(J(f'task{i}',tuple(v+rng.uniform(-.75,.75) for v in xy),spawn=2+2*i,
                      priority=rng.choice((1.,2.,4.)) if family=='conditions' else 2.,
                      deadline=rng.choice((20.,24.,28.)) if family=='conditions' else 24.,
                      service=2,fixed_owner=chr(67+i%2)))
    config=Config(f'{VERSION}-{split}-{family}-{stream}-{index}',tuple(robots),tuple(jobs),horizon=32,seed=seed)
    config.validate()
    payload={'generator':VERSION,'split':split,'family':family,'index':index,'stream':stream,'event_seed':seed,
             'config':asdict(config),'settings':asdict(Settings(sensor_radius=100,sensor_noise=0,radio_delay=0))}
    return {**payload,'scenario_id':f'{split}-{family}-{stream}-{index:05d}-{digest(payload)[:12]}','sha256':digest(payload)}


def native_env(spec):
    if spec['split'] not in ('train','validation','benchmark'):raise ValueError('Protected/unknown split rejected')
    env=ParallelCore(config_from_dict(spec['config']),Settings(**{**spec['settings'],'roster':tuple(spec['settings']['roster'])}),RewardWeights())
    obs,_=env.reset(spec['event_seed'],{'terminate_on_tasks_resolved':False})
    env.engine.rows[0]['phase3b2b']={k:spec[k] for k in ('scenario_id','sha256','split','family','stream','event_seed')}
    return env,obs


def validation_specs():return [scenario('validation',i,family=f) for f in FAMILIES for i in range(16)]
def benchmark_specs():return [scenario('benchmark',i,family=f) for f in FAMILIES for i in range(200)]

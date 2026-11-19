"""Cost gate, frozen protocol, hidden local worker processes and evaluation barrier."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import torch
from ..phase3b1.learner import TrainConfig
from ..phase3a.rewards import RewardWeights
from ..phase3b2a.scenarios import digest
from .scenarios import VERSION,SEEDS,STEPS,INTERVAL,FAMILIES,validation_specs
from .learner import BaselineLearner
from .evaluation import evaluate_many
from .io import save


def calibrate(output):
    output.mkdir(parents=True,exist_ok=False)
    learner=BaselineLearner(0,999)
    start=time.perf_counter()
    for _ in range(2):learner.update(learner.collect(256))
    seconds_per_step=(time.perf_counter()-start)/512
    start=time.perf_counter()
    sample=validation_specs()[:4]
    evaluate_many(sample,'stochastic',learner.net,log_dir=output/'raw_sample')
    seconds_per_eval=(time.perf_counter()-start)/len(sample)
    bytes_per_eval=sum(p.stat().st_size for p in (output/'raw_sample').glob('*.gz'))/len(sample)
    validation_episodes=5*14*32*2
    benchmark_episodes=5*400*4+6*400+5*32*5*2
    serial_seconds=seconds_per_step*STEPS*5+seconds_per_eval*(validation_episodes+benchmark_episodes)
    forecast_bytes=bytes_per_eval*benchmark_episodes+STEPS*5*1800+250*1024**2
    report={'calibration_steps':512,'calibration_seed':999,'seconds_per_training_step':seconds_per_step,
            'seconds_per_evaluation_episode':seconds_per_eval,'compressed_bytes_per_evaluation_episode':bytes_per_eval,
            'forecast_serial_seconds':serial_seconds,'forecast_disk_gib':forecast_bytes/1024**3,
            'planned_seeds':list(SEEDS),'steps_per_seed':STEPS,'checkpoint_interval':INTERVAL,'validation_interval':INTERVAL,
            'validation_scenarios':32,'benchmark_physical_scenarios':400,'stochastic_repeats':3,
            'slot_diagnostic_physical_scenarios':32,'slot_permutations':6,'forecast_evaluation_episodes':benchmark_episodes,
            'maximum_parallel_workers':5,'forecast_is_estimate':True,'budget_gate':serial_seconds<=8*3600 and forecast_bytes<=20*1024**3,
            'free_disk_gib':shutil.disk_usage(output).free/1024**3,'python':sys.version,'torch':torch.__version__,
            'numpy':importlib.metadata.version('numpy'),'device':'cpu','torch_threads':torch.get_num_threads()}
    save(output/'cost_forecast.json',report);print(json.dumps(report,indent=2),flush=True)


def prepare(output,calibration,root):
    cost=json.loads((calibration/'cost_forecast.json').read_text())
    if not cost['budget_gate']:raise RuntimeError('Forecast exceeds previous 8-hour/20-GiB envelope; obtain approval before expanding')
    output.mkdir(parents=True,exist_ok=False)
    plan={'protocol':VERSION,'date':'2026-11-19','seeds':list(SEEDS),'steps_per_seed':STEPS,
          'initialization':'independent from scratch','stream_independence':'separate policy/init RNG and nonoverlapping per-seed training event streams',
          'actor':[146,64,64,10],'critic':[141,64,64,1],'activation':'Tanh','shared_actor':True,
          'ppo_hyperparameters':{k:v for k,v in asdict(TrainConfig()).items() if k not in ('seed','environment_seed')},
          'reward_weights':asdict(RewardWeights()),
          'environment':{'robots':4,'capacity':1,'tasks_train':'1/2/3 with .2/.2/.6 probability','tasks_benchmark':3,
                         'task_spawn_times':[2,4,6],'service_seconds':2,'task_priorities':[1,2,4],
                         'task_deadlines':[20,24,28],'position_halfwidth_m':.75,'guard_position_halfwidth_m':.05,
                         'base_positions':[[0,1.5],[0,-1.5]],'base_priorities':[1,1],'horizon_seconds':32,
                         'sensor_radius':100,'sensor_noise':0,'radio_delay':0,'failures':False},
          'mask':'0 NOOP always valid; 1 keep; 2..9 locally known fresh job slots; no inactive actor loss',
          'termination':'32s truncation, bootstrap1 and continuation0; no early workload end',
          'cgr':'capacity-feasible physical matching, two persistent priority1 bases; denominator64 region-seconds',
          'checkpoint_interval':INTERVAL,'validation_interval':INTERVAL,'validation_scenarios':32,
          'selection':'min greedy/stochastic success, mean success, -worst CGR, -mean restricted response',
          'primary_slot_augmentation':False,'benchmark_families':list(FAMILIES),'episodes_per_family':200,
          'stochastic_repeats':3,'slot_probe_first_conditions_scenarios':32,'slot_probe_repeats':1,
          'reference_policies':['local_fixed','local_eta','central_fixed','central_greedy','central_exact','central_handover'],
          'exact_assignment_internal_costs':{'switching_cost':2.,'unassigned_cost':50.},
          'reference_information':'local actor packets vs privileged team reference; central_handover has extra preparation action',
          'holdout':'existing 400 remain sealed; not read; not evaluated; benchmark is not final holdout',
          'no_tuning_after_benchmark':True,'cpu_only':True,'parallel_worker_cap':5,
          'cost_forecast':cost}
    save(output/'frozen_plan.json',plan)
    save(output/'validation_manifest.json',validation_specs())
    source=[]
    for folder in ('phase2','phase3a','phase3b1','phase3b2b'):
        source.extend((root/'src/cah_mappo'/folder).glob('*.py'))
    for name in ('phase3b2a/baselines.py','phase3b2a/evaluation.py','phase3b2a/scenarios.py',
                 'phase3b2a_repair/environment.py','phase3b2a_repair/slots.py'):
        source.append(root/'src/cah_mappo'/name)
    save(output/'source_manifest.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(source))})
    save(output/'dependencies.json',{'python':sys.version,'platform':platform.platform(),
         'packages':{p:importlib.metadata.version(p) for p in ('torch','numpy')},'device':'cpu','cuda_runtime':torch.version.cuda,
         'new_packages_installed':False})
    save(output/'pretraining_freeze.json',{'plan_sha256':hashlib.sha256((output/'frozen_plan.json').read_bytes()).hexdigest(),
         'source_manifest_sha256':hashlib.sha256((output/'source_manifest.json').read_bytes()).hexdigest(),
         'benchmark_results_seen':False,'existing_holdout_opened':False})


def workers(root,run,commands):
    logdir=run/'worker_logs';logdir.mkdir(exist_ok=True)
    pending=list(commands);active=[];failed=[];start=time.monotonic()
    while pending or active:
        while pending and len(active)<5:
            label,args=pending.pop(0);stream=(logdir/f'{label}.log').open('x',encoding='utf-8')
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            process=subprocess.Popen([sys.executable,'-B',str(root/'phase3b2b.py'),*args],cwd=str(root.parents[1]),
                                     stdout=stream,stderr=subprocess.STDOUT,env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            active.append((label,process,stream));print(json.dumps({'started':label,'pid':process.pid}),flush=True)
        for item in active[:]:
            label,process,stream=item
            if process.poll() is not None:
                stream.close();active.remove(item)
                print(json.dumps({'finished':label,'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-start}),flush=True)
                if process.returncode:failed.append(label)
        if time.monotonic()-start>8*3600:
            for _,process,stream in active:
                process.terminate();process.wait(timeout=30);stream.close()
            raise RuntimeError('Eight-hour worker budget exceeded; stop expansion and report unfinished work')
        if active:time.sleep(2)
    if failed:raise RuntimeError(f'Workers failed; inspect logs, do not report success: {failed}')


def train_all(root,run):
    workers(root,run,[(f'train_seed{seed}',['train','--run',str(run),'--stream',str(i)]) for i,seed in enumerate(SEEDS)])
    frozen={}
    for seed in SEEDS:
        record=json.loads((run/f'seed_{seed}/selection.json').read_text())
        path=run/f'seed_{seed}'/record['selected_checkpoint']
        frozen[str(seed)]={'selection':record,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    save(run/'all_seeds_selection_freeze.json',{'seeds':frozen,'benchmark_started':False,'holdout_opened':False})


def benchmark_all(root,run):
    if not (run/'all_seeds_selection_freeze.json').is_file():raise RuntimeError('Finish and freeze all seed selections first')
    from .scenarios import benchmark_specs
    save(run/'benchmark_manifest.json',benchmark_specs())
    policies=('local_fixed','local_eta','central_fixed','central_greedy','central_exact','central_handover')
    jobs=[(f'benchmark_seed{seed}',['benchmark-seed','--run',str(run),'--stream',str(i)]) for i,seed in enumerate(SEEDS)]
    jobs += [(f'reference_{mode}',['benchmark-reference','--run',str(run),'--policy',mode]) for mode in policies]
    workers(root,run,jobs)
    workers(root,run,[(f'slots_seed{seed}',['benchmark-slots','--run',str(run),'--stream',str(i)]) for i,seed in enumerate(SEEDS)])
    from .statistics import make_report
    make_report(run)

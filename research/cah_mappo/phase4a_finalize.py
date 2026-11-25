"""Audit supplementary traces, frozen Actor inference and preservation; no writes to old data."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch
from cah_mappo.phase3b2b.io import read_raw,save
from cah_mappo.phase4a.metrics import score
from cah_mappo.phase4a.evaluation import load_network

BASE=ROOT/'phase4a_outputs/study_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(1);extra_count=0;max_error=0.;forward_count=0
    legacy=read(BASE/'legacy_paired_replay/paired_results.json')
    for seed,groups in legacy['seeds'].items():
        for label,group in groups.items():
            for e in group['episodes']:
                p=BASE/'legacy_paired_replay/raw'/f"seed{seed}_{e['scenario_id']}_r{e['repeat']}_{label}.jsonl.gz"
                actual=score(read_raw(p))
                for key,value in actual.items():assert value==e[key],(p.name,key)
                extra_count+=1
    transfer=read(BASE/'transfer_probe/results.json')
    for seed,groups in transfer['seeds'].items():
        for condition,group in groups.items():
            for e in group['episodes']:
                mode=e['policy'].removeprefix('mappo_')
                p=BASE/f'transfer_probe/seed_{seed}_{condition}'/f"{e['scenario_id']}_{mode}.jsonl.gz"
                actual=score(read_raw(p))
                for key,value in actual.items():assert value==e[key],(p.name,key)
                extra_count+=1
    freeze=read(BASE/'design/freeze.json')
    for seed,cp in freeze['checkpoints'].items():
        net=load_network(ROOT/cp['path'],cp['sha256'])
        for mode in ('greedy','stochastic'):
            for gate in ('original','ready','adaptive'):
                folder=BASE/f'comparison/seed_{seed}_{mode}_{gate}';entry=read(folder/'results.json')['episodes'][0]
                rows=read_raw(folder/'raw'/f"{entry['scenario_id']}_r0.jsonl.gz")
                for row in rows:
                    if row['type']!='transition':continue
                    records=row['records'];vectors=[];masks=[]
                    for name in ('A','B','C','D'):
                        r=records.get(name);vectors.append(r['actor_observation'] if r else [0.]*146)
                        masks.append(r['action_mask'] if r else [1]+[0]*9)
                    probs=net.distribution(torch.tensor(vectors,dtype=torch.float32),torch.tensor(masks,dtype=torch.bool)).probs
                    for i,name in enumerate(('A','B','C','D')):
                        if name in records:
                            error=max(abs(a-b) for a,b in zip(probs[i].tolist(),records[name]['probabilities']))
                            max_error=max(max_error,error);assert error<1e-10
                    forward_count+=1
    # Public reports preserve all numeric results when correcting the plotted point style.
    for p in (BASE/'analysis_extended').glob('*.json'):
        assert p.read_bytes()==(BASE/'analysis_extended_v2'/p.name).read_bytes()
    test_log=(BASE/'regression_tests.txt').read_text(encoding='utf-8')
    assert re.search(r'Ran 179 tests in [0-9.]+s\s+OK\s*$',test_log)
    old=read(BASE/'design/preservation.json');repo=ROOT.parents[1]
    for name,h in old['tracked_files'].items():assert sha(repo/name)==h
    for cp in freeze['checkpoints'].values():assert sha(ROOT/cp['path'])==cp['sha256']
    protected=ROOT/'phase3b2a_repair_outputs/pilot_v1/sealed_final_holdout_manifest.json'
    assert {'bytes':protected.stat().st_size,'mtime_ns':protected.stat().st_mtime_ns}==old['sealed_metadata_only']
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=repo).strip()
    # The repository already contained a newer, unrelated user commit than the
    # previous chat's hash. Preserve it; do not reset HEAD to an old chat value.
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()
    commit_time=int(subprocess.check_output(['git','log','-1','--format=%ct'],cwd=repo).decode())
    assert commit_time<=freeze['timestamp_ns']/1e9,'Repository commit postdates the experiment freeze'
    files=[*sorted((ROOT/'src/cah_mappo/phase4a').glob('*.py')),*sorted(ROOT.glob('phase4a*.py')),
           *sorted(ROOT.glob('PHASE4A*.md')),ROOT/'tests/test_phase4a.py']
    save(BASE/'final_integrity.json',{'status':'PASS','supplementary_raw_episodes_recomputed':extra_count,
        'actual_actor_forward_batches_replayed':forward_count,'max_actor_probability_error':max_error,
        'regression_tests':179,'old_tracked_files_unchanged':len(old['tracked_files']),
        'selected_checkpoint_hashes_unchanged':5,'protected_holdout_opened':False,'git_commits_or_pushes_performed':False,
        'observed_repository_head':head,'head_commit_predates_experiment_freeze':True,
        'audit_tool_correction':'Initial HEAD assertion used stale previous-chat hash; existing newer user commit preserved. Research-file/checkpoint/raw hash assertions were unchanged.',
        'code_document_tests_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
    print('final PASS',extra_count,'supplementary raw;',forward_count,'actor batches; max probability error',max_error)


if __name__=='__main__':main()

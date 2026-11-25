"""Final immutable manifest and compact data exports; no Git mutations."""
from pathlib import Path
import csv
import json
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.phase4b.experiment import read,sha
from cah_mappo.phase3b2b.io import save
from cah_mappo.phase3b2b.io import read_raw
from cah_mappo.phase4c.evaluation import coverage_denominators


def finalize(output):
    target=output/'completion_final';audit=read(target/'audit.json');assert audit['status']=='PASS'
    specs=read(ROOT/'phase4b_outputs/study_v1/design/diagnostic_manifest.json');extents=[]
    for s in specs:
        points=[r['position'] for r in s['config']['robots']]+[j['position'] for j in s['config']['jobs']]
        width=max(p[0] for p in points)-min(p[0] for p in points);height=max(p[1] for p in points)-min(p[1] for p in points)
        extents.append({'scenario_id':s['scenario_id'],'x_span_m':width,'y_span_m':height,
                        'exceeds_axis_aligned_practice_dimensions':width>3.9 or height>2.6})
    save(target/'geometry_review.json',{'worlds':extents,'exceeding_count':sum(x['exceeds_axis_aligned_practice_dimensions'] for x in extents),
        'scope':'axis-aligned bounding spans of initial robots and jobs only; no registered map transform or arbitrary-rotation feasibility claim'})
    denominators=[]
    for path in sorted((output/'rules').rglob('*.jsonl.gz')):
        denominators.append({'file':path.relative_to(output).as_posix(),**coverage_denominators(read_raw(path))})
    save(target/'coverage_denominator_details.json',denominators)
    summaries=read(output/'analysis_final/all_method_summaries.json');records=[]
    for name,groups in summaries.items():
        for condition,metrics in groups['by_condition'].items():
            records.append({'method':name,'condition':condition,**{k:v for k,v in metrics.items() if not isinstance(v,(list,dict))}})
    fields=sorted({k for r in records for k in r})
    with (target/'method_condition_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    before=read(output/'preservation_before.json');repo=ROOT.parents[1]
    for n,h in before['source_sha256'].items():assert sha(repo/n)==h,n
    for n,d in before['old_output_metadata'].items():
        s=(ROOT/n).stat();assert d=={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns},n
    for cp in before['checkpoints'].values():assert sha(ROOT/cp['path'])==cp['sha256']
    for manifest in ('read_sources_sha256.json','service_timer_sources_sha256.json'):
        for n,h in read(output/'evidence'/manifest).items():assert sha(Path(n))==h,n
    files=list(ROOT.glob('phase4c*.py'))+list((ROOT/'src/cah_mappo/phase4c').glob('*.py'))+[
        ROOT/'tests/test_phase4c.py',ROOT/'PHASE4C_PROTOCOL_KO.md',ROOT/'PHASE4C_RESULTS_KO.md',ROOT/'PHASE4C_README_KO.md',
        ROOT/'ROBOT_PARAMETER_EVIDENCE_KO.md',ROOT/'RESEARCH_DIRECTION_DECISION_KO.md']
    contents={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    save(target/'final_integrity.json',{'status':'PASS','code_document_tests_sha256':contents,
        'git_head_at_start':before['git_head'],'git_head_at_finish':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip(),
        'new_files_count':len(contents),'prior_source_files_unchanged':len(before['source_sha256']),
        'prior_output_metadata_unchanged':len(before['old_output_metadata']),'checkpoint_files_unchanged':len(before['checkpoints']),
        'decision':'C_empirical_comparison_scope','holdout_content_opened':False,'remote_or_device_execution':False,
        'retrained':False,'git_commit_push':False,'all_experiment_values_measured_by_saved_runs':True})
    print('FINAL INTEGRITY PASS;',len(contents),'new source/document/test files; decision C')

if __name__=='__main__':finalize(ROOT/'phase4c_outputs/study_v1')

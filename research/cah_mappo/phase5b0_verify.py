"""Actual offline replay verification; never launches ROS or controls a device."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.measurement.storage import read,save,sha
from cah_mappo.measurement.importers import extract_bag,extract_shadow
from cah_mappo.measurement.analysis import analyze_dataset
from phase5b0 import inside


def verify(study,output,test_python):
    output=inside(output);output.mkdir(parents=True,exist_ok=False);study=Path(study)
    originals=read(study/'original_hashes_before.json');replays=[]
    for name in ('walk','kick','shadow'):
        dataset=study/(name+'_dataset');m=read(dataset/'manifest.json');source=Path(m['sources'][0]['path'])
        replay=output/(name+'_reextracted')
        (extract_shadow if name=='shadow' else extract_bag)(source,replay)
        for file in ('manifest.json','records.jsonl.gz','seal.json'):assert sha(dataset/file)==sha(replay/file),(name,file)
        first=output/(name+'_analysis');second=output/(name+'_analysis_replay')
        report=analyze_dataset(dataset,first);analyze_dataset(dataset,second)
        for file in ('analysis.json','validation.json','command_segments.json'):assert sha(first/file)==sha(second/file),(name,file)
        validation=read(first/'validation.json')
        assert not any(e['reason']=='raw_message_decode_error' for e in validation['errors'])
        assert report['absolute_position_error']['value'] is None and report['actual_ground_speed_accuracy']['value'] is None
        assert report['one_way_network_delay']['value'] is None
        replays.append({'dataset':name,'records':report['records'],'normalized_bytes_reproduced':True,'analysis_bytes_reproduced':True,
            'validation_status':validation['status'],'observed_errors':len(validation['errors']),
            'interpretation':'Historical header-order errors are retained, not fixed or treated as a decoder failure.'})
        print('Replay PASS',name,report['records'],flush=True)
    for p,h in originals.items():assert sha(p)==h,p
    manual=read(study/'safety_source_review.json')
    for d in manual['manuals']:assert sha(d['source']['path'])==d['source']['sha256']
    result=subprocess.run([str(test_python),'-B','-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
                          capture_output=True,text=True,encoding='utf-8',errors='replace')
    with (output/'unittest.txt').open('x',encoding='utf-8') as f:f.write(result.stdout+result.stderr)
    if result.returncode!=0:raise RuntimeError(result.stderr[-4000:])
    import re
    tests=int(re.search(r'Ran (\d+) tests',result.stderr)[1])
    save(output/'verification.json',{'status':'PASS','replays':replays,'original_source_hashes_unchanged':len(originals),
        'tests':tests,'test_exit_code':result.returncode,'synthetic_tests_are_not_physical_measurements':True,
        'ROS_subscriber_runtime_executed':False,'actuated':False,'retrained':False,'packages_installed':False,
        'protected_holdout_opened':False,'one_way_latency_or_absolute_error_fabricated':False})
    print('FINAL OFFLINE VERIFICATION PASS; tests',tests,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--test-python',required=True,type=Path);a=p.parse_args();verify(a.study,a.output,a.test_python.resolve())

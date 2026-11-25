"""Close an API-level caveat: original native reset seeds differ across branches.

The public decision histories are equal; reset(seed) itself was not equal.
Existing deterministic native rules do not draw their stored RNG in decide().
Replay with one constant actual reset input and verify unchanged outcomes.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.phase4b.policies import Native
from cah_mappo.phase4b.experiment import read,POLICIES
from cah_mappo.phase4c.runs import execute
from cah_mappo.phase4c.information import compare_histories
from cah_mappo.phase3b2b.io import save,write_raw


class CommonResetNative(Native):
    def reset(self,seed):super().reset(7);self.effective_reset_seed=7


def run(output):
    old=ROOT/'phase4b_outputs/study_v1';target=output/'information/native_common_reset';target.mkdir(exist_ok=False)
    specs=read(old/'design/diagnostic_manifest.json');index={(s['family'],s['team'],s['index']):s for s in specs};pairs=[];records=[]
    for name in POLICIES:
        original={r['scenario_id']:r for r in read(old/'comparison'/name/'results.json')['episodes']}
        for team in (2,4):
            for i in range(6):
                histories=[]
                for family in ('A_near','D_visible'):
                    s=index[family,team,i];p=CommonResetNative(name,warning=4,lookahead=4);m,rows=execute(s,p);histories.append(rows)
                    for key,value in original[s['scenario_id']].items():
                        if key in m:assert m[key]==value,(name,s['scenario_id'],key)
                    filename=f"{name}_{s['scenario_id']}.jsonl.gz";write_raw(target/filename,rows)
                    records.append({'file':filename,'metrics':m})
                result=compare_histories(*histories);assert result['identical']
                pairs.append({'policy':name,'team':team,'index':i,**result})
    save(output/'information/native_reset_results.json',{'episodes':records,'pairs':pairs,'actual_effective_reset_seed':7,
        'original_reset_seeds_equal':False,'all_observed_outcomes_unchanged':True,
        'caveat':'original native API exposes a scenario-dependent reset seed, despite equal decide histories; do not make unrestricted information bounds for policies allowed to exploit that seed',
        'prediction_bound_scope':'admissible sensor/message/odometry histories with reset seeds not supplied as predictive features'})
    print('Common-reset native replay',len(records),'episodes;',len(pairs),'equal-history pairs; outcomes unchanged')

if __name__=='__main__':run(ROOT/'phase4c_outputs/study_v1')

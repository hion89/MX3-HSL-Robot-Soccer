"""Offline measurement tools. No publishers, motor/service commands or launch calls."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.measurement.storage import read,save,create_session,append_records,seal_session,archive_source,iter_records


def inside(path):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('All new outputs must remain inside research/cah_mappo')
    return p


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('study');a.add_argument('--output',type=Path,required=True);a.add_argument('--sources',type=Path,default=ROOT.parents[1].parent/'isaac_ws/sim2real_evidence')
    a=sub.add_parser('template');a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('init');a.add_argument('--session',type=Path,required=True);a.add_argument('--manifest',type=Path,required=True)
    for name in ('validate','analyze','seal','append','archive','import-gt','capture-ros'):
        a=sub.add_parser(name);a.add_argument('--session',type=Path,required=True)
        if name in ('validate','analyze'):a.add_argument('--output',type=Path,required=True)
        if name in ('append','archive','import-gt'):a.add_argument('--input',type=Path,required=True)
        if name=='import-gt':
            for field in ('clock','frame','device','calibration'):a.add_argument('--'+field,required=True)
        if name=='capture-ros':
            a.add_argument('--duration',type=float,default=60);a.add_argument('--allow-passive-subscriptions',action='store_true')
    a=p.parse_args()
    if a.command=='study':
        from cah_mappo.measurement.workflow import study
        study(ROOT,inside(a.output),a.sources)
    elif a.command=='template':
        from cah_mappo.measurement.collection import session_template
        save(inside(a.output),session_template())
    elif a.command=='init':create_session(inside(a.session),read(a.manifest))
    elif a.command=='append':append_records(inside(a.session),iter_records(a.input))
    elif a.command=='seal':seal_session(inside(a.session))
    elif a.command=='archive':print(archive_source(inside(a.session),a.input))
    elif a.command=='import-gt':
        from cah_mappo.measurement.collection import import_gt_csv
        print(import_gt_csv(inside(a.session),a.input,a.clock,a.frame,a.device,a.calibration))
    elif a.command=='capture-ros':
        from cah_mappo.measurement.collection import passive_ros_capture
        passive_ros_capture(inside(a.session),a.duration,a.allow_passive_subscriptions)
    elif a.command=='validate':
        from cah_mappo.measurement.validation import validate_dataset
        r=validate_dataset(a.session);save(inside(a.output),r)
        if r['status']!='PASS':raise SystemExit(2)
    elif a.command=='analyze':
        from cah_mappo.measurement.analysis import analyze_dataset
        analyze_dataset(a.session,inside(a.output))


if __name__=='__main__':main()

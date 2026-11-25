from pathlib import Path
import os
import subprocess
from ..phase3b2b.io import save
from ..phase4b.experiment import read,sha


def protect(root,output):
    output.mkdir(parents=True,exist_ok=False);repo=root.parents[1]
    names=[p for p in subprocess.check_output(['git','ls-files','-z','research'],cwd=repo).decode().split('\0') if p]
    hashes={n:sha(repo/n) for n in names}
    for phase in ('phase4a','phase4b'):
        p=root/(phase+'_outputs')/'study_v1'
        manifest=read(p/('completion/final_integrity.json' if phase=='phase4b' else 'final_integrity.json'))
        for n,h in manifest['code_document_tests_sha256'].items():
            assert sha(root/n)==h,n;hashes[(root/n).relative_to(repo).as_posix()]=h
    metadata={}
    for directory in root.iterdir():
        if not directory.is_dir() or directory==output.parent or not (directory.name=='outputs' or directory.name.endswith('_outputs')):continue
        for base,dirs,files in os.walk(directory):
            dirs[:]=[d for d in dirs if 'holdout' not in d.lower()]
            for name in files:
                p=Path(base)/name;stat=p.stat();metadata[p.relative_to(root).as_posix()]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    cp=read(root/'phase4b_outputs/study_v1/design/freeze.json')['checkpoints']
    for d in cp.values():assert sha(root/d['path'])==d['sha256']
    save(output/'preservation_before.json',{'source_sha256':hashes,'old_output_metadata':metadata,'checkpoints':cp,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip(),
        'initial_git_status':subprocess.check_output(['git','status','--short'],cwd=repo).decode(),
        'protected_holdout_content_read':False})
    print('Protected',len(hashes),'old source files and',len(metadata),'old output metadata records')


def evidence(root,output):
    from .evidence import collect
    collect(root,output)


def rules(root,output):
    from .runs import run_rules
    run_rules(root,output)


def plans(root,output):
    from .runs import run_plans
    run_plans(root,output)


def information(root,output):
    from .information import run_information
    run_information(root,output)


def sensitivity(root,output):
    from .runs import run_sensitivity
    run_sensitivity(root,output)


def classify(root,output):
    from .classification import classify_all
    classify_all(root,output)


def audit(root,output):
    from .audit import audit_all
    audit_all(root,output)

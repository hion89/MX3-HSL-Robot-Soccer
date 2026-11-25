"""Exclusive output creation, immutable sealing, local raw-source preservation."""
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
from .schema import validate_manifest,validate_record


def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(canonical(value)+'\n')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def create_session(path,manifest):
    path=Path(path);validate_manifest(manifest);path.mkdir(parents=True,exist_ok=False)
    save(path/'manifest.json',manifest)
    (path/'records.jsonl').open('x',encoding='utf-8').close()
    return path


@contextmanager
def journal_lock(path):
    """Persistent lock file; no raw files are deleted to unlock."""
    with (Path(path)/'journal.lock').open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if __import__('os').name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            try:yield
            finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def append_records(path,records):
    path=Path(path);records=list(records);m=read(path/'manifest.json')
    for r in records:validate_record(r,m)
    ids=[r['record_id'] for r in records]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate incoming record IDs')
    payload=''.join(canonical(r)+'\n' for r in records)
    with journal_lock(path):
        if (path/'seal.json').exists():raise ValueError('Sealed journal is immutable')
        old={r['record_id'] for r in iter_records(path/'records.jsonl')}
        if old.intersection(ids):raise ValueError('Record ID already exists')
        with (path/'records.jsonl').open('a',encoding='utf-8',newline='\n') as f:
            f.write(payload);f.flush();__import__('os').fsync(f.fileno())


def iter_records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            try:yield json.loads(line)
            except (ValueError,TypeError) as e:raise ValueError(f'Invalid/truncated JSONL line {i}') from e


class JournalWriter:
    """Hold one lock/read IDs once for bounded passive collection."""
    def __init__(self,path):self.path=Path(path)
    def __enter__(self):
        self.lock=journal_lock(self.path);self.lock.__enter__()
        try:
            if (self.path/'seal.json').exists():raise ValueError('Sealed journal is immutable')
            self.manifest=read(self.path/'manifest.json');self.ids={r['record_id'] for r in iter_records(self.path/'records.jsonl')}
            self.file=(self.path/'records.jsonl').open('a',encoding='utf-8',newline='\n');return self
        except BaseException:
            self.lock.__exit__(*__import__('sys').exc_info());raise
    def append(self,records):
        records=list(records);ids=[r['record_id'] for r in records]
        for r in records:validate_record(r,self.manifest)
        if len(ids)!=len(set(ids)) or self.ids.intersection(ids):raise ValueError('Duplicate record ID')
        self.file.write(''.join(canonical(r)+'\n' for r in records));self.file.flush()
        __import__('os').fsync(self.file.fileno());self.ids.update(ids)
    def __exit__(self,*exc):
        try:self.file.close()
        finally:self.lock.__exit__(*exc)


def seal_session(path):
    path=Path(path)
    with journal_lock(path):
        if (path/'seal.json').exists():raise FileExistsError('Already sealed')
        count=sum(1 for _ in iter_records(path/'records.jsonl'))
        save(path/'seal.json',{'manifest_sha256':sha(path/'manifest.json'),'records_sha256':sha(path/'records.jsonl'),
            'record_count':count,'data_origin':read(path/'manifest.json')['data_origin'],
            'archived_sources':{p.relative_to(path).as_posix():sha(p) for p in (path/'raw_sources').glob('*') if p.is_file()}})


def write_dataset(path,manifest,records):
    path=Path(path);validate_manifest(manifest);path.mkdir(parents=True,exist_ok=False);save(path/'manifest.json',manifest)
    count=0
    with (path/'records.jsonl.gz').open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as zipped:
            with io.TextIOWrapper(zipped,encoding='utf-8',newline='\n') as f:
                for record in records:
                    validate_record(record,manifest);f.write(canonical(record)+'\n');count+=1
    save(path/'seal.json',{'manifest_sha256':sha(path/'manifest.json'),'records_sha256':sha(path/'records.jsonl.gz'),
                           'record_count':count,'data_origin':manifest['data_origin']})
    return count


def archive_source(session,source):
    session,source=Path(session),Path(source).resolve()
    if (session/'seal.json').exists():raise ValueError('Cannot attach sources to sealed session')
    h=sha(source);folder=session/'raw_sources';folder.mkdir(exist_ok=True)
    target=folder/(h+'_'+source.name)
    with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    if sha(source)!=h or sha(target)!=h:raise ValueError('Source changed during archival')
    return {'path':str(source),'sha256':h,'archive':str(target),'bytes':source.stat().st_size}

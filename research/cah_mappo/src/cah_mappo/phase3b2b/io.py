"""Exclusive, streaming gzip JSONL storage. Protected holdout files are not inputs."""
import gzip
import json
from pathlib import Path


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def write_raw(path,rows):
    with gzip.open(path,'xt',encoding='utf-8',compresslevel=1) as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')


def read_raw(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]

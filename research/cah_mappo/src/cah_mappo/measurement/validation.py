"""Offline validation reports missing data without manufacturing measurements."""
from collections import Counter
from .schema import validate_manifest,validate_record,TIME_FIELDS,QUANTITIES
from .storage import read,iter_records,sha


def validate_dataset(path):
    from pathlib import Path
    path=Path(path);m=read(path/'manifest.json');errors=[];warnings=[];missing=Counter();ids=set();previous={};n=0
    try:validate_manifest(m)
    except ValueError as e:errors.append({'record':None,'reason':str(e)})
    file=path/('records.jsonl.gz' if (path/'records.jsonl.gz').is_file() else 'records.jsonl')
    try:
        for i,r in enumerate(iter_records(file),1):
            n+=1
            try:validate_record(r,m)
            except (ValueError,KeyError,TypeError) as e:errors.append({'record':r.get('record_id',i),'reason':str(e)})
            if r.get('record_id') in ids:errors.append({'record':r.get('record_id'),'reason':'duplicate_record_id'})
            ids.add(r.get('record_id'))
            for k in TIME_FIELDS+QUANTITIES+('success',):
                if r.get(k) is None:missing[k]+=1
            for field in ('timestamp','header_timestamp'):
                t=r.get(field)
                if not isinstance(t,dict) or not isinstance(t.get('value'),int):continue
                key=(r.get('source',{}).get('topic'),t.get('clock_id'),field);old=previous.get(key)
                if old and t['value']<old[0]:errors.append({'record':r['record_id'],'previous_record':old[1],'reason':'timestamp_reversal','field':field})
                if old and t['value']==old[0]:warnings.append({'record':r['record_id'],'reason':'repeated_timestamp','field':field})
                previous[key]=(t['value'],r['record_id'])
            if r.get('event_type')=='decode_error':errors.append({'record':r['record_id'],'reason':'raw_message_decode_error'})
    except (ValueError,OSError) as e:errors.append({'record':n+1,'reason':str(e)})
    for link in m.get('synchronization',[]):
        if link.get('offset_ns') is None or link.get('uncertainty_ns') is None:warnings.append({'record':None,'reason':'unmeasured_sync_offset_or_error'})
    clocks_used=set(previous_key[1] for previous_key in previous)
    if len(clocks_used)>1 and not m.get('synchronization'):warnings.append({'record':None,'reason':'multiple_clocks_no_measured_link'})
    seal=read(path/'seal.json') if (path/'seal.json').is_file() else None
    if seal:
        for key,p in [('manifest_sha256',path/'manifest.json'),('records_sha256',file)]:
            if sha(p)!=seal[key]:errors.append({'record':None,'reason':'sealed_content_changed','file':p.name})
        if n!=seal['record_count']:errors.append({'record':None,'reason':'sealed_count_mismatch'})
        for name,h in seal.get('archived_sources',{}).items():
            p=(path/name).resolve()
            if not p.is_relative_to(path.resolve()) or not p.is_file() or sha(p)!=h:errors.append({'record':None,'reason':'archived_source_changed','file':name})
    # Historical absence is reported as coverage, not rejected as fabricated GT.
    blockers=[]
    if m.get('mode')!='historical_forensic':
        safety=m.get('safety') or {}
        for key in ('temperature_stop_c','voltage_stop_v','current_stop_ma','balance_stop_definition'):
            item=safety.get(key)
            if not isinstance(item,dict) or item.get('value') is None or not item.get('source'):blockers.append('unverified_safety_'+key)
        if not safety.get('responsible_operator') or not safety.get('responsible_safety_person'):blockers.append('safety_roles_unconfirmed')
        if not safety.get('fall_prevention_checked') or not safety.get('emergency_stop_checked'):blockers.append('physical_safety_checks_unconfirmed')
        if n==0:blockers.append('no_recorded_samples')
        if missing['ground_truth_position']==n:blockers.append('no_independent_GT_samples')
        if len(clocks_used)>1 and not m.get('synchronization'):blockers.append('clock_alignment_unverified')
    return {'schema_version':m.get('schema_version'),'data_origin':m.get('data_origin'),'records':n,
        'status':'PASS' if not errors else 'FAIL','errors':errors,'warnings':warnings,
        'missing_counts':dict(missing),'motion_readiness':'NOT_AUTHORIZED_BY_THIS_TOOL',
        'preflight_blockers':blockers,'note':'Validation is never permission to actuate; historical null GT is not repaired.'}

"""Additional raw-GT diagnosis; does not replace any legacy/Phase 4B metric."""
from ..phase4b.model import match


def coverage_denominators(rows):
    from ..phase2.metrics import evaluate
    old=evaluate(rows);bases=sum(j['kind']=='base' for j in rows[0]['config']['jobs'])
    seconds=rows[-1]['time']*bases
    return {'legacy_weighted_gap_region_seconds':old['weighted_coverage_gap_s'],
        'legacy_weighted_denominator_region_seconds':old['coverage_weighted_time_s'],
        'legacy_weighted_cgr':old['coverage_gap_ratio'],
        'unweighted_gap_region_seconds':old['coverage_gap_s'],
        'unweighted_denominator_region_seconds':seconds,
        'unweighted_cgr':old['coverage_gap_s']/seconds}


def failure_service_recovery(rows):
    intervals=[r for r in rows if r['type']=='interval'];end=rows[-1]['time'];records=[]
    for event in rows:
        if event.get('event')!='failure':continue
        time=event['time'];name=event['target'];previous=[r for r in intervals if r['start']<time]
        state=previous[-1]['truth'] if previous else intervals[0]['truth']
        robot=state['robots'][name];base=robot['assignment']
        if base not in state['jobs'] or state['jobs'][base]['kind']!='base':base=None
        ready=None;at_fault=None
        if base:
            query={base:(base,1.)}
            for row in intervals:
                if row['start']<time:continue
                covered=base in match(row['truth'],query,{base:0.},at_point=True)
                if at_fault is None:at_fault=covered
                if covered:ready=row['start'];break
        records.append({'failed_robot':name,'failure_at':time,'directly_affected_base':base,
            'service_present_at_failure':at_fault,'first_at_point_service_after_failure':ready,
            'service_recovery_delay_s':None if ready is None else ready-time,
            'right_censored':bool(base and ready is None),'observed_until':end,
            'definition':'first operational capacity-free robot at the affected base after failure; zero means continuous initial service, not a repaired robot'})
    return records

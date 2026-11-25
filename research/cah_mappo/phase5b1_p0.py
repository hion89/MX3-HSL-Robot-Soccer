"""Offline P0 preparation/replay ONLY. No live collector or device entry point."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from cah_mappo.measurement.storage import read, save, sha, iter_records
from cah_mappo.measurement_p0.audit import audit_streams, timestamp_audit, readiness, raw_image_budget, P1_EVIDENCE
from cah_mappo.measurement_p0.static_review import review
from phase5b0_verify import verify


def bag_catalog(path):
    """Open SQLite query-only and retain offered QoS exactly as stored, if present."""
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as con:
        con.execute('PRAGMA query_only=ON')
        con.row_factory = sqlite3.Row
        return {'path': str(path), 'sha256': sha(path),
                'topics_raw_sqlite': [dict(r) for r in con.execute('SELECT * FROM topics ORDER BY id')],
                'scope': 'Stored offered QoS is historical metadata, not live DDS compatibility proof.'}


def prepare(study, output, test_python):
    study, output = Path(study).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError('output must be a new subdirectory inside research/cah_mappo')
    output.mkdir(parents=True, exist_ok=False)
    profiles = read(ROOT / 'p0_profiles.json')['profiles']
    save(output / 'profiles_frozen.json', read(ROOT / 'p0_profiles.json'))
    save(output / 'source_review.json', review(ROOT.parents[1] / 'MX3_Match_Sim' / 'src'))
    # Re-extract original bags/CSV, compare normalized bytes and repeat original analysis.
    # This also runs the full existing + new unit suite. It does not import/run ROS.
    verify(study, output / 'replay', Path(test_python).resolve())
    catalogs, reports = [], []
    for name in ('walk', 'kick', 'shadow'):
        dataset = study / (name + '_dataset')
        manifest = read(dataset / 'manifest.json')
        data = dataset / 'records.jsonl.gz'
        report = audit_streams(iter_records(data), manifest, profiles['historical_' + name])
        again = audit_streams(iter_records(data), manifest, profiles['historical_' + name])
        assert report == again, name
        save(output / (name + '_stream_audit.json'), report)
        assert not report['errors'], report['errors'][:3]
        reports.append({'name': name, 'records': report['records'], 'errors': len(report['errors']),
                        'GT_missing_records': report['GT_missing_records'], 'audit_reproduced': True})
        if name != 'shadow':
            catalogs.append(bag_catalog(manifest['sources'][0]['path']))
    save(output / 'bag_catalogs.json', catalogs)
    kick = study / 'kick_dataset' / 'records.jsonl.gz'
    stamps = timestamp_audit(iter_records(kick))
    assert stamps == timestamp_audit(iter_records(kick))
    assert len(stamps['reversals']) == stamps['cross_signature_reversals'] == 168
    assert all(g['header']['negative_intervals'] == 0 for g in stamps['signatures'])
    save(output / 'kick_timestamp_audit.json', stamps)
    # Cross-profile missing-source check is expected to fail, NOT a live P0 test.
    missing = audit_streams(iter_records(kick), read(study / 'kick_dataset' / 'manifest.json'), profiles['imu_candidate'])
    assert any(e['reason'] == 'required_stream_missing_or_short' and e['topic'] == '/imu/data' for e in missing['errors'])
    save(output / 'expected_missing_stream_demo.json', missing)
    evidence = {k: {'confirmed': False, 'source': None, 'signed_by': None, 'checked_at': None, 'origin': None}
                for k in P1_EVIDENCE}
    save(output / 'onsite_evidence_UNFILLED.json', evidence)
    save(output / 'readiness.json', readiness(evidence))
    disk = shutil.disk_usage(ROOT)
    save(output / 'storage_planning.json', {'local_PC_only': {'total': disk.total, 'used': disk.used, 'free': disk.free},
         'robot_free_bytes': None, 'robot_write_throughput_bytes_s': None,
         'conditional_image_payload_30s': raw_image_budget(),
         'conditional_image_payload_90s': raw_image_budget(seconds=90),
         'not_a_capacity_approval': True})
    save(output / 'completion.json', {'status': 'OFFLINE_PREPARATION_COMPLETE', 'reports': reports,
         'original_header_reversals_preserved': 168, 'clock_synchronization_verified': False,
         'ROS_runtime_executed': False, 'device_accessed': False, 'onsite_P0_complete': False,
         'P1_allowed': False, 'packages_installed': False, 'git_commit_or_push': False,
         'protected_holdout_opened': False,
         'test_summary': read(output / 'replay' / 'verification.json')})
    print('P0 OFFLINE PREPARATION COMPLETE; onsite P0 not run; P1 BLOCKED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--test-python', type=Path, required=True)
    args = p.parse_args()
    prepare(args.study, args.output, args.test_python)

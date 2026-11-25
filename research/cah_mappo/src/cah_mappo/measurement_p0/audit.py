"""Supplementary audits; original measurement-v1 and Phase5B0 stay unchanged."""
from collections import Counter, defaultdict
import math
from ..measurement.analysis import intervals
from ..measurement.schema import validate_record


def timestamp_audit(records, topic='/joint_goals'):
    """Preserve file order. Name signatures are NOT publisher identities."""
    groups = defaultdict(list)
    aggregate, received, reversals = [], [], []
    previous = None
    for r in records:
        if r.get('source', {}).get('topic') != topic:
            continue
        t, h = r.get('timestamp'), r.get('header_timestamp')
        if t:
            received.append((t['value'], r['record_id']))
        if not h:
            continue
        signature = tuple(r.get('decoded', {}).get('name', []))
        row = (h['value'], r['record_id'])
        aggregate.append(row)
        groups[signature].append(row)
        if previous and previous[0] == h['clock_id'] and row[0] < previous[1][0]:
            reversals.append({'previous_record': previous[1][1], 'record': row[1],
                              'previous_names': list(previous[2]), 'names': list(signature),
                              'cross_signature': previous[2] != signature,
                              'delta_ns': row[0] - previous[1][0],
                              'source': r['source']})
        previous = (h['clock_id'], row, signature)
    return {'topic': topic, 'aggregate_header': intervals(aggregate),
            'recorder': intervals(received), 'reversals': reversals,
            'cross_signature_reversals': sum(r['cross_signature'] for r in reversals),
            'signatures': [{'names': list(k), 'header': intervals(v)} for k, v in sorted(groups.items())],
            'clock_synchronization': 'NOT_VERIFIED', 'one_way_latency_s': None,
            'caveat': 'Signature is not DDS publisher GID; monotonicity does not prove synchronized clocks. '
                      'Header may be copied feedback time rather than publish/acquisition time.'}


def _stamp_ns(stamp):
    if not isinstance(stamp, dict):
        raise ValueError('missing_stamp')
    sec, ns = stamp.get('sec'), stamp.get('nanosec')
    if type(sec) is not int or type(ns) is not int or not 0 <= ns < 1_000_000_000:
        raise ValueError('invalid_stamp')
    return sec * 1_000_000_000 + ns


def audit_streams(records, manifest, profile):
    """Check expected streams, raw frames and nested TF, not just populated quantities.

    Thresholds here are recording diagnostics, never equipment safety limits.
    Static TF stamps may legitimately be zero. Event streams need not emit.
    """
    counts, types, times, headers = Counter(), defaultdict(set), defaultdict(list), defaultdict(list)
    errors, warnings, tf_previous, n = [], [], {}, 0
    gt_missing = 0
    for r in records:
        n += 1
        ident = r.get('record_id')
        topic = r.get('source', {}).get('topic', '')
        counts[topic] += 1
        types[topic].add(r.get('source', {}).get('message_type'))
        try:
            validate_record(r, manifest)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({'record': ident, 'reason': 'schema', 'detail': str(exc)})
        gt_missing += r.get('ground_truth_position') is None
        t = r.get('timestamp')
        if t and type(t.get('value')) is int:
            times[(topic, t.get('clock_id'))].append((t['value'], ident))
        d = r.get('decoded') or {}
        header = d.get('header')
        if isinstance(header, dict):
            frame = header.get('frame_id')
            need_frame = topic in profile.get('frame_required_topics', [])
            if need_frame and (not frame or frame not in manifest.get('frames', {})):
                errors.append({'record': ident, 'reason': 'missing_or_undeclared_frame', 'topic': topic})
            try:
                stamp = _stamp_ns(header.get('stamp'))
                if stamp <= 0:
                    warnings.append({'record': ident, 'reason': 'zero_or_negative_dynamic_stamp'})
                # Mixed JointState commands are separately audited by signature.
                signature = tuple(d.get('name', [])) if topic == '/joint_goals' else ()
                headers[(topic, signature)].append((stamp, ident))
            except ValueError as exc:
                errors.append({'record': ident, 'reason': str(exc), 'topic': topic})
        if topic in ('/tf', '/tf_static'):
            for tr in d.get('transforms', []):
                h = tr.get('header') or {}
                parent, child = h.get('frame_id'), tr.get('child_frame_id')
                if not parent or not child or parent == child:
                    errors.append({'record': ident, 'reason': 'invalid_tf_frames'})
                elif any(f not in manifest.get('frames', {}) for f in (parent, child)):
                    errors.append({'record': ident, 'reason': 'undeclared_tf_frame', 'frames': [parent, child]})
                try:
                    stamp = _stamp_ns(h.get('stamp'))
                    key = (topic, parent, child)
                    if topic == '/tf' and stamp <= 0:
                        warnings.append({'record': ident, 'reason': 'zero_or_negative_dynamic_tf_stamp'})
                    if key in tf_previous and stamp < tf_previous[key]:
                        errors.append({'record': ident, 'reason': 'nested_tf_timestamp_reversal', 'frames': [parent, child]})
                    tf_previous[key] = stamp
                except ValueError as exc:
                    errors.append({'record': ident, 'reason': str(exc), 'topic': topic})
    for topic, cfg in profile.get('streams', {}).items():
        if counts[topic] < cfg.get('min_messages', 0):
            errors.append({'record': None, 'reason': 'required_stream_missing_or_short', 'topic': topic,
                           'received': counts[topic], 'required': cfg['min_messages']})
        if counts[topic] and types[topic] != {cfg['type']}:
            errors.append({'record': None, 'reason': 'message_type_mismatch', 'topic': topic})
    stream_reports = []
    for (topic, clock), values in sorted(times.items()):
        report = intervals(values)
        stream_reports.append({'topic': topic, 'clock_id': clock, **report})
        if report['negative_intervals']:
            errors.append({'record': None, 'reason': 'receive_timestamp_reversal', 'topic': topic})
    for (topic, signature), values in headers.items():
        if intervals(values)['negative_intervals']:
            errors.append({'record': None, 'reason': 'header_timestamp_reversal', 'topic': topic, 'names': list(signature)})
    return {'profile': profile.get('id'), 'records': n, 'counts': dict(sorted(counts.items())),
            'errors': errors, 'warnings': warnings, 'receive_streams': stream_reports,
            'GT_missing_records': gt_missing, 'GT_accuracy_validated': False,
            'status': 'ERROR' if errors else 'OFFLINE_STREAM_CHECK_PASS',
            'onsite_validation_complete': False,
            'limitations': ['No DDS runtime or recorder throughput test.',
                           'Long intervals are candidates, not proven lost packets.',
                           'No independent GT means absolute position/speed error is unavailable.']}


P1_EVIDENCE = ('safety_person', 'equipment_operator', 'equipment_stop_limits',
               'emergency_stop', 'mechanical_protection', 'installed_runtime_and_policy',
               'independent_GT_calibration', 'clock_alignment', 'onsite_P0_recording',
               'storage_and_seal', 'onsite_experiment_approval')


def readiness(evidence):
    """Evidence checklist only; even complete paperwork never authorizes motion."""
    missing = []
    for key in P1_EVIDENCE:
        item = evidence.get(key) or {}
        if not (item.get('confirmed') is True and item.get('source') and item.get('signed_by')
                and item.get('checked_at') and item.get('origin') == 'onsite'):
            missing.append(key)
    for key, fields in {'clock_alignment': ('offset_ns', 'residual_error_ns', 'valid_from_ns', 'valid_to_ns'),
                        'independent_GT_calibration': ('position_uncertainty_m', 'yaw_uncertainty_rad'),
                        'equipment_stop_limits': ('temperature_stop_c', 'current_stop_ma', 'voltage_stop_v')}.items():
        item = evidence.get(key) or {}
        if not all(type(item.get(f)) in (int, float) and math.isfinite(item[f]) for f in fields):
            missing.append(key + ':unmeasured_values')
        if key == 'clock_alignment' and all(type(item.get(f)) in (int, float) for f in fields):
            if item['residual_error_ns'] < 0 or item['valid_from_ns'] >= item['valid_to_ns']:
                missing.append(key + ':invalid_range')
        if key == 'independent_GT_calibration' and any(isinstance(item.get(f), (int, float)) and item[f] < 0 for f in fields):
            missing.append(key + ':negative_uncertainty')
    return {'P0_offline_preparation': 'COMPLETE', 'onsite_P0': 'NOT_VERIFIED' if 'onsite_P0_recording' in missing else 'EVIDENCE_DECLARED_ONLY',
            'P1_evidence_status': 'BLOCKED' if missing else 'REQUIRES_HUMAN_REVIEW',
            'missing': missing, 'motion_authorized': False,
            'reason': 'No hardware was inspected or run by this tool; separate onsite authorization is required.'}


def raw_image_budget(width=848, height=480, fps=30, seconds=30):
    """Conditional uncompressed BGR8 + aligned 16UC1; not a measured rate."""
    if any(type(x) is not int or x <= 0 for x in (width, height, fps, seconds)):
        raise ValueError('positive_integer_dimensions_required')
    rate = width * height * (3 + 2) * fps
    return {'classification': 'C/D conditional planning assumption', 'bytes_per_second': rate,
            'duration_s': seconds, 'payload_bytes': rate * seconds,
            'assumptions': 'One BGR8 color and one 16UC1 depth stream; excludes overhead, GT video, duplicate copies and compression.'}

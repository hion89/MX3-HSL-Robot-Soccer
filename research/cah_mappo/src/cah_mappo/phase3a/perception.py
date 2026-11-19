"""Privileged sensor/transport simulation and a separate cache-only Actor builder."""
import copy
import math
import random
import hashlib
import json


class Perception:
    def __init__(self, settings, config):
        self.s, self.c = settings, config
        self.actual = tuple(r.id for r in config.robots)
        self.memory = {a: {'peers': {}, 'jobs': {}} for a in self.actual}
        self.slots = {a: [] for a in self.actual}
        self.self_records = {}
        self.queue = []

    def _cache(self, agent, category, entity, record, source):
        old = self.memory[agent][category].get(entity)
        # Equal-time direct measurements win over relayed self reports.
        if old is not None and (old['observed_at'] > record['observed_at']
                               or old['observed_at'] == record['observed_at'] and old['source'] == 'sensor'):
            return
        self.memory[agent][category][entity] = {**copy.deepcopy(record), 'source': source}
        if category == 'jobs' and entity not in self.slots[agent]:
            self.slots[agent].append(entity)  # no global/future job registry in actor slots

    def update(self, truth, t, event):
        for name, robot in truth['robots'].items():
            # Perfect own odometry/controller feedback is a declared local sensor.
            self.self_records[name] = copy.deepcopy(robot)
            if not robot['operational']:
                continue
            peers, jobs = {}, {}
            for other, target in sorted(truth['robots'].items()):
                if other == name or math.dist(robot['position'], target['position']) > self.s.sensor_radius:
                    continue
                rng = random.Random(f'sensor:{self.c.seed}:{t}:{name}:robot:{other}')
                record = {'position': [x + rng.uniform(-self.s.sensor_noise, self.s.sensor_noise) for x in target['position']],
                          'heading': target['heading'], 'speed': target['speed'],
                          'operational': target['operational'], 'observed_at': t}
                peers[other] = record
                self._cache(name, 'peers', other, record, 'sensor')
            for jid, job in sorted(truth['jobs'].items()):
                if job['status'] == 'not_spawned' or math.dist(robot['position'], job['position']) > self.s.sensor_radius:
                    continue
                rng = random.Random(f'sensor:{self.c.seed}:{t}:{name}:job:{jid}')
                record = {'position': [x + rng.uniform(-self.s.sensor_noise, self.s.sensor_noise) for x in job['position']],
                          'kind': job['kind'], 'status': job['status'], 'priority': job['priority'],
                          'deadline': job['deadline'],
                          'remaining': max(0.0, job['service'] - job['work']) if job['kind'] == 'task' else 0.0,
                          'observed_at': t}
                jobs[jid] = record
                self._cache(name, 'jobs', jid, record, 'sensor')
            if self.s.radio_enabled:
                report = {'position': list(robot['position']), 'heading': robot['heading'],
                          'speed': robot['speed'], 'operational': robot['operational'], 'observed_at': t}
                # Send only own state and locally inspected jobs, never hidden GT or forwarded caches.
                for recipient in self.actual:
                    if recipient != name:
                        self.queue.append({'sender': name, 'recipient': recipient, 'sent': t,
                                           'due': round(t + self.s.radio_delay, 10), 'peer': report, 'jobs': copy.deepcopy(jobs)})
                        event(t, 'radio_sent', sender=name, recipient=recipient, due=round(t + self.s.radio_delay, 10))
        due = [m for m in self.queue if m['due'] <= t + 1e-9]
        self.queue = [m for m in self.queue if m['due'] > t + 1e-9]
        for msg in due:
            recipient = msg['recipient']
            if not truth['robots'][recipient]['operational']:
                event(t, 'radio_dropped', sender=msg['sender'], recipient=recipient)
                continue
            self._cache(recipient, 'peers', msg['sender'], msg['peer'], 'message')
            for jid, record in sorted(msg['jobs'].items()):
                self._cache(recipient, 'jobs', jid, record, 'message')
            event(t, 'radio_received', sender=msg['sender'], recipient=recipient, sent=msg['sent'], age=t - msg['sent'])

    def actor(self, name, t):
        # This builder never receives or reads engine.truth(), peer GT, or event plans.
        return build_actor(self.self_records[name], self.memory[name], self.slots[name],
                           self.actual, self.s, t, self.c.horizon, name)


def build_actor(own, memory, slots, actual_roster, settings, t, horizon, name):
    assignment = own['assignment']
    own_assignment = [0.0] * (settings.max_jobs + 1)
    own_index = slots.index(assignment) + 1 if assignment in slots else 0
    own_assignment[own_index] = 1.0
    x, y = own['position']
    vector = [x / 10, y / 10, math.sin(own['heading']), math.cos(own['heading']),
              own['speed'] / 2, own['turn_speed'] / math.pi, float(own['operational']),
              float(own['base_role'] == 'guard'), t / horizon, *own_assignment]
    peer_mask = []
    for peer in settings.roster:
        record = memory['peers'].get(peer)
        if record is None or peer == name:
            vector.extend([0.0] * 8); peer_mask.append(0)
        else:
            age = t - record['observed_at']
            vector.extend([1.0, (record['position'][0] - x) / 10, (record['position'][1] - y) / 10,
                           math.sin(record['heading']), math.cos(record['heading']), record['speed'] / 2,
                           float(record['operational']), age / horizon])
            peer_mask.append(int(age <= settings.cache_ttl))
    job_mask, fresh_mask, select_mask, job_ids = [], [], [], []
    for index in range(settings.max_jobs):
        if index >= len(slots):
            vector.extend([0.0] * 9)
            job_mask.append(0); fresh_mask.append(0); select_mask.append(0); job_ids.append(None)
            continue
        jid = slots[index]; record = memory['jobs'][jid]; age = t - record['observed_at']
        fresh = age <= settings.cache_ttl
        vector.extend([1.0, float(record['status'] == 'active'), float(record['kind'] == 'base'),
                       (record['position'][0] - x) / 10, (record['position'][1] - y) / 10,
                       record['priority'] / 10, (record['deadline'] - t) / horizon,
                       record['remaining'] / horizon, age / horizon])
        job_mask.append(1); fresh_mask.append(int(fresh)); job_ids.append(jid)
        select_mask.append(int(fresh and record['status'] == 'active' and own['operational']))
    roster_mask = [int(peer in actual_roster) for peer in settings.roster]  # public team roster, not visibility
    vector.extend(roster_mask + peer_mask + job_mask + fresh_mask)
    mask = [1, int(own['operational'] and assignment is not None), *select_mask]
    if not all(math.isfinite(v) for v in vector):
        raise ValueError('Nonfinite actor features')
    result = {'observation': vector, 'action_mask': mask, 'robot_mask': roster_mask,
            'peer_fresh_mask': peer_mask, 'job_mask': job_mask, 'job_fresh_mask': fresh_mask,
            'job_ids': job_ids, 'self_assignment': assignment, 'active_mask': int(own['operational'])}
    result['observation_token'] = hashlib.sha256(json.dumps({'agent': name, 'time': t, 'view': result},
                                                           sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result


def build_critic(truth, t, horizon, settings, global_slots):
    """Current full GT only. No future jobs, events, RNG state, or private queues."""
    for jid, job in sorted(truth['jobs'].items()):
        if job['status'] != 'not_spawned' and jid not in global_slots:
            global_slots.append(jid)
    robots = truth['robots']; vector = [t / horizon]; robot_mask = []
    for name in settings.roster:
        r = robots.get(name)
        if r is None:
            vector.extend([0.0] * (8 + settings.max_jobs + 1)); robot_mask.append(0)
            continue
        code = [0.0] * (settings.max_jobs + 1)
        code[global_slots.index(r['assignment']) + 1 if r['assignment'] in global_slots else 0] = 1.0
        vector.extend([1.0, float(r['operational']), r['position'][0] / 10, r['position'][1] / 10,
                       math.sin(r['heading']), math.cos(r['heading']), r['speed'] / 2,
                       float(r['base_role'] == 'guard'), *code]); robot_mask.append(1)
    job_mask = []
    for index in range(settings.max_jobs):
        if index >= len(global_slots):
            vector.extend([0.0] * 9); job_mask.append(0); continue
        j = truth['jobs'][global_slots[index]]
        owner = settings.roster.index(j['owner']) + 1 if j['owner'] in settings.roster else 0
        vector.extend([1.0, float(j['status'] == 'active'), float(j['kind'] == 'base'),
                       j['position'][0] / 10, j['position'][1] / 10, j['priority'] / 10,
                       (j['deadline'] - t) / horizon,
                       j['work'] / j['service'] if j['service'] else 0.0, owner / len(settings.roster)])
        job_mask.append(1)
    if not all(math.isfinite(v) for v in vector):
        raise ValueError('Nonfinite critic state')
    return {'state': vector, 'robot_mask': robot_mask, 'job_mask': job_mask,
            'job_ids': [*global_slots, *([None] * (settings.max_jobs - len(global_slots)))]}

"""Non-mutating core adapter plus sensor/message/action timing instrumentation."""
import copy
from dataclasses import replace

from ..phase3a.contracts import SampledAction, validate_sample
from ..phase3a.env import ParallelCore
from ..phase3a.contracts import Settings
from ..phase3a.rewards import RewardWeights
from ..phase2.model import config_from_dict
from .slots import SlotMap


class AuditedEnv:
    def __init__(self, spec, order=(2, 3, 4), condition='controlled'):
        if spec['split'] == 'final_holdout':
            raise ValueError('Final holdout is sealed; no evaluation permitted in this phase')
        if condition not in ('controlled', 'realistic', 'zero_delay'):
            raise ValueError(condition)
        self.spec, self.layout, self.condition = spec, SlotMap(order), condition
        config = config_from_dict(spec['config'])
        settings = Settings(**{**spec['settings'], 'roster': tuple(spec['settings']['roster'])})
        if condition == 'controlled':settings = replace(settings, sensor_radius=100., sensor_noise=0., radio_delay=0.)
        if condition == 'zero_delay':settings = replace(settings, radio_delay=0.)
        self.core = ParallelCore(config, settings, RewardWeights())
        self.core.reset(spec['event_seed'], {'terminate_on_tasks_resolved': False})
        self.first = set(); self.now = 0.
        self._install_trace()
        self.obs = self._view()
        self._known()
        self.core.engine.rows[0]['repair_interface'] = {'scenario_id': spec['scenario_id'], 'split': spec['split'],
                                                       'condition': condition, 'new_to_old': list(self.layout.new_to_old)}

    def __getattr__(self, key):
        return getattr(self.core, key)

    def _event_once(self, event, agent, jid, **extra):
        key = (event, agent, jid)
        if key not in self.first:
            self.first.add(key)
            self.core.engine.event(self.now, event, robot=agent, job=jid, **extra)

    def _install_trace(self):
        perception = self.core.perception
        original = perception._cache
        def tracked(agent, category, entity, record, source):
            original(agent, category, entity, record, source)
            if category == 'jobs' and record['kind'] == 'task':
                event = 'task_sensor_first' if source == 'sensor' else 'task_message_first'
                self._event_once(event, agent, entity, measured_at=record['observed_at'])
        perception._cache = tracked
        # Reconstruct only reset-time actually measured/received records, never future tasks.
        for name, memory in perception.memory.items():
            for jid, record in memory['jobs'].items():
                if record['kind'] == 'task' and record['source'] == 'sensor':
                    self._event_once('task_sensor_first', name, jid, measured_at=record['observed_at'])
        received = [r for r in self.core.engine.rows if r.get('event') == 'radio_received' and r['time'] == 0]
        for event in received:
            for jid, record in perception.memory[event['sender']]['jobs'].items():
                if record['kind'] == 'task' and record['source'] == 'sensor':
                    self._event_once('task_message_first', event['recipient'], jid, measured_at=record['observed_at'])

    def _view(self):
        return {name: self.layout.packet(p) for name, p in self.core.observations.items()}

    def _known(self):
        for name, packet in self.obs.items():
            for i, jid in enumerate(packet['job_ids']):
                if jid is not None and not packet['observation'][52 + 9*i]:
                    self._event_once('task_actor_first_known', name, jid)

    def state(self):
        return self.layout.critic(self.core.state())

    def step(self, actions):
        if set(actions) != set(self.core.agents):raise ValueError('One action per present agent required')
        old_view, old_state = copy.deepcopy(self.obs), self.state()
        translated = {}; view_records = {}
        # Validate the complete batch before any transition or trace mutation.
        for name, item in actions.items():
            packet = self.obs[name]
            if isinstance(item, SampledAction):
                validate_sample(item, packet['action_mask'], packet['observation_token'])
                action = item.action
                translated[name] = SampledAction(self.layout.action_to_core(action), item.log_probability,
                                                 self.layout.to_core_distribution(item.action_mask),
                                                 self.layout.to_core_distribution(item.probabilities), item.policy_version,
                                                 self.core.observations[name]['observation_token'])
            else:
                if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 10 or not packet['action_mask'][item]:
                    raise ValueError('Invalid view action')
                action = item; translated[name] = self.layout.action_to_core(item)
            view_records[name] = {'sampled_action': action, 'core_action': self.layout.action_to_core(action),
                                  'job_ids': packet['job_ids'], 'action_mask': packet['action_mask']}
        self.now = self.core.time
        for name, record in view_records.items():
            action = record['sampled_action']; packet = self.obs[name]
            jid = None if action == 0 else packet['self_assignment'] if action == 1 else packet['job_ids'][action-2]
            if jid is not None and self.core.engine.jobs[jid]['kind'] == 'task':
                self._event_once('task_policy_first_selected', name, jid, action=action)
        self.now = round(self.core.time + self.core.c.dt, 10)
        _, rewards, terms, truncs, infos = self.core.step(translated)
        for name, info in infos.items():
            r = info['transition']; packet = old_view[name]; item = actions[name]
            r['sampled_action'] = view_records[name]['sampled_action']
            r['executed_action'] = self.layout.action_from_core(r['executed_action'])
            r['action_mask'] = packet['action_mask']
            r['actor_observation'] = packet['observation']
            r['observation_token'] = packet['observation_token']
            if isinstance(item, SampledAction):
                r['sample_observation_token'] = item.observation_token
                r['probabilities'] = list(item.probabilities)
            view_records[name]['transition'] = copy.deepcopy(r)
            info['critic_state'] = self.state()
        self.obs = self._view()
        self._known()
        # Trace records added after step must precede the terminal record for the legacy evaluator.
        rows = self.core.engine.rows
        terminal = next((r for r in reversed(rows) if r['type'] == 'terminal'), None) if self.done else None
        if terminal is not None:
            rows.remove(terminal); rows.append(terminal)
        transition = next(r for r in reversed(rows) if r['type'] == 'transition')
        transition['slot_adapter'] = {'new_to_old': list(self.layout.new_to_old), 'actor_before': old_view,
                                       'records': view_records, 'critic_before': old_state, 'critic_after': self.state()}
        for name, info in infos.items():
            if self.done:
                info['final_observation'] = copy.deepcopy(self.obs[name]); info['final_state'] = self.state()
        return copy.deepcopy(self.obs), rewards, terms, truncs, infos

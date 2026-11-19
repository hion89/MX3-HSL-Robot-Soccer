"""Fixed-size interface and checked behavior-policy action records."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Settings:
    roster: tuple = ('A', 'B', 'C', 'D')
    max_jobs: int = 8
    sensor_radius: float = 5.0
    sensor_noise: float = .02
    radio_enabled: bool = True
    radio_delay: float = 2.0
    cache_ttl: float = 4.0

    def validate(self, config):
        if len(self.roster) != 4 or len(set(self.roster)) != 4 or self.max_jobs < 1:
            raise ValueError('Four public roster slots and positive job capacity required')
        if not {r.id for r in config.robots} <= set(self.roster) or len(config.jobs) > self.max_jobs:
            raise ValueError('Episode exceeds fixed padding limits')
        values = (self.sensor_radius, self.sensor_noise, self.radio_delay, self.cache_ttl)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid sensing or communication setting')
        if abs(self.radio_delay / config.dt - round(self.radio_delay / config.dt)) > 1e-8:
            raise ValueError('Radio delay must be a multiple of dt')

    @property
    def n_actions(self):
        return self.max_jobs + 2  # 0 wait/release, 1 keep, 2+ local known-job slot


@dataclass(frozen=True)
class SampledAction:
    action: int
    log_probability: float
    action_mask: tuple
    probabilities: tuple
    policy_version: str = 'unspecified'
    observation_token: str | None = None


def validate_sample(sample, mask, observation_token=None):
    if observation_token is not None and sample.observation_token != observation_token:
        raise ValueError('Sample belongs to a different observation or agent')
    if isinstance(sample.action, bool) or not isinstance(sample.action, int) or not 0 <= sample.action < len(mask):
        raise ValueError('Discrete action out of range')
    if tuple(mask) != tuple(sample.action_mask) or not mask[sample.action]:
        raise ValueError('Stale mask or invalid sampled action')
    probs = sample.probabilities
    if len(probs) != len(mask) or any(not math.isfinite(p) or p < 0 for p in probs):
        raise ValueError('Invalid behavior probabilities')
    if abs(sum(probs) - 1) > 1e-7 or any(p != 0 for p, valid in zip(probs, mask) if not valid):
        raise ValueError('Behavior distribution disagrees with mask')
    if probs[sample.action] <= 0 or not math.isfinite(sample.log_probability):
        raise ValueError('Impossible action or nonfinite log probability')
    if abs(math.log(probs[sample.action]) - sample.log_probability) > 1e-6:
        raise ValueError('Behavior log-probability disagrees with sampled action')


def validate_ppo_record(record):
    """Validate the SAMPLED action, deliberately never the arbitrated action."""
    if record['log_probability'] is None or record['probabilities'] is None:
        raise ValueError('Scripted integer actions are not PPO training samples')
    validate_sample(SampledAction(record['sampled_action'], record['log_probability'],
                                  tuple(record['action_mask']), tuple(record['probabilities']),
                                  observation_token=record['sample_observation_token']),
                    record['action_mask'], record['observation_token'])
    return bool(record['policy_loss_mask'])

"""Fixed-size batch adapter and optional lazy PettingZoo/Gymnasium adapter."""
import copy


def padded_batch(observations, critic, settings):
    actor_size, state = 50 + 12 * settings.max_jobs, critic['state']
    actor, masks, active, present, states = [], [], [], [], []
    for name in settings.roster:
        obs = observations.get(name)
        actor.append(list(obs['observation']) if obs else [0.0] * actor_size)
        masks.append(list(obs['action_mask']) if obs else [1, *([0] * (settings.n_actions - 1))])
        active.append([obs['active_mask'] if obs else 0])
        present.append([int(obs is not None)])
        states.append(list(state) if obs else [0.0] * len(state))
    return {'actor_observations': actor, 'available_actions': masks, 'active_masks': active,
            'present_masks': present, 'critic_states': states}


def make_pettingzoo(core):
    """Not invoked during Phase 3A tests; installation is deliberately not automatic."""
    try:
        import numpy as np
        from gymnasium import spaces
        from pettingzoo import ParallelEnv
    except ImportError as exc:
        raise RuntimeError('Optional PettingZoo adapter requires a separately approved environment') from exc

    class Adapter(ParallelEnv):
        metadata = core.metadata
        render_mode = None

        def __init__(self):
            self.possible_agents = list(core.possible_agents)
            self.observation_spaces = {a: spaces.Dict({
                'observation': spaces.Box(-np.inf, np.inf, shape=(core.observation_size,), dtype=np.float32),
                'action_mask': spaces.MultiBinary(core.settings.n_actions)}) for a in self.possible_agents}
            self.action_spaces = {a: spaces.Discrete(core.settings.n_actions) for a in self.possible_agents}
            self.state_space = spaces.Box(-np.inf, np.inf, shape=(core.state_size,), dtype=np.float32)

        @property
        def agents(self):
            return core.agents

        def observation_space(self, agent):
            return self.observation_spaces[agent]

        def action_space(self, agent):
            return self.action_spaces[agent]

        def _convert(self, obs):
            return {a: {'observation': np.asarray(o['observation'], dtype=np.float32),
                         'action_mask': np.asarray(o['action_mask'], dtype=np.int8)} for a, o in obs.items()}

        def reset(self, seed=None, options=None):
            obs, infos = core.reset(seed, options)
            return self._convert(obs), infos

        def step(self, actions):
            obs, rewards, terms, truncs, infos = core.step(actions)
            return self._convert(obs), rewards, terms, truncs, infos

        def state(self):
            return np.asarray(core.state()['state'], dtype=np.float32)

        def close(self):
            core.close()

    return Adapter()

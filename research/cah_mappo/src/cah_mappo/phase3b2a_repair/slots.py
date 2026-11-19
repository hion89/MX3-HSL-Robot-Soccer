"""Bijective, episode-stable relabeling; no future job registry is inspected."""
import copy
import hashlib
import itertools
import json

PERMUTATIONS = tuple(itertools.permutations((2, 3, 4)))


class SlotMap:
    def __init__(self, order=(2, 3, 4)):
        if tuple(sorted(order)) != (2, 3, 4):
            raise ValueError('Permute logical slots 2/3/4 exactly once')
        self.new_to_old = (0, 1, *tuple(order), 5, 6, 7)
        self.old_to_new = tuple(self.new_to_old.index(i) for i in range(8))

    def action_to_core(self, action):
        return action if action < 2 else self.new_to_old[action - 2] + 2

    def action_from_core(self, action):
        return action if action < 2 else self.old_to_new[action - 2] + 2

    def to_core_distribution(self, values):
        return (values[0], values[1], *(values[self.old_to_new[i] + 2] for i in range(8)))

    def packet(self, packet):
        out = copy.deepcopy(packet); old = packet['observation']; order = self.new_to_old
        out['observation'] = (old[:10] + [old[10 + i] for i in order] + old[18:50]
                              + [v for i in order for v in old[50 + 9*i:59 + 9*i]]
                              + old[122:130] + [old[130+i] for i in order] + [old[138+i] for i in order])
        for key in ('job_ids', 'job_mask', 'job_fresh_mask'):
            out[key] = [packet[key][i] for i in order]
        out['action_mask'] = packet['action_mask'][:2] + [packet['action_mask'][i+2] for i in order]
        # The adapter token is separate from the core's original observation token.
        out['observation_token'] = hashlib.sha256(json.dumps({'core_token': packet['observation_token'],
                                                             'order': order, 'view': out}, sort_keys=True).encode()).hexdigest()
        return out

    def critic(self, state):
        out = copy.deepcopy(state); old = state['state']; order = self.new_to_old
        vector = [old[0]]
        for robot in range(4):
            start = 1 + robot * 17
            vector.extend(old[start:start+9])
            vector.extend(old[start+9+i] for i in order)
        vector.extend(v for i in order for v in old[69 + 9*i:78 + 9*i])
        out['state'] = vector
        for key in ('job_ids', 'job_mask'):out[key] = [state[key][i] for i in order]
        return out

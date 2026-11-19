"""Local policies take actor packets only, never an Environment or critic state."""
import math
import random


class LocalPolicy:
    def __init__(self, mode, seed=0):
        if mode not in ('local_hold', 'local_fixed', 'local_eta', 'local_random'):
            raise ValueError(mode)
        self.mode = mode
        self.rng = random.Random(f'local-policy:{mode}:{seed}')

    def decide(self, name, packet):
        o, mask = packet['observation'], packet['action_mask']
        valid = [i for i, bit in enumerate(mask) if bit]
        if self.mode == 'local_random':
            return self.rng.choice(valid)
        keep = 1 if mask[1] else 0
        if self.mode == 'local_hold' or not packet['active_mask']:
            return keep
        candidates = []
        for slot, jid in enumerate(packet['job_ids']):
            action = slot + 2
            if not mask[action]:
                continue
            row = o[50 + slot * 9:50 + (slot + 1) * 9]
            is_base, dx, dy, priority = bool(row[2]), row[3] * 10, row[4] * 10, row[5] * 10
            if self.mode == 'local_fixed':
                if is_base == bool(o[7]):
                    candidates.append((0 if jid == packet['self_assignment'] else 1, slot, action))
                continue
            # Translation + turn, using only own odometry and cached peer reports.
            own_heading, speed = math.atan2(o[2], o[3]), max(o[4] * 2, 1e-8)
            own_eta = math.hypot(dx, dy) / speed + abs(math.remainder(math.atan2(dy, dx) - own_heading, 2 * math.pi)) / max(o[5] * math.pi, 1e-8)
            nearest = (own_eta, name)
            for i, peer in enumerate(('A', 'B', 'C', 'D')):
                if not packet['peer_fresh_mask'][i]:
                    continue
                p = o[18 + 8 * i:26 + 8 * i]
                if not p[6]:
                    continue
                px, py = dx - p[1] * 10, dy - p[2] * 10
                peer_eta = math.hypot(px, py) / max(p[5] * 2, 1e-8)
                # Turn speed of peers is not in the packet; no privileged estimate.
                nearest = min(nearest, (peer_eta, peer))
            if nearest[1] == name or jid == packet['self_assignment']:
                candidates.append((-priority, own_eta, action))
        return min(candidates)[-1] if candidates else keep

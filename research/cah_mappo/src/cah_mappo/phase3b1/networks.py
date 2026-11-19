"""Separate shared Actor and environment-level Critic, always on CPU."""
import math
import torch
from torch import nn
from torch.distributions import Categorical


def mlp(inputs, outputs, hidden=64, output_gain=1.0):
    net = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, math.sqrt(2)); nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, output_gain)
    return net


class ActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.actor = mlp(146, 10, hidden, .01)
        self.critic = mlp(141, 1, hidden)

    def distribution(self, observations, masks):
        if observations.ndim != 2 or observations.shape[-1] != 146 or masks.shape != (observations.shape[0], 10):
            raise ValueError('Actor/mask shape mismatch')
        if not torch.isfinite(observations).all() or not ((masks == 0) | (masks == 1)).all() or not (masks.sum(-1) > 0).all():
            raise ValueError('Invalid actor input or all-masked distribution')
        # Float64 probabilities satisfy the strict stdlib action audit without
        # replacing the probability with a separately recomputed scalar.
        logits = self.actor(observations.float()).double().masked_fill(~masks.bool(), -torch.inf)
        return Categorical(logits=logits)

    def value(self, states):
        if states.ndim != 2 or states.shape[-1] != 141 or not torch.isfinite(states).all():
            raise ValueError('Critic shape or finiteness error')
        return self.critic(states.float()).squeeze(-1)

    def sample(self, observations, masks, generator):
        dist = self.distribution(observations, masks)
        actions = torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
        return actions, dist.log_prob(actions), dist.probs

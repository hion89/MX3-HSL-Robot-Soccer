"""킥 시계 — kick_*.py 가 '킥 시작부터 몇 초' 를 잴 때 쓰는 시각.

기본은 에피소드 시각 (episode_length_buf × step_dt) 그대로다. 보행·킥 통합 태스크
(RS02 WalkKickUnified) 는 걷는 도중에 킥을 시작하므로 env.kick_t0_steps (킥 시작 순간의
episode_length_buf) 를 두고, 그 값을 빼서 킥 시작 = 0 으로 맞춘다.
kick_t0_steps 가 없는 기존 태스크 (MX3 WalkKick, RS02 WalkKick-V0·V0-1) 는 값이 전과 같다.
"""
from __future__ import annotations

import torch


def kick_clock_steps(env) -> torch.Tensor:
    if hasattr(env, "kick_t0_steps"):
        return env.episode_length_buf - env.kick_t0_steps
    return env.episode_length_buf


def kick_clock_s(env) -> torch.Tensor:
    return kick_clock_steps(env).float() * env.step_dt

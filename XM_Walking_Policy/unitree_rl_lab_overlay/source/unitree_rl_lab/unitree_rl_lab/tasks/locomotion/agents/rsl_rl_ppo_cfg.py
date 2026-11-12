# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=3,       # 5 → 3 (양성 피드백 루프 약화)
        num_mini_batches=4,
        learning_rate=5.0e-4,        # 1e-3 → 5e-4 (value spike 시 업데이트 크기 절반)
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,  # 기본 1.0 → 0.5로
    )


@configclass
class GetupV12PPORunnerCfg(BasePPORunnerCfg):
    """Getup-V1-2 전용 — BasePPORunnerCfg + **액션 클리핑**.

    【왜 — 2026-09-14 액션 폭주로 롤백 반복】
      V1-2 2 판에서 롤백이 5 회 났다. value_loss 실측:
          62397155133915922432.0  (6.2e19)
          inf
          20816537909401530459488256.0  (2.1e25)
          inf
      텐서보드의 vloss 는 0.18 로 멀쩡해 보이는데, **롤백된 업데이트는
      기록되지 않기 때문**이다. 스파이크는 로그에 안 남는다.

      원인: clip_actions=None 이라 정책 출력에 제한이 없다. 관절 목표는
      DelayedJointPositionAction 이 target_limits 로 클램프하므로
      **로봇 동작에는 영향이 없고**, 정책은 "액션을 키워도 손해가 없다" 를
      학습할 수 있다. 그러다 값이 커지면 action_rate = |a_t - a_{t-1}|^2 이
      float32 에서 폭발하고 value_loss 가 inf 가 된다.

      V2 15 판에서 같은 증상을 겪었다 (action_rate 2.06e15, reward -inf,
      vloss inf, 로봇은 멀쩡히 기립). 거기서 clip_actions=100.0 으로 해결했다.

      ※ 1 판에는 안 나타났다. 페널티가 램프인 전이라 action_rate 항이
        -0.17 수준으로 작았기 때문이다. 2 판에서 페널티를 처음부터 전량으로
        켜자(-3 대) 정책이 그 항을 회피하려 액션 공간의 먼 곳으로 밀려나며
        폭발 구간에 들어갔다.

    【100.0 인 이유】
      scale=0.25 이므로 100 이면 관절 목표 25 rad. target_limits 가 어차피
      자르는 범위(최대 2.97 rad)보다 훨씬 넓어 **정상 동작에는 전혀 제약이
      안 된다.** 폭주만 막는다.

    ※ 미러 손실은 붙이지 않는다 — v1 계보의 설계 결정이다
      ("대칭은 하드 제약 아님: 중간 과정의 비대칭(옆으로 구르기)을 막지 않는다").
    """

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.clip_actions = 100.0

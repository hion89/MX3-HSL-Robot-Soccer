# rsl_rl_ppo_v1_cfg.py
# ============================================================================
# v1 환경 전용 PPO 설정.
#
# 기존 BasePPORunnerCfg 와의 차이는 "네트워크 배분" 하나뿐이다.
# 학습률/에포크/미니배치는 그대로 둔다 — 이 둘은 서로 얽혀 있어서
# 함께 바꾸면 원인 분리가 안 된다. (htwk 의 lr 5e-6 + 20 epoch 조합은
# full-batch + 매 epoch advantage 재계산과 세트라서 부분 이식이 위험하다)
#
# ── 왜 비대칭으로 바꾸나 ──
# 기존은 actor/critic 이 둘 다 [512, 256, 128] 로 동일하다. 그런데
#   - actor  : Jetson 에서 매 스텝 추론. sim-to-real 일반화가 걸림.
#              큰 네트워크는 시뮬레이터 특유의 패턴을 외우기 쉽다.
#   - critic : 학습 중에만 GPU 에서 돌고 배포 시 버려짐. 용량이 공짜.
#              게다가 DR 로 질량/COM/마찰이 매번 달라지는 상황에서
#              리턴을 예측해야 하니 과제 자체가 더 어렵다.
# 따라서 actor 를 줄이고 critic 은 유지하는 쪽이 합리적이다.
# htwk 도 actor [256,128,128] / critic [256,256,128] 로 critic 을 1.8 배
# 넓게 잡아 두었다.
#
# 파라미터 수 비교 (관측 차원 포함):
#   기존   actor 281k / critic 288k
#   v1     actor 118k / critic 288k     (관측 47*5=235 기준)
# ============================================================================

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class V1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # task name 과 동일하게 자동 설정됨
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        # init_noise_std 1.0 은 action_scale 0.25 기준 관절 타겟 +-0.25 rad
        # (= 14.3 deg) 의 초기 탐색 노이즈다. 액션 범위(+-1)와 같은 크기라
        # 초기 정책이 사실상 완전 랜덤이 된다.
        # 학습 로그의 Policy/mean_noise_std 가 수천 iter 뒤에도 0.5 이상이면
        # 이 값을 0.5 로 낮추거나 entropy_coef 를 0.005 로 줄일 것.
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],   # 축소 — 배포 대상
        critic_hidden_dims=[512, 256, 128],  # 유지 — DR 설명 부담이 큼
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class V1PPORunnerLongHorizonCfg(V1PPORunnerCfg):
    """gamma 를 0.995 로 올린 변형 (선택).

    50 Hz 기준 유효 지평이 2 s -> 4 s 로 늘어난다. 보행은 한 스텝의 결과가
    다음 스텝에 누적되므로 긴 지평이 유리하다. 다만 value 추정 분산이
    커지므로 critic 이 안정된 뒤에(= 기본 설정으로 한 번 수렴시킨 뒤)
    시도할 것.
    """

    def __post_init__(self):
        self.algorithm.gamma = 0.995

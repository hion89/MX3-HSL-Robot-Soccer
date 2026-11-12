# rsl_rl_ppo_sym_cfg.py
# ============================================================================
# BasePPORunnerCfg + 좌우 대칭(mirror) 손실
# ----------------------------------------------------------------------------
# A/B 실험 설계
#   실험군과 대조군의 차이가 "미러 손실 하나"가 되도록, 환경 설정
#   (velocity_v0_2_env_cfg) 과 PPO 하이퍼파라미터를 모두 공유한다.
#   BasePPORunnerCfg 를 상속하고 symmetry_cfg 만 덧붙이는 이유가 이것이다.
#   값을 복사해 두면 나중에 Base 를 고칠 때 두 곳이 어긋난다.
#
#       대조군: Unitree-MX3-Velocity-V0-2       (BasePPORunnerCfg)
#       실험군: Unitree-MX3-Velocity-V0-2-Sym   (SymmetryPPORunnerCfg)
#
# ── use_data_augmentation 을 끄는 이유 ─────────────────────────────────
#   데이터 증강은 미러링된 샘플을 미니배치에 추가해 학습한다. 그런데
#   PPO 는 on-policy 라 증강 샘플이 현재 정책의 행동 분포에서 나온 것이
#   아니고, IsaacLab 이슈 #2568 에서 "증강이 새로운 비대칭 아티팩트를
#   만든다"는 보고가 있었다.
#
#   미러 손실은 정책 출력에 대한 MSE 정칙화라 롤아웃 분포를 건드리지
#   않는다. 먼저 이쪽만 켜고, 효과를 확인한 뒤에 증강을 실험한다.
#   (변경점 하나씩 — v1 에서 8~9 가지를 한꺼번에 바꿔 원인 분리에
#    실패했던 것을 반복하지 않는다)
#
# ── mirror_loss_coeff 튜닝 ────────────────────────────────────────────
#   논문(arXiv 2511.03996) Appendix D 는 symmetry coefficient 10 을 쓴다.
#   다만 그쪽은 AMP 스타일 리워드가 함께 있어 손실 스케일이 다르므로
#   그대로 옮기면 안 된다. 1.0 에서 시작한다.
#
#   텐서보드의 Loss/symmetry 를 보고 판단한다.
#     - 초반부터 거의 0    → 정책이 이미 대칭. 계수를 올려도 이득 없음
#     - 계속 큰 값 유지    → 계수를 올리거나(3.0, 10.0) 매핑을 재확인
#     - track_lin_vel 이 눈에 띄게 떨어짐 → 계수가 과함. 0.3 으로 낮춤
#
#   ★ 학습 전에 반드시 scripts/rsl_rl/check_symmetry_mx3.py 를 통과시킬 것.
#     매핑이 틀린 채로 학습하면 미러 손실이 보행을 적극적으로 망가뜨린다.
# ============================================================================

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.locomotion.mdp.symmetry_mx3 import compute_symmetric_states


@configclass
class SymmetryPPORunnerCfg(BasePPORunnerCfg):
    """BasePPORunnerCfg 와 모든 하이퍼파라미터를 공유하고 미러 손실만 추가."""

    def __post_init__(self):
        # 상위 클래스에 __post_init__ 이 있으면 먼저 실행 (없어도 안전)
        parent_post = getattr(super(), "__post_init__", None)
        if parent_post is not None:
            parent_post()

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            # 데이터 증강 off — 롤아웃 분포를 건드리지 않는다 (위 주석 참조)
            use_data_augmentation=False,
            # 미러 손실 on — 정책 출력에 대한 MSE 정칙화
            use_mirror_loss=True,
            mirror_loss_coeff=1.0,
            data_augmentation_func=compute_symmetric_states,
        )


@configclass
class SymmetryMetricOnlyPPORunnerCfg(BasePPORunnerCfg):
    """미러 손실을 학습에 반영하지 않고 "지표로만" 기록하는 설정.

    rsl_rl 은 use_data_augmentation / use_mirror_loss 가 모두 False 여도
    data_augmentation_func 를 호출해 대칭 지표를 계산·로깅한다.

    용도
      - 기존 v0_2 학습을 그대로 재현하면서 비대칭 정도가 실제로 얼마나
        되는지 먼저 관측하고 싶을 때
      - 미러 매핑이 학습에 영향을 주기 전에 텐서보드에서 안전하게
        검증하고 싶을 때
    이 설정으로 몇백 iter 돌려보고 Loss/symmetry 가 유의미하게 크면
    SymmetryPPORunnerCfg 로 넘어가는 순서를 권장한다.
    """

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if parent_post is not None:
            parent_post()

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=compute_symmetric_states,
        )


@configclass
class WalkKickPPORunnerCfg(SymmetryPPORunnerCfg):
    """walk_kick_v0 용. 미러 손실 설정은 v0_3 와 동일하다.

    킥에서 미러 손실이 더 중요한 이유가 있다. 우연히 오른발 킥이 먼저
    조금 성공하면 정책이 공을 오른발 쪽에 두려고 몸을 돌리고, 그러면
    왼발 킥 데이터가 아예 안 쌓여서 격차가 벌어진다. 양의 되먹임이라
    한 번 굳으면 안 풀린다.

    arXiv 2511.03996 Appendix D 의 도입 동기가 정확히 이것이다
    ("consistently kicking with a single leg" 방지).

    텐서보드의 metric_kick_foot_balance 로 확인할 것.
    +1 이면 오른발만, -1 이면 왼발만 쓰는 중. 0 근처를 유지해야 한다.
    한쪽으로 치우치면 mirror_loss_coeff 를 올린다.
    """

    def __post_init__(self):
        super().__post_init__()
        # 킥은 좌우 편향 압력이 보행보다 강하므로 계수를 조금 높인다.
        # 논문은 10 을 쓰지만 그쪽은 AMP 손실이 함께 있어 스케일이 다르다.
        #
        # [2026-08-08] coeff 2.0 으로 학습한 v1 이 "왼발로만 킥" 편향으로
        # 굳었다(양의 되먹임 락 — 위 클래스 docstring 참조). soft 미러 손실은
        # 이미 굳은 편향을 잘 못 푸므로, 스크래치 재학습에서 편향이 굳기
        # 전부터 강한 대칭 압력을 걸어 예방한다. 2.0 → 5.0.
        # (보행 track_lin_vel 이 눈에 띄게 떨어지면 과함 — 3.0 으로 낮출 것)
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 5.0

        # [2026-08-13] soft 미러손실(coeff5)+wrong_foot 벌점으로도 킥이 한쪽 발로
        # 고착(오른발→왼발 뒤집힘)됨. 원인: 희귀 킥이 평균 미러손실에 희석 + 벌점
        # 스케일 약함. data augmentation 은 미러 샘플을 학습데이터에 직접 주입해
        # 좌/우를 완벽 균형시켜 한쪽 발 고착을 "구조적으로" 불가능하게 함.
        # (wrong_foot 벌점은 근/원발 대칭해 중 '가까운 발' 쪽으로 tie-break — 유지)
        self.algorithm.symmetry_cfg.use_data_augmentation = True


@configclass
class SymmetryPPOMirror3RunnerCfg(SymmetryPPORunnerCfg):
    """v0_4 전용 — mirror_loss_coeff 만 1.0 → 3.0.

    ── 왜 별도 클래스인가 ────────────────────────────────────────────
    SymmetryPPORunnerCfg 를 직접 고치면 Unitree-MX3-Velocity-V0-3 과
    WalkKickPPORunnerCfg(이 클래스를 상속한다)까지 함께 바뀐다. 실기 검증된
    베이스라인은 건드리지 않는다는 이 레포의 원칙에 어긋나므로 분리한다.

    ── 왜 3.0 인가 ──────────────────────────────────────────────────
    v0_4 2 차 런(2026-08-22_13-40-04) 관측:
        창3(iter 1000-1499)  Loss/symmetry 0.0042
        창10(iter 4500-4999) Loss/symmetry 0.0073   (+74%)
      pelvis_pitch_symmetry  -0.0385 → -0.0467
      joint_deviation_hip_yaw    -0.0402 → -0.0561
      joint_deviation_pelvis_roll -0.0482 → -0.0593
    entropy 가 -0.84 → 5.75 로 오르면서 좌우 관련 항목이 함께 나빠졌다.
    계수 1.0 의 미러 손실이 그 드리프트를 못 잡고 있다는 뜻이다.

    이 파일 상단 주석의 튜닝 지침("계속 큰 값 유지 → 계수를 올리거나
    3.0, 10.0")을 그대로 따른다. 10.0 이 아니라 3.0 부터 가는 이유는
    미러 손실이 과하면 track_lin_vel 이 떨어질 수 있어서다(상단 주석 참조).

    ── 판정 방법 ────────────────────────────────────────────────────
    같은 iteration 구간에서 2 차 런과 Loss/symmetry 를 비교한다.
    track_lin_vel_xy 가 눈에 띄게 떨어지면 계수가 과한 것이니 1.5~2.0 으로.
    변화가 없으면 10.0 을 시도한다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 3.0


@configclass
class GetupPPOMirror03RunnerCfg(SymmetryPPORunnerCfg):
    """기상 V2 전용 — mirror_loss_coeff 1.0 -> 0.3.

    【왜 3.0 이 아니라 0.3 인가 — 기상은 본질적으로 비대칭이다】
      보행은 좌우 대칭이 정답이라 3.0 이 잘 맞는다 (v0_5+). 그런데 기상은
      옆으로 구르고, 한쪽 팔로 짚고, 한 다리를 먼저 세운다. 그 과도구간에
      대칭을 강하게 걸면 과제와 정면 충돌한다.

      Getup-V2 9~10 판 실측 (coeff 3.0). Loss/symmetry 는 계수를 곱하기
      **전** 원시 MSE 다 (ppo.py: loss += coeff * symmetry_loss 후 detach):
          base_z 0.062 (누움)   sym 0.0033   <- 안정
          base_z 0.126          sym 0.0038
          base_z 0.206          sym 0.0616
          base_z 0.238          sym 0.8658   vloss   663.99
          base_z 0.158          sym 342.27   vloss 46139.79   <- 폭발
      **일어나는 동안 sym 이 5 자릿수로 폭발**한다. 그 결과 value_loss 가
      문턱(2000 으로 올린 뒤에도)을 넘어 롤백이 왕복했다.

      v1 설계 노트도 같은 말을 한다:
        "대칭은 하드 제약 아님: 중간 과정의 비대칭(옆으로 구르기)을 막지 않는다"

      9 판에서 레벨 0(5 도, 거의 서 있음)일 때 sym 0.0033 으로 잘 됐던 게
      오해를 줬다. 각도가 커져 실제 기상 동작이 필요해지자 무너졌다.

    【0.3 의 근거】
      3.0 대비 1/10. 폭발 구간 기여가 1026 -> 103 으로 줄어든다.
      정상 구간(sym 0.005~0.02)에서는 어차피 기여가 미미하므로,
      "기상 중 비대칭은 허용하고 정상 상태만 약하게 당긴다" 에 가까워진다.
      이걸로도 불안정하면 계수 튜닝이 아니라 **기립 후에만 거는 게이팅**이
      필요하다는 뜻이다 (rsl_rl symmetry 는 배치 단위라 구현 부담이 크다).
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 0.3


@configclass
class GetupPPOMirror03ClipRunnerCfg(GetupPPOMirror03RunnerCfg):
    """기상 V2 — 미러 0.3 + **액션 클리핑**.

    【왜 — 2026-09-11 액션 폭주】
      15 판에서 Train/mean_reward 가 -inf, Loss/value_function 이 inf 로
      나왔다. 그런데 관절 위치·토크는 전부 유한하고 로봇은 멀쩡히
      일어난다(stand 4.8, base_z 0.377). 원인을 항별로 추적하니:

          항                비유한   |값| 최대
          action_rate           0    2.06e+15   <<<
          base_height           0       11.82
          stand_success         0        5.00

      **action_rate 하나가 10^15 다.** action_rate = |a_t - a_{t-1}|^2 이므로
      액션이 10^7 규모라는 뜻이다. float32 누적 합에서 inf 로 넘어간다.

      원인: clip_actions=None 이라 정책 출력에 제한이 없다. 관절 목표는
      DelayedJointPositionAction 이 target_limits 로 클램프하므로 **로봇
      동작에는 영향이 없고**, 정책은 "액션을 키워도 손해가 없다" 를
      학습할 수 있다. action_rate(-0.15)가 그걸 벌주려 하지만 스케일이
      이미 깨졌다.

      ※ 이건 토크 페널티와 무관한 기존 문제다. 14 판의 vloss 11988 도
        같은 원인으로 보인다.

    【100.0 인 이유】 scale=0.25 이므로 100 이면 관절 목표 25 rad —
      어차피 리밋에 클램프되는 범위라 정상 동작에 영향이 없다.
      폭주(10^7)만 막는 안전장치다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.clip_actions = 100.0

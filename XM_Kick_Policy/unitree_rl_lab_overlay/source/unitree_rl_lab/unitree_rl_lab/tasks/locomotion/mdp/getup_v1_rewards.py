"""getup_v1 (supine 기상) 전용 리워드.

【설계 근거】
  구조는 HumanUP (RSS 2025, Unitree G1 supine/prone 기상, Apache 2.0,
  github.com/RunpeiDong/HumanUP) 의 검증된 구성을 참고했다. 다만 G1 은
  몸통 목표 0.728 m 로 MX3(0.39 m)의 약 2 배라, 가중치와 수식은 그대로
  옮기지 않고 [0,1] 정규화 후 MX3 실측값으로 다시 맞췄다.

  참고한 아이디어
    - 진행(delta) 보상    : 절대 높이가 평평한 구간에서도 신호가 안 죽는다
    - 발 하중 보상        : 기상의 본질은 접촉 모드 전환(등 → 발)이다
    - 자세 항은 낮은 비중 : 높이가 주 신호, 자세는 보조
    - 대칭은 하드 제약 아님: 중간 과정의 비대칭(옆으로 구르기)을 막지 않는다

  가져오지 않은 것
    - 수직 보조력(drag force 1500 N): 보수적으로 일단 뺐다. 희소 보상
      장벽을 못 넘는 게 확인되면 그때 넣는다.
    - 머리 높이: MX3 는 목이 짧아 증폭이 없고, 프로젝트에 head link 를
      쓰는 곳이 없다.

【자세와 projected_gravity_b 대응】
    직립   : (  0,  0, -1)      supine : ( -1,  0,  0)      prone : ( +1, 0, 0)
  따라서 g_z 는 supine 에서 0, 직립에서 -1 이다.
  (실측: check_getup_spawn.py → (-0.999, 0.000, +0.034))

【MX3 실측 기준값】
    supine 정착 base_z = 0.057   (check_getup_spawn.py 3 회)
    기립      base_z = 0.396   (보행 스모크 3 태스크)
    보행 리워드 목표    = 0.39    (velocity_v0_6:522, walk_kick_v1_4:931)

【홀드 마스킹】
  전 항이 hold_masked 로 감싸여 리셋 홀드(낙하·정착) 구간에서 0 이다.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .getup_v1_mdp import _feet_load_frac, hold_masked

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ═══════════════════════════════════════════════════════════════
#  공통 헬퍼
# ═══════════════════════════════════════════════════════════════
def _height_frac(env, asset_cfg, supine_height: float, stand_height: float):
    """몸통 높이를 [0,1] 로 정규화. supine 0 → 기립 1."""
    asset: RigidObject = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    return torch.clamp((z - supine_height) / (stand_height - supine_height), 0.0, 1.0)


def _stand_gate(env, supine_height: float, stand_height: float,
                gate_from: float = 0.85,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """기립 후에만 열리는 게이트 [0,1]. 안정화 항 전용.

    ★ height_frac 을 그대로 곱으면 안 된다 ★
      supine 정착 높이가 기준값 0.057 보다 살짝 높아(실측 0.0565~0.0580)
      height_frac 이 0 이 아니라 0.003~0.06 이 나온다. 그런데 안정화 항의
      **원값이 누운 자세에서 매우 크다** — 발바닥이 하늘을 보므로
      foot_flat_roll 원값이 20 을 넘는다. 작은 계수를 곱해도 무시 못 할
      벌점이 남는다 (실측: supine 에서 foot_flat_roll 1.26, base_roll 0.60).

      누워 있을 때 발바닥이 뒤집힌 건 당연하고 벌주면 안 된다. 그래서
      문턱(gate_from) 아래는 **확실히 0** 으로 만들고 그 위에서만 선형으로
      연다.

    ★ 문턱을 0.5 로 잡았다가 실패했다 (10차, iter 500) ★
      0.5 는 base_z 0.224 로 **아직 일어서는 중**인 높이다. 그 지점부터
      "가만히 있으면 보너스"(stand_still_lin/ang, 합 +0.76/초)가 켜지니
      정책이 거기서 멈추는 게 이득이 된다. 실측:
          height_frac 0.778 인데 upright_frac 0.80 (문턱 0.95 미달)
          stand_success 0.24 / 최대 5.0
          9차 동시점 성공률 0.9995 -> 10차 0.0600
      새 항의 **페널티**는 문제가 아니었다 (합 -0.19, action_rate -2.21
      대비 미미). 양수 보너스가 중간 높이에 국소해를 만든 것이다.

      0.85 = stand_success 의 height_frac_threshold 와 같은 값(base_z 0.340).
      실제로 다 일어선 뒤에만 켜지고, 중간에 멈추면 못 받는다.
    """
    h = _height_frac(env, asset_cfg, supine_height, stand_height)
    return torch.clamp((h - gate_from) / (1.0 - gate_from), 0.0, 1.0)


def _upright_frac(env, asset_cfg):
    """직립 정도를 [0,1] 로. prone 0 → supine 0.5 → 직립 1.

    ★ clamp(-g_z, 0, 1) 을 쓰면 안 된다 ★
      그 식은 "supine 에서 0" 이라는 겉보기 속성은 만족하지만, g_z > 0
      (엎드린 쪽) 반평면 전체에서 값이 0 으로 눌려 **기울기가 사라진다**.
      로봇은 supine(g_z≈0)에서 시작하므로 정확히 그 경계에 서 있고,
      한 번 엎드린 쪽으로 넘어가면 되돌아올 신호가 아예 없다.

      실측 (2026-08-28, 3 차 학습):
        iter  50  upright 0.0010
        iter 100  upright 0.0036   ← 최고점
        iter 250  upright 0.0000
        iter 500  upright 0.0000   ← 250 부터 계속 정확히 0
      upright==0 은 g_z>=0 을 뜻한다(함수 정의상 확정). 즉 정책이 사각
      지대로 들어가 눌러앉았다. base_height 도 iter 100 부터 400
      iteration 동안 0.61 에서 평평했다.

    (1 - g_z)/2 는 전 구간에 기울기가 있고 엎드린 쪽을 능동적으로 벌준다.
    supine 에서 0.5 를 지급하지만 그건 상수 오프셋이라 PPO 의 어드밴티지
    에서 상쇄된다 — 기울기가 있는 것이 훨씬 중요하다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.5 * (1.0 - asset.data.projected_gravity_b[:, 2])


# ═══════════════════════════════════════════════════════════════
#  태스크 — 주 신호
# ═══════════════════════════════════════════════════════════════
@hold_masked
def base_height_progress(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """몸통 높이 [0,1]. 이 태스크의 주 신호다.

    HumanUP 의 exp(clamp(z,0,target))-1 를 그대로 쓰지 않는다. 그 형태는
    절대 높이를 지수에 직접 넣어 로봇 크기에 민감하다. G1 은 값 폭이
    0.91 인데 MX3 에 그대로 쓰면 0.42 로 쪼그라들어, 의도치 않게 다른
    항 대비 비중이 절반이 된다.

    1 단계에서는 선형이다. 볼록(h**2 등)하게 만들면 초기 구간 기울기가
    죽어 "일단 조금이라도 올라가기" 가 어려워진다.
    """
    return _height_frac(env, asset_cfg, supine_height, stand_height)


@hold_masked
def height_rising(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """이번 스텝에 몸통이 올라갔으면 1, 아니면 0.

    절대 높이가 거의 안 변하는 구간(옆으로 구르는 중, 팔을 짚는 중)에서도
    신호가 남는다. 절대값 셰이핑만 쓰면 그 구간이 평평해져 탐색이 멈춘다.

    env 에 상태를 붙여 이전 스텝 높이를 기억한다. 리셋 시점은 홀드
    마스킹이 어차피 0 을 반환하므로 따로 처리하지 않는다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    last = getattr(env, "_getup_last_base_z", None)
    if last is None or last.shape != z.shape:
        last = z.clone()
    rew = (z > last).float()
    env._getup_last_base_z = z.clone()
    return rew


@hold_masked
def upright_progress(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """직립 정도 × 높이 비율. 둘 다 만족해야 만점이다.

    supine 시작점을 0 으로 만들려고 clamp 를 썼다가 기울기 사각지대를
    만들어 학습이 막혔다. 근거는 _upright_frac docstring 참조.

    ★ 높이 비율을 곱하는 이유 — "정체 지점에서 타먹는 보상" 제거 ★
      정직한 직립도만 쓰면, 앉은 채로 상체만 세워도 만점이 나온다.
      실측(5차 iter 8000, 정체 지점)의 리워드 구성:
          upright      2.312 / 초   ← 전체의 64%. 최댓값 2.4925 에 근접
          base_height  0.779
          height_rising 0.258
          feet_load    0.163
          walk_ready   0.100
      **upright 하나가 정체 지점 보상의 64%** 였다. 게다가 몸통이 이미
      수직(g_z -0.994)이라 최댓값이어서, 일어서려고 앞으로 숙이면
      오히려 손해다 — 30° 숙이면 0.16/초를 잃고, 그걸 즉시 상쇄하려면
      base_z 를 1.8 cm 벌어야 한다. 숙이는 동작 자체는 높이를 안 올리므로
      그 사이가 **낮은 보상의 계곡**이 되고, 정책은 계곡 앞에서 멈춘다.

      높이 비율을 곱하면:
          정체 지점(height_frac 0.26)  2.312 → 0.601   (타먹기 1/4 로)
          기립      (height_frac 1.00)  2.499 → 2.499   (그대로)
          정체/기립 총점 비             27%  →  13%
      하드 게이트("높이 0.35 이상일 때만 지급")보다 이쪽을 쓴다. 하드
      게이트는 상체를 세우는 과정 전체의 셰이핑을 0 으로 만드는데,
      밤사이 유일한 진전(상체 세우기)이 바로 그 셰이핑으로 나왔다.

    ※ walk_ready_pose 의 게이트와 stand_success 의 문턱은 _upright_frac 을
      **그대로** 쓴다. 그건 실제 자세 판정이지 셰이핑이 아니다.
    """
    up = _upright_frac(env, asset_cfg)
    h = _height_frac(env, asset_cfg, supine_height, stand_height)
    return up * h


@hold_masked
def feet_load_fraction(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    supine_height: float,
    stand_height: float,
    gait_command_name: str = "gait",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """발이 받치는 체중 비율 [0,1]. 기상의 본질인 접촉 모드 전환을 본다.

    HumanUP 의 stand_on_feet 은 '양발 접촉 & 발 높이<0.1' 인 이진값인데,
    MX3 는 보행 기본각(무릎 굽힘)으로 누워서 **등과 발이 함께 바닥에
    닿는다** (실측: 최저 바디가 l_ankle_roll_link, z=0.032). 그래서 그
    이진 조건은 시작부터 참이 되어 '움직이지 마라' 신호가 된다.

    ★ 하중 비율만으로도 부족하다 — 높이를 곱한다 ★
      하중 비율 단독으로 쓴 첫 판(iter 1500)에서 정책이 **다리를 앞으로
      쭉 뻗고 앉는 자세**로 수렴했다. 실측:
          base_z 0.1467 (기립 0.396)   무릎 0.019 (완전 신전, 기본 1.2)
          upright 0.9965 (몸통 수직)   feet_load 1.0 (포화)
      다리를 뻗으면 발에 체중이 실려 하중 비율이 최대가 된다. 즉 이 항이
      **일어서지 않고도 만점을 주는 자세를 능동적으로 유도했다.**
      "앉아서 발이 닿은 것" 과 "발로 서 있는 것" 을 구분하지 못한다.

      높이 비율을 곱하면 앉은 자세(0.27)에서는 0.27 배로 깎이고, 실제로
      일어섰을 때만 만점이 된다. 두 조건이 동시에 만족돼야 한다는
      의미가 식에 그대로 들어간다.
    """
    # 계산부는 getup_v1_mdp._feet_load_frac 과 공유한다. Critic 관측
    # (feet_load_obs) 과 값이 어긋나지 않게 하기 위함이다.
    load = _feet_load_frac(env, sensor_cfg, asset_cfg)
    h = _height_frac(env, asset_cfg, supine_height, stand_height)

    # ★ 기립 중 OR 완전정지에서만 (2026-09-03, 제자리걸음 도입) ★
    #   제자리걸음은 한 발을 드는 동작이라 하중 비율이 절반으로 떨어진다.
    #   그대로 두면 "발을 들어라"(feet_gait)와 정면 충돌한다.
    #   보행에는 이 항이 아예 없다 — 누운 상태에서 일어날 일이 없어서다.
    #
    #   ⚠ 완전히 빼면 안 된다. 기상에서 이 항은 "체중을 발로 옮겨라" 를
    #     가르치는 신호다. 없앴을 때(초기 판, iter 1500) 정책이 다리를
    #     앞으로 쭉 뻗고 앉는 자세로 수렴한 기록이 있다.
    #
    #   ⚠ 게이트를 (1 - stand_gate) 만으로 두면 안 된다. 그러면 보상이
    #     height_frac 0.85 에서 최대가 되어 **거기 머무는 유인**이 생긴다
    #     (2026-09-02 웅크림 실패와 같은 구조). 완전정지를 OR 로 묶으면
    #     기립 후(h=1.0)가 항상 더 크므로 그 함정이 없다.
    gate = torch.maximum(
        1.0 - _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg),
        full_stop_mask(env, gait_command_name),
    )
    return load * h * gate


# ═══════════════════════════════════════════════════════════════
#  태스크 — 최종 형태(기본자세로 서 있기)
# ═══════════════════════════════════════════════════════════════
@hold_masked
def self_collision(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """팔 링크가 몸통/골반/다리와 부딪히는 힘의 합 (페널티 — 음수 weight).

    【판정 방식】
      ContactSensor 의 filter_prim_paths_expr 로 **바디쌍 접촉**을 직접
      읽는다. force_matrix_w 형상 (N, B, M, 3) — 센서 바디 B 와 필터 바디 M
      사이의 힘이다. 바닥 접촉은 필터에 없으므로 애초에 섞이지 않는다.

      필터는 "센서 prim 이 환경당 단일 prim 일 때만 동작" 하므로
      (contact_sensor_cfg.py 주석) 팔 링크마다 센서를 따로 둔다.

    【처음에 틀렸던 방식 — 기록】
      "바닥이 평면뿐이니 바닥에서 떨어진 링크의 접촉 = 자기충돌" 로
      판정하려 했으나 **링크 원점 높이 != 접촉 지점 높이** 라 실패했다.
      실측에서 걸린 것이 l_ankle_roll_link 30 N — 그냥 발의 지지력이었다.
      힘 크기로는 발 지지력과 자기충돌을 구분할 수 없다.

    【실측 근거 (model_1900 기립 롤아웃)】
      r_elbow_pitch_link ↔ r_pelvis_roll_link 가 에피소드 내내 접촉:
          step  60  35.34 N     step 180  17.66 N     step 469  5.53 N
      다른 링크쌍은 전부 0.00 N. 오른팔꿈치가 오른골반을 계속 밀고 있다.
      체중이 55.9 N 이므로 35 N 은 상당한 힘이다.

    Returns:
        접촉력 합 [N]. 스케일이 크므로 weight 를 작게 쓴다.
    """
    total = None
    for name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]
        fm = sensor.data.force_matrix_w
        if fm is None:
            continue
        f = torch.norm(fm, dim=-1)                       # (N, B, M)
        f = torch.where(f > force_threshold, f, torch.zeros_like(f))
        v = f.sum(dim=(1, 2))
        total = v if total is None else total + v
    if total is None:
        return torch.zeros(env.num_envs, device=env.device)
    return total


@hold_masked
def walk_ready_pose(
    env: ManagerBasedRLEnv,
    target_pose: dict,
    std: float = 0.5,
    gate_from: float = 0.5,
    gait_command_name: str = "gait",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절이 보행 준비 자세에 가까울수록 보상. **직립 정도로 게이트**한다.

    ★ 게이트가 없으면 이 항은 '움직이지 마라' 가 된다 ★
      mx_bot 의 XM_robot04_getup 은 init_joint_pos 를 재정의하지 않고
      보행 기본각을 상속한다. 즉 시작 관절각이 곧 목표 자세라, 게이트가
      없으면 t=0 에서 이미 최대값이고 일어나려고 휘저을수록 값이 떨어진다.
      (구 getup_env_cfg.py 가 정확히 이 상태였다 — 시작 리워드의 63%)

    gate_from 을 넘어선 뒤부터 선형으로 지급한다. 즉 '움직이지 마라' 가
    아니라 '일어선 다음엔 보행 자세로 마무리해라' 가 된다.

    목표는 default_joint_pos = 보행 기본각이다. 최종 형태가 보행 준비
    자세이므로, 일어선 뒤 보행 정책으로 넘길 때 자세 불연속이 없다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # 목표는 **보행 기본각**이다. 최종 기립 자세는 보행이 정지 상태에서
    #   잡는 그 자세(팔꿈치 ±2.967 접힘)여야, 일어선 뒤 보행 정책으로 넘길 때
    #   자세 불연속이 없다.
    #   ★ target_pose 는 getup 기본각이 뭐든 항상 이 값이다. 2026-08-29 에
    #     getup 기본각을 ±1.5 -> ±2.967 로 되돌렸지만 target 은 그대로다.
    #     즉 **팔 각도에 대한 보상 지형은 그때나 지금이나 동일**하다.
    #     바뀐 것은 같은 각도에 도달하는 데 필요한 액션 크기뿐이다
    #     (팔꿈치 1.0 rad 까지: 이전 -2.0, 지금 -7.9 — 탐색 난이도 문제).
    #   ⚠ 기본각이 곧 target 이 된 지금은, 로그의 walk_ready_pose 값만으로
    #     팔이 움직였는지 알 수 없다(잠겨 있어도 만점). 팔 사용 여부는
    #     scripts/rsl_rl/eval_getup_arm.py 로 직접 재야 한다.
    #   dict 해석은 최초 1회만 하고 캐싱한다.
    key = "_getup_walk_ready_target"
    q_target = getattr(env, key, None)
    if q_target is None or q_target.shape != q.shape[1:]:
        names = [asset.data.joint_names[i] for i in asset_cfg.joint_ids]
        q_target = torch.tensor(
            [float(target_pose.get(n, 0.0)) for n in names],
            dtype=q.dtype, device=q.device,
        )
        setattr(env, key, q_target)

    pose = torch.exp(-torch.mean(torch.square(q - q_target), dim=1) / (std * std))

    up = _upright_frac(env, asset_cfg)
    gate = torch.clamp((up - gate_from) / (1.0 - gate_from), 0.0, 1.0)
    # ★ 완전정지일 때만 (2026-09-03, 제자리걸음 도입) ★
    #   제자리걸음 중에는 다리가 계속 목표 자세에서 벗어난다. 게이팅이
    #   없으면 "위상에 맞춰 발을 들어라"(feet_gait)와 "정지 자세로
    #   가만히"(이 항)가 정면 충돌한다. 보행이 stop_posture_v01 /
    #   stop_default_posture_v01 을 full_stop 으로 게이팅해 푼 방식 그대로다.
    return pose * gate * full_stop_mask(env, gait_command_name)


@hold_masked
def walk_pose_deadline(
    env: ManagerBasedRLEnv,
    target_pose: dict,
    supine_height: float,
    stand_height: float,
    deadline_s: float = 1.5,
    gait_command_name: str = "gait",
    gate: str = "height",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """기한이 지나도 보행 기본자세가 아니면 페널티 (음수 weight 로 쓴다).

    【왜 필요한가 — walk_ready_pose 가 포화한다】
      walk_ready_pose 는 exp(-err/std^2) 형태라 목표에 가까워질수록 기울기가
      죽는다. 실측: 1,300 iteration 동안 0.79 → 0.83 에서 사실상 정체.
      보행의 stop_default_posture_v01 은 같은 목적을 **sum(square(dev))
      페널티 x weight -1.0** 으로 푼다. 포화가 없어 끝까지 밀어준다.

    【시간 게이트를 쓰는 이유】
      기상은 실측상 1.0~1.2 초에 끝난다(base_z 0.099 -> 0.407 @ step 50~60).
      그 뒤로는 "보행 준비 자세로 서 있기" 만 남는다. 기한을 넘겨서도
      자세가 안 잡히면 벌주는 게 의미가 분명하다.

      deadline_s 기본 1.5 초 — 측정된 완료 시점(1.0~1.2)보다 여유를 둔다.
      1.0 초로 잡으면 기상 막바지가 벌점 구간에 걸려 **더 빨리 일어나라는
      압력**이 되는데, 이미 기상 구간 관절속도가 4.57 rad/s 로 액추에이터
      한계(4.5)를 넘고 있어 그 방향은 sim2real 을 악화시킨다.

    ★ 주의: 시간 기반 게이트는 상태 기반과 다르다 ★
      관측에 시간이 없으면 Critic 이 같은 자세를 t=0.5s 와 t=5s 에 같은
      값으로 본다. 그래서 Critic 관측에 에피소드 진행도를 넣는다
      (getup_v1_mdp.episode_phase_obs, sim 전용이라 실기 비용 0).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    key = "_getup_deadline_target"
    q_target = getattr(env, key, None)
    if q_target is None or q_target.shape != q.shape[1:]:
        names = [asset.data.joint_names[i] for i in asset_cfg.joint_ids]
        q_target = torch.tensor(
            [float(target_pose.get(n, 0.0)) for n in names],
            dtype=q.dtype, device=q.device,
        )
        setattr(env, key, q_target)

    dev = torch.sum(torch.square(q - q_target), dim=1)
    # ★ 시간 게이트 -> 높이 게이트 (2026-09-02) ★
    #   구: past = (경과시간 > deadline_s)
    #   시간 기준이라 **넘어뜨림 외란과 충돌**한다. 에피 4 초에 밀어
    #   쓰러뜨리면 5 초 시점엔 바닥에 있는데 "보행 자세가 아니다" 로
    #   벌점을 먹는다. 피할 수 없는 외란 때문에 벌받는 구조다.
    #
    #   부수 효과로 설계도 나아진다. 원 docstring 의 지적 — 시간 게이트는
    #   상태 기반이 아니라서 Critic 이 같은 자세를 t=0.5s 와 t=5s 에 같은
    #   값으로 본다 — 가 사라진다. 높이 게이트는 상태 기반이다.
    #
    #   다른 안정화 항(base_height_well / base_lin_vel_z / base_ang_vel_xy /
    #   base_roll)도 전부 같은 _stand_gate(0.85) 를 쓰므로 기준이 통일된다.
    #   deadline_s 인자는 하위호환으로 남기지만 쓰지 않는다.
    #
    # ★★ 2026-09-14 — 높이 게이트 -> 몸통 방향 게이트 ★★
    #   높이 게이트가 **도피로**였다. 이 항은 -1.0 으로 18 관절 전부를 덮는
    #   가장 큰 자세 항인데, 낮게 서면 게이트가 닫혀 사라진다. 그리고
    #   **다리를 벌리면 엉덩이가 내려간다** — 벌점을 피하는 방법이 곧 벌점
    #   대상을 악화시키는 것이었다.
    #
    #   V3 2 판 실측 (레벨 5 이후 300 iter 평균) — 벌점이 세지는데 자세가
    #   계속 나빠진다. 정상이라면 반대여야 한다:
    #       wdead  +0.0000 -0.0967 -0.1441 -0.2372   <- 램프인으로 커지는데
    #       wpose  +0.7305 +0.6198 +0.5937 +0.5163   <- 자세는 나빠지고
    #       proll  -0.0532 -0.0516 -0.0735 -0.0807   <- 다리는 더 벌어진다
    #       l_pelvis_roll -26 deg -> -53 deg,  높이 0.382 -> 0.369
    #
    #   _upright_frac 은 projected_gravity_b[:,2] 만 쓴다. 다리를 어떻게
    #   벌려도 몸통 방향은 안 바뀌므로 조작할 수 없다.
    #   ※ 일반 규칙: 게이트 입력 g 와 벌점 대상 x 에 대해 dg/dx != 0 이면
    #     그 게이트는 도피로다.
    #   2026-09-03: 여기에 full_stop 을 곱한다. 제자리걸음 중에는 다리가
    #   목표 자세를 벗어나는 것이 정상이므로 벌하면 안 된다.
    # ★ 2026-09-14 — 게이트를 인자로 뺐다 ★
    #   V3 작업 중 높이 게이트를 래치로 바꿨는데, 이 함수는 **v1 도 쓴다**.
    #   v1 계보를 그대로 재현하려면 기본값이 예전 동작(height)이어야 한다.
    #   래치를 쓰려면 cfg 에서 gate="latch" 를 명시한다.
    _g = (_stood_latch(env, supine_height, stand_height, asset_cfg=asset_cfg)
          if gate == "latch"
          else _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg))
    return (dev
            * _g
            * full_stop_mask(env, gait_command_name))


@hold_masked
def stand_success(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    height_frac_threshold: float = 0.85,
    upright_threshold: float = 0.90,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """직립 + 기립 높이를 동시에 만족할 때만 주는 희소 보너스.

    자세(관절각)까지 조건에 넣지 않는다. 조건을 겹겹이 쌓으면 한 번도
    안 터져서 희소 보너스가 죽는다. 자세는 walk_ready_pose 가 연속으로
    안내한다.
    """
    h = _height_frac(env, asset_cfg, supine_height, stand_height)
    up = _upright_frac(env, asset_cfg)
    return ((h > height_frac_threshold) & (up > upright_threshold)).float()


# ═══════════════════════════════════════════════════════════════
#  기립 후 안정화 — "보행에서 cmd=0" 과 같은 상태를 만든다
#  ─────────────────────────────────────────────────────────────
#  【왜 필요한가 — 실측】
#    기립 후(step 120~470) 정책이 계속 흔들린다:
#        수평속도 |v_xy| 0.102 m/s     각속도 |w_xy| 0.381 rad/s (22°/s)
#        수직속도 |v_z|  0.018 m/s     base_z 변동 0.34 mm
#    높이는 완벽한데 수평·회전으로 흔들려 발바닥이 안절부절이 된다.
#    원인은 기상 리워드에 **몸통 운동을 벌주는 항이 하나도 없어서**다.
#    기존 항(높이/자세각/발하중/관절각)은 전부 "흔들려도 만족되는" 양이고,
#    정규화 항(joint_vel, action_rate)은 관절 공간만 본다.
#
#  【가중치 출처】
#    전부 velocity_v0_6(실기 검증된 보행)에서 그대로 가져왔다. 기립 후
#    유지 구간은 본질적으로 "보행에서 cmd=0" 과 같은 상태이기 때문이다.
#        track_lin_vel_xy      +1.0    exp(-||v||^2/0.15)
#        track_ang_vel_z       +1.0
#        base_linear_velocity  -2.0    lin_vel_z_l2
#        base_angular_velocity -0.05   ang_vel_xy_l2
#        base_roll_l2          -5.0
#        foot_flat             -0.3
#    보행은 명령을 추종해야 해서 v_xy 를 못 벌주지만, 기상의 최종 상태는
#    제자리 정지이므로 cmd=0 을 상수로 박아 그대로 쓴다.
#
#  【전부 height_frac 을 곱한다】
#    일어서는 1.2 초 동안은 몸통도 발도 크게 움직여야 한다. 누워 있을 때
#    (height_frac~0) 0, 서 있을 때만 작동하게 한다.
# ═══════════════════════════════════════════════════════════════
_VEL_STD2 = 0.15          # velocity_v0_6 의 std=sqrt(0.15) 와 동일


@hold_masked
def stand_still_lin_vel(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """제자리 유지 보너스 (수평). 보행의 track_lin_vel_xy 를 cmd=0 으로 고정."""
    asset: RigidObject = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b[:, :2]
    r = torch.exp(-torch.sum(torch.square(v), dim=1) / _VEL_STD2)
    return r * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


@hold_masked
def stand_still_ang_vel(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """제자리 유지 보너스 (요 회전). 보행의 track_ang_vel_z 를 cmd=0 으로 고정."""
    asset: RigidObject = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_b[:, 2]
    r = torch.exp(-torch.square(w) / _VEL_STD2)
    return r * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


@hold_masked
def foot_flat_pitch(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """발바닥 앞뒤 기울임(pitch) 페널티 — **기상 전용**.

    보행의 foot_flat_orientation 은 roll 만 보고 pitch 를 일부러 제외한다.
    toe-off / heel-strike 가 정상 보행이고 추진력의 원천이기 때문이다
    (rewards.py:762 주석). **기상에는 그 예외가 필요 없다.** 서 있을 때
    발바닥은 앞뒤로도 평평해야 하고, 관측된 "발꿈치를 들었다 놨다" 가
    바로 이 pitch 성분이다.

    계산은 foot_flat_orientation 과 같은 방식 — 중력을 발 좌표계로 옮겨
    각도를 추정하되, y/z(roll) 대신 x/z(pitch) 를 본다.
    """
    from isaaclab.utils.math import quat_apply_inverse

    asset: RigidObject = env.scene[asset_cfg.name]
    quats = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    down = torch.zeros(env.num_envs, 3, device=env.device)
    down[:, 2] = -1.0

    penalty = torch.zeros(env.num_envs, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        g_f = quat_apply_inverse(quats[:, i, :], down)
        pitch = torch.atan2(-g_f[:, 0], -g_f[:, 2])
        penalty += torch.square(pitch)

    return penalty * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


from isaaclab.envs.mdp import rewards as _lab_rewards  # noqa: E402

from .getup_v1_mdp import full_stop_mask  # noqa: E402
from . import rewards as _mx_rewards  # noqa: E402

# ★ 아래는 제네릭 데코레이터로 감싸면 안 된다 ★
#   functools.wraps 가 __wrapped__ 를 달아 원본 시그니처를 노출하는데,
#   IsaacLab 매니저가 그걸로 파라미터를 검증한다. supine_height/stand_height
#   처럼 **인자를 추가**하는 경우엔 "expects ... but received ..." 로 죽는다.
#   (hold_masked 는 인자를 안 바꿔서 문제가 없다)
#   그래서 시그니처를 명시한 함수로 하나씩 쓴다.


@hold_masked
def base_height_well(
    env: ManagerBasedRLEnv,
    target_height: float,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """기립 후 몸통 높이를 target 에 **이차로 고정**. 보행 base_height_l2(-150) 이식.

    ★ 왜 필요한가 — 다 일어서면 높이 기울기가 정확히 0 이다 ★
      base_height 보상은 _height_frac = clamp((z-0.057)/(0.39-0.057), 0, 1)
      이라 기립하면 1.0 으로 포화한다. 그 위로는 아무 제약이 없어서
      까딱거려도 손해가 문자 그대로 0 이다. 보행은 대신 목표 높이 이차
      우물(-150)이 계속 당긴다.

      실측 비교 (2026-08-29, 256 env, 완전정지/기립 10 초):
                          보행 v0_6    겟업 model_21900
          base_z 표준편차   0.15 mm       5.78 mm   (39 배)
          수직속도 RMS      0.003 m/s     0.056 m/s (19 배)
          각속도 RMS        0.044 rad/s   0.164 rad/s
          발 접촉 전환      0.00 회/s     0.41 회/s
          평균 base_z       0.395 m       0.401 m   (6 mm 높음)

      발 흔들림은 학습이 알아서 해결한다(미학습 5.82 -> 완성 0.41 회/s).
      남는 격차는 **몸통의 수직 까딱임**이고 이 항이 그걸 겨냥한다.
      평균 높이가 6 mm 높은 것도 같이 잡혀, 기상 -> 보행 정책 인계 시
      자세 불연속이 줄어든다.

    ★ 게이트를 거는 이유 ★
      누운 자세는 z 가 0.057 이라 (z-0.39)^2 = 0.11 이다. -150 을 곱하면
      -16.6/초 로 다른 모든 항을 압도해 학습이 시작조차 못 한다.
      _stand_gate 는 h_frac 0.85(= stand_success 문턱, z 0.340)에서 0 으로
      시작해 1.0 에서 1 이 되므로 **불연속 없이** 열린다.

    ※ 10차 실패(FAILED_23-46-50)는 양수 보너스(stand_still_lin/ang)가 중간
      높이에 국소해를 만든 것이었다. 이 항은 페널티라 새 끌개가 없다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    gate = _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)
    return torch.square(z - target_height) * gate


@hold_masked
def base_lin_vel_z(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """수직 진동 페널티. 보행 base_linear_velocity(-2.0) 와 같은 함수."""
    h = _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)
    return _lab_rewards.lin_vel_z_l2(env, asset_cfg) * h


@hold_masked
def base_ang_vel_xy(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """롤·피치 각속도 페널티. 보행 base_angular_velocity(-0.05) 와 같은 함수."""
    h = _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)
    return _lab_rewards.ang_vel_xy_l2(env, asset_cfg) * h


@hold_masked
def base_roll(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """몸통 좌우 기울기 페널티. 보행 base_roll_l2(-5.0) 와 같은 함수."""
    h = _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)
    return _mx_rewards.base_roll_l2(env, asset_cfg) * h


@hold_masked
def foot_flat_roll(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """발바닥 좌우 비틀림 페널티. 보행 foot_flat(-0.3) 과 같은 함수.

    보행 쪽은 command_name=None 으로 정지 포함 항상 적용하도록 이미
    고쳐져 있어(velocity_v0_6:666) 여기서도 그대로 None 을 넘긴다.
    """
    h = _stand_gate(env, supine_height, stand_height)
    return _mx_rewards.foot_flat_orientation(env, asset_cfg, command_name=None) * h


# ═══════════════════════════════════════════════════════════════
#  정규화 — IsaacLab 기본 항에 홀드 마스킹만 씌워 재노출
#  (낙하·정착 중 속도/가속도를 정책 탓으로 벌주지 않기 위함)
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  구간별 부드러움 페널티 — ★ 현재 미사용 (2026-09-02 실패) ★
#  ─────────────────────────────────────────────────────────────
#  ⚠ 이 두 함수를 켜면 로봇이 **5 cm 낮게 웅크리고 선다.**
#    _stand_gate 는 height_frac 0.85(= 0.340 m)에서 열리는데, 게이트가
#    열리면 부드러움이 배수만큼 비싸지므로 정책이 문턱 바로 위
#    (실측 0.342 m, 게이트 0.04)에 머물러 게이트를 닫아둔다.
#    이전 정책들은 0.395~0.401 m 였다.
#
#    ★ 핵심은 순환 구조다 ★
#      이걸 막아야 할 base_height_well(-150, 목표 0.39)이 **같은 게이트로
#      묶여 있다.** 웅크리면 그 항도 같이 꺼진다 — "웅크림을 벌해야 할
#      항이 웅크림 때문에 꺼진다."
#
#    다시 쓰려면 게이트를 게임 불가능하게 바꿔야 한다:
#      "이번 에피소드에서 한 번이라도 height_frac 0.9 를 넘었는가" 래치.
#      한 번 제대로 서면 그 뒤로는 웅크려도 배수가 유지된다.
#    (자세한 기록: logs/.../FAILED_21-13-46_crouch_gate_gaming/RUN_NOTES.txt)
#
#  아래는 그때의 설계 근거다 — 구간을 나눠 거는 접근 자체는 유효했다.
#  실측으로 기립 중 0.934 유지 / 서 있음 0.024 -> 0.017 (-30 %) 였다.
#  ─────────────────────────────────────────────────────────────
#  실기: "기립은 아주 좋다. 근데 다 잘 일어나 놓고 서 있는 상태에서
#        과하게 움직여서 발산시키는 경향이 있다."
#
#  지금까지 action_rate / joint_acc 는 **게이트가 없어** 두 구간에
#  똑같이 걸렸다. 그래서 조이면 기립까지 굼떠지고, 풀면 서 있을 때
#  까불거리는 맞교환에 갇혔다. 실제로 -0.30 -> -0.60 으로 2 배 올렸을 때
#  기립 구간 action_rate_l2 는 1.211 -> 0.934 (-23 %) 로 줄었지만
#  joint_acc_l2 는 22,949 -> 23,142 로 꿈쩍도 안 했다 — 기립에 필요한
#  가속도는 과제가 요구하는 물리적 하한이기 때문이다.
#
#  해법: **기립 중에는 지금 세기 그대로, 기립 후에만 배수를 곱한다.**
#    gate = _stand_gate(0.85)  →  기립 전 0, 기립 후 1 (선형 램프)
#    배수 = 1 + (stand_mult - 1) × gate
#
#  실측 기준 (model_44400):
#      구간      action_rate_l2   joint_acc_l2
#      기립 중       0.934          23,142
#      서 있음       0.024             641
#      보행 정지     0.0026            —      <- 서 있음의 목표
#  서 있을 때가 보행의 9 배다. 여기를 좁히는 것이 목표다.
#
#  ⚠ 맞교환이 있다: 너무 세게 조이면 밀림 회복에 필요한 움직임까지
#    막힌다. 적용 후 eval_getup_recovery.py 로 방향별 회복률이
#    유지되는지 확인할 것.

@hold_masked
def action_rate_l2_phased(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    stand_mult: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """액션 변화율 페널티. 기립 후에는 stand_mult 배로 강해진다."""
    raw = _lab_rewards.action_rate_l2(env)
    gate = _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)
    return raw * (1.0 + (stand_mult - 1.0) * gate)


@hold_masked
def joint_acc_l2_phased(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    stand_mult: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절 가속도 페널티. 기립 후에는 stand_mult 배로 강해진다."""
    raw = _lab_rewards.joint_acc_l2(env, asset_cfg)
    gate = _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)
    return raw * (1.0 + (stand_mult - 1.0) * gate)


joint_vel_l2 = hold_masked(_lab_rewards.joint_vel_l2)
joint_acc_l2 = hold_masked(_lab_rewards.joint_acc_l2)
action_rate_l2 = hold_masked(_lab_rewards.action_rate_l2)
joint_pos_limits = hold_masked(_lab_rewards.joint_pos_limits)
joint_torques_l2 = hold_masked(_lab_rewards.joint_torques_l2)


# ═══════════════════════════════════════════════════════════════
#  자세 항 — 누운 상태에서도 성립하는 판 (2026-09-03 신설)
#  ─────────────────────────────────────────────────────────────
#  【왜 보행 함수를 못 쓰는가 — 실측】
#    보행의 base_roll_l2 / foot_flat_orientation 은 좌우 기울기를
#        roll = atan2(g_y, -g_z)
#    로 복원한다. 두 번째 인자 -g_z 는 "몸통 z 축이 위를 향하는가" 인데,
#    **90 도를 넘으면 부호가 뒤집혀 atan2 가 다른 사분면으로 넘어간다.**
#
#      몸통을 뒤로 a 도 눕혔을 때  g_b = [-sin a, 0, -cos a]
#         a      g_z     -g_z    roll     roll^2 x 5
#        85.0  -0.087   0.087   0.0000      0.000
#        89.5  -0.009   0.009   0.0000      0.000
#        90.5  +0.009  -0.009   3.1416     49.348   <- 여기서 튄다
#       120.0  +0.500  -0.500   3.1416     49.348
#
#    ★ 이때 실제 좌우 기울기 g_y 는 내내 0 이다. 등을 대고 반듯이 누워
#      좌우로 전혀 안 기울었는데 "180 도 옆으로 굴렀다" 로 읽는다.
#      g_y = 0 이면 atan2(0, 음수) = pi 이기 때문이다.
#
#    ★ 하필 MX3 의 supine 안착 자세가 경계 바로 너머다 —
#      projected_gravity = (-1.000, 0.000, +0.027), 즉 91.5 도.
#      부호만 반대였으면 페널티가 0 이었다. 측정된 -49.343 은
#      5 x pi^2 = 49.348 과 소수점 둘째 자리까지 일치한다.
#
#  【왜 치명적인가】
#    경계를 넘는 데 필요한 건 높이가 아니라 **각도**다. 초당 이득 비교:
#        기립 전체 (0.057 -> 0.390 m)
#           base_height          150 x (0.39-0.057)^2 = 16.63
#           base_height_progress 3.0                  =  3.00   합 19.63
#        몸통만 3.4 도 돌리기 (g_z +0.03 -> -0.03)
#           base_roll_l2         5 x pi^2             = 49.35
#    **3.4 도 회전이 기상 전체보다 2.5 배 이득이다.** 그리고 몸통 각도는
#    앉은 채로도 세울 수 있다. Getup-V2 2 판(2026-09-03_04-10-01)의 정책이
#    base_z 0.181 / g_z -0.90 에 고착한 것이 정확히 이 회수 동작이었다
#    (그 자세에서 base_roll_l2 = -0.017).
#
#  【해법 — g_y 를 직접 쓴다】
#    좌우 기울기는 중력이 몸통 y 축으로 새어 나온 양 그 자체다. atan2 로
#    각도를 복원할 필요가 없다. g_y^2 는 앞뒤로 눕는 것과 **완전히
#    독립**이고 불연속이 없다.
#        자세                 g_y    atan2 방식    g_y^2 방식
#        직립                 0.00      0.00         0.00
#        좌우 30 도           0.50      0.27         0.25
#        좌우 90 도(옆누움)   1.00      2.47         1.00
#        등 대고 누움         0.00      9.87  <-X    0.00  <-O
#    직립 근처에서 두 방식이 거의 같으므로 **보행 가중치를 그대로 써도
#    직립 구간 동작이 안 바뀐다** (30 도에서 0.27 vs 0.25).
#
#  ※ 보행/킥이 공유하는 mdp/rewards.py 는 건드리지 않는다. 그쪽 정책의
#    재현성을 깨지 않기 위해 기상 전용 함수로 분리했다.
# ═══════════════════════════════════════════════════════════════
def base_roll_gy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """몸통 좌우 기울기 페널티. 누운 상태에서도 성립한다.

    보행 base_roll_l2(-5.0) 의 대체. 위 섹션 주석 참조.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def foot_flat_gy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """발바닥 좌우 비틀림 페널티. 누운 상태에서도 성립한다.

    보행 foot_flat_orientation(-0.3) 의 대체. 발 2 개의 값을 더한다
    (원본과 같은 규약 — 그래서 가중치를 그대로 쓸 수 있다).

    pitch 를 안 보는 것도 원본과 같다: toe-off / heel-strike 의 발목
    pitch 회전은 정상 보행이고 추진력의 원천이다.
    """
    from isaaclab.utils.math import quat_apply_inverse

    asset: RigidObject = env.scene[asset_cfg.name]
    quats = asset.data.body_quat_w[:, asset_cfg.body_ids, :]

    down = torch.zeros(env.num_envs, 3, device=env.device)
    down[:, 2] = -1.0

    penalty = torch.zeros(env.num_envs, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        g_f = quat_apply_inverse(quats[:, i, :], down)
        penalty += torch.square(g_f[:, 1])
    return penalty


# ═══════════════════════════════════════════════════════════════
#  속도 추종 — 기립 후에만 채점 (2026-09-04 신설, Getup-V2 용)
#  ─────────────────────────────────────────────────────────────
#  【왜 필요한가】
#    Getup-V2 5 판까지는 rel_standing_envs=1.0 으로 전 env 속도 0 이었다.
#    그러면 track_lin_vel_xy / track_ang_vel_z 가 **가만히 있을 때 만점**이라
#    발을 뗄 이유가 사라진다. 실측(model_9600, 90 deg 기립 후):
#        발 접지율  오른 73.8 %  왼 99.3 %   <- 왼발은 아예 안 뗀다
#        gait 보상  0.628/초  (양발 붙이고 있으면 자동 0.550, 완벽 제자리걸음 1.000)
#    제자리걸음으로 얻을 최대 이득이 0.45/초인데, 발을 들면 upright_progress
#    (2.5) 와 stand_success(5.0) 를 흔들 위험이 있어 순손실이다.
#
#    v0_6(보행)은 env 의 85 % 가 실제 속도 명령을 받아 **발을 뗄 수밖에 없고**,
#    그렇게 배운 걸음을 정지 명령일 때 제자리에서 쓴다. gait 항은 위상 정렬
#    신호일 뿐 걸음을 만드는 힘이 아니었다.
#
#  【그런데 그대로 되살리면 안 된다】
#    누워 있는 동안에도 속도 추종이 채점되면 **바닥을 미끄러져 점수를 받는**
#    해가 열린다. 지금까지 다섯 판이 전부 "안 서고 점수 받는 법"을 찾아냈다.
#    -> _stand_gate 로 기립 전에는 끈다. v1 의 stand_still_lin_vel 과 같은 방식.
# ═══════════════════════════════════════════════════════════════
@hold_masked
def track_lin_vel_xy_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """보행 track_lin_vel_xy(+1.0) 를 기립 후에만 채점한다."""
    r = _mx_rewards.track_lin_vel_xy_hold_aware(
        env, command_name=command_name, std=std, asset_cfg=asset_cfg)
    return r * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


@hold_masked
def track_ang_vel_z_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """보행 track_ang_vel_z(+1.0) 를 기립 후에만 채점한다."""
    r = _mx_rewards.track_ang_vel_z_hold_aware(
        env, command_name=command_name, std=std, asset_cfg=asset_cfg)
    return r * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


# ═══════════════════════════════════════════════════════════════
#  팔 자세 고정 — 기립 후에만, 보행 중에도 (2026-09-04 신설)
#  ─────────────────────────────────────────────────────────────
#  【왜 필요한가 — 실측 (7 판 model_23500, 90 deg 기립 후 마지막 4 초)】
#    관절                 기본각   완전정지   제자리걸음
#    r_shoulder_pitch      0.00      0.02      -1.42   <- 앞으로 81 도
#    r_shoulder_roll       0.00      0.00      +0.59   <- 옆으로 34 도 벌림
#    r_elbow_pitch         1.50      1.52       0.35   <- 팔꿈치 폄
#    l_shoulder_pitch      0.00     -0.00      -1.58
#    l_shoulder_roll       0.00     -0.04      -0.33
#    l_elbow_pitch        -1.50     -1.49      -0.12
#
#    완전정지에서는 walk_ready_pose 가 팔을 완벽히 잡는다(편차 0.00~0.04).
#    그런데 그 항은 full_stop_mask 로 게이팅돼 **제자리걸음/보행 중에는
#    꺼진다.** 다리는 위상에 맞춰 발을 들어야 하므로 그게 맞지만,
#    **팔까지 같이 풀리는 건 의도가 아니었다.**
#
#  【게이트 선택】
#    _stand_gate  O  — 기상 중에는 팔로 바닥을 짚어야 한다. 항상 켜두면
#                      기상이 깨진다 (v1 에서 팔 기본각 하나로 학습 성패가
#                      갈렸다). 높이 85 % 이상에서만 건다.
#    full_stop_mask X — 걷든 제자리걸음이든 팔은 오므리고 있어야 한다.
#                      이게 walk_ready_pose 와의 차이점이다.
# ═══════════════════════════════════════════════════════════════
@hold_masked
def _stood_latch(env, supine_height: float, stand_height: float,
                 frac_threshold: float = 0.85,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """에피소드 중 **한 번이라도** 기립 높이에 닿았으면 1, 아니면 0. 래치.

    ★★ 왜 래치인가 — 게이트 두 종류가 각각 다르게 실패했다 ★★

      ① 높이 게이트 (_stand_gate) : **도피 가능**
         낮게 서면 게이트가 닫혀 벌점이 사라진다. 그리고 다리를 벌리면
         엉덩이가 내려간다 — 벌점을 피하는 방법이 곧 벌점 대상을 악화시킨다.
         V3 2 판 실측: 벌점 램프인 중 proll -0.053 -> -0.081 로 **더 나빠짐**,
         l_pelvis_roll -26 deg -> -53 deg, 높이 0.382 -> 0.369,
         stand 3.94 -> 2.87.

      ② 몸통 방향 게이트 (_upright_frac) : **너무 일찍 열린다**
         몸통 방향만 보므로 아직 웅크린 상태에서도 켜진다. 실측에서
         t=1.20 s 에 g_z -0.72 -> upright_frac 0.86 -> 게이트 44 % 인데
         그때 base_z 는 0.269 로 **기상 완료 1 초 전**이다.
         walk_pose_deadline(-1.0, 18 관절)이 기상 중에 "보행 자세를 잡아라"
         라고 요구해 동작이 무너진다.
         V3 3 판 실측: 램프인 중 ovs -0.012 -> -0.658 (53 배),
         proll -0.043 -> -0.121, stand 4.48 -> 3.45, reward 110 -> 24.5.

      ③ 래치 : 둘 다 피한다
         - 기상 중에는 안 켜진다 (실제 기립 높이에 닿아야 하므로)
         - 도피가 안 된다 (한 번 켜지면 낮게 서도 안 꺼짐)
         - 래치를 피하려고 아예 안 일어나면 stand_success(+5.0)를 통째로 잃는다

      ※ 이 프로젝트가 이미 쓰는 패턴이다 (킥의 ins_latched,
        arm_lift_power 의 적립액 예산).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    frac = ((z - supine_height) / max(stand_height - supine_height, 1e-6)).clamp(0.0, 1.0)

    if not hasattr(env, "_getup_stood_latch"):
        env._getup_stood_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # 에피소드 리셋 감지 — kick_inside_mdp / arm_lift_power 와 같은 방식
    just_reset = env.episode_length_buf <= 1
    if bool(just_reset.any()):
        env._getup_stood_latch = torch.where(
            just_reset, torch.zeros_like(env._getup_stood_latch), env._getup_stood_latch)
    env._getup_stood_latch = env._getup_stood_latch | (frac >= frac_threshold)
    return env._getup_stood_latch.float()


def joint_deviation_standing(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    gate_from: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """기립 후에만 적용되는 관절 편차 L1. 어느 관절 묶음에든 쓸 수 있다.

    ★ 왜 L1 이 따로 필요한가 (walk_ready_pose 가 있는데도) ★
      walk_ready_pose 는 exp(-sum(dev^2)/std^2) 지수 커널이다. V3 1 판
      실측에서 기립 후 자세가 이렇게 나왔다:
          l_pelvis_roll  -0.4523 rad (-26 deg)   목표 0
          r_pelvis_roll  +0.3565 rad (+20 deg)
          r_pelvis_yaw   -0.2725 rad (-16 deg)
          l_pelvis_yaw   +0.2544 rad
          r/l_ankle_roll +-0.10 rad
      편차 제곱합이 약 0.63, std 0.5 이므로 walk_ready_pose 가 1.0 중
      **0.08 만** 나온다. 즉 정책은 이 항의 92 % 를 이미 포기했고,
      **지수 함수의 평평한 꼬리에 있어 되돌아올 기울기가 거의 없다.**

      넓게 벌리면 지지면이 커져 균형에 유리하니 stand_success(+5.0) 와
      upright(+2.5) 로 상쇄하고도 남는다. 피치 관절은 편차가
      0.006~0.039 로 거의 완벽한데 롤·요만 벌어진 것이 그 증거다.

      L1 은 편차가 커도 기울기가 일정해서 꼬리 문제가 없다.
      (보행 v0_6 에는 joint_deviation_hip_yaw / _pelvis_roll 이 -0.5 로
       들어 있었다. v1 계보에는 없어서 V3 가 물려받지 못했다.)

    ※ 몸통 높이도 같이 고쳐진다 — 다리를 벌리면 같은 무릎 각도에서
      엉덩이가 내려간다. V3 1 판 실측 0.3820 m (목표 0.390).

    ★★ 2026-09-14 — 게이트를 높이 기준에서 **몸통 방향** 기준으로 바꿨다 ★★
      처음에 _stand_gate(몸통 **높이** 기준, height_frac 0.85 부터)를 썼다가
      **정책이 게이트를 닫아버렸다.**
          height_frac = (z - 0.057)/0.333,  0.85 부터 열림
          z = 0.3693 -> frac 0.938 -> 게이트 0.587 (59 % 만 열림)
          z < 0.340  -> frac < 0.85 -> 게이트 0   ← 벌점이 사라진다
      그리고 **다리를 벌리면 엉덩이가 내려간다.** 즉 다리 벌림과 낮게 서기가
      같은 도피로였다. V3 2 판 실측:
          l_pelvis_roll  -0.4523 (-26 deg) -> -0.9218 (-53 deg)  두 배 악화
          몸통 높이       0.3820 -> 0.3693
          벌점 램프인 중 proll -0.053 -> -0.080 으로 **더 나빠짐**
          stand 3.94 -> 2.87,  EMA 0.999 -> 0.969

      _upright_frac 은 projected_gravity_b[:,2] 만 쓴다. **다리를 어떻게
      벌려도 몸통 방향은 안 바뀌므로** 게이트를 조작할 수 없다.
      walk_ready_pose 가 같은 이유로 이미 이 방식(gate_from=0.75)을 쓴다.

      ※ 일반 규칙: 게이트 입력 g 와 벌점 대상 x 에 대해 d g / d x != 0 이면
        그 게이트는 도피로다. x 와 독립인 양을 골라야 한다.

    Args:
        gate_from: 이 upright_frac 이상에서 선형으로 열린다.
            prone 0 / supine 0.5 / 직립 1 스케일이므로 0.75 는 몸통이
            수평에서 60 deg 이상 일어선 상태다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - \
        asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    pen = torch.sum(torch.abs(dev), dim=1)

    return pen * _stood_latch(env, supine_height, stand_height, asset_cfg=asset_cfg)


def joint_deviation_arm_standing(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """팔 관절의 기본각 이탈 (L1). 기립 후에만 채점한다 — 페널티(음수 weight)."""
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - \
        asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    pen = torch.sum(torch.abs(dev), dim=1)
    return pen * _stand_gate(env, supine_height, stand_height)


# ═══════════════════════════════════════════════════════════════
#  토크 사용률 페널티 (2026-09-11 신설)
#  ─────────────────────────────────────────────────────────────
#  【왜】 실기 피드백: "과하게 밀어서 반대로 넘어진다",
#    "천천히 일어나도 되니 안정적으로 일어났으면".
#
#  실측 (model_34400, 90 도, 256 env, 기상 구간 0.4~1.2 초):
#      t[s]  base_z  sum(r^2)  로드율평균  포화env%
#      0.40   0.070    13.474    0.665     100.0%
#      0.80   0.215    13.194    0.684     100.0%
#      1.20   0.370     7.909    0.454     100.0%
#      1.60   0.407     2.395    0.209      54.3%   <- 기립 완료
#    ★ 기상 내내 **전 env 100 % 가 토크 포화**다. 로드율 최대 1.249 로
#      상한을 24 % 초과 요청한다(PhysX 가 클램프). 그리고 18 관절이
#      **전부** 최대 로드율 1.116 — 특정 관절이 아니라 온몸을 동시에
#      최대로 밀어붙인다. 그래서 1 초 만에 튀듯 일어나고, 그 각운동량이
#      실기에서 반대로 넘어가는 원인이다.
#    기립 유지 구간은 sum(r^2) 2.0~2.4 로 기상 구간의 1/6 이다.
#
#  【정규화를 env 별 상한으로 하는 이유】
#    randomize_effort_limit 으로 effort_limit 이 env 마다 4.0~5.0 이다.
#    스칼라(5.0)로 나누면 약한 모터 env 가 자기 상한을 다 써도 0.64 만
#    나와 압력이 덜 걸린다. 실제 상한으로 나눠야 "네가 가진 걸 다 쓰지
#    마라" 가 되고, 모터 개체차와 무관하게 같은 의미가 된다.
# ═══════════════════════════════════════════════════════════════
@hold_masked
def joint_load_ratio_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """정격 대비 토크 사용률의 제곱합 — 페널티(음의 weight).

    각 env 의 **실제** joint_effort_limits 로 정규화한다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jids = asset_cfg.joint_ids
    if jids is None:
        jids = slice(None)
    tau = asset.data.applied_torque[:, jids]
    lim = asset.data.joint_effort_limits[:, jids].clamp(min=1e-6)
    return torch.sum(torch.square(tau / lim), dim=1)


@hold_masked
def joint_load_excess_l2(
    env: ManagerBasedRLEnv,
    threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """정격 대비 토크 사용률이 threshold 를 **넘는 만큼만** 제곱해서 합산.

    【왜 joint_load_ratio_l2(전체 제곱합)로는 안 됐는가 — 14 판 실측】
      weight -0.05 로 넣었더니 에피소드 평균 sum(r^2) 이 13.4 -> 8.4 로
      줄어 효과가 있어 보였다. 그런데 시간축으로 보니 **감소분이 전부
      기립 후 구간**이었다:
          t[s]   13 판   14 판
          0.40  13.474  13.476   <- 기상 피크, 그대로
          0.80  13.194  14.112   <- 오히려 증가
          1.20   7.909   5.927
          2.40   2.241   1.309   <- -42 %, 여기서만 줄였다
      base_z 궤적도 그대로여서 **기상 시간이 안 늘었다**.

      이유: 전체 제곱합은 낮은 토크도 똑같이 벌준다. 정책 입장에서는
      이미 서 있어 토크가 덜 필요한 구간에서 줄이는 게 싸다. 기상 피크를
      줄이면 stand_success(5.0/초)를 늦게 받기 시작해 손해라 안 건드린다.

    【문턱 방식】 포화 구간에만 압력이 걸린다. 실측 대비(th=0.8):
          기상피크(0.3~1.2s) 0.814   vs   기립 후 0.045   = 18 배
      (전체 제곱합은 13.5 vs 2.0 = 7 배였다)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jids = asset_cfg.joint_ids
    if jids is None:
        jids = slice(None)
    tau = asset.data.applied_torque[:, jids]
    lim = asset.data.joint_effort_limits[:, jids].clamp(min=1e-6)
    excess = ((tau / lim).abs() - threshold).clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=1)


# ═══════════════════════════════════════════════════════════════
#  좌우 팔 부하 균형 (2026-09-12 신설)
#  ─────────────────────────────────────────────────────────────
#  【왜 — 실기 피드백】
#    "이건 그냥 한쪽팔로 땅을 밀어서 그 반동으로 일어나는방식이라
#     조금만 조건이 틀어져도 못일어날거같아"
#
#  실측 (model_58600, 90 도, 128 env, 관절 |토크| 합):
#      t[s]  base_z |   R팔    L팔  팔비대칭 |  R다리  L다리  다리비대칭
#      0.40   0.070 |  9.80   1.77   0.695  | 21.30  22.52    0.070
#      0.80   0.204 | 14.70   5.80   0.458  | 14.32  19.34    0.201
#      1.20   0.394 |  8.02   0.94   0.784  | 12.18   6.97    0.369
#    접촉력도 오른팔꿈치에만 15.45 N, 왼쪽 0 이었다.
#    ★ **다리는 이미 대칭(0.07~0.20)이고 팔만 비대칭(0.46~0.78)** 이다.
#      그래서 팔만 겨냥한다. 다리까지 묶으면 잘 되는 쪽에 불필요한 제약이다.
#
#  【미러 손실로는 왜 안 되는가】
#    미러 손실은 pi(mirror(s)) = mirror(pi(s)) 를 강제한다. "대칭 상태에는
#    대칭 행동" 이지 **실행된 동작이 좌우 대칭이어야 한다** 가 아니다.
#    supine 은 좌우 대칭이라 불안정 평형이고, 작은 비대칭이 한쪽으로
#    무너뜨리면 그 뒤로는 상태 자체가 비대칭이라 비대칭 행동이 정당하다.
#    Loss/symmetry 가 0.0065 로 낮은데도 한쪽 팔만 쓰는 게 모순이 아닌 이유다.
#
#  【기대 효과】
#    (1) 반동 상쇄 — 한 팔로 밀면 반작용이 몸통을 회전시켜 반대로 넘어가려
#        한다. 양팔이면 좌우 성분이 상쇄되고 위로만 작용한다.
#    (2) 자연히 느려짐 — 두 팔이 일을 나눠 각 팔의 순간 힘이 절반이 된다.
#    (3) 덜 취약 — 팔꿈치 접촉 하나에 의존하지 않는다.
#
#  ⚠ 강제가 아니라 **선호** 여야 한다. 스폰에 side_roll +-0.15 가 있어
#    한쪽 어깨로 기울어 눕는 경우가 있고, 그때는 비대칭이 정당하다.
#  ※ 기립 후에는 이미 대칭(0.20~0.25)이라 압력이 거의 안 걸린다. 게이팅 불필요.
# ═══════════════════════════════════════════════════════════════
def arm_load_balance_l2(
    env: ManagerBasedRLEnv,
    right_cfg: SceneEntityCfg,
    left_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """좌우 팔 토크 총량 차이의 제곱 — 페널티(음의 weight)."""
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque.abs()
    r = tau[:, right_cfg.joint_ids].sum(dim=1)
    l = tau[:, left_cfg.joint_ids].sum(dim=1)
    return torch.square(r - l)


# ─────────────────────────────────────────────────────────────────────────
# ★ 2026-09-12 — 양팔 지지 보상 (접촉력 기준) ★
#
#   왜 토크가 아니라 접촉력인가:
#     19 판에서 arm_load_balance_l2(= 좌우 팔 **관절 토크** 차이의 제곱)를
#     -0.05 로 올렸더니 토크 비대칭이 0.46~0.78 -> 0.05~0.18 로 떨어졌다.
#     그런데 접촉력으로 다시 재보니 앙와위 90 도에서
#         오른팔 충격량 4.349 N.s / 접촉시간 0.172 s / 피크 83.4 N
#         왼팔   충격량 0.000 N.s / 접촉시간 0.000 s / 피크  0.0 N
#         오른팔 분담률 1.000, 왼팔이 1 N 이상 닿은 env 0.0 %
#     **왼팔은 단 한 순간도 땅에 닿지 않았다.**
#
#     applied_torque 는 팔을 허공에서 휘둘러도 올라간다. 정책은 왼팔을
#     더 힘주어 움직이는 것으로 토크 균형 요구를 만족시키고 지지 구조는
#     그대로 뒀다. 대리 지표를 잘못 골라서, 스펙은 맞추고 의도는 비켜간
#     전형적인 경우다.
#
#   왜 min(F_R, F_L) 인가 (차이의 제곱이 아니라):
#     차이 제곱 penalty 는 **둘 다 0 으로 만들어도** 만족된다. 팔을 아예
#     안 쓰는 쪽으로 도망갈 길이 남는다. min 은 약한 쪽을 직접 보상하므로
#     양팔이 **둘 다** 실제로 땅을 밀어야 값이 생긴다. 허공에 휘두르면 0 이다.
#
#   왜 예산(budget)인가:
#     양수 보상이라 무한정 주면 "누워서 양팔로 바닥 누르기" 로 농사를
#     지을 수 있다. 예산 0.5 s x 가중치 15 = 에피소드당 최대 7.5 로 묶으면
#     서 있기 1 초치 보상(10.5)보다 작아서 농사가 성립하지 않는다.
#
#   왜 hold 를 제외하는가:
#     공중 스폰(0.14~0.18 m) 후 정착할 때 양 팔이 동시에 바닥을 때린다.
#     복와위 실측에서 t=0.10~0.12 s 에 양팔 55~72 N 이 찍혔는데 이건
#     기상 동작이 아니라 낙하 충격이다. hold(0.3 s) 동안은 세지 않는다.
#     실제 기상 밀기는 앙와위 0.66~0.90 s, 복와위 0.30~0.44 s 로 hold 밖이다.
# ─────────────────────────────────────────────────────────────────────────
def arm_support_both(
    env,
    right_cfg: SceneEntityCfg,
    left_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    force_ref: float = 20.0,
    budget_s: float = 0.5,
    min_rise_speed: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """양팔이 **동시에** 바닥을 지지할 때만 나오는 보상 [0, 1].

    Args:
        right_cfg / left_cfg: 접촉센서(contact_forces) 의 팔 링크.
            ⚠ 접촉센서의 body 순서는 articulation 과 다르다. 반드시
              SceneEntityCfg("contact_forces", ...) 의 body_ids 로만 색인한다.
        force_threshold: 이 힘[N] 을 넘어야 "닿았다" 로 본다. 잡음 제외.
        force_ref:  약한 쪽 팔이 이 힘[N] 을 내면 품질 1.0. 복와위 실측에서
                    양팔 지속 지지가 20~24 N 이었으므로 20 으로 잡았다.
        budget_s:   에피소드당 보상을 받을 수 있는 총 시간[s].
        min_rise_speed: base 상승 속도[m/s] 문턱. 이것을 넘을 때만 센다.

    ★ min_rise_speed 가 왜 필요한가 (스모크 테스트에서 잡은 구멍) ★
      액션을 전부 0 으로 주고(= 그냥 누워 있게) 돌렸더니 **예산 0.5 s 가
      전부 소모됐다.** 누워 있으면 두 팔이 그냥 바닥에 닿아 있고, 약한 쪽
      팔 힘이 9 N 이나 나온다(팔 자체 무게). 즉 "양팔 접촉" 만으로는
      기상 밀기와 가만히 누워 있기를 구별할 수 없다.

      밀기 구간에서는 base 가 실제로 올라간다 (실측 v_z):
          앙와위 밀기      0.50 ~ 0.73 m/s
          복와위 양팔 지지 0.13 ~ 0.22 m/s
      누워 있으면 0 이다.

      ★ 문턱은 0.10 m/s 다 (0.05 가 아니라) ★
        0.05 로 잡았더니 액션 0 스모크에서 **정착 중 튐**이 걸렸다.
        256 env x 6 라운드 계측:
            누수 빈도   env 의 0.3 %
            접촉력      양팔 26~27 N  <- 진짜 지지력이라 힘으로는 구별 불가
            v_z         0.06 ~ 0.07 m/s   <- 여기가 구별점
        평균 영향은 0.0008 s 로 무시 가능하지만, 걸린 env 는 예산 0.48 s 를
        정착에 날려 정작 밀기에 쓸 게 없어진다. 0.10 이면 누수(0.07)는
        끊고 복와위 최저값(0.13)은 30 % 여유로 통과한다.

    ★ force_threshold 는 1 N 이다 (10 N 이 아니라) ★
      처음에 5 -> 10 N 으로 올렸다가 1 N 으로 내렸다. 이유:
      **문턱을 높게 잡으면 앙와위에서 보상이 정확히 0 이 되어 기울기가
      사라진다.** 앙와위 실측에서 왼팔 접촉력은 0.00 N 이고 밀기 중
      왼팔은 공중에 있다. 문턱 10 N 이면 왼팔이 우연히 10 N 넘게 닿을
      때까지 PPO 가 개선 방향을 전혀 못 본다 (희소 보상 문제).

      1 N 이면 왼팔이 스치기만 해도 quality = min/20 = 0.05 가 나와
      **연속적인 기울기**가 생긴다. 농사 방지는 문턱이 아니라
      min_rise_speed 가 한다 — 액션 0 스모크 테스트로 확인했다.
      문턱의 역할은 접촉 잡음 제외뿐이고, 크기 판단은 quality 가 한다.
    """
    dev = env.device
    if not hasattr(env, "_arm_sup_used"):
        env._arm_sup_used = torch.zeros(env.num_envs, device=dev)

    # 에피소드 리셋 감지 — kick_inside_mdp 와 같은 방식.
    # IsaacLab step() 은 episode_length_buf 를 먼저 올리고 보상을 계산하므로
    # 리셋 직후 첫 보상 계산 시점의 값은 1 이다.
    just_reset = env.episode_length_buf <= 1
    if bool(just_reset.any()):
        env._arm_sup_used = torch.where(
            just_reset, torch.zeros_like(env._arm_sup_used), env._arm_sup_used
        )

    sensor = env.scene.sensors[right_cfg.name]
    f = sensor.data.net_forces_w.norm(dim=-1)          # [N, bodies]
    f_r = f[:, right_cfg.body_ids].sum(dim=1)
    f_l = f[:, left_cfg.body_ids].sum(dim=1)

    weak = torch.minimum(f_r, f_l)
    quality = (weak / force_ref).clamp(max=1.0)

    touching = (f_r > force_threshold) & (f_l > force_threshold)

    # 몸이 실제로 올라가는 중일 때만 센다 (누워서 팔 얹고 있기 배제)
    asset: Articulation = env.scene[asset_cfg.name]
    touching = touching & (asset.data.root_lin_vel_w[:, 2] > min_rise_speed)

    # reset hold 동안(공중 스폰 정착)은 세지 않는다
    if hasattr(env, "reset_hold_steps"):
        touching = touching & (env.reset_hold_steps <= 0)

    remain = (budget_s - env._arm_sup_used).clamp(min=0.0)
    active = touching & (remain > 0.0)

    env._arm_sup_used = env._arm_sup_used + torch.where(
        active, torch.full_like(env._arm_sup_used, env.step_dt),
        torch.zeros_like(env._arm_sup_used),
    )

    return torch.where(active, quality, torch.zeros_like(quality))


# ─────────────────────────────────────────────────────────────────────────
# ★ 2026-09-12 — 팔꿈치 높이 대칭 (앙와위 기울기 확보용) ★
#
#   왜 필요한가:
#     arm_support_both(접촉력 기준)은 앙와위에서 **정확히 0** 이다. 왼팔
#     접촉력이 0.00 N 이라 문턱을 1 N 으로 내려도 값이 안 생긴다.
#     보상이 0 이면 PPO 가 개선 방향을 못 본다.
#
#     팔꿈치 높이 실측 (앙와위 90 도, model_61300):
#         t      R팔꿈치z  L팔꿈치z   차이 | R어깨피치  L어깨피치
#       0.40      0.112    0.111   -0.001 |   +0.248    -0.245
#       0.50      0.158    0.165   +0.007 |   +0.592    -0.597
#       0.60      0.137    0.182   +0.045 |   +0.894    -0.508
#       0.70      0.057    0.243   +0.186 |   +1.243    -0.512
#       0.80      0.066    0.348   +0.282 |   +1.530    -0.677
#
#     **0.5 초까지는 완벽히 대칭이다** (차이 7 mm). 0.55~0.60 s 부터
#     오른 어깨만 +1.53 까지 계속 가서 팔꿈치를 바닥(0.057)에 붙이고,
#     왼 어깨는 -0.51 에서 멈춰 팔꿈치가 0.24 m 에 남는다. 간격 19~28 cm.
#     탐색 잡음으로 닿을 거리가 아니다.
#
#   왜 위치인가 (접촉력·토크가 아니라):
#     위치는 **언제나 정의**되어 있어 접촉이 없어도 기울기가 있다.
#     그리고 허공에서 팔을 휘둘러 우회할 수 없다 — 높이를 같게 만드는
#     유일한 방법이 놀고 있는 팔을 내리는 것이다.
#     (19 판에서 토크가 우회당한 것과 대비된다. 토크는 공중에서도 오른다.)
#
#   회피로 점검:
#     미는 팔을 들어올려 차이를 줄일 수는 있다. 하지만 그러면 기상을
#     못 하고 stand_success(5.0/s) + height(3.0) + upright(2.5) 를 다 잃는다.
#     이 항의 크기(초당 최대 0.4 수준)로는 이득이 안 된다.
#
#   L1 을 쓴다 (제곱이 아니라):
#     제곱은 0 근처에서 기울기가 사라져 마지막 몇 cm 를 못 좁힌다.
#     지금 목표는 19 cm 를 0 으로 끌고 가는 것이라 일정한 당김이 낫다.
# ─────────────────────────────────────────────────────────────────────────
def elbow_height_symmetry_l1(
    env,
    right_cfg: SceneEntityCfg,
    left_cfg: SceneEntityCfg,
    min_rise_speed: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """밀기 구간에서 좌우 팔꿈치 높이 차이 [m].

    Args:
        right_cfg / left_cfg: articulation 의 팔꿈치 링크 (body_ids 하나씩).
            ⚠ 접촉센서가 아니라 **robot(articulation)** 의 body_ids 다.
              두 순서는 다르므로 섞어 쓰면 조용히 틀린 값이 나온다.
        min_rise_speed: base 상승 속도[m/s] 문턱. arm_support_both 와 동일 기준.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z_r = asset.data.body_pos_w[:, right_cfg.body_ids, 2].squeeze(-1)
    z_l = asset.data.body_pos_w[:, left_cfg.body_ids, 2].squeeze(-1)
    diff = (z_r - z_l).abs()
    rising = asset.data.root_lin_vel_w[:, 2] > min_rise_speed
    return torch.where(rising, diff, torch.zeros_like(diff))


# ─────────────────────────────────────────────────────────────────────────
# ★ 2026-09-12 (2차) — 팔이 '들어올린 일' 로 재기 ★
#
#   arm_support_both(접촉력 norm + 상승속도 문턱)의 실패:
#     복와위 전 구간 실측 (model_71400) — 수직 성분
#         t           base_z          R / L [N]     무슨 일
#       0.08~0.12   0.166 -> 0.138 하강   55 / 58    낙하 충격
#       0.14~0.38   0.132 -> 0.096 하강   22 / 23    팔에 눌림
#       0.40~0.66   0.101 -> 0.151 상승   20 / 17    상체만 4 cm 들기
#       0.80~1.00   0.146 -> 0.243 상승   55 / 0.00  ★ 진짜 밀어올리기
#       1.10~       0.305 -> 0.39         0 /    0   다리로 기립
#
#     **양팔이 같이 눌린 건 넘어질 때와 가라앉는 동안이고, 실제로 몸을
#       밀어올리는 구간은 오른팔 100 % 다 (L 이 정확히 0.00).**
#
#     그런데 상승속도 문턱 0.10 m/s 는 예비 구간(v_z 0.22)을 통과시킨다.
#     정책은 **예산 0.5 s 를 값싼 예비 구간에서 다 쓰고 진짜 밀기는
#     한 팔로** 했다. 막았다고 생각한 예산 도둑질이 형태를 바꿔 돌아왔다.
#
#     앞선 "약한 팔이 체중 20 % 이상 받친 시간 0.459 s" 라는 보고는
#     **몸이 내려가는 구간을 센 것**이었다. norm 을 써서 수평 마찰력까지
#     포함한 것도 같은 오판에 기여했다.
#
#   처방 — 문턱을 올리지 않는다. 경계에서 또 새는 길이 생긴다.
#     힘 x 속도 = 일률(power)로 재면 값싼 구간이 자동으로 작아진다:
#         낙하/가라앉음  v_z 음수        -> 0
#         예비 들기      20 N x 0.22     -> 0.16
#         복와위 진짜밀기 0 N x 0.68      -> 0     <- 여기를 키우는 게 최대 이득
#         앙와위 밀기    17 N x 0.73     -> 0.31
#     속도가 가장 빠른 구간을 양팔로 바꾸는 것이 가장 크게 이득이 된다.
#
#   예산도 **시간이 아니라 적립액**으로 센다. 시간 예산은 값싼 구간이
#   그대로 먹어치운다. 적립액 예산이면 싼 구간은 예산을 조금만 쓴다.
# ─────────────────────────────────────────────────────────────────────────
def _body_weight(env, asset_cfg: SceneEntityCfg) -> float:
    """로봇 전체 체중 [N]. 한 번 계산해 캐싱."""
    if not hasattr(env, "_getup_body_weight"):
        asset: Articulation = env.scene[asset_cfg.name]
        m = float(asset.data.default_mass[0].sum())
        env._getup_body_weight = m * 9.81
    return env._getup_body_weight


def arm_lift_power(
    env,
    right_cfg: SceneEntityCfg,
    left_cfg: SceneEntityCfg,
    vel_ref: float = 0.5,
    budget: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """양팔이 **함께 몸을 들어올린 일률** [0, 1].

        min(Fz_R, Fz_L) x clamp(v_z, 0, vel_ref) / (체중 x vel_ref)

    약한 쪽 팔을 쓰므로 한 팔만 일하면 0 이다. 수직 성분만 쓰므로
    바닥을 옆으로 문지르는 것은 세지 않는다. v_z 가 음수인 낙하/침강
    구간도 0 이다.

    Args:
        right_cfg / left_cfg: **접촉센서(contact_forces)** 의 팔 링크.
            ⚠ articulation 과 body 순서가 다르다. 섞어 쓰면 조용히 틀린다.
        vel_ref: 이 상승 속도[m/s] 이상은 같게 본다. 실측 밀기 v_z 가
                 앙와위 0.5~0.73, 복와위 0.29~0.69 라 0.5 로 잡았다.
        budget:  에피소드당 **적립액** 상한 [unit.s]. 시간이 아니다.
                 실측 기준 앙와위 0.078, 복와위 예비 0.042 를 적립하므로
                 0.3 이면 개선 여지를 4 배 남기면서 농사는 막는다.
                 가중치 15 에서 에피소드당 최대 4.5 (Episode_Reward 0.45).
    """
    dev = env.device
    if not hasattr(env, "_arm_lift_credit"):
        env._arm_lift_credit = torch.zeros(env.num_envs, device=dev)
    just_reset = env.episode_length_buf <= 1
    if bool(just_reset.any()):
        env._arm_lift_credit = torch.where(
            just_reset, torch.zeros_like(env._arm_lift_credit), env._arm_lift_credit
        )

    sensor = env.scene.sensors[right_cfg.name]
    fz = sensor.data.net_forces_w[..., 2]
    z_r = fz[:, right_cfg.body_ids].sum(dim=1).clamp(min=0.0)
    z_l = fz[:, left_cfg.body_ids].sum(dim=1).clamp(min=0.0)
    weak = torch.minimum(z_r, z_l)

    asset: Articulation = env.scene[asset_cfg.name]
    v_z = asset.data.root_lin_vel_w[:, 2].clamp(min=0.0, max=vel_ref)

    val = (weak * v_z / (_body_weight(env, asset_cfg) * vel_ref)).clamp(max=1.0)

    if hasattr(env, "reset_hold_steps"):
        val = torch.where(env.reset_hold_steps > 0, torch.zeros_like(val), val)

    remain = (budget - env._arm_lift_credit).clamp(min=0.0)
    val = torch.where(remain > 0.0, val, torch.zeros_like(val))
    env._arm_lift_credit = env._arm_lift_credit + val * env.step_dt
    return val


def arm_lift_balance_l2(
    env,
    right_cfg: SceneEntityCfg,
    left_cfg: SceneEntityCfg,
    vel_ref: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """밀어올리는 **그 순간**의 좌우 수직 지지력 차이 (체중 정규화, 제곱).

    누적이 아니라 순간을 봐야 하는 이유: 앙와위 실측에서 누적 수직
    충격량은 R 5.557 / L 5.043 으로 거의 같은데, 매 순간에는 한쪽만
    일한다 (0.68 s 에 R 45.6 % / L 15.2 %, 0.86 s 에 R 28.3 % / L 47.4 %).
    **오른팔로 밀고 넘긴 뒤 왼팔로 미는 순차 동작**이다. 누적을 보는
    항으로는 절대 안 잡힌다.

    v_z 로 가중해 실제로 들어올리는 구간에만 압력이 걸리게 한다.
    이렇게 하면 "안 올라가면 벌점도 없다" 는 회피로가 생기지만, 그 길은
    stand_success(5.0/s)를 잃으므로 이득이 안 된다.
    """
    sensor = env.scene.sensors[right_cfg.name]
    fz = sensor.data.net_forces_w[..., 2]
    z_r = fz[:, right_cfg.body_ids].sum(dim=1).clamp(min=0.0)
    z_l = fz[:, left_cfg.body_ids].sum(dim=1).clamp(min=0.0)

    asset: Articulation = env.scene[asset_cfg.name]
    v_z = asset.data.root_lin_vel_w[:, 2].clamp(min=0.0, max=vel_ref)

    # ★ 제곱값을 1.0 에서 자른다 ★
    #   체중으로 나눴으니 최대 1 일 거라 봤는데 아니었다. 일부 env 의
    #   충격력이 체중의 3 배를 넘어(복와위 피크 222 N vs 체중 55.9 N)
    #   제곱하면 15 이 되고 env 평균이 그쪽으로 끌려간다. 실측에서
    #   복와위 0.80 s 의 항 값이 2.75, 가중치 -20 이면 그 순간 -55/s 로
    #   기상 보상 전체(10.5/s)의 다섯 배였다.
    #
    #   "한 팔이 체중만큼 더 받친다" 를 넘으면 이미 완전한 한팔이라 더
    #   벌줄 필요가 없다. 자르면 이 항은 "힘이 얼마나 큰가" 가 아니라
    #   "얼마나 한쪽으로 쏠렸나" 를 재게 되고, 충격 스파이크가 기울기를
    #   지배하는 것도 막힌다.
    diff = (z_r - z_l) / _body_weight(env, asset_cfg)
    val = torch.square(diff).clamp(max=1.0) * (v_z / vel_ref)

    if hasattr(env, "reset_hold_steps"):
        val = torch.where(env.reset_hold_steps > 0, torch.zeros_like(val), val)
    return val


# ─────────────────────────────────────────────────────────────────────────
# ★ 2026-09-13 — 팔 관절각 거울상 강제 (보상 5 차 시도) ★
#
#   앞선 네 판이 모두 우회당했다:
#     19  관절 토크 차이            -> 허공에서 왼팔 휘두르기
#     20  접촉력 norm + 시간예산    -> 낙하/침강 구간을 지지로 팔기
#     21  일률 + 순간차이(-40)      -> 팔 일을 다리로 넘기기
#     22  일률(+40) + 순간차이(-20) -> 한팔 밀기 힘 키우기
#   복와위 최종 밀기(0.80 s)의 왼팔 수직력은 네 판 모두 0.00 N 이었다.
#
#   이 항이 다른 점 두 가지:
#     ① **위치라서 도피로가 없다.** 힘·토크·접촉은 모두 "안 하면 0" 이라는
#        길이 있었다. 관절각 합은 팔이 어디 있든 항상 정의되고, 값을 줄이는
#        방법이 **자세를 거울상으로 만드는 것 하나**뿐이다.
#     ② **세 관절 전부를 묶는다.** elbow_height_sym 은 팔꿈치 높이 하나만
#        봐서 높이는 맞추고 자세는 다르게 할 수 있었다.
#
#   MX3 팔 관절은 좌우가 부호 반대다 (기본자세 r_elbow +1.49 / l_elbow -1.49).
#   따라서 q_R + q_L = 0 이 완전한 거울상이다.
#   실측 (앙와위 0.80 s): 오른 어깨피치 +1.530 / 왼 어깨피치 -0.677
#                        -> 합 0.853.  대칭이면 0.
# ─────────────────────────────────────────────────────────────────────────
def arm_joint_mirror_l1(
    env,
    right_cfg: SceneEntityCfg,
    left_cfg: SceneEntityCfg,
    gate_from_height: float | None = None,
    supine_height: float = 0.057,
    stand_height: float = 0.39,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """좌우 팔 관절각의 거울상 이탈량 [rad].

        Σ |q_R + q_L|        (좌우 부호가 반대이므로 합이 0 이면 거울상)

    Args:
        right_cfg / left_cfg: articulation 관절. **같은 순서로** 짝을 맞춰야
            한다 (어깨피치-어깨피치, 어깨롤-어깨롤, 팔꿈치-팔꿈치).
            preserve_order=True 필수.
        gate_from_height: None 이면 항상 적용. 값을 주면 그 height_frac
            이상에서 0 으로 끈다 (기립 후 팔을 자유롭게 두고 싶을 때).
            joint_deviation_arm 이 이미 기립 후 자세를 잡으므로 기본은 None.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q_r = asset.data.joint_pos[:, right_cfg.joint_ids]
    q_l = asset.data.joint_pos[:, left_cfg.joint_ids]
    dev = torch.sum(torch.abs(q_r + q_l), dim=1)

    if gate_from_height is not None:
        z = asset.data.root_pos_w[:, 2]
        frac = ((z - supine_height) / (stand_height - supine_height)).clamp(0.0, 1.0)
        dev = torch.where(frac >= gate_from_height, torch.zeros_like(dev), dev)

    if hasattr(env, "reset_hold_steps"):
        dev = torch.where(env.reset_hold_steps > 0, torch.zeros_like(dev), dev)
    return dev


# ═══════════════════════════════════════════════════════════════════════
# ★ 2026-09-13 — 보행 베이스 항들의 기상 구간 게이팅 ★
#
#   구간별 보상 감사(probe_audit.py, model_82100, 복와위 -90 도) 결과,
#   밀기 구간의 합계가 **-22.0/s** 였다. 정책 입장에서 기상은 "빨리
#   통과해야 하는 손해 구간" 이고, 그래서
#     - 기상 시간이 여섯 판 내내 1.2 s 에 고정됐고
#     - 밀기에서 팔을 더 쓰면 벌점만 늘어 양팔 전환이 안 됐다.
#
#   구간별 실측 (가중치 포함, /초):
#       항목                    누움      밀기    기립후
#       base_height           -12.515   -6.260   -0.103
#       arm_lift_balance       -0.096   -4.616   +0.000
#       elbow_height_sym       -0.147   -4.583   -0.011
#       base_roll_l2           -0.046   -2.234   -0.015
#       joint_load_ratio       -1.567   -1.943   -0.114
#       undesired_contacts     -2.667   -0.732   +0.000
#       base_pitch_target_l2   -2.116   -0.574   -0.002
#       stop_default_posture   -0.096   -0.477   -0.003
#       base_linear_velocity   -0.059   -0.298   -0.019
#       pelvis_pitch_symmetry  -0.036   -0.651   -0.060
#       gait                   +0.098   +0.215   +0.810
#       arm_lift_power         +1.547   +0.000   +0.000   <- 복와위
#       arm_lift_power         +0.000   +4.468   +0.000   <- 앙와위
#
#   ※ 복와위는 양팔 보상을 **누워 있는 동안 다 받고 정작 밀기에서는 0**
#     이었다. 같은 항이 두 자세에서 정반대로 작동했다.
# ═══════════════════════════════════════════════════════════════════════
def undesired_contacts_gated(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """undesired_contacts 를 기립 후에만 센다.

    ★ 이 항이 기상을 직접 벌주고 있었다 ★
      body_names=["(?!.*ankle.*).*"] 로 **발목 외 모든 바디의 접촉을
      개수로** 센다. 누워 있으면 몸통·어깨·팔꿈치가 다 닿으니 복와위
      누움 구간에 -2.667/s 다. 더 나쁜 것은 **팔로 밀면 팔 접촉이 늘어
      벌점이 커지고, 두 팔로 밀면 더 커진다** 는 점이다. 양팔 밀기를
      정면으로 반대하는 항이었다.

      기상 중 몸통·팔이 바닥에 닿는 것은 당연하고 벌주면 안 된다.
      기립 후에 몸통이 닿는 것만 이상 상황이다.
    """
    from isaaclab.envs.mdp.rewards import undesired_contacts as _uc
    r = _uc(env, threshold=threshold, sensor_cfg=sensor_cfg).float()
    return r * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


def lin_vel_z_l2_gated(
    env: ManagerBasedRLEnv,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """수직 속도 제곱 벌점을 기립 후에만 적용한다.

    보행에서는 통통 튀는 것을 막는 항인데, 기상에서는 **올라가는 동작
    자체를 벌준다**. arm_lift_power 는 같은 v_z 를 보상하므로 두 항이
    정면으로 싸웠다 (밀기 구간 -0.298/s).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    r = torch.square(asset.data.root_lin_vel_b[:, 2])
    return r * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


def stop_default_posture_gated(
    env: ManagerBasedRLEnv,
    gait_command_name: str,
    supine_height: float,
    stand_height: float,
    include_stopping: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """정지 구간 기본자세 벌점을 기립 후에만 적용한다.

    원 함수의 active 는 `~is_walking & ~is_stopping` 뿐이고 **높이 게이팅이
    없다.** 그래서 정지 명령을 받은 env(약 9 %)는 누워 있는 동안에도
    "선 자세 관절각에서 벗어났다" 고 벌점을 받았다 (밀기 -0.48~-1.08/s).
    기상은 관절을 크게 움직여야 하므로 정면 충돌이다.
    """
    from . import htwk_gait_v01 as _hg
    r = _hg.stop_default_posture_v01(
        env, asset_cfg=asset_cfg, gait_command_name=gait_command_name,
        include_stopping=include_stopping)
    return r * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


def pelvis_pitch_symmetry_gated(
    env: ManagerBasedRLEnv,
    window_size: int,
    command_name: str,
    supine_height: float,
    stand_height: float,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """보행용 골반 피치 대칭 벌점을 기립 후에만 적용한다 (밀기 -0.35~-0.65/s)."""
    r = _mx_rewards.pelvis_pitch_symmetry_windowed(
        env, window_size=window_size, command_name=command_name,
        command_threshold=command_threshold)
    return r * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


def feet_gait_gated(
    env: ManagerBasedRLEnv,
    gait_command_name: str,
    offset: tuple[float, float],
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """보행 위상 보상을 기립 후에만 준다.

    누워서 양발이 바닥에 닿아 있으면 anti-phase 중 한 발이 stance 와
    일치해 **절반 점수를 공짜로** 받았다 (누움 +0.098~+0.303/s).
    """
    from . import htwk_gait_v01 as _hg
    r = _hg.feet_gait_v01(
        env, gait_command_name=gait_command_name, offset=offset,
        threshold=threshold, sensor_cfg=sensor_cfg)
    return r * _stand_gate(env, supine_height, stand_height, asset_cfg=asset_cfg)


# ═══════════════════════════════════════════════════════════════════════
# ★ 2026-09-14 — 과회전(넘어감) 벌점 ★
#
#   【무엇을 봤나】 V3 2 판 model_7000, 복와위 90 도 실측:
#       t      base_z   g_x     g_z     뒤집힘%   upright  base_height
#       0.00    0.210   +0.99    0.00     0 %      0.000     0.000
#       0.80    0.260   +1.00   -0.01     0 %      0.768     1.826
#       1.20    0.268   +0.38   -0.72     0 %      1.368     1.904
#       1.40    0.242   -0.43   -0.62    79.7 %    1.127     1.664
#       1.60    0.086   -0.98   -0.03   100.0 %    0.124     0.261   <-
#       2.20    0.212   +0.14   -0.98     0 %      1.156     1.401
#       3.00    0.371    0.00   -1.00     0 %      2.357     2.832
#     엎드림(g_x +0.99)에서 상체를 들다가 **직립을 지나쳐 등으로 넘어간다**
#     (g_x -0.98). base_z 가 0.268 -> 0.086 으로 주저앉고, 1.6 초에 128 env
#     **전부** 뒤집힌다. 그 뒤 1.8~3.0 초는 앙와위 기상이다.
#
#   【굴러야 이득인가 — 아니다】
#     1.20 s (성공 직전)  upright 1.368 / base_height 1.904
#     1.60 s (쓰러진 뒤)  upright 0.124 / base_height 0.261
#     보상이 1/10 로 떨어진다. 이득을 보려고 구르는 게 아니라 **실패**다.
#
#   【왜 기존 항이 못 막았나】
#     base_ang_vel_xy 가 각속도를 잡는 항인데 _stand_gate(높이 0.340 이상)
#     로 게이팅돼 있다. 넘어가는 시점의 base_z 는 0.268 이라 **감쇠가
#     필요한 순간에 꺼져 있다.** 가중치를 올려도 이미 일어선 뒤에만 걸린다.
#
#   【왜 각속도가 아니라 이 형태인가】
#     각속도를 벌주면 기상에 **필요한** 회전까지 벌준다. 이 항은 "출발
#     방향을 지나친 정도" 만 세므로 회전은 자유롭고 넘어감만 막는다.
# ═══════════════════════════════════════════════════════════════════════
@hold_masked
def getup_overshoot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """출발 방향을 지나친 정도의 제곱 [0, 1].

        clamp(g_x * fall_sign, min=0) ** 2

    fall_sign 은 reset_root_supine 이 저장한다 (엎드림 -1 / 등 +1).
    출발 시 g_x ~= -fall_sign 이므로 곱이 음수 -> clamp 로 0 이다.
    직립이면 g_x ~= 0 -> 0. 반대쪽으로 넘어가야 양수가 된다.

        엎드림 출발(sign -1)   g_x +0.99 -> -0.99 -> 0     (정상)
                              g_x  0.00 ->  0.00 -> 0     (직립)
                              g_x -0.98 -> +0.98 -> 0.96  (넘어감)

    게이트를 걸지 않는다 — 과회전이 일어나는 base_z 0.24~0.27 은 어떤
    기립 게이트보다도 낮고, 기립 후에는 g_x ~= 0 이라 자동으로 0 이 된다.
    (홀드 구간은 @hold_masked 가 처리한다.)

    앙와위(sign +1)에도 대칭으로 적용돼 앞으로 고꾸라지는 것도 막는다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_x = asset.data.projected_gravity_b[:, 0]
    sign = getattr(env, "getup_fall_sign", None)
    if sign is None:
        return torch.zeros_like(g_x)
    return torch.square((g_x * sign).clamp(min=0.0))

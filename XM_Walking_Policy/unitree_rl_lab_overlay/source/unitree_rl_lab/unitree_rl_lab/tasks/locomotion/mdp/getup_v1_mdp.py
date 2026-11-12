"""getup_v1 (등 대고 누운 상태에서 일어나기) 전용 MDP 항목.

★ 분리 원칙 ★
  reset_hold.py 는 **한 줄도 수정하지 않는다**. 보행/킥이 쓰는
  DelayedJointPositionAction 을 상속만 해서 필요한 부분만 덮어쓴다.
  따라서 이 파일을 어떻게 고치든 보행/킥 태스크에는 영향이 없다.
"""

from __future__ import annotations

import functools
import math
import torch

from isaaclab.utils import math as math_utils

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .htwk_gait_v01 import PhaseOnlyGaitCommand, PhaseOnlyGaitCommandCfg

from .reset_hold import (
    DelayedJointPositionAction,
    DelayedJointPositionActionCfg,
    mask_rewards_during_reset_hold,
)


# ═══════════════════════════════════════════════════════════════
#  gait 명령 — 기립 후 제자리걸음 / 완전정지
# ═══════════════════════════════════════════════════════════════
#  ★ htwk_gait_v01.py 는 한 줄도 고치지 않는다 ★
#    보행이 쓰는 공유 파일이다. 상속만 해서 필요한 부분만 덮어쓴다.
#    (reset_hold.py 에 적용한 것과 같은 원칙)
#
#  왜 필요한가
#    실기: "다 잘 일어나 놓고 서 있는 상태에서 과하게 움직여 발산한다."
#    가만히 있으려 애쓰게 만드는 대신 **구조화된 리듬**을 준다. 한계주기가
#    생기면 움직임이 조직화되고, 밀렸을 때 이미 발을 떼고 있어 회복이 쉽다.
#    (실측: 0.6 m/s 밀림에서 발뗌 100 % — 스텝이 회복의 핵심이었다)
#
#  왜 서브클래스가 필요한가
#    원본 _standing_mask 는 velocity_command_name 이 가리키는 속도 명령의
#    노름으로 "정지 상태" 를 판정한다. 기상 태스크에는 속도 명령이 없다.
#    None 으로 두면 전부 False 가 되어 **항상 걷기만** 하고, 그러면
#    완전정지 모드가 영영 안 나와서 보행 인계 자세를 못 배운다.
#    -> 항상 True 로 두고 rel_stop_when_standing 이 비율을 정하게 한다.


class GetupGaitCommand(PhaseOnlyGaitCommand):
    """속도 명령이 없는 기상 태스크용 gait 명령.

    모든 env 를 "정지 가능" 으로 보고, rel_stop_when_standing 비율로
    제자리걸음과 완전정지를 섞는다. 그 외 동작(위상 적분, 정지 시 위상
    0 수렴, reset hold 동안 동결)은 부모 그대로다.
    """

    def _standing_mask(self) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)


@configclass
class GetupGaitCommandCfg(PhaseOnlyGaitCommandCfg):
    class_type: type = GetupGaitCommand
    velocity_command_name: str | None = None   # 속도 명령 없음


def full_stop_mask(env, gait_command_name: str = "gait") -> torch.Tensor:
    """완전정지(주파수 0 & 위상 0) 인 env 를 1.0 으로 주는 마스크 [N].

    보행의 stop_posture_v01 / stop_default_posture_v01 이 쓰는 판정과
    같다. "정지 자세로 가만히" 계열 보상을 이걸로 게이팅하면 제자리걸음
    과의 충돌이 구조적으로 사라진다 — 보행이 쓴 해법 그대로다.
    """
    try:
        term = env.command_manager.get_term(gait_command_name)
    except (KeyError, AttributeError):
        return torch.ones(env.num_envs, device=env.device)
    return ((~term.is_walking) & (~term.is_stopping)).float()


# ═══════════════════════════════════════════════════════════════
#  액션 — 리셋 홀드 동안 "스폰된 자세" 를 유지한다
# ═══════════════════════════════════════════════════════════════
class GetupJointPositionAction(DelayedJointPositionAction):
    """리셋 홀드 구간에서 기본각이 아니라 스폰 당시 관절각을 유지한다.

    【왜 필요한가】
      부모 클래스는 홀드 동안 action 을 0 으로 마스킹한다. 그러면
          target = default_joint_pos + 0 * scale = default_joint_pos
      이 되어 PD 가 관절을 기본각으로 끌고 간다. stiffness 50 / damping 2.5
      기준 정정시간이 약 0.25 s 라, 0.3 s 이상 홀드하면 리셋 이벤트가 준
      **관절 랜덤화가 통째로 지워진다.**

      기상 태스크는 "떨어뜨려서 정착시킨 뒤 시작" 하는 구조라 홀드가 길다.
      그동안 스폰 자세를 유지해야 랜덤화가 살아남는다.

      물리적으로도 이쪽이 맞다 — 실제로 넘어질 때 PD 는 정책이 마지막에
      명령한 자세를 잡고 있지, 기본각으로 되돌아가지 않는다.

    【리셋 순서 근거】
      manager_based_env.py::_reset_idx 에서
          scene.reset → event_manager.apply("reset") → action_manager.reset
      순으로 돌기 때문에, 이 클래스의 reset() 시점에는 이벤트가 쓴
      (랜덤화된) 관절 상태가 이미 data.joint_pos 에 반영돼 있다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 첫 reset 전에 참조될 일은 없지만 안전하게 기본각으로 초기화한다.
        self._q_at_reset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        # 이벤트가 이미 적용된 뒤라 이 값이 곧 "스폰된 자세" 다.
        self._q_at_reset[idx] = self._asset.data.joint_pos[idx][:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor):
        # super() 가 홀드 카운터를 깎으므로 마스크를 먼저 떠 둔다.
        hold_mask = None
        if hasattr(self._env, "reset_hold_steps"):
            hold_mask = self._env.reset_hold_steps > 0

        # 리워드 마스킹용으로 "깎기 전" 마스크를 env 에 남긴다.
        #   RewardManager 는 process_actions 이후에 돌기 때문에, 그때
        #   reset_hold_steps 를 다시 읽으면 마지막 홀드 스텝이 이미 0 으로
        #   깎여 있어 리워드가 한 스텝 새어 나간다. 여기서 박제해 둔다.
        if hold_mask is None:
            hold_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._env.getup_hold_active = hold_mask.clone()

        super().process_actions(actions)

        # 기본각 대신 스폰 자세를 목표로 준다.
        if torch.any(hold_mask):
            self._processed_actions[hold_mask] = self._q_at_reset[hold_mask]


@configclass
class GetupJointPositionActionCfg(DelayedJointPositionActionCfg):
    """DelayedJointPositionActionCfg 와 동일. class_type 만 바뀐다."""

    class_type = GetupJointPositionAction


# ═══════════════════════════════════════════════════════════════
#  리워드 홀드 마스킹
# ═══════════════════════════════════════════════════════════════
def hold_masked(func):
    """리셋 홀드(낙하·정착) 구간 동안 이 리워드를 0 으로 만든다.

    정착은 에피소드 밖 취급이다. 마스킹하지 않으면
      - alive 가 공짜로 쌓이고
      - 낙하 중 관절속도/가속도가 정책 탓으로 벌점 처리되고
      - upright / height 가 누운 상태 값으로 들어간다.

    functools.wraps 가 __wrapped__ 를 달아 주므로 IsaacLab 매니저의
    inspect.signature 검증은 원본 시그니처를 보고 통과한다.
    """

    @functools.wraps(func)
    def wrapper(env, *args, **kwargs):
        return mask_rewards_during_getup_hold(env, func(env, *args, **kwargs))

    return wrapper


def mask_rewards_during_getup_hold(env, reward: torch.Tensor) -> torch.Tensor:
    """홀드 구간의 리워드를 0 으로 만든다.

    reset_hold.mask_rewards_during_reset_hold 를 쓰지 않는 이유:
    그 함수는 env.reset_hold_steps 를 그 자리에서 읽는데, RewardManager 가
    도는 시점에는 process_actions 이 이미 카운터를 깎은 뒤다. 그래서 마지막
    홀드 스텝의 리워드가 마스킹되지 않고 새어 나간다(실측 0.0058).
    GetupJointPositionAction 이 깎기 전에 박제해 둔 마스크를 쓴다.
    """
    mask = getattr(env, "getup_hold_active", None)
    if mask is None:
        return reward
    if torch.any(mask):
        reward = reward.clone()
        reward[mask] = 0.0
    return reward


# ═══════════════════════════════════════════════════════════════
#  리셋 — 등을 대고 누운 자세 (월드/바디 축을 명시적으로 분리)
# ═══════════════════════════════════════════════════════════════
def reset_root_supine(
    env,
    env_ids,
    xy_range: tuple[float, float],
    height_range: tuple[float, float],
    yaw_range: tuple[float, float],
    side_roll_range: tuple[float, float],
    tilt_range: tuple[float, float],
    fall_angle_deg: float = 90.0,
    fall_angle_min_deg: float = 0.0,
    prone_ratio: float = 0.5,
    supine_height: float = 0.057,
    stand_height: float = 0.39,
    prone_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """등을 바닥에 댄 자세로 루트를 리셋한다.

    【왜 mdp.reset_root_state_uniform 을 안 쓰는가】
      그 함수는 orientations = quat_mul(q_default, q_delta) 로 델타를
      **바디 축** 에 곱한다. init_rot 이 직립이면 roll/pitch/yaw 가 그대로
      월드 축이라 직관적이지만, 이 태스크는 init_rot 이 supine(pitch −90°)
      이라 축이 통째로 뒤바뀐다:
          body x → world +z,  body y → world +y,  body z → world −x
      즉 'yaw' 키가 월드 yaw 가 아니라 '옆으로 구르기' 가 된다. 주석으로
      떼우면 반드시 나중에 헷갈리므로, 의미대로 이름 붙인 인자를 받는다.

    Args:
        xy_range: 평면 위치 흔들기 [m]
        height_range: 안착 높이 위로 띄울 높이 [m]. 여기서 떨어뜨려 정착시킨다.
            바닥에 파고든 채 스폰되면 depenetration 임펄스로 로봇이 튄다.
        fall_angle_deg: **역방향 커리큘럼용 넘어진 정도** [deg] — 밴드 상한.
            0 = 직립, 90 = 완전히 눕는다.
        prone_ratio: 엎드림(prone) 비율 [0,1]. 0=전부 supine, 1=전부 prone,
            0.5=반반(학습 기본값). **방향은 fall_angle_deg 의 부호가 아니라
            이 인자로 정한다** — 각도는 abs() 로 크기만 쓰기 때문이다.
        fall_angle_min_deg: 밴드 하한 [deg]. env 마다 |각도| 를
            [min, max] 에서 균일 추출하고 prone_ratio 로 부호를 붙인다.
            + = supine(등을 댐) / - = prone(엎드림).
            커리큘럼은 이전 레벨의 각도를 여기에 넣어 레벨별 밴드를 만든다
            (레벨 0 은 0). 예: 레벨 2 면 |각도| 가 20~40 도에서 뽑힌다.
            정책이 기립 상태를 한 번도 경험하지 못하는 것이 이 태스크의
            근본 난점이라(8 판 통틀어 stand_success 가 한 번도 발화하지
            않았다), 거의 서 있는 상태에서 시작해 점점 눕혀 간다.
            높이는 cos 보간으로 함께 낮춘다.
        supine_height / stand_height: 높이 보간 양 끝점 [m].
        prone_height: 엎드림(-) 쪽 안착 높이 [m]. None 이면 supine_height 를
            쓴다(= 예전 동작). MX3 실측 0.1099 — 앙와위 0.0577 보다 5.2 cm
            높으므로 복와위를 쓰는 태스크는 **반드시 지정**해야 한다.
        yaw_range: **월드 수직축** 회전 [rad]. 어느 방향을 보고 누웠는가.
            평지에 방향 선호가 없으므로 처음부터 ±π 로 열어도 난이도가
            올라가지 않는다. 강건성만 공짜로 얻는다.
        side_roll_range: **바디 z축**(머리-발 축) 회전 [rad]. 등이 아니라
            어깨 한쪽으로 기울어 눕는 정도. 커리큘럼으로 넓히는 주축이다.
        tilt_range: **바디 y축**(좌우 축) 회전 [rad]. 머리가 위/아래로
            끄덕이는 정도.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    n = len(env_ids)
    dev = asset.device
    root = asset.data.default_root_state[env_ids].clone()

    def _u(rng):
        return torch.empty(n, device=dev).uniform_(rng[0], rng[1])

    zeros = torch.zeros(n, device=dev)

    # --- 위치 ---
    #   기울기에 따라 안착 높이가 달라진다. 직립(0°)이면 stand_height,
    #   supine(90°)이면 supine_height. 강체 근사로 cos 보간한다.
    #   정확할 필요는 없다 — 살짝 띄워 떨어뜨리고 홀드 동안 정착시킨다.
    # ★ 2026-09-10 — 스칼라 각도 -> **env 별 밴드 추출 + 양방향** ★
    #   이전: 전 env 가 정확히 +fall_angle_deg. 즉 "뒤로 넘어짐" 한 방향만.
    #   실측(model_29800): supine +5~+90 전부 기립 100 %, 그런데
    #                      prone -30/-60/-90 은 0~9 %. 한 방향만 배웠다.
    #   실기에서 "첫 시도 실패 후 재시도가 안 되는" 원인이 이것이다 —
    #   한 번 넘어지면 어느 방향으로 눕는지 모르는데 학습은 한쪽만 봤다.
    #
    #   밴드로 뽑는 이유(A: 균일 [-a,+a] 를 안 쓰는 이유):
    #     균일이면 레벨 5(±90)에서도 -15, +8 같은 쉬운 각도가 섞여 승격
    #     판정(stand_success EMA >= 0.5)이 물러진다. 쉬운 env 들이 EMA 를
    #     밀어 올려 진짜 90 도를 못 해도 승격된다. 이 세션 1 판이 정확히
    #     그런 식(문턱이 헐거워서)으로 레벨만 올라가고 못 서던 실패를 했다.
    #     밴드는 레벨별 난이도를 유지하면서 양방향을 덮는다.
    #
    #   ※ 높이 보간은 cos 이라 짝함수 — ±ang 에 같은 값을 준다. 엎드림
    #     안착이 supine 보다 3.1 cm 높지만(실측 0.088 vs 0.057),
    #     ★ 2026-09-13 정정 — 이 가정이 틀렸다 ★
    #       "height_range(0.03~0.06)가 그 위로 띄우므로 파고들지 않는다" 고
    #       적혀 있었으나 **엎드림 안착 높이가 앙와위보다 5.2 cm 높다**.
    #       액션 0 으로 낙하시켜 실측 (probe_settle.py, 256 env):
    #           앙와위 안착  0.0577 m   (범위 0.0574~0.0578)
    #           복와위 안착  0.1099 m   (범위 0.1092~0.1104)
    #       supine_height(0.057) + 3~6 cm = 0.087~0.117 이므로 **범위의
    #       대부분이 바닥 아래**다. PhysX 가 침투를 해소하며 로봇을 공중으로
    #       날려버리고, 화면에서는 "뒤집혀서 튀어오르는" 것으로 보인다
    #       (실측: base_z 0.154 -> 0.355, 그동안 접촉력 0 = 자유 비행).
    #       -> prone_height 로 엎드림 쪽 보간 끝점을 따로 준다.
    lo = math.radians(min(abs(fall_angle_min_deg), abs(fall_angle_deg)))
    hi = math.radians(max(abs(fall_angle_min_deg), abs(fall_angle_deg)))
    mag = torch.empty(n, device=dev).uniform_(lo, hi) if hi > lo else \
        torch.full((n,), hi, device=dev)
    # prone_ratio: 0.0 = 전부 supine(등) / 1.0 = 전부 prone(엎드림) / 0.5 = 반반.
    #   학습은 0.5. 추론/계측에서 한 방향만 보려면 0 또는 1 을 준다.
    #   ★ fall_angle_deg 의 **부호로는 방향을 못 정한다** — 위에서 abs() 로
    #     크기만 쓰기 때문이다. 방향은 반드시 이 인자로 지정할 것.
    sign = torch.where(torch.rand(n, device=dev) < prone_ratio, -1.0, 1.0)
    ang = mag * sign                       # +는 supine(등), -는 prone(엎드림)

    # ★ 2026-09-14 — 넘어진 방향을 env 에 남긴다 (과회전 벌점용) ★
    #   getup_overshoot 가 "출발 방향을 지나쳤는가" 를 판정하려면 이 부호가
    #   필요하다. 여기 말고는 알 수 없다 (관측의 g_x 는 기상 중 계속 변한다).
    if not hasattr(env, "getup_fall_sign"):
        env.getup_fall_sign = torch.ones(env.num_envs, device=dev)
    env.getup_fall_sign[env_ids] = sign

    # 방향별 안착 높이로 보간한다. prone_height 를 안 주면 예전과 동일 동작
    #   (supine_height 를 양쪽에 쓴다) 이라 v1/v2 는 영향이 없다.
    _ph = supine_height if prone_height is None else prone_height
    base_h = torch.where(
        sign < 0.0,
        _ph + (stand_height - _ph) * torch.cos(ang),
        supine_height + (stand_height - supine_height) * torch.cos(ang),
    )
    pos = root[:, 0:3] + env.scene.env_origins[env_ids]
    pos[:, 0] += _u(xy_range)
    pos[:, 1] += _u(xy_range)
    pos[:, 2] = base_h + _u(height_range)

    # --- 자세 ---
    # q = R_z(월드 yaw) · R_y(-넘어진 각도) · R_z(옆구르기) · R_y(끄덕임)
    #   왼쪽 곱 = 월드 축 회전, 오른쪽 곱 = 바디 축 회전.
    #   pitch -90° 가 supine 이므로 넘어진 각도에 음수를 붙인다.
    #   (로봇 cfg 의 init_rot 은 여기서 쓰지 않는다 — 각도를 직접 만든다)
    q_def = quat_from_euler_xyz(zeros, -ang, zeros)
    q_body = quat_mul(
        quat_from_euler_xyz(zeros, zeros, _u(side_roll_range)),   # 바디 z
        quat_from_euler_xyz(zeros, _u(tilt_range), zeros),        # 바디 y
    )
    q_world_yaw = quat_from_euler_xyz(zeros, zeros, _u(yaw_range))
    quat = quat_mul(q_world_yaw, quat_mul(q_def, q_body))

    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    # 낙하는 중력에만 맡긴다. 초기 속도를 주면 정착 결과가 튄다.
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


# ═══════════════════════════════════════════════════════════════
#  Critic 전용 특권 관측
#  ─────────────────────────────────────────────────────────────
#  Critic 은 sim 에서만 돌기 때문에 실기에서 못 얻는 정보를 써도 된다
#  (비대칭 actor-critic). Policy 에는 절대 넣지 않는다 — 실기에 외부
#  트래킹이 없어 몸통 높이도 접촉력도 측정할 수 없다.
#
#  왜 필요한가: 이 태스크의 리워드 대부분(base_height 3.0 + stand_success
#  5.0 = 8.0/11.0)이 root_pos_w[:,2] 에서 나오는데, 그 높이를 Critic 이
#  못 보면 관절각과 중력벡터로 추론해야 한다. 가치 추정이 흐려지면
#  어드밴티지도 흐려지고, 희소 보상 태스크에서 그건 그대로 돌파 실패로
#  이어진다. feet_load(1.0) 도 접촉력 기반인데 마찬가지다.
# ═══════════════════════════════════════════════════════════════
def _feet_load_frac(env, sensor_cfg, asset_cfg):
    """발이 받치는 체중 비율 [0,1]. 리워드와 관측이 공유하는 계산부.

    ★ history 는 max 가 아니라 mean 을 쓴다 ★
      max 는 접촉 채터링의 피크를 잡아 정지 상태에서도 체중의 4 배가
      나온다. 실측(iter 1500, 완전 정지 상태): 양발 합 234.7 N,
      체중 55.88 N → 비율 1.61 이라 clamp 에 걸려 1.0 으로 포화된다.
      포화되면 기울기가 0 이라 신호가 죽는다.

    ★ 접촉센서 인덱싱 주의 ★
      sensor_cfg.body_ids 는 SceneEntityCfg 가 **센서 자체의** 바디
      순서로 푼 것이다. robot.data.body_names.index() 로 만든 인덱스를
      섞어 쓰면 엉뚱한 바디를 읽는다 (실측: 21 개 중 19 개 순서 불일치).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    fz = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    fz = fz.mean(dim=1).clamp(min=0.0).sum(dim=1)
    weight = asset.data.default_mass.to(fz.device).sum(dim=1) * 9.81
    return torch.clamp(fz / weight, 0.0, 1.0)


def feet_load_obs(env, sensor_cfg, asset_cfg=SceneEntityCfg("robot")):
    """발 하중 비율 (Critic 전용).

    리워드 쪽 feet_load_fraction 과 달리 홀드 마스킹을 하지 않는다.
    관측은 낙하·정착 구간에도 참값이어야 한다.
    """
    return _feet_load_frac(env, sensor_cfg, asset_cfg).unsqueeze(-1)


def episode_phase_obs(env):
    """에피소드 진행도 [0,1] (Critic 전용).

    walk_pose_deadline 처럼 **시간으로 켜지는** 리워드를 쓰면, 관측에
    시간이 없는 Critic 은 같은 자세를 t=0.5s 와 t=5s 에 같은 값으로 본다.
    가치 추정이 흐려지므로 진행도를 넣어 준다. Policy 에는 넣지 않는다 —
    실기에서 "지금 몇 초째인가" 는 낙하 감지 시점에 의존해 불확실하다.
    """
    return (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)


def body_contact_obs(env, sensor_cfg, threshold: float = 1.0):
    """바디별 접촉 여부 0/1 (Critic 전용).

    기상은 본질적으로 **접촉 모드 전환**(등 → 팔/엉덩이 → 발)이다.
    어느 부위가 바닥에 닿아 있는지가 곧 진행 단계라, Critic 이 이걸 보면
    가치 추정이 훨씬 쉬워진다. 관절각만으로는 같은 자세라도 공중인지
    바닥에 눌려 있는지 구분할 수 없다.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    f = torch.norm(f, dim=-1).max(dim=1)[0]
    return (f > threshold).float()


# ═══════════════════════════════════════════════════════════════
#  탐색 보조 — 중력 일부 상쇄 (기본 비활성)
#  ─────────────────────────────────────────────────────────────
#  ★ 기본 cfg 에는 들어 있지 않다 ★ 쓰려면 env cfg 의 events 에
#  apply_lift_assist, curriculum 에 lift_assist_curriculum 을 추가한다.
#
#  【왜 필요할 수 있나】
#    기상은 희소 보상 태스크다. 실측(4·5 차 학습, 서로 독립)에서 정책이
#    supine → 상체 세우기(1 초)까지는 배우지만 거기서 멈춘다:
#        base_z 0.139 에서 정체 (기립 0.396)
#        reward 33 / 이론 최대 123  = 27%
#    보상 헤드룸은 남아 있는데 경로를 못 찾는다. 앉은 자세에서 일어서려면
#    상체를 앞으로 숙여 골반 하중을 풀고 발로 옮긴 뒤 펴는 협응 동작이
#    필요한데, 그 과정에서 높이가 일시적으로 떨어지고 넘어질 수 있다.
#    즉 **낮은 보상의 계곡**을 건너야 한다.
#
#    HumanUP(RSS 2025)도 같은 문제를 머리에 1500 N 수직 보조력 +
#    커리큘럼 제거로 푼다.
#
#  【HumanUP 과 다르게 한 점】
#    그쪽은 특정 부위(머리)에 큰 힘을 주기적으로 건다. 여기서는 **바디별
#    질량에 비례한 중력 상쇄**로 구현했다. 한 점에 큰 힘을 주면 원치 않는
#    모멘트가 생기지만, 질량 비례 상쇄는 "중력이 약한 곳에서 연습한다" 는
#    의미가 그대로여서 해석이 쉽고 커리큘럼도 직관적이다 (alpha = 상쇄 비율).
# ═══════════════════════════════════════════════════════════════
def apply_lift_assist(env, env_ids, alpha: float, asset_cfg=SceneEntityCfg("robot")):
    """중력의 alpha 비율을 상쇄하는 수직 보조력을 건다 (월드 좌표 기준).

    alpha=0.6 이면 로봇이 체감하는 중력이 40% 가 된다. alpha=0 이면 무효과.

    set_external_force_and_torque 는 버퍼를 세팅하면 매 스텝 적용되므로
    reset 때 한 번만 걸면 에피소드 내내 유지된다.
    """
    asset = env.scene[asset_cfg.name]
    if alpha <= 0.0:
        return
    m = asset.data.default_mass.to(env.device)            # (num_envs, num_bodies)
    f = torch.zeros(env.num_envs, m.shape[1], 3, device=env.device)
    f[:, :, 2] = alpha * m * 9.81
    asset.set_external_force_and_torque(f, torch.zeros_like(f), is_global=True)


def lift_assist_curriculum(
    env,
    env_ids,
    stage_steps: tuple[int, ...],
    alpha: tuple[float, ...],
    term_name: str = "lift_assist",
):
    """보조력을 단계적으로 뺀다. alpha 의 마지막 원소는 0 이어야 한다.

    보조력을 끝까지 남기면 실기와 물리가 달라져 정책을 못 쓴다.
    """
    step = env.common_step_counter
    level = 0
    for i, s_ in enumerate(stage_steps):
        if step >= s_:
            level = i + 1
    level = min(level, len(alpha) - 1)

    cfg = env.event_manager.get_term_cfg(term_name)
    cfg.params["alpha"] = alpha[level]
    env.event_manager.set_term_cfg(term_name, cfg)
    return torch.tensor(alpha[level], dtype=torch.float32, device=env.device)


# ═══════════════════════════════════════════════════════════════
#  종료 — 수치 폭주 안전장치
# ═══════════════════════════════════════════════════════════════
def root_height_above_maximum(
    env,
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """몸통이 물리적으로 불가능한 높이까지 올라가면 종료한다.

    IsaacLab 에는 root_height_below_minimum 만 있고 위쪽 판정이 없어서
    직접 만든다.

    【왜 필요한가】
      기상 태스크는 관절 랜덤화 + 낙하 + self_collisions 조합이라,
      depenetration 임펄스(max_depenetration_velocity=5.0)로 로봇이
      튀어오를 수 있다. 종료 조건이 time_out 뿐이면 그 환경이 남은
      9.5 초 내내 쓰레기 샘플을 배치에 넣는다.

    【왜 아래쪽(below_minimum)은 못 쓰나】
      누운 자세(base_z 0.057)가 시작 상태다. 높이/자세 하한을 걸면
      매 에피소드가 첫 스텝에 끝난다. 위쪽만 걸 수 있다.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height


# ═══════════════════════════════════════════════════════════════
#  커리큘럼 — 리셋 범위를 단계적으로 넓힌다
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  외란 — 밀기(순간 충격) + 외력(지속 편향), 그리고 Critic 관측
# ═══════════════════════════════════════════════════════════════
#  둘 다 IsaacLab 원본을 감싸서 **가한 값을 env 에 기록**한다. Critic 이
#  그걸 보면 "정책이 못한 것" 과 "외란이 컸던 것" 을 구분할 수 있어
#  가치 추정의 잡음이 줄고 어드밴티지가 선명해진다. 비대칭 Critic 을
#  쓰는 이유가 그것이고, 관측 불가능한 외란을 넣으면서 Critic 에 안
#  알려주면 그 이점을 스스로 깎는 셈이다.
#
#  ★ 이 두 관측을 추가하면 Critic 입력 차원이 바뀐다 ★
#    기존 체크포인트에서 이어받기가 불가능해진다(shape mismatch).
#    그래서 이 항들을 넣는 판은 반드시 처음부터 학습해야 한다.

def push_recorded(
    env,
    env_ids,
    velocity_range: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gate_below_height_frac: float | None = None,
    gate_above_height_frac: float | None = None,
    supine_height: float = 0.057,
    stand_height: float = 0.39,
):
    """push_by_setting_velocity 와 동일하되 가한 속도 변화를 기록한다.

    기록: env._getup_last_push  [N,3] (선속도 xyz)
          env._getup_last_push_t [N]  (가한 시점의 common_step_counter)
    """
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [velocity_range.get(k, (0.0, 0.0)) for k in keys]

    # ★ 세기가 0 이면 **아무것도 하지 않는다** ★
    #   write_root_velocity_to_sim 은 값이 같아도 항등이 아니다. Articulation
    #   에서 루트 속도를 다시 써넣으면 PhysX 가 나머지 링크 속도와 정합을
    #   맞춰야 하고, IsaacLab 은 그때 body_acc_w 버퍼도 0 으로 지운다.
    #   램프 0 단계(레벨 0~4)에서 이게 0.3~1.0 초마다 기립 구간에 들어가
    #   레벨 1 을 못 넘겼다. 실측 (시드 42 로 동일, iter 361):
    #       외란 항 없음      sr 0.993  (iter 399 에 레벨 2)
    #       세기 0 짜리 밀기  sr 0.066  (레벨 1 정체)
    if max(max(abs(lo), abs(hi)) for lo, hi in range_list) == 0.0:
        return

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)

    # ★ 기립 전에만 민다 (2026-09-01, 실기 관찰 반영) ★
    #   서 있는 동안에도 0.3~1.0 초마다 밀면 정책이 "언제 밀릴지 모르니
    #   항상 대비" 하도록 배운다. sim 에서는 조용해 보여도(발 접촉 전환
    #   0.46 회/s) 실기에서는 센서 잡음·지연이 그 반응을 헛트리거해
    #   진동이 된다 — "일어난 뒤 과격한 동작으로 수렴을 못 한다".
    #   실측: 밀기 없음 0.46 회/s  vs  ±0.4 밀기 4.32 회/s (반응 준비 상태)
    #
    #   외란의 목적은 **일어나는 구간**의 강건성이다. 기립 후에는 필요
    #   없다. height_frac 이 문턱 아래인 env 만 민다.
    #   gate_above_height_frac 는 그 반대다 — **다 일어선 env 만** 민다.
    #   세고 드문 "넘어뜨리기" 용이다. 약하고 잦은 밀기와 성질이 다르다:
    #     약+잦음  버티면 이득 -> 정책이 항상 대비 -> 실기에서 진동
    #     세+드묾  저항이 무의미 -> 그냥 넘어지고 다시 일어나는 게 최적
    #   목적은 (a) 기상 샘플을 2~3 배 늘리고 (b) 인위적 supine 스폰이
    #   아니라 **실제로 넘어진 자세**에서 일어나게 하는 것이다.
    if gate_below_height_frac is not None or gate_above_height_frac is not None:
        z = asset.data.root_pos_w[env_ids, 2]
        hf = ((z - supine_height) / (stand_height - supine_height)).clamp(0.0, 1.0)
        keep = torch.ones_like(hf, dtype=torch.bool)
        if gate_below_height_frac is not None:
            keep &= hf < gate_below_height_frac
        if gate_above_height_frac is not None:
            keep &= hf > gate_above_height_frac
        env_ids = env_ids[keep]
        if len(env_ids) == 0:
            return

    vel_w = asset.data.root_vel_w[env_ids]
    ranges = torch.tensor(range_list, device=asset.device)
    delta = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    asset.write_root_velocity_to_sim(vel_w + delta, env_ids=env_ids)

    if not hasattr(env, "_getup_last_push"):
        env._getup_last_push = torch.zeros(env.num_envs, 3, device=env.device)
        env._getup_last_push_t = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._getup_last_push[env_ids] = delta[:, :3]
    env._getup_last_push_t[env_ids] = env.common_step_counter


def apply_base_force_recorded(
    env,
    env_ids,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """apply_external_force_torque 와 동일하되 걸린 힘을 기록한다.

    ★ asset_cfg 는 반드시 단일 바디(base_link)로 좁혀야 한다 ★
      원본은 **바디마다** 힘을 따로 뽑는다(size = (env, num_bodies, 3)).
      body_names=".*" 로 두면 21 개 바디에 각각 걸려 의도치 않게 커진다.

    permanent_wrench_composer 라 한 번 걸면 바꿀 때까지 계속 작용한다.
    mode="reset" 으로 쓰면 에피소드마다 새 편향, 그 안에서는 일정하다.

    기록: env._getup_ext_force [N,3]
    또한 밀기 기록을 리셋한다(새 에피소드에는 이전 밀기가 남으면 안 된다).
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)

    # ★ 세기가 0 이면 물리 호출을 **하지 않는다** ★
    #   WrenchComposer.set_forces_and_torques 는 값과 무관하게 _active=True
    #   로 놓는다(wrench_composer.py:201). 그러면 articulation.write_data_to_sim
    #   이 매 물리 스텝마다 apply_forces_and_torques_at_position 을 호출한다
    #   (articulation.py:229). 0 힘이라 수치적으로는 무해해 보여도, 램프
    #   0 단계에서 굳이 켤 이유가 없는 경로다.
    #   밀기(push_recorded)와 같은 함정이다 — 이벤트는 보상 항과 달리
    #   세기 0 이어도 실행된다.
    zero_f = max(abs(force_range[0]), abs(force_range[1])) == 0.0
    zero_t = max(abs(torque_range[0]), abs(torque_range[1])) == 0.0
    if zero_f and zero_t:
        if not hasattr(env, "_getup_ext_force"):
            env._getup_ext_force = torch.zeros(env.num_envs, 3, device=env.device)
        env._getup_ext_force[env_ids] = 0.0
        if hasattr(env, "_getup_last_push"):
            env._getup_last_push[env_ids] = 0.0
        return

    nb = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    size = (len(env_ids), nb, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torques, body_ids=asset_cfg.body_ids, env_ids=env_ids,
    )

    if not hasattr(env, "_getup_ext_force"):
        env._getup_ext_force = torch.zeros(env.num_envs, 3, device=env.device)
    env._getup_ext_force[env_ids] = forces[:, 0, :]

    if hasattr(env, "_getup_last_push"):
        env._getup_last_push[env_ids] = 0.0


def ext_force_obs(env, scale: float = 3.0):
    """Critic 전용 — base_link 에 걸린 지속 외력 [N,3]. 에피소드 내내 일정."""
    buf = getattr(env, "_getup_ext_force", None)
    if buf is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return buf / scale


def last_push_obs(env, tau: float = 0.5, scale: float = 0.4):
    """Critic 전용 — 최근 밀기 충격 [N,3]. 시간에 따라 지수 감쇠.

    감쇠를 쓰는 이유: 밀기는 순간 사건이라 그 스텝에만 신호를 주면
    Critic 이 "직후 몇 스텝의 나쁜 상태" 를 여전히 정책 탓으로 본다.
    tau=0.5 s 면 밀린 뒤 약 1 초간 흔적이 남아 그 구간을 설명해준다.

    부작용이 없는 순수 함수다 — 저장된 값과 경과 시간으로만 계산하므로
    관측 그룹이 여러 번 호출돼도 값이 달라지지 않는다.
    """
    v = getattr(env, "_getup_last_push", None)
    if v is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    elapsed = (env.common_step_counter - env._getup_last_push_t).clamp(min=0).float() * env.step_dt
    return (v / scale) * torch.exp(-elapsed / tau).unsqueeze(-1)


def disturbance_rampin_curriculum(
    env,
    env_ids,
    stage_steps: tuple[int, ...],
    push_vel: tuple[float, ...],
    ext_force: tuple[float, ...],
    knockdown_vel: tuple[float, ...] | None = None,
    from_level: int = 5,
    push_term: str = "push_robot",
    force_term: str = "push_force",
    knockdown_term: str = "push_knockdown",
):
    """외란을 **레벨 5(90 deg 증명 관문) 도달 후에** 켠다.

    부트스트랩에 외란을 섞으면 안 된다. 레벨 0~1 에서는 아직 못 일어나
    바닥에 있는데 0.3~1.0 초마다 밀리면, 페널티로 다섯 판 연속 막혔던
    그 구간이 더 나빠진다.

    목적은 손상되지 않는다 — 레벨 5 에서도 매 에피소드가 supine 에서
    시작하므로, 그때부터 **일어나는 구간에** 밀기가 들어간다. 시작이
    iter 1,000 이후일 뿐이다.
    """
    level = int(getattr(env, "_getup_level", 0))
    if level < from_level:
        idx = 0
    else:
        since = getattr(env, "_getup_level_since", 0)
        dwell = env.common_step_counter - since
        idx = 0
        for i, s_ in enumerate(stage_steps):
            if dwell >= s_:
                idx = i + 1
    idx = min(idx, len(push_vel) - 1)

    v, f = push_vel[idx], ext_force[idx]
    cfg = env.event_manager.get_term_cfg(push_term)
    cfg.params["velocity_range"] = {"x": (-v, v), "y": (-v, v), "z": (-v, v)}
    env.event_manager.set_term_cfg(push_term, cfg)

    cfg = env.event_manager.get_term_cfg(force_term)
    cfg.params["force_range"] = (-f, f)
    env.event_manager.set_term_cfg(force_term, cfg)

    if knockdown_vel is not None:
        k = knockdown_vel[min(idx, len(knockdown_vel) - 1)]
        cfg = env.event_manager.get_term_cfg(knockdown_term)
        cfg.params["velocity_range"] = {"x": (-k, k), "y": (-k, k)}
        env.event_manager.set_term_cfg(knockdown_term, cfg)

    return torch.tensor(float(idx), dtype=torch.float32, device=env.device)


def penalty_rampin_curriculum(
    env,
    env_ids,
    stage_steps: tuple[int, ...],
    walk_pose_deadline: tuple[float, ...],
    torque_envelope: tuple[float, ...],
    joint_vel_limit: tuple[float, ...],
    base_height_well: tuple[float, ...],
    from_level: int = 4,
):
    """실기 이관용 페널티를 **과제를 다 배운 뒤에** 켠다.

    【왜 필요한가 — 실측 근거】
      walk_pose_deadline(-1.0) 과 torque_envelope(-0.02) 은 마지막으로
      처음부터 커리큘럼을 완주한 판(OK_16-28-53) **이후에** 추가됐고,
      검증은 전부 **이미 일어설 줄 아는 정책에 이어받아서** 했다.
      처음부터 학습에서는 한 번도 통과한 적이 없다.

      ★ 2026-08-30 확정 — 팔과 페널티 **둘 다** 영향을 준다 ★
          팔 2.967 + 페널티 없음  -> 레벨 1 정체
          팔 2.967 + 페널티 있음  -> 레벨 1 정체
          팔 1.5   + 페널티 있음  -> 레벨 1 정체 (02-54-38)
          팔 1.5   + 페널티 없음  -> **완주** 199/410/610/810 (02-03-05)
        레벨 1(35 deg)이 부트스트랩 병목이고, 거기서 페널티가 양수 보상을
        상쇄해 회복 탄력을 없앤다. 실측(레벨 1 정체 구간, 초당):
            walk_pose_deadline -0.696   torque_envelope   -0.447
            joint_vel_limit    -0.007   base_height_well  -0.003
        앞의 둘이 추가 부담의 99% 다.

      그리고 **이미 일어설 줄 아는 정책에는 페널티가 문제가 아니다**:
        19-35-26 은 이어받기 + 페널티 5 개 전량으로 커리큘럼을 최소 체류
        (+333/+666/+999/+1333) 로 완주했다. 그래서 레벨 4 도달 후에 켠다.

      판별 기록 (전부 처음부터 학습, 안정화항 weight 0):
          추가 페널티 없음  OK_13-59-11   전환 333/666/999/1333 -> lv4
          추가 페널티 없음  OK_16-28-53   전환 333/666/999/1333 -> lv4
          foot_flat 2 개만  00-11-57      전환 333/666          -> lv2 (정상,
                                          iter 733 에서 중단한 것뿐)
          5 개 전부         23-18-37      전환 333, 이후 lv1 에 954 까지 갇힘
          5 개 전부         00-07-54      전환 333, 이후 lv1 에 1500+ 갇힘
      전환 시점 333 = min_steps_per_level 이 허용하는 이론상 최속이다.

      기전: 난이도가 오르면 양수 항은 무너지는데 이 페널티들은 상수처럼
      남는다. 실측(레벨 1, iter 1019):
          양수 합 +3.41/초   음수 합 -2.76/초   순 +0.65/초
      이 중 walk_pose_deadline -0.58, torque_envelope -0.51 이다. 둘이
      양수 총합의 32% 를 먹어, "일어서려 애쓰기" 와 "가만히 있기" 의
      점수 차가 사라진다.

      특히 walk_pose_deadline 은 게이트가 **높이가 아니라 시간**(1.5 초)
      이라, 아직 바닥에 있어도 벌점이 켜진다. 기상이 1.0~1.2 초에 끝난다는
      실측을 전제로 잡은 값인데, 부트스트랩에서는 그 전제가 깨진다.

    【왜 지우지 않고 나중에 켜는가】
      둘 다 실기 이관에 필요하다. torque_envelope 은 XM540 의 N-T 곡선을
      넘지 않게 하고, walk_pose_deadline 은 walk_ready_pose 의 exp() 포화를
      메워 보행 인계 자세를 끝까지 밀어준다. 완성 정책에는 있어야 한다.

    【왜 레벨 4 도달 이후인가】
      역사적으로 성공한 조합이 정확히 그것이다 — 이미 lv4 인 정책에
      이어받아 이 항들을 켠 판(02-20-25, 19-35-26)은 전부 잘 됐다.
      레벨 2~3 에서 켜면 남은 단계 전진을 다시 막을 위험이 있다.

    Args:
        from_level: 이 레벨에 **도달한 뒤부터** 램프를 시작한다.
        stage_steps: 그 레벨에 머문 step 수 기준 단계 문턱.
            (env.common_step_counter 기준. 16384 env 에서 24 step/iter)

    Returns:
        현재 단계 인덱스 (로깅용)
    """
    level = int(getattr(env, "_getup_level", 0))
    if level < from_level:
        idx = 0
    else:
        since = getattr(env, "_getup_level_since", 0)
        dwell = env.common_step_counter - since
        idx = 0
        for i, s_ in enumerate(stage_steps):
            if dwell >= s_:
                idx = i + 1
    idx = min(idx, len(walk_pose_deadline) - 1)

    for name, table in (("walk_pose_deadline", walk_pose_deadline),
                        ("torque_envelope", torque_envelope),
                        ("joint_vel_limit", joint_vel_limit),
                        ("base_height_well", base_height_well)):
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.weight = table[idx]
        env.reward_manager.set_term_cfg(name, cfg)

    return torch.tensor(float(idx), dtype=torch.float32, device=env.device)


def regularization_curriculum(
    env,
    env_ids,
    stage_steps: tuple[int, ...],
    action_rate: tuple[float, ...],
    joint_acc: tuple[float, ...],
    start_stage: int = 0,
):
    """부드러움 페널티를 단계적으로 강하게 만든다.

    【왜 고정값이 아닌가】
      기상은 희소 보상 태스크라 초기엔 크게 휘저어야 해를 찾는다. 처음부터
      실기 수준(보행 -0.15 / 킥 -0.6)으로 조이면 탐색이 죽는다. 반대로
      약하게(-0.01) 두면 학습이 되더라도 실기에서 못 쓴다.

      실측 근거 — action_rate weight -0.01 로 296 iteration 돌린 결과
      raw action_rate_l2 = 30.1 이었다. 18 관절이므로 관절당 RMS 액션
      델타 1.29, × scale 0.25 = 0.32 rad/step, ÷ 0.02 s = **16 rad/s**.
      액추에이터 속도 한계 4.5 rad/s 의 3.5 배다. sim 에서는 PD 가
      포화돼 안 따라갈 뿐이지만 실기에서는 휘적거림이 된다.

      HumanUP 도 같은 이유로 regularization_scale_curriculum 을 쓴다
      (범위 [0.8, 2.0], 달성 높이 기준). 여기서는 예측 가능하도록
      step 기준 단계로 단순화했다.

    Args:
        start_stage: 이 단계에서 시작한다 (step 기준 계산과 max 를 취한다).
            **이미 유능한 정책에 이어받을 때** 쓴다. 이어받으면
            common_step_counter 가 0 부터 다시 시작해 부드러움 페널티가
            가장 약한 값으로 되돌아가는데, 실기 이관 단계에서는 그게
            역행이다. 2026-09-01 실기 관찰(기립 후 과격한 동작으로 수렴
            실패)에 따라 이어받기 판에서 2 로 둔다.

    Returns:
        현재 단계 인덱스 (로깅용)
    """
    step = env.common_step_counter
    level = start_stage
    for i, s_ in enumerate(stage_steps):
        if step >= s_:
            level = max(level, i + 1)
    level = min(level, len(action_rate) - 1)

    for name, table in (("action_rate", action_rate), ("joint_acc", joint_acc)):
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.weight = table[level]
        env.reward_manager.set_term_cfg(name, cfg)

    return torch.tensor(level, dtype=torch.float32, device=env.device)


def reverse_getup_curriculum(
    env,
    env_ids,
    fall_angles: tuple[float, ...],
    success_threshold: float = 0.5,
    hold_frac: float = 0.25,
    ema_alpha: float = 0.05,
    min_steps_per_level: int = 4800,
    term_name: str = "reset_base_supine",
    reward_name: str = "stand_success",
    band: bool = True,
):
    """역방향 커리큘럼 — 거의 서 있는 상태에서 시작해 점점 눕힌다.

    【왜 필요한가】
      이 태스크의 근본 난점은 정책이 **기립 상태를 한 번도 경험하지
      못한다**는 것이다. stand_success(weight 5.0, 기립 시 보상의 절반)가
      6 판 통틀어 한 번도 발화하지 않았다. 존재를 모르는 보상을 향해
      갈 수는 없다. 시작 자세를 목표 근처에 두고 점점 뒤로 물린다.

    ★ 첫 구현이 iteration 200 만에 level 0→4 로 뛰어 실패했다. 원인 둘 ★
      (1) EMA 가 평활을 못 했다.
          alpha = len(env_ids)/window 로 잡았는데, IsaacLab 은 에피소드가
          동기화돼 16384 개 env 가 한꺼번에 리셋된다. alpha 가 1.0 으로
          클램프되어 EMA 가 그냥 "직전 에피소드 성공률" 이 됐고, 에피소드
          마다(약 21 iteration) 한 단계씩 올라갔다.
          → 배치 크기와 무관한 고정 alpha 를 쓴다.
      (2) 초기 단계가 스폰만으로 충족됐다.
          15° 에서 정착하면 upright 0.977 / height_frac 1.0 이라
          stand_success 조건을 그대로 만족한다. "한 스텝이라도 발화" 를
          성공으로 세면 아무것도 배우지 않고 통과한다.
          → **유지 시간**을 본다. 에피소드 누적이 최대치의 hold_frac
            이상이어야 성공이다 (0.25 면 약 2.5 초 유지).

    【단계 전진은 반드시 성공률 기반이어야 한다】
      step 카운트로 올리면 못 배운 채 난이도만 올라가 무너진다. 역방향
      커리큘럼의 알려진 실패 모드다. 내려가지는 않는다.

    【성공 판정 근거】
      manager_based_rl_env.py::_reset_idx 에서
          curriculum_manager.compute (356행)  →  reward_manager.reset (375행)
      순으로 돌기 때문에, 이 함수가 실행되는 시점에는 방금 끝난
      에피소드의 reward_manager._episode_sums 가 아직 살아 있다.

    Args:
        hold_frac: 성공으로 인정할 유지 시간 비율. stand_success 누적이
            weight x episode_length_s 의 이 비율을 넘어야 한다.
        ema_alpha: 성공률 EMA 계수. 배치 크기와 무관한 고정값.
        min_steps_per_level: 한 단계에 최소 이만큼 머문다(common_step_counter).
            EMA 가 안정될 시간을 벌어 연쇄 급등을 막는다.
            4,800 step = 200 iteration (16384 env 에서 24 step/iter).
            2026-08-30 에 8,000(333 iter) 에서 낮췄다 — 5 단계 완주 최소치가
            1,333 -> 800 iteration 이 된다. 실측상 레벨 0 EMA 는 iter 200 에
            0.85 에 도달하므로 EMA 조건이 먼저 충족되는 구간은 아니다.

    Returns:
        dict — level, fall_angle(deg), success_rate(EMA)
    """
    sums = getattr(env.reward_manager, "_episode_sums", None)
    level = getattr(env, "_getup_level", 0)
    ema = getattr(env, "_getup_succ_ema", 0.0)
    since = getattr(env, "_getup_level_since", 0)

    if sums is not None and reward_name in sums and env_ids is not None and len(env_ids) > 0:
        # 최대 누적 = weight x 에피소드 길이(초). func 가 1 로 붙어 있을 때.
        w = env.reward_manager.get_term_cfg(reward_name).weight
        need = hold_frac * abs(w) * env.max_episode_length_s
        succ = (sums[reward_name][env_ids] >= need).float().mean().item()
        ema = (1.0 - ema_alpha) * ema + ema_alpha * succ

    dwelled = (env.common_step_counter - since) >= min_steps_per_level
    if ema >= success_threshold and dwelled and level < len(fall_angles) - 1:
        level += 1
        ema = 0.0                      # 새 단계에서 다시 증명해야 한다
        since = env.common_step_counter

    env._getup_level = level
    env._getup_succ_ema = ema
    env._getup_level_since = since

    cfg = env.event_manager.get_term_cfg(term_name)
    # ★ 2026-09-10 — 레벨별 **밴드** 를 준다 (스칼라 각도 -> [이전, 현재]) ★
    #   레벨 0 은 [0, fall_angles[0]]. 레벨 k 는 [fall_angles[k-1], fall_angles[k]].
    #   reset_root_supine 이 이 구간에서 |각도| 를 뽑고 부호를 랜덤으로 붙인다.
    #   균일 [-a, +a] 를 안 쓰는 이유는 그쪽 docstring 참조 (승격 판정이 물러진다).
    # ★ 2026-09-14 — band 인자 추가 ★
    #   V2 에서 밴드 샘플링(레벨 k = [fall_angles[k-1], fall_angles[k]])을
    #   넣으면서 이 함수가 v1 계보로도 흘러들어갔다. v1 원판은 **고정 각도**다.
    #   band=False 면 상·하한을 같게 줘 밴드 폭을 0 으로 만든다.
    cfg.params["fall_angle_deg"] = float(fall_angles[level])
    if band:
        cfg.params["fall_angle_min_deg"] = float(fall_angles[level - 1]) if level > 0 else 0.0
    else:
        cfg.params["fall_angle_min_deg"] = float(fall_angles[level])
    env.event_manager.set_term_cfg(term_name, cfg)

    return {
        "level": torch.tensor(float(level), device=env.device),
        "fall_angle": torch.tensor(float(fall_angles[level]), device=env.device),
        "success_rate": torch.tensor(float(ema), device=env.device),
    }


def supine_reset_curriculum(
    env,
    env_ids,
    stage_steps: tuple[int, ...],
    side_roll: tuple[float, ...],
    tilt: tuple[float, ...],
    joint_noise: tuple[float, ...],
    base_term: str = "reset_base_supine",
    joint_term: str = "reset_joint_noise",
):
    """common_step_counter 에 따라 리셋 랜덤화 폭을 넓힌다.

    좁게 시작해 넓히는 것만 된다. 넓게 시작하면 태스크 분포가 폭발해서
    1 단계 자체가 안 뚫리고, 그 뒤에 좁혀도 되돌아오지 않는다.

    yaw 와 xy 는 처음부터 전범위이므로 여기서 건드리지 않는다.

    Returns:
        현재 단계 인덱스 (로깅용)
    """
    step = env.common_step_counter
    level = 0
    for i, s in enumerate(stage_steps):
        if step >= s:
            level = i + 1
    level = min(level, len(side_roll) - 1)

    base_cfg = env.event_manager.get_term_cfg(base_term)
    base_cfg.params["side_roll_range"] = (-side_roll[level], side_roll[level])
    base_cfg.params["tilt_range"] = (-tilt[level], tilt[level])
    env.event_manager.set_term_cfg(base_term, base_cfg)

    joint_cfg = env.event_manager.get_term_cfg(joint_term)
    joint_cfg.params["position_range"] = (-joint_noise[level], joint_noise[level])
    env.event_manager.set_term_cfg(joint_term, joint_cfg)

    return torch.tensor(level, dtype=torch.float32, device=env.device)


# ═══════════════════════════════════════════════════════════════
#  모터 최대 토크 랜덤화 (2026-09-10 신설)
#  ─────────────────────────────────────────────────────────────
#  【왜】 sim 은 ImplicitActuator 로 effort_limit 5.0 N*m 를 **관절속도와
#    무관하게 평평하게** 준다. 실기 XM540-W150 은 N-T 곡선이 있다:
#        tau(w) = 6.9 * (1 - w/5.24),  캡 5.0
#        w=0 -> 5.0(캡)   w=2 -> 4.26   w=3 -> 3.35   w=4 -> 2.04
#    실측(v0_6 model_13500, probe_speed_limit.py): vx 0.7 에서 힙피치
#    |w| p95 = 4.40, |tau| p95 = 5.00 포화, **실기 N-T 위반 13.2 %**.
#
#  【DC 모터 모델을 왜 안 쓰는가 — 2026-08-29 검증 후 철회】
#    use_dc_motor=True 로 실기 곡선을 그대로 넣었더니 **15 도 균형 회복조차
#    학습이 안 됐다** (성공률 0.42 에서 2,000 iteration 정체. 평평한 모델은
#    iter 400 에 0.99). 그런데 보행/킥은 평평한 5.0 으로 학습해 실기에서
#    잘 동작한다 — 모델이 실기보다 비관적이라는 뜻이다. 다이나믹셀은 내부
#    제어가 도는 스마트 액추에이터에 기어비 152.3:1 이라 관절 단 특성이
#    순수 DC 모터 곡선과 다르다.  ([[mx3-dcmotor-model-rejected]])
#
#  【그래서 절충】 곡선을 강요하는 대신 **상한을 env 마다 흩뿌린다.**
#    정책이 "토크가 얼마나 있는지 모르는 상태" 에서도 되게 만든다.
#    scale_range 하한을 너무 낮추면 DC 모델과 같은 실패가 나므로 주의.
#    0.8 이면 4.0 N*m — 실기 w=2.20 근처의 토크다.
# ═══════════════════════════════════════════════════════════════
def randomize_effort_limit(
    env,
    env_ids,
    scale_range: tuple[float, float] = (0.8, 1.0),
    per_joint: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """관절 최대 토크(effort limit)를 기준값의 scale_range 배로 랜덤화한다.

    Args:
        scale_range: 기준 effort_limit 에 곱할 배율 범위.
        per_joint: True 면 관절마다 다른 배율(모터 개체차), False 면 로봇
            하나당 한 값. 실기에서는 개체차가 관절별로도 있지만, False 가
            신호가 깨끗하고 "이 로봇은 토크가 약하다" 를 배우기 쉽다.
        asset_cfg: joint_ids 를 주면 그 관절만. 기본은 전 관절.

    ※ 기준값은 최초 1 회 캐싱한다. 매번 현재값에 곱하면 배율이 누적돼
      토크가 0 으로 수렴한다 (실수하기 쉬운 지점).
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    key = "_getup_effort_limit_base"
    base = getattr(env, key, None)
    if base is None:
        base = asset.data.joint_effort_limits.clone()
        setattr(env, key, base)

    jids = asset_cfg.joint_ids
    if jids is None or isinstance(jids, slice):
        jids = slice(None)
        n_j = base.shape[1]
    else:
        n_j = len(jids)

    n = len(env_ids)
    shape = (n, n_j) if per_joint else (n, 1)
    scale = torch.empty(shape, device=env.device).uniform_(*scale_range)

    new_limits = base[env_ids][:, jids] * scale
    asset.write_joint_effort_limit_to_sim(
        new_limits, joint_ids=jids, env_ids=env_ids
    )


# ═══════════════════════════════════════════════════════════════
#  스폰 침투 보정 (2026-09-11 신설)
#  ─────────────────────────────────────────────────────────────
#  【증상】 리셋 시 로봇이 튀어오른다.
#  【원인】 스폰 높이는 base_link 기준 cos 보간이라 **다른 바디가 바닥을
#    뚫는 걸 못 본다.** side_roll/tilt(+-0.15)와 관절 랜덤화가 얹히면
#    어깨나 무릎이 예상보다 내려간다. PhysX 가 depenetration 임펄스로
#    밀어내면서 튄다 (max_depenetration_velocity=5.0).
#
#    실측 (probe_spawn_pen.py, 256 env, 스폰 직후 최저 바디 z):
#       각도  방향     최저바디z 범위      침투env%   0.5s내 |v|>2 비율
#        90  supine  -0.0250 ~ 0.0467       7.0%        0.0%
#        90  prone   -0.0251 ~ 0.0230       9.0%       44.9%
#        60  supine  -0.0158 ~ 0.0415       5.1%       26.6%
#        60  prone    0.0312 ~ 0.0861       0.0%        0.0%
#     최대 2.5 cm 파고든다. prone 90 도가 44.9 % 로 가장 심하다.
#     (45/30 도 supine 의 |v| 2.2 는 침투가 아니라 자유낙하다 —
#      스폰 0.34 에서 안착 0.059 까지 28 cm 떨어진다. 그건 정상)
#
#  ★ 반드시 **관절 리셋 다음에** 실행할 것 ★
#    이벤트는 선언 순서대로 돈다. reset_base_supine 시점에는 관절이 아직
#    직전 에피소드 값이라, 거기서 재면 엉뚱한 자세의 최저점을 잰다.
# ═══════════════════════════════════════════════════════════════
def lift_above_ground(
    env,
    env_ids,
    margin: float = 0.005,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """어떤 바디도 바닥 아래로 내려가지 않도록 루트를 +z 로 들어올린다.

    Args:
        margin: 최저 바디를 바닥에서 이만큼 띄운다 [m]. 접촉 지터 여유.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # 루트/관절 쓰기를 링크 변환에 반영시킨다. 이걸 안 하면 body_pos_w 가
    # 직전 스텝 값이라 보정이 무의미하다.
    env.sim.forward()

    bz = asset.data.body_pos_w[env_ids][:, :, 2]
    ground = env.scene.env_origins[env_ids][:, 2]
    lowest = bz.min(dim=1).values - ground
    lift = (margin - lowest).clamp(min=0.0)

    if float(lift.max()) <= 0.0:
        return

    pose = asset.data.root_state_w[env_ids][:, :7].clone()
    pose[:, 2] += lift
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)


def randomize_velocity_limit(
    env,
    env_ids,
    scale_range: tuple[float, float] = (0.78, 1.0),
    per_joint: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """관절 최대 속도(velocity limit)를 기준값의 scale_range 배로 랜덤화한다.

    【왜 토크만으로는 부족한가 — 실측】
      randomize_effort_limit 만 켰을 때 (model_34400, 90 도, 512 env):
        토크버킷[N*m]  관절속도 최대   평균    기상시간
          4.0~4.2         4.74       4.52     1.081 s
          4.4~4.6         4.93       4.52     1.008 s
          4.8~5.0         4.63       4.52     0.973 s
      **관절속도 평균이 전 버킷 4.52 로 동일**하다. effort_limit 과
      velocity_limit 은 ImplicitActuator 에서 **별개 파라미터**라 토크를
      낮춰도 속도 천장이 안 내려간다. 기상 시간만 11 % 늘 뿐이다.

      실기 XM540 은 tau 와 w 가 결합돼 있다 (tau = 6.9(1 - w/5.24)) —
      고속에서 토크가 0 으로 간다. 그 결합을 sim 에서 그대로 만들려면
      DC 모터 모델이 필요한데 2026-08-29 에 철회했다(학습 불가).

      대신 속도 상한도 독립적으로 흩뿌려 "느린 모터 / 약한 모터" 조합을
      다양하게 겪게 한다. 곡선은 아니지만 커버 범위가 넓어진다.

    ※ effort 쪽과 같은 이유로 기준값을 최초 1 회 캐싱한다.
      현재값에 곱하면 배율이 누적돼 속도가 0 으로 수렴한다.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    key = "_getup_velocity_limit_base"
    base = getattr(env, key, None)
    if base is None:
        base = asset.data.joint_vel_limits.clone()
        setattr(env, key, base)

    jids = asset_cfg.joint_ids
    if jids is None or isinstance(jids, slice):
        jids = slice(None)
        n_j = base.shape[1]
    else:
        n_j = len(jids)

    n = len(env_ids)
    shape = (n, n_j) if per_joint else (n, 1)
    scale = torch.empty(shape, device=env.device).uniform_(*scale_range)

    asset.write_joint_velocity_limit_to_sim(
        base[env_ids][:, jids] * scale, joint_ids=jids, env_ids=env_ids
    )


# ═══════════════════════════════════════════════════════════════
#  env 당 균일한 마찰 랜덤화 (2026-09-12 신설)
#  ─────────────────────────────────────────────────────────────
#  【왜 IsaacLab 기본 함수로 안 되는가】
#    randomize_rigid_body_material 은 버킷 인덱스를
#        (num_instances, max_num_shapes)
#    정수 텐서로 뽑는다. 즉 **한 로봇 안에서 형상 42 개가 각각 다른 마찰**
#    을 갖는다. make_consistent 는 동마찰<=정마찰만 보장할 뿐이다.
#
#    실측 (0.25~1.0 범위, 1024 env):
#        env 동마찰(42 형상 평균)  0.512 ~ 0.741   <- 0.25~1.0 이 아니다
#        한 env 안에서의 편차       0.2173
#        1024 env 중 994 개가 0.55~0.70 버킷에 몰림
#    발은 0.3, 손은 0.9 같은 식으로 섞여 평균이 항상 0.6 근처다.
#    **"미끄러운 바닥" 을 겪은 env 가 하나도 없다.** 마찰 하한을 0.7 ->
#    0.25 로 내려도 성능이 안 변한 이유가 이것이다.
#
#  【이 함수】 env 당 값 하나를 뽑아 **전 형상에 같은 값**을 준다.
#    그래야 "이 바닥은 미끄럽다" 가 표현된다.
#
#  ※ 지형 마찰은 1.0 이고 combine_mode="multiply" 라 여기 값이 곧 실효 마찰.
#  ※ material 쓰기는 CPU 텐서다. startup 모드 권장.
# ═══════════════════════════════════════════════════════════════
def randomize_actuator_limits_per_env(
    env,
    env_ids,
    effort_scale_range: tuple[float, float] = (0.80, 1.00),
    velocity_scale_range: tuple[float, float] = (0.78, 1.00),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """env 하나당 토크/속도 한계 배율을 한 쌍 뽑아 그 로봇의 전 관절에 적용한다.

    【왜 env 당 하나인가】
      randomize_rigid_body_material 이 형상마다 독립 추출이라 env 평균이
      1.0 으로 수렴해 버렸던 것과 같은 함정을 피한다. 관절마다 따로 뽑으면
      "로봇 전체가 약한 개체" 가 한 번도 안 나온다. 실기 개체차는 관절별
      랜덤이 아니라 **로봇 단위의 편차**에 가깝다.

    【왜 mode="startup" 인가】
      [[dr-intensity-action-rate-tradeoff]] — DR 의 비용은 "에피소드 내내
      상수인가" 로 갈린다. 매 에피소드 바뀌는 DR 은 정책을 반응형으로 만들어
      action_rate 를 올리고 실기에서 휘적거림이 된다. startup 이면 env 당
      한 번만 뽑혀 에피소드 내내 상수다 (마찰에서 쓴 것과 같은 방식).

    【왜 두 군데에 쓰는가】
      write_joint_*_limit_to_sim 은 PhysX 솔버 한계만 바꾸고 액추에이터
      모델 버퍼는 안 건드린다 (IsaacLab #128, 독스트링에 명시).
      ImplicitActuator 는 PhysX 가 토크를 계산하므로 **동작은 솔버 쪽이
      결정**하지만, 모델 버퍼를 안 맞추면 applied_torque 로그와 실제가
      어긋나고 torque_envelope 페널티가 틀린 값을 본다.
      DC 모터 시절 이걸 놓쳐 솔버 한계가 USD 기본값 10 억이 되어 액션이
      폭주한 전례가 있다 ([[mx3-dcmotor-model-rejected]]).

    기본 범위 근거 (XM540-W150):
        effort   0.80~1.00 -> 4.0~5.0 N.m
        velocity 0.78~1.00 -> 3.5~4.5 rad/s
      sim 은 속도와 무관하게 평평한 5.0 을 주는데 실기는 tau(w)=7.3(1-w/5.55)
      로 떨어진다. 기상 동작의 N-T 곡선 위반률은 7.6~8.4 % 로 보행 vx0.7
      (13.2 %, 실기 동작함) 보다 낮지만, 기상은 한 번에 밀어올리는 동작이라
      결정적 순간에 모자라면 아예 못 일어난다. 보험으로 넣는다.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    n = len(env_ids)
    dev = asset.device

    # ArticulationData 에는 default_joint_effort_limits 가 없다
    # (default_joint_pos_limits 만 있다). startup 은 한 번만 돌므로 그 시점
    # 값이 곧 cfg 값이지만, 두 번 불려도 배율이 누적되지 않도록 캐시한다.
    if not hasattr(env, "_getup_base_effort_limits"):
        env._getup_base_effort_limits = asset.data.joint_effort_limits.clone()
        env._getup_base_vel_limits = asset.data.joint_vel_limits.clone()
    eff0 = env._getup_base_effort_limits[env_ids].clone()            # [n, J]
    vel0 = env._getup_base_vel_limits[env_ids].clone()
    es = torch.empty(n, 1, device=dev).uniform_(*effort_scale_range)
    vs = torch.empty(n, 1, device=dev).uniform_(*velocity_scale_range)

    eff = eff0 * es
    vel = vel0 * vs
    asset.write_joint_effort_limit_to_sim(eff, env_ids=env_ids)
    asset.write_joint_velocity_limit_to_sim(vel, env_ids=env_ids)

    # 액추에이터 모델 버퍼도 맞춘다 (applied_torque 클립 / 로그 정합)
    #   ⚠ act.effort_limit 의 2 번째 축은 **그 그룹 안에서의 지역 인덱스**다
    #     (shape [num_envs, 그룹 관절 수]). 전역 관절 인덱스로 접근하면
    #     CUDA index out of bounds 가 난다. 오른쪽만 전역 인덱스로 뽑고
    #     왼쪽은 축 전체를 쓴다.
    for act in asset.actuators.values():
        ji = act.joint_indices                       # 전역 인덱스 (slice 또는 tensor)
        for attr in ("effort_limit", "velocity_limit"):
            buf = getattr(act, attr, None)
            if buf is None or not torch.is_tensor(buf):
                continue
            src = (eff if attr == "effort_limit" else vel)[:, ji]
            if src.shape[1] != buf.shape[1]:
                continue                             # 형상이 다르면 건드리지 않는다
            buf[env_ids, :] = src


def randomize_friction_per_env(
    env,
    env_ids,
    static_friction_range: tuple[float, float] = (0.3, 1.2),
    dynamic_friction_range: tuple[float, float] = (0.25, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """env 하나당 마찰 한 쌍을 뽑아 그 로봇의 전 형상에 적용한다."""
    asset = env.scene[asset_cfg.name]
    view = asset.root_physx_view
    mats = view.get_material_properties().clone()      # [N, S, 3] (cpu)
    n = mats.shape[0]

    st = torch.empty(n, 1).uniform_(*static_friction_range)
    dy = torch.empty(n, 1).uniform_(*dynamic_friction_range)
    dy = torch.minimum(dy, st)                          # 동마찰 <= 정마찰

    mats[:, :, 0] = st.expand(-1, mats.shape[1])
    mats[:, :, 1] = dy.expand(-1, mats.shape[1])

    ids = torch.arange(n, dtype=torch.int32)
    view.set_material_properties(mats, ids)

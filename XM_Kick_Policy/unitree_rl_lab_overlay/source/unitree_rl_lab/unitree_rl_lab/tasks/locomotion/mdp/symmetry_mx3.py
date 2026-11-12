# symmetry_mx3.py
# ============================================================================
# MX3 좌우 대칭(mirror) 매핑 — RSL-RL symmetry augmentation / mirror loss 용
# ----------------------------------------------------------------------------
# 목적
#   보행 정책이 좌우 비대칭 전략(한쪽 발만 쓰기, 정지 시 한 발만 앞으로 내기)
#   으로 수렴하는 것을 막는다. 대칭 상태에는 대칭 행동이 나오도록 강제한다.
#
#   기존 velocity_v0_2 는 pelvis_pitch_symmetry(-1.0) 와
#   stop_default_posture(-1.0) 로 같은 문제를 리워드에서 부분적으로 잡고
#   있었다. 이 모듈은 그것을 손실 함수 수준에서 근본적으로 처리한다.
#   (미러 손실이 잘 동작하면 위 두 항의 가중치를 낮출 수 있는지 실험 가능)
#
# ── 관절 부호 규약 (중요) ──────────────────────────────────────────────
#   이 USD 는 좌측 관절 축을 "반사상의 부호 반전"으로 정의한다.
#       a_L = -M_pseudo(a_R)
#   여기서 M_pseudo 는 시상면 반사 시 유사벡터(회전축) 변환:
#       M_pseudo(ax, ay, az) = (-ax, ay, -az)
#
#   pitch 축 (0,1,0) → M_pseudo → (0,1,0) → 부호반전 → (0,-1,0)  : 축 뒤집힘
#   roll  축 (1,0,0) → M_pseudo → (-1,0,0) → 부호반전 → (1,0,0)  : 축 그대로
#   yaw   축 (0,0,1) → M_pseudo → (0,0,-1) → 부호반전 → (0,0,1)  : 축 그대로
#
#   축이 뒤집힌 경우와 아닌 경우가 섞여 있지만, "관절값의 미러 부호"는
#   세 축 모두 -1 로 동일하다. (축 반전과 각도 반전이 상쇄/누적되는 방식이
#   서로 달라 결과적으로 같은 부호가 된다)
#
#   자산에서의 근거 — mx_bot.py / walk_kick target_limits 전부 일치:
#       init_joint_pos : r_pelvis_pitch  0.8   / l_pelvis_pitch  -0.8
#                        r_knee_pitch    1.4   / l_knee_pitch    -1.4
#       target_limits  : r_pelvis_roll (-0.52, +1.57)
#                        l_pelvis_roll (-1.57, +0.52)   ← 정확히 부호 반전
#                        r_ankle_pitch (-0.87, +1.92)
#                        l_ankle_pitch (-1.92, +0.87)   ← 정확히 부호 반전
#
#   ⇒ 12 개 다리 관절 전부 미러 부호 -1. 예외 없음.
#      check_symmetry_mx3.py 의 [검사 1] 이 이것을 자산 실측으로 검증한다.
#
# ── 좌우 스왑이 6칸 시프트인 이유 ──────────────────────────────────────
#   MX_LEG_JOINTS 가 r 6 개 → l 6 개 순서이고, 각 절반의 sub-order 가
#   (pelvis_pitch, pelvis_roll, pelvis_yaw, knee, ankle_pitch, ankle_roll)
#   로 동일하다. 따라서 좌우 스왑 = 인덱스 6 칸 시프트.
#   관측/행동 모두 preserve_order=True 이므로 이 순서가 그대로 유지된다.
#
# ── 위상(gait_phase) 반전 이유 ─────────────────────────────────────────
#   feet_gait_v01 의 offset=(0.0, 0.5) — 좌우 발이 반주기 엇갈린다.
#   좌우를 바꾼 상태는 시간을 반주기 이동한 상태와 같으므로 φ → φ + 0.5.
#       cos(2π(φ+0.5)) = -cos(2πφ)
#       sin(2π(φ+0.5)) = -sin(2πφ)
#   frequency 는 스칼라라 불변.
#   이 항을 빠뜨리면 "오른발이 스탠스여야 할 위상"에 미러된 왼발 상태를
#   짝지어 학습시키게 되어, 미러 손실이 보행을 오히려 망가뜨린다.
#
# ── RSL-RL 3.1.2 인터페이스 ────────────────────────────────────────────
#   rsl_rl/algorithms/ppo.py 에서 다음 두 형태로 호출된다.
#     (1) use_data_augmentation=True  → obs, actions 둘 다 전달
#     (2) use_mirror_loss=True 만     → obs 만 전달되는 호출 + actions 만
#                                        전달되는 호출이 각각 들어옴
#   따라서 obs=None, actions=None 양쪽을 모두 처리해야 한다.
#
#   반환은 [원본; 미러] 를 dim 0 으로 concat 한 것 (num_aug = 2).
#   ppo.py 가 num_aug = obs.batch_size[0] / original_batch_size 로
#   증강 배수를 역산하므로 이 규약을 지켜야 한다.
#
#   obs 는 TensorDict 이고 키가 관측 그룹명("policy", "critic")이다.
#   두 그룹 모두 미러링해야 한다 — use_data_augmentation=True 일 때
#   critic 도 증강된 obs 로 평가되기 때문이다.
# ============================================================================

from __future__ import annotations

import torch

# ------------------------------------------------------------------
# 다리 관절 미러 (12 DOF)
#   perm : r 6 개 ↔ l 6 개 스왑
#   sign : 전부 -1 (위 부호 규약 참조)
# ------------------------------------------------------------------
_N_LEG = 12
_HALF = _N_LEG // 2

_LEG_PERM: list[int] = list(range(_HALF, _N_LEG)) + list(range(0, _HALF))
_LEG_SIGN: list[float] = [-1.0] * _N_LEG


# ------------------------------------------------------------------
# 관측 term 별 미러 스펙
#   key   : ObservationManager 의 active_terms 에 나오는 term 이름
#   value : (base_dim, perm, sign)
#           base_dim = 히스토리 1 프레임당 차원
#           perm     = term 내부 인덱스 순열
#           sign     = perm 적용 "후" 각 차원에 곱할 부호
#
#   벡터/유사벡터 구분이 부호를 결정한다. 시상면 반사(y → -y) 기준:
#       진짜 벡터   (v)  : (vx,  vy, vz) → ( vx, -vy,  vz)
#       유사벡터    (ω)  : (ωx,  ωy, ωz) → (-ωx,  ωy, -ωz)
#   base_ang_vel 는 각속도라 유사벡터, projected_gravity 는 진짜 벡터다.
#
#   ※ 새 관측 term 을 추가하면 여기에 반드시 등록해야 한다.
#     등록되지 않은 term 이 있으면 _build_index_map 이 KeyError 로
#     즉시 실패한다 (조용히 틀리는 것보다 낫다).
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 관절 항은 **정적 스펙을 쓰지 않는다** (2026-09-04)
#   보행/킥은 12 관절(MX_LEG_JOINTS), 기상 V2 는 18 관절
#   (MX_LEG_JOINTS + MX_ARM_JOINTS) 을 쓴다. base_dim 을 하나로 박으면
#   한쪽이 반드시 깨진다:
#       joint_pos_rel 차원  보행 60 (12x5)  /  기상 90 (18x5)
#       60 % 18 != 0,  90 % 12 != 0
#
#   그래서 관측 term 이 실제로 쓰는 **관절 이름**에서 순열을 만든다.
#   r_ <-> l_ 접두사를 바꾼 짝을 찾는 방식이라 관절 집합이 무엇이든
#   맞고, 짝이 없으면 즉시 실패한다 (조용히 틀리는 것보다 낫다).
#
#   부호는 전 관절 -1 이다. 다리 12 개는 자산 리밋 대칭으로,
#   팔 6 개는 probe_mirror_sign_mx3.py 실측으로 확인했다
#   (어깨 pitch 는 리밋이 대칭이라 실측으로만 결정 가능했다).
_JOINT_TERMS = ("joint_pos_rel", "joint_vel_rel", "last_action")

# ★ 2026-09-13 — 바디 인덱스 항 ★
#   Getup-V1/V3 의 critic 관측 body_contact 는 **바디마다 0/1** 이고
#   body_names=".*" 이라 로봇 바디 수만큼 차원이 나온다. 미러하면 좌우
#   바디가 교환돼야 한다. 관절 항과 같은 이유로 정적 스펙에 담을 수 없어
#   (로봇마다 바디 수가 다르다) 접촉센서의 실제 바디 이름에서 동적 생성한다.
#
#   ⚠ 접촉센서의 body 순서는 articulation 과 다르다. 반드시 센서의
#     body_names 를 써야 한다. 섞어 쓰면 조용히 틀린 순열이 나온다.
_BODY_TERMS = ("body_contact",)


def _mirror_partner(name: str) -> str:
    """r_xxx <-> l_xxx (MX3), right_xxx <-> left_xxx (rs_bot). 좌우 접두사가 없으면 자기 자신(중앙 관절).

    rs_bot 도 MX3 와 같은 "좌우 완전 거울" 관절 규약이라(12 관절 전부 미러 부호 -1)
    접두사만 추가하면 이 모듈을 그대로 쓴다. MX3 이름은 left_/right_ 로 시작하지 않으므로 영향 없음.
    """
    for a, b in (("r_", "l_"), ("right_", "left_")):
        if name.startswith(a):
            return b + name[len(a):]
        if name.startswith(b):
            return a + name[len(b):]
    return name


def _body_spec_from_names(names: list[str]) -> tuple[int, list[int], list[float]]:
    """바디 이름 목록 -> (base_dim, perm, sign).

    접촉 여부는 스칼라 플래그라 부호는 전부 +1 이고, 좌우 바디만 교환한다.
    짝이 없는 바디(base_link, neck 등)는 자기 자신으로 남는다.
    """
    idx = {n: i for i, n in enumerate(names)}
    perm: list[int] = []
    for n in names:
        p = _mirror_partner(n)
        perm.append(idx.get(p, idx[n]))
    return len(names), perm, [1.0] * len(names)


def _obs_term_body_names(env, group: str, term: str) -> list[str]:
    """관측 term 이 쓰는 접촉센서의 바디 이름 목록을 얻는다."""
    om = _unwrap(env).observation_manager
    cfg = om._group_obs_term_cfgs[group][list(om.active_terms[group]).index(term)]
    sensor_cfg = cfg.params.get("sensor_cfg")
    if sensor_cfg is None:
        raise KeyError(
            f"[symmetry_mx3] term '{term}' 에 sensor_cfg 파라미터가 없어 "
            f"바디 이름을 알 수 없습니다."
        )
    sensor = _unwrap(env).scene.sensors[sensor_cfg.name]
    all_names = list(sensor.body_names)
    ids = sensor_cfg.body_ids
    if isinstance(ids, slice):
        return all_names
    return [all_names[i] for i in ids]


def _joint_spec_from_names(names: list[str]) -> tuple[int, list[int], list[float]]:
    """관절 이름 목록 -> (base_dim, perm, sign). 이름으로 좌우 짝을 찾는다."""
    idx = {n: i for i, n in enumerate(names)}
    perm: list[int] = []
    for n in names:
        p = _mirror_partner(n)
        if p not in idx:
            raise KeyError(
                f"[symmetry_mx3] 관절 '{n}' 의 미러 짝 '{p}' 가 관측 관절 목록에 "
                f"없습니다. 좌우가 짝을 이루지 않는 집합은 미러링할 수 없습니다.\n"
                f"  관절 목록: {names}"
            )
        perm.append(idx[p])
    return len(names), perm, [-1.0] * len(names)


def _action_joint_names(env) -> list[str]:
    """액션이 제어하는 관절 이름을 순서대로. last_action 의 레이아웃이 이것이다."""
    uw = _unwrap(env)
    robot = uw.scene["robot"]
    out: list[str] = []
    for tname in uw.action_manager.active_terms:
        term = uw.action_manager.get_term(tname)
        jids = getattr(term, "_joint_ids", None)
        if jids is None:
            continue
        if isinstance(jids, slice):
            jids = list(range(len(robot.data.joint_names)))[jids]
        out.extend(robot.data.joint_names[i] for i in jids)
    if not out:
        raise KeyError(
            "[symmetry_mx3] 액션 term 에서 관절 목록을 찾지 못했습니다. "
            "last_action 의 미러 순열을 만들 수 없습니다."
        )
    return out


def _obs_term_joint_names(env, group: str, term_name: str) -> list[str]:
    """관측 term 이 실제로 쓰는 관절 이름을 순서대로 돌려준다.

    last_action 은 관측이 아니라 **액션 버퍼**라 asset_cfg 가 없다.
    이 항의 레이아웃은 액션 term 의 관절 순서와 같으므로 거기서 가져온다.
    """
    if term_name == "last_action":
        return _action_joint_names(env)

    om = _unwrap(env).observation_manager
    robot = _unwrap(env).scene["robot"]
    names = list(om.active_terms[group])
    cfgs = om._group_obs_term_cfgs[group]
    cfg = cfgs[names.index(term_name)]
    asset_cfg = cfg.params.get("asset_cfg", None)
    if asset_cfg is None or getattr(asset_cfg, "joint_ids", None) is None:
        # asset_cfg 가 없으면 액션 관절 순서를 쓴다 (기본 관절 전체와 동일한 경우)
        return _action_joint_names(env)
    jids = asset_cfg.joint_ids
    if isinstance(jids, slice):
        jids = list(range(len(robot.data.joint_names)))[jids]
    return [robot.data.joint_names[i] for i in jids]


_MIRROR_SPEC: dict[str, tuple[int, list[int], list[float]]] = {
    # -- 몸통 상태 --
    # 각속도: roll rate / yaw rate 반전, pitch rate 유지
    "base_ang_vel": (3, [0, 1, 2], [-1.0, +1.0, -1.0]),
    # 선속도(critic 전용): y 성분만 반전
    "base_lin_vel": (3, [0, 1, 2], [+1.0, -1.0, +1.0]),
    # 몸통 프레임 중력 방향: y 성분만 반전
    "projected_gravity": (3, [0, 1, 2], [+1.0, -1.0, +1.0]),

    # -- 명령 --
    # (vx, vy, wz): 횡속도와 요레이트 반전
    "velocity_commands": (3, [0, 1, 2], [+1.0, -1.0, -1.0]),
    # (cos, sin, freq/scale): 반주기 시프트 → cos/sin 동시 반전, freq 불변
    "gait_phase": (3, [0, 1, 2], [-1.0, -1.0, +1.0]),

    # -- 관절 --
    #   ★ 여기 없다. _JOINT_TERMS 로 분류돼 _build_index_map 에서
    #     관측 term 의 실제 관절 이름으로부터 동적 생성한다.
    #     (보행 12 / 기상 18 을 하나의 정적 스펙으로 못 담기 때문)

    # -- 기상 critic 특권 관측 (Getup-V1/V3) --
    # 몸통 높이 (스칼라) : 좌우 반사와 무관하므로 불변
    "base_height": (1, [0], [+1.0]),
    # 양발 합산 하중 비율 (스칼라) : 좌우를 합친 값이라 불변
    "feet_load": (1, [0], [+1.0]),
    #   ※ body_contact 는 _BODY_TERMS 로 분류돼 동적 생성된다.

    # -- 킥 태스크용 (Stage A 이후 사용, 지금은 미사용) --
    # 로봇 프레임 공 위치 (x, y) : y 반전
    "ball_pos_b": (2, [0, 1], [+1.0, -1.0]),
    # 로봇 프레임 공 속도 (x, y) : y 반전
    "ball_vel_b": (2, [0, 1], [+1.0, -1.0]),
    # 목표 방향 단위벡터 (cos, sin) : sin 반전
    "target_dir_b": (2, [0, 1], [+1.0, -1.0]),
    # 목표 거리 정규화 (스칼라) : 불변
    "target_dist_norm": (1, [0], [+1.0]),
    # 킥 세기 요청 (상수 1.0, 스칼라) : 불변 — walk_kick_v1
    "kick_strength": (1, [0], [+1.0]),

    # -- 백덤블링 (RS02 Backflip-V0-4, 11-08) — 뒤로 도는 회전은 좌우 거울에 불변 --
    # 시각 (t/total, 기준 회전/2π)
    "phase": (2, [0, 1], [+1.0, +1.0]),
    # 뒤로 돈 각 / 2π
    "flip_progress": (1, [0], [+1.0]),
    # 보조 배율 (critic)
    "assist": (1, [0], [+1.0]),
}


# ------------------------------------------------------------------
# 인덱스 맵 캐시
#   그룹당 한 번만 만들고 재사용한다. 매 미니배치마다 재구성하면
#   학습 속도에 영향이 있다.
#   키: (group_name, total_dim, device_str)
# ------------------------------------------------------------------
_INDEX_CACHE: dict[tuple[str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _unwrap(env):
    """RslRlVecEnvWrapper 등으로 감싸여 있어도 ManagerBasedRLEnv 를 얻는다."""
    return getattr(env, "unwrapped", env)


def _build_index_map(
    env, group: str, total_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """관측 그룹 하나에 대한 (평탄화 인덱스 순열, 부호 벡터) 를 만든다.

    IsaacLab ObservationManager 는 term 별로 히스토리 버퍼를 갖고,
    각 term 을 (N, history, dim) → (N, history*dim) 으로 평탄화한 뒤
    term 순서대로 concat 한다. 즉 전체 레이아웃은

        [term1 히스토리 전체][term2 히스토리 전체]...

    이고, term 내부는 (history, dim) 행우선 = 과거 프레임부터 순서대로다.
    따라서 term 별 (perm, sign) 을 히스토리 횟수만큼 반복하면 된다.

    ※ 이 레이아웃 가정은 check_symmetry_mx3.py 의 [검사 3] 이
      학습된 정책의 미러 잔차로 실증 검증한다. 가정이 틀리면
      잔차가 크게 나와서 바로 드러난다.
    """
    om = _unwrap(env).observation_manager

    term_names = list(om.active_terms[group])
    term_dims = list(om.group_obs_term_dim[group])

    perm: list[int] = []
    sign: list[float] = []
    offset = 0

    for name, shape in zip(term_names, term_dims):
        # shape 는 (dim,) 또는 (history*dim,) 형태의 튜플
        term_total = 1
        for s in shape:
            term_total *= int(s)

        if name in _JOINT_TERMS:
            # 관절 항: 관측 term 이 쓰는 관절 이름에서 순열을 만든다.
            jnames = _obs_term_joint_names(env, group, name)
            base_dim, term_perm, term_sign = _joint_spec_from_names(jnames)
        elif name in _BODY_TERMS:
            # 바디 항: 접촉센서의 실제 바디 이름에서 순열을 만든다.
            bnames = _obs_term_body_names(env, group, name)
            base_dim, term_perm, term_sign = _body_spec_from_names(bnames)
        elif name not in _MIRROR_SPEC:
            raise KeyError(
                f"[symmetry_mx3] 관측 term '{name}' (그룹 '{group}') 의 "
                f"미러 스펙이 없습니다. _MIRROR_SPEC 에 추가하세요."
            )
        else:
            base_dim, term_perm, term_sign = _MIRROR_SPEC[name]

        if term_total % base_dim != 0:
            raise ValueError(
                f"[symmetry_mx3] term '{name}' 차원 {term_total} 이 "
                f"base_dim {base_dim} 의 배수가 아닙니다. "
                f"_MIRROR_SPEC 의 base_dim 을 확인하세요."
            )

        history = term_total // base_dim

        # 히스토리 프레임 수만큼 같은 패턴을 반복
        for k in range(history):
            frame_offset = offset + k * base_dim
            perm.extend(frame_offset + j for j in term_perm)
            sign.extend(term_sign)

        offset += term_total

    if offset != total_dim:
        raise ValueError(
            f"[symmetry_mx3] 그룹 '{group}' term 차원 합계 {offset} 이 "
            f"실제 관측 차원 {total_dim} 과 다릅니다."
        )

    perm_t = torch.tensor(perm, dtype=torch.long, device=device)
    sign_t = torch.tensor(sign, dtype=torch.float, device=device)
    return perm_t, sign_t


def _get_index_map(
    env, group: str, total_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (group, total_dim, str(device))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = _build_index_map(env, group, total_dim, device)
    return _INDEX_CACHE[key]


def mirror_obs_group(env, group: str, tensor: torch.Tensor) -> torch.Tensor:
    """관측 그룹 텐서 (N, D) 하나를 미러링한다."""
    perm, sign = _get_index_map(env, group, tensor.shape[-1], tensor.device)
    return tensor[..., perm] * sign


# 액션 미러 스펙 캐시 — 차원(관절 수)별로 보관한다.
#   보행/킥 12, 기상 V2 18. mirror_actions 는 env 인자 없이도 불리므로
#   (check_symmetry_mx3 의 검사 2 등) 차원을 키로 캐싱한다.
_ACT_SPEC_CACHE: dict[int, tuple[list[int], list[float]]] = {}


def register_action_spec(env) -> None:
    """env 의 액션 관절 순서로부터 미러 스펙을 만들어 캐시에 넣는다.

    compute_symmetric_states 가 매번 호출하므로 학습 경로에서는 자동이다.
    env 없이 mirror_actions 를 먼저 부르는 경로(검증 스크립트)를 위해
    공개 함수로 둔다.
    """
    names = _action_joint_names(env)
    n, perm, sign = _joint_spec_from_names(names)
    _ACT_SPEC_CACHE[n] = (perm, sign)


def mirror_actions(actions: torch.Tensor, env=None) -> torch.Tensor:
    """행동 (N, J) 을 미러링한다.

    행동은 관절 목표 오프셋이라 관측의 joint_pos_rel 과 같은 규약을 쓴다.
    히스토리가 없으므로 순열/부호를 직접 적용한다.

    J 는 태스크마다 다르다 (보행/킥 12, 기상 V2 18). env 가 주어지면
    거기서 관절 이름을 읽어 스펙을 만들고, 아니면 캐시를 쓴다.
    """
    n = actions.shape[-1]
    if env is not None and n not in _ACT_SPEC_CACHE:
        register_action_spec(env)
    if n not in _ACT_SPEC_CACHE:
        raise ValueError(
            f"[symmetry_mx3] 행동 차원 {n} 에 대한 미러 스펙이 없습니다. "
            f"register_action_spec(env) 를 먼저 부르거나 "
            f"mirror_actions(actions, env=env) 로 호출하세요."
        )
    p, sg = _ACT_SPEC_CACHE[n]
    perm = torch.tensor(p, dtype=torch.long, device=actions.device)
    sign = torch.tensor(sg, dtype=torch.float, device=actions.device)
    return actions[..., perm] * sign


def compute_symmetric_states(env, obs=None, actions=None, obs_type=None):
    """RSL-RL symmetry 콜백. 2.3.x / 3.x 양쪽 호환.

    Args:
        env:      VecEnv (RslRlVecEnvWrapper). 관측 레이아웃 조회에만 사용.
        obs:      Tensor (2.3.x) | TensorDict (3.x) | None
        actions:  Tensor (N, 12) | None
        obs_type: "policy" | "critic" (2.3.x 만 전달). 3.x 는 None.

    Returns:
        (obs_out, actions_out) — 각각 [원본; 미러] 를 dim 0 으로 concat.
        입력이 None 이면 해당 출력도 None.

    ── 버전별 호출 방식 ────────────────────────────────────────────
      rsl_rl 2.3.x — obs 가 평범한 Tensor 이고 그룹별로 따로 호출된다.
          func(obs=obs_batch,        actions=..., env=..., obs_type="policy")
          func(obs=critic_obs_batch, actions=None, env=..., obs_type="critic")

      rsl_rl 3.x — obs 가 TensorDict 하나에 전 그룹이 담겨 온다.
          func(obs=obs_batch, actions=..., env=...)

      obs_type 을 안 받으면 2.3.x 에서 TypeError 가 난다. 검사 스크립트는
      mirror_obs_group / mirror_actions 를 직접 부르므로 이 경로를 타지
      않아, 검증이 통과해도 학습에서 터질 수 있다. 주의.

    ※ ppo.py 가 num_aug = out.shape[0] / original_batch_size 로 증강
      배수를 역산하므로, 반드시 원본을 앞에 두고 2 배 길이로 반환해야
      한다. 순서를 바꾸면 미러 손실이 엉뚱한 쌍을 비교한다.
    """
    obs_out = None
    if obs is not None:
        if isinstance(obs, torch.Tensor):
            # -- rsl_rl 2.3.x 경로 --
            group = obs_type if obs_type is not None else "policy"
            obs_out = torch.cat(
                [obs, mirror_obs_group(env, group, obs)], dim=0
            )
        else:
            # -- rsl_rl 3.x 경로 (TensorDict) --
            # 원본을 in-place 로 고치지 않고 새로 만든다.
            # (롤아웃 스토리지가 오염될 수 있음)
            from tensordict import TensorDict

            num_orig = int(obs.batch_size[0])
            data = {}
            for group in obs.keys():
                tensor = obs[group]
                data[group] = torch.cat(
                    [tensor, mirror_obs_group(env, group, tensor)], dim=0
                )
            obs_out = TensorDict(
                data, batch_size=[2 * num_orig], device=obs.device
            )

    actions_out = None
    if actions is not None:
        actions_out = torch.cat([actions, mirror_actions(actions, env=env)], dim=0)

    return obs_out, actions_out

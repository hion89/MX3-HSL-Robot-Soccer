# unitree_rl_lab/assets/robots/mx_bot.py

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from pathlib import Path

from unitree_rl_lab.utils import joint_friction_compat

# 아래 friction=1.1 은 Isaac Sim 4.5 의 관절 마찰 "계수" (하중 비례) 기준 값이다. Isaac Sim 5.x 는 기본으로
# 이 값을 정지 마찰 1.1 N·m 로만 넣어 움직이는 관절 마찰이 0 이 되므로, 5.x 에서는 4.5 계수 모델로 되돌린다.
joint_friction_compat.apply()

# ── repo root ────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[5]

# ╔══════════════════════════════════════════════════════════════╗
# ║  ★ 여기만 바꾸면 로봇 전체 설정이 전환됩니다 ★               ║
# ╠══════════════════════════════════════════════════════════════╣
ACTIVE_ROBOT = "XM_robot04"   # ← "mx_robot02" / "mx_robot03" 등
# ╚══════════════════════════════════════════════════════════════╝


# ── 공통 joint 이름 (모든 로봇이 같은 12-DOF 다리 구조라면 공유) ──
MX_LEG_JOINTS = [
    "r_pelvis_pitch_joint", "r_pelvis_roll_joint", "r_pelvis_yaw_joint",
    "r_knee_pitch_joint",   "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_pelvis_pitch_joint", "l_pelvis_roll_joint", "l_pelvis_yaw_joint",
    "l_knee_pitch_joint",   "l_ankle_pitch_joint", "l_ankle_roll_joint",
]

MX_ARM_JOINTS = [
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_elbow_pitch_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_elbow_pitch_joint",
]

# 목 관절 (학습 미사용 — 기본자세 0으로 잠금 처리)
MX_NECK_JOINTS = [
    "neck_yaw_joint", "neck_pitch_joint",
]

# ═══════════════════════════════════════════════════════════════
#  로봇별 설정 레지스트리
#  - 새 로봇을 추가하려면 ROBOT_CONFIGS에 키를 하나 더 넣으면 끝
#  - 한 로봇의 "태스크별 변형"도 여기에 항목으로 넣는다. build_robot_cfg()
#    로 항목마다 독립된 cfg 를 만들 수 있으므로, ACTIVE_ROBOT 을 바꾸거나
#    파일을 복사할 필요가 없다.
#
#  【항목 키 레퍼런스】
#    usd_path          : str, 필수
#    init_pos          : (x, y, z), 필수 — 스폰 위치
#    init_rot          : (w, x, y, z), 선택 — 스폰 자세. 생략하면 직립
#                        (1,0,0,0). pitch +90°=엎드림, -90°=등 대고 누움
#    init_joint_pos    : dict, 필수 — 관절 기본각
#                        ★ 이 값은 세 가지로 동시에 쓰인다:
#                          (1) 액션 원점 (target = default + action*scale)
#                          (2) reset_joints_by_scale 의 기준 자세
#                          (3) default_joint_pos 를 읽는 리워드의 목표 자세
#                        태스크별로 (1)과 (3)이 충돌할 수 있으니 주의.
#    self_collisions   : bool, 기본 False
#    actuator_per_type : bool — True 면 _PER_TYPE_MAP 6그룹, False 면
#                        pelvis/knee/ankle 3그룹
#    <그룹명>          : dict(effort, vel, stiffness, damping, armature,
#                        friction) — per_type 이면 pelvis_pitch/knee_pitch/
#                        ankle_pitch/pelvis_roll/ankle_roll/pelvis_yaw
#    arm / neck        : dict, 선택 — 있을 때만 해당 그룹이 생긴다
#    leg_joints        : list|None — None 이면 MX_LEG_JOINTS
#    joint_sdk_names   : list, 선택 — deploy export 관절 순서.
#                        생략하면 다리 관절만. 팔까지 쓰는 태스크는 명시
#    inherit           : str, 선택 — 다른 항목 키를 상속한다.
#                        ★ 최상위 키 단위 얕은 덮어쓰기. 자식이 dict 를
#                        주면 부모 dict 는 통째로 버려진다(부분 merge 아님)
# ═══════════════════════════════════════════════════════════════
ROBOT_CONFIGS = {

    # ──────────────── XM_robot04 ──────────────
    # 관절 타입별 분리 튜닝 (L/R 동일 파라미터)
    "XM_robot04": dict(
        usd_path=str(_REPO_ROOT / "robotmodels/mx_bot/usd/XM_robot04/XM_robot04.usd"),
        self_collisions=True ,  # ← 추가
        init_pos=(0.0, 0.0, 0.44),
        init_joint_pos={
            "r_pelvis_pitch_joint":  0.62,
            "r_knee_pitch_joint":    1.2,
            "r_ankle_pitch_joint":   0.62,
            "l_pelvis_pitch_joint": -0.62,
            "l_knee_pitch_joint":   -1.2,
            "l_ankle_pitch_joint":  -0.62,
            # 팔꿈치 접힌 상태
            "r_elbow_pitch_joint":   2.967,   # 170°
            "l_elbow_pitch_joint":  -2.967,   # -170°
            # 목 기본자세 0 (잠금)
            "neck_yaw_joint":        0.0,
            "neck_pitch_joint":      0.0,
        },

        # ★ 관절 타입별 분리 튜닝 활성화 ★
        actuator_per_type=True,

        # pitch 관절 (주 운동축 — 가장 큰 토크/움직임)
        pelvis_pitch=dict(effort=5.0, vel=4.5, stiffness=50.0, damping=2.5, armature=0.05, friction=1.1),
        knee_pitch  =dict(effort=5.0, vel=4.5, stiffness=50.0, damping=2.5, armature=0.05, friction=1.1),
        ankle_pitch =dict(effort=5.0, vel=4.5, stiffness=50.0, damping=2.5, armature=0.05, friction=1.1),

        # roll 관절 (좌우 균형)
        pelvis_roll =dict(effort=5.0, vel=4.5, stiffness=50.0, damping=2.5, armature=0.05, friction=1.1),
        ankle_roll  =dict(effort=5.0, vel=4.5, stiffness=50.0, damping=2.5, armature=0.05, friction=1.1),

        # yaw 관절 (회전)
        pelvis_yaw  =dict(effort=5.0, vel=4.5, stiffness=50.0, damping=2.5, armature=0.05, friction=1.1),

        # 팔 고정용 (약한 stiffness로 접힌 상태 유지)
        arm = dict(effort=5.0, vel=4.5, stiffness=0.0, damping=50, armature=0.05, friction=1.1),

        # 목 고정용 (학습 미사용 — 팔과 동일, 기본자세 0 유지)
        neck = dict(effort=5.0, vel=4.5, stiffness=0.0, damping=50, armature=0.05, friction=1.1),

        leg_joints=None,
    ),

    # ──────────── XM_robot04_getup ────────────
    # 등을 대고 누운 상태에서 일어나기(supine get-up) 전용.
    # XM_robot04 를 상속하므로 USD / 다리 게인 / 기본각이 전부 따라온다.
    # 보행·킥 태스크는 XM_robot04 를 그대로 쓰므로 영향받지 않는다.
    "XM_robot04_getup": dict(
        inherit="XM_robot04",

        # ── 팔꿈치 원점을 ±1.5 로 되돌린다 (2026-08-30, 대조 실험) ──
        #   보행 기본각(±2.967)을 상속하면 **레벨 0(15 deg)부터** 학습이
        #   절반밖에 안 된다. 실측 (iter 250, 레벨 0, 초당 값):
        #                      00-11-57(팔 1.5)   팔 2.967
        #       stand_success        4.535          2.173
        #       upright              2.332          1.150
        #       base_height          2.822          1.828
        #       성공률 EMA           0.959          0.430
        #   격차 4.69 의 99.8% 가 **양수 항**에서 나온다. 페널티 탓이 아니다.
        #   실제로 페널티 4 개를 끄고 joint_acc 도 되돌린 판(01-42-44)이
        #   레벨 1 에서 성공률 0.444 -> 0.004 로 오히려 악화됐다.
        #
        #   ★ "어깨가 움직이니 팔은 문제 없다" 는 판단이 틀렸다 ★
        #     어깨 dev_max 51.8 deg 로 분명히 움직인다. 하지만 팔꿈치가
        #     170 deg 접히면 팔이 몸통 앞에 뭉쳐 무게중심이 앞으로 쏠리고
        #     균형에 쓸 관성 팔길이도 짧아진다. 움직이는 것과 유효한 것은
        #     다르다.
        #
        #   【실기에서의 대가】
        #     넘어지면 팔은 damping=50 이 잡아 접힌 채(±2.967)다. 이 원점을
        #     쓰려면 기상 직전에 팔을 ±1.5 로 세팅하는 절차가 필요하다.
        #     성능과 실기 편의의 맞교환이며, 이번 판으로 그 대가가 얼마인지
        #     확정한다.
        #   ⚠ 병합은 최상위 얕은 덮어쓰기다. init_joint_pos 를 주면 부모의
        #     것이 통째로 버려지므로 **전체를 다시 적는다** (다리·목 포함).
        init_joint_pos={
            "r_pelvis_pitch_joint":  0.62,
            "r_knee_pitch_joint":    1.2,
            "r_ankle_pitch_joint":   0.62,
            "l_pelvis_pitch_joint": -0.62,
            "l_knee_pitch_joint":   -1.2,
            "l_ankle_pitch_joint":  -0.62,
            # ★ 2026-09-15 — 팔 기본자세 변경 (사용자 지정) ★
            #   기존: 어깨 0 / 팔꿈치 ±1.5 (리밋 중점)
            #   변경: 어깨피치 ∓0.3 / 팔꿈치 ±1.0
            #   use_default_offset=True 이므로 이 값이 **액션의 원점**이고,
            #   동시에 스폰 자세다. _WALK_READY_POSE(종료 목표)도 같은 값으로
            #   맞춘다 — 그쪽은 getup_v1_2_env_cfg.py 에서 덮어쓴다.
            #   ⚠ 팔 자세가 바뀌면 누웠을 때 안착 높이가 달라진다.
            #     init_pos / _SUPINE_HEIGHT / _PRONE_HEIGHT 를 다시 잴 것.
            "r_shoulder_pitch_joint": -0.3,
            "r_shoulder_roll_joint":   0.0,
            "r_elbow_pitch_joint":     1.0,
            "l_shoulder_pitch_joint":  0.3,
            "l_shoulder_roll_joint":   0.0,
            "l_elbow_pitch_joint":    -1.0,
            "neck_yaw_joint":        0.0,
            "neck_pitch_joint":      0.0,
        },

        # 스폰 자세 — 등이 바닥.
        #   pitch −90° 이면 x_body(정면)가 위를 향하고 등이 아래로 간다.
        #   (+90° 는 엎드림 — 별도 태스크로 다룰 것. supine 과 섞어
        #    학습하면 롤아웃 간섭으로 성능이 나빠진다는 보고가 있다)
        #   quat(w,x,y,z) = (cos(−45°), 0, sin(−45°), 0)
        init_rot=(0.70711, 0.0, -0.70711, 0.0),

        # 등을 대고 누웠을 때 base_link 높이 — 실측값 (2026-08-28).
        #   scripts/rsl_rl/check_getup_spawn.py 로 0.15 에서 떨어뜨려 정착시킨
        #   결과. 보행 기본각(무릎 굽힘)이라 등과 발이 함께 바닥에 닿는다.
        #   projected_gravity=(-1.000, 0.000, +0.027) → supine 확인.
        #   기본각을 바꾸면 이 값도 다시 재야 한다.
        init_pos=(0.0, 0.0, 0.056),

        # ── DC 모터 모델은 쓰지 않는다 (2026-08-29 검증 후 철회) ──
        #   실기 토크 곡선을 넣으려고 use_dc_motor=True 를 켰다가 되돌렸다.
        #
        #   【왜 켰나】 sim 은 ω 무관 5.0 N·m 평평, 실기 XM540 스펙은
        #     τ=6.9(1-ω/5.24) 라 ω=4.57 에서 1.29 뿐이다. 기상이 그 구간을 쓴다.
        #
        #   【왜 되돌렸나】
        #     (1) DC 로는 15° 균형 회복조차 학습이 안 됐다. 성공률이 0.42 에서
        #         2,000 iteration 정체 (평평한 모델은 iter 400 에 0.99).
        #     (2) 그런데 **보행·킥은 평평한 5.0 으로 학습해 실기에서 잘 동작한다.**
        #         모델이 실기보다 비관적이라는 뜻이다.
        #     (3) 다이나믹셀은 내부 제어가 도는 스마트 액추에이터에 기어비
        #         152.3:1 이라, 관절 단 특성이 순수 DC 모터 곡선과 다르다.
        #
        #   ※ 되살리려면 use_dc_motor=True 한 줄이면 된다. 그때는 _make_actuator
        #     의 *_sim 인자도 반드시 함께 확인할 것 (물리 솔버 한계가 USD
        #     기본값 10 억으로 뚫려 액션이 폭주했던 전례가 있다).

        # 팔 활성화 — 기상에는 팔이 필수다.
        # 다리와 동일한 게인. (보행용 XM_robot04 는 stiffness=0, damping=50)
        arm=dict(effort=5.0, vel=4.5, stiffness=50.0, damping=2.5,
                 armature=0.05, friction=1.1),

        # deploy export 관절 순서 — 다리 12 + 팔 6 = 18.
        # 보행 정책(12)과 다르므로 실기 전환 시 게인 세트도 같이 바꿔야 한다.
        joint_sdk_names=MX_LEG_JOINTS + MX_ARM_JOINTS,
    ),

    # ──────── XM_robot04_convexhull (2026-10-17) ────────
    # 충돌 모델만 바꾼 USD. 수동 캡슐 대신 URDF 메시를 Isaac 자동 변환(링크당
    # convex hull)으로 만들고, 거짓 충돌 쌍(무릎↔발, 몸통↔골반 roll, 좌우)과
    # 부모-자식 쌍을 USD 에 충돌 제외(FilteredPairsAPI)로 기록했다.
    #   - 질량·관성·관절 이름·한계는 XM_robot04 와 같다 (neck_pitch 한계만 −60°, 기존 −50°)
    #   - 발 충돌 형상이 수동 캡슐 11 개 → 발 메시 convex hull 로 바뀌었다
    #   - 생성 도구 / 검증 기록: ~/Documents/MX3_collision_autofit (README)
    #   ⚠ Isaac Sim GUI 의 Physics Inspector 는 충돌 제외를 적용하지 않는 것으로 보인다
    #     (학습·스크립트 시뮬레이션에서는 적용됨, 실측 확인).
    "XM_robot04_convexhull": dict(
        inherit="XM_robot04",
        usd_path=str(_REPO_ROOT / "robotmodels/mx_bot/usd/XM_robot04_convexhull/XM_robot04_convexhull.usd"),
    ),
    "XM_robot04_getup_convexhull": dict(
        inherit="XM_robot04_getup",
        usd_path=str(_REPO_ROOT / "robotmodels/mx_bot/usd/XM_robot04_convexhull/XM_robot04_convexhull.usd"),
    ),

    # 필요하면 여기에 "mx_robot04": dict(...) 계속 추가
}


# ═══════════════════════════════════════════════════════════════
#  아래부터는 수정할 필요 없음 — build_robot_cfg() 가 자동 구성
# ═══════════════════════════════════════════════════════════════


@configclass
class MXArticulationCfg(ArticulationCfg):
    """MX robot articulation config with Unitree-style deployment metadata."""
    joint_sdk_names: list[str] | None = None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ ★ 액추에이터 모델 선택 (2026-08-19) — 킥 sim2real 갭의 핵심                ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║ USE_DC_MOTOR = False (기본, 기존과 100% 동일)                              ║
# ║   ImplicitActuatorCfg. effort_limit 을 관절속도와 무관하게 평평히 준다.    ║
# ║                                                                          ║
# ║ USE_DC_MOTOR = True  (실기 정합)                                          ║
# ║   DCMotorCfg. IsaacLab 이 내부적으로 선형 N-T 곡선을 적용한다:             ║
# ║       τ_max(ω) = clamp( saturation_effort·(1 − ω/velocity_limit),        ║
# ║                         0, effort_limit )                                ║
# ║                                                                          ║
# ║ 【왜 필요한가 — XM540-W150 실측 스펙(ROBOTIS 공식)】                       ║
# ║   12.0V : 스톨 7.3 N·m / 무부하 53 rpm (5.55 rad/s)                       ║
# ║   11.1V : 스톨 6.9 N·m / 무부하 50 rpm (5.24 rad/s)                       ║
# ║   연속 정격 1.46 N·m (스톨의 20%). 기어비 152.3:1                          ║
# ║                                                                          ║
# ║   τ(ω) = 7.3·(1 − ω/5.55) [12V]                                          ║
# ║     ω=0.0 → 7.30   ω=2.0 → 4.67   ω=3.0 → 3.35                           ║
# ║     ω=3.5 → 2.70   ω=4.0 → 2.04   ω=4.5 → 1.38  [N·m]                    ║
# ║                                                                          ║
# ║   현재 sim: ω 무관 5.0 N·m 평평 → ω=4.0 에서 실기의 2.5배, 4.5 에서 3.6배. ║
# ║   보행은 ω<2 라 갭이 작지만, 킥 스윙은 ω 3~4.5 라 정확히 그 구간에서 어긋   ║
# ║   난다. "걷기는 잘 맞는데 킥에서만 실기가 불안정" 과 정확히 일치한다.       ║
# ║                                                                          ║
# ║ 【주의】 dynamics 가 바뀌므로 기존 정책을 그대로 못 쓴다. 재학습 필요.      ║
# ║   먼저 walk_kick_v1_4 의 torque_envelope_penalty(물리 불변, 파인튜닝 가능)  ║
# ║   로 시도하고, 그걸로 부족할 때 이 스위치를 켠다.                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
USE_DC_MOTOR = False

# 보수적으로 배터리 새그(11.1V) 기준. 경기 후반에도 되어야 하기 때문.
DC_STALL_TORQUE = 6.9      # [N·m] saturation_effort
DC_NO_LOAD_SPEED = 5.24    # [rad/s] velocity_limit (= 50 rpm)


def _make_actuator(joint_names, params, use_dc=None):
    """관절 그룹 액추에이터 cfg 생성.

    use_dc 가 None 이면 모듈 전역 USE_DC_MOTOR 를 따른다. 레지스트리 항목이
    "use_dc_motor" 키를 주면 **그 항목만** 다른 모델을 쓴다 — 기상은 실기
    토크 곡선이 필요하지만 보행/킥은 기존 물리를 유지해야 하기 때문이다.
    stiffness / damping / armature / friction 은 두 경우 모두 동일하게 전달해
    스위치를 켜고 끌 때 그 값들이 바뀌지 않도록 한다(변인 하나만 바꾸기).
    """
    if use_dc is None:
        use_dc = USE_DC_MOTOR

    if not use_dc:
        return ImplicitActuatorCfg(
            joint_names_expr=joint_names,
            effort_limit_sim=params["effort"],
            velocity_limit_sim=params["vel"],
            stiffness=params["stiffness"],
            damping=params["damping"],
            armature=params["armature"],
            friction=params.get("friction", 0.0),
        )

    # DCMotor: 저속 상한은 params["effort"](열/전류 캡), 곡선은 실측 스펙으로.
    # velocity_limit 은 "무부하 속도" 여야 곡선 기울기가 맞는다.
    # params["vel"](4.5)은 sim 편의값이라 여기 쓰면 안 된다.
    #  ★ *_sim 을 반드시 함께 준다 (2026-08-29 실측으로 발견) ★
    #    effort_limit / velocity_limit 은 **액추에이터 모델**이 쓰는 값이고,
    #    effort_limit_sim / velocity_limit_sim 은 **물리 솔버**에 적용된다.
    #    (actuator_base_cfg.py 문서: "None 이면 USD joint prim 값을 쓴다")
    #
    #    처음에 모델용만 줬더니 물리 솔버 effort 한계가 USD 기본값
    #    **1,000,000,000** 이 됐다. ImplicitActuator 분기는 effort_limit_sim=5.0
    #    으로 물리가 한 번 더 잘라주는데 DC 분기는 그 안전망이 없었다.
    #    PD 게인 50 에 오차가 크면 순간 토크가 그대로 통과해 액션이 폭주했다
    #    (학습 실패: action_rate -32.68, value_loss 47.2, reward -290).
    #
    #    두 모델을 비교하려면 **물리 솔버 설정이 같아야** 한다.
    #    바뀌는 것은 오직 "속도에 따라 토크가 깎이는가" 하나여야 한다.
    return DCMotorCfg(
        joint_names_expr=joint_names,
        # ── 액추에이터 모델 (N-T 곡선) ──
        effort_limit=params["effort"],        # 연속 토크 (짧은 시간 상한)
        saturation_effort=DC_STALL_TORQUE,    # 스톨 토크
        velocity_limit=DC_NO_LOAD_SPEED,      # 무부하 속도 (곡선 기울기)
        # ── 물리 솔버 (ImplicitActuator 분기와 동일하게) ──
        effort_limit_sim=params["effort"],
        velocity_limit_sim=params["vel"],
        stiffness=params["stiffness"],
        damping=params["damping"],
        armature=params["armature"],
        friction=params.get("friction", 0.0),
    )


# ═══════════════════════════════════════════════════════════════
#  레지스트리 → ArticulationCfg 빌더
# ═══════════════════════════════════════════════════════════════

# 관절 타입별 액추에이터 그룹 (actuator_per_type=True 일 때)
# joint name 패턴: {l,r}_{location}_{type}_joint
_PER_TYPE_MAP = {
    "pelvis_pitch": ["r_pelvis_pitch_joint", "l_pelvis_pitch_joint"],
    "pelvis_roll":  ["r_pelvis_roll_joint",  "l_pelvis_roll_joint"],
    "pelvis_yaw":   ["r_pelvis_yaw_joint",   "l_pelvis_yaw_joint"],
    "knee_pitch":   ["r_knee_pitch_joint",   "l_knee_pitch_joint"],
    "ankle_pitch":  ["r_ankle_pitch_joint",  "l_ankle_pitch_joint"],
    "ankle_roll":   ["r_ankle_roll_joint",   "l_ankle_roll_joint"],
}


def _resolve_entry(key):
    """ROBOT_CONFIGS[key] 를 'inherit' 체인을 따라 펼쳐 평평한 dict 로 반환.

    병합은 **최상위 키 단위의 얕은 덮어쓰기**다. 자식이 init_joint_pos 를
    주면 부모의 init_joint_pos 는 통째로 버려진다 (부분 merge 아님).
    일부러 이렇게 했다 — 부분 merge 는 "이 관절 기본각이 어디서 온
    값인지"를 추적 불가능하게 만든다. 자식은 필요한 dict 를 전부 다시 쓴다.

    반대로 다리 게인(pelvis_pitch/knee_pitch/...)이나 usd_path 처럼 자식이
    안 건드리는 키는 부모 것이 그대로 따라온다. 이게 inherit 의 목적이다 —
    XM_robot04 의 다리 튜닝을 바꾸면 파생 항목도 같이 따라온다.
    """
    chain = []
    seen = []
    k = key
    while True:
        if k not in ROBOT_CONFIGS:
            raise KeyError(
                f"[mx_bot] '{k}' 은(는) ROBOT_CONFIGS에 없습니다. "
                f"사용 가능: {list(ROBOT_CONFIGS.keys())}"
            )
        if k in seen:
            raise ValueError(
                f"[mx_bot] 'inherit' 순환 참조: {' -> '.join(seen + [k])}"
            )
        seen.append(k)
        chain.append(ROBOT_CONFIGS[k])
        parent = ROBOT_CONFIGS[k].get("inherit")
        if parent is None:
            break
        k = parent

    merged = {}
    for entry in reversed(chain):          # 조상 → 자손 순으로 덮어쓰기
        merged.update({kk: vv for kk, vv in entry.items() if kk != "inherit"})
    return merged


def _build_actuators(cfg, joints):
    """해결된 cfg dict 로부터 actuator 그룹 dict 를 만든다."""
    use_dc = cfg.get("use_dc_motor")          # None 이면 전역값을 따른다
    if cfg.get("actuator_per_type", False):
        # ─── 관절 타입별 분리 (6그룹, L/R 묶음) ───
        actuators = {}
        for type_name, type_joints in _PER_TYPE_MAP.items():
            if type_name in cfg:
                actuators[type_name] = _make_actuator(type_joints, cfg[type_name], use_dc)
            else:
                print(f"[mx_bot] WARNING: '{type_name}' not in config, skipping")
    else:
        # ─── 기존 방식 (pelvis/knee/ankle 3그룹) ───
        actuators = {
            "pelvis": _make_actuator([j for j in joints if "pelvis" in j], cfg["pelvis"], use_dc),
            "knee":   _make_actuator([j for j in joints if "knee"   in j], cfg["knee"], use_dc),
            "ankle":  _make_actuator([j for j in joints if "ankle"  in j], cfg["ankle"], use_dc),
        }

    # 팔 actuator 추가 (cfg에 arm 키가 있을 때만)
    if "arm" in cfg:
        actuators["arm"] = _make_actuator(MX_ARM_JOINTS, cfg["arm"], use_dc)

    # 목 actuator 추가 (cfg에 neck 키가 있을 때만) — 학습 미사용, 기본자세 0 고정
    if "neck" in cfg:
        actuators["neck"] = _make_actuator(MX_NECK_JOINTS, cfg["neck"], use_dc)

    return actuators


def build_robot_cfg(key):
    """ROBOT_CONFIGS 의 한 항목으로부터 MXArticulationCfg 를 만든다.

    같은 레지스트리에서 서로 다른 로봇 설정을 **여러 개** 만들 수 있다.
    태스크마다 팔 활성화 여부나 기본 자세가 다를 때 쓴다:

        MX_BOT_CFG       = build_robot_cfg("XM_robot04")          # 보행/킥
        MX_BOT_GETUP_CFG = build_robot_cfg("XM_robot04_getup")    # 기상

    매 호출이 새 객체를 만든다. IsaacLab 의 configclass.replace() 는
    dataclasses.replace (얕은 복사)라 actuators dict 가 공유되므로,
    태스크별 변형은 .replace() 로 덮어쓰지 말고 여기서 따로 빌드해야 한다.
    """
    cfg = _resolve_entry(key)
    joints = cfg.get("leg_joints") or MX_LEG_JOINTS

    init_state_kwargs = dict(
        pos=cfg["init_pos"],
        joint_pos=cfg["init_joint_pos"],
        joint_vel={".*": 0.0},
    )
    # init_rot 은 항목이 명시했을 때만 넘긴다. 안 주면 InitialStateCfg 의
    # 기본값 (1,0,0,0 = 직립)이 그대로 쓰여 기존 항목의 동작이 안 바뀐다.
    if cfg.get("init_rot") is not None:
        init_state_kwargs["rot"] = cfg["init_rot"]

    return MXArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=cfg["usd_path"],
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=cfg.get("self_collisions", False),
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(**init_state_kwargs),
        actuators=_build_actuators(cfg, joints),
        # deploy export 용 관절 순서. 팔까지 쓰는 태스크는 항목에서
        # joint_sdk_names 를 직접 지정한다 (다리 12 + 팔 6 등).
        joint_sdk_names=list(cfg.get("joint_sdk_names") or joints),
    )


# ── 기본(활성) 로봇 — 보행/킥 등 기존 모든 태스크가 쓰는 것 ──
MX_BOT_CFG = build_robot_cfg(ACTIVE_ROBOT)
MX_BOT_USD_PATH = _resolve_entry(ACTIVE_ROBOT)["usd_path"]

# ── 현재 활성 로봇 확인용 (디버그) ──
print(f"[mx_bot] Active robot: {ACTIVE_ROBOT}  →  {MX_BOT_USD_PATH}")
print(f"[mx_bot] Actuator groups: {list(MX_BOT_CFG.actuators.keys())}")

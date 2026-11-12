# velocity_v0_6_env_cfg.py
# ============================================================================
# MX3 Velocity — v0_6 (v0_5 + 가변 보행 주파수)
# ----------------------------------------------------------------------------
# v0_5 대비 변경점은 **딱 한 줄**이다.
#       gait.frequency_range = (1.7, 2.4)     # v0_5 는 None (2.0 Hz 고정)
#   다음으로 확인할 수 있다(그 한 줄 외 차이가 없어야 정상).
#       diff <(sed -n '/^import math/,$p' velocity_v0_5_env_cfg.py) \
#            <(sed -n '/^import math/,$p' velocity_v0_6_env_cfg.py)
#
# ── 왜 가변 주파수인가 ─────────────────────────────────────────────────
#   지금까지 보행 주파수는 2.0 Hz 고정이었다. 그러면 속도를 올릴 때
#   **보폭만** 늘어난다. 한 걸음 보폭 = v / (2f) 이므로
#       0.5 m/s @ 2.0 Hz -> 0.125 m
#       0.7 m/s @ 2.0 Hz -> 0.175 m   (40% 증가)
#       0.7 m/s @ 2.4 Hz -> 0.146 m
#   실제 이족보행은 속도를 올릴 때 보폭과 주파수를 **함께** 올린다.
#   주파수를 못 올리면 고속에서 보폭이 과해지고, 그게 v0_4 실기 관찰
#   ("발을 휘적거림")의 한 축일 수 있다.
#
# ── 왜 속도와 결합하지 않고 독립 샘플인가 ──────────────────────────────
#   f = f(v) 로 묶으면 주파수가 속도의 함수가 되어 **독립 제어 입력이 못 된다.**
#   독립 샘플이면 실기에서 같은 속도라도 "빠른 잔걸음 / 느린 성큼걸음" 을
#   골라 명령할 수 있다. 지형·상황에 따라 쓸 수 있는 자유도가 하나 생긴다.
#   대신 "고속 + 저주파"(보폭 과대) 조합도 학습에 섞이므로 범위를 넓히면
#   안 된다. 그래서 1.7~2.4 로 좁게 잡았다.
#       0.7 m/s 기준 보폭:  1.7 Hz -> 0.206 m  /  2.4 Hz -> 0.146 m
#   잘 되면 다음 런에서 1.5~2.8 로 넓히는 것이 후속 실험이다.
#
# ── 공짜로 되는 것 (관측·미러 수정 불필요) ─────────────────────────────
#   gait_phase 는 이미 3 차원 (cos, sin, freq/frequency_scale) 이고
#   include_frequency=True 다. 즉 **정책은 지금까지도 주파수를 보고 있었다.**
#   그 값이 항상 1.0 (= 2.0/2.0) 이었을 뿐이다. 이제 0.85~1.20 으로 변한다.
#     · 관측 차원 불변 : policy 48 x5 = 240 / critic 51 x5 = 255
#       -> 실기 러너 수정 불필요, v0_5 체크포인트에서 resume 가능
#     · _MIRROR_SPEC 수정 불필요 : gait_phase 부호가 [-1, -1, +1] 인데
#       주파수 성분은 좌우 반전에 불변이므로 그대로 맞다
#     · feet_gait_v01 리워드 자동 대응 : term 의 위상을 참조하므로
#       주파수가 바뀌면 접지 타이밍 요구도 같이 따라간다
#     · stop_rate / 완전정지 / freeze_during_reset_hold 전부 그대로 동작
#
# ── 구현 (htwk_gait_v01.py, 하위 호환) ─────────────────────────────────
#   PhaseOnlyGaitCommandCfg.frequency_range 를 신설했다.
#     None (기본)  : 전 env 가 cfg.frequency 로 고정 — 기존 환경 동작 불변
#     (lo, hi)     : _resample_command 에서 env 별로 균등 샘플
#   v0_1~v0_5 와 walk_kick 계열은 None 이므로 한 글자도 안 달라진다.
#
# ── 확인할 것 ──────────────────────────────────────────────────────────
#   1) Metrics/gait/gait_frequency — 1.7~2.4 사이에 분포해야 한다.
#      v0_5 는 1.83 근처 한 점이었다(정지 env 가 0 을 섞어 2.0 보다 낮다).
#   2) action_rate — 이 런의 본래 동기다. v0_5 최종 -0.494 / 최적 -0.206.
#      주파수 자유도가 늘어 보폭 부담이 줄면 개선되어야 한다.
#   3) 커리큘럼 — 과제가 어려워지므로 lin 0.72 / ang 0.62 문턱으로 여전히
#      열리는지. v0_5 는 lin iter 1000 이내 / ang iter 1417 이었다.
#   4) Loss/symmetry — v0_5 수준(0.0025 근처) 유지되는지.
#
# gym id: Unitree-MX3-Velocity-V0-6 / -V0-6-Play / -V0-6-Metric
# ============================================================================
import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.mx_bot import MX_BOT_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


# ------------------------------------------------------------
# Terrain — flat 100% (v0는 지형 난이도 없음)
# ------------------------------------------------------------
FLAT_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)

# ------------------------------------------------------------
# MX naming
# ------------------------------------------------------------
MX_BASE_LINK = "base_link"

MX_FEET = ["r_ankle_roll_link", "l_ankle_roll_link"]

MX_LEG_JOINTS = [
    "r_pelvis_pitch_joint", "r_pelvis_roll_joint", "r_pelvis_yaw_joint",
    "r_knee_pitch_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_pelvis_pitch_joint", "l_pelvis_roll_joint", "l_pelvis_yaw_joint",
    "l_knee_pitch_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
]


@configclass
class RobotSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_ROAD_CFG,
        max_init_terrain_level=FLAT_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                f"TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                f"kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


# ================================================================
# Events — DR 최소 (physics_material / reset_base / reset_joints / push)
# ================================================================
@configclass
class EventCfg:

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.01, 0.01),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.0),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-0.02, 0.02),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # 오프셋 노이즈 — reset_joints_by_scale "이후"에 실행돼야 함 (선언 순서 = 실행 순서).
    # scale은 default=0 관절(pelvis_roll/yaw, ankle_roll)을 못 흔드므로, 여기서 ±0.02 덧셈.
    reset_joints_noise = EventTerm(
        func=mdp.reset_joints_additive_noise,
        mode="reset",
        params={
            "noise_range": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
    )

    # reset 후 0.3s 동안 정책 출력을 마스킹 → 랜덤화된 초기 관절각이
    # 정착하는 과도기를 확보 (DelayedJointPositionAction이 실제 마스킹 수행)
    reset_hold = EventTerm(
        func=mdp.set_reset_hold_steps,
        mode="reset",
        params={"hold_time_s": 0.3},
    )


# ================================================================
# Commands — 표준 (hold 없음). 좁게 시작 → 커리큘럼으로 limit 확장
# ================================================================
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.15,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # 초기 학습 범위 (좁게)
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.5, 0.5),
        ),
        # 최종 목표 범위 (커리큘럼이 여기까지 확장)
        # ★ v0_5 변경: lin_vel_x 상한 0.5 → 0.7 (전진만. 후진 -0.5 유지)
        #   커리큘럼은 상·하한에 ±delta 를 동시에 더한 뒤 limit 로 클램프하므로
        #   하한은 레벨 4 에서 -0.5 에 걸려 멈추고 상한만 레벨 8 에서 0.7 도달.
        #   명령 샘플링이 r.uniform_(lo,hi) 라 비대칭을 가정 없이 처리한다.
        #   ※ v0_4 는 0.8 이었다. 0.7 로 내린 이유는 gait 2.0 Hz 고정에서
        #     0.8 m/s 가 한 걸음 0.2 m 를 요구해 stride 가 과해지기 때문이다
        #     (0.7 이면 0.175 m). 실기 휘적거림의 한 축일 수 있다.
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.7),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.5, 1.5),
        ),
    )

    # ── [A] 보행 위상 (v0 대비 신규) ─────────────────────────────────
    # ★ 선언 순서 중요 ★
    # CommandManager 는 선언 순서대로 갱신한다. gait 의 정지 판정이
    # base_velocity 를 읽으므로 base_velocity 가 반드시 먼저 와야 한다.
    #
    # 주파수 2.0 Hz 고정 = v0 feet_gait 의 period 0.5 s 와 동일.
    # 속도 0 이면 주파수 0 -> 위상이 0 으로 최단 경로 수렴 (0.25 s).
    # 관측 2 차원 (cos, sin), 게이팅 없음 — 히스토리로 정지/보행 구분.
    gait = mdp.PhaseOnlyGaitCommandCfg(
        resampling_time_range=(10.0, 10.0),
        frequency=2.0,
        stop_rate=2.0,
        # ── [D] 제자리 걷기 (v0_1 대비 신규) ─────────────────────────
        # 관측에 주파수를 넣어 "속도 0 + 발을 드는 중"과
        # "속도 0 + 가만히 서 있음"을 구분한다.
        # 위상 cos/sin 만으로는 이 구분의 근거가 약하다.
        include_frequency=True,
        frequency_scale=2.0,
        # 속도 0 인 env(15%) 중 60% 완전 정지 / 40% 제자리 걷기
        #   완전 정지  ~ 9 %
        #   제자리걷기 ~ 6 %
        rel_stop_when_standing=0.6,
        velocity_command_name="base_velocity",
        standing_threshold=0.1,
        # ★ v0_6 유일한 변경점: 주파수 고정(2.0) → env 별 랜덤 1.7~2.4 Hz.
        #   리샘플(10 s)마다 갱신되므로 한 에피소드(20 s)에 두 값을 겪는다.
        #   frequency=2.0 은 이제 안 쓰이지만(범위가 우선) 참고용으로 남긴다.
        #   되돌리려면 None. 넓히려면 (1.5, 2.8) 이 다음 후보다.
        frequency_range=(1.7, 2.4),
        reset_phase_on_resample=True,
        freeze_during_reset_hold=True,
        debug_vis=False,
    )


# ================================================================
# Actions — DelayedJointPositionAction (sim-to-real 1순위)
#   - action_delay 1~2 step: 실기 구동 지연 내성 (정지 떨림 완화 핵심)
#   - reset_hold 마스킹: 리셋 후 hold 동안 정책 출력 0으로 (아래 reset_hold 이벤트와 짝)
#   - target_limits=None: XM_robot04 USD 자체 관절한계로 클램핑 (로봇 고유값)
# ================================================================
@configclass
class ActionsCfg:
    JointPositionAction = mdp.DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=MX_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        target_limits=None,
        action_delay_range=(1, 2),
    )


# ================================================================
# Observations — 표준 함수만 (hold_aware 제거)
# ================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands_with_reset_hold,
            params={"command_name": "base_velocity"},
        )
        # ── [B] 보행 위상 관측 (v0 대비 신규, +2 차원) ───────────────
        # v0 는 위상을 리워드에만 쓰고 관측에 넣지 않았다. 히스토리
        # 5 프레임 = 100 ms 는 보행 주기 500 ms 보다 짧아서 정책이
        # 절대 위상을 복원할 수 없었다. 즉 정책이 볼 수 없는 시계로
        # 채점하고 있었던 셈이다.
        #
        # 노이즈는 주지 않는다. 위상은 실기에서도 내부 시계로 정확히
        # 생성되는 값이라 관측 오차가 없다.
        gait_phase = ObsTerm(
            func=mdp.gait_phase_v01,
            params={"gait_command_name": "gait"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=MX_LEG_JOINTS, preserve_order=True
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=MX_LEG_JOINTS, preserve_order=True
                ),
            },
            scale=0.05,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands_with_reset_hold,
            params={"command_name": "base_velocity"},
        )
        gait_phase = ObsTerm(
            func=mdp.gait_phase_v01,
            params={"gait_command_name": "gait"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=MX_LEG_JOINTS, preserve_order=True
                ),
            },
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=MX_LEG_JOINTS, preserve_order=True
                ),
            },
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ================================================================
# Rewards — v0 최소 세트 (11개, 전부 표준 함수)
# ================================================================
@configclass
class RewardsCfg:
    # -- task (이름은 커리큘럼이 참조하므로 track_lin_vel_xy / track_ang_vel_z 고정)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_hold_aware,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.15)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_hold_aware,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.15)},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base 안정화
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    # 몸통 자세: roll/pitch 분리 (베스트 모델 방식)
    #  - roll 은 좌우 균형이라 강하게(-5.0)
    #  - pitch target 0° (수직). 이전 -2° 앞숙임 -> 0 으로 변경
    base_roll_l2 = RewTerm(func=mdp.base_roll_l2, weight=-5.0)
    base_pitch_target_l2 = RewTerm(
        func=mdp.base_pitch_target_l2,
        weight=-1.0,
        params={"target_pitch": math.radians(0.0)},
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-150.0,
        params={"target_height": 0.39},
    )

    # -- 접촉 (발목 외 링크 접촉 억제)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["(?!.*ankle.*).*"]
            ),
        },
    )

    # -- 팔자걸음(out-toeing) 교정: 엉덩이 yaw를 기본자세(발끝 정면)로 유지.
    #    회전(yaw 명령)에도 쓰이는 관절이라 약하게 시작 → 회전 굼뜨면 낮추기.
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["r_pelvis_yaw_joint", "l_pelvis_yaw_joint"],
            ),
        },
    )

    # -- 좌우 발 전후 비대칭 교정 (A안): pelvis pitch 창-평균 대칭.
    #    기존 순간 abs 버그 제거 버전. 대칭 gait면 ~0, 한쪽 치우침이면 페널티.
    pelvis_pitch_symmetry = RewTerm(
        func=mdp.pelvis_pitch_symmetry_windowed,
        weight=-1.0,
        params={
            "window_size": 30,
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    # -- 보행 주기 강제 (발산 방지 핵심): 0.5s 주기, 좌우 반주기 엇갈림.
    #    규칙적 리듬을 부여해 emergent gait의 발산을 막음. command-gated(정지 시 off).
    # ── [C] feet_gait 를 term 위상 참조로 교체 ───────────────────────
    # v0 는 위상을 episode_length_buf 로 매번 재계산했다. 그 위상은
    # 관측에 안 들어갔으므로 관측과 리워드가 서로 다른 시계를 볼
    # 위험이 있었다. 이제 gait term 하나가 위상을 소유하고 관측과
    # 리워드가 같은 값을 참조한다. 단일 진실 원천.
    #
    # 가중치 0.5 는 v0 그대로. period/offset/threshold 의미도 동일하다.
    # (period 는 term 의 frequency=2.0 이 대신하므로 파라미터에서 빠짐)
    #
    # 정지 판정도 term 이 담당하므로 command_name 게이팅이 불필요하다.
    # 정지 과도 구간(위상 0 수렴 중)에는 켜져 있어 양발 접지를 유도하고,
    # 완전 정지 후에는 꺼진다. 그 상태는 아래 feet_contact_standing 담당.
    gait = RewTerm(
        func=mdp.feet_gait_v01,
        weight=0.5,
        params={
            "gait_command_name": "gait",
            "offset": (0.0, 0.5),
            "threshold": 0.55,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=MX_FEET),
        },
    )

    # 진단 전용 (weight=0). 텐서보드에서 위상 수렴을 확인한다.
    metric_phase = RewTerm(
        func=mdp.metric_phase_v01,
        weight=0.0,
        params={"gait_command_name": "gait"},
    )

    # ── [F] 정지 자세 보상 — 판정 기준을 속도에서 주파수로 교체 ──────
    # v0/v0_1 은 feet_contact_without_cmd(속도 명령 기준)를 썼다.
    # 그런데 제자리 걷기는 "속도 0 이면서 발을 들어야 하는" 상태다.
    # 속도 기준을 그대로 두면
    #     feet_gait      : 위상에 맞춰 발을 들어라
    #     feet_contact.. : 양발을 붙여라
    # 두 항이 정면으로 충돌해 제자리 걷기가 학습되지 않는다.
    #
    # stop_posture_v01 은 "주파수 0 이고 위상도 0" 일 때만 보상한다.
    #     완전 정지   -> 보상 대상
    #     제자리 걷기 -> 대상 아님 (발을 들어도 벌하지 않음)
    #
    # 가중치 0.15 는 v0 그대로 유지.
    stop_posture = RewTerm(
        func=mdp.stop_posture_v01,
        weight=0.15,
        params={
            "gait_command_name": "gait",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=MX_FEET),
        },
    )

    # 🆕 완전정지 시 다리를 기본자세(발 모음 중립)로 당김.
    #    stop_posture(양발 접지)만으로는 "한 발 앞으로 낸 staggered 자세" 도
    #    만족돼 정지 시 한쪽 발을 앞으로 내는 문제가 생긴다. 이 항이 교정한다.
    #    완전정지에서만 작동(전환은 feet_gait 담당). weight 는 play 보고 조정.
    stop_default_posture = RewTerm(
        func=mdp.stop_default_posture_v01,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MX_LEG_JOINTS),
            "gait_command_name": "gait",
            "include_stopping": False,
        },
    )

    # ── 진단 지표 (weight=0, 텐서보드 추적 전용) ──────────────────────
    # 학습 초반에 이 둘의 비율이 의도한 9% / 6% 근처로 나오는지 확인할 것.
    metric_is_full_stop = RewTerm(
        func=mdp.metric_is_full_stop_v01,
        weight=0.0,
        params={"gait_command_name": "gait"},
    )
    metric_is_marching = RewTerm(
        func=mdp.metric_is_marching_v01,
        weight=0.0,
        params={
            "gait_command_name": "gait",
            "velocity_command_name": "base_velocity",
            "standing_threshold": 0.1,
        },
    )

    # [일부 복원] feet_slide / feet_clearance 는 아직 제외.

    # 골반 roll을 기본자세로 유지 → 좌우 균형 + foot_flat과 짝.
    # (없으면 골반 roll이 자유로워 한쪽으로 꺾어 버티는 degenerate 자세가 나옴)
    joint_deviation_pelvis_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["r_pelvis_roll_joint", "l_pelvis_roll_joint"],
            ),
        },
    )

    # 발바닥 좌우 비틀림(roll) 억제 → 지면과 평행 (사용자 요청)
    # weight 약하게(-0.3): -2.0은 발목 롤(좌우 균형)을 죽여 걷기와 충돌 → 완화
    # command_name=None: 게이팅 제거 → 정지(cmd=0) 포함 항상 적용.
    #   (기존 command_threshold=0.0 + `cmd_norm > threshold` 는 정지에서 꺼지는 경계버그)
    foot_flat = RewTerm(
        func=mdp.foot_flat_orientation,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_FEET),
            "command_name": None,
        },
    )


# ================================================================
# Terminations
# ================================================================
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.25}
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": 0.8}
    )


# ================================================================
# Curriculum — velocity command level만 (terrain level 없음)
# ================================================================
@configclass
class CurriculumCfg:
    # v0.1: 커리큘럼 완화 — 초기 v0에서 ~1000 iter만에 full range가 열려
    #       정책이 못 따라잡고 악화됨. threshold↑ + delta↓ 로 천천히 확장.
    #       (v0 원래값: threshold 0.6/0.55, delta 0.1)
    # ★ v0_5 문턱 조정 — 둘 다 "종속 변경" 이다 (독립 실험 변수가 아니다)
    #
    #   [lin] 0.78 → 0.72 : lin_vel_x 목표가 0.5 → 0.7 로 넓어진 것의 종속 변경.
    #     v0_3 실측(DR 없음, 목표 0.5): lin_rew 지속 0.744~0.806 / 최댓값 0.842.
    #     문턱 0.78 은 그 지속 구간 한가운데라 목표 0.5 를 겨우 채웠다
    #     (iter 292 도달). 목표를 0.7 로 넓히면 레벨업마다 lin_rew 가 더
    #     내려가므로 0.78 로는 0.55~0.6 부근에서 멈춘다.
    #     0.72 로 잡으면 예상 지속대(0.72~0.78) 아래에 걸치면서도, 학습
    #     극초기(v0_3 iter 100 = 0.679)는 통과 못 해 게이팅이 유지된다.
    #
    #   [ang] 0.72 → 0.62 : mirror_loss_coeff 3.0 의 종속 변경.
    #     v0_3(계수 1.0)의 ang_rew 최댓값이 0.738 로, 문턱 0.72 는 천장에
    #     거의 붙어 있었다(첫 레벨업이 iter 917 에야 나온 이유).
    #     v0_4 3 차에서 계수를 3.0 으로 올리자 ang_rew 천장이 약 0.05 내려가
    #     문턱 0.55 에서도 1.25 에 멈췄다. 계수 3.0 을 쓰면서 문턱 0.72 를
    #     그대로 두면 v0_4 1 차처럼 **한 번도 안 열릴 가능성이 높다.**
    #     0.62 는 v0_3 천장(0.738)에서 계수 상승분(~0.05)과 안전여유(~0.06)를
    #     뺀 값이다. DR 이 없으므로 v0_4 3 차(0.568)보다는 높게 나올 것이다.
    #
    #   delta 는 0.05 그대로. 확장 "속도" 가 아니라 "문턱" 만 낮춘다.
    #   평가 주기 = max_episode_length(1000) / num_steps_per_env(24) ≈ 42 iter.
    #   lin 은 0.3→0.7 로 8 레벨, ang 은 0.5→1.5 로 20 레벨이 필요하다.
    #
    #   ★ 학습 초반에 Curriculum/* 두 값이 오르는지 반드시 확인할 것.
    #     안 오르면 "설정이 나쁘다" 가 아니라 "범위가 안 열렸다" 이고,
    #     그 창은 entropy 가 낮은 초기에만 있다(resume 으로 못 되살림).
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={"reward_term_name": "track_lin_vel_xy", "threshold": 0.72, "delta": 0.05},
    )
    ang_vel_cmd_levels = CurrTerm(
        func=mdp.ang_vel_cmd_levels,
        params={"reward_term_name": "track_ang_vel_z", "threshold": 0.62, "delta": 0.05},
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # 16384 envs 동시 리셋 스파이크(~355k) 대비 655,360 으로 증설
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt

        # v0: terrain 난이도 커리큘럼 없음 → flat 고정
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10

        # play: 전진 0.5 고정 (걷는지 눈으로 확인용)
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.rel_standing_envs = 0.0

# walk_kick_v1_4_env.py
# ============================================================================
# ★★★ v1_4 — 킥 후 실기 안정성 (sim2real 갭 축소, 2026-08-19) ★★★
#   v1_3 로 인사이드 킥 성공. sim 99% 안정인데 실기는 5회 중 1회 킥 후 전도.
#
#   【진단 전제】 sim 99% 는 좋은 소식이 아니라 경고다.
#     실패율 1% 면 정책이 "안정 마진" 을 배울 gradient 가 없다. 넘어지는 env 가
#     없으니 더 안전해질 압력이 없고, 마진 없이 딱 되는 해에 수렴한다.
#     실기에서만 나는 실패는 보상 튜닝이 아니라 **sim 을 어렵게 만들어** 고친다.
#     목표: sim 성공률을 85~90% 로 떨어뜨린 뒤 그 위에서 다시 올린다.
#
#   【가장 유력한 물리 원인 — 속도-토크 곡선 미모델링】
#     ImplicitActuator 는 effort_limit 5.0 N·m 를 관절속도와 무관하게 평평히 준다.
#     실제 XM540-W150(12V): 스톨 7.3 N·m / 무부하 53 rpm(5.55 rad/s)
#       τ(ω) ≈ 7.3(1 − ω/5.55):  ω=3.0→3.35  4.0→2.04  4.5→1.38 [N·m]
#     걷기(ω<2)는 갭이 작지만 킥 스윙(ω 3~4.5)은 sim 이 실기의 2.5~3.6 배.
#     → 킥에서만 어긋나는 현상과 정확히 일치. 배터리 새그면 더 나빠진다.
#
#   【2026-08-19 관찰 갱신 — 진단 재정렬】
#     · 전압은 항상 안정 → 전원(축 B) 배제.
#     · 킥 자체는 대부분 성공 → 액추에이터 토크 한계(축 A)가 주범은 아님.
#       토크가 binding 이면 킥 자체가 약하거나 들쭉날쭉해야 하는데 그렇지 않다.
#     · "지면 상태 / 킥 후 반동이 매번 다르다" → **분산(variance) 문제**.
#       계통 오차(bias)가 아니라 미모델 변동에 대한 강건성 부족이다.
#     → 우선순위 재정렬: STAGE_HARD·STAGE_VARIATION 이 주 레버, ENVELOPE 는 보험.
#
#   【핵심 갭 — sim 은 변동이 없다】
#     · 지면: FLAT_ROAD_CFG 는 완전 평면 100%. 요철 0.
#       실제 인조잔디는 44 cm 로봇 기준 cm 급 요철이 있다.
#     · 마찰: robot material 랜덤이 mode="startup" 이라 env 당 1 회 고정.
#       범위도 0.7~1.2 로 좁다(먼지·습기·잔디 눕는 방향에 따라 실제는 더 넓다).
#     · 공: 질량·마찰·반발이 전부 고정. 실제 공 상태와 접촉점은 매번 다르다.
#     · 킥 반동: sim 접촉은 거의 결정적. 실기는 접촉점·발바닥 변형·스핀으로 매번 다르다.
#     → 정책이 "매번 같은 반동" 을 전제로 수렴했고, 실기의 변동 폭을 못 받아낸다.
#
#   【v1_4 수정 — 5 갈래】
#     ① 모터 포락선 벌점 (STAGE_ENVELOPE)
#        dynamics 는 그대로 두고 실현 불가 영역에만 비용 → 파인튜닝으로 이전 가능.
#        torque_envelope_penalty + joint_vel_over_limit_penalty
#     ② 안정 마진 보상 (STAGE_MARGIN)
#        capture point 마진 / 지지발 미끄러짐 / 양발공중 / 킥윈도우 각속도
#        자세각 벌점은 "이미 기울어진 뒤" 반응하지만 capture point 는 한 박자 빠르다.
#     ③ DR 강화 + 종료 강화 (STAGE_HARD)
#        actuator gain ±5%→±20%, armature/friction 랜덤, 지연 1~2→1~4,
#        IMU 상수 바이어스, 킥 직후 표적 외란,
#        bad_orientation 0.8→0.6 rad / base_height 0.25→0.28
#     ④ base_height 데드존 (STAGE_HEIGHT)
#        weight −150 평지붕 벌점이 지지무릎 굽힘 여유를 뺏는다 → ±0.025 m 무벌점
#     ⑤ 접촉 변동 (STAGE_VARIATION) ★ 이번 관찰로 승격된 주 레버
#        마찰 per-episode 재샘플 + 범위 확대 / 공 물성 랜덤 / 킥 직후 반동 외란
#        ※ 지형 요철(TERRAIN_ROUGH)은 별도이며 **기본 OFF** 다.
#          이 로봇(발 18×9cm, 고관절 39cm)에 1.5cm 요철은 발 roll 18° 외란이고,
#          더 결정적으로 공이 깨진다 — reset_ball_around_robot 이 ball_z 를
#          world 절대 0.065 로 하드코딩하는데다, 경사에서 공이 굴러가면
#          frozen 관측(ball_pos_b)이 옛 위치를 가리킨다. 파일 상단 주석 참조.
#
#   【실기 없이 검증하는 법】
#     로봇을 못 돌리는 동안의 대체 지표는 **DR 하에서의 sim 실패율** 이다.
#     TensorBoard 의 Episode_Termination/bad_orientation + base_height 합이
#     10~15% 가 목표. 이게 1% 로 유지되면 DR 이 아직 약한 것이고 실기 개선도 없다.
#     추가로 -Stress-* play 태스크로 조건별 성공률을 눈으로 확인할 수 있다.
#
#   experiment=unitree_mx3_walkkick_v1_4. v1_3 체크포인트에서 resume 권장.
# ----------------------------------------------------------------------------
#   (아래는 walk_kick_v1 원본 헤더 — 참고용)
# ----------------------------------------------------------------------------
# (원 v1) HARD KICK 탄성공 + 상수강타 기반. 이후 공상한3.0/스무딩/±90°→±70°/
#   가까운공/data-aug 발균형 등으로 발전한 최신 config 가 여기 복사돼 있다.
# ============================================================================
# MX3 Walk + Kick — v1
# ----------------------------------------------------------------------------
# 방식 전환: 검증된 옛 walk_kick_env_cfg.py 의 "킥 레이어" 를 그대로 쓰고,
# "보행 레이어" 만 velocity_v0_3_env_cfg.py 로 교체한다.
#
#   ★ 킥 레이어 = 옛 walk_kick 그대로 (kick_mdp.py 함수들) ★
#     - 공 스폰 / 목표 / no-kick 데드라인 이벤트
#     - 관측: ball_pos_b(frozen) / target_dir_b / kick_strength (+critic ball_vel_b)
#     - velocity_commands 관측: commands_zero_on_kick_or_hold (킥 후 명령=0)
#     - 리워드: ball_velocity(proj²) / ball_z / kick_success_bonus(+8) /
#               ball_approach(절대거리 가우시안) / joint_pose_to_default
#     - 종료: kick_resolved(킥+정지) / no_kick_deadline
#     - track_lin_vel_x (x만 추종, y 자유) 는 hold-aware (hold 구간 0)
#     → 투터치 방지 = 첫 접촉 래치(speed_threshold 0.01) + 명령 관측 0 세트
#
#   ★ 보행 레이어 = v0_3 (v0_2 본문) ★
#     - PhaseOnlyGaitCommand + gait_phase 관측 + feet_gait_v01
#     - stop_posture / stop_default_posture (정지/제자리걷기)
#     - base 안정화, 미러 손실 호환 관측 레이아웃
#     - DelayedJointPositionAction (지연 1~2 step)
#
# 이후 세부(커리큘럼, 킥 리워드 스케일, 미러 계수)는 학습 보며 조정한다.
# gym id: Unitree-MX3-WalkKick-V1 / -Play
# ============================================================================

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from unitree_rl_lab.tasks.locomotion.mdp import kick_mdp
from unitree_rl_lab.tasks.locomotion.mdp import kick_mdp_v0
from unitree_rl_lab.tasks.locomotion.mdp import kick_inside_mdp
from unitree_rl_lab.tasks.locomotion.mdp import kick_stability_mdp


# ------------------------------------------------------------
# Terrain — v0_3 와 동일 (flat 100%)
# ------------------------------------------------------------
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ ★ 지형 요철 — 기본 OFF (2026-08-19 재검토) ★                             ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║ 처음엔 ±1.5 cm 요철을 켰다가 되돌렸다. 이유 3 가지.                       ║
# ║                                                                          ║
# ║ 【1】 이 로봇 크기에 1.5 cm 는 크다                                        ║
# ║   발 18×9 cm / 고관절 39 cm / 스텝 7.5 cm.                                ║
# ║   horizontal_scale=0.1 이라 셀 하나가 10×10 cm — 발이 셀 2 개를 걸친다.    ║
# ║   즉 발이 경계에 착지하면 그 높이차가 그대로 발 기울기가 된다:             ║
# ║       요철 15 mm → pitch 9.5° / roll 18.4°                               ║
# ║       요철  6 mm → pitch 3.8° / roll  7.6°                               ║
# ║       요철  3 mm → pitch 1.9° / roll  3.8°                               ║
# ║   킥 직후 단각지지 상태에서 roll 18° 외란은 발목 roll 권한을 넘어선다.     ║
# ║   (성인 G1 고관절 72 cm 기준 2 cm ≈ MX3 의 10.8 mm. 1.5 cm 는 그 이상.)   ║
# ║                                                                          ║
# ║ 【2】 ★ 공이 깨진다 — 이게 더 큰 문제 ★                                   ║
# ║   kick_mdp.reset_ball_around_robot 은 ball_z 를 world 절대 0.065 로       ║
# ║   하드코딩한다(지면 z=0 가정). 요철 지형에서는                            ║
# ║     · 높은 셀 → 공이 지면에 파묻힌 채 스폰 → depenetration 으로 튕겨나감   ║
# ║     · 낮은 셀 → 공중에서 낙하 → restitution 0.8 이라 튀어 오름            ║
# ║   게다가 경사에서 공이 굴러간다. 구름 가속 a=(5/7)g·sinθ 로 계산하면       ║
# ║     15 mm/10 cm(8.5°) → 2 초에 2.08 m,  6 mm(3.4°) → 0.84 m 이동.        ║
# ║   그런데 policy 관측 ball_pos_b 는 **리셋 시점에 frozen** 이다.           ║
# ║   → 정책은 "이미 굴러가 버린 옛 위치" 를 보고 찬다. 킥 태스크가 조용히     ║
# ║     망가지고, 성공률이 떨어져도 원인을 지형이 아닌 다른 데서 찾게 된다.    ║
# ║                                                                          ║
# ║ 【3】 "지면이 매번 다르다" 의 지배적 성분은 높이가 아니라 마찰이다          ║
# ║   인조잔디에서 미끄러짐·반작용을 좌우하는 건 잔디 눕는 방향·먼지·습기 →    ║
# ║   마찰 변동이다. 그건 아래 physics_material 을 mode="reset" 으로 바꾸고    ║
# ║   범위를 0.45~1.35 로 넓혀 이미 다루고 있다. 공을 안 깨뜨리면서 같은       ║
# ║   목적을 달성한다.                                                        ║
# ║                                                                          ║
# ║ 【켜려면 — 선행 조건】                                                    ║
# ║   ① reset_ball_around_robot 을 지형 높이 인식형으로 고칠 것               ║
# ║      (RayCasterCfg 추가 후 공 xy 에서 지면 높이를 읽어 ball_z 계산)        ║
# ║   ② 아래 값(6 mm / proportion 0.3)으로 시작. 1.5 cm 는 쓰지 말 것.        ║
# ║   ③ 그래도 공은 경사에서 조금씩 구른다. frozen 관측과의 충돌을 감수할지    ║
# ║      먼저 판단할 것.                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
TERRAIN_ROUGH = False

_SUB_TERRAINS = (
    {
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            # 6 mm — 발 기울기 pitch 3.8° / roll 7.6° 수준. 이 로봇 크기의 상한.
            noise_range=(0.0, 0.006),
            noise_step=0.002,
            border_width=0.25,
        ),
    }
    if TERRAIN_ROUGH
    else {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
)

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
    sub_terrains=_SUB_TERRAINS,
)

# ------------------------------------------------------------
# MX naming — v0_3 와 동일
# ------------------------------------------------------------
MX_BASE_LINK = "base_link"
MX_FEET = ["r_ankle_roll_link", "l_ankle_roll_link"]
MX_LEG_JOINTS = [
    "r_pelvis_pitch_joint", "r_pelvis_roll_joint", "r_pelvis_yaw_joint",
    "r_knee_pitch_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_pelvis_pitch_joint", "l_pelvis_roll_joint", "l_pelvis_yaw_joint",
    "l_knee_pitch_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
]

# ------------------------------------------------------------
# Ball / kick spec — 옛 walk_kick 그대로
# ------------------------------------------------------------
BALL_RADIUS = 0.065
BALL_MASS = 0.15   # 질량은 킥속도 레버 아님(공이 이미 발보다 가벼움) → 원복
# ── 공 스폰: 극좌표 (거리 + 방위각), ±60° 로 확장 ──────────────────────
# 정면 좁은 상자(옛것)에서 극좌표로 교체. 거리·방위가 독립이라 정책이
# ball_pos_b 를 실제 입력으로 쓰게 된다. 방위 범위는 0 중심 좌우대칭이어야
# 미러 손실과 안 싸운다(±60°).
BALL_DIST_RANGE = (0.2, 0.5)                              # 20~50cm (2026-08-13)
BALL_BEARING_RANGE = (-math.radians(60.0), math.radians(60.0))  # ±60° (2026-08-15, 70→60)
# 목표 방향: 로봇 리셋 정면 기준 오프셋. ±60° (0 중심 대칭)
TARGET_DIR_ANGLE_RANGE = (-math.radians(60.0), math.radians(60.0))  # ±60° (2026-08-15, 90→60)
TARGET_DISTANCE = (2.0, 4.0)
KICK_HOLD_TIME_S = 3.0                 # 킥 후 정지 유지 시간(성공 판정)

# ------------------------------------------------------------
# ★ 인사이드 킥 실험 스위치 (증분 A/B 용) — 코드 수정 없이 단계별 검증
# ------------------------------------------------------------
# STAGE_INSIDE : False → 인사이드 품질 배율을 끄고(=floor 1.0) 스윙 shaping 도 0.
#                이 상태는 "v1_1 킥 파워 그대로 + 드리블 차단만" 이라
#                Stage1(드리블 제거)이 먹었는지를 단독으로 검증할 수 있다.
#                True  → 인사이드 품질 배율 + 스윙 shaping 활성 (본 목표).
# STAGE_HIPYAW : False → 골반 yaw 벌점 원래대로(항상 적용).
#                True  → 공 0.5m 안쪽에서 벌점 해제(킥 자세 자유).
#                inside_q 가 0.5 근처에서 정체할 때만 켠다.
STAGE_INSIDE = True    # 2단계: STAGE1(하드킥) resume 위에 인사이드 품질 배율 켬 (2026-08-16)
STAGE_HIPYAW = False

_Q_FLOOR = 0.3 if STAGE_INSIDE else 1.0     # 1.0 이면 옛 ball_velocity 와 수학적 동일
_W_SWING = 1.0 if STAGE_INSIDE else 0.0
_HIPYAW_RELEASE_R = 0.5 if STAGE_HIPYAW else 0.0   # 0.0 이면 옛 joint_deviation_l1 과 동일

# ------------------------------------------------------------
# ★ 인사이드 킥 공유 파라미터 — kick_inside_mdp 를 쓰는 모든 항이 동일해야 함
# ------------------------------------------------------------
# kick_inside_mdp._update_state 는 "스텝당 1 회" 만 돌기 때문에, 항마다 값이
# 다르면 먼저 평가된 항의 값이 이기고 나머지는 조용히 무시된다.
# (kick_mdp._maybe_record_kick 의 공유 래치와 정확히 같은 함정)
# → 모듈이 첫 스텝에 예외를 던져 잡아 주지만, 애초에 여기 상수 하나로 묶어 둔다.
INSIDE_MIN_STRIKE_SPEED = 1.8   # [m/s] 강타 래치 문턱. 최우선 튜너블.
INSIDE_WINDOW_STEPS = 20        # 래치 후 보상 스텝 수 (0.4 s @50Hz)
INSIDE_TOUCH_FREE_STEPS = 10    # 벌점 없이 허용되는 누적 접촉 스텝 (0.20 s)

# 발 SceneEntityCfg 도 동일 객체 구성으로 통일 (body_ids 순서 = [우발, 좌발])


def _feet_cfg():
    """인사이드 킥 항 공용 발 cfg. preserve_order 필수 ([우발, 좌발] 순서)."""
    return SceneEntityCfg("robot", body_names=MX_FEET, preserve_order=True)


# ------------------------------------------------------------
# ★ 킥 판정 속도 문턱 — kick_mdp._maybe_record_kick 공유 래치
# ------------------------------------------------------------
# 0.01 → 0.8. 세 곳(kick_success_bonus / kick_resolved / early_kick_done)이
# 같은 래치를 공유하므로 값이 다르면 먼저 평가되는 항의 문턱이 이겨 조용히 깨진다.
# 반드시 이 상수 하나만 참조할 것.
#
# ※ 스크래치 학습 시 주의: 0.8 은 "약한 킥은 아예 킥이 아님" 이라는 뜻이라,
#   초기 정책이 0.5 m/s 밖에 못 밀면 성공 신호가 0 이 되어 "안 차기" 로 붕괴할
#   수 있다(과거 peak-fire-once 실험의 킥률 1% 와 같은 실패 모드).
#   v1_1 체크포인트(이미 강타 가능)에서 이어 학습하면 안전하다.
#   스크래치라면 0.4 로 시작해 학습이 붙은 뒤 0.8 로 올릴 것.
KICK_SPEED_THRESHOLD = 0.8   # STAGE2: resume 베이스가 하드킥이라 0.8 안전(그레이즈 배제)

# ------------------------------------------------------------
# ★ v1_4 안정성 스위치 — 증분 검증용 (모두 독립)
# ------------------------------------------------------------
#   STAGE_ENVELOPE : 모터 속도-토크 포락선 벌점. 실기가 못 내는 궤적 억제.
#   STAGE_MARGIN   : capture point / 미끄러짐 / 양발공중 / 킥윈도우 각속도.
#   STAGE_HARD     : DR 강화 + 종료 문턱 강화. sim 성공률을 일부러 떨어뜨린다.
#   STAGE_HEIGHT   : base_height 데드존(지지무릎 굽힘 여유 확보).
#
#   ※ 한 번에 다 켜지 말 것. ①→②→③→④ 순서로 하나씩 켜고 실기 확인.
#     특히 ③은 sim 성공률이 눈에 띄게 떨어지므로 ①②가 붙은 뒤에 켠다.
#   STAGE_VARIATION: 마찰 재샘플·확대 + 공 물성 랜덤 + 킥 반동 외란.
#                    ★ 지형 기하는 포함하지 않는다 (파일 상단 TERRAIN_ROUGH 별도)
#
# ── 2026-08-19 기본값 재설정 ────────────────────────────────────────────────
#   관찰: 전압 안정 / 킥 자체는 성공 / "지면·반동이 매번 다르다".
#   → 계통 오차가 아니라 **분산** 문제다. 강건성 스택을 기본으로 켠다.
#   로봇을 못 돌리는 동안에는 실기 A/B 가 불가능하므로 "한 번에 하나" 의 이득이
#   작다. 대신 sim 실패율(Episode_Termination)과 -Stress-* 태스크로 검증한다.
#
#   ※ 2026-08-19 재검토: STAGE_VARIATION 을 TERRAIN_ROUGH 와 **분리**했다.
#     원래 하나로 묶어 뒀는데, 지형 요철은 공 스폰·구름 문제로 껐지만
#     마찰/공물성/반동 외란은 공을 안 깨뜨리면서 같은 목적(변동 주입)을
#     달성하므로 켜 두어야 한다. 묶어 두면 하나를 끄면서 나머지까지 꺼진다.
STAGE_ENVELOPE = True    # 보험 성격. 킥이 잘 되므로 주범은 아니지만 유지.
STAGE_MARGIN = True      # 변동을 흡수할 자세 마진 — 이번 증상에 직결.
STAGE_HARD = True        # ★ 주 레버. sim 실패율이 10~15% 로 오르는 게 정상.
STAGE_HEIGHT = True      # 반동을 다리로 흡수할 여유 확보.
STAGE_VARIATION = True   # ★ 주 레버. 마찰/공물성/반동 변동 (지형 기하와 무관).

# ------------------------------------------------------------
# ★ 실측 모터 스펙 (XM540-W150, ROBOTIS 공식값)
# ------------------------------------------------------------
#   12.0V : 스톨 7.3 N·m / 무부하 53 rpm (= 5.55 rad/s)
#   11.1V : 스톨 6.9 N·m / 무부하 50 rpm (= 5.24 rad/s)  ← 배터리 새그 시
#   연속 정격 : 1.46 N·m (스톨의 20%)
#
#   τ(ω) = stall · (1 − ω/ω_noload) 표 (12V):
#       ω=0.0 → 7.30   ω=2.0 → 4.67   ω=3.0 → 3.35
#       ω=3.5 → 2.70   ω=4.0 → 2.04   ω=4.5 → 1.38  [N·m]
#   현재 sim actuator 는 ω 무관 5.0 N·m 평평 → ω=4.0 에서 실기의 2.5배.
#
#   ★ 보수적으로 새그 조건(11.1V)을 기준으로 잡는다. 경기 후반 배터리가
#     떨어진 상태에서도 되어야 하기 때문.
MOTOR_STALL_TORQUE = 6.9      # [N·m] 11.1V 기준 (12V 면 7.3)
MOTOR_NO_LOAD_SPEED = 5.24    # [rad/s] 50 rpm (12V 면 5.55)
MOTOR_EFFORT_CAP = 5.0        # [N·m] sim actuator effort 와 일치시킬 것
MOTOR_ENVELOPE_MARGIN = 0.8   # 곡선의 80% 안에서 놀게 (실성능 < 스톨 스펙)
JOINT_VEL_SOFT_LIMIT = 3.5    # [rad/s] 이 위는 가용토크 2.3 N·m 미만 영역

# 킥 직후 안정성 압력을 키울 구간 [s]. 전도는 이 구간에서 난다.
KICK_STAB_WINDOW_S = 1.0

_W_ENVELOPE = -0.02 if STAGE_ENVELOPE else 0.0   # 제곱합이라 스케일 작게
_W_JOINTVEL = -0.02 if STAGE_ENVELOPE else 0.0
_W_CAPTURE = 1.0 if STAGE_MARGIN else 0.0
_W_SLIP = -0.5 if STAGE_MARGIN else 0.0
_W_FLIGHT = -1.0 if STAGE_MARGIN else 0.0
_W_KICK_ANGVEL = -0.10 if STAGE_MARGIN else 0.0

# ------------------------------------------------------------
# ★ 버전 호환 — mdp.randomize_joint_parameters 존재 확인
# ------------------------------------------------------------
# EventTerm 은 클래스 정의 시점에 func 속성에 접근하므로, IsaacLab 버전에 이
# 함수가 없으면 import 단계에서 AttributeError 로 죽는다. 미리 확인해서 없으면
# no-op 을 꽂고 경고만 남긴다(그래야 나머지 DR 은 정상 동작한다).
# IsaacLab 2.3.x 에는 존재하므로 정상 환경에서는 그대로 쓰인다.
_JOINT_PARAM_FUNC = getattr(
    mdp, "randomize_joint_parameters", kick_stability_mdp.noop_event
)


# ================================================================
# Scene — v0_3 씬 + 공 (옛 walk_kick)
# ================================================================
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

    # 공 — 옛 walk_kick 그대로
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.2,
                angular_damping=0.2,
                max_linear_velocity=20.0,
                max_angular_velocity=50.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6, restitution=0.8,
                # combine="max" → 공-발 충돌에서 공의 높은 반발이 채택되어
                # 탄성 킥이 된다 (기존 multiply×발(~0) = 비탄성 → 공 안 튕김).
                restitution_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.4, 0.1)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, BALL_RADIUS)),
    )

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
# Events — v0_3 보행 이벤트 + 옛 walk_kick 킥 이벤트
# ================================================================
@configclass
class EventCfg:

    # ★ v1_4 STAGE_VARIATION: 마찰 범위 확대 + per-episode 재샘플.
    #   기존은 mode="startup" 이라 env 당 1 회 고정 — 한 env 는 학습 내내 같은
    #   마찰만 본다. mode="reset" 으로 바꾸면 같은 env 가 매 에피소드 다른
    #   마찰을 겪으므로 "이 지면이 어떤지 모른 채로도 버티기" 를 학습한다.
    #   범위도 0.7~1.2 → 0.45~1.35. 인조잔디는 먼지·습기·잔디 눕는 방향에 따라
    #   실제 마찰 편차가 이보다 좁을 이유가 없다. 정책은 마찰을 관측하지 못하므로
    #   (적응이 아니라) 강건해지는 수밖에 없고, 그게 우리가 원하는 것이다.
    #
    #   ★ 지형 요철을 껐으므로 "지면이 매번 다르다" 는 이 항이 거의 전담한다.
    #     터레인 기하와 달리 공 스폰/구름을 건드리지 않으면서 미끄러짐·반작용
    #     변동을 그대로 만든다. 실제로 인조잔디에서 지배적인 변동원도 마찰이다.
    #
    #   ※ 성능 주의: randomize_rigid_body_material 은 CPU 텐서로 물성을 쓴다.
    #     IsaacLab 문서도 startup 사용을 권장한다. 4096 env 에서 매 리셋마다
    #     돌리면 FPS 가 눈에 띄게 떨어질 수 있다. 학습 속도가 30% 이상 느려지면
    #     num_buckets 를 64 → 16 으로 줄이거나, mode 를 "startup" 으로 되돌리고
    #     대신 범위만 넓게 유지할 것 (env 간 다양성은 남고 env 내 변동만 사라짐).
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset" if STAGE_VARIATION else "startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.45, 1.35) if STAGE_VARIATION else (0.7, 1.2),
            "dynamic_friction_range": (0.40, 1.10) if STAGE_VARIATION else (0.7, 1.0),
            "restitution_range": (0.0, 0.2) if STAGE_VARIATION else (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    # ★ v1_4 STAGE_VARIATION: 공 물성 랜덤화.
    #   sim 은 질량 0.15 kg / 마찰 0.8 / 반발 0.8 고정이라 임팩트 반동이 거의
    #   결정적이다. 실기는 공 상태(공기압·표면·젖음)와 접촉점이 매번 달라
    #   같은 스윙에도 반동이 다르게 온다. "킥 후 반동이 매번 다르다" 의 직접
    #   대응항이다. 반발을 낮은 쪽으로도 열어야 (=공이 덜 튀어 발에 오래 붙는)
    #   반동이 큰 케이스까지 커버된다.
    randomize_ball_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (0.5, 1.1) if STAGE_VARIATION else (0.8, 0.8),
            "dynamic_friction_range": (0.35, 0.85) if STAGE_VARIATION else (0.6, 0.6),
            "restitution_range": (0.55, 0.9) if STAGE_VARIATION else (0.8, 0.8),
            "num_buckets": 32,
        },
    )

    randomize_ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (0.8, 1.3) if STAGE_VARIATION else (1.0, 1.0),
            "operation": "scale",
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

    reset_joints_noise = EventTerm(
        func=mdp.reset_joints_additive_noise,
        mode="reset",
        params={
            "noise_range": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ── 킥 이벤트 (옛 walk_kick) ★ 순서 중요: reset_base 다음 ★ ──────────
    #   ball → target → kick_state 순. 로봇 랜덤화된 root 를 기준으로 놓는다.
    # 극좌표 스폰: 거리 0.3~1.0 m, 방위 ±60° (정면 기준)
    reset_ball = EventTerm(
        func=kick_mdp.reset_ball_around_robot,
        mode="reset",
        params={
            "radius_range": BALL_DIST_RANGE,
            "front_angle_range": BALL_BEARING_RANGE,
            "height": BALL_RADIUS,
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    reset_target = EventTerm(
        func=kick_mdp.reset_kick_target,
        mode="reset",
        params={
            "distance_range": TARGET_DISTANCE,
            "target_angle_range": TARGET_DIR_ANGLE_RANGE,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # (킥세기 상수 버전 — kick_strength 상수 1.0, reset_kick_strength 제거)

    # per-env no-kick 데드라인 = 공거리 / cmd_vx + margin
    # 자유 내비게이션은 직진이 아니라 곡선 경로(몸 틀어 접근)라 실제 이동
    # 거리가 초기 직선거리보다 길다. margin·cap 을 소폭 완화해 옆 공을
    # 도는 시간을 보장 (margin 1.0→2.0, cap 3.0→5.0).
    reset_kick_state = EventTerm(
        func=kick_mdp.reset_kick_state_from_cmd,
        mode="reset",
        params={
            "command_name": "base_velocity",
            "margin_s": 2.0,
            "min_walk_speed": 0.1,
            "max_deadline_s": 5.0,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # reset 후 0.3 s 정책 출력 마스킹 (DelayedJointPositionAction 이 수행)
    reset_hold = EventTerm(
        func=mdp.set_reset_hold_steps,
        mode="reset",
        params={"hold_time_s": 0.3},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),  # 3~5 → 1~3 (외란 잦게, 회복훈련↑, 2026-08-14)
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )

    # ── 실기 강건성 DR 추가 (2026-08-14) — 킥 후 균형/sim2real ──────────
    #   CoM/외란토크는 현재 isaaclab 확정 존재. mass/actuator_gains 는
    #   버전에 따라 없을 수 있어 빌드검증 필요(없으면 그 항목만 드롭).
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_BASE_LINK),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.03, 0.03)},
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_BASE_LINK),
            "force_range": (-1.5, 1.5),
            "torque_range": (-0.1, 0.1),
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_BASE_LINK),
            "mass_distribution_params": (-0.3, 0.5),
            "operation": "add",
        },
    )

    # ★ v1_4: gain 랜덤 폭 확대 (STAGE_HARD).
    #   ±5% 는 실기 불확실성에 비해 너무 좁다. Dynamixel 의 내부 P/D 게인이
    #   sim stiffness/damping 으로 정확히 환산된다는 보장이 없고, 개체차·온도·
    #   전압에 따라 실효 게인이 변한다. ±20% 로 넓혀 게인 오차에 강건하게 만든다.
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MX_LEG_JOINTS),
            "stiffness_distribution_params": (0.80, 1.20) if STAGE_HARD else (0.95, 1.05),
            "damping_distribution_params": (0.80, 1.20) if STAGE_HARD else (0.95, 1.05),
            "operation": "scale",
        },
    )

    # ★ v1_4: armature / joint friction 랜덤 (STAGE_HARD)
    #   armature = N²·J_rotor 는 sim2real 에서 가장 불확실하고 가장 영향이 큰
    #   파라미터다(XM540 은 N=152.3 이라 reflected inertia 가 지배적).
    #   현재 0.05 를 "정답" 으로 가정하고 학습하면 실측이 0.08 일 때 킥 같은
    #   고가속 동작에서 바로 어긋난다. 폭넓게 랜덤화해 값 자체에 둔감하게 만든다.
    #   ※ IsaacLab 버전에 따라 randomize_joint_parameters 인자명이 다를 수 있다.
    #     빌드 에러가 나면 이 항만 주석 처리하고 나머지를 먼저 검증할 것.
    randomize_joint_params = EventTerm(
        func=_JOINT_PARAM_FUNC,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MX_LEG_JOINTS),
            "armature_distribution_params": (0.6, 1.6) if STAGE_HARD else (1.0, 1.0),
            "friction_distribution_params": (0.6, 1.6) if STAGE_HARD else (1.0, 1.0),
            "operation": "scale",
        },
    )

    # ★ v1_4: IMU 상수 바이어스 (STAGE_HARD)
    #   grav 0.05 ≈ 2.9° 장착/캘리브 오차. gyro 0.05 rad/s 바이어스.
    reset_imu_bias = EventTerm(
        func=kick_stability_mdp.reset_imu_bias,
        mode="reset",
        params={
            "grav_bias_range": 0.05 if STAGE_HARD else 0.0,
            "gyro_bias_range": 0.05 if STAGE_HARD else 0.0,
        },
    )

    # ★ v1_4: 킥 직후 표적 외란 (STAGE_HARD)
    #   기존 push_robot 은 1~3 초 간격 무작위라 정작 전도가 나는 "킥 직후 0.6 초"
    #   에 걸릴 확률이 낮다. 실패가 나는 구간을 직접 겨냥해 훈련한다.
    push_on_kick = EventTerm(
        func=kick_stability_mdp.push_robot_on_kick,
        mode="interval",
        interval_range_s=(0.02, 0.02),   # 매 스텝 평가, 내부에서 확률 게이팅
        params={
            # ★ 2026-08-19: STAGE_HARD → STAGE_VARIATION 으로 승격.
            #   "킥 후 반동이 매번 다르다" 가 주 증상이므로, 이 항이 그 변동을
            #   직접 대표한다. 확률 0.03 이면 0.6 s(30스텝) 안에 약 60%.
            #   세기 0.45 m/s 는 로봇 질량 대비 꽤 큰 외란이지만, 이걸 버티는
            #   정책이라야 실기의 미지 반동을 받아낸다.
            "post_kick_window_s": 0.6,
            "probability": 0.03 if STAGE_VARIATION else 0.0,
            "velocity_range": {"x": (-0.45, 0.45), "y": (-0.45, 0.45)},
        },
    )


# ================================================================
# Commands — 킥 태스크(정면 고정) + v0_3 gait 위상
# ================================================================
@configclass
class CommandsCfg:
    # 자유 내비게이션 재설계:
    #   base_velocity 는 "추종 목표" 가 아니다. 역할은 두 가지뿐 —
    #     (a) 전진 속도 상한(cap) 기준값 (planar_speed_cap_penalty 가 참조)
    #     (b) gait 걷기/정지 스위치 (norm<0.1 이면 gait 정지)
    #   정책은 vy/wz 를 자유롭게 써서 공으로 접근한다 (track 리워드 제거).
    #   ★ sim2real: 킥 후 cmd 를 0 으로 바꾸지 않는다. base_velocity 는
    #     에피소드 내내 상수(킥 전 값 유지). 실 로봇에선 킥 순간을 못 잡아
    #     정확한 타이밍에 0 을 줄 수 없으므로, 그냥 상수 상한만 준다.
    #     (킥 후 깔끔한 정지는 sim2real 안전한 방식으로 나중에 재설계)
    #   정지 env 없음(0.0): 킥 태스크는 항상 공으로 걸어감.
    # 원조 walk_kick 방식 (2026-08-14): cmd 전진속도 0.1~0.5 양수 → track 추종으로
    #   앞으로 걸어가 접근/킥. 킥 후 정지 없음(cmd 상수 유지, 실기안전). y=나중에.
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )

    # ★ 선언 순서 중요 ★ gait 정지판정이 base_velocity 를 읽는다.
    gait = mdp.PhaseOnlyGaitCommandCfg(
        resampling_time_range=(10.0, 10.0),
        frequency=2.0,
        stop_rate=2.0,
        include_frequency=True,
        frequency_scale=2.0,
        rel_stop_when_standing=0.6,
        velocity_command_name="base_velocity",
        standing_threshold=0.1,
        reset_phase_on_resample=True,
        freeze_during_reset_hold=True,
        debug_vis=False,
    )


# ================================================================
# Actions — v0_3 (DelayedJointPositionAction)
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
        # ★ v1_4: 지연 1~2(20~40ms) → 1~4(20~80ms) (STAGE_HARD).
        #   실기 지연 = 정책추론 + ROS2 + U2D2 시리얼 + SyncWrite + 서보 내부
        #   제어주기. 18축 SyncWrite 왕복까지 더하면 40ms 는 낙관적이다.
        #   실측해서 값을 맞추는 게 정석이고, 실측 전까지는 넓게 잡는다.
        action_delay_range=(1, 4) if STAGE_HARD else (1, 2),
    )


# ================================================================
# Observations — v0_3 보행 관측 + 옛 walk_kick 킥 관측
# ================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ★ 2026-08-22 수정: 자이로 바이어스 배선 누락.
        #   reset_imu_bias 가 gyro_bias_range 로 env.imu_gyro_bias 를 매 리셋
        #   샘플하는데, 이 항이 mdp.base_ang_vel(원본)이라 그 버퍼를 읽는 곳이
        #   없었다 → 자이로 상수 바이어스가 학습에 전혀 반영되지 않았다.
        #   (projected_gravity 만 바이어스가 걸려 있었다.)
        #   base_ang_vel_biased 로 교체. STAGE_HARD=False 면 폭 0 이라
        #   mdp.base_ang_vel 과 수학적으로 동일하고, 차원도 (N,3) 그대로다.
        base_ang_vel = ObsTerm(
            func=kick_stability_mdp.base_ang_vel_biased,
            scale=0.2,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        # ★ v1_4: i.i.d. 노이즈 + **에피소드 상수 바이어스**.
        #   실기 IMU 오차는 i.i.d. 가 아니다. 장착각 오차/캘리브 잔차는 에피소드
        #   내내 일정하고, Madgwick 은 킥의 큰 선형가속도를 중력으로 오인한다.
        #   i.i.d. 노이즈는 history_length=5 스택에서 상쇄돼 정책이 무시하는 법을
        #   배우지만, 상수 바이어스는 상쇄되지 않는다 → 실기 1~3° 오차에 취약해진다.
        #   STAGE_HARD=False 면 바이어스 폭이 0 이라 mdp.projected_gravity 와 동일.
        projected_gravity = ObsTerm(
            func=kick_stability_mdp.projected_gravity_biased,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        # 명령 관측 = 상수(reset_hold 때만 0). 킥 후 0화 제거 — sim2real:
        # 실 로봇에선 킥 순간을 잡아 cmd를 정확히 0으로 줄 수 없으므로,
        # base_velocity를 킥 전 값 그대로 유지한다(정책이 보는 값도 상수).
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

        # ── 킥 관측 (옛 walk_kick) ────────────────────────────────────
        ball_pos_b = ObsTerm(
            func=kick_mdp.ball_pos_xy_in_robot_frame_frozen,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        target_dir_b = ObsTerm(
            func=kick_mdp.target_dir_in_robot_frame,
            noise=Unoise(n_min=-0.052, n_max=0.052),
        )
        kick_strength = ObsTerm(func=kick_mdp.ones_obs, params={"dim": 1})

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

        # 킥 관측 (critic 은 특권 정보 ball_vel_b 추가)
        ball_pos_b = ObsTerm(func=kick_mdp.ball_pos_xy_in_robot_frame)
        # 2D(xy) 공속도 — 미러 스펙(2D)과 정합. 옛 ball_lin_vel_in_robot_frame
        # 은 3D(x,y,z)라 미러 base_dim 과 안 맞아, 2D 버전을 쓴다 (critic
        # 특권 관측이라 z 성분 불필요, ball_z 페널티가 수직은 따로 처리).
        ball_vel_b = ObsTerm(func=kick_mdp_v0.ball_vel_b)
        target_dir_b = ObsTerm(func=kick_mdp.target_dir_in_robot_frame)
        kick_strength = ObsTerm(func=kick_mdp.ones_obs, params={"dim": 1})

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ================================================================
# Rewards — v0_3 보행 리워드 + 옛 walk_kick 킥 리워드
# ================================================================
@configclass
class RewardsCfg:
    # -- 속도 추종 복원 (원조 walk_kick 방식, 2026-08-14) --
    #   cmd 전진속도(0.1~0.5)를 추종해 앞으로 걸어감 → 정면 접근/킥.
    #   hold_aware(kick_aware 아님) = 킥 후에도 계속 추종(정지 없음, 실기안전).
    #   ※ 옆공 접근 위해 x(전진)만 추종, y(옆)는 자유 (2026-08-15, wz처럼 풀음).
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_hold_aware,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_hold_aware,
        weight=0.0,  # wz 추종 끔(1.0→0): 옆공/꺾인목표 위해 회전 자유 (2026-08-15)
        params={"command_name": "base_velocity", "std": 0.25},
    )
    # speed_cap 끔(track 이 속도 제어 담당). 0 으로 비활성.
    speed_cap = RewTerm(
        func=kick_mdp_v0.planar_speed_cap_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base 안정화 (v0_3) --
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 안정화 스무딩(2026-08-12): 강한 3.0 킥에서 resume 하며 완만화.
    #   action_rate −0.4→−0.6 (추가 완만화, model_3100 에서 이어서).
    #   joint_acc −2.5e−6 유지. (resume 이라 킥 보존)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.6)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    base_roll_l2 = RewTerm(func=mdp.base_roll_l2, weight=-5.0)
    base_pitch_target_l2 = RewTerm(
        func=mdp.base_pitch_target_l2,
        weight=-1.0,
        params={"target_pitch": math.radians(0.0)},
    )
    # ★ v1_4: 데드존 옵션 (STAGE_HEIGHT).
    #   weight −150 의 평지붕 벌점은 킥 반작용을 흡수할 지지무릎 굽힘 여유를
    #   뺏는다(2cm 만 내려가도 −0.06 ×150 = −9). 실기에서 충격을 다리로 흡수
    #   못 하면 그대로 몸통 각운동량이 되어 넘어진다.
    #   ±0.025 m 는 무벌점으로 두어 "살짝 앉으며 받아내기" 를 허용한다.
    base_height = RewTerm(
        func=mdp.base_height_l2_deadzone if STAGE_HEIGHT else mdp.base_height_l2,
        weight=-150.0,
        params=(
            {"target_height": 0.39, "deadzone": 0.025}
            if STAGE_HEIGHT
            else {"target_height": 0.39}
        ),
    )

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

    # 골반 yaw 편차 벌점 — 공 근처에서 해제 가능(STAGE_HIPYAW).
    #   인사이드 킥은 정의상 골반 yaw 를 돌리는 동작인데, 이 항이 그걸 직접 벌한다.
    #   45° = 0.785 rad → 스텝당 -0.39. 킥 자세를 만들 때마다 벌을 받으니
    #   "골반은 안 돌리고 발목만 트는" poke 로 흐르기 쉽다(지난 45° 트위스트 실패).
    #   release_radius=0.0 이면 mdp.joint_deviation_l1 과 동일 동작(스위치 OFF).
    joint_deviation_hip_yaw = RewTerm(
        func=kick_inside_mdp.joint_deviation_l1_ball_gated,
        weight=-0.5,
        params={
            "release_radius": _HIPYAW_RELEASE_R,
            "falloff": 0.15,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["r_pelvis_yaw_joint", "l_pelvis_yaw_joint"],
            ),
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
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

    # 킥 실제 속도(오버슛) 캡 → 파워=윈드업 축소 (2026-08-13).
    #   ball_velocity 는 3.0 에서 보상만 포화(초과 벌점 없음) → 실제 킥 4+ m/s.
    #   목표 3.0 초과분을 벌해 실제 킥을 3.0 으로 캡 → 윈드업/다리 앞뒤벌림 감소.
    #   3.0 목표(더 낮추면 드리블로 흐름). feet_fb(간접·너무 약함)는 제거하고 대체.
    ball_overspeed = RewTerm(
        func=kick_mdp.ball_overspeed_penalty,
        weight=-2.0,  # model_14500 학습값. −5 는 이득없어 되돌림 (2026-08-13)
        params={"target_speed": 3.0, "post_kick_window_s": 0.4},
    )

    # 먼 발(공 반대쪽 발) 킥 억제 → 양발 사용 (오른발 편향 교정, 2026-08-13).
    #   공 왼쪽(frozen+y)인데 오른발 / 오른쪽인데 왼발로 차면 벌. 미러 대칭이라
    #   미러손실과 안 싸움. 킥 접촉 순간에만 걸려 희석 안 됨.
    kick_wrong_foot = RewTerm(
        func=kick_mdp_v0.kick_wrong_foot_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_FEET, preserve_order=True),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    pelvis_pitch_symmetry = RewTerm(
        func=mdp.pelvis_pitch_symmetry_windowed,
        weight=-1.0,
        params={
            "window_size": 30,
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    # -- 보행 위상 (v0_3) --
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
    metric_phase = RewTerm(
        func=mdp.metric_phase_v01,
        weight=0.0,
        params={"gait_command_name": "gait"},
    )
    stop_posture = RewTerm(
        func=mdp.stop_posture_v01,
        weight=0.15,
        params={
            "gait_command_name": "gait",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=MX_FEET),
        },
    )
    stop_default_posture = RewTerm(
        func=mdp.stop_default_posture_v01,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MX_LEG_JOINTS),
            "gait_command_name": "gait",
            "include_stopping": False,
        },
    )
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

    foot_flat = RewTerm(
        func=mdp.foot_flat_orientation,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_FEET),
            "command_name": None,
        },
    )

    # ================================================================
    # 킥 리워드 — 옛 walk_kick 그대로
    # ================================================================
    # ── 킥 세기: 최대 파워 proj² (킥세기 상수 버전 = model_8800 구성) ──────
    # 파워 조절(kick_power_match)은 실패(드리블/정체)로 폐기. 항상 최대로
    # 세게 proj², 탄성 공에서 strike 학습된 그 구성. max_speed 6.0(탄성으로
    # 공이 발속도 초과), post_kick_window 0.4(임펄스만 보상).
    ball_velocity = RewTerm(
        func=kick_inside_mdp.ball_velocity_inside_gated,
        weight=2.0,
        params={
            "max_speed": 3.0,
            "post_kick_window_s": 0.4,
            # 인사이드가 아닐 때 남는 최소 배율. 0 으로 두면 학습 초기에
            # "아직 인사이드가 아니니 아예 차지 말자" 로 붕괴한다.
            "quality_floor": _Q_FLOOR,
            "min_strike_speed": INSIDE_MIN_STRIKE_SPEED,
            "window_steps": INSIDE_WINDOW_STEPS,
            "asset_cfg": _feet_cfg(),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # ================================================================
    # ★ v1_3 인사이드 킥 (2026-08-16)
    #   v1_2 의 dense 3종(더하기)을 폐기하고 "파워 × 품질(곱하기)" 로 재설계.
    #   남은 항은 탐색 보조(swing) + 드리블 차단(touch_budget) + 정확도(offaxis).
    # ================================================================
    # 스윙 shaping — 래치된 (인사이드 품질 × 정규화 타격속도) 를 윈도우 동안 dense.
    #   ball_velocity 만으로는 "안쪽면 정렬 스윙" 을 초기에 못 찾는다(신호가 늦음).
    #   farm 방지: 에피소드당 1회 래치 + 강타에서만 래치 + 총량 상한 = window_steps.
    inside_swing = RewTerm(
        func=kick_inside_mdp.inside_swing_shaping,
        weight=_W_SWING,
        params={
            "ref_speed": 2.5,
            "min_strike_speed": INSIDE_MIN_STRIKE_SPEED,
            "window_steps": INSIDE_WINDOW_STEPS,
            "asset_cfg": _feet_cfg(),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    # 접촉 지속시간 예산 — 드리블 직접 차단.
    #   정상 강타 접촉 = 3~5스텝(0.06~0.10s) → 벌점 0
    #   드리블        = 수십~150스텝        → 스텝당 -1.0 (100스텝이면 -100)
    touch_budget = RewTerm(
        func=kick_inside_mdp.touch_budget_penalty,
        weight=-1.0,
        params={
            "free_steps": INSIDE_TOUCH_FREE_STEPS,
            "min_strike_speed": INSIDE_MIN_STRIKE_SPEED,
            "window_steps": INSIDE_WINDOW_STEPS,
            "asset_cfg": _feet_cfg(),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    # 공이 목표에서 옆으로 새는 성분 벌 → 방향 정확도.
    #   weight 는 작게. -1.0 은 "안 차는 게 이득" 이 되어 하드킥을 죽였다(v1_2 밤튜닝).
    ball_offaxis = RewTerm(
        func=kick_inside_mdp.ball_offaxis_penalty,
        weight=-0.3,
        params={"post_kick_window_s": 0.4, "ball_cfg": SceneEntityCfg("ball")},
    )

    # (제거 2026-08-16) v1_2 의 kick_push / kick_inside_align 삭제.
    #   dense 접촉 보상이라 최적해가 "접촉시간 최대화" = 인사이드 드리블이었다.
    #   두 항의 역할은 ball_velocity 의 품질 배율 + inside_swing 이 대체한다.

    # (제거 2026-08-15) early_kick 벌점(-1) + kick_time_gauss(12) 삭제 — 조기킥 억제는
    #   종료 early_kick_done 로 일원화. 둘 다 타이밍용이었으나 종료가 대체 → 중복.
    #   함수(early_kick_penalty / kick_time_gauss_reward)는 kick_mdp.py 에 dormant 로 남음.

    ball_z_velocity = RewTerm(
        func=kick_mdp.ball_z_velocity_l2,
        weight=-2.0,
    )
    # 킥 + 정지(hold) 성공 시 +8. dense farming 대신 결정적 단일 킥 유도.
    kick_success_bonus = RewTerm(
        func=kick_mdp.kick_then_stand_bonus,
        weight=8.0,
        params={
            "hold_time_s": KICK_HOLD_TIME_S,
            "speed_threshold": KICK_SPEED_THRESHOLD,
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    # 로봇↔공 거리 어트랙터 exp(-d/std). 킥 후 1.0 잠금은 함수 내부.
    # ★ 자유 내비게이션의 주동력 ★ — 추종 리워드를 뗐으므로 이 항이
    #   "공까지 걸어가라" 를 담당한다. std 0.3 은 1.0m 에서 0.036 으로
    #   먼 공 유인이 거의 없어(추종이 대신 끌던 것) std 1.0 으로 넓힌다:
    #     d=1.0→0.37, d=0.5→0.61, d=0.3→0.74. 전 범위에 유의미한 gradient.
    #   weight 도 0.5→1.5 (제거된 추종 2.0 만큼 접근 구동 보강).
    ball_approach = RewTerm(
        func=kick_mdp.ball_approach_reward,
        weight=1.5,
        params={
            "std": 1.0,
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    # 킥 후 다리 원위치 → 깔끔한 정지
    joint_pose_to_default = RewTerm(
        func=kick_mdp.joint_pose_to_default,
        weight=0.5,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MX_LEG_JOINTS),
        },
    )

    # ================================================================
    # ★ 메트릭 (weight=0.0) — 학습에 영향 없음, TensorBoard 검증 전용
    # ----------------------------------------------------------------
    # 학습 곡선만으로는 드리블 / poke / 인사이드킥을 구분할 수 없다.
    # v1_2 의 실패를 며칠 늦게 알아챈 이유가 이거라, 이번엔 항상 켜 둔다.
    #
    #   판정표                드리블   poke   목표(인사이드킥)
    #   metric_touch_steps     40~150   3~5     3~6
    #   metric_max_run         40~150   3~5     3~6
    #   metric_latch_rate      낮음     높음    높음
    #   metric_inside_q        0.3~0.6  <0.3    >0.6
    #   metric_ball_dir_err    10~40°   15~30°  <8°
    #
    # metric_* 는 파라미터를 받지 않는다 — kick_inside_mdp._update_state 의
    # "스텝당 1회" 가드와 얽혀 파라미터 불일치 함정이 생기는 걸 막기 위함.
    # 버퍼를 읽기만 하므로 선언 순서에 따라 1스텝 늦을 수 있으나 로깅엔 무해.
    # ================================================================
    metric_inside_q = RewTerm(
        func=kick_inside_mdp.metric_inside_quality, weight=0.0
    )
    metric_latch_rate = RewTerm(
        func=kick_inside_mdp.metric_latch_rate, weight=0.0
    )
    # ★ INSIDE_MIN_STRIKE_SPEED 튜닝의 직접 근거 ★
    #   latch_rate 가 낮은데 max_vlong 이 1.2 근처 정체 → 문턱을 1.2 아래로.
    #   latch_rate 는 높은데 드리블이 남음 → 문턱이 낮아 느린 터치가 래치 중 → 올림.
    metric_max_vlong = RewTerm(
        func=kick_inside_mdp.metric_max_vlong, weight=0.0
    )
    metric_touch_steps = RewTerm(
        func=kick_inside_mdp.metric_touch_steps, weight=0.0
    )
    # 누적(touch_steps)은 큰데 최장런(max_run)이 짧으면 드리블이 아니라
    # "여러 번 툭툭 건드림" 이다. INSIDE_TOUCH_FREE_STEPS 튜닝 근거.
    metric_max_run = RewTerm(
        func=kick_inside_mdp.metric_max_run, weight=0.0
    )
    # 최종 목표 지표 — 공 진행방향의 목표 대비 각도 오차 [deg]
    metric_ball_dir_err = RewTerm(
        func=kick_inside_mdp.metric_ball_dir_err_deg,
        weight=0.0,
        params={"min_speed": 0.5, "ball_cfg": SceneEntityCfg("ball")},
    )

    # ================================================================
    # ★ v1_4 안정성 항 (2026-08-19) — 킥 후 실기 전도 대응
    # ================================================================
    # ① 모터 속도-토크 포락선 위반 벌점 (STAGE_ENVELOPE)
    #   sim 은 ω 무관 5.0 N·m 를 주지만 실기는 ω=4.0 에서 1.6 N·m 뿐이다.
    #   dynamics 는 그대로 두고 "실현 불가 영역" 에만 비용을 매겨, 기존
    #   체크포인트에서 파인튜닝으로 실현 가능 영역 안으로 밀어 넣는다.
    torque_envelope = RewTerm(
        func=kick_stability_mdp.torque_envelope_penalty,
        weight=_W_ENVELOPE,
        params={
            "stall_torque": MOTOR_STALL_TORQUE,
            "no_load_speed": MOTOR_NO_LOAD_SPEED,
            "effort_cap": MOTOR_EFFORT_CAP,
            "margin": MOTOR_ENVELOPE_MARGIN,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=MX_LEG_JOINTS, preserve_order=True
            ),
        },
    )
    # 포락선 벌점의 강건 버전 — applied_torque 구현에 안 의존하고 속도만 본다.
    joint_vel_limit = RewTerm(
        func=kick_stability_mdp.joint_vel_over_limit_penalty,
        weight=_W_JOINTVEL,
        params={
            "vel_limit": JOINT_VEL_SOFT_LIMIT,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=MX_LEG_JOINTS, preserve_order=True
            ),
        },
    )

    # ② Capture point 마진 (STAGE_MARGIN)
    #   자세각 벌점은 "이미 기울어진 뒤" 반응한다. 킥 전도는 자세가 멀쩡한데
    #   수평 속도가 이미 지지면 밖을 향해서 일어난다. capture point 는 그
    #   속도를 위치로 환산하므로 한 박자 먼저 반응한다.
    capture_point = RewTerm(
        func=kick_stability_mdp.capture_point_margin_reward,
        weight=_W_CAPTURE,
        params={
            "std": 0.10,
            "post_kick_window_s": KICK_STAB_WINDOW_S,
            "boost": 3.0,
            "force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=MX_FEET,
                                         preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_FEET,
                                        preserve_order=True),
        },
    )
    # 지지발 미끄러짐 — sim 마찰(0.7~1.2)이 실제 필드보다 후하면 sim 에서만 버틴다.
    support_slip = RewTerm(
        func=kick_stability_mdp.support_foot_slip_penalty,
        weight=_W_SLIP,
        params={
            "force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=MX_FEET,
                                         preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_FEET,
                                        preserve_order=True),
        },
    )
    # 양발 동시 공중 — 그 순간부터 어떤 제어도 자세를 못 바꾼다. 단각지지는 정상.
    flight_phase = RewTerm(
        func=kick_stability_mdp.flight_phase_penalty,
        weight=_W_FLIGHT,
        params={
            "post_kick_window_s": KICK_STAB_WINDOW_S,
            "boost": 3.0,
            "force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=MX_FEET,
                                         preserve_order=True),
        },
    )
    # 킥 윈도우 각속도 — 팔이 고정(stiffness 0)이라 스윙 각운동량을 상쇄할
    # 자유도가 없다. 지지발 토크로 미리 상쇄하도록 그 구간만 강하게 벌한다.
    kick_ang_vel = RewTerm(
        func=kick_stability_mdp.base_ang_vel_kick_windowed,
        weight=_W_KICK_ANGVEL,
        params={"post_kick_window_s": KICK_STAB_WINDOW_S, "boost": 5.0},
    )

    # ── 안정성 메트릭 (weight=0.0) ─────────────────────────────────
    #   metric_cp_dist        : capture point 마진 [m]. 0.10 넘으면 위험 구간.
    #   metric_max_qd         : 최대 관절속도 [rad/s]. 3.5 넘으면 실기 토크 부족.
    #   metric_tau_violation  : 포락선 초과 토크 [N·m]. 0 이어야 실기 재현 가능.
    #   metric_tau_available  : 그 순간 실기 가용 토크 [N·m]. 1.5 아래면 위험.
    metric_cp_dist = RewTerm(
        func=kick_stability_mdp.metric_capture_point_dist,
        weight=0.0,
        params={
            "force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=MX_FEET,
                                         preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_FEET,
                                        preserve_order=True),
        },
    )
    metric_max_qd = RewTerm(
        func=kick_stability_mdp.metric_max_joint_vel,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=MX_LEG_JOINTS, preserve_order=True
            ),
        },
    )
    metric_tau_violation = RewTerm(
        func=kick_stability_mdp.metric_torque_envelope_violation,
        weight=0.0,
        params={
            "stall_torque": MOTOR_STALL_TORQUE,
            "no_load_speed": MOTOR_NO_LOAD_SPEED,
            "effort_cap": MOTOR_EFFORT_CAP,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=MX_LEG_JOINTS, preserve_order=True
            ),
        },
    )
    metric_tau_available = RewTerm(
        func=kick_stability_mdp.metric_available_torque_at_speed,
        weight=0.0,
        params={
            "stall_torque": MOTOR_STALL_TORQUE,
            "no_load_speed": MOTOR_NO_LOAD_SPEED,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=MX_LEG_JOINTS, preserve_order=True
            ),
        },
    )


# ================================================================
# Terminations — v0_3 보행 + 옛 walk_kick 킥
# ================================================================
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # ★ v1_4: 종료 문턱 강화 (STAGE_HARD)
    #   0.8 rad(46°)는 너무 관대하다. 실기는 30° 만 넘어도 회복이 거의 불가능한데
    #   sim 은 46° 까지 버티게 두니 "위험하게 기울었다가 살아남는" 해가 성공으로
    #   집계된다. 0.6 rad(34°)로 조이면 그 해들이 실패로 바뀌어 마진이 학습된다.
    #   ※ 이 한 줄이 sim 성공률을 가장 크게 떨어뜨린다. 의도된 것이다.
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.28 if STAGE_HARD else 0.25},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.6 if STAGE_HARD else 0.8},
    )
    # 킥 + hold(2 s) 후 종료 (성공)
    kick_resolved = DoneTerm(
        func=kick_mdp.kick_resolved_termination,
        params={
            "hold_time_s": KICK_HOLD_TIME_S,
            "speed_threshold": KICK_SPEED_THRESHOLD,
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    # 데드라인까지 안 차면 종료 (실패)
    no_kick_deadline = DoneTerm(
        func=kick_mdp.no_kick_timeout_termination,
        params={"fallback_s": 5.0},
    )
    # 조기킥 종료 (2026-08-15 개정): 시작 후 1초 이내 킥이면 종료 → 초반 즉시킥/런지만 억제.
    #   deadline 상대 방식이 타이밍 부적절 + 강한킥 억제(약한킥 평탄화) → 절대 1초 문턱으로.
    early_kick_done = DoneTerm(
        func=kick_mdp.early_kick_termination,
        params={
            "min_kick_time_s": 1.0,
            "speed_threshold": KICK_SPEED_THRESHOLD,
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )


# ================================================================
# Curriculum — 없음 (첫 패스; 명령 정면 고정이라 v0_3 속도 커리큘럼 부적합)
# ================================================================
@configclass
class CurriculumCfg:
    pass


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
        self.episode_length_s = 15.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # 공이 env 당 1 개 추가되므로 여유를 더 준다
        self.sim.physx.gpu_max_rigid_patch_count = 24 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10

        # play: 전진 0.3 고정 (공까지 걸어가 차는지 눈으로 확인)
        self.commands.base_velocity.ranges = (
            mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(0.3, 0.3),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            )
        )
        self.commands.base_velocity.rel_standing_envs = 0.0


# ================================================================
# 실기 대조용 고정 시나리오 (sim-to-real 비교)
#   ball_pos_b   = (0.40, 0.00)  ← 공 40cm 정면 고정 (frozen 관측)
#   target_dir_b = (1.0, 0.0)    ← 타겟 정면(각0)
#   cmd_vx       = 0.3           ← 전진 명령 0.3 (vy=wz=0)
#   외란(push)·물성 랜덤 끔 → 결정적. gym id: -V1-Fixed
# ================================================================
@configclass
class RobotFixedTestEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()  # cmd_vx=0.3 은 여기서 이미 설정됨

        # 공: 정면 40cm 고정 → ball_pos_b(frozen) = (0.40, 0.00)
        self.events.reset_ball.params["radius_range"] = (0.40, 0.40)
        self.events.reset_ball.params["front_angle_range"] = (0.0, 0.0)

        # 타겟: 정면(각0) → target_dir_b = (1.0, 0.0)
        self.events.reset_target.params["target_angle_range"] = (0.0, 0.0)

        # cmd_vx=0.3 재확인 (limit 도 0.3 고정 — 커리큘럼 확장 방지)
        self.commands.base_velocity.limit_ranges = (
            mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(0.3, 0.3),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            )
        )

        # 결정적 대조: 외란/도메인랜덤 끔
        self.events.push_robot = None
        self.events.physics_material = None

        # ★ v1_4 에서 추가된 확률적 이벤트도 전부 꺼야 "결정적" 이 유지된다.
        #   이걸 안 끄면 -Fixed 가 더 이상 결정적이지 않아 sim↔실기 대조와
        #   A/B 비교가 무의미해진다. 새 이벤트를 추가할 때마다 여기도 갱신할 것.
        self.events.push_on_kick = None            # 킥 직후 표적 외란
        self.events.randomize_ball_material = None  # 공 마찰/반발
        self.events.randomize_ball_mass = None      # 공 질량
        self.events.randomize_joint_params = None   # armature/joint friction
        self.events.reset_imu_bias = None           # IMU 상수 바이어스
        self.events.randomize_actuator_gains = None  # 게인 오차
        self.events.randomize_base_com = None        # CoM 오프셋
        self.events.add_base_mass = None             # 몸통 질량
        self.events.base_external_force_torque = None  # 초기 외력


# ================================================================
# 좌/우 발 사용 대칭성 체크 — 공을 옆쪽 원뿔(cone)로 스윕
#   L: 정면기준 +15°~+55°(왼쪽), R: -55°~-15°(오른쪽), 거리 0.40 고정.
#   각 env 가 이 범위에서 무작위 각을 뽑으므로 한 번의 play 로 옆 전체를
#   커버 → "어느 각도에서든 왼발을 쓰는가"를 결정적으로 관찰.
#   y 환산: 15°→ +0.10m, 55°→ +0.33m (거리 0.40 기준).
#   reset_ball 규약상 +각도 = 왼쪽(+y). target 각0·cmd_vx 0.3·외란 off 상속.
#   gym id: -V1-Fixed-L / -V1-Fixed-R
# ================================================================
_FBALL_DIST = 0.40
_FBALL_ANG_LO = math.radians(15.0)
_FBALL_ANG_HI = math.radians(55.0)


@configclass
class RobotFixedTestLEnvCfg(RobotFixedTestEnvCfg):
    """공을 왼쪽 원뿔(+15°~+55°, 거리0.40)로 스윕 → 왼발 쓰는지 확인."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_ball.params["radius_range"] = (_FBALL_DIST, _FBALL_DIST)
        self.events.reset_ball.params["front_angle_range"] = (_FBALL_ANG_LO, _FBALL_ANG_HI)


@configclass
class RobotFixedTestREnvCfg(RobotFixedTestEnvCfg):
    """공을 오른쪽 원뿔(-55°~-15°, 거리0.40)로 스윕 → 오른발 쓰는지 확인."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_ball.params["radius_range"] = (_FBALL_DIST, _FBALL_DIST)
        self.events.reset_ball.params["front_angle_range"] = (-_FBALL_ANG_HI, -_FBALL_ANG_LO)


# ================================================================
# ★ 스트레스 테스트 (2026-08-19) — 실기 없이 강건성을 재는 대체 수단
# ----------------------------------------------------------------
# 로봇을 못 돌리는 동안, "실기에서 나아졌는가" 를 대신할 지표가 필요하다.
# 조건을 하나씩 악화시킨 play 태스크를 두고 32 env 중 몇 개가 살아남는지 본다.
#
#   기준선: -V1-4-Play (기본 DR)
#   축별  : -Stress-Fric / -Stress-Delay / -Stress-Push / -Stress-Rough
#   종합  : -Stress-All  (경기장 최악 조건 가정)
#
# 【판정 방법】
#   play 를 돌리며 넘어지는 env 수를 세면 된다. 더 정확히는 rsl_rl 학습 로그의
#   Episode_Termination/bad_orientation + base_height 합을 보는 게 편하다.
#
# 【해석】
#   v1_3 정책을 -Stress-Fric 에 넣었을 때 성공률이 크게 떨어지면, 그게 바로
#   "지면 상태가 매번 다르면 넘어진다" 의 sim 재현이다. 실기 로그 없이도
#   원인이 확인되고, v1_4 학습 후 같은 태스크에서 회복되는지로 검증할 수 있다.
#
# 【주의】 이 클래스들은 PlayEnvCfg 를 상속하므로 num_envs=32 다.
#   통계를 원하면 --num_envs 로 256 정도까지 올려서 돌릴 것.
# ================================================================
@configclass
class RobotStressFrictionEnvCfg(RobotPlayEnvCfg):
    """저마찰 지면 고정 — 미끄러운 인조잔디/먼지 낀 매트 가정."""

    def __post_init__(self):
        super().__post_init__()
        # 범위가 아니라 낮은 값으로 고정해야 "저마찰에서의 성공률" 이 측정된다.
        self.events.physics_material.params["static_friction_range"] = (0.45, 0.55)
        self.events.physics_material.params["dynamic_friction_range"] = (0.40, 0.50)


@configclass
class RobotStressDelayEnvCfg(RobotPlayEnvCfg):
    """제어 지연 악화 — 60~100 ms. 통신 지연 실측 전 상한 확인용."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.JointPositionAction.action_delay_range = (3, 5)


@configclass
class RobotStressPushEnvCfg(RobotPlayEnvCfg):
    """킥 직후 반동 외란 강화 — "킥 후 반동이 매번 다르다" 의 sim 재현."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_on_kick.params["probability"] = 0.06
        self.events.push_on_kick.params["velocity_range"] = {
            "x": (-0.6, 0.6), "y": (-0.6, 0.6),
        }


@configclass
class RobotStressAllEnvCfg(RobotPlayEnvCfg):
    """종합 최악 조건 — 저마찰 + 지연 + 강한 반동 + 게인/관성 오차."""

    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material.params["static_friction_range"] = (0.45, 0.60)
        self.events.physics_material.params["dynamic_friction_range"] = (0.40, 0.55)
        self.actions.JointPositionAction.action_delay_range = (2, 5)
        self.events.push_on_kick.params["probability"] = 0.05
        self.events.push_on_kick.params["velocity_range"] = {
            "x": (-0.55, 0.55), "y": (-0.55, 0.55),
        }
        self.events.randomize_actuator_gains.params[
            "stiffness_distribution_params"] = (0.75, 1.25)
        self.events.randomize_actuator_gains.params[
            "damping_distribution_params"] = (0.75, 1.25)
        self.events.randomize_joint_params.params[
            "armature_distribution_params"] = (0.5, 1.8)


# ================================================================
# ★ W90 — 킥 목표 방향 ±60° → ±90° 확장 (2026-08-19)
# ----------------------------------------------------------------
# 【배경】 TARGET_DIR_ANGLE_RANGE 는 원래 ±90°(v1, 2026-08-13) 였다가
#   2026-08-15 에 ±60° 로 줄였다. 그 무렵 인사이드킥 분기가 "45°+ 정확도
#   미해결" 로 폐기된 기록이 v1_1 헤더에 남아 있다. 즉 넓은 각도에서
#   방향 정확도가 무너지는 것이 축소 사유였을 가능성이 높다.
#
# 【지금 다시 넓히는 근거】 그때와 상황이 다르다. v1_3 에서 파워×품질 곱
#   구조로 인사이드킥이 풀렸고, v1_4 (model_16700) 는 ±60° 에서
#   ball_dir_err 평균 4.25° / p90 8.9° 로 여유가 있다.
#
# 【주의】 목표 방향만 넓힌다. 공 스폰 방위(BALL_BEARING_RANGE)는 ±60° 그대로다.
#   둘 다 넓히면 공↔목표 상대각이 최악 150° 까지 벌어져 난이도가 급등한다.
#   또 no_kick_deadline 이 5.0 s 이므로, 90° 를 돌아서 접근·타격할 시간이
#   빠듯하면 max_deadline_s 를 먼저 늘려야 한다 (지금은 건드리지 않았다).
#
# 【판정】 학습 후 eval_inside_kick 의 ball_dir_err 을 ±60° 결과(4.25°)와
#   비교할 것. 크게 나빠지면 각도를 75° 정도로 되돌리거나, 커리큘럼
#   (CurriculumCfg 가 지금 비어 있으므로 신규 구현)이 필요하다는 뜻이다.
# ================================================================
TARGET_DIR_ANGLE_RANGE_W90 = (-math.radians(90.0), math.radians(90.0))


@configclass
class RobotW90EnvCfg(RobotEnvCfg):
    """학습용 — 킥 목표 방향 ±90°. 나머지는 v1_4 와 완전히 동일."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_target.params["target_angle_range"] = TARGET_DIR_ANGLE_RANGE_W90


@configclass
class RobotW90PlayEnvCfg(RobotPlayEnvCfg):
    """추론용 — 킥 목표 방향 ±90°."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_target.params["target_angle_range"] = TARGET_DIR_ANGLE_RANGE_W90


# ================================================================
# ★ W90B70 — 목표 방향 ±90° + 공 스폰 방위 ±70° (2026-08-19)
# ----------------------------------------------------------------
# W90 위에 공 스폰 방위(BALL_BEARING_RANGE)까지 ±60° → ±70° 로 넓힌다.
# 이 값은 reset_ball_around_robot 의 front_angle_range 로, 극좌표(거리+방위)
# 스폰의 방위각이다. 즉 "공의 좌우(y) 범위" 를 각도로 표현한 것.
#   거리 0.2~0.5 m 기준 좌우 최대 오프셋: 0.5·sin(70°) = 0.47 m (60°면 0.43 m)
#
# 【이력】 BALL_BEARING_RANGE 도 원래 ±70° 였다가 2026-08-15 에 ±60° 로
#   줄였다(TARGET_DIR_ANGLE_RANGE 90→60 과 같은 날). 되돌리는 것이다.
#
# 【난이도】 공↔목표 상대각이 최악 160°(공 -70° / 목표 +90°)까지 벌어진다.
#   이 조합은 로봇이 공을 크게 돌아 들어가야 하므로, no_kick_deadline
#   (공까지 걷는 시간 + 여유 2.0s, 상한 5.0s)이 빠듯해질 수 있다.
#   ★ 학습 시작 후 Episode_Termination/no_kick_deadline 을 반드시 볼 것.
#     단 초반 수십 iter 는 에피소드 통계 버퍼가 안 차서 0.7 같은 값이
#     나온다 — 그건 warm-up 이지 문제가 아니다(W90 에서 확인됨).
#     버퍼가 찬 뒤에도 높으면 max_deadline_s / fallback_s 를 5.0 → 7.0 으로.
# ================================================================
BALL_BEARING_RANGE_B70 = (-math.radians(70.0), math.radians(70.0))


@configclass
class RobotW90B70EnvCfg(RobotW90EnvCfg):
    """학습용 — 목표 ±90° + 공 스폰 방위 ±70°."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_ball.params["front_angle_range"] = BALL_BEARING_RANGE_B70


@configclass
class RobotW90B70PlayEnvCfg(RobotW90PlayEnvCfg):
    """추론용 — 목표 ±90° + 공 스폰 방위 ±70°."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_ball.params["front_angle_range"] = BALL_BEARING_RANGE_B70

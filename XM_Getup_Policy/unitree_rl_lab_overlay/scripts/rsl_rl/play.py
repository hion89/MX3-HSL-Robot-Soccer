#!/usr/bin/env python3
# play_with_sim_real_log.py
#
# 기존 play_io_log_full_obs_exact_labels.py 원본 코드 100% 유지.
# CSV 출력만 actor_onnx_real_runner.py의 onnx_log 포맷으로 변경.
# → sim_log_*.csv 와 onnx_log_*.csv 를 바로 diff 비교 가능.
#
# 변경점 요약 (🆕 마크):
#   1. POLICY_JOINTS 상수 추가
#   2. SimRealCSVLogger 클래스 추가
#   3. main() 내 CSV 로깅을 SimRealCSVLogger로 교체
#   4. 나머지 코드(obs label helpers, export 등) 전부 원본 그대로

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# 🎥 카메라 프리셋 (azim°, elev°) — az=0 앞(얼굴), 180 뒤(등), 90 좌, 270 우
#   키보드 훅의 숫자키 1~5 와 같은 값이다.
_CAM_PRESETS = [
    ("뒤(chase)", 180.0, 20.0),
    ("앞(정면)",    0.0, 20.0),
    ("좌측면",     90.0, 15.0),
    ("우측면",    270.0, 15.0),
    ("위(top)",   180.0, 80.0),
]

parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and export exact obs/action labels.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# 🎥 카메라 각도 스윕 — 한 번 실행으로 여러 시점의 영상을 연속 녹화한다.
#   video_length 스텝마다 카메라를 다음 프리셋(_CAM_PRESETS)으로 바꾸고
#   RecordVideo 가 같은 경계에서 새 영상을 연다. 총 5 개가 나온다.
parser.add_argument("--cam_sweep", action="store_true", default=False,
                    help="여러 카메라 각도로 연속 녹화 (--video 와 함께). 각 각도마다 video_length 스텝.")
parser.add_argument("--cam_preset", type=int, default=None, metavar="N",
                    help="시작 카메라 시점 고정 1~5 (1 뒤 / 2 앞 / 3 좌 / 4 우 / 5 위).")
# 🎲 시드 고정 — 같은 시드면 같은 스폰·같은 동작이 재현된다.
#   각도별로 따로 녹화할 때 **동일한 기상 장면**을 얻으려면 필수다.
parser.add_argument("--seed", type=int, default=None,
                    help="환경 시드 고정. 같은 값이면 같은 동작이 재현된다.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--use-train-cfg", action="store_true", default=False,
    help="Use the saved env config from the training log directory instead of current code.",
)
# 🆕 명령 시나리오 직접 지정 — cfg 수정 없이 추론할 때 cmd 고정 가능
parser.add_argument("--cmd_vx", type=float, default=None, help="Fixed lin_vel_x command [m/s]")
parser.add_argument("--cmd_vy", type=float, default=None, help="Fixed lin_vel_y command [m/s]")
parser.add_argument("--cmd_wz", type=float, default=None, help="Fixed ang_vel_z command [rad/s]")
parser.add_argument("--cmd_stop_after", type=float, default=None,
                    help="지정 초 후 속도명령을 0으로 바꿔 '전진→정지' 시연 (v1s stop-aware 확인용)")
parser.add_argument("--cmd_stop_sec", type=float, default=None,
                    help="함께 주면 '전진(cmd_stop_after초)↔정지(cmd_stop_sec초)' 를 반복 (가다 서다 반복)")
parser.add_argument("--friction", type=float, default=None, metavar="MU",
                    help="바닥 마찰 고정 [0~1.2]. 지면이 1.0 이고 combine=multiply 라 "
                         "이 값이 곧 실효 마찰이다. 미끄러운 바닥 재현용 "
                         "(장판 0.3~0.4 / 콘크리트 0.5). 미지정이면 랜덤화 그대로.")
parser.add_argument("--fall_angle", type=float, default=None, metavar="DEG",
                    help="기상 태스크 스폰 기울기 [deg]. 부호가 방향이다 — "
                         "양수 = supine(등을 댐) / 음수 = prone(엎드림). "
                         "예: 90 = 완전히 등을 댐, -90 = 완전히 엎드림. "
                         "지정하면 각도가 그 값으로 고정되고(밴드 폭 0) "
                         "역방향 커리큘럼도 끈다.")
parser.add_argument("--march", action="store_true", default=False,
                    help="제자리 걷기 강제 (속도명령 0 + rel_stop_when_standing=0.0). v0_2/v1s 처럼 제자리걷기 학습된 정책용")
parser.add_argument("--print_pose", type=int, default=0, metavar="N",
                    help="N 스텝마다 몸통 pitch/roll 실측값을 출력한다(0=끔). "
                         "예: 25 이면 0.5 초마다. 목표 피치 추종 확인용.")
parser.add_argument("--episode_len", type=float, default=None,
                    help="에피소드 길이 [s] (기본: 환경값 20초). 이 시간마다 처음 위치로 리스폰된다. "
                         "추론 전용 — 학습 설정에는 영향 없음.")
parser.add_argument("--resample_s", type=float, default=None,
                    help="속도명령 리샘플 주기 [s] (기본: 환경값 10초). 리샘플마다 명령이 새로 뽑히고 "
                         "보행 위상이 0 으로 리셋돼 걸음이 한 번 끊긴다. 크게 주면 끊김이 사라진다.")
parser.add_argument("--no_play_log", action="store_true", default=False,
                    help="Disable per-step CSV logging (sim_real/diag) for faster/real-time playback.")
parser.add_argument("--kick_strength", type=float, default=None,
                    help="Fixed kick strength command [0.1~1.0] (kick task)")
parser.add_argument("--use_train_cfg", choices=["auto", "on", "off"], default="auto",
                    help="auto=학습 시 저장된 env_cfg.py 자동 사용 (없으면 현재 코드), "
                         "on=강제 사용 (없으면 에러), off=항상 현재 코드 사용")
parser.add_argument("--ball_x", type=float, default=None,
                    help="Fix ball spawn x [m] (range degenerates to single value)")
parser.add_argument("--ball_y", type=float, default=None,
                    help="Fix ball spawn y [m] (range degenerates to single value)")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os
import sys
import time
import inspect
from datetime import datetime
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# ═══════════════════════════════════════════════════════════════
# 🆕 sim-real 비교용 관절 이름 (actor_onnx_real_runner.py와 동일)
# ═══════════════════════════════════════════════════════════════
POLICY_JOINTS = [
    "r_pelvis_pitch_joint", "r_pelvis_roll_joint", "r_pelvis_yaw_joint",
    "r_knee_pitch_joint",   "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_pelvis_pitch_joint", "l_pelvis_roll_joint", "l_pelvis_yaw_joint",
    "l_knee_pitch_joint",   "l_ankle_pitch_joint", "l_ankle_roll_joint",
]


def _load_train_env_cfg(log_dir: str, device: str, num_envs: int | None, use_fabric: bool):
    import importlib.util
    import glob

    params_dir = os.path.join(log_dir, "params")
    if not os.path.isdir(params_dir):
        raise FileNotFoundError(f"params directory not found: {params_dir}")

    py_files = glob.glob(os.path.join(params_dir, "*env_cfg*.py"))
    if not py_files:
        raise FileNotFoundError(f"No *env_cfg*.py found in {params_dir}")

    cfg_path = py_files[0]
    print(f"[INFO] Loading saved train config from: {cfg_path}")

    spec = importlib.util.spec_from_file_location("_saved_env_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod.__name__] = mod
    spec.loader.exec_module(mod)

    if hasattr(mod, "RobotPlayEnvCfg"):
        env_cfg = mod.RobotPlayEnvCfg()
        print(f"[INFO] Using RobotPlayEnvCfg (play 설정 적용됨)")
    elif hasattr(mod, "RobotEnvCfg"):
        env_cfg = mod.RobotEnvCfg()
        print(f"[INFO] RobotPlayEnvCfg not found, using RobotEnvCfg")
    else:
        raise AttributeError(f"RobotEnvCfg not found in {cfg_path}")

    env_cfg.scene.num_envs = num_envs if num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = device
    if use_fabric is not None:
        env_cfg.sim.use_fabric = use_fabric

    print(f"[INFO] Loaded config: {env_cfg.__class__.__name__} from saved file")
    return env_cfg


# =========================================================
# generic tensor/obs helpers (원본 그대로)
# =========================================================
def _is_tensordict(x) -> bool:
    return x.__class__.__module__.startswith("tensordict") and x.__class__.__name__ in ("TensorDict", "TensorDictBase")

def _iter_kv_container(x):
    if isinstance(x, dict):
        return x.items()
    if _is_tensordict(x):
        return x.items()
    raise TypeError(f"Not a kv-container: {type(x)}")

def _obs_to_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, dict) or _is_tensordict(x):
        parts = []
        for _, v in _iter_kv_container(x):
            parts.append(_obs_to_tensor(v))
        if len(parts) == 0:
            raise ValueError("Empty dict/TensorDict observation.")
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    if isinstance(x, (list, tuple)):
        parts = [_obs_to_tensor(v) for v in x]
        parts = [p for p in parts if p is not None]
        if len(parts) == 0:
            raise ValueError("Empty list/tuple observation.")
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    raise TypeError(f"Unsupported obs type: {type(x)}")

def _has_key(x, key: str) -> bool:
    try:
        return key in x
    except Exception:
        return False

def _get_item(x, key: str):
    return x[key]

def _extract_group_tensor(obs, group_name="policy"):
    if (isinstance(obs, dict) or _is_tensordict(obs)) and _has_key(obs, group_name):
        return _obs_to_tensor(_get_item(obs, group_name))
    if (isinstance(obs, dict) or _is_tensordict(obs)):
        try:
            keys = list(obs.keys())
        except Exception:
            keys = []
        if len(keys) == 1:
            return _obs_to_tensor(_get_item(obs, keys[0]))
    return _obs_to_tensor(obs)


# =========================================================
# cfg / action / obs label helpers (원본 그대로)
# =========================================================
def _get_public_fields_in_order(cfg_obj):
    out = []
    if hasattr(cfg_obj, "__dataclass_fields__"):
        for name in cfg_obj.__dataclass_fields__.keys():
            if not name.startswith("_"):
                out.append((name, getattr(cfg_obj, name)))
        return out
    for name, value in vars(cfg_obj).items():
        if not name.startswith("_"):
            out.append((name, value))
    return out

def _is_obs_term_cfg(x):
    return hasattr(x, "func") and hasattr(x, "params")

def _safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default

def _find_action_joint_names_from_cfg(unwrapped):
    actions_cfg = _safe_getattr(unwrapped.cfg, "actions", None)
    if actions_cfg is None:
        return None, None
    for attr_name, attr_value in _get_public_fields_in_order(actions_cfg):
        joint_names = _safe_getattr(attr_value, "joint_names", None)
        if joint_names:
            return list(joint_names), attr_name
    for name in dir(actions_cfg):
        if name.startswith("_"):
            continue
        value = _safe_getattr(actions_cfg, name, None)
        joint_names = _safe_getattr(value, "joint_names", None)
        if joint_names:
            return list(joint_names), name
    return None, None

def _resolve_joint_names_from_asset_cfg(asset_cfg, robot_joint_names):
    if asset_cfg is None:
        return None
    joint_names = _safe_getattr(asset_cfg, "joint_names", None)
    if joint_names:
        return list(joint_names)
    joint_ids = _safe_getattr(asset_cfg, "joint_ids", None)
    if joint_ids is not None and robot_joint_names is not None:
        try:
            return [robot_joint_names[i] for i in joint_ids]
        except Exception:
            pass
    return None

def _make_known_term_labels(term_name, term_cfg, robot_joint_names, action_joint_names):
    params = _safe_getattr(term_cfg, "params", {}) or {}
    asset_cfg = params.get("asset_cfg", None) if isinstance(params, dict) else None
    term_name_l = term_name.lower()
    if term_name_l in ("base_ang_vel",):
        return ["wx", "wy", "wz"]
    if term_name_l in ("projected_gravity",):
        return ["gx", "gy", "gz"]
    if term_name_l in ("velocity_commands", "commands", "generated_commands", "command"):
        return ["cmd_vx", "cmd_vy", "cmd_wz"]
    if term_name_l in ("joint_pos", "joint_pos_rel", "joint_positions"):
        jn = _resolve_joint_names_from_asset_cfg(asset_cfg, robot_joint_names)
        if jn is None:
            jn = list(robot_joint_names) if robot_joint_names is not None else None
        if jn is not None:
            return [f"q_{n}" for n in jn]
    if term_name_l in ("joint_vel", "joint_vel_rel", "joint_velocities"):
        jn = _resolve_joint_names_from_asset_cfg(asset_cfg, robot_joint_names)
        if jn is None:
            jn = list(robot_joint_names) if robot_joint_names is not None else None
        if jn is not None:
            return [f"qd_{n}" for n in jn]
    if term_name_l in ("last_action", "actions", "previous_action", "prev_action"):
        if action_joint_names is not None:
            return [f"a_prev_{n}" for n in action_joint_names]
    return None

def _infer_term_dim_by_call(term_name, term_cfg, env_unwrapped):
    func = _safe_getattr(term_cfg, "func", None)
    params = _safe_getattr(term_cfg, "params", {}) or {}
    if func is None:
        return None
    try:
        sig = inspect.signature(func)
        kwargs = {}
        for pname, p in sig.parameters.items():
            if pname == "env":
                continue
            if pname in params:
                kwargs[pname] = params[pname]
            elif p.default is inspect._empty:
                return None
        out = func(env_unwrapped, **kwargs)
        if torch.is_tensor(out):
            return int(out.shape[-1]) if out.ndim > 1 else int(out.shape[0])
        else:
            t = _obs_to_tensor(out)
            return int(t.shape[-1])
    except Exception as e:
        print(f"[WARN] Failed to infer dim by calling term '{term_name}': {e}")
        return None

def _build_policy_term_info(unwrapped, robot_joint_names, action_joint_names):
    policy_cfg = _safe_getattr(_safe_getattr(unwrapped.cfg, "observations", None), "policy", None)
    if policy_cfg is None:
        raise RuntimeError("Could not find unwrapped.cfg.observations.policy")
    term_infos = []
    obs_labels_1f = []
    for term_name, term_cfg in _get_public_fields_in_order(policy_cfg):
        if not _is_obs_term_cfg(term_cfg):
            continue
        labels = _make_known_term_labels(term_name, term_cfg, robot_joint_names, action_joint_names)
        if labels is None:
            dim = _infer_term_dim_by_call(term_name, term_cfg, unwrapped)
            if dim is None:
                raise RuntimeError(
                    f"Could not determine dimension/labels for observation term '{term_name}'.")
            labels = [f"{term_name}_{i:03d}" for i in range(dim)]
        else:
            dim = len(labels)
        term_infos.append({"name": term_name, "dim": dim, "labels": labels})
        obs_labels_1f.extend(labels)
    return term_infos, obs_labels_1f

def _build_history_labels_term_major(term_infos, history_length):
    labels = []
    for term in term_infos:
        for h in range(history_length):
            labels.extend([f"h{h}_{name}" for name in term["labels"]])
    return labels

def _build_history_labels_frame_major(obs_labels_1f, history_length):
    labels = []
    for h in range(history_length):
        labels.extend([f"h{h}_{name}" for name in obs_labels_1f])
    return labels

def _infer_history_length(unwrapped, obs_dim_full, no_stack_obs_dim):
    H = None
    try:
        H = int(getattr(unwrapped.cfg.observations.policy, "history_length"))
        if H <= 0:
            H = None
    except Exception:
        H = None
    if H is None and no_stack_obs_dim > 0 and obs_dim_full % no_stack_obs_dim == 0:
        H = obs_dim_full // no_stack_obs_dim
    return H

def _print_action_order(action_joint_names, action_cfg_name):
    print("\n===================================")
    print("MODEL OUTPUT (ACTION) ORDER")
    print("===================================")
    if action_joint_names is None:
        print("Could not read action joint_names from env cfg.")
    else:
        print(f"[INFO] action source cfg = {action_cfg_name}")
        for i, name in enumerate(action_joint_names):
            print(f"{i:02d} : {name}")
    print("===================================\n")

def _print_obs_term_info(term_infos):
    print("\n===================================")
    print("MODEL INPUT (OBS) ORDER - 1 frame (EXACT FROM CFG)")
    print("===================================")
    cursor = 0
    for term in term_infos:
        start = cursor
        end = cursor + term["dim"] - 1
        print(f"\n[{term['name']}]  dim={term['dim']}  idx=[{start}..{end}]")
        for i, name in enumerate(term["labels"]):
            print(f"  {start + i:03d} : {name}")
        cursor += term["dim"]
    print("\n===================================")
    print(f"TOTAL OBS DIM (NO STACK) = {cursor}")
    print("===================================\n")


# =========================================================
# 🆕 sim-real 비교용 CSV 로거 (onnx_log 호환 포맷)
# =========================================================
class SimRealCSVLogger:
    """
    kick 정책 입출력 매칭 CSV 로거.
    - 입력: base_ang_vel (raw), projected_gravity, joint_pos/vel (raw + scaled),
            last_action, ball_pos_b (frozen), target_dir_b, kick_strength
    - 출력: a_raw, a_clip, q_target (= q_default + 0.25 × a_clip)
    """

    def __init__(self, csv_path: str, action_scale: float = 0.25, control_dt: float = 0.02, joint_names=None):
        self.action_scale = action_scale
        self.control_dt = control_dt
        self.tick = 0

        self.f = open(csv_path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)

        header = ["tick", "time_sec"]
        # IMU (raw, before policy ×0.2 scale)
        header += ["imu_wx", "imu_wy", "imu_wz"]
        # Projected gravity in body frame
        header += ["grav_x", "grav_y", "grav_z"]
        # Kick-specific inputs
        header += ["ball_pos_b_x", "ball_pos_b_y", "ball_pos_b_z"]   # FROZEN at reset
        header += ["target_dir_b_x", "target_dir_b_y", "target_dir_b_z"]
        header += ["kick_strength"]
        # Per-joint data (12 joints × 9 columns)
        for prefix in ["q", "qd", "q_rel", "qd_scaled", "a_prev",
                        "a_raw", "a_clip", "q_target", "q_default"]:
            for jn in (joint_names or POLICY_JOINTS):
                header.append(f"{prefix}_{jn}")
        header += ["inference_ms"]

        self.w.writerow(header)
        self.f.flush()
        print(f"[SimRealCSV] Logging to: {csv_path} ({len(header)} columns, dt={self.control_dt:.4f}s sim-time)")

    def log(self, robot, env, joint_indices, default_q12,
            actions, prev_actions, env_index=0, inference_ms=0.0,
            joint_vel_scale=0.05):

        elapsed = self.tick * self.control_dt
        row = [self.tick, f"{elapsed:.4f}"]

        ang_vel = robot.data.root_ang_vel_b[env_index].cpu().numpy()
        proj_grav = robot.data.projected_gravity_b[env_index].cpu().numpy()
        q_all = robot.data.joint_pos[env_index].cpu().numpy()
        qd_all = robot.data.joint_vel[env_index].cpu().numpy()

        idx = joint_indices
        q12 = q_all[idx]
        qd12 = qd_all[idx]
        q_def = default_q12.cpu().numpy() if torch.is_tensor(default_q12) else default_q12

        q_rel = q12 - q_def
        qd_scaled = qd12 * joint_vel_scale

        a_raw = actions.cpu().numpy() if torch.is_tensor(actions) else actions
        a_clip = a_raw.copy()
        a_prev = prev_actions.cpu().numpy() if torch.is_tensor(prev_actions) else prev_actions
        q_target = q_def + self.action_scale * a_clip

        # Kick-specific inputs from env state
        if hasattr(env, "ball_pos_b_frozen"):
            ball_pos_b = env.ball_pos_b_frozen[env_index].cpu().numpy()
        else:
            ball_pos_b = [0.0, 0.0, 0.0]

        # target_dir_b: world dir rotated into body frame (yaw-only via quat)
        if hasattr(env, "kick_target_dir_w"):
            try:
                from isaaclab.utils.math import quat_apply_inverse as _qai
            except ImportError:
                from isaaclab.utils.math import quat_rotate_inverse as _qai
            dir_w = env.kick_target_dir_w[env_index:env_index+1]
            quat = robot.data.root_quat_w[env_index:env_index+1]
            target_dir_b = _qai(quat, dir_w)[0].cpu().numpy()
        else:
            target_dir_b = [1.0, 0.0, 0.0]

        kick_strength = float(env.kick_strength_cmd[env_index].item()) \
            if hasattr(env, "kick_strength_cmd") else 1.0

        row += [f"{ang_vel[0]:.6f}", f"{ang_vel[1]:.6f}", f"{ang_vel[2]:.6f}"]
        row += [f"{proj_grav[0]:.6f}", f"{proj_grav[1]:.6f}", f"{proj_grav[2]:.6f}"]
        row += [f"{ball_pos_b[0]:.6f}", f"{ball_pos_b[1]:.6f}", f"{ball_pos_b[2]:.6f}"]
        row += [f"{target_dir_b[0]:.6f}", f"{target_dir_b[1]:.6f}", f"{target_dir_b[2]:.6f}"]
        row += [f"{kick_strength:.4f}"]

        for arr in [q12, qd12, q_rel, qd_scaled, a_prev,
                    a_raw, a_clip, q_target, q_def]:
            for v in arr:
                row.append(f"{float(v):.6f}")
        row.append(f"{inference_ms:.3f}")

        self.w.writerow(row)
        self.tick += 1
        if self.tick % 25 == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()
            print(f"[SimRealCSV] Closed: {self.tick} rows")
        except Exception:
            pass


# =========================================================
# 🆕 진단용 CSV 로거 (foot xyz/자세/contact + base 좌표/속도)
# =========================================================
class DiagnosticCSVLogger:
    """sim-real onnx 호환과 별도로, hack 검증 / 보행 분석용 진단 컬럼만 기록.

    포함 컬럼:
        - base_x/y/z, base_roll/pitch/yaw (world)
        - base_lin_vel_x/y/z (body frame) — cmd 와 직접 비교 가능
        - base_ang_vel_x/y/z (body frame)
        - 각 발 ankle_roll_link 의 world xyz
        - 각 발 ankle_roll_link 의 world roll/pitch/yaw (quat → euler)
        - 각 발 net contact force norm 과 is_contact (force > 5N)
        - swing_max_z = max(feet_z * is_swing) — contact-aware swing height (sim 측정값)

    까치발 hack 같은 reward exploitation 을 offline 으로 정량 검증하기 위한 데이터.
    """

    @staticmethod
    def _quat_to_euler(quat):
        """[w,x,y,z] → (roll, pitch, yaw) [rad].
        IsaacLab body_quat_w 는 wxyz 순서."""
        import math
        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def __init__(self, csv_path: str, foot_body_names=("r_ankle_roll_link", "l_ankle_roll_link"),
                 contact_force_threshold: float = 5.0, control_dt: float = 0.02):
        self.tick = 0
        self.contact_thresh = contact_force_threshold
        self.control_dt = control_dt
        self.foot_body_names = list(foot_body_names)
        self._foot_body_ids = None
        self._foot_sensor_ids = None
        self._sensor = None

        self.f = open(csv_path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)

        header = ["tick", "time_sec"]
        header += ["base_x", "base_y", "base_z"]
        header += ["base_roll", "base_pitch", "base_yaw"]
        # base linear velocity 둘 다 — body frame 은 cmd 와 직접 비교용, world frame 은 절대속도/이동 분석용
        header += ["base_lin_vel_x_b", "base_lin_vel_y_b", "base_lin_vel_z_b"]
        header += ["base_lin_vel_x_w", "base_lin_vel_y_w", "base_lin_vel_z_w"]
        header += ["base_ang_vel_x_b", "base_ang_vel_y_b", "base_ang_vel_z_b"]
        for fb in self.foot_body_names:
            short = fb.replace("_link", "")
            header += [f"{short}_x", f"{short}_y", f"{short}_z"]
            header += [f"{short}_roll", f"{short}_pitch", f"{short}_yaw"]
            # 발 world linear velocity — swing speed / stance slide 직접 측정용
            header += [f"{short}_lin_vel_x_w", f"{short}_lin_vel_y_w", f"{short}_lin_vel_z_w"]
            header += [f"{short}_contact_force_n", f"{short}_is_contact"]
        header += ["swing_max_z", "any_swing"]

        self.w.writerow(header)
        self.f.flush()
        print(f"[DiagCSV] Logging to: {csv_path} ({len(header)} columns)")

    def _resolve_ids(self, robot, scene):
        body_names = list(robot.data.body_names)
        self._foot_body_ids = [body_names.index(fb) for fb in self.foot_body_names]
        # contact sensor 가 있는지 시도
        try:
            sensor = scene.sensors["contact_forces"]
            sensor_body_names = list(sensor.body_names)
            # contact_forces 는 모든 body 를 추적하므로 발 이름 그대로 찾을 수 있음
            self._foot_sensor_ids = [sensor_body_names.index(fb) for fb in self.foot_body_names]
            self._sensor = sensor
        except (KeyError, AttributeError, ValueError):
            self._sensor = None
            self._foot_sensor_ids = None
            print("[DiagCSV] WARN: contact_forces sensor not available, contact force columns will be 0.")

    def log(self, robot, scene, env_index=0):
        if self._foot_body_ids is None:
            self._resolve_ids(robot, scene)

        elapsed = self.tick * self.control_dt
        row = [self.tick, f"{elapsed:.4f}"]

        # base
        base_pos = robot.data.root_pos_w[env_index].cpu().numpy()
        base_quat = robot.data.root_quat_w[env_index].cpu().numpy()
        base_lin_b = robot.data.root_lin_vel_b[env_index].cpu().numpy()
        base_lin_w = robot.data.root_lin_vel_w[env_index].cpu().numpy()
        base_ang_b = robot.data.root_ang_vel_b[env_index].cpu().numpy()
        roll, pitch, yaw = self._quat_to_euler(base_quat)
        row += [f"{base_pos[0]:.6f}", f"{base_pos[1]:.6f}", f"{base_pos[2]:.6f}"]
        row += [f"{roll:.6f}", f"{pitch:.6f}", f"{yaw:.6f}"]
        row += [f"{base_lin_b[0]:.6f}", f"{base_lin_b[1]:.6f}", f"{base_lin_b[2]:.6f}"]
        row += [f"{base_lin_w[0]:.6f}", f"{base_lin_w[1]:.6f}", f"{base_lin_w[2]:.6f}"]
        row += [f"{base_ang_b[0]:.6f}", f"{base_ang_b[1]:.6f}", f"{base_ang_b[2]:.6f}"]

        # foot per-body
        swing_zs = []
        any_swing = False
        for i, fb_id in enumerate(self._foot_body_ids):
            fpos = robot.data.body_pos_w[env_index, fb_id].cpu().numpy()
            fquat = robot.data.body_quat_w[env_index, fb_id].cpu().numpy()
            fvel_w = robot.data.body_lin_vel_w[env_index, fb_id].cpu().numpy()
            froll, fpitch, fyaw = self._quat_to_euler(fquat)

            if self._sensor is not None and self._foot_sensor_ids is not None:
                fid = self._foot_sensor_ids[i]
                fforce_vec = self._sensor.data.net_forces_w[env_index, fid].cpu().numpy()
                fforce_n = float((fforce_vec ** 2).sum() ** 0.5)
            else:
                fforce_n = 0.0
            is_contact = 1 if fforce_n >= self.contact_thresh else 0
            if not is_contact:
                swing_zs.append(float(fpos[2]))
                any_swing = True

            row += [f"{fpos[0]:.6f}", f"{fpos[1]:.6f}", f"{fpos[2]:.6f}"]
            row += [f"{froll:.6f}", f"{fpitch:.6f}", f"{fyaw:.6f}"]
            row += [f"{fvel_w[0]:.6f}", f"{fvel_w[1]:.6f}", f"{fvel_w[2]:.6f}"]
            row += [f"{fforce_n:.4f}", f"{is_contact}"]

        swing_max_z = max(swing_zs) if swing_zs else 0.0
        row += [f"{swing_max_z:.6f}", f"{1 if any_swing else 0}"]

        self.w.writerow(row)
        self.tick += 1
        if self.tick % 25 == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()
            print(f"[DiagCSV] Closed: {self.tick} rows")
        except Exception:
            pass


# =========================================================
# main
# =========================================================
def main():
    """Play with RSL-RL agent and dump sim-real comparable CSV."""

    # ---------------------------------------------------------
    # 1) agent config + resume path
    # ---------------------------------------------------------
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # ---------------------------------------------------------
    # 2) env config
    # ---------------------------------------------------------
    # --- env_cfg 로드 전략 ---
    # auto: log_dir/params/*env_cfg*.py 가 있으면 사용, 없으면 현재 코드
    # on:   반드시 saved env_cfg.py 사용 (없으면 에러)
    # off:  항상 현재 코드 사용
    import glob as _glob
    saved_cfg_files = _glob.glob(os.path.join(log_dir, "params", "*env_cfg*.py"))
    has_saved_cfg = len(saved_cfg_files) > 0

    if args_cli.use_train_cfg == "on":
        if not has_saved_cfg:
            raise FileNotFoundError(
                f"--use_train_cfg=on 인데 saved env_cfg.py 없음: {log_dir}/params/"
            )
        use_saved = True
    elif args_cli.use_train_cfg == "off":
        use_saved = False
    else:  # auto
        use_saved = has_saved_cfg
        if has_saved_cfg:
            print(f"[INFO] auto 모드: saved env_cfg.py 발견 → 사용 ({saved_cfg_files[0]})")
        else:
            print(f"[INFO] auto 모드: saved env_cfg.py 없음 → 현재 코드 사용")

    if use_saved:
        env_cfg = _load_train_env_cfg(
            log_dir=log_dir,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
    else:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
            entry_point_key="play_env_cfg_entry_point",
        )

    # 로봇 추적 카메라(asset_root) — 보는 방향은 재생 중 키보드로 전환 (아래 키보드 훅)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.5, 2.5, 1.6)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # 🆕 CLI 로 cmd 직접 지정 — 시나리오별 추론 편의용
    if args_cli.cmd_vx is not None or args_cli.cmd_vy is not None or args_cli.cmd_wz is not None:
        try:
            cur = env_cfg.commands.base_velocity.ranges
            vx = args_cli.cmd_vx if args_cli.cmd_vx is not None else (cur.lin_vel_x[0] + cur.lin_vel_x[1]) / 2
            vy = args_cli.cmd_vy if args_cli.cmd_vy is not None else (cur.lin_vel_y[0] + cur.lin_vel_y[1]) / 2
            wz = args_cli.cmd_wz if args_cli.cmd_wz is not None else (cur.ang_vel_z[0] + cur.ang_vel_z[1]) / 2
            env_cfg.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
            env_cfg.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
            env_cfg.commands.base_velocity.ranges.ang_vel_z = (wz, wz)
            env_cfg.commands.base_velocity.rel_standing_envs = 0.0
            print(f"[INFO] CLI cmd override: vx={vx:+.3f}  vy={vy:+.3f}  wz={wz:+.3f}")
        except Exception as e:
            print(f"[WARN] Failed to override cmd from CLI: {e}")

    # 🆕 에피소드 길이 / 리샘플 주기 조절 (추론 전용).
    #   · episode_len : 이 시간마다 time_out 으로 종료되고 reset_base 가 로봇을
    #     env 원점 근처로 되돌린다 = "처음 위치로 리스폰". 환경 기본값 20 초.
    #   · resample_s  : base_velocity 재샘플 주기. gait 의
    #     reset_phase_on_resample=True 때문에 이때 보행 위상이 0 으로 리셋돼
    #     걸음이 한 번 끊긴다. 환경 기본값 10 초.
    #   ※ --march / --cmd_stop_after 가 뒤에서 이 값들을 덮어쓸 수 있다.
    # 🎲 시드 고정 (각도별 녹화에서 같은 장면을 재현하기 위해)
    if args_cli.seed is not None:
        import torch as _torch
        import numpy as _np
        import random as _random
        env_cfg.seed = args_cli.seed
        _torch.manual_seed(args_cli.seed)
        _torch.cuda.manual_seed_all(args_cli.seed)
        _np.random.seed(args_cli.seed)
        _random.seed(args_cli.seed)
        print(f"[INFO] seed = {args_cli.seed} (동작 재현용)")

    if args_cli.episode_len is not None:
        env_cfg.episode_length_s = args_cli.episode_len
        print(f"[INFO] episode_length_s = {args_cli.episode_len:.1f}s "
              f"(이 주기로 처음 위치 리스폰)")
    if args_cli.resample_s is not None:
        env_cfg.commands.base_velocity.resampling_time_range = (
            args_cli.resample_s, args_cli.resample_s)
        print(f"[INFO] 명령 리샘플 주기 = {args_cli.resample_s:.1f}s")

    # 🆕 '전진→정지' 시연: 지정 초 후 속도명령을 런타임에 0 으로 덮어쓴다.
    #    - resample 이 override 를 되돌리지 않도록 base_velocity 재샘플을 사실상 끈다.
    #    - v1s 는 정지 시 60/40 로 stop/march 를 뽑는데, 시연에선 완전정지로 고정.
    if args_cli.cmd_stop_after is not None:
        try:
            env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
            _gait = getattr(env_cfg.commands, "gait", None)
            if _gait is not None and hasattr(_gait, "rel_stop_when_standing"):
                _gait.rel_stop_when_standing = 1.0
                print("[INFO] stop 시연: 정지 시 완전정지 강제 (rel_stop_when_standing=1.0)")
            print(f"[INFO] {args_cli.cmd_stop_after:.1f}s 후 속도명령 0 으로 전환")
        except Exception as e:
            print(f"[WARN] cmd_stop_after 설정 실패: {e}")

    # 🆕 바닥 마찰 고정 (2026-09-12). 미끄러짐 재현/비교용.
    #   randomize_friction_per_env 는 일반 함수라 env 생성 후에도 적용된다
    #   (IsaacLab 기본 randomize_rigid_body_material 은 클래스라 안 된다).
    if args_cli.friction is not None:
        try:
            _t = env_cfg.events.physics_material
            _f = float(args_cli.friction)
            _t.params["static_friction_range"] = (_f, _f)
            _t.params["dynamic_friction_range"] = (_f, _f)
            print(f"[INFO] --friction: 마찰 {_f:.2f} 고정 (전 형상 균일)")
        except AttributeError:
            print("[WARN] --friction: physics_material 이벤트가 없는 태스크. 무시한다.")

    # 🆕 기상 태스크 스폰 기울기 강제 (2026-09-04).
    #   기본 태스크는 역방향 커리큘럼이 매 리셋마다 fall_angle_deg 를 덮어쓴다.
    #   커리큘럼 상태는 체크포인트에 안 들어 있어 재생 시 항상 레벨 0(5 deg,
    #   거의 직립)에서 시작한다. 원래 과제인 supine 을 보려면 90 을 준다.
    if args_cli.fall_angle is not None:
        try:
            term = env_cfg.events.reset_base_supine
            # ★ 2026-09-10 — 밴드 도입 후 각도는 abs() 로 크기만 쓰고 방향은
            #   prone_ratio 가 정한다. 상·하한을 같게 줘 **밴드 폭을 0** 으로
            #   만들어야 지정한 각도가 정확히 나온다.
            #   (이걸 안 하면 fall_angle_min_deg 가 0 으로 남아
            #    [0, |각도|] 균일 + 부호 랜덤이 되어 매번 다른 자세가 나온다)
            _a = float(args_cli.fall_angle)
            term.params["fall_angle_deg"] = abs(_a)
            term.params["fall_angle_min_deg"] = abs(_a)
            term.params["prone_ratio"] = 1.0 if _a < 0 else 0.0
            # 커리큘럼이 되돌리지 못하게 제거
            killed = []
            for _n in list(vars(env_cfg.curriculum)):
                if _n.startswith("_"):
                    continue
                if getattr(env_cfg.curriculum, _n) is not None and "getup" in _n:
                    setattr(env_cfg.curriculum, _n, None)
                    killed.append(_n)
            print(f"[INFO] --fall_angle: |{abs(_a):.0f}|도 "
                  f"{'prone(엎드림)' if _a < 0 else 'supine(등)'} 고정")
            if killed:
                print(f"[INFO]   역방향 커리큘럼 끔: {killed}")
        except AttributeError:
            print("[WARN] --fall_angle: 이 태스크에 reset_base_supine 이벤트가 없음 "
                  "(기상 태스크가 아님). 무시한다.")

    # 🆕 제자리 걷기 강제: 속도명령 0 + 정지 시 완전정지 대신 제자리걷기.
    #    gait 의 rel_stop_when_standing=0.0 이면 속도 0 env 가 전부 제자리걷기.
    if args_cli.march:
        try:
            env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
            env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            env_cfg.commands.base_velocity.rel_standing_envs = 0.0
            # 연속 녹화용: 명령 리샘플 끔(위상 10초마다 튐 방지) + 에피소드 길이 확장.
            env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
            env_cfg.episode_length_s = 60.0
            print("[INFO] --march: 리샘플 끔 + episode_length_s=60 (연속 제자리걷기)")
            _gait = getattr(env_cfg.commands, "gait", None)
            if _gait is not None and hasattr(_gait, "rel_stop_when_standing"):
                _gait.rel_stop_when_standing = 0.0
                print("[INFO] 제자리걷기 강제: 속도 0 + rel_stop_when_standing=0.0")
            else:
                print("[WARN] --march: gait 에 rel_stop_when_standing 없음 (제자리걷기 미지원 태스크?)")
        except Exception as e:
            print(f"[WARN] --march 설정 실패: {e}")

    # 🆕 CLI 로 ball spawn 위치 고정 — keyframe 재생용 결정론적 추론
    if args_cli.ball_x is not None or args_cli.ball_y is not None:
        try:
            reset_ball_params = env_cfg.events.reset_ball.params
            if args_cli.ball_x is not None:
                bx = float(args_cli.ball_x)
                reset_ball_params["x_range"] = (bx, bx)
            if args_cli.ball_y is not None:
                by = float(args_cli.ball_y)
                reset_ball_params["y_range"] = (by, by)
            print(
                f"[INFO] CLI ball spawn override: "
                f"x_range={reset_ball_params['x_range']}  y_range={reset_ball_params['y_range']}"
            )
        except Exception as e:
            print(f"[WARN] Failed to override ball spawn from CLI: {e}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # -----------------------------------------------------
    # read exact robot/action/obs metadata BEFORE wrapper
    # -----------------------------------------------------
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    robot_joint_names = list(robot.data.joint_names)

    action_joint_names, action_cfg_name = _find_action_joint_names_from_cfg(unwrapped)
    act_names = [f"a_{n}" for n in action_joint_names] if action_joint_names is not None else []

    _print_action_order(action_joint_names, action_cfg_name)

    term_infos, obs_labels_1f = _build_policy_term_info(
        unwrapped=unwrapped,
        robot_joint_names=robot_joint_names,
        action_joint_names=action_joint_names,
    )
    _print_obs_term_info(term_infos)

    no_stack_obs_dim = len(obs_labels_1f)

    # -----------------------------------------------------
    # 🆕 sim-real CSV 로거 설정
    # -----------------------------------------------------
    LOG_ENV_INDEX = 0
    LOG_MAX_STEPS = 2000

    joint_name_to_idx = {name: i for i, name in enumerate(robot_joint_names)}
    # 로봇 cfg 의 joint_sdk_names(정책 입출력 순서)를 쓴다. 없으면 MX3 목록 (rs_bot 등 다른 로봇 대응)
    _sdk_names = _safe_getattr(unwrapped.cfg.scene.robot, "joint_sdk_names", None)
    policy_joints = list(_sdk_names) if _sdk_names else POLICY_JOINTS
    joint_indices = [joint_name_to_idx[jn] for jn in policy_joints]

    default_all = robot.data.default_joint_pos[0].cpu()
    default_q12 = default_all[joint_indices]

    action_scale = 0.25
    actions_cfg = _safe_getattr(unwrapped.cfg, "actions", None)
    if actions_cfg:
        for attr_name, attr_value in _get_public_fields_in_order(actions_cfg):
            s = _safe_getattr(attr_value, "scale", None)
            if s is not None:
                action_scale = float(s)
                break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"sim_log_{timestamp}.csv")
    sim_real_logger = SimRealCSVLogger(csv_path, action_scale=action_scale,
                                       control_dt=unwrapped.step_dt, joint_names=policy_joints)

    # 🆕 진단 CSV (발 좌표/자세, contact, base 좌표/속도)
    diag_csv_path = os.path.join(log_dir, f"sim_diag_{timestamp}.csv")
    _feet = (("right_ankle_roll_link", "left_ankle_roll_link") if "right_ankle_roll_link" in robot.body_names
             else ("r_ankle_roll_link", "l_ankle_roll_link"))
    diag_logger = DiagnosticCSVLogger(diag_csv_path, foot_body_names=_feet, control_dt=unwrapped.step_dt)

    print(f"[INFO] action_scale = {action_scale}")
    print(f"[INFO] control_dt = {unwrapped.step_dt:.4f}s (sim-time per tick)")
    print(f"[INFO] default_q12 = {default_q12.numpy()}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        _vlen = args_cli.video_length
        # 🎥 --cam_preset 이면 각도별 폴더로 나눈다.
        #   같은 시드로 각도만 바꿔 여러 번 돌릴 때 서로 덮어쓰지 않게 한다.
        _vsub = "play"
        if args_cli.cam_preset is not None:
            _i = (args_cli.cam_preset - 1) % len(_CAM_PRESETS)
            _vsub = f"play_cam{args_cli.cam_preset}_{_CAM_PRESETS[_i][0]}"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", _vsub),
            # 스윕이면 video_length 마다 새 영상을 시작한다 (각도별 1 개)
            "step_trigger": (lambda step: step % _vlen == 0) if args_cli.cam_sweep
                            else (lambda step: step == 0),
            "video_length": _vlen,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -----------------------------------------------------
    # load runner / policy
    # -----------------------------------------------------
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # ★ 2026-09-14 — 팔 거울상 강제를 **내보낸 정책 안에** 넣는다 ★
    #   Getup-V3 는 MirroredArmJointPositionAction 이 액션 항에서
    #       a_L = -a_R
    #   로 왼팔 출력을 덮어쓴다. 이건 **신경망 밖**이라, 내보낸 정책을
    #   그대로 실기에 올리면 왼팔이 전혀 다르게 움직인다.
    #
    #   실측 (model_10200, 앙와위 90 도, 128 env):
    #       a_R  [-0.285, -5.805, 14.530]
    #       a_L  [-10.244, 14.523, 14.253]   <- 거울상이면 [+0.285,+5.805,-14.530]
    #       |a_L + a_R|  평균 13.08  최대 29.12  (관절각 환산 3.27 / 7.28 rad)
    #   왼팔 출력에는 보상 기울기가 전혀 안 간다(물리에 영향이 없으므로).
    #   그래서 미러 손실만으로는 묶이지 않고 값이 자유롭게 떠돈다.
    #
    #   내보낸 정책이 sim 과 다르게 동작하는 건 그 자체가 버그이므로,
    #   액션 항이 미러링을 쓰면 내보내기도 같은 변환을 포함하게 감싼다.
    #   ⚠ 내보내기 함수는 policy.actor 를 꺼내 deepcopy 한다
    #     (isaaclab_rl/rsl_rl/exporter.py). 그래서 정책 **바깥**을 감싸면
    #     래퍼가 통째로 무시된다. actor 를 감싸고, 내보내기가 기대하는
    #     속성(is_recurrent / actor)을 그대로 노출하는 얇은 껍데기를 준다.
    class _MirroredArmActor(torch.nn.Module):
        """actor 출력에 a_L = -a_R 을 적용한다."""

        def __init__(self, actor, r_start: int, n_arm: int):
            super().__init__()
            self.actor = actor
            self.r_start = r_start
            self.n_arm = n_arm

        def forward(self, x):
            a = self.actor(x)
            r0, n = self.r_start, self.n_arm
            # 팔은 액션 벡터 끝에 [오른팔 n | 왼팔 n] 으로 연속 배치돼 있다.
            return torch.cat([a[..., : r0 + n], -a[..., r0 : r0 + n]], dim=-1)

        def __getitem__(self, i):
            # ONNX 내보내기가 actor[0].in_features 로 입력 차원을 읽는다
            # (isaaclab_rl/rsl_rl/exporter.py:205). 안쪽 actor 로 위임한다.
            return self.actor[i]

    class _MirroredArmPolicy(torch.nn.Module):
        """내보내기 함수가 보는 껍데기 — .actor 와 .is_recurrent 만 제공한다."""

        def __init__(self, inner, r_start: int, n_arm: int):
            super().__init__()
            self.actor = _MirroredArmActor(inner.actor, r_start, n_arm)
            self.is_recurrent = False

    _at = env.unwrapped.action_manager.get_term("JointPositionAction")
    if bool(getattr(_at.cfg, "mirror_arms", False)):
        _names = list(_at.cfg.joint_names)
        _r = [i for i, nm in enumerate(_names)
              if nm.startswith("r_") and any(k in nm for k in ("shoulder", "elbow", "wrist", "arm"))]
        _l = [i for i, nm in enumerate(_names)
              if nm.startswith("l_") and any(k in nm for k in ("shoulder", "elbow", "wrist", "arm"))]
        _ok = (len(_r) == len(_l) and len(_r) > 0
               and _r == list(range(_r[0], _r[0] + len(_r)))
               and _l == list(range(_r[0] + len(_r), _r[0] + 2 * len(_r)))
               and _l[-1] == len(_names) - 1)
        if _ok and getattr(policy_nn, "is_recurrent", False):
            print("[WARN] 순환 정책은 actor 래핑으로 안전하게 감쌀 수 없다. "
                  "내보내기 래핑을 건너뛴다 — 실기에서 a_L = -a_R 을 직접 적용할 것.")
        elif _ok and hasattr(policy_nn, "actor"):
            policy_nn = _MirroredArmPolicy(policy_nn, _r[0], len(_r))
            print(f"[INFO] 내보내기: 팔 거울상 포함 (오른팔 {_r} -> 왼팔 {_l})")
        else:
            print("[WARN] 팔 인덱스가 연속 배치가 아니라 내보내기 래핑을 건너뛴다. "
                  f"R={_r} L={_l} — 실기 코드에서 a_L = -a_R 을 직접 적용할 것.")

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # -----------------------------------------------------
    # initial observation
    # -----------------------------------------------------
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    # obs 차원 정보 출력
    policy_obs = _extract_group_tensor(obs, "policy")
    obs_dim_full = int(policy_obs.shape[-1])
    H = _infer_history_length(unwrapped, obs_dim_full, no_stack_obs_dim)
    print(f"[INFO] FULL POLICY OBS DIM = {obs_dim_full}")
    print(f"[INFO] NO-STACK OBS DIM    = {no_stack_obs_dim}")
    print(f"[INFO] HISTORY_LENGTH     = {H}")
    print(f"[INFO] ACTION DIM         = {len(act_names)}")

    prev_actions = torch.zeros(12, device=env.unwrapped.device)

    # -----------------------------------------------------
    # Kick-task debug viz: arrow above ball pointing to target
    # -----------------------------------------------------
    kick_arrow_markers = None
    if "ball" in unwrapped.scene.keys() and hasattr(unwrapped, "kick_target_dir_w"):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG

        arrow_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        arrow_cfg.prim_path = "/Visuals/KickTargetArrow"
        arrow_cfg.markers["arrow"].scale = (0.4, 0.04, 0.44)
        kick_arrow_markers = VisualizationMarkers(arrow_cfg)
        print("[INFO] Kick target-direction arrow enabled (green arrow above ball)")

    def _update_kick_arrow():
        if kick_arrow_markers is None:
            return
        ball = unwrapped.scene["ball"]
        n = unwrapped.num_envs
        pos = ball.data.root_pos_w.clone()
        pos[:, 2] += 0.20  # 20cm above ball
        # Absolute-direction mode: arrow points along the FIXED world target
        # direction (kick_target_dir_w), so it stays put as the robot moves.
        dir_w = unwrapped.kick_target_dir_w
        yaw = torch.atan2(dir_w[:, 1], dir_w[:, 0])
        half = 0.5 * yaw
        quat = torch.zeros(n, 4, device=env.unwrapped.device)
        quat[:, 0] = torch.cos(half)
        quat[:, 3] = torch.sin(half)
        kick_arrow_markers.visualize(translations=pos, orientations=quat)

    # -----------------------------------------------------
    # Kick strength override (kick task only)
    # -----------------------------------------------------
    if args_cli.kick_strength is not None:
        ks = float(args_cli.kick_strength)
        ks = max(0.0, min(1.0, ks))
        unwrapped.kick_strength_fixed = ks
        if hasattr(unwrapped, "kick_strength_cmd"):
            unwrapped.kick_strength_cmd.fill_(ks)
        print(f"[INFO] kick_strength fixed to {ks:.2f} (target speed = {ks * 3.5:.2f} m/s)")

    # -----------------------------------------------------
    # Ball velocity tracker (kick task only)
    # -----------------------------------------------------
    ball_obj = None
    ball_peak_speed = None
    ball_peak_fwd = None  # projection onto target direction
    ball_was_moving = None
    ball_kicks_total = []   # total speed peaks across all envs
    ball_kicks_fwd = []     # forward (toward-target) speed peaks
    if "ball" in unwrapped.scene.keys():
        ball_obj = unwrapped.scene["ball"]
        N = unwrapped.num_envs
        dev = env.unwrapped.device
        ball_peak_speed = torch.zeros(N, device=dev)
        ball_peak_fwd = torch.zeros(N, device=dev)
        ball_was_moving = torch.zeros(N, dtype=torch.bool, device=dev)
        print("[INFO] Ball velocity tracker enabled (threshold 0.3 m/s to detect movement)")

    # -----------------------------------------------------
    # 🎥 키보드 카메라 시점 전환 (MuJoCo 스타일) — 로봇 추적 유지 + 방향만 전환
    #   default_cam_eye(오프셋)를 런타임에 갈아끼우면 추적 콜백이 매 프레임 적용.
    #   숫자키 = 프리셋 시점 / [ ] = 좌우 회전 / ; ' = 상하 / - = 줌인·줌아웃
    # -----------------------------------------------------
    _cam_set = None          # 🎥 프리셋 setter — 뷰포트가 있을 때만 채워진다
    _cam_ctrl = getattr(unwrapped, "viewport_camera_controller", None)
    if _cam_ctrl is not None:
        import math as _math
        import numpy as _np
        import carb.input
        import omni.appwindow

        # 현재 기본 eye=(1.5,2.5,1.6) 을 (azim, elev, radius)로 환산해 초기값
        _e0 = (1.5, 2.5, 1.6)
        _cam_state = {
            "azim": _math.degrees(_math.atan2(_e0[1], _e0[0])),        # ≈59°
            "elev": _math.degrees(_math.atan2(_e0[2], _math.hypot(_e0[0], _e0[1]))),  # ≈29°
            "radius": _math.sqrt(_e0[0] ** 2 + _e0[1] ** 2 + _e0[2] ** 2),  # ≈3.33
        }

        def _cam_apply():
            az = _math.radians(_cam_state["azim"])
            el = _math.radians(_cam_state["elev"])
            r = _cam_state["radius"]
            ex = r * _math.cos(el) * _math.cos(az)
            ey = r * _math.cos(el) * _math.sin(az)
            ez = r * _math.sin(el)
            _cam_ctrl.default_cam_eye = _np.array([ex, ey, ez], dtype=float)
            _cam_ctrl.default_cam_lookat = _np.array([0.0, 0.0, 0.0], dtype=float)

        _KI = carb.input.KeyboardInput
        # 숫자키 프리셋 — 모듈 상수 _CAM_PRESETS 와 공유한다 (스윕과 같은 값)
        _cam_presets = {
            k: _CAM_PRESETS[i]
            for i, k in enumerate((_KI.KEY_1, _KI.KEY_2, _KI.KEY_3, _KI.KEY_4, _KI.KEY_5))
        }

        # 🎥 스윕/프리셋 지정에서 쓸 setter 를 밖으로 노출
        def _cam_set_preset(idx: int):
            """0-based 프리셋 인덱스로 카메라 각도를 바꾼다."""
            name, az, el = _CAM_PRESETS[idx % len(_CAM_PRESETS)]
            _cam_state["azim"], _cam_state["elev"] = az, el
            _cam_apply()
            return name

        _cam_set = _cam_set_preset
        # --cam_preset 로 시작 시점을 고정
        if args_cli.cam_preset is not None:
            _n = _cam_set_preset(args_cli.cam_preset - 1)
            print(f"[CAM] 시작 시점 고정: {_n}")

        def _on_kbd(e, *_):
            if e.type == carb.input.KeyboardEventType.KEY_PRESS:
                k = e.input
                if k in _cam_presets:
                    name, az, el = _cam_presets[k]
                    _cam_state["azim"], _cam_state["elev"] = az, el
                    _cam_apply(); print(f"[CAM] {name}  (azim={az:.0f} elev={el:.0f})")
                elif k == _KI.LEFT_BRACKET:
                    _cam_state["azim"] = (_cam_state["azim"] - 15) % 360; _cam_apply()
                elif k == _KI.RIGHT_BRACKET:
                    _cam_state["azim"] = (_cam_state["azim"] + 15) % 360; _cam_apply()
                elif k == _KI.SEMICOLON:
                    _cam_state["elev"] = min(89.0, _cam_state["elev"] + 10); _cam_apply()
                elif k == _KI.APOSTROPHE:
                    _cam_state["elev"] = max(-20.0, _cam_state["elev"] - 10); _cam_apply()
                elif k == _KI.MINUS:
                    _cam_state["radius"] *= 1.15; _cam_apply()
                elif k == _KI.EQUAL:
                    _cam_state["radius"] = max(0.5, _cam_state["radius"] / 1.15); _cam_apply()
            return True

        try:
            _appwin = omni.appwindow.get_default_app_window()
            _kbd = _appwin.get_keyboard()
            _input_iface = carb.input.acquire_input_interface()
            _kbd_sub = _input_iface.subscribe_to_keyboard_events(_kbd, _on_kbd)
            print("[CAM] 키보드 시점 전환 활성:  1=뒤 2=앞 3=좌 4=우 5=위 | [ ]=좌우회전 ; '=상하 -/= =줌")
        except Exception as _e:
            print(f"[CAM] 키보드 훅 실패(무시): {_e}")

    # -----------------------------------------------------
    # simulation loop
    # -----------------------------------------------------
    timestep = 0
    print(f"\n[INFO] Starting play loop (logging env {LOG_ENV_INDEX}, max {LOG_MAX_STEPS} steps)")

    # ── 🔧 TEMP DEBUG: 발 link 위치 + 계산된 toe/heel 검증 (50 step마다, ~1초) ──
    import torch as _t
    _foot_ids = []
    for _name in ["r_ankle_roll_link", "l_ankle_roll_link"]:
        try:
            _foot_ids.append(robot.find_bodies(_name)[0][0])
        except Exception:
            pass
    _TOE_OFFSET  = 0.132    # 사용자 측정: 132mm
    _HEEL_OFFSET = -0.028   # 사용자 측정: -28mm

    def _quat_apply(q, v):
        # q: (4,) [w,x,y,z], v: (3,)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        vx, vy, vz = v[0], v[1], v[2]
        # rotated = q * v * q^-1
        # Using the formula: v' = v + 2*q_vec × (q_vec × v + q_w * v)
        t_x = 2 * (qy * vz - qz * vy)
        t_y = 2 * (qz * vx - qx * vz)
        t_z = 2 * (qx * vy - qy * vx)
        rx = vx + qw * t_x + (qy * t_z - qz * t_y)
        ry = vy + qw * t_y + (qz * t_x - qx * t_z)
        rz = vz + qw * t_z + (qx * t_y - qy * t_x)
        return _t.tensor([rx, ry, rz], device=q.device)

    # ── 킥-허용 시각화 마커 (2026-08-15) ─────────────────────────────
    #   로봇 위 구체 색: 🔴 아직 차면 종료(earliest 전) / 🟢 킥 허용 / 🔵 이미 킥.
    #   allowed = episode_time >= min_kick_time_s (조기킥 종료 임계와 동일, 절대 1초 문턱).
    _kick_marker = None
    _KICK_MIN_TIME_S = 1.0
    try:
        import isaaclab.sim as _sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        try:
            _tcfg = unwrapped.termination_manager.get_term_cfg("early_kick_done")
            _KICK_MIN_TIME_S = float(_tcfg.params.get("min_kick_time_s", 1.0))
        except Exception:
            pass

        _mk_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/kick_allowed",
            markers={
                "wait": _sim_utils.SphereCfg(
                    radius=0.07,
                    visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
                ),
                "allowed": _sim_utils.SphereCfg(
                    radius=0.07,
                    visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
                ),
                "kicked": _sim_utils.SphereCfg(
                    radius=0.07,
                    visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
            },
        )
        _kick_marker = VisualizationMarkers(_mk_cfg)
        print(f"[INFO] 킥-허용 마커 활성 (min_kick_time_s={_KICK_MIN_TIME_S}): "
              f"🔴차면종료 / 🟢킥허용 / 🔵이미킥")
    except Exception as _e:
        print(f"[WARN] 킥-허용 마커 비활성: {_e}")

    # 재생 배속(real-time factor) 측정용 — 최근 구간 sim-time / wall-time
    _rtf_last_wall = time.time()
    _rtf_last_step = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            t_infer = time.perf_counter()
            actions = policy(obs)
            inference_ms = (time.perf_counter() - t_infer) * 1000.0

            # ── 🆕 kick 정책 입출력 CSV 로깅 (--no_play_log 로 끄면 배속↑) ──
            if timestep < LOG_MAX_STEPS and not args_cli.no_play_log:
                act_env = actions[LOG_ENV_INDEX].detach()

                sim_real_logger.log(
                    robot=robot,
                    env=unwrapped,
                    joint_indices=joint_indices,
                    default_q12=default_q12,
                    actions=act_env,
                    prev_actions=prev_actions,
                    env_index=LOG_ENV_INDEX,
                    inference_ms=inference_ms,
                )
                diag_logger.log(robot=robot, scene=unwrapped.scene, env_index=LOG_ENV_INDEX)
                prev_actions = act_env.clone()

            obs, _, _, _ = env.step(actions)

            # ── 🆕 몸통 자세 실측 출력 (--print_pose N) ──────────────────
            #   목표 피치 명령이 실제로 자세를 바꾸는지 눈이 아니라 숫자로
            #   확인하기 위한 것. pitch 계산은 base_pitch_target_l2 와 동일한
            #   방식(projected_gravity 기반)이라 리워드가 보는 값과 같다.
            if args_cli.print_pose and timestep % args_cli.print_pose == 0:
                _g = robot.data.projected_gravity_b[LOG_ENV_INDEX]
                _pitch = torch.atan2(-_g[0], torch.sqrt(_g[1] ** 2 + _g[2] ** 2))
                _roll = torch.atan2(_g[1], -_g[2])
                _pd = float(torch.rad2deg(_pitch))
                _rd = float(torch.rad2deg(_roll))
                _tgt = ""
                _bp = unwrapped.command_manager._terms.get("base_pitch")
                if _bp is not None:
                    _t = float(torch.rad2deg(_bp.target_pitch[LOG_ENV_INDEX]))
                    _tgt = f"  목표 {_t:+6.2f}°  오차 {_pd - _t:+6.2f}°"
                # 부호 검증용: 몸통 x축(전방)의 월드 z 성분.
                #   nose_z < 0 이면 코가 아래 = 앞으로 숙임, > 0 이면 뒤로 젖힘.
                #   pitch 부호와 대조하면 규약이 확정된다.
                _q = robot.data.root_quat_w[LOG_ENV_INDEX]
                _w, _x, _y, _z = _q[0], _q[1], _q[2], _q[3]
                _nose_z = float(2.0 * (_x * _z - _w * _y))
                _dir = "앞으로숙임" if _nose_z < 0 else "뒤로젖힘"
                print(f"[pose] t={timestep:5d}  pitch {_pd:+6.2f}°  roll {_rd:+6.2f}°{_tgt}"
                      f"  | nose_z {_nose_z:+.4f} ({_dir})")

            # ── 킥-허용 마커 갱신 (로봇 위 구체 색) ──
            #   env마다 3색(빨강0/초록1/파랑2) 인스턴스를 항상 유지 — 활성 색만 머리 위,
            #   나머지 2색은 땅 밑(z=-100)으로 숨김. (프로토마다 인스턴스≥1 → USD 경고 방지)
            if _kick_marker is not None:
                try:
                    _n = unwrapped.num_envs
                    _dev = unwrapped.device
                    _cur_t = unwrapped.episode_length_buf.float() * unwrapped.step_dt
                    _allowed = _cur_t >= _KICK_MIN_TIME_S   # 시작 1초 이후 킥 허용
                    if hasattr(unwrapped, "kick_time_s"):
                        _kicked = unwrapped.kick_time_s >= 0
                    else:
                        _kicked = _t.zeros(_n, dtype=_t.bool, device=_dev)
                    # 상태 per env: 0=wait(빨강) / 1=allowed(초록) / 2=kicked(파랑)
                    _state = _t.where(
                        _kicked, _t.full((_n,), 2, dtype=_t.long, device=_dev),
                        _t.where(_allowed, _t.ones(_n, dtype=_t.long, device=_dev),
                                 _t.zeros(_n, dtype=_t.long, device=_dev)),
                    )
                    _head = robot.data.root_pos_w.clone()
                    _head[:, 2] += 0.65   # 로봇 머리 위
                    # (N,3)색 각각 인스턴스 → (3N,3). marker_indices=[0,1,2,0,1,2,...] 고정.
                    _tr = _head.unsqueeze(1).repeat(1, 3, 1).reshape(-1, 3)
                    _proto = _t.arange(3, device=_dev).unsqueeze(0).expand(_n, 3).reshape(-1)
                    _state_rep = _state.unsqueeze(1).expand(_n, 3).reshape(-1)
                    _bury = _proto != _state_rep          # 활성 색 외에는 땅 밑으로
                    _tr[_bury, 2] = -100.0
                    _kick_marker.visualize(translations=_tr, marker_indices=_proto)
                except Exception:
                    pass

            # 🆕 '전진→정지(→다시 전진)' 시연: 스케줄에 따라 매 스텝 속도명령을 덮어쓴다.
            #    base_velocity 재샘플을 껐으므로 덮어쓰기가 유지되고,
            #    다음 스텝 gait 정지판정이 이를 읽어 정지/보행으로 들어간다.
            #      - cmd_stop_sec 없음: cmd_stop_after 초 후 계속 정지
            #      - cmd_stop_sec 있음: 전진(cmd_stop_after) ↔ 정지(cmd_stop_sec) 반복
            if args_cli.cmd_stop_after is not None:
                _t = timestep * dt
                if args_cli.cmd_stop_sec is not None:
                    _period = args_cli.cmd_stop_after + args_cli.cmd_stop_sec
                    _walking = (_t % _period) < args_cli.cmd_stop_after
                else:
                    _walking = _t < args_cli.cmd_stop_after
                _cmd = unwrapped.command_manager.get_command("base_velocity")
                if _walking:
                    _cmd[:, 0] = args_cli.cmd_vx if args_cli.cmd_vx is not None else 0.0
                    _cmd[:, 1] = args_cli.cmd_vy if args_cli.cmd_vy is not None else 0.0
                    _cmd[:, 2] = args_cli.cmd_wz if args_cli.cmd_wz is not None else 0.0
                else:
                    _cmd[:, :] = 0.0

            _update_kick_arrow()

            # ── Ball velocity tracking ──
            if ball_obj is not None:
                v_xy = ball_obj.data.root_lin_vel_w[:, :2]
                speed = v_xy.norm(dim=-1)
                # forward speed = projection onto target direction
                dir_xy = unwrapped.kick_target_dir_w[:, :2]
                fwd = (v_xy * dir_xy).sum(dim=-1).clamp(min=0.0)
                moving = speed > 0.3
                ball_peak_speed = torch.maximum(ball_peak_speed, speed)
                ball_peak_fwd = torch.maximum(ball_peak_fwd, fwd)
                stopped = ball_was_moving & (~moving)
                if stopped.any():
                    peaks_s = ball_peak_speed[stopped].cpu().tolist()
                    peaks_f = ball_peak_fwd[stopped].cpu().tolist()
                    for ps, pf in zip(peaks_s, peaks_f):
                        ball_kicks_total.append(ps)
                        ball_kicks_fwd.append(pf)
                        # print latest kick
                        print(f"  [KICK #{len(ball_kicks_total):3d}] peak total speed = {ps:.2f} m/s, forward = {pf:.2f} m/s")
                    ball_peak_speed = torch.where(stopped, torch.zeros_like(ball_peak_speed), ball_peak_speed)
                    ball_peak_fwd = torch.where(stopped, torch.zeros_like(ball_peak_fwd), ball_peak_fwd)
                ball_was_moving = moving

            base_h = env.unwrapped.scene["robot"].data.root_pos_w[0, 2].item()
            if timestep % 50 == 0:
                # 실제 이동속도 (body frame: vx=전진, vy=횡)
                v_b = robot.data.root_lin_vel_b[0]
                vx_a, vy_a = v_b[0].item(), v_b[1].item()
                spd = (vx_a * vx_a + vy_a * vy_a) ** 0.5
                # 속도 명령이 없는 태스크(기상 등)도 있다. commands=None 이면
                # get_command 가 KeyError 를 낸다. 보행/킥은 기존과 동일하게 동작한다.
                try:
                    cmd_vx = unwrapped.command_manager.get_command("base_velocity")[0, 0].item()
                except Exception:
                    cmd_vx = None
                # 재생 배속: 최근 구간 (스텝수 × dt) / 실제 경과 wall-time
                _now = time.time()
                _wall = _now - _rtf_last_wall
                _rtf = ((timestep - _rtf_last_step) * dt / _wall) if (_wall > 0 and timestep > 0) else 0.0
                _rtf_last_wall = _now
                _rtf_last_step = timestep
                _cmd_s = f" (cmd vx={cmd_vx:+.2f})" if cmd_vx is not None else ""
                print(
                    f"[step {timestep:5d}] base_height: {base_h:.4f} m | "
                    f"이동속도 vx={vx_a:+.2f} vy={vy_a:+.2f} |v|={spd:.2f} m/s{_cmd_s} | "
                    f"재생 {_rtf:.2f}x"
                )

        if args_cli.video:
            timestep += 1
            # 🎥 각도 스윕 — video_length 스텝마다 다음 프리셋으로 바꾸고
            #    RecordVideo 의 step_trigger 가 같은 경계에서 새 영상을 연다.
            if args_cli.cam_sweep:
                _n_seg = len(_CAM_PRESETS)
                if timestep >= args_cli.video_length * _n_seg:
                    break
                if timestep % args_cli.video_length == 0:
                    _idx = timestep // args_cli.video_length
                    if _cam_set is not None:
                        _nm = _cam_set(_idx)
                        print(f"[CAM] 구간 {_idx + 1}/{_n_seg} → {_nm}")
                    else:
                        print(f"[CAM] 구간 {_idx + 1}/{_n_seg} (뷰포트 없음 — 각도 고정)")
            elif timestep == args_cli.video_length:
                break
        else:
            timestep += 1
            if timestep >= LOG_MAX_STEPS:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    sim_real_logger.close()
    diag_logger.close()

    # ── Ball velocity summary ──
    if ball_kicks_total:
        n = len(ball_kicks_total)
        mean_s = sum(ball_kicks_total) / n
        max_s = max(ball_kicks_total)
        min_s = min(ball_kicks_total)
        sorted_s = sorted(ball_kicks_total)
        median_s = sorted_s[n // 2]
        mean_f = sum(ball_kicks_fwd) / n
        max_f = max(ball_kicks_fwd)
        median_f = sorted(ball_kicks_fwd)[n // 2]
        print(f"\n========== Ball Velocity Summary ==========")
        print(f"Total kicks recorded: {n}")
        print(f"--- TOTAL speed (xy norm) ---")
        print(f"  mean:   {mean_s:.2f} m/s")
        print(f"  median: {median_s:.2f} m/s")
        print(f"  max:    {max_s:.2f} m/s")
        print(f"  min:    {min_s:.2f} m/s")
        print(f"--- FORWARD speed (toward target) ---")
        print(f"  mean:   {mean_f:.2f} m/s")
        print(f"  median: {median_f:.2f} m/s")
        print(f"  max:    {max_f:.2f} m/s")
        print(f"===========================================\n")
    else:
        print("\n[Ball Velocity] No kicks recorded during play.\n")

    env.close()
    print(f"[DONE] {timestep} steps")


if __name__ == "__main__":
    main()
    simulation_app.close()
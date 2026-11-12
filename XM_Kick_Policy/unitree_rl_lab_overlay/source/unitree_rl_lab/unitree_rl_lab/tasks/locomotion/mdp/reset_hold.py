import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


def set_reset_hold_steps(env, env_ids, hold_time_s: float):
    """Reset된 env들에 대해 reset hold step 카운터를 설정한다."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    hold_steps = max(0, int(round(hold_time_s / env.step_dt)))

    if not hasattr(env, "reset_hold_steps"):
        env.reset_hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env.reset_hold_steps[env_ids] = hold_steps


def generated_commands_with_reset_hold(env, command_name: str):
    """reset hold 동안에는 command를 0으로 반환."""
    cmd = env.command_manager.get_command(command_name)

    if not hasattr(env, "reset_hold_steps"):
        return cmd

    hold_mask = env.reset_hold_steps > 0
    if torch.any(hold_mask):
        cmd = cmd.clone()
        cmd[hold_mask] = 0.0

    return cmd


def mask_rewards_during_reset_hold(env, reward: torch.Tensor):
    """원하면 reward term 안에서 이 함수를 써서 hold 동안 reward를 0으로 만들 수 있음."""
    if not hasattr(env, "reset_hold_steps"):
        return reward

    hold_mask = env.reset_hold_steps > 0
    if torch.any(hold_mask):
        reward = reward.clone()
        reward[hold_mask] = 0.0
    return reward


class DelayedJointPositionAction(JointPositionAction):
    """Reset hold + 관절 한계 클램핑 + Action delay가 적용된 JointPositionAction.

    1. reset hold 기간 동안 action을 0으로 마스킹
    2. target position을 관절 한계 내로 클램핑
    3. action delay: 정책 출력을 N step 뒤에 적용 (sim-to-real gap 축소)
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # --- action delay buffer 초기화 ---
        if cfg.action_delay_range is not None:
            d_min, d_max = cfg.action_delay_range
            self._delay_max = d_max
            self._delay_min = d_min

            # ring buffer: [num_envs, buffer_size, action_dim]
            buf_size = d_max + 1
            act_dim = self._processed_actions.shape[-1]
            self._delay_buf = torch.zeros(
                env.num_envs, buf_size, act_dim, device=env.device
            )
            self._delay_write_idx = torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            )
            # 각 env별 delay step 수 (reset 시 랜덤 재설정)
            self._delay_steps = torch.randint(
                d_min, d_max + 1, (env.num_envs,), device=env.device
            )
            self._use_delay = True
        else:
            self._use_delay = False

    def process_actions(self, actions: torch.Tensor):
        # --- reset hold masking ---
        if not hasattr(self._env, "reset_hold_steps"):
            self._env.reset_hold_steps = torch.zeros(
                self._env.num_envs, dtype=torch.long, device=self._env.device
            )

        hold_mask = self._env.reset_hold_steps > 0

        if torch.any(hold_mask):
            actions = actions.clone()
            actions[hold_mask] = 0.0

        # --- action delay 적용 ---
        if self._use_delay:
            actions = self._apply_delay(actions)

        # --- 기본 처리: target = default + action * scale ---
        super().process_actions(actions)

        # --- target을 관절 한계 내로 클램핑 ---
        if not hasattr(self, "_clamp_lo"):
            self._build_clamp_limits()

        self._processed_actions[:] = torch.clamp(
            self._processed_actions,
            min=self._clamp_lo,
            max=self._clamp_hi,
        )

        # --- hold 카운트다운 ---
        if torch.any(hold_mask):
            self._env.reset_hold_steps[hold_mask] -= 1
            self._env.reset_hold_steps.clamp_(min=0)

    def _apply_delay(self, actions: torch.Tensor) -> torch.Tensor:
        """Ring buffer를 이용한 action delay.

        현재 action을 buffer에 쓰고, N step 전의 action을 꺼내서 반환한다.
        텐서 연산으로 구현 (for loop 없음).
        """
        buf_size = self._delay_max + 1
        n = self._env.num_envs

        # 현재 action을 buffer에 기록 (advanced indexing)
        env_idx = torch.arange(n, device=actions.device)
        self._delay_buf[env_idx, self._delay_write_idx] = actions

        # N step 전의 action 읽기
        read_idx = (self._delay_write_idx - self._delay_steps) % buf_size
        delayed_actions = self._delay_buf[env_idx, read_idx]

        # write index 전진
        self._delay_write_idx = (self._delay_write_idx + 1) % buf_size

        return delayed_actions

    def reset(self, env_ids):
        """Reset 시 해당 env의 delay buffer를 초기화."""
        super().reset(env_ids)

        if self._use_delay and env_ids is not None and len(env_ids) > 0:
            self._delay_buf[env_ids] = 0.0
            self._delay_write_idx[env_ids] = 0

            # delay step 수 랜덤 재설정 (domain randomization)
            self._delay_steps[env_ids] = torch.randint(
                self._delay_min, self._delay_max + 1,
                (len(env_ids),), device=self._env.device,
            )

    def _build_clamp_limits(self):
        """클램핑 한계를 한 번 만들고 캐싱."""
        cfg = self.cfg

        if cfg.target_limits is not None:
            lo_list = []
            hi_list = []
            for jname in cfg.joint_names:
                if jname in cfg.target_limits:
                    lo_list.append(cfg.target_limits[jname][0])
                    hi_list.append(cfg.target_limits[jname][1])
                else:
                    lo_list.append(-1e6)
                    hi_list.append(1e6)

            self._clamp_lo = torch.tensor(lo_list, dtype=torch.float32, device=self._env.device)
            self._clamp_hi = torch.tensor(hi_list, dtype=torch.float32, device=self._env.device)
        else:
            limits = self._asset.data.soft_joint_pos_limits[0, self._joint_ids, :]
            self._clamp_lo = limits[:, 0]
            self._clamp_hi = limits[:, 1]


def reset_joints_additive_noise(
    env,
    env_ids,
    noise_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """reset_joints_by_scale 이후에 호출하여 additive uniform noise를 추가."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[env_ids].clone()
    noise = torch.empty_like(joint_pos).uniform_(noise_range[0], noise_range[1])
    new_pos = joint_pos + noise

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    new_pos = torch.clamp(new_pos, joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    asset.write_joint_state_to_sim(
        new_pos,
        asset.data.joint_vel[env_ids],
        env_ids=env_ids,
    )


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type = DelayedJointPositionAction

    # 관절별 target position 한계 (선택)
    target_limits: dict[str, tuple[float, float]] | None = None

    # Action delay 범위 (선택)
    # (min_steps, max_steps) — 각 env마다 범위 내 랜덤 delay, reset 시 재설정
    # None이면 delay 없음
    #
    # 50Hz 제어 기준:
    #   (2, 4) → 40~80ms (실측 3~4 step 기준)
    #   (1, 2) → 20~40ms (보수적)
    action_delay_range: tuple[int, int] | None = None
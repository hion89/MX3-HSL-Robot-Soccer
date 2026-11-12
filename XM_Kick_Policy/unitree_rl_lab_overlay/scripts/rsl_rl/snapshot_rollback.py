"""
Snapshot Rollback Wrapper for RSL-RL OnPolicyRunner (rsl-rl v3.1.2)
===================================================================
value_loss 발산 감지 시 자동으로 이전 스냅샷으로 복구합니다.

사용법 (train.py에서):
    from snapshot_rollback import learn_with_rollback
    learn_with_rollback(runner, max_iterations=agent_cfg.max_iterations)
"""

import gc
import io
import os
import time
from collections import deque

import torch


def _state_dict_to_cpu(state_dict):
    """state_dict의 모든 텐서를 CPU로 복사. GPU 메모리 참조를 끊음."""
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}


def _optimizer_state_to_cpu(opt_state):
    """optimizer state_dict 내부의 모든 텐서를 CPU로 복사."""
    cpu_state = {}
    for k, v in opt_state.items():
        if k == 'state':
            cpu_state['state'] = {}
            for param_id, param_state in v.items():
                cpu_state['state'][param_id] = {}
                for sk, sv in param_state.items():
                    if isinstance(sv, torch.Tensor):
                        cpu_state['state'][param_id][sk] = sv.cpu()
                    else:
                        cpu_state['state'][param_id][sk] = sv
        else:
            cpu_state[k] = v
    return cpu_state


def learn_with_rollback(
    runner,
    max_iterations: int,
    snapshot_interval: int = 10,
    max_snapshots: int = 20,
    value_loss_threshold: float = 200.0,
    init_at_random_ep_len: bool = True,
    randomize_after_rollback: bool = True,
):
    # ── 1. runner.learn() 초기화 부분 그대로 재현 ──
    runner._prepare_logging_writer()

    if init_at_random_ep_len:
        runner.env.episode_length_buf = torch.randint_like(
            runner.env.episode_length_buf,
            high=int(runner.env.max_episode_length),
        )

    obs = runner.env.get_observations().to(runner.device)
    runner.train_mode()

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
    cur_episode_length = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)

    # RND 지원
    if runner.alg.rnd:
        erewbuffer = deque(maxlen=100)
        irewbuffer = deque(maxlen=100)
        cur_ereward_sum = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
        cur_ireward_sum = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)

    # Distributed 동기화
    if runner.is_distributed:
        print(f"Synchronizing parameters for rank {runner.gpu_global_rank}...")
        runner.alg.broadcast_parameters()

    # ── 2. 스냅샷 관리 (CPU 복사 + BytesIO) ──
    snapshots = []
    rollback_count = 0

    def save_snapshot(iteration):
        # GPU → CPU 복사 후 직렬화 (GPU 메모리 참조 완전히 끊음)
        cpu_model = _state_dict_to_cpu(runner.alg.policy.state_dict())
        cpu_optimizer = _optimizer_state_to_cpu(runner.alg.optimizer.state_dict())

        buf = io.BytesIO()
        torch.save({"model": cpu_model, "optimizer": cpu_optimizer}, buf)

        # 오래된 스냅샷 명시적 해제
        while len(snapshots) >= max_snapshots:
            old = snapshots.pop(0)
            old["buf"].close()
            del old

        snapshots.append({"buf": buf, "iteration": iteration})
        print(f"[SNAPSHOT] Saved at iteration {iteration} ({len(snapshots)}/{max_snapshots}) | Total rollbacks: {rollback_count}")

    def load_snapshot():
        if not snapshots:
            print("[SNAPSHOT] No snapshot available!")
            return False
        snapshot = snapshots[-1]
        snapshot["buf"].seek(0)
        state = torch.load(snapshot["buf"], map_location=runner.device, weights_only=False)
        runner.alg.policy.load_state_dict(state["model"])
        runner.alg.optimizer.load_state_dict(state["optimizer"])
        rollback_iter = snapshot["iteration"]

        # 사용한 스냅샷 해제
        snapshot["buf"].close()
        snapshots.pop()
        del state

        print(f"[SNAPSHOT] *** ROLLBACK to iteration {rollback_iter} ***")
        print(f"[SNAPSHOT] Remaining snapshots: {len(snapshots)}")
        return True

    save_snapshot(0)

    # ── 3. 학습 루프 (runner.learn()과 동일 구조) ──
    start_iter = runner.current_learning_iteration
    tot_iter = start_iter + max_iterations
    num_learning_iterations = max_iterations

    for it in range(start_iter, tot_iter):
        start = time.time()

        # ── 3a. Rollout ──
        with torch.inference_mode():
            for _ in range(runner.num_steps_per_env):
                actions = runner.alg.act(obs)
                obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
                obs, rewards, dones = (
                    obs.to(runner.device),
                    rewards.to(runner.device),
                    dones.to(runner.device),
                )
                runner.alg.process_env_step(obs, rewards, dones, extras)

                intrinsic_rewards = runner.alg.intrinsic_rewards if runner.alg.rnd else None

                if runner.log_dir is not None:
                    if "episode" in extras:
                        ep_infos.append(extras["episode"])
                    elif "log" in extras:
                        ep_infos.append(extras["log"])

                    if runner.alg.rnd:
                        cur_ereward_sum += rewards
                        cur_ireward_sum += intrinsic_rewards
                        cur_reward_sum += rewards + intrinsic_rewards
                    else:
                        cur_reward_sum += rewards

                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                    if runner.alg.rnd:
                        erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_ereward_sum[new_ids] = 0
                        cur_ireward_sum[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            runner.alg.compute_returns(obs)

        # ── 3b. Policy update ──
        loss_dict = runner.alg.update()

        stop = time.time()
        learn_time = stop - start
        runner.current_learning_iteration = it

        # ── 3c. 발산 감지 & 롤백 ──
        value_loss = loss_dict.get("value_function", 0.0)

        if value_loss > value_loss_threshold or not torch.isfinite(torch.tensor(value_loss)):
            print(f"\n[SNAPSHOT] !!! Divergence at iteration {it} !!!")
            print(f"[SNAPSHOT] value_loss = {value_loss:.4f} (threshold: {value_loss_threshold})")
            print(f"[SNAPSHOT] Total rollbacks so far: {rollback_count}")

            success = load_snapshot()
            if success:
                rollback_count += 1

                # 환경 리셋
                with torch.inference_mode():
                    runner.env.reset()
                    obs = runner.env.get_observations().to(runner.device)

                # 로깅 버퍼 초기화 (쓰레기값 방지)
                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                rewbuffer.clear()
                lenbuffer.clear()
                ep_infos.clear()

                # 에피소드 길이 재랜덤화 (동시 리셋 방지). 에피소드 시각으로 사건을 일으키는 태스크 (KickSwitch: 킥 요청 시각) 는
                # 끌 것 — 거의 전 env 가 동시에 그 사건을 겪어 다시 발산 (11-07 KickSwitch-V0 2 차 연속 롤백)
                if randomize_after_rollback:
                    runner.env.episode_length_buf = torch.randint_like(
                        runner.env.episode_length_buf,
                        high=int(runner.env.max_episode_length),
                    )

                runner.train_mode()
                gc.collect()
                print(f"[SNAPSHOT] Rollback #{rollback_count} complete.\n")
                continue
            else:
                print("[SNAPSHOT] No snapshots left! Continuing without rollback.")

        # ── 3d. 정상일 때만 스냅샷 갱신 ──
        if (it + 1) % snapshot_interval == 0 and value_loss < value_loss_threshold:
            save_snapshot(it + 1)

        # ── 3e. 로깅 ──
        if runner.log_dir is not None and not runner.disable_logs:
            runner.log(locals())
            ep_infos.clear()
            if it % runner.save_interval == 0:
                runner.save(os.path.join(runner.log_dir, f"model_{it}.pt"))

    # ── 4. 최종 저장 ──
    if runner.log_dir is not None:
        runner.save(os.path.join(runner.log_dir, "model_final.pt"))

    print(f"\n[SNAPSHOT] Training complete.")
    print(f"[SNAPSHOT] Total rollbacks: {rollback_count}")
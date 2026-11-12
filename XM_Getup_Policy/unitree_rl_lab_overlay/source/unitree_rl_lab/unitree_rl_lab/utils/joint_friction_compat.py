"""Isaac Sim 5.x 에서도 관절 마찰을 4.5 와 같게 — 옛 PhysX 마찰 "계수" 모델로 되돌린다.

배경 (2026-11-01 측정, scripts/kick_v1_5/probe_sim_version.py):
  Isaac Sim 5.x (PhysX 107) 는 관절 마찰을 (정지 effort [N·m], 운동 effort [N·m], 점성 계수) 로 바꿨고,
  IsaacLab 은 actuator friction 을 정지 effort 에만 넣는다 (운동 0). 그래서 friction=1.1 이 "멈춰 있을 때만
  1.1 N·m" 가 되고 움직이는 관절엔 마찰이 없다 → 5.1 기본 = 4.5 에서 마찰 0 인 로봇.
  4.5 의 계수 모델 (관절 하중에 비례, 움직이는 동안에도 걸림) 은 5.x 에 폐기 예정 API 로 남아 있고,
  그걸로 계수 1.1 을 넣으면 4.5 와 같은 킥이 나온다 (v1_5b 22300 s 1: 공속 2.79 vs 2.83, 발속도 2.35 vs 2.38).

동작: 5.x 에서 Articulation.write_joint_friction_coefficient_to_sim 을 바꿔
  friction → 옛 계수 (set_dof_friction_coefficients), 새 모델 정지 effort 는 0.
  운동·점성 마찰은 명시적으로 준 값만 새 모델로 그대로 쓴다 (우리 cfg 는 안 씀 → 0).
  actuator 초기화와 randomize_joint_parameters (마찰 DR) 가 모두 이 함수를 거치므로 둘 다 4.5 와 같아진다.
4.x 에서는 아무것도 안 한다. 5.x 의 새 모델을 쓰려면 UNITREE_JOINT_FRICTION=native.
폐기 예정 API 가 없어진 버전이면 조용히 넘어가지 않고 에러를 낸다.
"""
import os

_ENV = "UNITREE_JOINT_FRICTION"
_applied = False


def apply():
    global _applied
    if _applied:
        return
    _applied = True
    from isaaclab.utils.version import get_isaac_sim_version

    ver = get_isaac_sim_version()
    if ver.major < 5:
        return
    mode = os.environ.get(_ENV, "legacy").lower()
    if mode == "native":
        print(f"[joint_friction_compat] Isaac Sim {ver}: {_ENV}=native → 5.x 새 마찰 모델 (4.5 와 다름)", flush=True)
        return
    if mode != "legacy":
        raise ValueError(f"{_ENV}={mode!r} — 'legacy' (기본) 또는 'native'")

    from isaaclab.assets import Articulation

    def write_joint_friction_coefficient_to_sim(self, joint_friction_coeff, joint_dynamic_friction_coeff=None,
                                                joint_viscous_friction_coeff=None, joint_ids=None, env_ids=None):
        view = self.root_physx_view
        if not hasattr(view, "set_dof_friction_coefficients"):
            raise RuntimeError(f"Isaac Sim {ver} 에 옛 관절 마찰 계수 API 가 없다 — 4.5 와 같은 마찰을 못 만든다. "
                               f"{_ENV}=native 로 새 모델을 쓰거나 마찰을 다시 맞출 것.")
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        self._data.joint_friction_coeff[env_ids, joint_ids] = joint_friction_coeff
        if joint_dynamic_friction_coeff is not None:
            self._data.joint_dynamic_friction_coeff[env_ids, joint_ids] = joint_dynamic_friction_coeff
        if joint_viscous_friction_coeff is not None:
            self._data.joint_viscous_friction_coeff[env_ids, joint_ids] = joint_viscous_friction_coeff
        ids_cpu = physx_env_ids.cpu()
        # 새 모델: 정지 effort 0 (마찰은 옛 계수로), 운동·점성은 준 값만
        props = view.get_dof_friction_properties()
        props[ids_cpu, :, 0] = 0.0
        if joint_dynamic_friction_coeff is not None:
            props[ids_cpu, :, 1] = self._data.joint_dynamic_friction_coeff[ids_cpu, :].cpu()
        if joint_viscous_friction_coeff is not None:
            props[ids_cpu, :, 2] = self._data.joint_viscous_friction_coeff[ids_cpu, :].cpu()
        view.set_dof_friction_properties(props, indices=ids_cpu)
        # 옛 모델: 4.5 와 같은 계수
        view.set_dof_friction_coefficients(self._data.joint_friction_coeff.cpu(), indices=ids_cpu)

    Articulation.write_joint_friction_coefficient_to_sim = write_joint_friction_coefficient_to_sim
    print(f"[joint_friction_compat] Isaac Sim {ver}: 관절 마찰을 4.5 계수 모델로 (끄려면 {_ENV}=native)", flush=True)

# ═══════════════════════════════════════════════════════════════════════
#  Getup-V3 전용 MDP — 팔 좌우 구조적 거울상 강제
#
#  【왜 구조로 강제하는가】
#    v2 에서 보상으로 네 판을 시도해 전부 우회당했다:
#        19 판  관절 토크 차이 제곱       -> 허공에서 왼팔 휘두르기
#        20 판  접촉력 norm + 시간예산    -> 낙하/침강 구간을 지지로 팔기
#        21 판  일률 + 순간차이(-40)      -> 팔 일을 다리로 넘기기
#        22 판  일률(+40) + 순간차이(-20) -> 한팔 밀기 힘 키우기
#    복와위 최종 밀기(0.80 s)의 왼팔 수직력은 네 판 모두 0.00 N 이었다.
#    23 판(보상 정리)으로 앙와위 지표는 개선됐지만 **동작은 동일**했다.
#
#    보상 경로는 "측정이 맞아야 성립하는 계획" 인데, 측정과 실제 동작이
#    세 번 갈렸다. 구조적 강제는 결과가 측정에 의존하지 않는다.
#
#  【무엇을 보장하는가】
#    팔 오프셋을 좌우 부호 반전으로 묶으므로 팔 자세는 **물리적으로**
#    거울상이다. 다섯 번째 우회로가 원리적으로 존재하지 않는다.
#
#  【대가】
#    옆으로 기울어 넘어진 경우(side_roll 큰 값) 비대칭 팔 회복이 불가능하다.
#    커리큘럼이 side_roll 을 0.15 -> 0.90 까지 넓히므로, 큰 롤에서 대칭
#    팔만으로 일어날 수 있는지는 **학습해 봐야 아는 미지**다. 안 되면
#    side_roll 범위를 줄이거나 mirror_arms=False 로 끈다.
#
#  【액션 차원을 18 로 유지하는 이유】
#    팔 출력을 3 개로 줄이면(15 차원) symmetry_mx3 의 미러 스펙과
#    실기 배포 코드를 함께 고쳐야 한다. 18 을 유지하고 왼팔 출력을
#    오른팔의 거울상으로 **덮어쓰면** 물리적 보장은 같다.
#    왼팔 출력 3 개는 무시되지만, 미러 손실이 등변성을 요구하므로
#    정책은 자연히 그 값을 오른팔의 거울상으로 맞추게 된다 — 낭비가 아니라
#    미러 손실과 같은 방향이다.
# ═══════════════════════════════════════════════════════════════════════
from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .getup_v1_mdp import GetupJointPositionAction, GetupJointPositionActionCfg


class MirroredArmJointPositionAction(GetupJointPositionAction):
    """팔 오프셋을 좌우 거울상으로 묶는다.

    MX3 팔 관절은 좌우 부호가 반대다 (기본각 r_elbow +1.5 / l_elbow -1.5,
    어깨는 둘 다 0). use_default_offset=True 이므로 액션은 기본각에서의
    오프셋이고, 목표각이 거울상이 되려면

        target_L = -target_R
        default_L + a_L * scale = -(default_R + a_R * scale)
        default_L = -default_R  (실측 확인)  =>  a_L = -a_R

    즉 **오프셋의 부호만 뒤집으면** 된다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._mirror_arms = bool(getattr(cfg, "mirror_arms", True))
        self._ri, self._li = None, None
        if self._mirror_arms:
            self._build_arm_index(cfg)

    def _build_arm_index(self, cfg):
        """액션 벡터 안에서 좌우 팔 관절의 짝을 찾는다 (인덱스 하드코딩 금지)."""
        names = list(cfg.joint_names)
        ri, li = [], []
        for i, n in enumerate(names):
            if not n.startswith("r_"):
                continue
            if not any(k in n for k in ("shoulder", "elbow", "wrist", "arm")):
                continue
            partner = "l_" + n[2:]
            if partner in names:
                ri.append(i)
                li.append(names.index(partner))
        if not ri:
            self._mirror_arms = False
            print("[WARN] MirroredArmJointPositionAction: 팔 관절 짝을 못 찾았다. 미러 끔.")
            return
        dev = self._env.device
        self._ri = torch.tensor(ri, dtype=torch.long, device=dev)
        self._li = torch.tensor(li, dtype=torch.long, device=dev)
        pairs = ", ".join(f"{names[a]}<->{names[b]}" for a, b in zip(ri, li))
        print(f"[INFO] 팔 거울상 강제 ON — {pairs}")

    def process_actions(self, actions: torch.Tensor):
        if self._mirror_arms:
            actions = actions.clone()
            actions[:, self._li] = -actions[:, self._ri]
        super().process_actions(actions)


@configclass
class MirroredArmJointPositionActionCfg(GetupJointPositionActionCfg):
    """GetupJointPositionActionCfg + 팔 거울상 강제 플래그."""

    class_type = MirroredArmJointPositionAction

    mirror_arms: bool = True
    """False 로 두면 GetupJointPositionAction 과 완전히 동일하게 동작한다.

    1 판에서 "대칭 팔로 일어날 수 있는가" 를 확인하고, 물리적으로 불가능하면
    이 값을 False 로 바꿔 보상 경로로 되돌아간다.
    """

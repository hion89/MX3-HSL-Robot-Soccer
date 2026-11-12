from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .reset_hold import *
# ── supine 기상 v1 전용 (reset_hold.py 를 수정하지 않고 상속만 한다) ──
from .getup_v1_mdp import *  # noqa: F401, F403
from . import getup_v1_mdp  # noqa: F401
from . import getup_v1_rewards  # noqa: F401
from . import getup_v3_mdp  # noqa: F401
from . import kick_mdp  # noqa: F401
from . import kick_mdp_v0  # noqa: F401
from . import kick_inside_mdp  # noqa: F401
from . import kick_stability_mdp  # noqa: F401
from . import symmetry_mx3  # noqa: F401
from . import csv_replay_mdp  # noqa: F401

# ── v0_1 전용: 보행 위상만 격리한 CommandTerm ─────────────────────
# v1/v1s 의 htwk_* 모듈과 독립적이다. v0 를 쓰는 기존 환경들은
# 이 import 가 추가돼도 영향을 받지 않는다.
from .htwk_gait_v01 import *  # noqa: F401, F403

# ── 밀림 회복: 지속 외력 / 임펄스 기록 / 정지 보상 게이팅 (RS01 V0-1 도입, 기존 환경은 안 씀) ──
from .push_recovery import *  # noqa: F401, F403

# ── 모터 부하 한도 벌점 / 지형 기준 높이 종료 (RS01 V0-2 도입, 기존 환경은 안 씀) ──
from .motor_load import *  # noqa: F401, F403

# ── 구동기 게인 DR (kp·kd 같은 배율, RS01 V0-5 4 차 도입, 기존 환경은 안 씀) ──
from .actuator_dr import *  # noqa: F401, F403

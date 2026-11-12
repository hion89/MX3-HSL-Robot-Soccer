"""팀 전략 — B-Human Behavior/Strategy (Tactic·Position·SetPlay·ActiveRole) 의 4 인 팀용 단순화.

B-Human 구조:
  - Tactic = 위치(Position) 목록 + 우선순위 그룹. 활동 선수가 줄면 우선순위대로 채운다.
  - 매 프레임 '공을 다룰 선수(PlayBall)' = 공 도달 시간이 가장 짧은 선수 (현재 담당에 보너스, 진동 방지).
  - 나머지는 위치를 비용 행렬(거리)로 배정. 모든 로봇이 같은 팀 정보로 같은 계산 → 합의.
  - Set play (킥오프 등) 는 READY 위치를 따로 둔다.
여기:
  - 골키퍼 = 1 번 고정 (HSL 규칙상 GC 의 goalkeeper 번호). 페널티 에어리어 안의 공만 직접 처리.
  - 필드 선수 위치 = 공 위치의 함수 (B-Human 의 Voronoi 영역 대신 규칙식):
      defender  : 우리 골대 중앙 → 공 선분 위, 골에서 1.5 m 이상 / 공에서 1.2 m 이상
      supporter : 공보다 앞쪽 대각 (리바운드·패스), 공과 반대쪽 y
      midfielder: 공 뒤 1.8 m (4 명 모두 살아 있고 GK 가 공을 안 볼 때만)
    우선순위 = defender > supporter > midfielder (B-Human t112: defender 1, forward 2, midfielder 3).
"""
import itertools
import math
from dataclasses import dataclass, field

from .geometry import Field, Pose, clamp, norm_angle

ROLE_NONE, ROLE_GOALKEEPER, ROLE_DEFENDER, ROLE_MIDFIELDER, ROLE_FORWARD = 0, 1, 2, 3, 4
ROLE_NAMES = {ROLE_NONE: "none", ROLE_GOALKEEPER: "goalkeeper", ROLE_DEFENDER: "defender",
              ROLE_MIDFIELDER: "midfielder", ROLE_FORWARD: "supporter"}


@dataclass
class Agent:
    """전략 계산에 필요한 선수 1 명의 상태 (나 또는 팀 메시지)."""
    number: int
    pose: Pose
    validity: float = 1.0
    time_to_ball: float = 1e3
    playing_ball: bool = False
    role: int = ROLE_NONE
    available: bool = True     # 페널티·넘어짐이면 False


@dataclass
class Decision:
    role: int = ROLE_NONE
    active: bool = False
    active_player: int = 0
    target: Pose = field(default_factory=Pose)
    tactic: str = ""


# ── 공 도달 시간 (B-Human ExpectedGoals/TimeToReachBall 단순화) ──
def time_to_ball(pose, ball, kick_target, v_walk=0.30, w_turn=0.8, behind_penalty=3.0, validity=1.0,
                 ball_age=0.0):
    """pose 에서 공 뒤 (킥 방향 반대쪽 0.35 m) 까지 걸어가 킥 방향을 보는 데 걸리는 시간 [s]."""
    bx, by = ball
    kx, ky = kick_target
    ka = math.atan2(ky - by, kx - bx)
    ax, ay = bx - 0.35 * math.cos(ka), by - 0.35 * math.sin(ka)
    d = math.hypot(ax - pose.x, ay - pose.y)
    face = math.atan2(ay - pose.y, ax - pose.x) if d > 0.3 else ka
    turn = abs(norm_angle(face - pose.th)) + abs(norm_angle(ka - face))
    t = d / v_walk + turn / w_turn
    # 공보다 앞(상대 골 쪽)에 있으면 돌아가야 한다
    rx, ry = pose.x - bx, pose.y - by
    if rx * math.cos(ka) + ry * math.sin(ka) > 0.0:
        t += behind_penalty
    t += (1.0 - clamp(validity, 0.0, 1.0)) * 3.0 + min(ball_age, 10.0) * 0.5
    return t


def kick_target_for(ball, f: Field):
    """상대 골대 안 목표점 (공 y 쪽으로 약간 치우침, 골대 기둥 안쪽 0.3 m)."""
    bx, by = ball
    lim = f.goal_width / 2 - 0.3
    return f.hl + 0.2, clamp(by * 0.3, -lim, lim)


# ── 위치 생성 ──
def goalkeeper_pose(ball, f: Field):
    bx, by = ball
    gx = -f.hl + 0.35
    gy = clamp(by * 0.35, -(f.goal_width / 2 - 0.15), f.goal_width / 2 - 0.15)
    return Pose(gx, gy, math.atan2(by - gy, bx - gx))


def defender_pose(ball, f: Field):
    bx, by = ball
    gx, gy = -f.hl, 0.0
    d = math.hypot(bx - gx, by - gy)
    scale = min(1.0, f.length / 9.0)
    goal_clearance, ball_clearance = 1.5 * scale, 1.2 * scale
    along = clamp(0.45 * d, goal_clearance, max(goal_clearance, d - ball_clearance))
    ux, uy = (bx - gx) / max(d, 1e-6), (by - gy) / max(d, 1e-6)
    x, y = gx + ux * along, gy + uy * along
    x = min(x, 0.5)
    x, y = f.clip_inside(x, y, 0.4)
    return Pose(x, y, math.atan2(by - y, bx - x))


def supporter_pose(ball, f: Field):
    bx, by = ball
    side = -1.0 if by > 0 else 1.0
    x = clamp(bx + 1.3, -1.0, f.hl - 1.3)
    y = clamp(by + side * 1.4, -f.hw + 0.6, f.hw - 0.6)
    if abs(y - by) < 1.0:
        y = by + side * 1.0
    x, y = f.clip_inside(x, y, 0.5)
    return Pose(x, y, math.atan2(by - y, bx - x))


def midfielder_pose(ball, f: Field):
    bx, by = ball
    x = clamp(bx - 1.8, -f.hl + 2.2, f.hl - 2.5)
    y = clamp(by * 0.4, -1.5, 1.5)
    if math.hypot(x - bx, y - by) < 1.2:
        y += 1.0 if y <= by else -1.0
    return Pose(x, y, math.atan2(by - y, bx - x))


FIELD_ROLE_POSES = [(ROLE_DEFENDER, defender_pose), (ROLE_FORWARD, supporter_pose),
                    (ROLE_MIDFIELDER, midfielder_pose)]


# ── READY 위치 (B-Human t121 킥오프 배치를 4 인·S 필드로) ──
def kickoff_poses(we_kick, f: Field):
    """GK 를 뺀 필드 선수 3 자리. 우리 킥오프면 1 명은 센터 서클 안 (공 뒤 0.5 m)."""
    if we_kick:
        return [(ROLE_FORWARD, Pose(-0.5, 0.0, 0.0)),
                (ROLE_MIDFIELDER, Pose(-1.2, 1.6, -0.4)),
                (ROLE_DEFENDER, Pose(-2.6, -0.8, 0.2))]
    r = f.center_circle_radius + 0.45
    return [(ROLE_FORWARD, Pose(-r, 0.0, 0.0)),
            (ROLE_MIDFIELDER, Pose(-1.6, 1.6, -0.5)),
            (ROLE_DEFENDER, Pose(-2.6, -0.8, 0.2))]


def assign(agents, slots, current_roles, hysteresis=0.6):
    """선수 → 자리 배정 (총 이동 거리 최소, 지금 맡은 역할엔 보너스). 브루트포스 (최대 4!)."""
    if not agents or not slots:
        return {}
    slots = slots[:len(agents)]
    best, best_cost = None, float("inf")
    for perm in itertools.permutations(range(len(agents)), len(slots)):
        cost = 0.0
        for si, ai in enumerate(perm):
            role, p = slots[si]
            a = agents[ai]
            c = a.pose.dist(p.x, p.y)
            if current_roles.get(a.number) == role:
                c -= hysteresis
            cost += c
        # 동률이면 번호 작은 선수가 앞 자리 (모든 로봇이 같은 답)
        if cost < best_cost - 1e-6:
            best, best_cost = perm, cost
    return {agents[ai].number: slots[si] for si, ai in enumerate(best)}


def choose_active(agents, ball_xy, f: Field, current_active, gk_number=1, bonus=1.5):
    """PlayBall 담당 = 공 도달 시간 최소. 현재 담당은 bonus 만큼 유리. GK 는 공이 우리 페널티 에어리어 근처일 때만."""
    best, best_t = 0, float("inf")
    for a in agents:
        if not a.available:
            continue
        if a.number == gk_number and not f.in_own_penalty_area(ball_xy[0], ball_xy[1], 0.3):
            continue
        t = a.time_to_ball - (bonus if (a.number == current_active or a.playing_ball) else 0.0)
        if t < best_t - 1e-6 or (abs(t - best_t) <= 1e-6 and a.number < best):
            best, best_t = a.number, t
    return best


def decide_playing(me, mates, ball_xy, f: Field, current_active, current_roles, gk_number=1):
    """PLAYING 중 내 역할·목표 결정. 반환 Decision."""
    agents = [me] + [m for m in mates if m.available]
    active = choose_active(agents, ball_xy, f, current_active, gk_number)
    d = Decision(active_player=active, active=(active == me.number), tactic="dynamic")
    if d.active:
        d.role = ROLE_GOALKEEPER if me.number == gk_number else (current_roles.get(me.number) or ROLE_FORWARD)
        return d
    if me.number == gk_number:
        d.role, d.target = ROLE_GOALKEEPER, goalkeeper_pose(ball_xy, f)
        return d
    field_agents = sorted([a for a in agents if a.number != gk_number and a.number != active], key=lambda a: a.number)
    slots = [(r, fn(ball_xy, f)) for r, fn in FIELD_ROLE_POSES]
    # GK 가 공을 잡으러 나갔으면 defender 가 골대 앞을 메운다
    if active == gk_number:
        slots[0] = (ROLE_DEFENDER, Pose(-f.hl + 0.6, clamp(ball_xy[1] * 0.3, -0.6, 0.6), 0.0))
    table = assign(field_agents, slots, current_roles)
    role, pose = table.get(me.number, (ROLE_DEFENDER, defender_pose(ball_xy, f)))
    d.role, d.target = role, pose
    return d


def decide_assigned_role(assigned_role, me, ball_xy, f: Field):
    """Explicit role assignment for a single real robot; reuse the team skills/poses."""
    if assigned_role == "attacker":
        return Decision(role=ROLE_FORWARD, active=True, active_player=me.number, tactic="play_ball")
    if assigned_role == "defender":
        return Decision(role=ROLE_DEFENDER, target=defender_pose(ball_xy, f), tactic="defend")
    if assigned_role == "goalkeeper":
        active = f.in_own_penalty_area(ball_xy[0], ball_xy[1], 0.3)
        return Decision(role=ROLE_GOALKEEPER, active=active,
                        active_player=me.number if active else 0,
                        target=goalkeeper_pose(ball_xy, f), tactic="goalkeeper")
    if assigned_role != "auto":
        raise ValueError("assigned_role must be auto, attacker, defender or goalkeeper")
    return None


def decide_ready(me, mates, f: Field, we_kick, current_roles, gk_number=1):
    if me.number == gk_number:
        return Decision(role=ROLE_GOALKEEPER, target=Pose(-f.hl + 0.5, 0.0, 0.0), tactic="kickoff")
    agents = sorted([a for a in [me] + mates if a.available and a.number != gk_number], key=lambda a: a.number)
    table = assign(agents, kickoff_poses(we_kick, f), current_roles, hysteresis=0.3)
    role, pose = table.get(me.number, (ROLE_DEFENDER, Pose(-2.6, 0.0, 0.0)))
    return Decision(role=role, target=pose, tactic="kickoff_own" if we_kick else "kickoff_opp")


def search_pose(number, f: Field, gk_number=1):
    """공을 아무도 모를 때 흩어지는 자리 (B-Human 의 탐색 파티클 대신 고정 분산)."""
    if number == gk_number:
        return Pose(-f.hl + 0.5, 0.0, 0.0)
    spots = {2: Pose(-1.5, 0.0, 0.0), 3: Pose(0.8, 1.4, 0.0), 4: Pose(0.8, -1.4, 0.0)}
    return spots.get(number, Pose(0.0, 0.0, 0.0))

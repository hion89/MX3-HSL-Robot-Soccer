"""공 다툼 (duel) 판단 — 상대가 공 가까이 있을 때 공 담당이 어떻게 할지.

시뮬 4v4 에서 넘어짐 대부분이 '양 팀 공 담당이 같은 공을 동시에 차며 부딪힘' 이었다 (DESIGN 5-3).
규칙 (모두 로봇 기준 좌표, m / rad):
  - 다툼 = 상대가 공에서 contest_radius 안.
  - 상대가 나보다 공에 closer_margin 넘게 가까우면 → 'block': 차러 들어가지 않고 공과 우리 골대 사이에 선다
    (상대가 차면 공이 우리 몸에 막힌다. 부딪히지 않는다).
  - 내가 더 가깝거나 비슷하면 → 'deflect': 킥 방향을 상대 쪽에서 sep_min 이상 벗어나게 튼다 (골 방향 ±max_dev 안).
  - 상대가 몸 바로 앞(front_block 안, 좌우 0.3 m)이면 → 'hold': 킥을 시작하지 않는다 (킥 정책은 앞으로 걸어 들어간다).
B-Human 은 상대 위치를 킥 선택(KickSelection)과 경로 계획에 넣는다. 여기선 그 최소판.
"""
import math

from .geometry import clamp, norm_angle


def opponents_from_obstacles(obstacles, mates_local, mate_radius=0.45):
    """로봇 검출(팀원 포함) 에서 팀원이 보고한 위치 근처를 빼 상대만 남긴다."""
    out = []
    for ox, oy in obstacles:
        if all(math.hypot(ox - mx, oy - my) > mate_radius for mx, my in mates_local):
            out.append((ox, oy))
    return out


def duel_decision(ball, opponents, ka, contest_radius=0.6, closer_margin=0.10,
                  sep_min=math.radians(50), max_dev=math.radians(60), front_block=0.40):
    """ball=(bx, by), opponents=[(x, y)], ka = 골 쪽 킥 방향.
    반환 (mode, kick_angle, opponent): mode ∈ none / block / deflect / hold."""
    bx, by = ball
    near = [(math.hypot(ox - bx, oy - by), (ox, oy)) for ox, oy in opponents]
    near = [n for n in near if n[0] < contest_radius]
    for ox, oy in opponents:
        if 0.0 < ox < front_block and abs(oy) < 0.3:
            return "hold", ka, (ox, oy)
    if not near:
        return "none", ka, None
    d_ob, opp = min(near)
    d_mb = math.hypot(bx, by)
    if d_ob < d_mb - closer_margin:
        return "block", None, opp
    beta = math.atan2(opp[1] - by, opp[0] - bx)       # 공에서 본 상대 방향
    sep = norm_angle(ka - beta)
    if abs(sep) >= sep_min:
        return "deflect", ka, opp
    side = 1.0 if sep >= 0 else -1.0
    cand = norm_angle(beta + side * sep_min)
    # 골 방향에서 너무 벗어나면 반대쪽으로 돌려 보고, 그래도 넘으면 한계까지만
    if abs(norm_angle(cand - ka)) > max_dev:
        other = norm_angle(beta - side * sep_min)
        cand = other if abs(norm_angle(other - ka)) < abs(norm_angle(cand - ka)) else cand
        dev = clamp(norm_angle(cand - ka), -max_dev, max_dev)
        cand = norm_angle(ka + dev)
    return "deflect", cand, opp


def block_pose(ball_field, own_goal=(-4.5, 0.0), dist=0.6):
    """공과 우리 골대 중앙을 잇는 선 위, 공에서 dist 떨어진 자리 (필드 좌표) 와 공을 보는 방향."""
    bx, by = ball_field
    gx, gy = own_goal
    d = math.hypot(gx - bx, gy - by)
    ux, uy = (gx - bx) / max(d, 1e-6), (gy - by) / max(d, 1e-6)
    x, y = bx + dist * ux, by + dist * uy
    return x, y, math.atan2(by - y, bx - x)

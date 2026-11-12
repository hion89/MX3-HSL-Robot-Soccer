import math

from hsl_behavior import strategy as S
from hsl_behavior.geometry import Field, Pose
from hsl_behavior.skills import CommandShaper, GoToBallAndKick, KickEnvelope, WalkLimits, WalkToPose

F = Field()


def agents(poses):
    return [S.Agent(number=i + 1, pose=p) for i, p in enumerate(poses)]


def test_ready_assignment_unique():
    a = agents([Pose(-3.8, 3.4, -1.57), Pose(-3.1, 3.4, -1.57), Pose(-2.4, 3.4, -1.57), Pose(-1.7, 3.4, -1.57)])
    targets = set()
    for me in a[1:]:
        d = S.decide_ready(me, [m for m in a if m is not me], F, True, {})
        targets.add((d.role, round(d.target.x, 2), round(d.target.y, 2)))
    assert len(targets) == 3, targets


def test_single_active_and_gk_stays():
    ball = (0.5, 0.3)
    a = agents([Pose(-4.0, 0, 0), Pose(-1.0, 0, 0), Pose(0.0, 2.0, 0), Pose(-2.0, -1.0, 0)])
    for x in a:
        x.time_to_ball = S.time_to_ball(x.pose, ball, S.kick_target_for(ball, F))
    actives, roles = set(), []
    for me in a:
        d = S.decide_playing(me, [m for m in a if m is not me], ball, F, 0, {})
        actives.add(d.active_player)
        roles.append((me.number, d.role, d.active))
    assert len(actives) == 1 and 1 not in actives, (actives, roles)
    field_roles = [r for n, r, act in roles if n != 1 and not act]
    assert len(set(field_roles)) == len(field_roles), roles


def test_gk_active_in_penalty_area():
    ball = (-3.9, 0.2)
    a = agents([Pose(-4.2, 0, 0), Pose(-1.0, 0, 0), Pose(0.0, 2.0, 0), Pose(-2.0, -1.0, 0)])
    for x in a:
        x.time_to_ball = S.time_to_ball(x.pose, ball, S.kick_target_for(ball, F))
    assert S.choose_active(a, ball, F, 0) == 1


def test_kick_handoff_envelope():
    lim = WalkLimits()
    k = GoToBallAndKick(lim, KickEnvelope())
    res, val = k(0.35, 0.05, 0.1, 0.1, 0.0, 0.0)
    assert res == "kick"
    res, val = k(0.35, 0.05, math.radians(80), 0.1, 0.0, 0.0)
    assert res == "walk"
    res, val = k(3.0, 0.0, 0.0, 0.1, 0.0, 0.0)
    assert res == "walk" and val[0] > 0.2


def test_orbit_when_in_front_of_ball():
    k = GoToBallAndKick(WalkLimits(), KickEnvelope())
    # 공이 로봇 앞 0.5 m, 킥 방향은 로봇 쪽 (= 로봇이 공과 목표 사이에 서 있음)
    res, val = k(0.5, 0.0, math.pi, 0.1, 0.0, 0.0)
    assert res == "walk" and k.phase == "orbit"


def test_shaper_min_norm():
    s = CommandShaper(WalkLimits())
    out = s((0.05, 0.0, 0.0), 0.05)
    assert math.hypot(*out) >= 0.119
    out = s((0.0, 0.0, 0.0), 0.05)
    assert out == (0.0, 0.0, 0.0)


def test_walk_arrival_latch():
    w = WalkToPose(WalkLimits())
    assert w(0.05, 0.0, 0.0) == (0.0, 0.0, 0.0) and w.arrived
    assert w(0.2, 0.0, 0.0) == (0.0, 0.0, 0.0)       # 래치 유지
    assert w(0.5, 0.0, 0.0) != (0.0, 0.0, 0.0)       # 이탈


def test_duel_modes():
    from hsl_behavior import duel
    ka = 0.0
    # 상대 없음
    assert duel.duel_decision((0.4, 0.0), [], ka)[0] == "none"
    # 상대가 공 건너편, 나보다 공에 가까움 → 막기
    assert duel.duel_decision((0.8, 0.0), [(1.1, 0.0)], ka)[0] == "block"
    # 내가 더 가깝고 상대가 킥 방향 정면 → 50° 이상 비켜 차기, 골 방향 ±60° 안
    mode, k2, _ = duel.duel_decision((0.4, 0.0), [(0.9, 0.05)], ka)
    beta = math.atan2(0.05, 0.5)                       # 공에서 본 상대 방향
    assert mode == "deflect", mode
    assert abs(math.remainder(k2 - beta, 2 * math.pi)) >= math.radians(49.9), math.degrees(k2)
    assert abs(k2 - ka) <= math.radians(60.1), math.degrees(k2)
    # 상대가 몸 바로 앞 → 킥 시작 안 함
    assert duel.duel_decision((0.4, 0.0), [(0.3, 0.1)], ka)[0] == "hold"
    # 팀원 위치 근처 검출은 상대가 아니다
    assert duel.opponents_from_obstacles([(1.0, 0.0), (2.0, 1.0)], [(1.1, 0.05)]) == [(2.0, 1.0)]

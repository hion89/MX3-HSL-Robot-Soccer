"""스킬 — B-Human Skills (WalkToPoint / GoToBallAndKick / SearchForBall) 의 우리 정책용 단순화.

전부 '로봇 기준 좌표' 에서 계산한다 (x 앞, y 왼쪽). 출력은 보행 정책 명령 (vx, vy, wz).
보행 정책 규약: 명령 노름 ‖(vx, vy, wz)‖ < standing_threshold(0.1) 이면 제자리 정지 → 움직일 땐 최소 노름을 보장.

킥 핸드오프 (기존 visual_servo 와 가장 다른 점):
  기존 서보는 공을 정해진 스위트스폿 (0.30, 0.05) ±수 cm 로 '정밀 정렬' 한 뒤 킥 정책을 불렀다.
  그 정렬 루프가 공 앞에서 앞뒤로 왔다갔다 하는 진동의 원인이었다.
  킥 정책은 공 0.2~0.5 m · 방위 ±90° · 목표 ±90° 에서 '걸어가서 차도록' 학습됐으므로
  여기서는 학습 분포 안쪽 여유 영역 (거리 0.25~0.45, 방위 ±55°, 목표 ±35°) 에 들어오면 바로 넘긴다.
  B-Human 도 InWalkKick 으로 '걸으면서' 차고, 정렬은 대략적으로만 한다.
"""
import math
from dataclasses import dataclass

from .geometry import clamp, norm_angle


@dataclass
class WalkLimits:
    vx_max: float = 0.40
    vx_back: float = 0.20
    vy_max: float = 0.25
    wz_max: float = 1.0
    min_norm: float = 0.12        # 보행 정책 standing_threshold(0.1) + 여유
    accel_lin: float = 0.6        # m/s²
    accel_ang: float = 2.5        # rad/s²


class CommandShaper:
    """최소 노름 보장 + 가속 제한 (급출발·급정지로 넘어지지 않게)."""

    def __init__(self, lim: WalkLimits):
        self.lim = lim
        self.prev = (0.0, 0.0, 0.0)

    def __call__(self, cmd, dt):
        vx, vy, wz = cmd
        n = math.sqrt(vx * vx + vy * vy + wz * wz)
        if 1e-6 < n < self.lim.min_norm:
            s = self.lim.min_norm / n
            vx, vy, wz = vx * s, vy * s, wz * s
        px, py, pw = self.prev
        a, aw = self.lim.accel_lin * dt, self.lim.accel_ang * dt
        if n <= 1e-6:      # 정지는 즉시 (정책이 알아서 멈춘다)
            out = (0.0, 0.0, 0.0)
        else:
            out = (px + clamp(vx - px, -a, a), py + clamp(vy - py, -a, a), pw + clamp(wz - pw, -aw, aw))
            on = math.sqrt(sum(v * v for v in out))
            if on < self.lim.min_norm:      # 출발 직후에도 정지 판정이 안 나게: 명령 방향으로 최소 노름
                s = self.lim.min_norm / max(math.sqrt(vx * vx + vy * vy + wz * wz), 1e-6)
                out = (vx * s, vy * s, wz * s)
        self.prev = out
        return out


class WalkToPose:
    """목표 pose 로 이동 (멀면 몸을 경로로 돌려 전진, 가까우면 전방향). 도착 래치 (히스테리시스)."""

    def __init__(self, lim: WalkLimits, pos_tol=0.15, yaw_tol=math.radians(15), leave_pos=0.35,
                 leave_yaw=math.radians(35)):
        self.lim = lim
        self.pos_tol, self.yaw_tol, self.leave_pos, self.leave_yaw = pos_tol, yaw_tol, leave_pos, leave_yaw
        self.arrived = False
        self.target_local = None  # Display telemetry; never used to compute motion.

    def reset(self):
        self.arrived = False
        self.target_local = None

    def __call__(self, tx, ty, tth, final_heading=True, avoid=(0.0, 0.0)):
        """tx, ty: 로봇 기준 목표점. tth: 로봇 기준 목표 heading 오차. avoid: 회피 속도 (vx, vy) 가산."""
        L = self.lim
        d = math.hypot(tx, ty)
        if not final_heading:
            tth = math.atan2(ty, tx) if d > 0.2 else 0.0
        self.target_local = (tx, ty, tth)
        if self.arrived:
            if d > self.leave_pos or abs(tth) > self.leave_yaw:
                self.arrived = False
            else:
                return 0.0, 0.0, 0.0
        if d < self.pos_tol and abs(tth) < self.yaw_tol:
            self.arrived = True
            return 0.0, 0.0, 0.0
        path = math.atan2(ty, tx)
        if d > 0.7:
            # 멀리: 경로를 바라보고 전진. 1.5 m 안쪽부터 최종 heading 쪽으로 서서히 섞는다
            w = clamp((1.5 - d) / 0.8, 0.0, 1.0)
            face = path if not final_heading else norm_angle(path + w * norm_angle(tth - path))
            if abs(path) > math.radians(55):
                return 0.0, 0.0, clamp(1.4 * path, -L.wz_max, L.wz_max)
            vx = L.vx_max * clamp(d / 1.0, 0.4, 1.0) * max(0.0, math.cos(path))
            vy = clamp(0.6 * ty, -L.vy_max, L.vy_max) if abs(path) < math.radians(30) else 0.0
            wz = clamp(1.2 * face, -L.wz_max, L.wz_max)
            vx, vy = vx + avoid[0], vy + avoid[1]
            return clamp(vx, -L.vx_back, L.vx_max), clamp(vy, -L.vy_max, L.vy_max), wz
        # 가까이: 전방향
        vx = clamp(0.9 * tx, -L.vx_back, L.vx_max * 0.7) + avoid[0]
        vy = clamp(1.0 * ty, -L.vy_max, L.vy_max) + avoid[1]
        wz = clamp(1.2 * tth, -L.wz_max, L.wz_max)
        return clamp(vx, -L.vx_back, L.vx_max), clamp(vy, -L.vy_max, L.vy_max), wz


@dataclass
class KickEnvelope:
    """핸드오프 영역 (킥 정책 학습 분포: 거리 0.2~0.5, 방위 ±90°, 목표 ±90° 의 안쪽)."""
    dist_min: float = 0.25
    dist_max: float = 0.45
    bearing_max: float = math.radians(55)
    target_max: float = math.radians(35)
    target_max_relaxed: float = math.radians(60)   # 공 옆에서 오래 못 맞추면 완화
    relax_after_s: float = 6.0
    max_ball_age: float = 0.4
    max_ball_speed: float = 0.4
    standoff: float = 0.36                         # 공 뒤 대기 거리 (킥 정책 기본 공 거리 0.35)


class GoToBallAndKick:
    """공 뒤로 가서 킥 정책에 넘긴다. 반환: ('walk', cmd) 또는 ('kick', (ball_x, ball_y, target_yaw))."""

    def __init__(self, lim: WalkLimits, env: KickEnvelope):
        self.lim, self.env = lim, env
        self.walk = WalkToPose(lim, pos_tol=0.04, yaw_tol=math.radians(8), leave_pos=0.10,
                               leave_yaw=math.radians(15))
        self.near_since = None
        self.phase = "far"

    def reset(self):
        self.walk.reset()
        self.near_since = None
        self.phase = "far"

    def in_envelope(self, bx, by, ka, t_near):
        e = self.env
        d = math.hypot(bx, by)
        tmax = e.target_max_relaxed if (t_near is not None and t_near > e.relax_after_s) else e.target_max
        return e.dist_min <= d <= e.dist_max and abs(math.atan2(by, bx)) <= e.bearing_max and abs(ka) <= tmax

    def __call__(self, bx, by, ka, ball_age, ball_speed, now, avoid=(0.0, 0.0)):
        """bx, by: 로봇 기준 공. ka: 로봇 기준 킥 방향 (공 → 목표). now: 초."""
        e = self.env
        d = math.hypot(bx, by)
        if d < 1.0:
            if self.near_since is None:
                self.near_since = now
        else:
            self.near_since = None
        t_near = None if self.near_since is None else now - self.near_since
        if ball_age <= e.max_ball_age and ball_speed <= e.max_ball_speed and self.in_envelope(bx, by, ka, t_near):
            return "kick", (bx, by, ka)

        ux, uy = math.cos(ka), math.sin(ka)
        # 공에서 본 로봇 방향과 '공 뒤' 방향(−u) 사이 각
        rx, ry = -bx, -by
        phi = norm_angle(math.atan2(ry, rx) - math.atan2(-uy, -ux))
        if d > 1.6:
            # 멀리: 공 뒤 0.6 m 지점으로 경로를 보며 걷기
            self.phase = "far"
            ax, ay = bx - 0.6 * ux, by - 0.6 * uy
            if abs(phi) > math.radians(100):     # 공 앞쪽에 있으면 옆으로 크게 돌아 들어갈 지점
                s = 1.0 if phi > 0 else -1.0
                ox, oy = rot(-ux, -uy, s * math.radians(80))
                ax, ay = bx + 0.7 * ox, by + 0.7 * oy
            cmd = self.walk(ax, ay, 0.0, final_heading=False, avoid=avoid)
            return "walk", cmd
        if abs(phi) > math.radians(50):
            # 공을 중심으로 돌기 (반지름 0.5, 한 번에 50° 씩 공 뒤쪽으로), 몸은 공을 본다
            self.phase = "orbit"
            s = 1.0 if phi > 0 else -1.0
            wphi = phi - s * math.radians(50)
            ox, oy = rot(-ux, -uy, wphi)
            wx, wy = bx + 0.5 * ox, by + 0.5 * oy
            face = math.atan2(by, bx)
            self.walk.arrived = False
            return "walk", self.walk(wx, wy, face, final_heading=True, avoid=avoid)
        # 공 뒤 standoff 지점, 킥 방향을 본다
        self.phase = "align"
        px, py = bx - e.standoff * ux, by - e.standoff * uy
        self.walk.arrived = False
        cmd = self.walk(px, py, ka, final_heading=True, avoid=(0.0, 0.0))
        return "walk", cmd


def rot(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return c * x - s * y, s * x + c * y


class SearchForBall:
    """제자리 한 바퀴 (목은 좌우 훑기) → 탐색 위치로 이동 → 다시 한 바퀴."""

    def __init__(self, lim: WalkLimits, turn_speed=0.6):
        self.lim, self.turn_speed = lim, turn_speed
        self.walk = WalkToPose(lim, pos_tol=0.3, yaw_tol=math.pi, leave_pos=0.8, leave_yaw=math.pi)
        self.reset()

    def reset(self, now=None):
        self.turned = 0.0
        self.phase = "turn"
        self.last = now

    def __call__(self, now, local_target, dt, turn_dir=1.0):
        if self.phase == "turn":
            self.turned += abs(self.turn_speed) * dt
            if self.turned >= 2 * math.pi:
                self.phase = "move"
                self.walk.reset()
            return 0.0, 0.0, turn_dir * self.turn_speed
        tx, ty = local_target
        cmd = self.walk(tx, ty, 0.0, final_heading=False)
        if self.walk.arrived:
            self.phase, self.turned = "turn", 0.0
        return cmd

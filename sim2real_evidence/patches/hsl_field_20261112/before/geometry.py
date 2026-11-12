"""필드 치수와 2D 기하 (hsl_world_model/field_model.hpp 와 같은 값).

좌표: 원점 = 센터, +x = 상대 골대 방향, +y = 왼쪽, m / rad. 모든 팀원이 '자기 팀 기준' 같은 좌표를 쓴다.
"""
import math
from dataclasses import dataclass


def norm_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    th: float = 0.0

    def to_local(self, px, py):
        """필드 점 → 이 pose 기준 (x 앞, y 왼쪽)."""
        dx, dy = px - self.x, py - self.y
        c, s = math.cos(self.th), math.sin(self.th)
        return c * dx + s * dy, -s * dx + c * dy

    def to_field(self, lx, ly):
        c, s = math.cos(self.th), math.sin(self.th)
        return self.x + c * lx - s * ly, self.y + s * lx + c * ly

    def dist(self, px, py):
        return math.hypot(px - self.x, py - self.y)


@dataclass
class Field:
    length: float = 9.0
    width: float = 6.0
    center_circle_radius: float = 0.75
    penalty_mark_dist: float = 1.5
    penalty_area_depth: float = 2.0
    penalty_area_width: float = 4.0
    goal_area_depth: float = 1.0
    goal_area_width: float = 3.0
    goal_width: float = 1.8

    @property
    def hl(self):
        return self.length / 2

    @property
    def hw(self):
        return self.width / 2

    @staticmethod
    def for_profile(profile):
        f = Field()
        if profile == "practice":
            s = 3.9 / 9.0
            f.length, f.width = 3.9, 2.6
            for k in ("center_circle_radius", "penalty_mark_dist", "penalty_area_depth", "penalty_area_width",
                      "goal_area_depth", "goal_area_width", "goal_width"):
                setattr(f, k, getattr(f, k) * s)
        return f

    def in_own_penalty_area(self, x, y, margin=0.0):
        return x < -self.hl + self.penalty_area_depth + margin and abs(y) < self.penalty_area_width / 2 + margin

    def clip_inside(self, x, y, margin=0.3):
        return clamp(x, -self.hl + margin, self.hl - margin), clamp(y, -self.hw + margin, self.hw - margin)

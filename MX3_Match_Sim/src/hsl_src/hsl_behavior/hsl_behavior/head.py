"""목 제어 — B-Human HeadControl / LookActive 단순화.

추적 (track): 목표점을 TF 로 카메라 광학 좌표에 옮겨, 기존 neck_ball_tracker 와 같은 법칙으로 닫힌 루프
    yaw ← 현재 yaw − kp·deg(atan2(x, z)),  pitch ← 현재 pitch − kp·deg(atan2(y, z))
  카메라 장착 오프셋을 몰라도 된다 (TF 가 안다). 목표가 카메라 뒤면 몸통 방위로 yaw 를 직접 준다.
LookActive: 공을 보다가 주기적으로 좌우 '랜드마크 훑기' (측위용). 공이 가깝거나 킥 직전엔 공만 본다.
탐색 (scan): yaw 를 ±70° 사인파로, 한 번 훑을 때마다 pitch 를 가까이/멀리 번갈아.
"""
import math

from .geometry import clamp


class HeadControl:
    def __init__(self, yaw_lim=(-80.0, 80.0), pitch_lim=(-60.0, -25.0), kp=0.6, max_step=6.0,
                 glance_period=5.0, glance_duration=1.6, scan_period=4.0):
        self.yaw_lim, self.pitch_lim = yaw_lim, pitch_lim
        self.kp, self.max_step = kp, max_step
        self.glance_period, self.glance_duration, self.scan_period = glance_period, glance_duration, scan_period
        self.yaw, self.pitch = 0.0, -30.0           # 마지막 명령 [deg]
        self.cur_yaw, self.cur_pitch = None, None    # 측정값 [deg]
        self.last_glance_end = 0.0
        self.glance_start = None
        self.glance_side = 1.0
        self.scan_t0 = None

    def _step(self, yaw, pitch):
        yaw = clamp(yaw, *self.yaw_lim)
        pitch = clamp(pitch, *self.pitch_lim)
        self.yaw += clamp(yaw - self.yaw, -self.max_step, self.max_step)
        self.pitch += clamp(pitch - self.pitch, -self.max_step, self.max_step)
        return self.yaw, self.pitch

    def track(self, p_cam, p_base):
        """p_cam: 카메라 광학 좌표 (x 오른쪽, y 아래, z 앞) 또는 None. p_base: 몸통 좌표 (x, y, z)."""
        cy = self.cur_yaw if self.cur_yaw is not None else self.yaw
        cp = self.cur_pitch if self.cur_pitch is not None else self.pitch
        if p_cam is None or p_cam[2] < 0.05:
            yaw = math.degrees(math.atan2(p_base[1], p_base[0]))
            d = math.hypot(p_base[0], p_base[1])
            pitch = -25.0 - 35.0 * clamp((1.5 - d) / 1.2, 0.0, 1.0)
            return self._step(yaw, pitch)
        eh = math.degrees(math.atan2(p_cam[0], p_cam[2]))
        ev = math.degrees(math.atan2(p_cam[1], p_cam[2]))
        yaw = cy - self.kp * eh if abs(eh) > 1.0 else self.yaw
        pitch = cp - self.kp * ev if abs(ev) > 1.0 else self.pitch
        return self._step(yaw, pitch)

    def scan(self, now, pitch_pair=(-28.0, -42.0), amplitude=70.0):
        if self.scan_t0 is None:
            self.scan_t0 = now
        t = now - self.scan_t0
        yaw = amplitude * math.sin(2 * math.pi * t / self.scan_period)
        k = int(t / (self.scan_period / 2)) % 2
        return self._step(yaw, pitch_pair[k])

    def stop_scan(self):
        self.scan_t0 = None

    def look_active(self, now, p_cam, p_base, allow_glance):
        """공 추적 + 주기적 좌우 훑기. 훑는 중이면 True 반환 (두 번째 값)."""
        if self.glance_start is not None:
            t = now - self.glance_start
            if t < self.glance_duration and allow_glance:
                yaw = self.glance_side * 60.0 * math.sin(math.pi * t / self.glance_duration)
                return self._step(yaw, -27.0), True
            self.glance_start = None
            self.last_glance_end = now
            self.glance_side = -self.glance_side
        if allow_glance and now - self.last_glance_end > self.glance_period:
            self.glance_start = now
        return self.track(p_cam, p_base), False

    def neutral(self):
        return self._step(0.0, -30.0)

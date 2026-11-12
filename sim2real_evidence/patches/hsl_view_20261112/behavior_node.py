#!/usr/bin/env python3
"""HSL 행동 노드 — B-Human Behavior (Strategy → Skill → Motion/Head 요청) 의 ROS 이식.

입력  /game_state (GC 수신), /hsl/robot_pose, /hsl/ball, /hsl/team_ball, /hsl/team_state, /robot_state,
      /kick_done, /neck/{yaw,pitch}_deg_state, TF (base_link → 카메라)
출력  cmd_vel_behavior (twist_mux behavior 슬롯 → 보행 정책), /neck/{yaw,pitch}_deg,
      /kick_trigger + 킥 노드 SetParameters (ball_pos_b_x/y, kick_target_yaw_offset_rad),
      /hsl/strategy_state (팀 통신이 실어 보냄), /hsl/behavior_debug (JSON)
경기 상태: INITIAL·SET·FINISHED·페널티 = 정지 (twist_mux 락도 game_state_to_lock 이 건다)
           READY = 킥오프 위치로 / PLAYING = 전략 (strategy.py) + 스킬 (skills.py)
"""
import json
import math
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue, SetParametersResult
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import PoseArray, Twist
from std_msgs.msg import Bool, Float32, String

from hsl_interfaces.msg import BallEstimate, RobotPoseEstimate, StrategyState, TeamBall, TeamState
from humanoid_behavior_interfaces.msg import GameState

from . import duel
from . import strategy as S
from .geometry import Field, Pose, clamp, norm_angle
from .head import HeadControl
from .skills import CommandShaper, GoToBallAndKick, KickEnvelope, SearchForBall, WalkLimits, WalkToPose

STATE_NAMES = {0: "INITIAL", 1: "READY", 2: "SET", 3: "PLAYING", 4: "FINISHED"}


class Behavior(Node):
    def __init__(self):
        super().__init__("hsl_behavior")
        P = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        self.player = int(P("player_number", 1))
        self.gk = int(P("goalkeeper_number", 1))
        self.field = Field.for_profile(str(P("field_profile", "s_field")))
        self.kick_node = str(P("kick_node", "actor_onnx_walkkick_v1_node"))
        self.default_state = str(P("default_game_state", "initial")).upper()
        self.assigned_role = str(P("assigned_role", "auto")).lower()
        if self.assigned_role not in ("auto", "attacker", "defender", "goalkeeper"):
            raise ValueError("Invalid assigned_role")
        self.opp_kickoff_wait = float(P("opponent_kickoff_wait", 10.0))
        self.set_play_distance = float(P("opponent_set_play_distance", 1.2))
        self.ball_lost_time = float(P("team_ball_lost_time", 3.0))
        self.own_ball_fresh = float(P("own_ball_fresh_time", 1.0))
        self.kick_timeout = float(P("kick_timeout", 6.0))
        self.kick_cooldown = float(P("kick_cooldown", 1.0))
        self.enable_kick = bool(P("enable_kick", True))
        self.ttb_speed = float(P("time_to_ball_walk_speed", 0.30))
        self.obstacle_ttl = float(P("obstacle_ttl", 1.5))           # 로봇 검출 유지 시간 [s]
        self.obstacle_radius = float(P("obstacle_avoid_radius", 0.9))
        # 공 다툼 (duel.py): 상대가 공 0.6 m 안 → 상대가 더 가까우면 막기, 아니면 상대를 피해 비켜 차기
        self.duel_enable = bool(P("duel_enable", True))
        self.duel_contest_radius = float(P("duel_contest_radius", 0.6))
        self.duel_sep_min = math.radians(float(P("duel_deflect_min_deg", 50.0)))
        self.duel_max_dev = math.radians(float(P("duel_deflect_max_deg", 60.0)))
        self.duel_mode = "none"
        rate = float(P("rate_hz", 20.0))
        self.lim = WalkLimits(vx_max=float(P("vx_max", 0.40)), vy_max=float(P("vy_max", 0.25)),
                              wz_max=float(P("wz_max", 1.0)), min_norm=float(P("min_cmd_norm", 0.12)))
        env = KickEnvelope(dist_min=float(P("kick_dist_min", 0.25)), dist_max=float(P("kick_dist_max", 0.45)),
                           bearing_max=math.radians(float(P("kick_bearing_max_deg", 55.0))),
                           target_max=math.radians(float(P("kick_target_max_deg", 35.0))),
                           target_max_relaxed=math.radians(float(P("kick_target_max_relaxed_deg", 60.0))))

        self.shaper = CommandShaper(self.lim)
        self.walk = WalkToPose(self.lim)
        self.play = GoToBallAndKick(self.lim, env)
        self.search = SearchForBall(self.lim)
        self.head = HeadControl()

        self.gc = None
        self.gc_rx = 0.0
        self.pose = None
        self.ball = None
        self.team_ball = None
        self.team = None
        self.robot_state = "NORMAL"
        self.team_ball_seen = 0.0         # 팀 공이 마지막으로 유효했던 시각
        self.last_team_ball = None
        self.active_player = 0
        self.roles = {}
        self.my_role = S.ROLE_NONE
        self.skill = "stand"
        self.prev_state = None
        self.playing_since = None
        self.kickoff_ball = None
        self.kick_phase = None            # None / arming / kicking / cooldown
        self.kick_t = 0.0
        self.kick_info = None
        self.kicks = 0
        self.last_t = time.monotonic()
        self.dbg_t = 0.0
        self.obstacles = []               # [(x, y)] 몸통 좌표, YOLO 'robot' 검출 (ball_tracker 가 odom 에서 유지)
        self.obstacles_t = 0.0

        self.create_subscription(GameState, "/game_state", self._on_gc, 10)
        self.create_subscription(RobotPoseEstimate, "/hsl/robot_pose", lambda m: setattr(self, "pose", m), 10)
        self.create_subscription(BallEstimate, "/hsl/ball", lambda m: setattr(self, "ball", m), 10)
        self.create_subscription(TeamBall, "/hsl/team_ball", lambda m: setattr(self, "team_ball", m), 10)
        self.create_subscription(TeamState, "/hsl/team_state", lambda m: setattr(self, "team", m), 10)
        self.create_subscription(String, "/robot_state", lambda m: setattr(self, "robot_state", m.data), 10)
        self.create_subscription(Bool, "/kick_done", self._on_kick_done, 10)
        # 장애물·카메라 기준 공은 ball_tracker(C++)가 계산해 준다 — Python TF 리스너는 로봇당 CPU 15~30 % 라 뺐다
        self.create_subscription(PoseArray, "/hsl/obstacles", self._on_obstacles, 10)
        self.create_subscription(Float32, "/neck/yaw_deg_state", self._on_neck_yaw, 10)
        self.create_subscription(Float32, "/neck/pitch_deg_state",
                                 lambda m: setattr(self.head, "cur_pitch", m.data), 10)
        self.pub_cmd = self.create_publisher(Twist, "cmd_vel_behavior", 10)
        # 정지 상태(INITIAL·SET·FINISHED·페널티·넘어짐)에서는 0 을 teleop 슬롯(우선순위 100)으로도 보낸다.
        # twist_mux 는 game_stop_lock(50) 이 걸리면 0 을 내는 게 아니라 '출력을 끊어서' 보행 러너가 마지막 명령을
        # 계속 실행한다 (Jazzy 에서 확인: 회전 중 페널티 → 30 s 내내 회전). SET 중 이동 반칙·페널티 중 이동을 막는다.
        self.stop_via_teleop = bool(P("stop_via_teleop_topic", True))
        self.pub_stop = self.create_publisher(Twist, "cmd_vel_teleop", 10)
        self.pub_game_lock = (self.create_publisher(Bool, "/game_stop_lock", 10)
                              if bool(P("publish_game_stop_lock", False)) else None)
        self.pub_yaw = self.create_publisher(Float32, "/neck/yaw_deg", 10)
        self.pub_pitch = self.create_publisher(Float32, "/neck/pitch_deg", 10)
        # 목 속도 (MX-64 Profile Velocity). neck_control 은 첫 목 상태를 낼 때 자기 기본값(실기 10 rpm)을 한 번 써 넣으므로,
        # 그 상태를 받은 직후(또는 목 상태가 2 s 넘게 끊겼다 다시 올 때 = neck_control 재시작)에 덮어쓴다.
        self.neck_speed_rpm = float(P("neck_speed_rpm", 15.0))
        self.pub_neck_speed = self.create_publisher(Float32, "/neck/speed_rpm", 10)
        self.neck_state_t = None
        self.neck_speed_sent = False
        self.pub_kick = self.create_publisher(Bool, "/kick_trigger", 10)
        self.pub_state = self.create_publisher(StrategyState, "/hsl/strategy_state", 10)
        self.pub_dbg = self.create_publisher(String, "/hsl/behavior_debug", 10)
        self.kick_cli = self.create_client(SetParameters, f"/{self.kick_node}/set_parameters")
        self.add_on_set_parameters_callback(self._on_parameters)
        self.create_timer(1.0 / rate, self._cycle)
        self.get_logger().info(f"behavior: player {self.player} (GK={self.gk}), field {self.field.length}x"
                               f"{self.field.width}, kick node /{self.kick_node}")

    def _on_parameters(self, params):
        for p in params:
            if p.name == "assigned_role":
                if str(p.value).lower() not in ("auto", "attacker", "defender", "goalkeeper"):
                    return SetParametersResult(successful=False, reason="Invalid assigned_role")
                if self.kick_phase in ("arming", "kicking"):
                    return SetParametersResult(successful=False, reason="Stop the current kick before changing role")
            if p.name == "default_game_state" and str(p.value).upper() not in STATE_NAMES.values():
                return SetParametersResult(successful=False, reason="Invalid default_game_state")
        for p in params:
            if p.name == "assigned_role":
                self.assigned_role = str(p.value).lower()
                self.active_player, self.my_role = 0, S.ROLE_NONE
                self.play.reset()
                self.walk.reset()
                self.get_logger().info(f"지정 역할 → {self.assigned_role}")
            elif p.name == "default_game_state":
                self.default_state = str(p.value).upper()
                if self.default_state != "PLAYING":
                    self.kick_phase = None
                    self.play.reset()
        return SetParametersResult(successful=True)

    # ── 콜백 ──
    def _on_gc(self, m):
        self.gc, self.gc_rx = m, time.monotonic()

    def _on_obstacles(self, m):
        self.obstacles = [(p.position.x, p.position.y) for p in m.poses]
        self.obstacles_t = time.monotonic()

    def _obstacles_local(self):
        """최근 로봇 검출 (몸통 좌표) [(x, y)]."""
        if time.monotonic() - self.obstacles_t > 0.6:
            return []
        return self.obstacles

    def _on_kick_done(self, m):
        if m.data and self.kick_phase in ("arming", "kicking"):
            self.kick_phase, self.kick_t = "cooldown", time.monotonic()

    # ── 보조 ──
    def _game(self):
        """(state, penalized, we_kick, set_play)."""
        if self.gc is None or time.monotonic() - self.gc_rx > 3.0 or not self.gc.team_found:
            st = {v: k for k, v in STATE_NAMES.items()}.get(self.default_state, 0)
            return st, False, True, 0
        i = self.player - 1
        pen = 0 <= i < len(self.gc.penalties) and self.gc.penalties[i] != 0
        return self.gc.state, pen, self.gc.we_are_kicking, self.gc.set_play

    def _my_pose(self):
        p = self.pose
        return Pose(p.x, p.y, p.theta) if p is not None else Pose(-self.field.hl, self.field.hw, -math.pi / 2)

    def _ball_field(self, now):
        """전략용 공 (필드 좌표, 팀 공). 잠깐 끊겨도 ball_lost_time 동안은 마지막 값을 유지."""
        tb = self.team_ball
        if tb is not None and tb.valid:
            self.team_ball_seen = now
            self.last_team_ball = (tb.field_x, tb.field_y)
            return self.last_team_ball
        if self.last_team_ball is not None and now - self.team_ball_seen < self.ball_lost_time:
            return self.last_team_ball
        return None

    def _ball_local(self, me, ball_field):
        """(bx, by, age, speed) 로봇 기준. 자기 공이 신선하면 그것 (오도메트리 좌표 추적이라 안정적)."""
        b = self.ball
        if b is not None and b.valid and b.time_since_seen < self.own_ball_fresh:
            return b.rel_x, b.rel_y, b.time_since_seen, math.hypot(b.vel_x, b.vel_y)
        if ball_field is None:
            return None
        lx, ly = me.to_local(*ball_field)
        return lx, ly, 1.0, 0.0

    def _on_neck_yaw(self, m):
        self.head.cur_yaw = m.data
        now = time.monotonic()
        if self.neck_state_t is None or now - self.neck_state_t > 2.0:
            self.neck_speed_sent = False
        self.neck_state_t = now
        # 구독자가 아직 연결 안 됐으면 보내도 사라지므로 다음 상태 때 다시 시도
        if self.neck_speed_rpm > 0.0 and not self.neck_speed_sent and self.pub_neck_speed.get_subscription_count() > 0:
            self.pub_neck_speed.publish(Float32(data=self.neck_speed_rpm))
            self.neck_speed_sent = True
            self.get_logger().info(f"목 속도 {self.neck_speed_rpm:.1f} rpm 요청 (/neck/speed_rpm)")

    def _mates(self):
        out = []
        if self.team is None:
            return out
        for m in self.team.teammates:
            out.append(S.Agent(number=m.player_number, pose=Pose(m.x, m.y, m.theta), validity=m.pose_validity,
                               time_to_ball=m.time_to_ball, playing_ball=m.playing_ball, role=m.position_role,
                               available=not (m.penalized or m.fallen)))
        return out

    def _avoid(self, me, mates, ball_field, avoid_ball, near_ball_ok=None):
        """앞쪽 장애물 (팀원 pose, YOLO 로봇 검출, 공 담당이 아니면 공) 을 옆으로 비킨다.
        near_ball_ok = 로봇 기준 공 위치: 공 0.7 m 안의 로봇은 무시 (공 다툼은 허용)."""
        obs = [(*me.to_local(m.pose.x, m.pose.y), self.obstacle_radius) for m in mates]
        obs += [(x, y, self.obstacle_radius) for x, y in self._obstacles_local()]
        if avoid_ball and ball_field is not None:
            obs.append((*me.to_local(*ball_field), 1.0))     # 공 담당이 아니면 공 1 m 안으로 안 들어감
        ax = ay = 0.0
        block = False
        for lx, ly, r in obs:
            if near_ball_ok is not None and math.hypot(lx - near_ball_ok[0], ly - near_ball_ok[1]) < 0.7 and r > 0.6:
                continue
            d = math.hypot(lx, ly)
            if 0.05 < d < r and lx > -0.2:
                k = 1.0 - d / r
                ay += -math.copysign(0.35 * k, ly if abs(ly) > 0.02 else -1.0)
                ax += -0.4 * k * max(0.0, lx / d)
                if d < 0.45 and lx > 0.1 and abs(ly) < 0.35 and r > 0.6:
                    block = True      # 바로 앞 로봇: 전진 금지 (4v4 넘어짐 대부분이 센터 근처 정면 충돌)
        if block:
            ax = -10.0                # WalkToPose 에서 vx 하한(후진 한계)으로 잘린다
        return ax, ay

    def _walk_to(self, me, target, mates, ball_field, avoid_ball=True, final_heading=True):
        lx, ly = me.to_local(target.x, target.y)
        dth = norm_angle(target.th - me.th)
        return self.walk(lx, ly, dth, final_heading=final_heading,
                         avoid=self._avoid(me, mates, ball_field, avoid_ball))

    def _head_ball(self, now, me, ball_field, allow_glance):
        bl = self._ball_local(me, ball_field)
        if bl is None:
            return self.head.scan(now)
        self.head.stop_scan()
        pb = (bl[0], bl[1], 0.065)
        b = self.ball
        own = b is not None and b.valid and b.cam_valid and b.time_since_seen < self.own_ball_fresh
        pc = (b.cam_x, b.cam_y, b.cam_z) if own else None
        (yaw, pitch), _ = self.head.look_active(now, pc, pb, allow_glance)
        return yaw, pitch

    # ── 킥 ──
    def _start_kick(self, bx, by, ka, now):
        if not self.kick_cli.service_is_ready():
            self.get_logger().warn("킥 노드 파라미터 서비스 없음 — 킥 생략", throttle_duration_sec=5.0)
            return
        d = math.hypot(bx, by)
        if d > 0.48:      # 학습 분포 경계 안으로
            bx, by = bx * 0.48 / d, by * 0.48 / d
        req = SetParameters.Request()
        for name, val in (("ball_pos_b_x", bx), ("ball_pos_b_y", by), ("kick_target_yaw_offset_rad", ka)):
            req.parameters.append(Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                                                             double_value=float(val))))
        self.kick_phase, self.kick_t, self.kick_info = "arming", now, (bx, by, ka)
        fut = self.kick_cli.call_async(req)
        fut.add_done_callback(self._kick_params_set)

    def _kick_params_set(self, fut):
        if self.kick_phase != "arming":
            return
        try:
            res = fut.result()
            ok = all(r.successful for r in res.results)
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            self.get_logger().warn("킥 파라미터 설정 실패")
            self.kick_phase, self.kick_t = "cooldown", time.monotonic()
            return
        self.pub_kick.publish(Bool(data=True))
        self.kick_phase, self.kick_t = "kicking", time.monotonic()
        self.kicks += 1
        bx, by, ka = self.kick_info
        self.get_logger().info(f"킥 #{self.kicks}: 공 ({bx:.2f}, {by:+.2f}) 목표 {math.degrees(ka):+.0f}°")

    # ── 메인 ──
    def _cycle(self):
        now = time.monotonic()
        dt = clamp(now - self.last_t, 0.0, 0.2)
        self.last_t = now
        state, penalized, we_kick, set_play = self._game()
        me_pose = self._my_pose()
        # Clear display telemetry only; preserve all arrival and motion state.
        for walker in (self.walk, self.play.walk, self.search.walk):
            walker.target_local = None
        mates = self._mates()
        ball_field = self._ball_field(now)
        fallen = self.robot_state != "NORMAL"

        if state != self.prev_state:
            self.get_logger().info(f"경기 상태 → {STATE_NAMES.get(state, state)}")
            if state == 3:
                self.playing_since = now
                self.kickoff_ball = ball_field
            self.walk.reset()
            self.play.reset()
            self.search.reset(now)
            self.prev_state = state

        # 내 공 도달 시간
        own_age = self.ball.time_since_seen if (self.ball is not None and self.ball.valid) else 5.0
        if ball_field is not None and not penalized and not fallen:
            kt = S.kick_target_for(ball_field, self.field)
            validity = self.pose.validity if self.pose is not None else 0.0
            my_ttb = S.time_to_ball(me_pose, ball_field, kt, v_walk=self.ttb_speed, validity=validity,
                                    ball_age=min(own_age, 3.0))
        else:
            my_ttb = 1e3
        me = S.Agent(number=self.player, pose=me_pose, validity=self.pose.validity if self.pose else 0.0,
                     time_to_ball=my_ttb, playing_ball=(self.active_player == self.player),
                     role=self.my_role, available=not (penalized or fallen))
        self.roles = {m.number: m.role for m in mates}
        self.roles[self.player] = self.my_role

        cmd = (0.0, 0.0, 0.0)
        head = None
        tactic = ""
        self.skill = "stand"

        if self.kick_phase in ("arming", "kicking"):
            if now - self.kick_t > self.kick_timeout:
                self.get_logger().warn("킥 완료 신호 시간 초과")
                self.kick_phase, self.kick_t = "cooldown", now
        if self.kick_phase == "cooldown" and now - self.kick_t > self.kick_cooldown:
            self.kick_phase = None
            self.play.reset()

        if state != 3:
            self.active_player = 0      # 공 담당은 PLAYING 에서만 (팀 메시지의 playing_ball 도 같이 꺼진다)
        if fallen or penalized or state in (0, 4):
            self.active_player = 0
            self.skill = "fallen" if fallen else "penalized" if penalized else "stand"
            head = self.head.neutral() if state == 0 else None
            self.walk.reset()
        elif state == 1:   # READY
            d = S.decide_ready(me, mates, self.field, we_kick, self.roles, self.gk)
            self.my_role, tactic = d.role, d.tactic
            cmd = self._walk_to(me_pose, d.target, mates, None, avoid_ball=False)
            self.skill = "walk_to_kickoff" + ("(도착)" if self.walk.arrived else "")
            self.dbg_target = d.target
        elif state == 2:   # SET
            self.skill = "set"
            self.walk.reset()
        elif state == 3:   # PLAYING
            cmd, tactic = self._playing(now, dt, me, me_pose, mates, ball_field, we_kick, set_play)

        if self.kick_phase in ("arming", "kicking"):
            cmd = (0.0, 0.0, 0.0)
        cmd = self.shaper(cmd, dt) if state in (1, 3) and not (fallen or penalized) else self.shaper((0, 0, 0), dt)

        if head is None:
            near = False
            if ball_field is not None:
                near = me_pose.dist(*ball_field) < 1.3 and self.active_player == self.player
            head = self._head_ball(now, me_pose, ball_field, allow_glance=not near and self.kick_phase is None)
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(cmd[0]), float(cmd[1]), float(cmd[2])
        self.pub_cmd.publish(t)
        must_stop = fallen or penalized or state in (0, 2, 4)
        if self.pub_game_lock is not None:
            self.pub_game_lock.publish(Bool(data=must_stop))
        if self.stop_via_teleop and must_stop and not fallen:
            self.pub_stop.publish(Twist())
        self.pub_yaw.publish(Float32(data=float(head[0])))
        self.pub_pitch.publish(Float32(data=float(head[1])))

        st = StrategyState()
        st.header.stamp = self.get_clock().now().to_msg()
        st.game_state = int(state)
        st.tactic = tactic
        st.set_play = str(set_play)
        st.position_role = int(self.my_role)
        st.active = self.active_player == self.player
        st.active_player = int(self.active_player)
        st.my_time_to_ball = float(min(my_ttb, 1e3))
        st.skill = self.skill
        self.pub_state.publish(st)
        if now - self.dbg_t > 0.2:
            self.dbg_t = now
            target = None
            for walker in (self.walk, self.play.walk, self.search.walk):
                if walker.target_local is not None:
                    lx, ly, heading = walker.target_local
                    tx, ty = me_pose.to_field(lx, ly)
                    target = [round(tx, 3), round(ty, 3),
                              round(math.degrees(norm_angle(me_pose.th + heading)), 1)]
            self.pub_dbg.publish(String(data=json.dumps(dict(
                player=self.player, state=STATE_NAMES.get(state, state),
                role=(S.ROLE_NAMES.get(self.my_role) if self.assigned_role == "auto" else self.assigned_role),
                assigned_role=self.assigned_role,
                target=target, target_kind="walk" if target is not None else None,
                active=self.active_player, skill=self.skill, kick=self.kick_phase, kicks=self.kicks,
                pose=[round(me_pose.x, 2), round(me_pose.y, 2), round(math.degrees(me_pose.th))],
                ball=None if ball_field is None else [round(ball_field[0], 2), round(ball_field[1], 2)],
                ttb=round(min(my_ttb, 999.0), 1), cmd=[round(c, 2) for c in cmd]))))

    def _playing(self, now, dt, me, me_pose, mates, ball_field, we_kick, set_play):
        f = self.field
        if ball_field is None:
            # 아무도 공을 모름 → 탐색 (B-Human SearchForBall: 돌기 → 흩어진 탐색 자리로)
            self.active_player = 0
            role_gk = self.player if self.assigned_role == "goalkeeper" else self.gk
            self.my_role = S.ROLE_GOALKEEPER if self.player == role_gk else self.my_role
            self.my_role = {"attacker": S.ROLE_FORWARD, "defender": S.ROLE_DEFENDER,
                            "goalkeeper": S.ROLE_GOALKEEPER}.get(self.assigned_role, self.my_role)
            sp = (S.defender_pose((0.0, 0.0), f) if self.assigned_role == "defender"
                  else S.search_pose(self.player, f, role_gk))
            sp.x, sp.y = f.clip_inside(sp.x, sp.y, 0.3)
            lx, ly = me_pose.to_local(sp.x, sp.y)
            self.skill = f"search_{self.search.phase}"
            turn_dir = 1.0 if self.player % 2 else -1.0
            return self.search(now, (lx, ly), dt, turn_dir), "search"
        self.search.reset(now)

        # 공이 아직 인플레이가 아님: 상대 킥오프 (10 s 또는 공 움직임) / 상대 세트플레이
        ball_moved = (self.kickoff_ball is not None and
                      math.hypot(ball_field[0] - self.kickoff_ball[0], ball_field[1] - self.kickoff_ball[1]) > 0.3)
        opp_kickoff = (not we_kick and self.playing_since is not None and
                       now - self.playing_since < self.opp_kickoff_wait and not ball_moved and set_play == 0)
        opp_set_play = set_play != 0 and not we_kick
        d = S.decide_assigned_role(self.assigned_role, me, ball_field, f)
        if d is None:
            d = S.decide_playing(me, mates, ball_field, f, self.active_player, self.roles, self.gk)
        if d.active_player != self.active_player:
            self.get_logger().info(f"공 담당: {self.active_player} → {d.active_player}")
            self.play.reset()
        self.active_player = d.active_player
        self.my_role = d.role
        if opp_kickoff or opp_set_play:
            keep = f.center_circle_radius + 0.3 if opp_kickoff else self.set_play_distance
            target = d.target if not d.active else S.supporter_pose(ball_field, f)
            if d.active:   # 공 담당도 공에서 거리 유지, 공 쪽을 보고 대기
                bx, by = ball_field
                ang = math.atan2(me_pose.y - by, me_pose.x - bx)
                target = Pose(bx + (keep + 0.2) * math.cos(ang), by + (keep + 0.2) * math.sin(ang),
                              norm_angle(ang + math.pi))
            dx, dy = target.x - ball_field[0], target.y - ball_field[1]
            dd = math.hypot(dx, dy)
            if dd < keep:
                s = keep / max(dd, 1e-3)
                target = Pose(ball_field[0] + dx * s, ball_field[1] + dy * s, target.th)
            # 거리 두는 자리가 필드 밖이면 (터치라인 킥인 때 로봇이 공보다 바깥쪽에 있으면 생긴다 →
            # 장외 이탈 페널티 2 회, 2026-10-31) 공에서 우리 골대 쪽으로 keep 떨어진 자리로 바꾼다
            if abs(target.x) > f.hl - 0.3 or abs(target.y) > f.hw - 0.3:
                gx, gy = -f.hl - ball_field[0], -ball_field[1]
                gn = max(math.hypot(gx, gy), 1e-3)
                tx, ty = f.clip_inside(ball_field[0] + keep * gx / gn, ball_field[1] + keep * gy / gn, 0.3)
                target = Pose(tx, ty, math.atan2(ball_field[1] - ty, ball_field[0] - tx))
            self.skill = "wait_ball_in_play"
            return self._walk_to(me_pose, target, mates, ball_field), "wait"

        if d.active:
            if self.kick_phase is not None:
                self.skill = f"kick_{self.kick_phase}"
                return (0.0, 0.0, 0.0), "play_ball"
            bl = self._ball_local(me_pose, ball_field)
            kt = S.kick_target_for(ball_field, f)
            ka = norm_angle(math.atan2(kt[1] - ball_field[1], kt[0] - ball_field[0]) - me_pose.th)
            bx, by, age, speed = bl
            mode, ka_use = "none", ka
            if self.duel_enable:
                mates_l = [me_pose.to_local(m.pose.x, m.pose.y) for m in mates]
                opps = duel.opponents_from_obstacles(self._obstacles_local(), mates_l)
                margin = -0.10 if self.duel_mode == "block" else 0.10      # 막기 ↔ 차기 진동 방지
                mode, ka2, _opp = duel.duel_decision((bx, by), opps, ka, self.duel_contest_radius, margin,
                                                     self.duel_sep_min, self.duel_max_dev)
                if mode != self.duel_mode:
                    self.get_logger().info(f"공 다툼: {self.duel_mode} → {mode}")
                self.duel_mode = mode
                if mode == "block":
                    x, y, th = duel.block_pose(ball_field, (-f.hl, 0.0))
                    self.play.reset()
                    self.skill = "duel_block"
                    return self._walk_to(me_pose, Pose(x, y, th), mates, None, avoid_ball=False), "duel"
                if mode == "deflect":
                    ka_use = ka2
            far = math.hypot(bx, by) > 1.0
            res, val = self.play(bx, by, ka_use, age, speed, now,
                                 avoid=self._avoid(me_pose, mates, None, False, near_ball_ok=(bx, by))
                                 if far else (0.0, 0.0))
            self.skill = f"play_ball_{self.play.phase}" + ("_deflect" if ka_use != ka else "")
            if res == "kick" and mode == "hold":
                self.skill = "duel_hold"           # 상대가 몸 바로 앞 — 킥 정책이 걸어 들어가면 부딪힌다
                return (0.0, 0.0, 0.0), "duel"
            if res == "kick":
                if self.enable_kick:
                    self._start_kick(*val, now)
                    self.skill = "kick_arming"
                return (0.0, 0.0, 0.0), "play_ball"
            return val, "play_ball"
        self.play.reset()
        self.skill = f"position_{S.ROLE_NAMES.get(d.role)}" + ("(도착)" if self.walk.arrived else "")
        return self._walk_to(me_pose, d.target, mates, ball_field), "dynamic"


def main():
    rclpy.init()
    node = Behavior()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

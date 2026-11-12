# robot_init_behavior.py

import math

def _raise_goal_pwm(hw):
    """Goal PWM 상한을 100% 로 올린다 (구버전 컨트롤러 호환).

    robot_init 은 팔을 arm_init_q(팔꿈치 ±3.04rad)까지 '이동' 시켜야 하고,
    도달 판정(tol/settle)이 팔도 검사한다. 상한이 낮으면 팔이 목표에
    도달하지 못해 init 이 timeout 으로 끝난다.
    (실측: 1.0rad 이동에 최소 150 필요, 유지는 30이면 충분)
    """
    fn = getattr(hw, "set_goal_pwm_full", None)
    if fn is not None:
        fn()


def _restore_goal_pwm(hw):
    """Goal PWM 상한을 게인 프로파일의 운용값으로 되돌린다."""
    fn = getattr(hw, "restore_goal_pwm", None)
    if fn is not None:
        fn()


class RobotInitBehavior:
    """
    joint_bridge가 가진 hw API를 이용해서:
      - Torque ON
      - Profile Velocity = init_rpm
      - 타겟 자세로 이동
          * start()      : legs -> leg_init_q, arms -> arm_init_q (정책 기본 자세)
          * start_zero() : 모든 조인트 -> 0 rad (정비/안전 자세)
      - 도착 판정 (tol + settle)
      - Profile Velocity = final_rpm
      - 종료
    """

    def __init__(self, hw):
        self.hw = hw
        self._active = False
        self._done = False

        self.targets = {}          # {joint_name: rad}
        self.tol = 0.03
        self.settle_need = 10
        self.settle_now = 0
        self.timeout_sec = 20.0
        self.t_start = 0.0
        self.final_rpm = 45.0

        self.leg_init_q = {
                    "r_pelvis_pitch_joint": 0.62,
                    "r_pelvis_roll_joint": 0.0,
                    "r_pelvis_yaw_joint": 0.0,
                    "r_knee_pitch_joint": 1.2,
                    "r_ankle_pitch_joint": 0.62,
                    "r_ankle_roll_joint": 0.0,
        
                    "l_pelvis_pitch_joint": -0.62,
                    "l_pelvis_roll_joint": 0.0,
                    "l_pelvis_yaw_joint": 0.0,
                    "l_knee_pitch_joint": -1.2,
                    "l_ankle_pitch_joint": -0.62,
                    "l_ankle_roll_joint": 0.0,
        }


        # self.leg_init_q = {
        #     "r_pelvis_pitch_joint": 0.6,
        #     "r_pelvis_roll_joint": 0.0,
        #     "r_pelvis_yaw_joint": 0.0,
        #     "r_knee_pitch_joint": 1.2,
        #     "r_ankle_pitch_joint": 0.6,
        #     "r_ankle_roll_joint": 0.0,

        #     "l_pelvis_pitch_joint": -0.6,
        #     "l_pelvis_roll_joint": 0.0,
        #     "l_pelvis_yaw_joint": 0.0,
        #     "l_knee_pitch_joint": -1.2,
        #     "l_ankle_pitch_joint": -0.6,
        #     "l_ankle_roll_joint": 0.0,
        # }
        
        self.arm_init_q = {
            "r_shoulder_pitch_joint": -0.3,
            "r_shoulder_roll_joint": 0.0,
            "r_elbow_pitch_joint": 1.00,
            "l_shoulder_pitch_joint": 0.3,
            "l_shoulder_roll_joint": 0.0,
            "l_elbow_pitch_joint": -1.00,
        }

        # ── 목(Neck) 초기 자세 ─────────────────────────────────
        # (2026-08) 목 모터가 AX-18A(MCU 버스) → MX-64(U2D2 버스)로 이관되어,
        # 이제 목도 self.hw.send_goals() 로 직접 갈 수 있다. 그래서 예전처럼
        # deg 를 보관했다가 BodyMotorController 가 /neck/* 토픽으로 따로
        # 발행하던 우회 경로가 사라졌고, 아래 값은 다리/팔과 똑같이
        # "URDF 관절 rad" 로 적어서 targets 에 그대로 합쳐 넣는다.
        #
        # ★ 단위가 deg → rad 로 바뀌었으니 값을 고칠 때 주의 ★
        #
        # 부호 규약 (URDF 관절 기준. neck_control_node 의 pitch_sign=-1 반영):
        #   neck_yaw_joint   : + = 왼쪽 (REP-103 yaw, 반시계방향이 양수)
        #   neck_pitch_joint : + = 아래를 봄, - = 위를 봄
        #                      (API 의 pitch_deg 와 부호가 반대다.
        #                       pitch_deg -25° = 아래로 25° = rad +0.436)
        #
        # init 자세: 아래로 25° 응시.
        # (RoboCup 경기에서 발 앞 공과 필드 라인이 화면에 같이 들어오는 각도)
        NECK_INIT_PITCH_DEG = -25.0        # API 규약(+ = 위) 기준 값
        self.neck_init_q = {
            "neck_yaw_joint":   0.0,                                  # 정면
            "neck_pitch_joint": -math.radians(NECK_INIT_PITCH_DEG),   # = +0.4363 rad
        }

        # motor_zero(start_zero) 모드에서 사용할 목 자세.
        # start_zero 는 "전 조인트 0 rad" 이므로 목도 0 이면 되지만,
        # 의도를 분리해 두면 향후 다른 캘리브레이션 자세로 분기시키기 쉽다.
        self.neck_zero_q = {
            "neck_yaw_joint":   0.0,
            "neck_pitch_joint": 0.0,
        }

        # ── 목(Neck) 초기화 속도 ───────────────────────────────
        # 목이 너무 빨리 돌면:
        #   1) 카메라(ZED 2i)가 머리에 마운트돼 있어 영상이 흔들리고,
        #   2) 갑작스런 가속으로 과전류 / 위치 오버슛 위험.
        # 따라서 본체 init_rpm(20) / final_rpm(45) 과 별개로 느린 값을 쓴다.
        #
        # 이제 목도 MX-64(Protocol 2.0)라 몸통과 같은 Profile Velocity
        # 레지스터(addr 112, 0.229 rpm/LSB)를 쓴다. 예전 AX MOVING_SPEED
        # raw(0.111 rpm/unit) 환산은 더 이상 필요 없다.
        #
        # ★ 0 은 Dynamixel 에서 "무제한(=Velocity Limit 최대속도)" 이라
        #   절대 쓰면 안 된다. 목이 최대 속도로 튄다.
        #
        # servo_map.yaml 의 limits.profile_velocity_exclude 에 목이 들어 있어
        # set_profile_velocity_all() 이 목을 건너뛴다. 그래서 이 값은
        # set_profile_velocity_joints() 로 목에만 개별 기입되고, 이후
        # robot_init 이 끝나면서 몸통이 final_rpm(45)으로 올라가도 목은
        # 계속 이 느린 속도를 유지한다.
        self.neck_init_rpm = 10.0    # rpm ≈ 60 °/s

        # ── 낙상 보호 자세 (protective pose) ───────────────────
        # 로봇이 넘어지는 것을 감지했을 때(기울기 45도 초과) 즉시
        # 이 자세로 전환하여 충격을 완화하고, standup 시퀀스가
        # 예측 가능한 자세에서 시작될 수 있도록 한다.
        #
        # 설계 의도:
        #   - 다리: 무릎을 적당히 굽혀 웅크린 자세.
        #           완전히 편 상태로 넘어지면 무릎/발목이 먼저
        #           충격을 받아 파손 위험이 크다. 굽혀서 충격을
        #           여러 관절에 분산시킨다.
        #   - 팔  : 팔꿈치를 굽혀 몸 앞쪽으로 모아 머리/몸통을
        #           보호하고, 바닥에 닿을 때 완충 역할을 한다.
        #
        # 주의: 아래 값은 KHUBER 의 기구 형상에 따라 자기간섭
        #       (self-collision)이 없는지 실제 로봇에서 반드시
        #       확인한 뒤 조정해야 한다. 우선은 standup 의 첫
        #       단계(crouch 류)와 유사한 안전 범위로 잡았다.
        #       값 부호 규약은 leg_init_q / arm_init_q 와 동일하다.
        self.protective_q = {
            # 다리: 가볍게 웅크림 (무릎 굽힘)
            "r_pelvis_pitch_joint": 0.6,
            "r_pelvis_roll_joint": 0.0,
            "r_pelvis_yaw_joint": 0.0,
            "r_knee_pitch_joint": 1.2,
            "r_ankle_pitch_joint": 0.6,
            "r_ankle_roll_joint": 0.0,

            "l_pelvis_pitch_joint": -0.6,
            "l_pelvis_roll_joint": 0.0,
            "l_pelvis_yaw_joint": 0.0,
            "l_knee_pitch_joint": -1.2,
            "l_ankle_pitch_joint": -0.6,
            "l_ankle_roll_joint": 0.0,

            # 팔: 팔꿈치를 굽혀 몸 앞으로 모아 머리/몸통 보호
            
            "r_shoulder_pitch_joint": 0.2,
            "r_shoulder_roll_joint": -0.3,
            "r_elbow_pitch_joint": 3.04,
            "l_shoulder_pitch_joint": -0.2,
            "l_shoulder_roll_joint": 0.3,
            "l_elbow_pitch_joint": -3.04,
        }

    
    def is_active(self) -> bool:
        return bool(self._active)

    def _apply_neck_profile_velocity(self, neck_targets: dict):
        """목 관절에만 느린 Profile Velocity(neck_init_rpm)를 기입한다.

        왜 따로 부르나:
          set_profile_velocity_all() 은 servo_map 의 profile_velocity_exclude
          때문에 목을 건너뛴다. 목 속도를 아무도 안 정해주면 모터에 남아 있는
          직전 값(전원 투입 직후엔 EEPROM 기본값 = 사실상 최대속도)으로
          움직여 카메라가 크게 흔들린다.

        순서 주의:
          반드시 send_goals() 보다 '먼저' 불러야 한다. 다이나믹셀은
          Goal Position 을 쓰는 그 순간의 Profile Velocity 로 이동을 시작한다.
          (이 메서드는 즉시 SyncWrite 하는 동기 호출이라, 이어서 send_goals 를
           불러도 순서가 뒤집히지 않는다)

        구버전 노드 호환:
          hw 에 set_profile_velocity_joints 가 없으면(구버전 컨트롤러) 조용히
          건너뛴다. 목 속도만 기본값이 될 뿐 init 자체는 정상 동작한다.
        """
        if not neck_targets:
            return
        fn = getattr(self.hw, "set_profile_velocity_joints", None)
        if fn is None:
            self.hw.log_warn(
                "hw.set_profile_velocity_joints 없음 → 목 속도 기본값 사용 "
                "(body_motor_controller 를 최신본으로 빌드했는지 확인)")
            return
        # 0 은 '무제한(최대속도)' 특수값이라 절대 내려가면 안 된다.
        rpm = max(0.1, float(self.neck_init_rpm))
        fn({name: rpm for name in neck_targets})

    def start(self, init_rpm=100.0, final_rpm=200.0, tol=0.03, settle_count=10, timeout_sec=20.0):
        if self._active:
            return False, "robot_init already running"
        self._active = True
        self._done = False

        # 1) torque ON
        self.hw.set_torque_all(True)
        # ★ 팔을 arm_init_q 까지 '이동' 시켜야 하므로 Goal PWM 상한을 100% 로.
        _raise_goal_pwm(self.hw)

        # 2) set profile velocity (init)
        #    목은 servo_map 의 profile_velocity_exclude 때문에 여기서 빠지고,
        #    아래 3) 에서 개별 속도로 따로 기입된다.
        self.hw.set_profile_velocity_all(init_rpm)

        # 3) build targets by joint name
        targets = {}

        # legs: policy 기본 자세 사용
        for name, rad in self.leg_init_q.items():
            if self.hw.has_joint(name):
                targets[name] = float(rad)
        # arms: name 기반
        for name, rad in self.arm_init_q.items():
            if self.hw.has_joint(name):
                targets[name] = float(rad)
        # neck: 몸통과 같은 send_goals 경로를 타되, 속도만 따로 지정한다.
        #   ★ 목이 targets 에 들어가면 도착 판정(tol/settle)도 목을 기다린다.
        #     목 모터가 죽어 있거나 기구에 걸리면 init 이 timeout 으로 끝난다
        #     (timeout 경로도 final_rpm 복원은 하므로 보행은 가능).
        #     목을 판정에서 빼고 싶으면 아래 블록만 지우고 send_goals 를
        #     따로 한 번 더 부르면 된다.
        neck_targets = {}
        for name, rad in self.neck_init_q.items():
            if self.hw.has_joint(name):
                targets[name] = float(rad)
                neck_targets[name] = float(rad)
        self._apply_neck_profile_velocity(neck_targets)

        if not targets:
            return False, "no targets built (check joint_map has ids 1~20)"

        self.targets = targets
        self.tol = float(tol)
        self.settle_need = int(settle_count)
        self.settle_now = 0
        self.timeout_sec = float(timeout_sec)
        self.final_rpm = float(final_rpm)
        self.t_start = self.hw.now_sec()

        # 4) send goal
        self.hw.send_goals(self.targets)

        self._active = True
        return True, "robot_init started"

    def start_zero(self, init_rpm=20.0, final_rpm=0.0, tol=0.03, settle_count=10, timeout_sec=20.0):
        """
        모든 등록된 조인트를 0 rad 위치로 보내는 모드.
        정비 / 캘리브레이션 / 안전 자세 복귀 용도.
        """
        if self._active:
            return False, "robot_init already running"
        self._active = True
        self._done = False

        # 1) torque ON
        self.hw.set_torque_all(True)
        # ★ 팔을 arm_init_q 까지 '이동' 시켜야 하므로 Goal PWM 상한을 100% 로.
        _raise_goal_pwm(self.hw)

        # 2) set profile velocity (init)
        self.hw.set_profile_velocity_all(init_rpm)

        # 3) build targets: legs + arms + neck 전 조인트를 0 rad로
        targets = {}
        for name in self.leg_init_q.keys():
            if self.hw.has_joint(name):
                targets[name] = 0.0
        for name in self.arm_init_q.keys():
            if self.hw.has_joint(name):
                targets[name] = 0.0
        # 목도 "전 조인트 0 rad" 에 포함한다 (= 정면 수평 응시).
        neck_targets = {}
        for name, rad in self.neck_zero_q.items():
            if self.hw.has_joint(name):
                targets[name] = float(rad)
                neck_targets[name] = float(rad)
        self._apply_neck_profile_velocity(neck_targets)

        if not targets:
            self._active = False
            return False, "no targets built (check joint_map has ids 1~20)"

        self.targets = targets
        self.tol = float(tol)
        self.settle_need = int(settle_count)
        self.settle_now = 0
        self.timeout_sec = float(timeout_sec)
        self.final_rpm = float(final_rpm)
        self.t_start = self.hw.now_sec()

        # 4) send goal
        self.hw.send_goals(self.targets)

        self.hw.log_info(f"motor_zero started: {len(targets)} joints -> 0 rad")
        return True, "motor_zero started"

    def tick(self, last_js):
        """
        joint_bridge의 read_and_publish()에서 매 tick마다 호출.
        last_js: sensor_msgs/JointState (최근 publish한 값)
        """
        if not self._active:
            return

        now = self.hw.now_sec()
        if now - self.t_start > self.timeout_sec:
            # [중요] timeout 으로 끝나도 반드시 final_rpm 으로 복원한다.
            #   - 예전엔 여기서 final_rpm 을 안 써서, 자세에 못 닿아 timeout 나면
            #     Profile Velocity 가 init_rpm(예: start_zero 의 20rpm)에 그대로 남았다.
            #   - RL 러너는 PV 를 안 건드리므로(=이 값을 그대로 물려받음),
            #     그 상태에서 킥하면 다리 속도가 20rpm 으로 캡되어 "약한 킥"이 되었다.
            #   - settle 완료 경로(아래)와 동일하게 final_rpm 을 써서,
            #     robot_init 이 '어떤 경로로 끝나든' 항상 정해진 속도로 마무리되게 한다.
            #     (버튼을 여러 번 눌러도 마지막에 항상 final_rpm 으로 수렴)
            self.hw.set_profile_velocity_all(self.final_rpm)
            _restore_goal_pwm(self.hw)
            self.hw.log_warn(
                f"robot_init timeout → Profile Velocity 를 final_rpm={self.final_rpm} 로 복원")
            self._active = False
            return

        if last_js is None or len(last_js.name) == 0:
            self.settle_now = 0
            return

        cur = {n: p for n, p in zip(last_js.name, last_js.position)}

        all_ok = True
        for n, trg in self.targets.items():
            if n not in cur:
                all_ok = False
                break
            if abs(cur[n] - trg) > self.tol:
                all_ok = False
                break

        if all_ok:
            self.settle_now += 1
        else:
            self.settle_now = 0

        if self.settle_now >= self.settle_need:
            self.hw.log_info(f"robot_init arrived: tol={self.tol:.3f}, settle={self.settle_need}")
            self.hw.set_profile_velocity_all(self.final_rpm)
            _restore_goal_pwm(self.hw)
            self._active = False
"""시뮬 전용: 정책 러너(actor_onnx_*)의 onnxruntime 스레드를 제한한다.

러너 코드는 SessionOptions 없이 InferenceSession 을 만든다 → ORT 기본값은 물리 코어 수만큼
intra-op 스레드 + 바쁜 대기(spinning). 작은 MLP 하나에 2 코어 이상을 태우고, 다른 노드와 CPU 를
다투다 /joint_states 를 놓친다 (실측: /joint_goals 43~46 Hz, 러너 CPU 209 %).
로봇 소스는 건드리지 않고, run_stack.sh·run_robot.sh 가 PYTHONPATH 에 이 폴더를 넣을 때만 적용된다.
  SIM_ORT_THREADS=1 (기본, run_robot.sh) / 비우면 패치 안 함
"""
import os
import sys

_n = os.environ.get("SIM_ORT_THREADS", "")
if _n and any("actor_onnx" in a for a in sys.argv[:1]):
    try:
        import onnxruntime as _ort

        _Orig = _ort.InferenceSession

        class _LimitedSession(_Orig):
            def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kw):
                if sess_options is None:
                    sess_options = _ort.SessionOptions()
                    sess_options.intra_op_num_threads = int(_n)
                    sess_options.inter_op_num_threads = 1
                    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                super().__init__(path_or_bytes, sess_options, providers, provider_options, **kw)

        _ort.InferenceSession = _LimitedSession
    except Exception as e:  # 패치 실패해도 러너는 원래대로 돈다
        print(f"[sim sitecustomize] ORT 패치 실패: {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# 시뮬 전용 시험 수정: visual servo GETBEHIND 교착 (SIM_SERVO_FIX=1 일 때만)
#
# perpendicular 모드에서 _delta_target 은 ±perp(80°)만 목표로 본다.
#   - ALIGN→GETBEHIND: |delta − 목표| > 60° (공 바로 뒤 delta≈0 이면 80° 라 진입)
#   - GETBEHIND→ALIGN: |delta − 목표| ≤ 45° (delta ≥ 35° 필요)
#   - 그런데 GETBEHIND(direct)는 로봇을 골대와 일직선 뒤(delta=0)로 몰아간다 → 영원히 못 나온다.
# 수정: 목표 후보에 0(일직선 뒤)을 추가. 공 뒤에서는 바로 ALIGN 에 들어가 ALIGN 이 공 주위를 돌아
#   80° 자세를 만들고, GETBEHIND 는 공 반대편(골대가 등 뒤)일 때만 쓰인다.
# 원본 visual_servo_kick_node.py 는 그대로다. 효과 확인 후 원본 반영은 사용자 승인으로.
# ─────────────────────────────────────────────────────────────────────────────
if os.environ.get("SIM_SERVO_FIX") == "1" and any("visual_servo_kick_node" in a for a in sys.argv[:1]):
    try:
        import humanoid_localization.visual_servo_kick_node as _vs
        from humanoid_localization.visual_servo_kick_node import wrap_angle as _wrap

        def _delta_target_fixed(self, delta):
            if self._kick_mode != "perpendicular":
                return 0.0
            cands = (0.0, self._perp_target, -self._perp_target)
            return min(cands, key=lambda t: abs(_wrap(delta - t)))

        _vs.VisualServoKickNode._delta_target = _delta_target_fixed
        print("[sim sitecustomize] SIM_SERVO_FIX: _delta_target 후보에 0 추가", file=sys.stderr)
    except Exception as e:
        print(f"[sim sitecustomize] servo 패치 실패: {e}", file=sys.stderr)

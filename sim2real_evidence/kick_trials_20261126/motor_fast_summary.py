# 로봇에서 실행 (읽기 전용): bag 의 /motor_fast 관절별 전류·PWM 와 목표-실제 각도 차. python3 motor_fast_summary.py <bag> [t0 t1]
import sys, math, json
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
ty = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
d = {k: [] for k in ty}
while r.has_next():
    n, raw, t = r.read_next(); d[n].append((t, deserialize_message(raw, ty[n])))
T0 = d["/motor_fast"][0][0]
a, b = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.0, 1e9)
sel = lambda top: [(t, m) for t, m in d[top] if a <= (t - T0) / 1e9 <= b and m.name]
mf, js, jg = sel("/motor_fast"), sel("/joint_states"), sel("/joint_goals")
names = list(mf[0][1].name)
cur = np.array([m.effort for _, m in mf]); pwm = np.array([m.position for _, m in mf])
pos = {n: np.array([dict(zip(m.name, m.position))[n] for _, m in js]) for n in names}
gl = {}
for _, m in jg:
    for n, p in zip(m.name, m.position): gl.setdefault(n, []).append(p)
print("samples motor_fast %d (%.1f Hz), joint_states %d" % (len(mf), (len(mf) - 1) / max(1e-9, (mf[-1][0] - mf[0][0]) / 1e9), len(js)))
for i, n in enumerate(names):
    if "shoulder" in n or "elbow" in n or "neck" in n: continue
    err = (np.mean(gl[n]) - pos[n].mean()) if n in gl else float("nan")
    print("%-22s I mean %+5.2f A (|max| %4.2f) | PWM mean %+5.1f %% (|max| %5.1f) | goal-act %+5.1f deg"
          % (n.replace("_joint", ""), cur[:, i].mean(), np.abs(cur[:, i]).max(), 100 * pwm[:, i].mean(), 100 * np.abs(pwm[:, i]).max(), math.degrees(err)))
dg = [json.loads(m.data) for _, m in sel("/motor_diagnostics")]
if dg:
    v = [x["voltage_v"] for g in dg for x in g["motors"] if x["id"] <= 12]
    print("voltage legs %.1f-%.1f V" % (min(v), max(v)))

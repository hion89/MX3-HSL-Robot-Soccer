# 로봇에서 실행: 킥 시험 bag 요약 (읽기 전용). python3 kick_bag_summary.py <bag_dir>
import math, sys, json
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
d = {k: [] for k in types}
while r.has_next():
    n, raw, t = r.read_next(); d[n].append((t, deserialize_message(raw, types[n])))
t0 = d["/kick_trigger"][0][0]
s = lambda t: (t - t0) / 1e9
def rpy(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))
    return roll, pitch
imu = [(s(t), *rpy(m.orientation)) for t, m in d["/imu/data"]]
base = [x for x in imu if -1.0 < x[0] < 0]
r0 = sum(x[1] for x in base) / len(base); p0 = sum(x[2] for x in base) / len(base)
win = [x for x in imu if 0 <= x[0] <= 6]
rm = max(win, key=lambda x: abs(x[1] - r0)); pm = max(win, key=lambda x: abs(x[2] - p0))
print(f"standing roll {r0:+.1f} pitch {p0:+.1f}")
print(f"max |roll dev| {rm[1]-r0:+.1f} deg at {rm[0]:.2f}s, max |pitch dev| {pm[2]-p0:+.1f} deg at {pm[0]:.2f}s")
for a, b in ((0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 6)):
    w = [x for x in imu if a <= x[0] < b]
    if w: print(f"  {a:.1f}-{b:.1f}s roll {max(x[1]-r0 for x in w):+6.1f}/{min(x[1]-r0 for x in w):+6.1f} pitch {max(x[2]-p0 for x in w):+6.1f}/{min(x[2]-p0 for x in w):+6.1f}")
J = ["r_pelvis_roll_joint", "l_pelvis_roll_joint", "r_pelvis_pitch_joint", "l_pelvis_pitch_joint", "r_knee_joint", "l_knee_joint"]
def series(topic):
    out = []
    for t, m in d[topic]:
        if not m.name: continue
        out.append((s(t), dict(zip(m.name, m.position))))
    return out
g, js = series("/joint_goals"), series("/joint_states")
def base_of(ser, j):
    v = [p[j] for tt, p in ser if -1 < tt < 0 and j in p]; return sum(v) / len(v) if v else 0.0
for j in J:
    gb, sb = base_of(g, j), base_of(js, j)
    gk = [(tt, p[j] - gb) for tt, p in g if 0 <= tt <= 3 and j in p]
    sk = [(tt, p[j] - sb) for tt, p in js if 0 <= tt <= 3 and j in p]
    if not gk: continue
    ga = max(gk, key=lambda x: abs(x[1])); sa = max(sk, key=lambda x: abs(x[1]))
    print(f"{j:22s} goal dev max {math.degrees(ga[1]):+6.1f} @{ga[0]:.2f}s | actual dev max {math.degrees(sa[1]):+6.1f} @{sa[0]:.2f}s")
print("kick_done:", [round(s(t), 2) for t, m in d["/kick_done"]])
st = []
for t, m in d["/robot_state"]:
    if not st or st[-1][1] != m.data: st.append((round(s(t), 2), m.data[:60]))
print("robot_state:", st[:12])
for t, m in d["/motor_diagnostics"]:
    if 0 <= s(t) <= 4: print(f"diag {s(t):.2f}s: {m.data[:400]}")

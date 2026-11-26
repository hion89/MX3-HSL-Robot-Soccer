# 로봇에서 실행 (읽기 전용): 킥 bag → npz. 시간은 /kick_trigger 기준 [s] (트리거 없으면 첫 메시지 기준).
#   goal_t, goal (N,20) / js_t, q, qd / mf_t, pwm, cur / imu_t, quat(wxyz), gyro / names (20)
import sys, json
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
bag, out = sys.argv[1], sys.argv[2]
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
ty = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
d = {k: [] for k in ty}
while r.has_next():
    n, raw, t = r.read_next(); d[n].append((t, deserialize_message(raw, ty[n])))
t0 = d["/kick_trigger"][0][0] if d.get("/kick_trigger") else min(v[0][0] for v in d.values() if v)
s = lambda t: (t - t0) / 1e9
names = [m.name for _, m in d["/joint_states"] if len(m.name) == 20][0]
def series(topic, attr):
    T, X = [], []
    for t, m in d.get(topic, []):
        if not m.name: continue
        dm = dict(zip(m.name, getattr(m, attr)))
        T.append(s(t)); X.append([dm.get(n, np.nan) for n in names])
    return np.array(T), np.array(X)
# joint_goals 는 부분 메시지 (몸/목) 가 섞인다 → 마지막 값 유지 (forward fill)
gt, gx = [], []
last = {}
for t, m in d["/joint_goals"]:
    for n, p in zip(m.name, m.position): last[n] = p
    gt.append(s(t)); gx.append([last.get(n, np.nan) for n in names])
js_t, q = series("/joint_states", "position"); _, qd = series("/joint_states", "velocity")
mf_t, pwm = series("/motor_fast", "position"); _, cur = series("/motor_fast", "effort")
it = np.array([s(t) for t, _ in d["/imu/data"]])
quat = np.array([[m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z] for _, m in d["/imu/data"]])
gyro = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for _, m in d["/imu/data"]])
st = [(s(t), m.data) for t, m in d.get("/robot_state", [])]
volt = [(s(t), np.mean([x["voltage_v"] for x in json.loads(m.data)["motors"] if x["id"] <= 12])) for t, m in d.get("/motor_diagnostics", [])]
np.savez_compressed(out, names=np.array(names), goal_t=np.array(gt), goal=np.array(gx), js_t=js_t, q=q, qd=qd, mf_t=mf_t, pwm=pwm, cur=cur,
                    imu_t=it, quat=quat, gyro=gyro, state_t=np.array([a for a, _ in st]), state=np.array([b for _, b in st]),
                    volt_t=np.array([a for a, _ in volt]), volt=np.array([b for _, b in volt]))
print(out, len(gt), len(js_t), len(mf_t), len(it))

"""실물 시험 조건의 킥을 화면에 반복 재생한다 (로봇·ROS 무관).

MuJoCo 기본 뷰어(launch_passive)는 이 PC 그래픽 드라이버에서 검게 나와서,
오프스크린 렌더(mujoco.Renderer) 프레임을 OpenCV 창에 띄운다.

  python kick_view.py --policy kick_stable_27500.onnx --strength 0.5 [--foot box8] [--slow 1.5] [--push 8]
키: q/ESC 종료, space 일시정지, 1/2/3 카메라 (정면/옆/뒤)
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time

import cv2
import mujoco
import numpy as np

import kick_foot_sweep as kf
import kick_real_compare as kc
import mujoco_kick_stress as ks


class Quit(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--foot", default="mesh")
    ap.add_argument("--solref", type=float, default=0.02)
    ap.add_argument("--slow", type=float, default=1.5, help="1 = 실시간, 2 = 절반 속도")
    ap.add_argument("--sim-model", action="store_true", help="원래 MuJoCo 모델 (질량 5.7 kg, kp 50)")
    ap.add_argument("--size", default="960x720")
    ap.add_argument("--dxl", action="store_true", help="Dynamixel 모터 모델 (실물 재생 보정, 10.3~11.3 V)")
    ap.add_argument("--mu-scale", type=float, default=1.0, help="발-바닥 마찰 배율 (기준 미끄럼 0.6, 비틀림 0.1)")
    ap.add_argument("--pos", default="", help="창 위치 x,y (나란히 띄울 때)")
    ap.add_argument("--push", type=float, default=0.0, help="킥 디딤발 쪽 외력 N (0 = 없음, 0.5 s, 벌림 40° 에서 시작)")
    ap.add_argument("--rand-push", type=float, default=0.0,
                    help="킥 중 무작위 외력 N (방향·시점·지속 무작위, kick_real_compare 와 같은 규칙). 빨간 화살표로 표시")
    args = ap.parse_args()
    import onnxruntime as ort
    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    sim = kf.make_sim(args.foot)
    kc._DEF.clear()
    kc.apply_model(sim, **({} if args.sim_model else dict(base_add_kg=0.8, kp=30.0)))
    c = dataclasses.replace(ks.Cond(), name="real_like", volt=10.3, imu_delay_ms=10.0, hold_s=0.6, cmd_vx=0.20,
                            filter_tau=0.02, mu=0.6 * args.mu_scale, mu_torsion=0.1 * args.mu_scale,
                            solref_tc=args.solref, strength=args.strength,
                            push_N=args.push, dxl=args.dxl)
    W, H = (int(x) for x in args.size.split("x"))
    sim.m.vis.global_.offwidth = max(sim.m.vis.global_.offwidth, W)
    sim.m.vis.global_.offheight = max(sim.m.vis.global_.offheight, H)
    ren = mujoco.Renderer(sim.m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 1.2, -12.0
    view = {"az": 0.0, "paused": False}   # 상대 방위: 0 = 로봇 정면에서 봄
    name = args.policy.replace(chr(92), "/").split("/")[-1].removesuffix(".onnx")
    win = f"{name} push {args.push:g}N rp {args.rand_push:g}N mu x{args.mu_scale:g}{' DXL' if args.dxl else ''} - q: quit, space: pause, 1/2/3: front/side/back"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    if args.pos:
        cv2.moveWindow(win, *(int(v) for v in args.pos.split(",")))
    n_sub = int(round(ks.POLICY_DT / sim.dt))
    cnt = [0]
    last = [time.perf_counter()]
    info = {"text": ""}
    orig = sim.step_physics

    def show():
        yaw = math.degrees(ks.yaw_of(sim.R_base()))
        cam.azimuth = yaw + 180.0 + view["az"]
        cam.lookat[:] = sim.d.xpos[sim.base] + np.array([0, 0, -0.1])
        ren.update_scene(sim.d, cam)
        F = sim.d.xfrc_applied[sim.base, :3]
        fn = float(np.linalg.norm(F))
        if fn > 1e-6 and ren.scene.ngeom < ren.scene.maxgeom:
            p0 = sim.d.xpos[sim.base].copy()
            p1 = p0 + F / fn * (0.08 + 0.02 * fn)
            p0, p1 = p0 - (p1 - p0), p0           # 화살표가 몸통을 향해 민다
            g = ren.scene.geoms[ren.scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array([1.0, 0.1, 0.1, 1.0], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, p0, p1)
            ren.scene.ngeom += 1
        img = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, info["text"], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        if fn > 1e-6:
            cv2.putText(img, f"PUSH {fn:.0f} N", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 230), 2)
        cv2.imshow(win, img)
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                raise Quit
            if k == ord(" "):
                view["paused"] = not view["paused"]
            if k in (ord("1"), ord("2"), ord("3")):
                view["az"] = {ord("1"): 0.0, ord("2"): 90.0, ord("3"): 180.0}[k]
            if not view["paused"]:
                break
            time.sleep(0.03)

    def step(cond, prof):
        orig(cond, prof)
        cnt[0] += 1
        if cnt[0] % n_sub:
            return
        show()
        dt = ks.POLICY_DT * args.slow - (time.perf_counter() - last[0])
        if dt > 0:
            time.sleep(dt)
        last[0] = time.perf_counter()

    sim.step_physics = step
    seed = 1000
    try:
        while True:
            info["text"] = (f"{args.policy.replace(chr(92), '/').split('/')[-1]}  strength {args.strength}  "
                            f"push {args.push:g} N  rp {args.rand_push:g} N  mu x{args.mu_scale:g}{'  DXL' if args.dxl else ''}  seed {seed}")
            rows, fell_t = kc.run(sim, sess, c, seed, 2.0, 1.5, 0.03, rand_N=args.rand_push)
            s = kc.summarize(rows, fell_t)
            rp = rows[0].get("rand_push", {})
            print(f"seed {seed}: push {rp.get('N', 0):g}N ang {rp.get('ang_deg', 0):.0f} t {rp.get('start', 0):.2f}s "
                  f"dur {rp.get('dur', 0):.2f}s fall={s['fell']} roll_max={s['roll_max']:.0f} kick_leg={s['kick_leg']} "
                  f"kick_hip_roll={s['kick_hip_roll_max']:.0f} vball={s['vball_max']:.2f}", flush=True)
            info["text"] += f"  | fall={s['fell']} roll max {s['roll_max']:.0f} deg"
            t_end = time.perf_counter() + 1.5
            while time.perf_counter() < t_end:
                show()
                time.sleep(0.03)
            seed += 1
    except Quit:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

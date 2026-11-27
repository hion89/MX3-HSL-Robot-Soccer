"""V1-6QC 마찰 커리큘럼 제어 (PC 에서 실행, 2026-11-26 사용자 요청).

발-바닥 마찰을 단계별로 고정해 올리며 이어 학습한다 (서버 ~/kick16_runs/qc_train.sh).
- 단계마다 500 iteration. 100 iteration 마다 MuJoCo 게이트 (일반 300, push 150, dxl_real 12 회를 단계 마찰과 0.45 에서).
- 통과한 체크포인트 중 가장 나은 것에서 다음 단계를 시작한다.
- 통과가 없으면 같은 마찰로 2000 iteration 장기 학습 (200 마다 게이트, 연속 3 번 킥 < 50 % 면 중단).
- 그래도 안 되면 마지막 통과 마찰과의 중간값을 해 보고, 통과하면 실패했던 마찰을 한 번 더 시도한다.
상태: qc_state.json, 기록: qc_curriculum_log.txt (이 폴더).
"""
import csv, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/.mujoco_env/Scripts/python.exe"
CK_DIR = r"C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/kick_robust_20261116/ckpt_v16qc"
KEY = os.path.expanduser("~/.ssh/claude_isaac_server_ed25519")
HOST = "bami122@163.180.145.122"
SSH = ["ssh", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
EXP = "isaac_ws/unitree_rl_lab_kick16/logs/rsl_rl/unitree_mx3_walkkick_v1_6qc"
STAGES = [0.45, 0.6, 0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 2.0]
SHORT, LONG = 500, 2000
# 통과 기준 (L3 31800 과 Q15 31900 은 통과, Q13F 31900 (킥 76 %) 은 탈락하는 수준)
MIN_KICK, MIN_V, MAX_REAL_LIKE, MAX_DXL_FALL, MIN_DXL_VBALL = 90.0, 3.3, 12.0, 2.0 / 12 + 1e-6, 2.5
LOG = os.path.join(HERE, "qc_curriculum_log.txt")
STATE = os.path.join(HERE, "qc_state.json")


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ssh(cmd, timeout=60):
    for _ in range(5):
        try:
            r = subprocess.run(SSH + [HOST, cmd], capture_output=True, text=True, timeout=timeout)
            if r.returncode != 255:
                return r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(30)
    return 255, ""


def launch(mu, iters, load_run, ckpt, rn):
    cmd = (f"cd ~/kick16_runs && setsid nohup bash qc_train.sh {mu} {iters} {load_run} {ckpt} {rn} "
           f"> qc/{rn}_driver.log 2>&1 < /dev/null & disown")
    try:
        subprocess.run(SSH + [HOST, "mkdir -p ~/kick16_runs/qc; " + cmd], timeout=20, capture_output=True)
    except subprocess.TimeoutExpired:
        pass  # setsid 로 띄우면 ssh 가 바로 안 끝나는 경우가 있다 (원격 작업은 계속 돈다)


def status(rn):
    """상태 파일의 줄 머리말만 본다 (첫 줄은 git log 라 제외)."""
    out = ssh(f"cat ~/kick16_runs/qc/{rn}.status 2>/dev/null")[1].splitlines()[1:]
    return out


def has(st, word):
    return any(l.startswith(word) for l in st)


def gate(mu, run_dir, n, tag):
    """체크포인트 하나를 게이트한다. 결과 dict."""
    os.makedirs(CK_DIR, exist_ok=True)
    tag = f"{tag}_{run_dir[11:19]}"  # run 시각: 같은 마찰을 다시 해도 예전 결과·ONNX 를 쓰지 않는다
    local = os.path.join(CK_DIR, f"{tag}_model_{n}.pt")
    subprocess.run(["scp", "-q", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                    f"{HOST}:{EXP}/{run_dir}/model_{n}.pt", local], timeout=300)
    o = os.path.join(HERE, f"results_v16qc_{tag}_{n}")
    import shutil
    for suf in ("_gate", "_push", "_dxl"):
        shutil.rmtree(o + suf, ignore_errors=True)
    def run(args, logf):
        with open(logf, "w") as f:
            subprocess.run([PY] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    run(["gate_checkpoints.py", "--ckpt", local, "--trials", "300", "--out", o + "_gate"], o + "_gate.log")
    run(["gate_checkpoints.py", "--ckpt", local, "--trials", "150", "--suite", "push", "--out", o + "_push"], o + "_push.log")
    onnx = sorted(os.listdir(o + "_gate/onnx"))[0]
    mus = [mu] if abs(mu - 0.45) < 1e-6 else [mu, 0.45]
    args = ["kick_real_compare.py", "--dxl-real", "--trials", "12", "--strength", "1.0",
            "--policy", os.path.join(o + "_gate/onnx", onnx), "--out", o + "_dxl"]
    for m in mus:
        args += ["--mu", str(m)]
    run(args, o + "_dxl.log")
    g = [r for r in csv.DictReader(open(o + "_gate/ranking.csv")) if r["policy"] != "baseline"][0]
    p = [r for r in csv.DictReader(open(o + "_push/ranking.csv")) if r["policy"] != "baseline"][0]
    dx = {d["cond"]: d for d in json.load(open(o + "_dxl/summary.json"))}
    ds = dx[f"dxl_real_mu{mu:g}"]
    d45 = dx["dxl_real_mu0.45"]
    r = dict(n=n, kick=float(g["min_kick"]), v=float(g["v_nominal"]), real_like=float(g["real_like"]),
             worst=float(g["worst_fall"]), push_mean=float(p["mean_fall"]),
             dxl_fall=float(ds["fell"]), dxl_vball=float(ds["vball_max"]), dxl45_fall=float(d45["fell"]))
    r["pass"] = (r["kick"] >= MIN_KICK and r["v"] >= MIN_V and r["real_like"] <= MAX_REAL_LIKE
                 and r["dxl_fall"] <= MAX_DXL_FALL and r["dxl_vball"] >= MIN_DXL_VBALL)
    r["score"] = r["real_like"] + 100 * r["dxl_fall"] + 0.3 * r["push_mean"]
    log(f"  {tag} {n}: kick {r['kick']:.0f}% v {r['v']:.2f} real_like {r['real_like']:.1f}% push {r['push_mean']:.1f}% "
        f"dxl@{mu:g} fall {100*r['dxl_fall']:.0f}% vball {r['dxl_vball']:.2f} dxl@0.45 {100*r['dxl45_fall']:.0f}% "
        f"-> {'PASS' if r['pass'] else 'fail'} (score {r['score']:.1f})")
    return r


def stage(mu, start, iters, every, kind):
    """start = (load_run, ckpt, iteration). 반환: 가장 나은 통과 체크포인트 (load_run, ckpt, it) 또는 None."""
    load_run, ckpt, it0 = start
    tag = f"mu{mu:g}".replace(".", "p") + f"_{kind}"
    rn = f"qc_{tag}"
    log(f"== 단계 마찰 {mu:g} {kind} ({iters} it), 시작 {load_run}/{ckpt}")
    launch(mu, iters, load_run, ckpt, rn)
    run_dir = None
    t0 = time.time()
    while run_dir is None:
        st = status(rn)
        if has(st, "started"):
            i = [k for k, l in enumerate(st) if l.startswith("started")][0]
            run_dir = st[i].split("run ")[1].split()[0]
            log(f"  학습 시작: {run_dir} | env.yaml 마찰: {st[i + 1] if i + 1 < len(st) else '?'}")
        elif has(st, "failed"):
            log("  학습 시작 실패:\n" + "\n".join(st))
            return None, "launch_failed"
        elif time.time() - t0 > 3 * 3600:
            log("  학습 시작 대기 3 시간 초과")
            return None, "launch_failed"
        else:
            time.sleep(60)
    targets = list(range(it0 + every, it0 + iters, every))
    results, bad_streak = [], 0
    for n in targets:
        while True:
            rc, _ = ssh(f"test -f {EXP}/{run_dir}/model_{n}.pt")
            if rc == 0:
                break
            if has(status(rn), "done"):
                rc, _ = ssh(f"test -f {EXP}/{run_dir}/model_{n}.pt")
                if rc != 0:
                    log(f"  model_{n} 없음 (학습 종료)")
                break
            time.sleep(60)
        if ssh(f"test -f {EXP}/{run_dir}/model_{n}.pt")[0] != 0:
            continue
        time.sleep(10)
        try:
            r = gate(mu, run_dir, n, tag)
        except Exception as e:  # 게이트 하나가 실패해도 커리큘럼은 계속
            log(f"  {tag} {n}: 게이트 오류 {e!r}")
            continue
        results.append(r)
        bad_streak = bad_streak + 1 if r["kick"] < 50 else 0
        if kind == "long" and bad_streak >= 3 and not any(x["pass"] for x in results):
            log("  연속 3 번 킥 < 50 %: 장기 학습 중단")
            ssh("pkill -9 -f 'python scripts/rsl_rl/[t]rain.py --task Unitree-MX3-WalkKick-V1-6QC '")
            break
    while not has(status(rn), "done") and ssh("pgrep -f 'scripts/rsl_rl/[t]rain.py --task Unitree-MX3-WalkKick-V1-6QC'")[0] == 0:
        time.sleep(60)
    ok = [r for r in results if r["pass"]]
    if not ok:
        log(f"  마찰 {mu:g} {kind}: 통과 없음")
        return None, "no_pass"
    best = min(ok, key=lambda r: r["score"])
    log(f"  마찰 {mu:g} {kind}: 가장 나은 통과 {best['n']} (score {best['score']:.1f})")
    return (run_dir, f"model_{best['n']}.pt", best["n"]), "pass"


def try_mu(mu, start):
    best, why = stage(mu, start, SHORT, 100, "short")
    if best is None and why != "launch_failed":
        best, why = stage(mu, start, LONG, 200, "long")
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="qc_state.json 에서 이어 간다")
    ap.add_argument("--stages", default="", help="남은 마찰 목록 (쉼표)")
    ap.add_argument("--retried", default="", help="이미 중간값을 거쳐 다시 시도한 마찰 (쉼표)")
    a = ap.parse_args()
    state = dict(passed=[], start=["init_l3_31800", "model_31800.pt", 31800], ok_mu=None)
    if a.resume:
        state = json.load(open(STATE))
    queue = [float(x) for x in a.stages.split(",")] if a.stages else list(STAGES)
    retried = set(float(x) for x in a.retried.split(",")) if a.retried else set()
    log("V1-6QC 마찰 커리큘럼 " + ("재개" if a.resume else "시작") + ": " + ", ".join(f"{m:g}" for m in queue)
        + f" | 시작점 {state['start']}")
    while queue:
        mu = queue.pop(0)
        best = try_mu(mu, tuple(state["start"]))
        if best:
            state["start"], state["ok_mu"] = list(best), mu
            state["passed"].append(dict(mu=mu, run=best[0], ckpt=best[1]))
            json.dump(state, open(STATE, "w"), indent=1)
            continue
        if state["ok_mu"] is None or mu in retried:
            log(f"마찰 {mu:g} 에서 막힘. 중단.")
            break
        mid = round((state["ok_mu"] + mu) / 2, 3)
        log(f"마찰 {mu:g} 실패 → 중간값 {mid:g} 시도")
        best = try_mu(mid, tuple(state["start"]))
        if not best:
            log(f"중간값 {mid:g} 도 실패. 중단.")
            break
        state["start"], state["ok_mu"] = list(best), mid
        state["passed"].append(dict(mu=mid, run=best[0], ckpt=best[1]))
        json.dump(state, open(STATE, "w"), indent=1)
        retried.add(mu)
        queue.insert(0, mu)
    log("최종: " + json.dumps(state, ensure_ascii=False))
    log("QC_ALLDONE")


if __name__ == "__main__":
    main()

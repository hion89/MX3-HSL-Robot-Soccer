"""2026-11-27: 캐시된 ONNX 때문에 잘못 평가된 마찰 0.6 두 번째 시도 (02:44 short, 03:11 long) 를 다시 게이트하고,
통과가 있으면 그 체크포인트에서 0.8 부터 커리큘럼을 재개한다."""
import json, subprocess, sys
import qc_curriculum as q
RUN = "2026-11-27_03-11-18_qc_mu0p6_long"
q.log("== 재게이트: 마찰 0.6 두 번째 시도 (ONNX 캐시 버그로 32300~33800 이 첫 시도 값으로 평가됨)")
res = []
for n in (32300, 32400, 32800, 33000, 33200, 33400, 33600, 33800):
    try:
        res.append(q.gate(0.6, RUN, n, "mu0p6_long_regate"))
    except Exception as e:
        q.log(f"  {n}: 게이트 오류 {e!r}")
ok = [r for r in res if r["pass"]]
if not ok:
    q.log("재게이트: 통과 없음. 마찰 0.6 에서 막힘 (중간값 0.525 다음 재시도도 실패).")
    q.log("QC_ALLDONE")
    sys.exit(0)
best = min(ok, key=lambda r: r["score"])
q.log(f"재게이트: 통과 {best['n']} (score {best['score']:.1f}) → 0.8 부터 재개")
st = json.load(open(q.STATE))
st["start"], st["ok_mu"] = [RUN, f"model_{best['n']}.pt", best["n"]], 0.6
st["passed"].append(dict(mu=0.6, run=RUN, ckpt=f"model_{best['n']}.pt"))
json.dump(st, open(q.STATE, "w"), indent=1)
subprocess.run([sys.executable, "-u", "qc_curriculum.py", "--resume", "--stages", "0.8,1.0,1.2,1.3,1.5,1.7,2.0"])

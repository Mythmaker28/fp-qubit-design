import json, math, sys

EXPECTED = {
  "r2": -0.18031132289321805,
  "mae": 8.490189850602476,
  "baseline_mae_mean": 11.20515714256465,
  "baseline_mae_median": 8.640995475113122,
  "delta_mae_percent": 24.22962264089023,
  "coverage_90_percent": 0.9004524886877828,
  "ece_abs_error": 0.0004524886877828038
}

PATH = "deliverables/lab_v2_2_2/cv_metrics.json"

def close(a, b, tol_rel=1e-6, tol_abs=1e-9):
    return abs(a-b) <= max(tol_abs, tol_rel*max(abs(a), abs(b)))

def main():
    with open(PATH, "r", encoding="utf-8") as f:
        got = json.load(f)
    keys = sorted(EXPECTED.keys())
    ok = True
    for k in keys:
        if k not in got:
            print(f"[FAIL] Missing key: {{k}}")
            ok = False
            continue
        ea = EXPECTED[k]
        ga = got[k]
        if isinstance(ea, (int, float)) and isinstance(ga, (int, float)):
            if not close(ea, ga, tol_rel=1e-6, tol_abs=1e-9):
                print(f"[FAIL] {{k}} mismatch: expected={{ea}}, got={{ga}}")
                ok = False
        else:
            if ea != ga:
                print(f"[FAIL] {{k}} mismatch (non-numeric): expected={{ea}}, got={{ga}}")
                ok = False
    if ok:
        print("[OK] cv_metrics match expected values. NO-GO (pred) / GO (lab) framing is anchored.")
        sys.exit(0)
    else:
        print("[FAIL] cv_metrics deviate from repository-anchored values.")
        sys.exit(1)

if __name__ == "__main__":
    main()

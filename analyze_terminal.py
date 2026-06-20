"""
analyze_terminal.py

Terminal-phase focused MC analysis for the uncooperative docking scenario.
Run after monte_carlo.py completes and saves results.

Usage:
    python analyze_terminal.py                     # uses mc_results.pkl
    python analyze_terminal.py mc_results_run2.pkl
"""

import sys
import pickle
import numpy as np

# ── Load results ─────────────────────────────────────────────────────────────

path = sys.argv[1] if len(sys.argv) > 1 else "mc_results.pkl"
if path.endswith(".npz"):
    # Load from npz: reconstruct list of dicts
    _npz = np.load(path, allow_pickle=True)
    _keys = list(_npz.keys())
    N_raw = len(_npz[_keys[0]])
    results = [{k: _npz[k][i] for k in _keys} for i in range(N_raw)]
    # Normalise types used downstream
    for r in results:
        r["docked"] = bool(r["docked"])
        r["survey_engaged"] = bool(r.get("survey_engaged", False))
        r["uncoop_override_fired"] = bool(r.get("uncoop_override_fired", False))
        r["capture_timeout"] = bool(r.get("capture_timeout", False))
        r["soft_capture_seen"] = bool(r.get("soft_capture_seen", False))
        r["stress_case"] = str(r.get("stress_case", "nominal"))
        r["failure_reason"] = str(r.get("failure_reason", ""))
else:
    with open(path, "rb") as f:
        results = pickle.load(f)

N = len(results)
r = results  # shorthand


def _get(key, default=np.nan):
    return np.array([trial.get(key, default) for trial in r])


def pct(n, total=N):
    return 100.0 * n / total if total else 0.0


# ── Core masks ────────────────────────────────────────────────────────────────

docked          = np.array([t["docked"] for t in r])
survey_engaged  = np.array([t.get("survey_engaged", False) for t in r])
override_fired  = np.array([t.get("uncoop_override_fired", False) for t in r])
timeout         = np.array([t.get("capture_timeout", False) for t in r])
crashed         = np.array([not t["docked"] and not t.get("capture_timeout", False)
                             and np.isnan(t.get("total_dv_ms", np.nan)) for t in r])

n_docked   = docked.sum()
n_survey   = survey_engaged.sum()
n_override = override_fired.sum()
n_timeout  = timeout.sum()
n_failed   = N - n_docked

# ── Nozzle estimator metrics ──────────────────────────────────────────────────

max_conf        = _get("max_nozzle_conf", 0.0)
conf_at_term    = _get("nozzle_conf_at_terminal")
conf_at_cap     = _get("nozzle_conf_at_capture")
term_range      = _get("terminal_entry_range_m")
com_at_cap      = _get("com_at_capture_m")

# Failed to reach TERMINAL at all
never_terminal  = survey_engaged & ~docked & np.isnan(conf_at_term)
low_conf        = survey_engaged & ~docked & (max_conf < 0.6)

# ── DV / timing ──────────────────────────────────────────────────────────────

term_dv   = _get("term_dv_ms")
total_dv  = _get("total_dv_ms")
t_dock    = _get("t_dock_hr")

# ── Print report ─────────────────────────────────────────────────────────────

SEP = "=" * 62

print(f"\n{SEP}")
print(f"  TERMINAL PHASE ANALYSIS  —  {N} trials")
print(SEP)

print(f"\n── MISSION FUNNEL ──────────────────────────────────────────")
print(f"  Total trials            : {N}")
print(f"  SURVEY engaged          : {n_survey:3d}  ({pct(n_survey):.0f}%)")
print(f"  TERMINAL reached        : {int((~np.isnan(conf_at_term)).sum()):3d}  ({pct((~np.isnan(conf_at_term)).sum()):.0f}%)")
print(f"  SOFT CAPTURE triggered  : {int(np.array([t.get('soft_capture_seen', False) for t in r]).sum()):3d}  ({pct(np.array([t.get('soft_capture_seen', False) for t in r]).sum()):.0f}%)")
print(f"  DOCKED (hard capture)   : {n_docked:3d}  ({pct(n_docked):.0f}%)")

print(f"\n── CAPTURE METHOD ──────────────────────────────────────────")
normal_cap = docked & ~override_fired
print(f"  Normal gate             : {normal_cap.sum():3d}")
print(f"  UNCOOP override (COM)   : {(docked & override_fired).sum():3d}")
if n_docked > 0:
    print(f"  Override fraction       : {pct((docked & override_fired).sum(), n_docked):.0f}% of successes")

print(f"\n── NOZZLE ESTIMATOR PERFORMANCE ────────────────────────────")
valid_max = max_conf[max_conf > 0]
if len(valid_max):
    print(f"  Max confidence reached  : mean={valid_max.mean():.2f}  "
          f"min={valid_max.min():.2f}  max={valid_max.max():.2f}")
    print(f"  Trials conf never > 0.6 : {(max_conf < 0.6).sum():3d}")

term_mask = ~np.isnan(conf_at_term)
if term_mask.sum():
    print(f"  Conf at TERMINAL entry  : mean={np.nanmean(conf_at_term):.2f}  "
          f"min={np.nanmin(conf_at_term):.2f}  max={np.nanmax(conf_at_term):.2f}")
    print(f"  Range at TERMINAL entry : mean={np.nanmean(term_range):.1f}m  "
          f"min={np.nanmin(term_range):.1f}m  max={np.nanmax(term_range):.1f}m")

override_mask = ~np.isnan(conf_at_cap)
if override_mask.sum():
    print(f"  Conf at override fire   : mean={np.nanmean(conf_at_cap):.2f}  "
          f"min={np.nanmin(conf_at_cap):.2f}")
    print(f"  CoM dist at override    : mean={np.nanmean(com_at_cap):.3f}m  "
          f"max={np.nanmax(com_at_cap):.3f}m")

print(f"\n── FAILURE BREAKDOWN ───────────────────────────────────────")
print(f"  Total failures          : {n_failed:3d}")
print(f"    Survey never engaged  : {(~survey_engaged & ~docked).sum():3d}")
print(f"    SURVEY conf never 0.6 : {low_conf.sum():3d}")
print(f"    TERMINAL but no cap   : {(~np.isnan(conf_at_term) & ~docked).sum():3d}")
print(f"    Capture timeout       : {n_timeout:3d}")
fail_reasons = [t.get("failure_reason", "") for t in r if not t["docked"]]
reason_counts = {}
for fr in fail_reasons:
    if fr:
        reason_counts[fr] = reason_counts.get(fr, 0) + 1
for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"      {reason[:50]:<50} x{count}")

print(f"\n── TERMINAL DV & TIMING (successes only) ───────────────────")
if n_docked > 0:
    td = term_dv[docked]
    tot = total_dv[docked]
    th = t_dock[docked]
    print(f"  Term ΔV (m/s)  : mean={td.mean():.3f}  "
          f"min={td.min():.3f}  max={td.max():.3f}  p95={np.percentile(td,95):.3f}")
    print(f"  Total ΔV (m/s) : mean={tot.mean():.3f}  "
          f"min={tot.min():.3f}  max={tot.max():.3f}  p95={np.percentile(tot,95):.3f}")
    print(f"  Dock time (hr) : mean={th.mean():.2f}  "
          f"min={th.min():.2f}  max={th.max():.2f}  p95={np.percentile(th,95):.2f}")

print(f"\n── STRESS CASE BREAKDOWN ───────────────────────────────────")
stress_cases = sorted(set(t.get("stress_case", "nominal") for t in r))
for sc in stress_cases:
    sc_trials = [t for t in r if t.get("stress_case", "nominal") == sc]
    sc_docked = sum(t["docked"] for t in sc_trials)
    sc_conf   = np.nanmean([t.get("max_nozzle_conf", np.nan) for t in sc_trials])
    print(f"  {sc:<20} : {sc_docked}/{len(sc_trials)} docked "
          f"({pct(sc_docked, len(sc_trials)):.0f}%)  "
          f"mean_max_conf={sc_conf:.2f}")

print(f"\n── CHIEF TUMBLE RATE VS OUTCOME ────────────────────────────")
omega = _get("chief_omega_dps", 0.0)
bins  = [(0.0, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 1.0)]
for lo, hi in bins:
    mask = (omega >= lo) & (omega < hi)
    if mask.sum() == 0:
        continue
    nd = docked[mask].sum()
    mc = np.nanmean(max_conf[mask])
    print(f"  ω {lo:.2f}–{hi:.2f} deg/s : {nd}/{mask.sum()} docked  "
          f"({pct(nd, mask.sum()):.0f}%)  mean_max_conf={mc:.2f}")

print(f"\n{SEP}\n")

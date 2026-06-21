"""
RPOD Telemetry Analyser
=======================
Loads rpod_telemetry.npz (single-run output from main.py) and produces:
  - Printed summary with phase breakdown, alignment diagnostics, nav errors
  - 3x3 figure covering all key GNC subsystems

Units actually stored in the NPZ (as written by main.py):
  rate       deg/s      (np.degrees applied before storage)
  hx/hy/hz   mNms       (stored as rw.h * 1e3)
  T_gg/T_srp nNm        (stored as |T| * 1e9)
  err_deg    (N,2) where col0 = time [s], col1 = MEKF pointing error [deg]
  rn_dv      m/s        (cumulative |accel|*dt)

Usage
-----
  python tools/analyze_rpod_telemetry.py                         # default path
  python tools/analyze_rpod_telemetry.py path/to/telemetry.npz
  python tools/analyze_rpod_telemetry.py --show                  # open window
"""

import sys
import pathlib
import numpy as np
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Path handling ──────────────────────────────────────────────────────────────
NPZ_DEFAULT = pathlib.Path(__file__).parent.parent / "rpod_telemetry.npz"
npz_path    = pathlib.Path(sys.argv[1]) if (len(sys.argv) > 1
                and not sys.argv[1].startswith("--")) else NPZ_DEFAULT

if not npz_path.exists():
    print(f"[ERROR] File not found: {npz_path}")
    sys.exit(1)

d = np.load(npz_path, allow_pickle=True)

def _arr(key, default=None, n=None):
    """Load array, return default (or NaN array of length n) if key absent."""
    if key in d:
        return np.asarray(d[key], dtype=float)
    if default is not None:
        return np.asarray(default, dtype=float)
    return np.full(n or 0, np.nan)

def _scalar(key, default=None):
    val = d.get(key)
    if val is None:
        return default
    return np.asarray(val).item()

# ── RPOD mode map ──────────────────────────────────────────────────────────────
RPOD_NAMES  = {1:"FORM_HOLD", 2:"LAMBERT", 3:"PROX_OPS",
               4:"TERMINAL",  5:"SOFT_CAP", 6:"DOCKING", 7:"LOST_TGT"}
RPOD_COLORS = {1:"#aec6cf", 2:"#b5ead7", 3:"#ffd700",
               4:"#ff9933", 5:"#ff4444", 6:"#00cc44", 7:"#cc00cc"}

# ── ADCS time series ───────────────────────────────────────────────────────────
t_adcs = _arr("t")                        # s
mode_a = _arr("mode").astype(int)
rate   = _arr("rate")                     # deg/s  (already converted in main.py)
hx     = _arr("hx")                       # mNms
hy     = _arr("hy")
hz     = _arr("hz")
h_mag  = np.sqrt(hx**2 + hy**2 + hz**2)  # mNms
T_gg   = _arr("T_gg")                     # nNm
T_srp  = _arr("T_srp")                    # nNm
ecl    = _arr("eclipse_nu")

# err_deg: col0 = time [s], col1 = MEKF pointing error [deg]
_ed = _arr("err_deg")
if _ed.ndim == 2 and _ed.shape[1] >= 2:
    t_err  = _ed[:, 0]
    e_err  = _ed[:, 1]
else:
    t_err = t_adcs
    e_err = _ed if _ed.ndim == 1 else np.full(len(t_adcs), np.nan)

# ── RPOD nav time series ───────────────────────────────────────────────────────
rn_t     = _arr("rn_t")
rn_mode  = _arr("rn_mode").astype(int)
rn_range = _arr("rn_range")               # truth range [m]
rn_est   = _arr("rn_est_range")           # EKF range [m]
rn_dv    = _arr("rn_dv")                  # cumulative DV [m/s]
rn_pos_e = _arr("rn_pos_err")             # EKF pos error [m]
rn_vel_e = _arr("rn_vel_err")             # EKF vel error [m/s]
rn_pr    = _arr("rn_port_range")          # truth port range [m]
rn_pv    = _arr("rn_port_vrel")           # port relative speed [m/s]
rn_align = _arr("rn_align_deg")           # truth dock-axis alignment [deg]
rn_cone  = _arr("rn_cone_error_deg")      # cone error [deg]
rn_lat   = _arr("rn_lateral_m")           # lateral offset [m]
rn_ax    = _arr("rn_axial_m")             # axial offset [m]
rn_dx    = _arr("rn_dx")
rn_dy    = _arr("rn_dy")
rn_dz    = _arr("rn_dz")

# estimated alignment (logged if main.py is new enough, else NaN)
rn_est_align = _arr("rn_est_align_deg", n=len(rn_t))
rn_align_bias = _arr("rn_align_bias_deg", n=len(rn_t))
rn_pose_status = _arr("rn_pose_status", default=np.zeros(len(rn_t)), n=len(rn_t)).astype(int)
rn_pose_visible_count = _arr("rn_pose_visible_count", default=np.zeros(len(rn_t)), n=len(rn_t))
rn_pose_visible_mask = _arr("rn_pose_visible_mask", default=np.zeros(len(rn_t)), n=len(rn_t)).astype(np.int64)
rn_pose_stub_visible = _arr("rn_pose_stub_visible", default=np.zeros(len(rn_t)), n=len(rn_t)).astype(bool)
rn_pose_age_s = _arr("rn_pose_age_s", n=len(rn_t))
rn_pose_reproj_rms_px = _arr("rn_pose_reproj_rms_px", n=len(rn_t))
rn_pose_pca_cond = _arr("rn_pose_pca_cond", n=len(rn_t))

POSE_STATUS_NAMES = {
    0: "NONE",
    1: "ACCEPTED",
    2: "REJECTED",
    3: "COAST",
    4: "NO_VISIBLE",
    5: "PNP_FAIL",
    6: "RMS_REJECT",
    7: "ACQUIRE",
}

# ── Scalar metadata ────────────────────────────────────────────────────────────
docked           = bool(_scalar("docked", False))
cap_timeout      = bool(_scalar("capture_timeout", False))
cap_detail       = str(_scalar("capture_timeout_detail", ""))
total_t_s        = float(_scalar("total_time_s", rn_t[-1] if len(rn_t) else 0))
dock_range       = float(_scalar("dock_range_m", 0.30))
hard_range       = float(_scalar("hard_capture_range_m", 0.08))
soft_range       = float(_scalar("soft_capture_range_m", 0.30))
cone_half        = float(_scalar("dock_cone_half_angle_deg", 15.0))
total_dv_ms      = float(rn_dv[-1]) * 1000 if len(rn_dv) else 0.0  # mm/s
uncooperative    = bool(_scalar("uncooperative_mode", False))

# ── Helpers ────────────────────────────────────────────────────────────────────
def pm(mv):
    return rn_mode == mv

def hr(s):
    return s / 3600.0

def phase_span(ax):
    if len(rn_mode) == 0:
        return
    prev_m = rn_mode[0]; seg_t = rn_t[0]
    for i in range(1, len(rn_mode)):
        if rn_mode[i] != prev_m or i == len(rn_mode) - 1:
            c = RPOD_COLORS.get(int(prev_m), "#dddddd")
            ax.axvspan(hr(seg_t), hr(rn_t[i]), alpha=0.12, color=c, linewidth=0)
            prev_m = rn_mode[i]; seg_t = rn_t[i]

def _finite_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "n=0"
    return (f"n={x.size} mean={np.mean(x):.2f} "
            f"p50={np.percentile(x, 50):.2f} "
            f"p95={np.percentile(x, 95):.2f} "
            f"max={np.max(x):.2f}")

def _status_counts(mask):
    if not np.any(mask):
        return "-"
    vals, counts = np.unique(rn_pose_status[mask], return_counts=True)
    return ", ".join(
        f"{POSE_STATUS_NAMES.get(int(v), str(int(v)))}={int(c)}"
        for v, c in zip(vals, counts))

def _visible_indices(mask):
    if not np.any(mask):
        return "-"
    merged = 0
    for v in rn_pose_visible_mask[mask]:
        if np.isfinite(v):
            merged |= int(v)
    idx = [str(i) for i in range(63) if merged & (1 << i)]
    return ",".join(idx) if idx else "-"

# ─────────────────────────────────────────────────────────────────────────────
# PRINTED SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 65
print(SEP)
print("  RPOD Telemetry Analysis")
print(SEP)
_mode_str = "NON-COOPERATIVE (engine-bell grapple)" if uncooperative else "COOPERATIVE (dock-port capture)"
print(f"  Scenario mode : {_mode_str}")

if docked:
    result_str = "DOCKED"
elif cap_timeout:
    result_str = f"CAPTURE_TIMEOUT  ({cap_detail})"
else:
    result_str = "TIMEOUT (increase T_SIM_MAX)"
print(f"  Result        : {result_str}")
print(f"  Total time    : {total_t_s:.1f}s  ({hr(total_t_s):.2f}hr)")
print(f"  Total DV      : {total_dv_ms:.1f}mm/s  ({total_dv_ms/1000:.3f}m/s)")
ISP = 220.0; G0 = 9.80665; MASS = 50.0
prop_g = MASS * (1 - np.exp(-(total_dv_ms/1000) / (ISP * G0))) * 1000
print(f"  Propellant    : {prop_g:.1f}g  (Isp={ISP}s, m0={MASS}kg)")
print()

# Per-phase breakdown
print(f"  {'Phase':<12} {'Dur(s)':>8} {'DV(mm/s)':>10} {'Rng_end(m)':>11}")
print(f"  {'-'*45}")
for mv in sorted(set(rn_mode)):
    mk = pm(mv)
    if not np.any(mk):
        continue
    ts   = rn_t[mk]
    dur  = ts[-1] - ts[0]
    dv_d = (rn_dv[mk][-1] - rn_dv[mk][0]) * 1000
    r_e  = rn_range[mk][-1]
    nm   = RPOD_NAMES.get(int(mv), str(mv))
    print(f"  {nm:<12} {dur:>8.0f} {dv_d:>10.1f}   {r_e:>8.2f}m")
print()

# Terminal alignment quality
# rn_align_deg = deputy +Z vs anti-parallel chief +Z (nozzle axis in both modes)
t_mask = pm(4)
if np.any(t_mask):
    a = rn_align[t_mask]
    fin = np.isfinite(a)
    if np.any(fin):
        a = a[fin]
        _align_label = "nozzle-axis align" if uncooperative else "dock-axis align"
        print(f"  TERMINAL truth align ({_align_label}) : min={a.min():.1f}°  mean={a.mean():.1f}°  "
              f"pct<30°={100*(a<30).mean():.0f}%  pct<15°={100*(a<15).mean():.0f}%")
        ea_t = rn_est_align[t_mask]
        fin_e = np.isfinite(ea_t) & (ea_t < 180)
        if np.any(fin_e):
            ea = ea_t[fin_e]
            b = rn_align[t_mask][fin_e] - ea
            print(f"  TERMINAL est align   : min={ea.min():.1f}°  mean={ea.mean():.1f}°  "
                  f"pct<30°={100*(ea<30).mean():.0f}%  pct<15°={100*(ea<15).mean():.0f}%")
            print(f"  TERMINAL align bias  : truth-est mean={b.mean():.1f}°  "
                  f"maxabs={np.abs(b).max():.1f}°")
        # detect oscillation (many crossings of mean)
        a_c = a - a.mean()
        n_cross = np.sum(np.diff(np.sign(a_c)) != 0)
        if n_cross > 20:
            print(f"  TERMINAL align : *** OSCILLATING ({n_cross} sign-crossings) — "
                  f"spin-sync may be diverged ***")

    print(f"  TERMINAL pose estimator:")
    print(f"    status counts : {_status_counts(t_mask)}")
    _t_status     = rn_pose_status[t_mask]
    _n_total      = len(_t_status)
    _n_accepted   = int(np.sum(_t_status == 1))
    _n_rejected   = int(np.sum(_t_status == 2))
    _n_no_visible = int(np.sum(_t_status == 4))
    _n_pnp_fail   = int(np.sum(_t_status == 5))
    if _n_total > 0:
        _accept_pct = 100 * _n_accepted / _n_total
        _nv_pct     = 100 * _n_no_visible / _n_total
        _rej_pct    = 100 * _n_rejected / _n_total
        if _nv_pct > 50:
            print(f"    *** CAMERA BLIND {_nv_pct:.0f}% of TERMINAL — "
                  f"spin-sync pointing camera away from chief ***")
        elif _rej_pct > 30:
            print(f"    *** GATE REJECTING {_rej_pct:.0f}% of TERMINAL — "
                  f"Mahalanobis gate too tight or EKF prior diverged ***")
        elif _n_pnp_fail > 0 and _n_pnp_fail / _n_total > 0.1:
            print(f"    *** PNP_FAIL {100*_n_pnp_fail/_n_total:.0f}% of TERMINAL — "
                  f"EPnP numerically failing ***")
        if _accept_pct < 10:
            print(f"    *** Only {_accept_pct:.0f}% of TERMINAL steps accepted "
                  f"— pose coasted through final approach ***")
    print(f"    visible count : {_finite_stats(rn_pose_visible_count[t_mask])}")
    print(f"    stub visible  : {100*np.mean(rn_pose_stub_visible[t_mask]):.1f}%")
    print(f"    pose age [s]  : {_finite_stats(rn_pose_age_s[t_mask])}")
    print(f"    reproj RMS px : {_finite_stats(rn_pose_reproj_rms_px[t_mask])}")
    print(f"    PCA cond      : {_finite_stats(rn_pose_pca_cond[t_mask])}")

# SOFT_CAPTURE detail
sc_mask = pm(5)
if np.any(sc_mask):
    a_sc   = rn_align[sc_mask]
    t_sc   = rn_t[sc_mask]
    pr_sc  = rn_pr[sc_mask]
    pv_sc  = rn_pv[sc_mask]
    dur_sc = t_sc[-1] - t_sc[0]
    best_i = int(np.nanargmin(a_sc))
    best_t = t_sc[best_i] - t_sc[0]

    _sc_align_label = "nozzle-axis align" if uncooperative else "dock-axis align"
    print(f"\n  SOFT_CAPTURE   : duration={dur_sc:.0f}s")
    print(f"    truth {_sc_align_label} entry / best / final : "
          f"{a_sc[0]:.1f}° / {a_sc[best_i]:.1f}° (at +{best_t:.0f}s) / {a_sc[-1]:.1f}°")
    print(f"      truth pct<30°={100*(a_sc<30).mean():.0f}%  "
          f"pct<15°={100*(a_sc<15).mean():.0f}%  pct<10°={100*(a_sc<10).mean():.0f}%")
    ea_sc = rn_est_align[sc_mask]
    fe_sc = np.isfinite(ea_sc) & (ea_sc < 180)
    if np.any(fe_sc):
        print(f"    est align   entry / best / final : "
              f"{ea_sc[fe_sc][0]:.1f}° / {np.nanmin(ea_sc[fe_sc]):.1f}° / {ea_sc[fe_sc][-1]:.1f}°")
        print(f"      est   pct<30°={100*(ea_sc[fe_sc]<30).mean():.0f}%  "
              f"pct<15°={100*(ea_sc[fe_sc]<15).mean():.0f}%  pct<10°={100*(ea_sc[fe_sc]<10).mean():.0f}%")
    if uncooperative:
        print(f"    port   entry / best / final : "
              f"{pr_sc[0]*100:.1f}cm / {pr_sc.min()*100:.1f}cm / {pr_sc[-1]*100:.1f}cm"
              f"  [cooperative-port dist — not nozzle criterion]")
    else:
        print(f"    port   entry / best / final : "
              f"{pr_sc[0]*100:.1f}cm / {pr_sc.min()*100:.1f}cm / {pr_sc[-1]*100:.1f}cm")
    print(f"    vrel   entry / best / final : "
          f"{pv_sc[0]*1e3:.2f} / {pv_sc.min()*1e3:.2f} / {pv_sc[-1]*1e3:.2f} mm/s")

    # Convergence type: slow-linear vs oscillating
    a_c = a_sc - a_sc.mean()
    n_cross = np.sum(np.diff(np.sign(a_c)) != 0)
    if n_cross > 10:
        print(f"    alignment OSCILLATING ({n_cross} crossings) — "
              f"spin-sync diverged after near-miss")
        # estimate oscillation period from first few crossings
        ci = np.where(np.diff(np.sign(a_c)) != 0)[0]
        if len(ci) >= 4:
            half_periods = np.diff(t_sc[ci])[:8]
            print(f"    approx oscillation half-periods: "
                  f"{', '.join(f'{p:.0f}s' for p in half_periods)}")
    else:
        # linear convergence
        if dur_sc > 1.0 and a_sc[0] > a_sc[-1]:
            rate_c = (a_sc[0] - a_sc[-1]) / dur_sc   # deg/s
            rem    = max(a_sc[-1] - 10.0, 0.0)
            eta    = rem / max(rate_c, 1e-9)
            print(f"    convergence: {rate_c*1000:.2f} mdeg/s  "
                  f"remaining-to-10°: {rem:.1f}°  ETA: {eta:.0f}s more needed")

    # estimated vs truth alignment bias
    ea_sc = rn_est_align[sc_mask]
    fin_e = np.isfinite(ea_sc) & (ea_sc < 180)
    if np.any(fin_e):
        bias  = a_sc[fin_e] - ea_sc[fin_e]
        print(f"    pose-est bias : truth-est mean={bias.mean():.1f}°  "
              f"max={np.abs(bias).max():.1f}°")
        if np.abs(bias).mean() > 20:
            if np.nanmin(a_sc) <= 10 and np.nanmin(ea_sc[fin_e]) > 10:
                print(f"    *** LARGE POSE-ESTIMATOR BIAS: "
                      f"truth reaches hard gate, estimate does not ***")
            elif np.nanmin(a_sc) > 10:
                print(f"    *** LARGE POSE-ESTIMATOR BIAS: "
                      f"truth also misses hard gate ***")
            else:
                print(f"    *** LARGE POSE-ESTIMATOR BIAS: "
                      f"truth/estimate disagree during soft capture ***")

    print("    pose estimator:")
    print(f"      status counts : {_status_counts(sc_mask)}")
    print(f"      visible count : {_finite_stats(rn_pose_visible_count[sc_mask])}")
    print(f"      stub visible  : {100*np.mean(rn_pose_stub_visible[sc_mask]):.1f}%")
    print(f"      visible idx   : {_visible_indices(sc_mask)}")
    print(f"      pose age [s]  : {_finite_stats(rn_pose_age_s[sc_mask])}")
    print(f"      reproj RMS px : {_finite_stats(rn_pose_reproj_rms_px[sc_mask])}")
    print(f"      PCA cond      : {_finite_stats(rn_pose_pca_cond[sc_mask])}")

print()
print(f"  EKF pos error  : mean={rn_pos_e.mean():.2f}m  max={rn_pos_e.max():.2f}m")
print(f"  EKF vel error  : mean={rn_vel_e.mean()*1e3:.1f}mm/s  max={rn_vel_e.max()*1e3:.1f}mm/s")
print(f"  Wheel |h| max  : {h_mag.max():.1f} mNms  final={h_mag[-1]:.1f} mNms")
print(f"  Body rate max  : {rate.max():.2f} deg/s  "
      f"final={rate[-1]:.3f} deg/s")
if len(e_err):
    print(f"  MEKF point.err : mean={e_err.mean():.3f}°  max={e_err.max():.3f}°")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE  (3 × 3)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
title_str = ("DOCKED" if docked else
             f"FAILED — {cap_detail}" if cap_timeout else "TIMEOUT")
fig.suptitle(
    f"RPOD Telemetry — {title_str}   "
    f"ΔV={total_dv_ms/1000:.2f}m/s   t={hr(total_t_s):.2f}hr",
    fontsize=13, fontweight="bold")
gs = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

t_hr  = hr(rn_t)
ta_hr = hr(t_adcs)
te_hr = hr(t_err)

legend_patches = [
    mpatches.Patch(color=RPOD_COLORS.get(m, "#ccc"), alpha=0.5,
                   label=RPOD_NAMES.get(m, str(m)))
    for m in sorted(set(rn_mode))
]

# ── 1. Relative range (log) ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
phase_span(ax1)
ax1.semilogy(t_hr, rn_range, 'k-', lw=0.9, label='Truth')
ax1.semilogy(t_hr, np.clip(rn_est, 1e-2, None), 'b--', lw=0.7, alpha=0.7, label='EKF')
ax1.axhline(dock_range, color='r',      lw=0.8, ls='--', label=f'Dock {dock_range}m')
ax1.axhline(soft_range, color='orange', lw=0.7, ls=':',  label=f'Soft {soft_range}m')
ax1.set(xlabel="Time [hr]", ylabel="Range [m]", title="Relative Range")
ax1.legend(fontsize=6); ax1.grid(True, alpha=0.3)

# ── 2. Port approach (terminal + SC) ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
close = rn_mode >= 4
if np.any(close):
    tc = t_hr[close]
    phase_span(ax2)
    ax2b = ax2.twinx()
    ax2.plot(tc, rn_pr[close]*100,  'b-',  lw=1.1, label='Port range [cm]')
    ax2b.plot(tc, rn_pv[close]*1e3, 'r-',  lw=0.9, alpha=0.7, label='v_rel [mm/s]')
    ax2.axhline(hard_range*100, color='r', lw=0.8, ls='--')
    ax2.set(xlabel="Time [hr]", ylabel="Port range [cm]",
            title="Port Approach (Terminal+)")
    ax2b.set_ylabel("v_rel [mm/s]", color='r')
    ax2.set_xlim(tc[0] - 0.01, tc[-1] + 0.01)
    l1, b1 = ax2.get_legend_handles_labels()
    l2, b2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1+l2, b1+b2, fontsize=6)
ax2.grid(True, alpha=0.3)

# ── 3. Dock-axis alignment ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
phase_span(ax3)
fin_a = np.isfinite(rn_align) & (rn_align < 180.1)
ax3.plot(t_hr[fin_a], rn_align[fin_a], 'k-', lw=0.9, label='Truth align')

# estimated alignment (new runs only)
fin_e = np.isfinite(rn_est_align) & (rn_est_align < 180.1)
if np.any(fin_e):
    ax3.plot(t_hr[fin_e], rn_est_align[fin_e],
             'b--', lw=0.8, alpha=0.75, label='Est align')

ax3.axhline(10, color='r',      lw=1.0, ls='--', label='Gate 10°')
ax3.axhline(30, color='orange', lw=0.8, ls=':',  label='SC entry 30°')
if np.any(sc_mask):
    best = rn_align[sc_mask][np.nanargmin(rn_align[sc_mask])]
    ax3.axhline(best, color='green', lw=0.8, ls='-.', label=f'Best {best:.1f}°')
    # shade SOFT_CAPTURE period
    t_sc_s = rn_t[sc_mask]; t_sc_e = t_sc_s[-1]
    ax3.axvspan(hr(t_sc_s[0]), hr(t_sc_e), alpha=0.08, color='deeppink')
ax3.set(xlabel="Time [hr]", ylabel="Alignment [deg]",
        title="Dock-Axis Alignment")
ax3.legend(fontsize=6)
ax3.set_ylim(0, min(180, np.nanmax(rn_align[fin_a])*1.05 + 5) if np.any(fin_a) else 180)
ax3.grid(True, alpha=0.3)

# ── 4. Reaction wheel momentum ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(ta_hr, hx, lw=0.7, label='hx [mNms]')
ax4.plot(ta_hr, hy, lw=0.7, label='hy [mNms]')
ax4.plot(ta_hr, hz, lw=0.7, label='hz [mNms]')
ax4.plot(ta_hr, h_mag, 'k-', lw=1.0, label='|h| [mNms]')
ax4.axhline(3000,  color='r', lw=0.8, ls='--', label='Dump thr 3Nms')
ax4.axhline(-3000, color='r', lw=0.8, ls='--')
ax4.set(xlabel="Time [hr]", ylabel="Momentum [mNms]",
        title="Reaction Wheel Momentum")
ax4.legend(fontsize=6); ax4.grid(True, alpha=0.3)

# ── 5. Body rate + MEKF pointing error ────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(ta_hr, rate, 'b-', lw=0.7, label='Body rate [deg/s]')
ax5b = ax5.twinx()
if len(e_err):
    ax5b.plot(te_hr, e_err, 'r-', lw=0.7, alpha=0.8, label='Pointing err [deg]')
    ax5b.set_ylabel("Pointing error [deg]", color='r')
ax5.set(xlabel="Time [hr]", ylabel="Body rate [deg/s]",
        title="ADCS: Rate & Pointing Error")
l1, b1 = ax5.get_legend_handles_labels()
l2, b2 = ax5b.get_legend_handles_labels()
ax5.legend(l1+l2, b1+b2, fontsize=6); ax5.grid(True, alpha=0.3)

# ── 6. DV accumulation ────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
phase_span(ax6)
ax6.plot(t_hr, rn_dv * 1000, 'k-', lw=1.1)
for mv in sorted(set(rn_mode)):
    mk = pm(mv)
    if np.any(mk):
        ax6.axvline(hr(rn_t[mk][0]),
                    color=RPOD_COLORS.get(int(mv), '#888'), lw=0.7, ls=':')
ax6.set(xlabel="Time [hr]", ylabel="Cumulative DV [mm/s]",
        title="ΔV Accumulation")
ax6.legend(handles=legend_patches, fontsize=5, ncol=2)
ax6.grid(True, alpha=0.3)

# ── 7. EKF nav error ──────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
phase_span(ax7)
ax7.semilogy(t_hr, rn_pos_e, 'b-', lw=0.9, label='Pos err [m]')
ax7b = ax7.twinx()
ax7b.semilogy(t_hr, rn_vel_e*1e3, 'r-', lw=0.9, alpha=0.7, label='Vel err [mm/s]')
ax7.set(xlabel="Time [hr]", ylabel="Pos error [m]", title="EKF Navigation Error")
ax7b.set_ylabel("Vel error [mm/s]", color='r')
l1, b1 = ax7.get_legend_handles_labels()
l2, b2 = ax7b.get_legend_handles_labels()
ax7.legend(l1+l2, b1+b2, fontsize=6); ax7.grid(True, alpha=0.3)

# ── 8. SOFT_CAPTURE close-up: alignment trajectory ────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
if np.any(sc_mask):
    t_sc_hr = hr(rn_t[sc_mask])
    a_sc    = rn_align[sc_mask]
    ax8.plot(t_sc_hr, a_sc, 'k-', lw=1.1, label='Truth align')
    ea_sc = rn_est_align[sc_mask]
    fe    = np.isfinite(ea_sc) & (ea_sc < 180)
    if np.any(fe):
        ax8.plot(t_sc_hr[fe], ea_sc[fe], 'b--', lw=0.9, alpha=0.8, label='Est align')
    ax8.axhline(10, color='r',      lw=1.0, ls='--', label='Gate 10°')
    ax8.axhline(30, color='orange', lw=0.8, ls=':')
    # mark best
    bi = int(np.nanargmin(a_sc))
    ax8.scatter([t_sc_hr[bi]], [a_sc[bi]], c='green', s=60, zorder=5,
                label=f'Best {a_sc[bi]:.1f}° @+{rn_t[sc_mask][bi]-rn_t[sc_mask][0]:.0f}s')
    ax8.set(xlabel="Time [hr]", ylabel="Alignment [deg]",
            title="SOFT_CAPTURE Alignment (close-up)")
    ax8.legend(fontsize=6); ax8.grid(True, alpha=0.3)
else:
    ax8.text(0.5, 0.5, "No SOFT_CAPTURE phase", ha='center', va='center',
             transform=ax8.transAxes)
    ax8.set_title("SOFT_CAPTURE Alignment")

# ── 9. CW trajectory top-down ─────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
sc9 = ax9.scatter(rn_dy, rn_dx, c=t_hr, s=0.4, cmap='viridis', alpha=0.6)
ax9.scatter([0], [0], marker='*', s=120, c='r', zorder=5, label='Chief')
ax9.set(xlabel="Along-track y [m]", ylabel="Cross-track x [m]",
        title="CW Trajectory (top-down)")
ax9.legend(fontsize=7); ax9.grid(True, alpha=0.3)
plt.colorbar(sc9, ax=ax9, label='Time [hr]', fraction=0.046, pad=0.04)

out_path = npz_path.parent / "rpod_telemetry_plots.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Plot saved: {out_path}")
if "--show" in sys.argv:
    plt.show()

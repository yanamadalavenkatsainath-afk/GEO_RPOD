import argparse
import os
from collections import Counter, defaultdict

import numpy as np


DEFAULT_NPZ = "monte_carlo_results.npz"


def load(path=DEFAULT_NPZ):
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(os.path.dirname(__file__), "..", path),
        os.path.join(os.path.dirname(__file__), "..", "mc_results", path),
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate, np.load(candidate, allow_pickle=True)
    raise FileNotFoundError(path)


def has(d, key):
    return key in d.files


def arr(d, key, default=np.nan, dtype=float):
    if key not in d.files:
        return None
    try:
        return np.asarray(d[key], dtype=dtype)
    except (TypeError, ValueError):
        return np.asarray(d[key])


def scalar(d, key, i, default=np.nan):
    if key not in d.files:
        return default
    try:
        return float(d[key][i])
    except (TypeError, ValueError, IndexError):
        return default


def text(d, key, i, default=""):
    if key not in d.files:
        return default
    try:
        return str(d[key][i])
    except IndexError:
        return default


def flag(d, key, i, default=False):
    if key not in d.files:
        return default
    try:
        return bool(d[key][i])
    except IndexError:
        return default


def finite(v):
    return np.isfinite(v)


def fmt(v, unit="", digits=3, missing="---"):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return missing
    if not np.isfinite(v):
        return missing
    return f"{v:.{digits}f}{unit}"


def pct(num, den):
    return 100.0 * num / max(den, 1)


def qstats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def print_stats(label, values, unit="", digits=3):
    st = qstats(values)
    if st is None:
        print(f"  {label:<24} no finite data")
        return
    print(
        f"  {label:<24} n={st['n']:>3}  "
        f"mean={fmt(st['mean'], unit, digits):>10}  "
        f"p50={fmt(st['p50'], unit, digits):>10}  "
        f"p95={fmt(st['p95'], unit, digits):>10}  "
        f"min/max={fmt(st['min'], unit, digits):>10}/{fmt(st['max'], unit, digits):<10}"
    )


def classify_signature(d, i, docked):
    reason = text(d, "failure_reason", i, "UNKNOWN")
    detail = text(d, "capture_timeout_detail", i, "NONE")
    soft_seen = flag(d, "soft_capture_seen", i)
    final_hard = flag(d, "final_hard_strict", i)
    final_soft = flag(d, "final_soft_stable", i)
    geom = flag(d, "final_geometry_ok", i)
    align = scalar(d, "final_align_deg", i)
    align_min = scalar(d, "soft_capture_align_min_deg", i)
    cone = scalar(d, "final_cone_err_deg", i)
    lateral = scalar(d, "final_lateral_m", i)
    min_port = scalar(d, "min_port_range_m", i)
    final_port = scalar(d, "final_port_range_m", i)
    final_v = scalar(d, "final_port_vrel_ms", i)
    nav = scalar(d, "max_nav_err_m", i)
    cam_drop = scalar(d, "camera_drop_s", i, 0.0)
    range_drop = scalar(d, "range_drop_s", i, 0.0)

    if docked:
        if final_hard:
            return "DOCKED_HARD"
        if flag(d, "final_soft_certified", i):
            return "DOCKED_SOFT_CERT"
        return "DOCKED_NONSTRICT"

    if reason == "CAPTURE_TIMEOUT":
        if detail and detail != "NONE":
            return detail
        if soft_seen and final_soft and finite(align) and align > 10.0:
            return "SOFT_ALIGN_TIMEOUT"
        if soft_seen and not final_soft:
            return "ESCAPED_SOFT_CAPTURE"
        if soft_seen and not geom:
            return "BAD_DOCKING_GEOMETRY"
        return "CAPTURE_TIMEOUT_UNCLASSIFIED"

    if reason == "LOST_TARGET_TIMEOUT":
        return "LOST_TARGET_TIMEOUT"
    if "ADCS" in reason:
        return "ADCS_GATE_FAILURE"
    if reason in ("TERMINAL_NOT_REACHED", "DOCK_TIMEOUT"):
        if finite(min_port) and min_port < 0.5:
            return "REACHED_PORT_NO_CAPTURE"
        if finite(nav) and nav > 10.0:
            return "NAV_DIVERGENCE"
        if cam_drop > 300.0:
            return "CAMERA_DROPOUT_DOMINANT"
        if range_drop > 300.0:
            return "RANGE_DROPOUT_DOMINANT"
        return reason

    if finite(final_port) and final_port < 0.3 and finite(final_v) and final_v > 0.05:
        return "CONTACT_TOO_FAST"
    if finite(align_min) and align_min <= 10.0 and finite(align) and align > 30.0:
        return "ALIGNMENT_REGRESSED"
    if finite(cone) and cone > 15.0:
        return "APPROACH_CONE_MISS"
    if finite(lateral) and lateral > 0.3:
        return "LATERAL_MISS"
    return reason or "UNKNOWN"


def soft_capture_diagnosis(entry, best, final, hold_s, chief_omega_dps):
    """
    Classify why a SOFT_ALIGN_TIMEOUT occurred.

    Returns (tag, explanation) where tag is one of:
      DIVERGED        — alignment got WORSE than entry (spin sync counterproductive or disabled)
      STAGNANT        — alignment barely changed from entry (spin sync never engaged)
      SLOW_CONVERGE   — converging but hold time ran out
      NEAR_MISS       — got within 5° of threshold, then diverged or ran out of time
      INSUFFICIENT    — improvement < 30% of what was needed
    """
    if not (finite(entry) and finite(best) and finite(final)):
        return "UNKNOWN", "missing alignment data"

    improvement = entry - best          # positive = got better
    regression  = final - best          # positive = regressed after best point
    needed      = max(entry - 10.0, 0.0)
    hold_s      = max(hold_s, 1.0)

    actual_rate  = improvement / hold_s   # deg/s convergence rate
    needed_rate  = needed / 1200.0        # deg/s needed to certify in max hold time

    if final > entry + 15.0:
        return ("DIVERGED",
                f"align went {entry:.1f}°->{final:.1f}° (WORSE by {final-entry:.1f}°). "
                f"Spin sync disabled or counterproductive. "
                f"chief_omega={chief_omega_dps:.3f}°/s")

    if improvement < 3.0:
        return ("STAGNANT",
                f"align stayed at ~{entry:.1f}° (only {improvement:.1f}° improvement). "
                f"Spin sync likely not engaged or chief omega estimate wrong. "
                f"chief_omega={chief_omega_dps:.3f}°/s")

    if best < 15.0:
        return ("NEAR_MISS",
                f"reached {best:.1f}° (within 5° of 10° threshold) but didn't certify. "
                f"Then regressed to {final:.1f}°. "
                f"Increase SOFT_CAPTURE_MAX_HOLD_S or tighten attitude gains.")

    if actual_rate < needed_rate * 0.5:
        return ("INSUFFICIENT",
                f"converging at {actual_rate*1e3:.2f} mdeg/s but need {needed_rate*1e3:.2f} mdeg/s. "
                f"Rate too slow for 1200s hold. chief_omega={chief_omega_dps:.3f}°/s")

    return ("SLOW_CONVERGE",
            f"entry={entry:.1f}° best={best:.1f}° needed={needed_rate*1e3:.2f}mdeg/s "
            f"actual={actual_rate*1e3:.2f}mdeg/s. "
            f"Needs longer hold or faster convergence.")


def criticality_score(d, i, docked):
    score = 0.0
    if not docked:
        score += 100.0
    min_port = scalar(d, "min_port_range_m", i)
    final_port = scalar(d, "final_port_range_m", i)
    final_v = scalar(d, "final_port_vrel_ms", i)
    align = scalar(d, "final_align_deg", i)
    nav = scalar(d, "max_nav_err_m", i)
    term_dv = scalar(d, "term_dv_ms", i)

    if finite(min_port):
        score += max(0.0, 5.0 - min_port) * 8.0
    if finite(final_port):
        score += max(0.0, 1.0 - final_port) * 12.0
    if finite(final_v):
        score += max(0.0, final_v - 0.01) * 500.0
    if finite(align):
        score += max(0.0, align - 10.0) * 0.8
    if finite(nav):
        score += min(nav, 50.0)
    if finite(term_dv):
        score += min(term_dv, 10.0) * 2.0
    if flag(d, "soft_capture_seen", i) and not docked:
        score += 40.0
    return score


def replay_hint(d, i, master_seed):
    stress = text(d, "stress_case", i, "mixed")
    return (
        f"py -3.11 monte_carlo.py --trials {i + 1} --seed {master_seed} "
        f"--workers 1 --stress-mode mixed    # inspect trial index {i}, stress={stress}"
    )


def print_header(title):
    print()
    print("=" * 118)
    print(f"  {title}")
    print("=" * 118)


def main():
    parser = argparse.ArgumentParser(
        description="Critical diagnostic analyzer for monte_carlo_results.npz"
    )
    parser.add_argument("path", nargs="?", default=DEFAULT_NPZ,
                        help="Path to monte_carlo_results.npz")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of worst/suspicious trials to print")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed used for replay hints")
    parser.add_argument("--all-trials", action="store_true",
                        help="Print full per-trial table")
    args = parser.parse_args()

    path, d = load(args.path)
    if "docked" not in d.files:
        raise KeyError("NPZ does not contain required field 'docked'")

    n = len(d["docked"])
    docked = np.asarray(d["docked"], dtype=bool)
    failed = ~docked
    stress = arr(d, "stress_case", dtype=str)
    reasons = arr(d, "failure_reason", dtype=str)
    signatures = np.array([classify_signature(d, i, docked[i]) for i in range(n)])
    scores = np.array([criticality_score(d, i, docked[i]) for i in range(n)])

    print_header("MC CRITICAL DIAGNOSTIC REPORT")
    print(f"  Source       : {path}")
    print(f"  Trials       : {n}")
    print(f"  Docked       : {int(docked.sum())}/{n} ({pct(int(docked.sum()), n):.1f}%)")
    print(f"  Failed       : {int(failed.sum())}/{n} ({pct(int(failed.sum()), n):.1f}%)")
    print(f"  Fields       : {', '.join(d.files)}")

    # ── Outcome funnel ────────────────────────────────────────────────
    print_header("OUTCOME FUNNEL")
    soft_seen = arr(d, "soft_capture_seen", dtype=bool)
    hard = arr(d, "final_hard_strict", dtype=bool)
    soft_cert = arr(d, "final_soft_certified", dtype=bool)
    cap_timeout = arr(d, "capture_timeout", dtype=bool)
    print(f"  Soft capture seen       : {int(soft_seen.sum()) if soft_seen is not None else '---'}/{n}")
    print(f"  Hard strict final       : {int(hard.sum()) if hard is not None else '---'}/{n}")
    print(f"  Soft certified final    : {int(soft_cert.sum()) if soft_cert is not None else '---'}/{n}")
    print(f"  Capture timeout flag    : {int(cap_timeout.sum()) if cap_timeout is not None else '---'}/{n}")

    # ── Failure reason counts ─────────────────────────────────────────
    print_header("FAILURE REASON COUNTS")
    if reasons is not None:
        for key, count in Counter(reasons[failed]).most_common():
            print(f"  {key:<32} {count:>4}")
    else:
        print("  failure_reason field missing")

    # ── Diagnostic signature counts ───────────────────────────────────
    print_header("DIAGNOSTIC SIGNATURE COUNTS")
    for key, count in Counter(signatures).most_common():
        marker = "FAIL" if key not in ("DOCKED_HARD", "DOCKED_SOFT_CERT", "DOCKED_NONSTRICT") else "PASS"
        print(f"  {marker:<4}  {key:<34} {count:>4}")

    # ── Stress case breakdown ─────────────────────────────────────────
    if stress is not None:
        print_header("STRESS CASE BREAKDOWN")
        for case in sorted(set(stress)):
            idx = stress == case
            nd = int(docked[idx].sum())
            nt = int(idx.sum())
            dominant = Counter(signatures[idx & failed]).most_common(1)
            dom = dominant[0][0] if dominant else "-"
            print(f"  {case:<20} docked={nd:>3}/{nt:<3} ({pct(nd, nt):5.1f}%)  dominant_fail={dom}")

    # ── Numeric health stats ──────────────────────────────────────────
    print_header("NUMERIC HEALTH STATS")
    print_stats("Total DV", arr(d, "total_dv_ms"), " m/s", 3)
    print_stats("PROX DV", arr(d, "prox_dv_ms"), " m/s", 3)
    print_stats("TERMINAL DV", arr(d, "term_dv_ms"), " m/s", 3)
    print_stats("Time to dock", arr(d, "t_dock_hr"), " hr", 3)
    print_stats("Min port range", arr(d, "min_port_range_m"), " m", 4)
    print_stats("Final port range", arr(d, "final_port_range_m"), " m", 4)
    print_stats("Final port vrel", arr(d, "final_port_vrel_ms"), " m/s", 4)
    print_stats("Final align", arr(d, "final_align_deg"), " deg", 2)
    print_stats("Final cone err", arr(d, "final_cone_err_deg"), " deg", 2)
    print_stats("Final lateral", arr(d, "final_lateral_m"), " m", 4)
    print_stats("Max nav err", arr(d, "max_nav_err_m"), " m", 3)
    print_stats("Camera dropout", arr(d, "camera_drop_s"), " s", 1)
    print_stats("Range dropout", arr(d, "range_drop_s"), " s", 1)

    # ── Soft capture phase stats ──────────────────────────────────────
    print_header("SOFT CAPTURE PHASE STATS")
    entry_aligns = arr(d, "soft_capture_align_entry_deg")
    min_aligns   = arr(d, "soft_capture_align_min_deg")
    hold_times   = arr(d, "max_capture_hold_s")
    final_aligns = arr(d, "final_align_deg")

    if entry_aligns is not None:
        print_stats("SC entry align",     entry_aligns,                          " deg", 2)
        print_stats("SC best align",      min_aligns,                            " deg", 2)
        print_stats("SC final align",     final_aligns,                          " deg", 2)
        print_stats("SC hold time",       hold_times,                            " s",   1)

        # Convergence: how much alignment improved from entry to best
        improvement = entry_aligns - min_aligns
        print_stats("SC improvement",     improvement,                           " deg", 2)

        # Delta: did final align get WORSE than entry? (divergence indicator)
        delta_final  = final_aligns - entry_aligns   # positive = diverged, negative = improved
        print_stats("SC delta(final-entry)", delta_final,                        " deg", 2)

        # Convergence rate (deg/s) for trials that had a hold
        with np.errstate(divide='ignore', invalid='ignore'):
            conv_rate = np.where(hold_times > 1, improvement / hold_times, np.nan)
        print_stats("SC conv rate",       conv_rate * 1e3,                       " mdeg/s", 2)

        # Separate stats for failed vs passed
        fail_mask = ~docked
        pass_mask = docked
        if fail_mask.any():
            print()
            print("  --- FAILED trials only ---")
            print_stats("  entry align",     entry_aligns[fail_mask],            " deg", 2)
            print_stats("  best align",      min_aligns[fail_mask],              " deg", 2)
            print_stats("  final align",     final_aligns[fail_mask],            " deg", 2)
            print_stats("  improvement",     improvement[fail_mask],             " deg", 2)
            print_stats("  delta(final-ent)",delta_final[fail_mask],             " deg", 2)
        if pass_mask.any():
            print()
            print("  --- PASSED trials only ---")
            print_stats("  entry align",     entry_aligns[pass_mask],            " deg", 2)
            print_stats("  best align",      min_aligns[pass_mask],              " deg", 2)
            print_stats("  final align",     final_aligns[pass_mask],            " deg", 2)
            print_stats("  improvement",     improvement[pass_mask],             " deg", 2)

    # ── Alignment divergence breakdown ────────────────────────────────
    print_header("ALIGNMENT DIVERGENCE ANALYSIS  (failures only)")
    print(
        f"  {'#':>3}  {'stress':<16} {'entry':>6} {'best':>6} {'final':>6} "
        f"{'improve':>8} {'delta':>7} {'hold_s':>7} {'omega':>7}  diagnosis"
    )
    print("  " + "-" * 115)
    for i in range(n):
        if docked[i]:
            continue
        sig = signatures[i]
        if "SOFT_ALIGN" not in sig and "ESCAPED" not in sig:
            continue
        entry  = scalar(d, "soft_capture_align_entry_deg", i)
        best   = scalar(d, "soft_capture_align_min_deg", i)
        final  = scalar(d, "final_align_deg", i)
        hold   = scalar(d, "max_capture_hold_s", i)
        omega  = scalar(d, "chief_omega_dps", i)

        tag, explanation = soft_capture_diagnosis(entry, best, final, hold, omega)
        impr  = entry - best if finite(entry) and finite(best) else np.nan
        delta = final - entry if finite(final) and finite(entry) else np.nan

        print(
            f"  #{i:>2}  {text(d,'stress_case',i,'?'):<16} "
            f"{fmt(entry,'°',1):>6} {fmt(best,'°',1):>6} {fmt(final,'°',1):>6} "
            f"{fmt(impr,'°',1):>8} {fmt(delta,'°',1):>7} "
            f"{fmt(hold,'s',0):>7} {fmt(omega,'°/s',3):>7}  "
            f"[{tag}] {explanation}"
        )

    # ── Chief tumble rate vs outcome ──────────────────────────────────
    print_header("CHIEF TUMBLE RATE vs OUTCOME")
    omega_arr = arr(d, "chief_omega_dps")
    if omega_arr is not None:
        print_stats("  omega (all)",    omega_arr,             " °/s", 3)
        print_stats("  omega (pass)",   omega_arr[docked],     " °/s", 3)
        print_stats("  omega (fail)",   omega_arr[~docked],    " °/s", 3)
        # Bins
        bins = [(0, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30)]
        print()
        print(f"  {'omega range':<18} {'n':>4} {'docked':>7}  pass%")
        for lo, hi in bins:
            mask = (omega_arr >= lo) & (omega_arr < hi)
            if mask.sum() == 0:
                continue
            nd = int(docked[mask].sum())
            nt = int(mask.sum())
            print(f"  {lo:.2f}–{hi:.2f} °/s       {nt:>4} {nd:>4}/{nt:<3}  {pct(nd,nt):.0f}%")

    # ── Gate margin audit ─────────────────────────────────────────────
    print_header("GATE MARGIN AUDIT")
    gate_specs = [
        ("soft range",  "final_port_range_m",  0.30,  "<=", "m"),
        ("hard range",  "final_port_range_m",  0.08,  "<=", "m"),
        ("hard vrel",   "final_port_vrel_ms",  0.010, "<=", "m/s"),
        ("dock align",  "final_align_deg",     10.0,  "<=", "deg"),
        ("cone err",    "final_cone_err_deg",  15.0,  "<=", "deg"),
        ("lateral",     "final_lateral_m",     0.30,  "<=", "m"),
        ("SC entry align", "soft_capture_align_entry_deg", 30.0, "<=", "deg"),
        ("SC best align",  "soft_capture_align_min_deg",   10.0, "<=", "deg"),
    ]
    for label, key, threshold, op, unit in gate_specs:
        values = arr(d, key)
        if values is None:
            print(f"  {label:<22} missing field {key}")
            continue
        if op == "<=":
            ok = np.isfinite(values) & (values <= threshold)
            margin = threshold - values
        else:
            ok = np.isfinite(values) & (values >= threshold)
            margin = values - threshold
        worst_i = int(np.nanargmin(margin)) if np.isfinite(margin).any() else -1
        print(
            f"  {label:<22} pass={int(ok.sum()):>3}/{n:<3}  "
            f"threshold={threshold:g} {unit:<4}  "
            f"worst=#{worst_i:02d}  margin={fmt(margin[worst_i] if worst_i >= 0 else np.nan, ' '+unit, 4)}"
        )

    # ── Spin sync proxy ───────────────────────────────────────────────
    print_header("SPIN SYNC PROXY  (alignment trajectory classification)")
    print(f"  {'Category':<28} {'count':>5}  {'meaning'}")
    print(f"  {'-'*90}")
    if entry_aligns is not None and final_aligns is not None and min_aligns is not None:
        delta_final = final_aligns - entry_aligns
        improvement = entry_aligns - min_aligns

        # Classify each trial by its alignment trajectory
        cats = {"DIVERGED": 0, "STAGNANT": 0, "CONVERGING": 0, "CERTIFIED": 0}
        for i in range(n):
            if docked[i]:
                cats["CERTIFIED"] += 1
                continue
            e = entry_aligns[i]; b = min_aligns[i]; f = final_aligns[i]
            if not (finite(e) and finite(b) and finite(f)):
                continue
            if f > e + 15.0:
                cats["DIVERGED"] += 1
            elif (e - b) < 3.0:
                cats["STAGNANT"] += 1
            else:
                cats["CONVERGING"] += 1

        print(f"  {'CERTIFIED (docked)':<28} {cats['CERTIFIED']:>5}  alignment reached <=10°")
        print(f"  {'DIVERGED (final > entry+15°)':<28} {cats['DIVERGED']:>5}  spin sync disabled or counterproductive")
        print(f"  {'STAGNANT (<3° improvement)':<28} {cats['STAGNANT']:>5}  spin sync not engaging")
        print(f"  {'CONVERGING (improving, not enough)':<28} {cats['CONVERGING']:>5}  right direction but rate/time insufficient")

    # ── Worst / most actionable ───────────────────────────────────────
    print_header("WORST / MOST ACTIONABLE TRIALS")
    order = np.argsort(-scores)
    printed = 0
    for i in order:
        if printed >= args.top:
            break
        if docked[i] and printed >= max(3, args.top // 3):
            continue
        stress_i = text(d, "stress_case", i, "nominal")
        reason   = text(d, "failure_reason", i, "-")
        detail   = text(d, "capture_timeout_detail", i, "-")
        entry    = scalar(d, "soft_capture_align_entry_deg", i)
        best_a   = scalar(d, "soft_capture_align_min_deg", i)
        final_a  = scalar(d, "final_align_deg", i)
        hold     = scalar(d, "max_capture_hold_s", i)
        omega    = scalar(d, "chief_omega_dps", i)

        tag, explanation = soft_capture_diagnosis(entry, best_a, final_a, hold, omega)

        print(
            f"  #{i:03d} score={scores[i]:6.1f}  "
            f"{'DOCKED' if docked[i] else 'FAIL':<6}  "
            f"stress={stress_i:<16} sig={signatures[i]}"
        )
        print(
            f"       dv={fmt(scalar(d,'total_dv_ms',i),' m/s',3)} "
            f"term_dv={fmt(scalar(d,'term_dv_ms',i),' m/s',3)} "
            f"tdock={fmt(scalar(d,'t_dock_hr',i),' hr',3)} "
            f"nav_err={fmt(scalar(d,'max_nav_err_m',i),' m',1)}"
        )
        print(
            f"       SC: entry={fmt(entry,'°',1)} best={fmt(best_a,'°',1)} "
            f"final={fmt(final_a,'°',1)} hold={fmt(hold,'s',0)} "
            f"chief_omega={fmt(omega,'°/s',3)}"
        )
        if not docked[i]:
            print(f"       diagnosis: [{tag}] {explanation}")
        print(f"       replay: {replay_hint(d, i, args.seed)}")
        printed += 1

    # ── Near misses / suspicious passes ──────────────────────────────
    print_header("NEAR MISSES / SUSPICIOUS PASSES")
    suspicious = []
    for i in range(n):
        if not docked[i]:
            continue
        flags = []
        if scalar(d, "final_port_range_m", i) > 0.08:
            flags.append("loose_final_range")
        if scalar(d, "final_port_vrel_ms", i) > 0.010:
            flags.append("loose_final_vrel")
        if scalar(d, "final_align_deg", i) > 10.0:
            flags.append("loose_align")
        term_dv_arr = arr(d, "term_dv_ms")
        if term_dv_arr is not None and scalar(d, "term_dv_ms", i) > np.nanpercentile(term_dv_arr, 90):
            flags.append("high_terminal_dv")
        if not flag(d, "final_hard_strict", i):
            flags.append("not_hard_strict")
        if flags:
            suspicious.append((scores[i], i, flags))
    for _, i, flags in sorted(suspicious, reverse=True)[: args.top]:
        print(
            f"  #{i:03d} stress={text(d,'stress_case',i,'nominal'):<16} "
            f"flags={','.join(flags):<45} "
            f"dv={fmt(scalar(d,'total_dv_ms',i),' m/s',3)} "
            f"SC_entry={fmt(scalar(d,'soft_capture_align_entry_deg',i),'°',1)} "
            f"SC_best={fmt(scalar(d,'soft_capture_align_min_deg',i),'°',1)} "
            f"final_align={fmt(scalar(d,'final_align_deg',i),'°',1)}"
        )
    if not suspicious:
        print("  No suspicious docked trials under current audit thresholds.")

    # ── Full per-trial table ──────────────────────────────────────────
    if args.all_trials:
        print_header("FULL PER-TRIAL TABLE")
        print(
            f"  {'#':>3} {'res':<5} {'stress':<16} {'sig':<26} "
            f"{'dv':>7} {'term_dv':>7} {'SC_entry':>9} {'SC_best':>8} "
            f"{'SC_final':>9} {'hold_s':>7} {'omega':>7}"
        )
        print("  " + "-" * 110)
        for i in range(n):
            entry = scalar(d, "soft_capture_align_entry_deg", i)
            best  = scalar(d, "soft_capture_align_min_deg", i)
            final = scalar(d, "final_align_deg", i)
            hold  = scalar(d, "max_capture_hold_s", i)
            omega = scalar(d, "chief_omega_dps", i)
            diverged = "DIVE" if (finite(final) and finite(entry) and final > entry + 15.0) else ""
            print(
                f"  {i:>3} {'DOCK' if docked[i] else 'FAIL':<5} "
                f"{text(d,'stress_case',i,'?'):<16} "
                f"{signatures[i]:<26} "
                f"{fmt(scalar(d,'total_dv_ms',i),'',2):>7} "
                f"{fmt(scalar(d,'term_dv_ms',i),'',2):>7} "
                f"{fmt(entry,'°',1):>9} "
                f"{fmt(best,'°',1):>8} "
                f"{fmt(final,'°',1):>9} "
                f"{fmt(hold,'',0):>7} "
                f"{fmt(omega,'',3):>7}  {diverged}"
            )

    print()
    print("=" * 118)


if __name__ == "__main__":
    main()

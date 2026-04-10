# GEO Rendezvous and Proximity Operations (RPOD) Simulation

A full-stack, flight-representative GNC simulation of an autonomous GEO rendezvous and docking mission, implemented entirely in Python with no proprietary dependencies. Core navigation and guidance algorithms are ported to C and verified through a Software-in-the-Loop (SIL) framework.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![C](https://img.shields.io/badge/C-Flight%20Code-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![SIL](https://img.shields.io/badge/SIL-Verified%207nm-brightgreen)

---

## Results Summary

| Metric | Result |
|--------|--------|
| Docking capture | range < 10 cm AND v_rel < 10 mm/s |
| Lambert ΔV budget | < 250 mm/s (typical ~180 mm/s) |
| TH-EKF position error | < 2 m during far-field coast |
| MEKF pointing accuracy | < 0.05° steady-state |
| SIL C vs Python divergence | < 7 nm over 1 full GEO orbit |
| Simulation ceiling | 80,000 s (~22 hr) |

---

## What This Is

A closed-loop GNC simulation covering the complete autonomous rendezvous mission for a 50 kg GEO servicing jetpack. The simulation models the full pipeline from initial spacecraft tumbling through docking:

- **Environment**: Full-force GEO orbit propagator (J2 + SRP), IGRF-13 magnetic field, NRLMSISE-00 atmospheric density, dual-cone eclipse model, gravity gradient torque, differential SRP between chief and deputy
- **Sensors**: ADIS16488A gyroscope (Allan variance), magnetometer, sun sensor, star tracker, ranging and bearing sensor
- **Estimator (attitude)**: QUEST (Wahba's problem, 3-vector) for initialisation, 6-state MEKF with Joseph-form covariance update
- **Estimator (relative nav)**: TH-EKF (Tschauner-Hempel EKF) with CW STM + true anomaly correction for GEO eccentricity
- **Guidance**: Two-impulse Lambert transfer (universal variable method, Stumpff functions), PROX_OPS velocity-profile PD closure, TERMINAL range-proportional deceleration
- **Control**: B-dot detumbling, PD reaction wheel attitude control, cross-product momentum desaturation, formation hold PD
- **FSW**: 10-phase hierarchical state machine (DETUMBLE → SUN_ACQ → FINE_POINTING → FORMATION_HOLD → LAMBERT → COAST → PROX_OPS → TERMINAL → DOCKING)
- **SIL Framework**: TH-EKF, MEKF, and RPOD guidance ported to C with CMSIS-DSP matrix operations, verified to 7 nm numerical parity against Python golden model

---

## Mission Sequence

| Phase | Mode | Exit Condition |
|-------|------|----------------|
| 1 | DETUMBLE | \|ω\| < 3.5 deg/s |
| 2 | SUN_ACQUISITION | Pointing error < 5° |
| 3 | FINE_POINTING | Error < 1°, sustained 100 steps |
| 4 | FORMATION_HOLD | 300 s EKF settle |
| 5 | LAMBERT burn 1 | Burn applied |
| 6 | COAST | t >= t_burn2 (~4 hr arc) |
| 7 | LAMBERT burn 2 | v_rel nulled |
| 8 | PROX_OPS | range < 0.8 m |
| 9 | TERMINAL | range < 10 cm |
| 10 | DOCKING | range < 10 cm AND v_rel < 10 mm/s |

---

## Project Structure

```
flight sim/
│
├── main.py                          # Full mission simulation entry point
│
├── plant/
│   └── spacecraft.py                # Rigid-body dynamics (Euler + quaternion, RK4)
│
├── environment/
│   ├── geo_orbit.py                 # Full-force GEO propagator (J2 + SRP, RK4)
│   ├── cw_dynamics.py               # Clohessy-Wiltshire relative dynamics
│   ├── magnetic_field.py            # IGRF-13 geomagnetic field (degree/order 6)
│   ├── solar_radiation_pressure.py  # SRP torque + dual-cone eclipse model
│   ├── gravity_gradient.py          # Gravity gradient torque
│   └── aerodynamic_drag.py          # NRLMSISE-00 drag (negligible at GEO)
│
├── sensors/
│   ├── gyro.py                      # ADIS16488A model (ARW + BI + RRW)
│   ├── magnetometer.py              # 3-axis mag with Gaussian noise
│   ├── sun_sensor.py                # Coarse sun sensor with eclipse masking
│   ├── star_tracker.py              # Quaternion output with noise
│   └── ranging_sensor.py            # Range + azimuth + elevation sensor
│
├── estimation/
│   ├── quest.py                     # QUEST attitude initialisation
│   ├── mekf.py                      # 6-state MEKF (Joseph form)
│   └── th_ekf.py                    # TH-EKF relative navigation filter
│
├── control/
│   ├── lambert_controller.py        # RPOD guidance + FSW mode machine
│   └── lambert_solver.py            # Universal variable Lambert solver
│
├── actuators/
│   ├── reaction_wheel.py            # RW momentum model + saturation
│   ├── magnetorquer.py              # MTQ torque + desaturation law
│   └── bdot.py                      # B-dot detumble controller
│
├── utils/
│   └── quaternion.py                # Quaternion algebra
│
└── Satellite_GNC/                   # C SIL framework
    ├── src_c/
    │   ├── linalg.h                 # Static matrix math (no malloc)
    │   ├── th_ekf.h / th_ekf.c     # C port of TH-EKF
    │   ├── mekf.h / mekf.c         # C port of MEKF (CMSIS-DSP)
    │   └── rpod_ctrl.h / rpod_ctrl.c  # C port of PROX_OPS + TERMINAL
    ├── sim_python/
    │   ├── wrapper.py               # ctypes bridge (Python → C)
    │   └── verify_sil.py            # SIL numerical parity verification
    ├── tests/
    │   ├── test_thekf.c             # TH-EKF C unit tests
    │   ├── test_mekf.c              # MEKF C unit tests
    │   └── test_rpod.c              # RPOD guidance C unit tests
    └── build.bat                    # Compile + run all tests
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install numpy matplotlib nrlmsise00
```

### 2. Run full mission simulation

```bash
python main.py
```

Produces two matplotlib figures:
- **Figure 1**: Full mission overview — ADCS rates, pointing error, FSW mode timeline, RPOD range, ΔV budget, EKF covariance
- **Figure 2**: Close-approach phase — PROX_OPS and TERMINAL guidance, range vs time, v_close profile

### 3. Build and run C SIL framework (Windows)

```bash
cd Satellite_GNC
$env:Path += ";C:\msys64\mingw64\bin"   # add gcc to PATH
.\build.bat
```

Expected output:
```
Built: gnc_lib.dll
=== TH-EKF C Verification === ... ALL PASS
=== MEKF C Verification    === ... ALL PASS
=== RPOD C Verification    === ... ALL PASS (0 failures)
✓ ALL PASS — C EKF matches Python golden model
   Position divergence: 6.73e-09 m  (threshold 1e-04 m)
```

---

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Chief orbit | a = 42164 km, e = 0.0003, i = 0.8°, 342°E | `main.py` |
| Deputy mass | 50 kg | `main.py` |
| Deputy thrust | 1.0 N → accel_max = 20 mm/s² | `main.py` |
| Deputy A/m | 0.00720 m²/kg | `main.py` |
| Chief A/m | 0.01500 m²/kg | `main.py` |
| Diff. SRP | 53.4 nm/s² (chief − deputy) | Computed |
| Initial standoff | 1000 m trailing (LVLH y = −1000 m) | `main.py` |
| Outer loop rate | 10 Hz (dt = 0.1 s) | `main.py` |
| Inner ADCS rate | 100 Hz (dt = 0.01 s) | `main.py` |
| Formation hold | 300 s EKF settle | `main.py` |
| Lambert ΔV cap | 2.0 m/s | `lambert_controller.py` |
| FAR_FIELD threshold | 500 m (Lambert → PROX_OPS) | `lambert_controller.py` |
| TERMINAL threshold | 0.8 m (PROX_OPS → TERMINAL) | `lambert_controller.py` |
| PROX_OPS time constant | 5 s | `lambert_controller.py` |
| TERMINAL speed gain | k = 0.01 s⁻¹, v_max = 5 mm/s | `lambert_controller.py` |
| Docking capture | range < 10 cm AND v_rel < 10 mm/s | `main.py` |
| Simulation ceiling | 80,000 s (~22 hr) | `main.py` |

---

## Algorithm Notes

### Lambert Solver (universal variable method)
Solves the two-point boundary value problem for a given time-of-flight using the universal variable z and Stumpff functions C(z), S(z). These unify the elliptic, parabolic, and hyperbolic cases into one formulation. Newton-Raphson iteration on the TOF equation converges to the required z. A TOF scan over candidate arcs [1, 2, 4, 6, 16] hr selects the minimum total ΔV solution within the 2 m/s budget cap.

### TH-EKF (Tschauner-Hempel EKF)
6-state relative navigation filter with state x = [δr, δv] in LVLH. Uses the Clohessy-Wiltshire State Transition Matrix with a true-anomaly correction via RK4 integration of dν/dt = h(1 + e·cos ν)²/p² — this corrects CW for the GEO eccentricity e = 0.0003, reducing the orbit-period error from ~42 m to below 1 m. Measurement model: nonlinear h(x) = [range, azimuth, elevation] with analytical Jacobian. Joseph-form update with 50-sigma Mahalanobis gate (far-field) and 5-sigma (nominal).

### MEKF
6-state error-state formulation [dθ, dbias]. Predicts using gyro-corrected angular rate via quaternion kinematics Ω(ω) matrix. Updates from magnetometer, sun sensor, and star tracker using the skew-symmetric Jacobian H = [−S(z_pred) | 0]. Joseph form used for numerical stability. QUEST reseeds the MEKF if attitude error exceeds 25°.

### PROX_OPS Controller
Velocity-profile PD law: desired closing speed is selected from a lookup table based on range (200 mm/s at 500 m stepping down to 3 mm/s at 2 m). Acceleration command: `accel = −(v − v_des)/τ − ω_pos² · pos` with τ = 5 s and ω_pos = 0.5·n for a weak position restoring term. Hard-clamped to 20 mm/s².

### TERMINAL Controller
Range-proportional speed law: `v_des = min(k·range, 5 mm/s)` with k = 0.01 s⁻¹. As range → 0, v_des → 0 — the deputy decelerates to rest at the docking port without an explicit brake command. At 0.8 m: 5 mm/s. At 0.1 m: 1 mm/s.

### SIL Framework
TH-EKF, MEKF, and RPOD guidance ported to C with static memory allocation (no malloc). MEKF uses CMSIS-DSP `arm_mat_mult_f32` on ARM targets; plain C loops on desktop via `−DMEKF_NO_CMSIS`. The `wrapper.py` ctypes bridge injects identical sensor data into both implementations. Verified: position divergence 6.73 nm, velocity divergence 6.6 pm/s, covariance error 4.0e-12 over one full GEO orbit.

---

## Notes on Differential SRP

The chief (A/m = 0.0150) and deputy (A/m = 0.0072) experience different SRP accelerations — a differential of 53.4 nm/s². This creates a natural GEO equilibrium separation point at approximately 307 m. The PROX_OPS and TERMINAL controllers overcome this through their velocity-profile and range-proportional laws. This differential is a physically correct and important characteristic of GEO on-orbit servicing scenarios — not a simulation artefact.

---

## References

- Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*, Springer 2014
- Vallado & McClain, *Fundamentals of Astrodynamics and Applications*, 4th ed., Microcosm Press 2013
- Clohessy & Wiltshire, "Terminal Guidance System for Satellite Rendezvous," *J. Aerosp. Sci.*, 1960
- Yamanaka & Ankersen, "New State Transition Matrix for Relative Motion on an Arbitrary Elliptical Orbit," *JGCD*, 2002
- Battin, *An Introduction to the Mathematics and Methods of Astrodynamics*, AIAA 1999
- Mignard & Farinella, "The theory of satellite orbits," *Celest. Mech.*, 1984
- IGRF-13: Alken et al., *Earth, Planets and Space*, 2021
- NRLMSISE-00: Picone et al., *J. Geophys. Res.*, 2002
- ARM, *CMSIS-DSP Software Library*, developer.arm.com

---

# 3U CubeSat ADCS Simulation

A full-stack, flight-representative Attitude Determination and Control System (ADCS) simulation for a 3U CubeSat, implemented entirely in Python with no proprietary dependencies.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-100%20runs-orange)

---

## Results Summary

| Metric | Result |
|--------|--------|
| QUEST acceptance rate | 99 / 100 runs |
| MEKF convergence rate | 95 / 100 runs |
| Steady-state pointing error | < 0.5° in 95% of runs |
| Wheel saturation events | 0 / 100 runs |
| End-to-end mission success | 95 / 100 runs |

---

## What This Is

A closed-loop GNC simulation covering the full pipeline from orbital mechanics to attitude control:

- **Environment**: SGP4 orbit propagation, IGRF-13 magnetic field, NRLMSISE-00 atmospheric density, SRP with dual-cone eclipse model, gravity gradient torque
- **Sensors**: Gyro (Allan variance noise model), magnetometer, sun sensor — all with hardware-representative noise
- **Estimator**: QUEST (3-vector: mag + sun + nadir) for initialisation, 6-state MEKF with Joseph-form covariance update for fine pointing
- **Control**: B-dot detumbling, PD attitude controller, cross-product momentum desaturation
- **FSW**: 5-mode hierarchical state machine (SAFE → DETUMBLE → SUN_ACQUISITION → FINE_POINTING → MOMENTUM_DUMP)
- **Validation**: 100-run Monte Carlo with randomised tumble, orbit epoch, solar activity, and gyro bias

---

## Project Structure

```
flight sim/
│
├── main.py                    # Single-run simulation entry point
├── monte_carlo.py             # 100-run Monte Carlo validation
├── requirements.txt           # Python dependencies
│
├── plant/
│   └── spacecraft.py          # Rigid body dynamics (Euler equations + quaternion kinematics)
│
├── sensors/
│   ├── gyro.py                # Allan variance gyro model (ARW + BI + RRW)
│   ├── magnetometer.py        # MEMS magnetometer with Gaussian noise
│   └── sun_sensor.py          # Coarse sun sensor array model
│
├── environment/
│   ├── orbit.py               # SGP4/SDP4 orbit propagation
│   ├── magnetic_field.py      # IGRF-13 geomagnetic field
│   ├── sun_model.py           # Sun vector ephemeris
│   ├── aerodynamic_drag.py    # NRLMSISE-00 drag torque
│   ├── gravity_gradient.py    # Gravity gradient torque
│   └── solar_radiation_pressure.py  # SRP torque + eclipse
│
├── estimation/
│   ├── quest.py               # QUEST algorithm (Wahba's problem, 3-vector)
│   └── mekf.py                # 6-state MEKF with Joseph form
│
├── control/
│   └── attitude_controller.py # PD attitude controller
│
├── actuators/
│   ├── reaction_wheel.py      # Reaction wheel momentum model
│   ├── magnetorquer.py        # Magnetorquer torque + desaturation law
│   └── bdot.py                # B-dot detumble controller
│
├── fsw/
│   └── mode_manager.py        # 5-mode FSW state machine
│
├── utils/
│   └── quaternion.py          # Quaternion algebra (multiply, error, rotation matrix)
│
└── telemetry/
    └── logger.py              # Telemetry storage and export
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/cubesat-adcs.git
cd cubesat-adcs
pip install -r requirements.txt
```

### 2. Run single simulation (60 min mission)

```bash
python main.py
```

Produces two matplotlib figures:
- **Figure 1**: Full mission overview — angular rates, reaction wheel momentum, disturbance torques, atmospheric density, FSW mode timeline
- **Figure 2**: MEKF attitude estimation error during fine pointing phase

### 3. Run Monte Carlo validation (100 runs, ~60 min wall time)

```bash
python monte_carlo.py
```

Produces `monte_carlo_results.png` with 8 subplots covering detumble time distribution, QUEST acceptance, MEKF convergence, pointing error CDF, 3-sigma per run, and mode reach distribution.

---

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Inertia matrix | diag(0.030, 0.025, 0.010) kg·m² | `main.py` |
| Outer loop rate | 10 Hz (dt = 0.1s) | `main.py` |
| Inner loop rate | 100 Hz (dt = 0.01s) | `main.py` |
| B-dot gain k_bdot | 2×10⁵ A·m²·s/T | `main.py` |
| PD gains Kp / Kd | 0.0005 / 0.008 | `main.py` |
| Reaction wheel h_max | 4 mN·m·s | `main.py` |
| QUEST quality threshold | 0.01 | `main.py` |
| MEKF Mahalanobis gate | 16.0 (4-sigma) | `mekf.py` |
| Detumble threshold | 3.5 deg/s | `mode_manager.py` |
| SAFE_RATE_THRESHOLD | 40 deg/s | `mode_manager.py` |

---

## Algorithm Notes

### QUEST (QUaternion ESTimator)
Solves Wahba's problem using three reference vectors simultaneously. The K matrix eigenvalue problem is solved via Newton-Raphson iteration. Solution quality assessed from eigenvalue gap — gap < 0.01 triggers gyro bridging fallback. During eclipse, falls back to 2-vector solution (magnetometer + nadir, weights 0.85/0.15).

### MEKF (Multiplicative Extended Kalman Filter)
6-state error state: [dtheta (3), dbias (3)]. Key implementation choices:
- **Joseph form**: `P = (I-KH)P(I-KH)ᵀ + KRKᵀ` — numerically stable for large initial errors
- **Vector normalisation**: all measurement vectors normalised to unit vectors before update
- **QUEST-assisted convergence**: if attitude error > 25°, `ekf.q` is reinitialised from fresh QUEST solution each outer step until filter enters linear regime

### Spacecraft Dynamics
Euler's equation with gyroscopic coupling:
```
I·ω̇ = τ_ext + τ_rw - ω×(I·ω) - ω×h_rw
```
The `ω×h_rw` term is essential for correct desaturation physics — without it, momentum exchange timing is ~15% slower than physical.

---

## Systems Engineering Documentation

The following documents are available in `/docs`:

| Document | Contents |
|----------|----------|
| `ADCS_Requirements_FlowDown.docx` | Mission → system → subsystem requirements with verification methods |
| `ADCS_FMECA.docx` | Failure modes, effects and criticality analysis (8 entries) |
| `ADCS_Technical_Analysis.docx` | Disturbance margins, power budget, verification strategy |
| `CubeSat_ADCS_Brief.docx` | Full technical project brief |

---

## References

- Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*, Springer 2014
- Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.
- IGRF-13: Alken et al., *Earth, Planets and Space*, 2021
- NRLMSISE-00: Picone et al., *Journal of Geophysical Research*, 2002
- ECSS-E-ST-60-30C: Satellite Attitude and Orbit Control System Standard

---

## License

MIT License — free to use, modify, and distribute with attribution.

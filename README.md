# GEO RPOD Python Simulation

Python reference simulation for GEO rendezvous, proximity operations, and docking
of a 50 kg servicing deputy approaching a tumbling GEO chief.

The current repository is an active engineering sandbox. The core Python stack
models attitude dynamics, relative navigation, Lambert transfer, PROX_OPS,
terminal docking-port approach, finite-body checks, soft/hard capture gates,
Monte Carlo campaigns, and post-run visualization.

> Status: research/prototype simulation. This is not flight-certified software.

## Current Phase

The current development phase is integration tuning with high-fidelity toggles
being enabled progressively. The immediate focus is terminal capture stability:
PROX_OPS reaches terminal, but terminal guidance is being debugged around
close-range reversal, port targeting, applied thruster acceleration, keep-out
effects, and navigation velocity estimates.

Current main-run feature state:

| Feature | State |
|---|---|
| Physical thruster layout | On |
| Finite-body collision checks | On |
| Coupled contact dynamics | Off |
| Body-mounted camera FOV gate | Off in `main.py` |
| Keep-out avoidance | On |
| Spin sync | On |
| Terminal handoff override | `MAIN_TERMINAL_M = 10.0 m` |
| Terminal nav filter | `alpha=0.25`, `beta=0.02`, `vmax=0.05 m/s`, `gate=0.25 m` |

Monte Carlo currently has its own toggle set and should be treated as a campaign
harness, not the source of truth for single-run tuning until `main.py` and
`monte_carlo.py` are fully reconciled.

## Quick Start

Use Python 3.11.

```powershell
py -3.11 -m pip install -r Requirements.txt
```

Run one end-to-end simulation:

```powershell
py -3.11 main.py
```

Run a Monte Carlo campaign:

```powershell
py -3.11 monte_carlo.py --trials 20 --workers 8 --seed 42
```

Run the post-run visualizer after `main.py` writes telemetry:

```powershell
py -3.11 visualiser.py
```

Save visual outputs:

```powershell
py -3.11 visualiser.py --save visualiser_dashboard.png --animate --save-animation visualiser_replay.gif
```

Analyze a single-run telemetry file:

```powershell
py -3.11 tools\analyze_rpod_telemetry.py
```

Analyze Monte Carlo results:

```powershell
py -3.11 tools\analyze_mc_results.py
```

## Main Files

| Path | Purpose |
|---|---|
| `main.py` | End-to-end single-run mission simulation and telemetry export |
| `monte_carlo.py` | Parallel Monte Carlo campaign runner and summary generation |
| `visualiser.py` | Post-run dashboard, close-range replay, and Earth-orbit replay |
| `control/lambert_controller.py` | RPOD state machine, Lambert, PROX_OPS, terminal guidance, lost-target handling |
| `control/attitude_controller.py` | Reaction-wheel attitude controller |
| `control/keepout_planner.py` | Keep-out-zone acceleration scaffold |
| `control/spin_sync_controller.py` | Deputy/chief rate matching scaffold |
| `plant/spacecraft.py` | Deputy rigid-body attitude plant |
| `plant/thruster_layout.py` | Physical thruster allocation model |
| `plant/finite_body.py` | Box-body clearance/collision checks |
| `plant/contact_dynamics.py` | Soft-capture/contact surrogate |
| `chief_attitude.py` | Tumbling chief and body-fixed docking port |
| `chief_pose_estimator.py` | Estimated chief pose/rate chain |
| `estimation/th_ekf.py` | Relative translation EKF |
| `estimation/mekf.py` | Attitude MEKF |
| `estimation/terminal_nav_filter.py` | Terminal alpha-beta relative navigation filter |
| `estimation/port_tracker.py` | Gated close-range port tracker |
| `sensors/` | Gyro, sun, magnetometer, star tracker, ranging, camera/FOV sensors |
| `environment/` | GEO orbit, CW dynamics, SRP, drag, magnetic field, gravity gradient, sun model |
| `fsw/mode_manager.py` | High-level ADCS mode manager |
| `tools/` | Analysis/report helper scripts |

## Mission Flow

The simulation starts with attitude acquisition, then enables RPOD after ADCS and
navigation are ready.

```text
DETUMBLE
  -> SUN_ACQUISITION
  -> FINE_POINTING
  -> FORMATION_HOLD
  -> LAMBERT / PROX_OPS
  -> TERMINAL
  -> SOFT_CAPTURE
  -> DOCKING
```

In the Python implementation, RPOD modes are handled by
`control/lambert_controller.py` and the top-level phase glue in `main.py`.

## Current Scenario

| Item | Current value |
|---|---:|
| Chief orbit | GEO, `a = 42164 km`, `e = 0.0003`, `i = 0.8 deg` |
| Chief longitude | `342 deg E` |
| Deputy mass | `50 kg` |
| Deputy thrust | `1 N` |
| Max ideal acceleration | `20 mm/s^2` |
| Initial standoff | `1000 m` trailing |
| Inner ADCS step | `0.01 s` |
| RPOD outer-loop step | `0.1 s` |
| Main terminal handoff | `10 m` |
| Soft-capture gate | `0.30 m`, `0.05 m/s` |
| Hard-capture gate | `0.08 m`, `0.010 m/s`, held for `5 s` |
| Soft-capture entry align gate | `30 deg` |
| Hard docking align gate | `10 deg` |

Chief docking geometry:

```text
DOCK_PORT_BODY = [0, 0, 0.5] m
DOCK_AXIS_BODY = [0, 0, 1]
```

## Guidance And Navigation

The simulation separates plant truth, sensor generation, estimation, and flight
software inputs:

- Plant truth propagates the deputy/chief physical states.
- Sensors generate noisy measurements from truth.
- MEKF estimates deputy attitude.
- TH-EKF estimates relative translation.
- Chief pose estimation provides docking-port axis/rate information.
- RPOD guidance receives estimated state, not the truth state.

Truth is still used for simulation-only scoring, contact/collision checks,
telemetry error calculations, and sensor measurement generation.

## Terminal Debugging

The current code includes extra terminal debug prints to isolate the close-range
failure mode.

Useful log lines:

```text
[TERMDBG ...]
```

Shows terminal guidance intent:

- selected target: port, COM fallback, or sanity fallback
- velocity along the target line
- desired velocity along the target line
- lateral velocity
- acceleration along the target line

```text
[PLANTDBG ...]
```

Shows guidance-versus-plant execution:

- COM and port range
- COM/port closing velocity
- commanded versus applied acceleration
- keep-out contribution
- physical thruster allocation error
- navigation position/velocity error

When debugging terminal failure, capture the logs around the first point where
range reaches roughly `0.3-1.0 m` and then starts increasing again.

## Outputs

Single run outputs:

```text
rpod_telemetry.npz
rpod_telemetry_plots.png
Figure_1.png
Figure_2.png
```

Monte Carlo outputs:

```text
monte_carlo_results.npz
monte_carlo_summary.txt
monte_carlo_plots.png
mc_results/
```

Visualizer outputs are optional and depend on CLI arguments:

```text
visualiser_dashboard.png
visualiser_replay.gif
```

Generated telemetry, plots, and replay artifacts should generally not be
committed unless they are intentionally being archived as a result baseline.

## Monte Carlo CLI

```powershell
py -3.11 monte_carlo.py --trials 300 --workers 8 --seed 42 --stress-mode mixed
```

Arguments:

| Argument | Meaning |
|---|---|
| `--trials` | Number of trials |
| `--workers` | Parallel worker processes |
| `--seed` | Master seed |
| `--outdir` | Output directory |
| `--stress-mode` | `mixed`, `nominal`, or `sweep` |

Current stress cases include nominal and degraded runs such as camera dropout,
gyro bias, high pose noise, range dropout, slow detumble, and weak thruster.

## Development Notes

Recommended workflow:

1. Tune and inspect one run through `main.py`.
2. Review `TERMDBG` / `PLANTDBG` around terminal reversal.
3. Run `tools\analyze_rpod_telemetry.py`.
4. Once a single run is stable, run a 20-trial Monte Carlo.
5. Only then scale to 100 or 300 trials.

Known active work:

- Reconcile duplicated logic between `main.py` and `monte_carlo.py`.
- Stabilize terminal port targeting with physical thrusters enabled.
- Decide when keep-out avoidance should be enabled during terminal tuning.
- Finish estimate-only FSW cleanup by removing remaining truth-assisted seed/reset paths.
- Add replay by seed/trial for failed Monte Carlo cases.
- Add unit/regression tests for contact, thruster allocation, docking geometry, camera dropout, gyro bias, and weak-thruster cases.

## Suggested Git Hygiene

Commit source/config/documentation changes:

```text
*.py
README.md
Requirements.txt
```

Avoid committing generated artifacts unless preserving a named baseline:

```text
rpod_telemetry.npz
*_plots.png
visualiser_*.png
*.gif
monte_carlo_results.npz
mc_results/
```

## Author

Venkat Sainath

MSc Space Engineering

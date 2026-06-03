# GEO RPOD Python Reference Plant

Python reference plant and guidance sandbox for GEO rendezvous, proximity
operations, and docking of a 50 kg servicing deputy approaching a tumbling GEO
chief.

This repository is now organized as:

- a high-fidelity Python plant/reference simulation,
- C-portable flight-software kernels under `fsw/`,
- explicit C-struct-style state specs under `spec/`,
- unit tests that protect the ported logic and plant helpers.

> Status: engineering prototype / reference plant. This is not flight-certified
> software, but the architecture is now set up for an embedded C port.

## Current Baseline

Latest Monte Carlo baseline recorded from `monte_carlo_summary.txt`:

| Metric | Result |
|---|---:|
| Trials | 20 |
| Docking success | 19 / 20 (95.0%) |
| Soft capture reached | 20 / 20 (100.0%) |
| Hard strict certified | 19 / 20 (95.0%) |
| Soft certified | 19 / 20 (95.0%) |
| Capture timeout | 1 / 20 |
| Dominant failure | `SOFT_ALIGN_TIMEOUT` |
| Mean total DV | 3.737 m/s |
| 95th percentile total DV | 5.382 m/s |
| Worst docked DV | 10.861 m/s |
| Mean propellant | 86.5 g |
| 95th percentile propellant | 124.5 g |

Stress-case pass rates from the same baseline:

| Stress case | Pass rate |
|---|---:|
| nominal | 6 / 6 |
| gyro_bias | 6 / 6 |
| camera_dropout | 2 / 2 |
| range_dropout | 1 / 1 |
| weak_thruster | 2 / 2 |
| high_pose_noise | 2 / 3 |

Interpretation: the current product-level issue is no longer reaching contact.
All trials reached soft capture. The remaining failure mode is a soft-capture
attitude convergence timeout in one high-pose-noise-style case.

## Test Status

Current unit-test baseline supplied from local run:

```text
python -m pytest tests/ -v
48 passed
```

Covered areas:

- contact dynamics,
- docking geometry metrics,
- capture-gate classification,
- RPOD guidance kernels,
- RPOD mode-transition sanity,
- physical thruster allocation.

## Architecture

Core layout:

| Path | Purpose |
|---|---|
| `main.py` | Single-run closed-loop simulation using the Python plant |
| `monte_carlo.py` | Monte Carlo campaign runner |
| `sim_config.py` | Shared mission/config constants for `main.py` and MC |
| `spec/rpod_state.py` | C-struct-like guidance state definitions |
| `fsw/rpod_guidance.py` | Pure C-portable PROX/TERMINAL guidance kernels |
| `fsw/capture_gate.py` | Pure C-portable capture classification |
| `fsw/mode_manager.py` | ADCS mode manager |
| `control/lambert_controller.py` | Python RPOD controller wrapper and Lambert logic |
| `utils/docking_metrics.py` | Shared docking geometry/alignment helpers |
| `plant/spacecraft.py` | Rigid-body deputy attitude plant |
| `plant/thruster_layout.py` | Physical bounded thruster allocation |
| `plant/finite_body.py` | Coarse finite-body collision/clearance checks |
| `plant/contact_dynamics.py` | Soft-capture/contact impulse surrogate |
| `estimation/` | MEKF, TH-EKF, port tracker, terminal nav filter |
| `sensors/` | Sensor models and camera/FOV logic |
| `environment/` | GEO orbit, CW dynamics, gravity gradient, SRP, drag, sun, magnetic field |
| `visualiser.py` | Post-run visualization and replay |

The intended embedded flow is:

1. Keep Python as the truth plant and Monte Carlo environment.
2. Port only pure kernels from `fsw/` and explicit structs from `spec/`.
3. Validate C against Python with golden-vector tests.
4. Run the C FSW inside the Python plant for closed-loop verification.

Do not port `main.py` directly. It contains simulation orchestration, truth
state, telemetry, plotting, and debug plumbing.

## Mission Flow

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

RPOD guidance uses estimated navigation state. Plant truth is used for sensor
generation, scoring, contact/collision checks, and telemetry analysis.

## Current Scenario

| Item | Value |
|---|---:|
| Chief orbit | GEO, a = 42164 km, e = 0.0003, i = 0.8 deg |
| Chief longitude | 342 deg E |
| Deputy mass | 50 kg |
| Deputy thrust | 1 N |
| Max acceleration | 20 mm/s^2 |
| Initial standoff | 1000 m trailing |
| Inner ADCS step | 0.01 s |
| RPOD outer-loop step | 0.1 s |
| Main terminal handoff | 10 m |
| Soft-capture gate | 0.30 m, 0.05 m/s |
| Hard-capture gate | 0.08 m, 0.010 m/s, held 5 s |
| Soft-capture entry align gate | 60 deg |
| Hard docking align gate | 10 deg |

Chief docking geometry:

```text
DOCK_PORT_BODY = [0, 0, 0.5] m
DOCK_AXIS_BODY = [0, 0, 1]
```

## Quick Start

Install dependencies:

```powershell
python -m pip install -r Requirements.txt
```

Run one simulation:

```powershell
python main.py
```

Run a 20-trial Monte Carlo:

```powershell
python monte_carlo.py --trials 20 --workers 8 --seed 42
```

Run tests:

```powershell
python -m pytest tests/ -v
```

Run post-run visualizer:

```powershell
python visualiser.py
```

## Outputs

Generated outputs are intentionally not part of the source baseline:

```text
rpod_telemetry.npz
rpod_telemetry_plots.png
Figure_1.png
Figure_2.png
monte_carlo_results.npz
monte_carlo_summary.txt
monte_carlo_plots.png
mc_results/
```

Preserve important campaign results in this README or in a deliberate report,
not by committing raw generated artifacts by accident.

## Development Roadmap

Near-term:

1. Add golden-vector tests for `fsw/rpod_guidance.py` and `fsw/capture_gate.py`.
2. Add exact failed-trial replay to `monte_carlo.py`.
3. Create C headers matching `spec/rpod_state.py`.
4. Port `fsw/rpod_guidance.py` and `fsw/capture_gate.py` to embedded C.
5. Run Python-vs-C comparison tests.
6. Run C FSW inside the Python plant for closed-loop verification.

Product-readiness work:

- clean CLI,
- reproducible result folders,
- archived named baseline configs,
- campaign report generation,
- demo replay/animation,
- final 300-run campaign with all accepted toggles.

## Git Hygiene

Commit source, tests, config, specs, and docs:

```text
*.py
README.md
Requirements.txt
tests/
fsw/
spec/
utils/
```

Do not commit generated telemetry, plots, caches, or local result folders unless
you are intentionally archiving a named baseline.

## Author

Venkat Sainath

MSc Space Engineering

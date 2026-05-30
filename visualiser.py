"""
RPOD post-run visualizer for main.py.

Run:
    py -3.11 main.py
    py -3.11 visualiser.py

main.py writes rpod_telemetry.npz at the end of a run. This visualizer reads
that file and shows the actual end-to-end RPOD timeline, close approach, and
docking-port approach cone used by the current simulation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parent
DEFAULT_TELEMETRY = ROOT / "rpod_telemetry.npz"

RPOD_MODE_LABELS = {
    1: "FORMATION_HOLD",
    2: "LAMBERT",
    3: "PROX_OPS",
    4: "TERMINAL",
    5: "SOFT_CAPTURE",
    6: "DOCKING",
    7: "LOST_TARGET",
}

RPOD_MODE_COLORS = {
    1: "#5b8db8",
    2: "#dd8a28",
    3: "#d75b45",
    4: "#b8223c",
    5: "#c61d83",
    6: "#c8a100",
    7: "#777777",
}

EARTH_RADIUS_KM = 6371.0


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run main.py first so it can write "
            "rpod_telemetry.npz."
        )
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def arr(data: dict[str, np.ndarray], key: str, default: float = np.nan) -> np.ndarray:
    if key in data:
        return np.asarray(data[key])
    n = len(data.get("rn_t", []))
    return np.full(n, default)


def scalar(data: dict[str, np.ndarray], key: str, default):
    value = data.get(key)
    if value is None:
        return default
    return np.asarray(value).item()


def finite_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def add_mode_bands(ax, t: np.ndarray, mode: np.ndarray) -> None:
    if len(t) == 0:
        return
    edges = np.concatenate(([0], np.where(np.diff(mode))[0] + 1, [len(mode)]))
    for start, stop in zip(edges[:-1], edges[1:]):
        m = int(mode[start])
        ax.axvspan(
            t[start] / 3600.0,
            t[stop - 1] / 3600.0,
            color=RPOD_MODE_COLORS.get(m, "#999999"),
            alpha=0.10,
            lw=0,
        )


def colored_path(ax, x: np.ndarray, y: np.ndarray, c: np.ndarray, cmap="viridis"):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=1.7, alpha=0.9)
    lc.set_array(c[:-1])
    ax.add_collection(lc)
    return lc


def cone_polygon(port_y: float, port_x: float, half_angle_deg: float,
                 length_m: float) -> Polygon:
    half_width = math.tan(math.radians(half_angle_deg)) * length_m
    # Main view uses horizontal y and vertical x. The port axis is shown as
    # the +y approach corridor in this 2D projection.
    pts = np.array([
        [port_y, port_x],
        [port_y + length_m, port_x + half_width],
        [port_y + length_m, port_x - half_width],
    ])
    return Polygon(pts, closed=True, facecolor="#4c9f70", alpha=0.14,
                   edgecolor="#2f7d53", lw=1.2)


def draw_chief(ax, half_extents: np.ndarray) -> None:
    hx, hy = float(half_extents[0]), float(half_extents[1])
    ax.add_patch(Rectangle((-hy, -hx), 2 * hy, 2 * hx, facecolor="#e8e8e8",
                           edgecolor="#222222", lw=1.2, zorder=2))
    ax.plot(0.0, 0.0, "k+", ms=8, mew=1.2, label="chief COM", zorder=4)


def set_axes_equal_3d(ax, radius: float) -> None:
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1, 1, 1))


def draw_earth(ax) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(0.0, np.pi, 40)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#3f80c2", alpha=0.55, linewidth=0,
                    shade=True, zorder=0)
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    ax.plot(EARTH_RADIUS_KM * np.cos(theta),
            EARTH_RADIUS_KM * np.sin(theta),
            np.zeros_like(theta), color="white", lw=0.8, alpha=0.7)


def mission_dashboard(data: dict[str, np.ndarray], save_path: Path | None) -> None:
    t = arr(data, "rn_t").astype(float)
    mode = arr(data, "rn_mode").astype(int)
    dx = arr(data, "rn_dx").astype(float)
    dy = arr(data, "rn_dy").astype(float)
    dz = arr(data, "rn_dz").astype(float)
    ex = arr(data, "rn_edx").astype(float)
    ey = arr(data, "rn_edy").astype(float)
    ez = arr(data, "rn_edz").astype(float)
    rng = arr(data, "rn_range").astype(float)
    est_rng = arr(data, "rn_est_range").astype(float)
    dv = arr(data, "rn_dv").astype(float)
    pos_err = arr(data, "rn_pos_err").astype(float)
    port_x = arr(data, "rn_port_dx").astype(float)
    port_y = arr(data, "rn_port_dy").astype(float)
    port_z = arr(data, "rn_port_dz").astype(float)
    port_rng = arr(data, "rn_port_range").astype(float)
    port_vrel = arr(data, "rn_port_vrel").astype(float)
    align = arr(data, "rn_align_deg").astype(float)
    cone_err = arr(data, "rn_cone_error_deg").astype(float)
    lateral = arr(data, "rn_lateral_m").astype(float)

    if len(t) == 0:
        raise ValueError("Telemetry file has no RPOD samples.")

    docked = bool(scalar(data, "docked", False))
    timeout = bool(scalar(data, "capture_timeout", False))
    timeout_detail = str(scalar(data, "capture_timeout_detail", ""))
    cone_half = float(scalar(data, "dock_cone_half_angle_deg", 15.0))
    dock_range = float(scalar(data, "dock_range_m", 0.30))
    hard_range = float(scalar(data, "hard_capture_range_m", 0.08))
    chief_half = np.asarray(data.get("chief_body_half_extents_m", [0.8, 0.8, 0.5]),
                            dtype=float)

    status = "DOCKED" if docked else ("CAPTURE TIMEOUT" if timeout else "NOT DOCKED")
    if timeout_detail and timeout:
        status += f" ({timeout_detail})"

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0])
    fig.suptitle(f"main.py RPOD replay - {status}", fontsize=14, fontweight="bold")

    ax_path = fig.add_subplot(gs[:, 0])
    ax_close = fig.add_subplot(gs[0, 1])
    ax_range = fig.add_subplot(gs[0, 2])
    ax_align = fig.add_subplot(gs[1, 1])
    ax_err = fig.add_subplot(gs[1, 2])

    lc = colored_path(ax_path, dy, dx, t / 3600.0)
    ax_path.plot(dy[0], dx[0], "o", color="#1f8f4d", ms=7, label="start")
    ax_path.plot(dy[-1], dx[-1], "s", color="#b8223c", ms=7, label="end")
    ax_path.plot(port_y, port_x, color="#333333", lw=0.9, alpha=0.55,
                 label="dock port")
    ax_path.plot(0.0, 0.0, "k*", ms=12, label="chief COM")
    ax_path.set_title("End-to-end LVLH trajectory")
    ax_path.set_xlabel("along-track y [m]")
    ax_path.set_ylabel("radial x [m]")
    ax_path.axis("equal")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(fontsize=8, loc="best")
    cbar = fig.colorbar(lc, ax=ax_path, fraction=0.046, pad=0.04)
    cbar.set_label("mission time [hr]")

    close_mask = rng < max(2.0, np.nanmin(rng) + 2.0)
    if not np.any(close_mask):
        close_mask = np.arange(len(t)) > max(0, len(t) - 500)
    draw_chief(ax_close, chief_half)
    final_port_y = float(port_y[np.isfinite(port_y)][-1]) if np.any(np.isfinite(port_y)) else 0.5
    final_port_x = float(port_x[np.isfinite(port_x)][-1]) if np.any(np.isfinite(port_x)) else 0.0
    ax_close.add_patch(cone_polygon(final_port_y, final_port_x, cone_half, 0.75))
    ax_close.add_patch(Circle((final_port_y, final_port_x), dock_range,
                              fill=False, ec="#b8223c", lw=1.3,
                              label="soft-capture range"))
    ax_close.add_patch(Circle((final_port_y, final_port_x), hard_range,
                              fill=False, ec="#c8a100", lw=1.2,
                              ls="--", label="hard-capture range"))
    ax_close.plot(port_y[close_mask], port_x[close_mask], color="#333333",
                  lw=1.0, alpha=0.8, label="port")
    lc2 = colored_path(ax_close, dy[close_mask], dx[close_mask],
                       t[close_mask] / 3600.0, cmap="plasma")
    ax_close.plot(dy[-1], dx[-1], "s", color="#b8223c", ms=7, label="deputy end")
    ax_close.set_title("Close approach with docking cone")
    ax_close.set_xlabel("along-track y [m]")
    ax_close.set_ylabel("radial x [m]")
    ax_close.axis("equal")
    ax_close.set_xlim(final_port_y - 0.8, final_port_y + 1.0)
    ax_close.set_ylim(final_port_x - 0.8, final_port_x + 0.8)
    ax_close.grid(True, alpha=0.3)
    ax_close.legend(fontsize=7, loc="best")

    add_mode_bands(ax_range, t, mode)
    ax_range.semilogy(t / 3600.0, np.maximum(rng, 1e-4),
                      color="#553C9A", label="COM range")
    ax_range.semilogy(t / 3600.0, np.maximum(port_rng, 1e-4),
                      color="#b8223c", label="port range")
    ax_range.semilogy(t / 3600.0, np.maximum(est_rng, 1e-4),
                      color="#2b6cb0", ls="--", alpha=0.8, label="EKF range")
    ax_range.axhline(dock_range, color="#b8223c", ls=":", lw=1)
    ax_range.set_title("Range closure")
    ax_range.set_xlabel("time [hr]")
    ax_range.set_ylabel("range [m]")
    ax_range.legend(fontsize=8)

    add_mode_bands(ax_align, t, mode)
    ax_align.plot(t / 3600.0, align, color="#b8223c", label="axis alignment")
    ax_align.plot(t / 3600.0, cone_err, color="#2f7d53", label="cone error")
    ax_align.axhline(30.0, color="#777777", ls=":", lw=1, label="soft entry")
    ax_align.axhline(10.0, color="#222222", ls="--", lw=1, label="hard align")
    ax_align.set_title("Attitude and approach cone")
    ax_align.set_xlabel("time [hr]")
    ax_align.set_ylabel("deg")
    ax_align.set_ylim(bottom=0)
    ax_align.legend(fontsize=8)

    add_mode_bands(ax_err, t, mode)
    ax_err.plot(t / 3600.0, dv * 1e3, color="#2b6cb0", label="cum DV [mm/s]")
    ax_err.set_title("DV, estimation, contact quality")
    ax_err.set_xlabel("time [hr]")
    ax_err.set_ylabel("DV [mm/s]", color="#2b6cb0")
    ax_err.tick_params(axis="y", labelcolor="#2b6cb0")
    ax2 = ax_err.twinx()
    ax2.semilogy(t / 3600.0, np.maximum(pos_err, 1e-4),
                 color="#b8223c", ls="--", alpha=0.8, label="pos err [m]")
    ax2.semilogy(t / 3600.0, np.maximum(port_vrel * 1e3, 1e-3),
                 color="#555555", ls=":", alpha=0.8, label="port vrel [mm/s]")
    ax2.semilogy(t / 3600.0, np.maximum(lateral, 1e-4),
                 color="#2f7d53", alpha=0.8, label="lateral [m]")
    ax2.set_ylabel("log scale")
    lines, labels = ax_err.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_err.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")

    mode_handles = [
        Rectangle((0, 0), 1, 1, color=color, alpha=0.25,
                  label=RPOD_MODE_LABELS.get(mode_id, str(mode_id)))
        for mode_id, color in RPOD_MODE_COLORS.items()
        if np.any(mode == mode_id)
    ]
    if mode_handles:
        fig.legend(handles=mode_handles, loc="lower center", ncol=len(mode_handles),
                   fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=180)
        print(f"Saved visualizer figure: {save_path}")


def mission_animation(data: dict[str, np.ndarray], save_path: Path | None) -> None:
    t = arr(data, "rn_t").astype(float)
    dx = arr(data, "rn_dx").astype(float)
    dy = arr(data, "rn_dy").astype(float)
    rng = arr(data, "rn_range").astype(float)
    port_x = arr(data, "rn_port_dx").astype(float)
    port_y = arr(data, "rn_port_dy").astype(float)
    align = arr(data, "rn_align_deg").astype(float)
    port_rng = arr(data, "rn_port_range").astype(float)
    cone_half = float(scalar(data, "dock_cone_half_angle_deg", 15.0))
    dock_range = float(scalar(data, "dock_range_m", 0.30))
    chief_half = np.asarray(data.get("chief_body_half_extents_m", [0.8, 0.8, 0.5]),
                            dtype=float)

    finite = np.isfinite(dx) & np.isfinite(dy)
    idx = np.where(finite)[0]
    if len(idx) == 0:
        raise ValueError("No finite trajectory samples for animation.")
    stride = max(1, len(idx) // 900)
    frames = idx[::stride]

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_chief(ax, chief_half)
    final_port_y = float(port_y[np.isfinite(port_y)][-1]) if np.any(np.isfinite(port_y)) else 0.5
    final_port_x = float(port_x[np.isfinite(port_x)][-1]) if np.any(np.isfinite(port_x)) else 0.0
    ax.add_patch(cone_polygon(final_port_y, final_port_x, cone_half, 0.75))
    ax.add_patch(Circle((final_port_y, final_port_x), dock_range, fill=False,
                        ec="#b8223c", lw=1.2))
    trace, = ax.plot([], [], color="#553C9A", lw=1.5)
    deputy, = ax.plot([], [], "o", color="#b8223c", ms=8)
    port_dot, = ax.plot([], [], "s", color="#222222", ms=5)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                  ha="left", fontsize=10,
                  bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax.set_title("Close-range RPOD replay")
    ax.set_xlabel("along-track y [m]")
    ax.set_ylabel("radial x [m]")
    ax.axis("equal")
    ax.set_xlim(final_port_y - 0.9, final_port_y + 1.0)
    ax.set_ylim(final_port_x - 0.9, final_port_x + 0.9)
    ax.grid(True, alpha=0.3)

    def update(frame_index: int):
        i = frames[frame_index]
        trail_start = max(0, i - 1200)
        trace.set_data(dy[trail_start:i + 1], dx[trail_start:i + 1])
        deputy.set_data([dy[i]], [dx[i]])
        port_dot.set_data([port_y[i]], [port_x[i]])
        hud.set_text(
            f"t={t[i] / 3600.0:.2f} hr\n"
            f"COM range={rng[i]:.3f} m\n"
            f"port range={port_rng[i]:.3f} m\n"
            f"align={align[i]:.1f} deg"
        )
        return trace, deputy, port_dot, hud

    anim = FuncAnimation(fig, update, frames=len(frames), interval=35, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow", fps=24)
        print(f"Saved visualizer animation: {save_path}")
    return anim


def orbit_animation(data: dict[str, np.ndarray], save_path: Path | None):
    t = arr(data, "viz_t").astype(float)
    cx = arr(data, "viz_chief_x_km").astype(float)
    cy = arr(data, "viz_chief_y_km").astype(float)
    cz = arr(data, "viz_chief_z_km").astype(float)
    dx = arr(data, "viz_dep_x_km").astype(float)
    dy = arr(data, "viz_dep_y_km").astype(float)
    dz = arr(data, "viz_dep_z_km").astype(float)
    rpod_mode = arr(data, "viz_rpod_mode", default=0).astype(int)

    finite = (np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cz)
              & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz))
    idx = np.where(finite)[0]
    if len(idx) < 2:
        print("No ECI orbit samples in telemetry; run main.py again with the updated exporter.")
        return None

    stride = max(1, len(idx) // 1200)
    frames = idx[::stride]
    orbit_radius = max(
        float(np.nanmax(np.linalg.norm(np.column_stack([cx, cy, cz]), axis=1))),
        EARTH_RADIUS_KM * 1.2,
    )
    view_radius = orbit_radius * 1.12

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("GMAT-style ECI orbit replay from main.py", fontsize=14,
                 fontweight="bold")
    draw_earth(ax)
    ax.plot(cx[idx], cy[idx], cz[idx], color="#c8a100", lw=0.9, alpha=0.35,
            label="chief orbit")
    ax.plot(dx[idx], dy[idx], dz[idx], color="#b8223c", lw=0.8, alpha=0.25,
            label="deputy path")
    chief_trail, = ax.plot([], [], [], color="#ffd34d", lw=2.0)
    deputy_trail, = ax.plot([], [], [], color="#ff4d5e", lw=2.0)
    chief_dot, = ax.plot([], [], [], "o", color="#ffd34d", ms=8,
                         label="chief")
    deputy_dot, = ax.plot([], [], [], "o", color="#ff4d5e", ms=6,
                          label="deputy")
    tether, = ax.plot([], [], [], color="white", lw=1.1, alpha=0.75)
    hud = ax.text2D(0.02, 0.96, "", transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.35,
                              edgecolor="none"),
                    color="white")
    ax.set_xlabel("ECI X [km]")
    ax.set_ylabel("ECI Y [km]")
    ax.set_zlabel("ECI Z [km]")
    set_axes_equal_3d(ax, view_radius)
    ax.view_init(elev=24, azim=38)
    ax.legend(loc="upper right")

    def update(frame_index: int):
        i = frames[frame_index]
        trail_start = max(0, i - 180)
        chief_trail.set_data(cx[trail_start:i + 1], cy[trail_start:i + 1])
        chief_trail.set_3d_properties(cz[trail_start:i + 1])
        deputy_trail.set_data(dx[trail_start:i + 1], dy[trail_start:i + 1])
        deputy_trail.set_3d_properties(dz[trail_start:i + 1])
        chief_dot.set_data([cx[i]], [cy[i]])
        chief_dot.set_3d_properties([cz[i]])
        deputy_dot.set_data([dx[i]], [dy[i]])
        deputy_dot.set_3d_properties([dz[i]])
        tether.set_data([cx[i], dx[i]], [cy[i], dy[i]])
        tether.set_3d_properties([cz[i], dz[i]])
        rel_m = 1000.0 * np.linalg.norm([dx[i] - cx[i], dy[i] - cy[i], dz[i] - cz[i]])
        mode_name = RPOD_MODE_LABELS.get(int(rpod_mode[i]), "ADCS/SETUP")
        hud.set_text(
            f"T+ {t[i] / 3600.0:.2f} hr\n"
            f"mode: {mode_name}\n"
            f"chief radius: {np.linalg.norm([cx[i], cy[i], cz[i]]):.0f} km\n"
            f"separation: {rel_m:.2f} m"
        )
        ax.view_init(elev=24, azim=38 + 0.02 * frame_index)
        return chief_trail, deputy_trail, chief_dot, deputy_dot, tether, hud

    anim = FuncAnimation(fig, update, frames=len(frames), interval=35, blit=False)
    if save_path:
        anim.save(save_path, writer="pillow", fps=24)
        print(f"Saved orbit animation: {save_path}")
    return anim


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize main.py RPOD telemetry.")
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY,
                        help="Path to rpod_telemetry.npz from main.py.")
    parser.add_argument("--save", type=Path, default=None,
                        help="Optional PNG path for the dashboard.")
    parser.add_argument("--animate", action="store_true",
                        help="Also open a close-range replay animation.")
    parser.add_argument("--no-orbit", action="store_true",
                        help="Do not open the GMAT-style Earth orbit replay.")
    parser.add_argument("--save-animation", type=Path, default=None,
                        help="Optional GIF path for --animate.")
    parser.add_argument("--save-orbit-animation", type=Path, default=None,
                        help="Optional GIF path for the Earth orbit replay.")
    args = parser.parse_args()

    data = load_npz(args.telemetry)
    animations = []
    mission_dashboard(data, args.save)
    if not args.no_orbit or args.save_orbit_animation:
        anim = orbit_animation(data, args.save_orbit_animation)
        if anim is not None:
            animations.append(anim)
    if args.animate or args.save_animation:
        anim = mission_animation(data, args.save_animation)
        if anim is not None:
            animations.append(anim)
    plt.show()


if __name__ == "__main__":
    main()

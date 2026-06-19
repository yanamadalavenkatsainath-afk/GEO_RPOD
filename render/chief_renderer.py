"""
Low-poly software rasterizer for the chief satellite.

Renders a synthetic camera image of the chief (bus box + 2 solar-array wings
+ asymmetric retroreflector pattern) from the deputy's camera, using the SAME
camera frame convention and intrinsics as sensors/camera_sensor.py and
chief_pose_estimator.py's PnP solver.

Orientation observability improvements over the original renderer:
  - Asymmetric retroreflector pattern (unique fingerprint per face)
  - Distinct solar-panel front (dark cell side) vs back (bright substrate)
  - Phong specular highlight so the bright spot moves with sun angle

No OpenGL/pyrender dependency -- the geometry is ~30 triangles, so a pure
NumPy z-buffered rasterizer is simpler and avoids driver/context issues under
multiprocessing (Monte Carlo runs many worker processes in parallel).
"""

import numpy as np

from sim_config import CHIEF_BODY_HALF_EXTENTS_M, CHIEF_SOLAR_ARRAY_HALF_SPAN_M

_HX, _HY, _HZ = CHIEF_BODY_HALF_EXTENTS_M
_ARRAY_ROOT_X  = 1.50
_ARRAY_TIP_X   = CHIEF_SOLAR_ARRAY_HALF_SPAN_M
_ARRAY_HALF_W  = 1.0
_MRK           = 0.15   # retroreflector half-size [m]

# Per-face bus albedos break cubic symmetry even before retroreflectors
_ALBEDO = {
    "bus_pz":       0.65,   # +Z top face (dock face, lightest)
    "bus_mz":       0.35,   # -Z bottom face (darkest)
    "bus_px":       0.58,   # +X face
    "bus_mx":       0.48,   # -X face
    "bus_py":       0.62,   # +Y face
    "bus_my":       0.42,   # -Y face (darkest side face)
    "panel_front":  0.20,   # dark solar cell side
    "panel_back":   0.65,   # bright white substrate
    "marker":       0.95,   # dock-face retroreflector (existing)
    "retro":        0.98,   # asymmetric retroreflectors (new)
}
_SPECULAR = {
    "bus_pz":       0.10,
    "bus_mz":       0.10,
    "bus_px":       0.10,
    "bus_mx":       0.10,
    "bus_py":       0.10,
    "bus_my":       0.10,
    "panel_front":  0.05,
    "panel_back":   0.15,
    "marker":       0.80,
    "retro":        0.95,
}
_SHININESS = 32.0


def _quad(p00, p10, p01, p11):
    """Return 4 vertices and 2 triangle indices (local 0-3)."""
    return np.array([p00, p10, p01, p11], dtype=float), \
           np.array([[0,1,2],[1,3,2]])


def _build_mesh():
    """Returns (vertices[N,3], triangles[M,3] int idx, face_materials[M] str)."""
    all_v_list, all_t_list, all_m = [], [], []

    def add(v_block, t_local, material, n_times=1):
        base = sum(len(v) for v in all_v_list)
        all_v_list.append(v_block)
        all_t_list.append(t_local + base)
        all_m.extend([material] * (len(t_local) * n_times // len(t_local)))

    # ── Bus box ──────────────────────────────────────────────────────────
    c = np.array([
        [ _HX,  _HY,  _HZ], [ _HX, -_HY,  _HZ],
        [-_HX,  _HY,  _HZ], [-_HX, -_HY,  _HZ],
        [ _HX,  _HY, -_HZ], [ _HX, -_HY, -_HZ],
        [-_HX,  _HY, -_HZ], [-_HX, -_HY, -_HZ],
    ], dtype=float)
    box_t = np.array([
        [0,1,2],[1,3,2],  # +Z
        [4,6,5],[5,6,7],  # -Z
        [0,4,1],[1,4,5],  # +X
        [2,3,6],[3,7,6],  # -X
        [0,2,4],[2,6,4],  # +Y
        [1,5,3],[3,5,7],  # -Y
    ])
    all_v_list.append(c)
    all_t_list.append(box_t)
    all_m.extend(["bus_pz","bus_pz", "bus_mz","bus_mz",
                   "bus_px","bus_px", "bus_mx","bus_mx",
                   "bus_py","bus_py", "bus_my","bus_my"])

    # ── Solar arrays — front (cell) and back (substrate) ─────────────────
    panel_p = np.array([
        [ _ARRAY_ROOT_X,  _ARRAY_HALF_W, 0.0],  # 0
        [ _ARRAY_ROOT_X, -_ARRAY_HALF_W, 0.0],  # 1
        [ _ARRAY_TIP_X,   _ARRAY_HALF_W, 0.0],  # 2
        [ _ARRAY_TIP_X,  -_ARRAY_HALF_W, 0.0],  # 3
        [-_ARRAY_ROOT_X,  _ARRAY_HALF_W, 0.0],  # 4
        [-_ARRAY_ROOT_X, -_ARRAY_HALF_W, 0.0],  # 5
        [-_ARRAY_TIP_X,   _ARRAY_HALF_W, 0.0],  # 6
        [-_ARRAY_TIP_X,  -_ARRAY_HALF_W, 0.0],  # 7
    ], dtype=float)
    base_p = len(c)
    all_v_list.append(panel_p)
    panel_t = np.array([
        [0,2,1],[1,2,3],  # +X wing front (+Z normal = cell side)
        [0,1,2],[1,3,2],  # +X wing back  (-Z normal = substrate)
        [4,5,6],[5,7,6],  # -X wing front
        [4,6,5],[5,6,7],  # -X wing back
    ]) + base_p
    all_t_list.append(panel_t)
    all_m.extend(["panel_front","panel_front","panel_back","panel_back",
                   "panel_front","panel_front","panel_back","panel_back"])

    dm = _MRK

    # ── Dock-face marker on +Z, offset +Y ────────────────────────────────
    base_mk = sum(len(v) for v in all_v_list)
    mk0_v = np.array([
        [-dm, 0.40-dm, _HZ], [ dm, 0.40-dm, _HZ],
        [-dm, 0.40+dm, _HZ], [ dm, 0.40+dm, _HZ],
    ], dtype=float)
    all_v_list.append(mk0_v)
    all_t_list.append(np.array([[0,1,2],[1,3,2]]) + base_mk)
    all_m.extend(["marker", "marker"])

    # ── Asymmetric retroreflectors (one unique pattern per face) ──────────
    # +X face: 1 retro at (HX, +0.30, +0.20)
    retros = [
        ("retro", np.array([
            [_HX, 0.30-dm, 0.20-dm], [_HX, 0.30+dm, 0.20-dm],
            [_HX, 0.30-dm, 0.20+dm], [_HX, 0.30+dm, 0.20+dm],
        ], dtype=float)),
        # -X face upper retro
        ("retro", np.array([
            [-_HX, -0.20-dm,  0.25-dm], [-_HX, -0.20+dm,  0.25-dm],
            [-_HX, -0.20-dm,  0.25+dm], [-_HX, -0.20+dm,  0.25+dm],
        ], dtype=float)),
        # -X face lower retro (two retros on -X → unique vs +X with one)
        ("retro", np.array([
            [-_HX, -0.20-dm, -0.25-dm], [-_HX, -0.20+dm, -0.25-dm],
            [-_HX, -0.20-dm, -0.25+dm], [-_HX, -0.20+dm, -0.25+dm],
        ], dtype=float)),
        # +Y face: 1 retro offset toward -X/-Z
        ("retro", np.array([
            [-0.40-dm, _HY, -0.15-dm], [-0.40+dm, _HY, -0.15-dm],
            [-0.40-dm, _HY, -0.15+dm], [-0.40+dm, _HY, -0.15+dm],
        ], dtype=float)),
        # -Y face: none → visually different from +Y
    ]
    for mat, rv in retros:
        base_r = sum(len(v) for v in all_v_list)
        all_v_list.append(rv)
        all_t_list.append(np.array([[0,1,2],[1,3,2]]) + base_r)
        all_m.extend([mat, mat])

    all_verts = np.vstack(all_v_list)
    all_tris  = np.vstack(all_t_list)
    all_tris  = _fix_outward_winding(all_verts, all_tris)
    return all_verts, all_tris, all_m


def _fix_outward_winding(verts, tris):
    fixed = tris.copy()
    for idx, (i, j, k) in enumerate(tris):
        p0, p1, p2 = verts[i], verts[j], verts[k]
        normal   = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        if np.dot(normal, centroid) < 0:
            fixed[idx] = [i, k, j]
    return fixed


_VERTS_BODY, _TRIS, _MATERIALS = _build_mesh()


def _rot_matrix(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def _camera_frame(dr_lvlh):
    """Same convention as CameraSensor._camera_frame / ChiefPoseEstimator."""
    r     = np.linalg.norm(dr_lvlh)
    r_hat = dr_lvlh / r
    up    = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(r_hat, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    cam_x = np.cross(up, r_hat);  cam_x /= np.linalg.norm(cam_x)
    cam_y = np.cross(r_hat, cam_x)
    return np.vstack([cam_x, cam_y, r_hat])   # R_l2c


def render_chief(dr_lvlh, q_chief, sun_lvlh, image_size_px=(640, 480), f_px=800.0,
                  ambient=0.12, read_noise_sigma=3.0, rng=None):
    """
    Render a grayscale image of the chief as seen from the deputy's camera.

    Parameters
    ----------
    dr_lvlh        : (3,) chief position relative to deputy, LVLH frame [m]
    q_chief        : (4,) chief body->LVLH quaternion [w,x,y,z]
    sun_lvlh       : (3,) unit sun direction, LVLH frame
    image_size_px  : (W, H)
    f_px           : focal length [px]
    ambient        : ambient light fraction
    read_noise_sigma : Gaussian sensor read noise [0-255 scale]

    Returns
    -------
    img : (H, W) uint8 grayscale image
    """
    rng  = rng if rng is not None else np.random
    W, H = image_size_px
    cx, cy = W / 2.0, H / 2.0

    r_l2c  = _camera_frame(dr_lvlh)
    r_b2l  = _rot_matrix(q_chief)
    r_b2c  = r_l2c @ r_b2l
    t_cam  = r_l2c @ dr_lvlh
    sun_cam = r_l2c @ (sun_lvlh / np.linalg.norm(sun_lvlh))

    verts_cam = (r_b2c @ _VERTS_BODY.T).T + t_cam

    img  = np.zeros((H, W), dtype=np.float64)
    zbuf = np.full((H, W), np.inf)

    for tri_idx, (i0, i1, i2) in enumerate(_TRIS):
        p0, p1, p2 = verts_cam[i0], verts_cam[i1], verts_cam[i2]
        if p0[2] <= 0.01 or p1[2] <= 0.01 or p2[2] <= 0.01:
            continue

        normal = np.cross(p1 - p0, p2 - p0)
        nlen   = np.linalg.norm(normal)
        if nlen < 1e-12:
            continue
        normal /= nlen
        centroid = (p0 + p1 + p2) / 3.0
        if np.dot(normal, centroid) >= 0:
            continue   # back-facing

        u0 = f_px*p0[0]/p0[2]+cx;  v0 = f_px*p0[1]/p0[2]+cy
        u1 = f_px*p1[0]/p1[2]+cx;  v1 = f_px*p1[1]/p1[2]+cy
        u2 = f_px*p2[0]/p2[2]+cx;  v2 = f_px*p2[1]/p2[2]+cy

        x_min = max(int(np.floor(min(u0,u1,u2))), 0)
        x_max = min(int(np.ceil( max(u0,u1,u2))), W-1)
        y_min = max(int(np.floor(min(v0,v1,v2))), 0)
        y_max = min(int(np.ceil( max(v0,v1,v2))), H-1)
        if x_min > x_max or y_min > y_max:
            continue

        xs, ys = np.meshgrid(np.arange(x_min, x_max+1),
                              np.arange(y_min, y_max+1))
        px = xs.astype(np.float64) + 0.5
        py = ys.astype(np.float64) + 0.5

        denom = (u1-u0)*(v2-v0) - (u2-u0)*(v1-v0)
        if abs(denom) < 1e-9:
            continue
        w1 = ((px-u0)*(v2-v0) - (u2-u0)*(py-v0)) / denom
        w2 = ((u1-u0)*(py-v0) - (px-u0)*(v1-v0)) / denom
        w0 = 1.0 - w1 - w2

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(inside):
            continue

        z_interp = w0*p0[2] + w1*p1[2] + w2*p2[2]
        mat      = _MATERIALS[tri_idx]

        # Lambertian diffuse
        diffuse  = max(0.0, float(np.dot(normal, sun_cam)))
        # Phong specular
        view_dir = -centroid / (np.linalg.norm(centroid) + 1e-12)
        reflect  = 2.0 * np.dot(sun_cam, normal) * normal - sun_cam
        specular = max(0.0, float(np.dot(reflect, view_dir))) ** _SHININESS

        intensity = (_ALBEDO[mat] * (ambient + (1-ambient) * diffuse)
                     + _SPECULAR[mat] * specular)
        intensity_255 = np.clip(intensity * 255.0, 0.0, 255.0)

        closer = inside & (z_interp < zbuf[ys, xs])
        if np.any(closer):
            img[ys[closer], xs[closer]]  = intensity_255
            zbuf[ys[closer], xs[closer]] = z_interp[closer]

    if read_noise_sigma > 0:
        img = img + rng.normal(0.0, read_noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

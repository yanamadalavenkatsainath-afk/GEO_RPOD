import numpy as np


class ContactResult:
    def __init__(self, rel_vel_after, impulse_ns, captured, severity,
                 impulse_vec=None, deputy_delta_omega=None,
                 chief_delta_omega=None):
        self.rel_vel_after = rel_vel_after
        self.impulse_ns = float(impulse_ns)
        self.captured = bool(captured)
        self.severity = float(severity)
        self.impulse_vec = (np.zeros(3) if impulse_vec is None
                            else np.asarray(impulse_vec, dtype=float))
        self.deputy_delta_omega = (
            np.zeros(3) if deputy_delta_omega is None
            else np.asarray(deputy_delta_omega, dtype=float))
        self.chief_delta_omega = (
            np.zeros(3) if chief_delta_omega is None
            else np.asarray(chief_delta_omega, dtype=float))


class DockingContactModel:
    """
    Low-order rigid/contact surrogate for soft capture.

    This is simulation validation logic, not flight code. It estimates the
    velocity change and impulse at first contact so the GNC layer can be
    tested against momentum-transfer effects before a high-fidelity contact
    solver is added.
    """

    def __init__(self,
                 restitution=0.10,
                 tangential_damping=0.30,
                 capture_vrel_ms=0.05,
                 max_impulse_ns=3.0):
        self.restitution = float(restitution)
        self.tangential_damping = float(tangential_damping)
        self.capture_vrel_ms = float(capture_vrel_ms)
        self.max_impulse_ns = float(max_impulse_ns)

    def resolve(self, rel_vel, normal_hat, reduced_mass_kg):
        rel_vel = np.asarray(rel_vel, dtype=float)
        n = np.asarray(normal_hat, dtype=float)
        n /= max(np.linalg.norm(n), 1e-12)

        v_n = float(np.dot(rel_vel, n))
        v_t = rel_vel - v_n * n
        rel_after = rel_vel.copy()

        if v_n < 0.0:
            rel_after += -(1.0 + self.restitution) * v_n * n
        rel_after += -self.tangential_damping * v_t

        delta_v = rel_after - rel_vel
        impulse = float(reduced_mass_kg * np.linalg.norm(delta_v))
        severity = impulse / max(self.max_impulse_ns, 1e-9)
        captured = (np.linalg.norm(rel_vel) <= self.capture_vrel_ms
                    and severity <= 1.0)
        return ContactResult(rel_after, impulse, captured, severity,
                             impulse_vec=reduced_mass_kg * delta_v)

    def ideal_latch(self, rel_pos, rel_vel, deputy_mass_kg):
        """
        Idealized soft-capture latch for the baseline controller.

        It removes the residual port-relative position and velocity so the
        translational latch stays active while attitude alignment settles.
        Coupled/bouncy contact behavior remains in resolve_coupled().
        """
        rel_pos = np.asarray(rel_pos, dtype=float)
        rel_vel = np.asarray(rel_vel, dtype=float)
        pos_delta = -rel_pos
        vel_delta = -rel_vel
        impulse = float(deputy_mass_kg * np.linalg.norm(vel_delta))
        severity = impulse / max(self.max_impulse_ns, 1e-9)
        return pos_delta, vel_delta, ContactResult(
            np.zeros(3), impulse, True, severity,
            impulse_vec=deputy_mass_kg * vel_delta)

    def resolve_coupled(self, rel_vel, normal_hat, deputy_mass_kg,
                        chief_mass_kg, deputy_I_body, chief_I_body,
                        r_dep_contact_body, r_chief_contact_body,
                        R_dep_body_to_world, R_chief_body_to_world):
        """
        Contact impulse with linear and angular momentum transfer.

        All velocities and normals are in the world frame used by the caller
        (LVLH in this sim). Contact lever arms and inertia tensors are supplied
        in each spacecraft body frame.
        """
        reduced_mass = (deputy_mass_kg * chief_mass_kg
                        / max(deputy_mass_kg + chief_mass_kg, 1e-9))
        linear = self.resolve(rel_vel, normal_hat, reduced_mass)
        impulse_world = linear.impulse_vec

        R_dep = np.asarray(R_dep_body_to_world, dtype=float)
        R_chief = np.asarray(R_chief_body_to_world, dtype=float)
        r_dep_world = R_dep @ np.asarray(r_dep_contact_body, dtype=float)
        r_chief_world = R_chief @ np.asarray(r_chief_contact_body, dtype=float)

        J_dep_body = R_dep.T @ impulse_world
        J_chief_body = R_chief.T @ (-impulse_world)
        I_dep_inv = np.linalg.inv(np.asarray(deputy_I_body, dtype=float))
        I_chief_inv = np.linalg.inv(np.asarray(chief_I_body, dtype=float))

        d_omega_dep_body = I_dep_inv @ np.cross(r_dep_contact_body, J_dep_body)
        d_omega_chief_body = I_chief_inv @ np.cross(r_chief_contact_body, J_chief_body)

        return ContactResult(
            linear.rel_vel_after,
            linear.impulse_ns,
            linear.captured,
            linear.severity,
            impulse_vec=impulse_world,
            deputy_delta_omega=d_omega_dep_body,
            chief_delta_omega=d_omega_chief_body)

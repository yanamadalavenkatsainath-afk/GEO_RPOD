"""Composable numerical Cartesian propagation and variational transport."""
import numpy as np

from utils.validation import finite_array, finite_scalar, vector


def propagate_cartesian(state, *, time, to_time, acceleration, max_step_s=10., acceleration_jacobian=None,allow_backward=False):
    """RK4 [r,v] and STM with explicit timestamped acceleration provider.

    Optional acceleration_jacobian returns 3x6 d(acceleration)/d[r,v]. Otherwise
    central differences are used, including velocity derivatives (e.g. drag).
    Select time step and force/provider accuracy for the mission under study.
    """
    state = vector(state, "state", size=6)
    dt = to_time.seconds_since(time)
    if not isinstance(allow_backward,bool):raise ValueError('allow_backward must be boolean')
    if dt < 0 and not allow_backward:
        raise ValueError("Propagation cannot move backwards")
    step = finite_scalar(max_step_s, "max_step_s", strictly_positive=True)
    if not callable(acceleration):
        raise ValueError("Acceleration provider is required")
    steps = max(1, int(np.ceil(abs(dt)/step)))
    h = dt/steps
    def rhs(x, phi, stamp):
        a = vector(acceleration(stamp, x[:3].copy(), x[3:].copy()), "acceleration", size=3)
        if acceleration_jacobian is None:
            derivative = np.empty((3, 6))
            for j in range(6):
                scale = max(1., abs(x[j]))*np.cbrt(np.finfo(float).eps)
                perturb = np.eye(6)[j]*scale
                plus, minus = x+perturb, x-perturb
                derivative[:, j] = (vector(acceleration(stamp, plus[:3], plus[3:]), "acceleration", size=3)-
                    vector(acceleration(stamp, minus[:3], minus[3:]), "acceleration", size=3))/(2*scale)
        else:
            derivative = finite_array(acceleration_jacobian(stamp, x[:3], x[3:]), "acceleration_jacobian", shape=(3, 6))
        f = np.zeros((6, 6))
        f[:3, 3:], f[3:] = np.eye(3), derivative
        return np.r_[x[3:], a], f @ phi
    transition = np.eye(6)
    for i in range(steps):
        a, pa = rhs(state, transition, time.shifted(i*h))
        b, pb = rhs(state+h*a/2, transition+h*pa/2, time.shifted((i+.5)*h))
        c, pc = rhs(state+h*b/2, transition+h*pb/2, time.shifted((i+.5)*h))
        d, pd = rhs(state+h*c, transition+h*pc, time.shifted((i+1)*h))
        state += h*(a+2*b+2*c+d)/6
        transition += h*(pa+2*pb+2*pc+pd)/6
    return state, transition

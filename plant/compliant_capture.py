"""Finite compliant fixture loads for two independently propagated rigid bodies.

The model is an engineering spring/damper, not a calibrated bell or docking
mechanism. Integration must resolve the contact natural frequency. No state
projection, zero-velocity reset or automatic rigid attachment is performed.
"""
from dataclasses import dataclass
import numpy as np
from plant.rigid_contact import RigidBodyState
from utils.quaternion import rot_matrix, quat_from_matrix, rotvec_from_quat
from utils.validation import finite_scalar, vector


@dataclass(frozen=True)
class CaptureLoads:
    force_first_reference_n: np.ndarray
    torque_first_body_nm: np.ndarray
    torque_second_body_nm: np.ndarray
    displacement_m: float
    relative_speed_m_s: float
    alignment_rad: float
    stored_energy_j: float
    broken: bool
    demanded_force_n: float = 0.
    demanded_couple_nm: float = 0.


class CompliantCapture:
    def __init__(self, *, first_fixture_body_m, second_fixture_body_m,
                 stiffness_n_m=500., damping_n_s_m=200.,
                 rotational_stiffness_nm_rad=20., rotational_damping_nm_s_rad=10.,
                 maximum_force_n=100., maximum_torque_nm=20., maximum_extension_m=.1):
        self.first_fixture=vector(first_fixture_body_m,'first_fixture_body_m',size=3)
        self.second_fixture=vector(second_fixture_body_m,'second_fixture_body_m',size=3)
        for name,value in [('k',stiffness_n_m),('c',damping_n_s_m),
                           ('kr',rotational_stiffness_nm_rad),('cr',rotational_damping_nm_s_rad),
                           ('force_limit',maximum_force_n),('torque_limit',maximum_torque_nm),
                           ('extension_limit',maximum_extension_m)]:
            setattr(self,name,finite_scalar(value,name,strictly_positive=True))
        self.engaged=False;self.broken=False;self.rest_offset=None;self.rest_rotation=None

    def engage(self,first,second):
        self._compatible(first,second)
        if self.engaged or self.broken:raise ValueError('Capture is already engaged or broken')
        a,b=rot_matrix(first.quaternion),rot_matrix(second.quaternion)
        # Engage at the accepted pose; do not move either body or remove energy.
        self.rest_offset=b.T@(first.position_m+a@self.first_fixture-second.position_m-b@self.second_fixture)
        self.rest_rotation=b.T@a
        self.engaged=True

    def release(self):
        self.engaged=False

    @staticmethod
    def _compatible(first,second):
        if not isinstance(first,RigidBodyState) or not isinstance(second,RigidBodyState):
            raise ValueError('Two rigid body states required')
        if first.time!=second.time or first.reference_frame!=second.reference_frame:
            raise ValueError('Fixture states must share time and reference frame')

    def loads(self,first,second):
        self._compatible(first,second)
        a,b=rot_matrix(first.quaternion),rot_matrix(second.quaternion)
        ra,rb=a@self.first_fixture,b@self.second_fixture
        wa,wb=a@first.omega_body_rad_s,b@second.omega_body_rad_s
        delta=first.position_m+ra-second.position_m-rb
        velocity=first.velocity_m_s+np.cross(wa,ra)-second.velocity_m_s-np.cross(wb,rb)
        if not self.engaged:
            return CaptureLoads(np.zeros(3),np.zeros(3),np.zeros(3),float(np.linalg.norm(delta)),
                float(np.linalg.norm(velocity)),0.,0.,self.broken)
        held=b@self.rest_offset
        error=delta-held
        relative=velocity-np.cross(wb,held)
        angle=rotvec_from_quat(quat_from_matrix(b@self.rest_rotation@a.T))
        force=-self.k*error-self.c*relative
        couple=self.kr*angle+self.cr*(wb-wa)
        # Displaced anchors need an additional reaction couple to conserve
        # total angular momentum while their finite gap is being damped.
        torque_a=a.T@(np.cross(ra,force)+couple)
        torque_b=b.T@(np.cross(rb,-force)-couple-np.cross(delta,force))
        extension=float(np.linalg.norm(error));alignment=float(np.linalg.norm(angle))
        demanded_force=float(np.linalg.norm(force));demanded_couple=float(np.linalg.norm(couple))
        if (demanded_force>self.force_limit or demanded_couple>self.torque_limit
                or extension>self.extension_limit):
            self.engaged=False;self.broken=True
        # The overload sample is reported as a demanded load but not applied:
        # instantaneous failure is the declared mechanism failure model.
        if self.broken:force=np.zeros(3);torque_a=np.zeros(3);torque_b=np.zeros(3)
        return CaptureLoads(force,torque_a,torque_b,extension,float(np.linalg.norm(relative)),
            alignment,.5*self.k*extension**2+.5*self.kr*alignment**2,self.broken,demanded_force,demanded_couple)

"""Calibrated pixel-level landmark camera and terrain-relative pose adapter."""
import numpy as np
from sensors.base import Readout
from estimation.camera_pose import solve_landmark_pose
from utils.validation import finite_scalar,finite_array,rotation_matrix,vector
from utils.quaternion import rot_matrix,quat_from_matrix
from utils.random_streams import random_generator


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


class TerrainLandmarkCamera:
    """Known map correspondences -> noisy pixels -> joint position/attitude.

    Pose is solved from the pixels, not noisy truth position. Landmark IDs and
    map coordinates are assumed supplied by the frontend; this does not claim
    raw-image descriptor matching, map discovery, or planar PnP support.
    Covariance is local pixel-noise linearization conditional on the map and
    calibration; map/calibration errors must be separately configured/evaluated.
    """
    def __init__(self,*,camera_matrix=None,image_size_px=(1024,1024),sigma_px=.5,
                 camera_to_body=None,lever_arm_body_m=(0.,0.,0.),rng=None):
        self.matrix=finite_array(np.array([[700.,0,512.],[0,700.,512.],[0,0,1.]]) if camera_matrix is None else camera_matrix,
            'camera_matrix',shape=(3,3))
        self.size=vector(image_size_px,'image_size_px',size=2)
        if np.any(self.size<=0):raise ValueError('Positive image dimensions required')
        self.sigma=finite_scalar(sigma_px,'sigma_px',strictly_positive=True)
        self.mount=rotation_matrix(np.diag([1.,-1.,-1.]) if camera_to_body is None else camera_to_body)
        self.lever=vector(lever_arm_body_m,'lever_arm_body_m',size=3)
        self.rng=random_generator(rng=rng)
        self.last_pixels=np.empty((0,2));self.last_ids=np.empty(0,dtype=int);self.last_reason='NOT_SAMPLED'

    def observe(self,*,position_m,quaternion_body_to_local,landmarks_local_m,terrain=None):
        points=finite_array(landmarks_local_m,'landmarks_local_m')
        if points.ndim!=2 or points.shape[1]!=3:raise ValueError('Nx3 map required')
        rotation=rot_matrix(quaternion_body_to_local)
        camera_rotation=rotation@self.mount
        origin=vector(position_m,'position_m',size=3)+rotation@self.lever
        relative=(points-origin)@camera_rotation
        visible=relative[:,2]>1e-6
        pixels=np.full((len(points),2),np.nan)
        projected=relative[visible]@self.matrix.T
        pixels[visible]=projected[:,:2]/projected[:,2,None]
        visible&=np.all((pixels>=0)&(pixels<self.size),axis=1)
        if terrain is not None:
            for index in np.flatnonzero(visible):
                distance=np.linalg.norm(points[index]-origin)
                contact=terrain.ray_range(origin,points[index]-origin,maximum_range_m=distance+.01)
                if contact is None or contact<distance-.05:visible[index]=False
        self.last_ids=np.flatnonzero(visible)
        self.last_pixels=pixels[visible]+self.rng.normal(0,self.sigma,(int(sum(visible)),2))
        if sum(visible)<6:return Readout(valid=False,reason='INSUFFICIENT_MAP_FEATURES')
        try:
            pose=solve_landmark_pose(points[visible],self.last_pixels,self.matrix,sigma_px=self.sigma)
        except (ValueError,np.linalg.LinAlgError):
            return Readout(valid=False,reason='MAP_POSE_UNOBSERVABLE_OR_REJECTED')
        camera_to_local=pose.body_to_camera.T
        estimated_rotation=camera_to_local@self.mount.T
        estimated_camera=-camera_to_local@pose.translation_camera_m
        estimated_position=estimated_camera-estimated_rotation@self.lever
        # Solver covariance is [translation_camera, left angle_camera].
        jac=np.zeros((6,6));jac[:3,:3]=-camera_to_local
        jac[:3,3:]=-camera_to_local@skew(pose.translation_camera_m)-camera_to_local@skew(self.mount.T@self.lever)
        jac[3:,3:]=-self.mount
        noise=jac@pose.covariance@jac.T
        return Readout(np.r_[estimated_position,quat_from_matrix(estimated_rotation)],(noise+noise.T)/2)


class LandmarkPoseCamera(TerrainLandmarkCamera):
    """Known target-map features to measured target pose in camera axes."""
    def __init__(self,**parameters):
        parameters.setdefault('camera_to_body',np.eye(3))
        super().__init__(**parameters)

    def observe(self,*,observer_position_m,observer_quaternion,target_position_m,target_quaternion,
                points_body_m,normals_body=None,actual_points_body_m=None):
        points=finite_array(points_body_m,'points_body_m')
        actual=points if actual_points_body_m is None else finite_array(actual_points_body_m,'actual_points_body_m',shape=points.shape)
        if points.ndim!=2 or points.shape[1]!=3:raise ValueError('Nx3 landmark map required')
        observer=vector(observer_position_m,'observer_position_m',size=3)
        target=vector(target_position_m,'target_position_m',size=3)
        observer_rotation=rot_matrix(observer_quaternion);target_rotation=rot_matrix(target_quaternion)
        origin=observer+observer_rotation@self.lever
        camera_rotation=observer_rotation@self.mount
        world=actual@target_rotation.T+target
        relative=(world-origin)@camera_rotation
        visible=relative[:,2]>1e-6
        pixels=np.full((len(points),2),np.nan)
        homogeneous=relative[visible]@self.matrix.T
        pixels[visible]=homogeneous[:,:2]/homogeneous[:,2,None]
        visible&=np.all((pixels>=0)&(pixels<self.size),axis=1)
        if normals_body is not None:
            normals=finite_array(normals_body,'normals_body',shape=points.shape)@target_rotation.T
            visible&=np.einsum('ij,ij->i',normals,origin-world)>0
        self.last_ids=np.flatnonzero(visible)
        self.last_pixels=pixels[visible]+self.rng.normal(0,self.sigma,(int(sum(visible)),2))
        if sum(visible)<6:return Readout(valid=False,reason='INSUFFICIENT_TARGET_FEATURES')
        try:pose=solve_landmark_pose(points[visible],self.last_pixels,self.matrix,sigma_px=self.sigma)
        except (ValueError,np.linalg.LinAlgError):return Readout(valid=False,reason='TARGET_POSE_REJECTED')
        return Readout(np.r_[pose.translation_camera_m,quat_from_matrix(pose.body_to_camera)],pose.covariance)

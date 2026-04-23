"""
Camera Sensor — Real Vision Pipeline
=====================================
Changes from v1:
  ransac_thresh_px: 5.0 -> 8.0  (was rejecting too aggressively)
  MIN_INLIERS:      4   -> 3    (allows measurement with fewer good points)

These values were tuned from MC analysis:
  - 78/300 runs had cam_ok < 1% with old params
  - All 17 failures had cam_ok < 5%
  - Relaxing RANSAC expected to raise cam_ok in low-P_detect runs
  - Still physically meaningful: 3 inliers = minimum for 2D-3D pose solve
"""

import numpy as np

CUBESAT_CORNERS_BODY = np.array([
    [ 0.15,  0.15,  0.30],[ 0.15, -0.15,  0.30],
    [-0.15,  0.15,  0.30],[-0.15, -0.15,  0.30],
    [ 0.15,  0.15, -0.30],[ 0.15, -0.15, -0.30],
    [-0.15,  0.15, -0.30],[-0.15, -0.15, -0.30],
], dtype=float)

MIN_INLIERS = 3   # was 4 — relaxed to allow sparse detection scenarios

def _rot_matrix(q):
    q = q / np.linalg.norm(q); w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

class CameraSensor:
    def __init__(self, focal_length_px=800., image_size_px=(640,480),
                 sigma_px=1.5, P_detect=0.75, P_mismatch=0.10,
                 lambda_fp=1.5, ransac_thresh_px=8.0,   # was 5.0
                 min_range_m=0.05, max_range_m=600., model_points=None):
        self.f=focal_length_px; self.W,self.H=image_size_px
        self.cx=self.W/2.; self.cy=self.H/2.
        self.sigma_px=sigma_px; self.P_detect=P_detect
        self.P_mismatch=P_mismatch; self.lambda_fp=lambda_fp
        self.ransac_thresh_px=ransac_thresh_px
        self.min_range=min_range_m; self.max_range=max_range_m
        self.model_pts=(model_points if model_points is not None
                        else CUBESAT_CORNERS_BODY.copy())
        self.N=len(self.model_pts)
        self._last_R=np.eye(3)*(0.1**2)
        # Track consecutive failures for dropout detection
        self._consecutive_failures = 0

    def measure(self, dr_lvlh, q_chief=None):
        r=float(np.linalg.norm(dr_lvlh))
        if r<self.min_range or r>self.max_range:
            self._consecutive_failures += 1
            return None, self._noise_cov(r)
        R_l2c,R_b2c=self._camera_frame(dr_lvlh,q_chief)
        gt_px,vis_idx=self._project(dr_lvlh,R_l2c,R_b2c)
        if len(vis_idx)<MIN_INLIERS:
            self._consecutive_failures += 1
            return None,self._noise_cov(r)
        det_px,det_3d=self._detect(gt_px,vis_idx)
        det_px,det_3d=self._add_fp(det_px,det_3d)
        det_px,det_3d=self._mismatch(det_px,det_3d)
        if len(det_3d)<MIN_INLIERS:
            self._consecutive_failures += 1
            return None,self._noise_cov(r)
        noisy=det_px+np.random.normal(0,self.sigma_px,det_px.shape)
        in_px,in_3d=self._ransac(noisy,det_3d,R_l2c,r)
        if len(in_3d)<MIN_INLIERS:
            self._consecutive_failures += 1
            return None,self._noise_cov(r)
        t=self._pnp(in_px,np.array(in_3d),r)
        if t is None:
            self._consecutive_failures += 1
            return None,self._noise_cov(r)
        self._consecutive_failures = 0
        pos_lvlh=R_l2c.T@t
        R_meas=self._noise_cov(r); self._last_R=R_meas
        return pos_lvlh,R_meas

    @property
    def is_lost(self, threshold=50):
        """True if camera has failed for > threshold consecutive steps."""
        return self._consecutive_failures > threshold

    @property
    def consecutive_failures(self):
        return self._consecutive_failures

    def _camera_frame(self,dr,q):
        r=np.linalg.norm(dr); rh=dr/r
        up=np.array([0.,0.,1.])
        if abs(np.dot(rh,up))>0.99: up=np.array([0.,1.,0.])
        cx=np.cross(up,rh); cx/=np.linalg.norm(cx)
        cy=np.cross(rh,cx)
        R_l2c=np.vstack([cx,cy,rh])
        Rb2l=_rot_matrix(q) if q is not None else np.eye(3)
        return R_l2c, R_l2c@Rb2l

    def _project(self,dr,R_l2c,R_b2c):
        com_cam=R_l2c@dr
        pts_cam=(R_b2c@self.model_pts.T).T+com_cam
        px=[]; idx=[]
        for i,P in enumerate(pts_cam):
            Z=P[2]
            if Z<=0.01: continue
            u=self.f*P[0]/Z+self.cx; v=self.f*P[1]/Z+self.cy
            if 0<=u<self.W and 0<=v<self.H:
                px.append([u,v]); idx.append(i)
        return np.array(px) if px else np.empty((0,2)), idx

    def _detect(self,gt_px,vis_idx):
        dpx=[]; d3d=[]
        for p,i in zip(gt_px,vis_idx):
            if np.random.random()<self.P_detect:
                dpx.append(p); d3d.append(self.model_pts[i])
        if not dpx: return np.empty((0,2)),[]
        return np.array(dpx),d3d

    def _add_fp(self,dpx,d3d):
        n=np.random.poisson(self.lambda_fp)
        if n==0 or len(d3d)==0: return dpx,d3d
        fp=np.column_stack([np.random.uniform(0,self.W,n),
                            np.random.uniform(0,self.H,n)])
        fp3d=[self.model_pts[np.random.randint(self.N)] for _ in range(n)]
        if len(dpx)==0: return fp,fp3d
        return np.vstack([dpx,fp]), list(d3d)+fp3d

    def _mismatch(self,dpx,d3d):
        if len(d3d)<2: return dpx,d3d
        d3d=list(d3d)
        for i in range(len(d3d)):
            if np.random.random()<self.P_mismatch:
                d3d[i]=self.model_pts[np.random.randint(self.N)]
        return dpx,d3d

    def _ransac(self,px,d3d,R_l2c,r,max_iter=30):
        M=len(d3d)
        if M<MIN_INLIERS: return np.empty((0,2)),[]
        d3d_arr=np.array(d3d); best=[]
        min_sample=max(3,MIN_INLIERS)
        for _ in range(max_iter):
            if M<min_sample: break
            si=np.random.choice(M,min_sample,replace=False)
            t=self._pnp(px[si],d3d_arr[si],r)
            if t is None: continue
            inl=[]
            for i in range(M):
                pt=R_l2c@d3d_arr[i]+t
                if pt[2]<=0: continue
                u=self.f*pt[0]/pt[2]+self.cx; v=self.f*pt[1]/pt[2]+self.cy
                if np.linalg.norm(px[i]-[u,v])<self.ransac_thresh_px:
                    inl.append(i)
            if len(inl)>len(best): best=inl
        if len(best)<MIN_INLIERS: return np.empty((0,2)),[]
        bi=np.array(best)
        return px[bi],[d3d[i] for i in bi]

    def _pnp(self,px,pts3d,r):
        M=len(pts3d)
        if M<MIN_INLIERS: return None
        rays=np.column_stack([(px[:,0]-self.cx)/self.f,
                               (px[:,1]-self.cy)/self.f,
                               np.ones(M)])
        rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        c3=np.mean(pts3d,axis=0); cr=np.mean(rays,axis=0)
        cr/=np.linalg.norm(cr)
        bv=pts3d-c3; cv=rays*r; cv-=np.mean(cv,axis=0)
        W=bv.T@cv
        try:
            U,S,Vt=np.linalg.svd(W)
        except np.linalg.LinAlgError:
            return None
        d=np.linalg.det(Vt.T@U.T)
        R=Vt.T@np.diag([1.,1.,d])@U.T
        return cr*r - R@c3

    def _noise_cov(self,r):
        s=np.clip(self.sigma_px*max(r,self.min_range)/self.f,0.05,5.)
        return np.diag([s**2,s**2,s**2])

    @property
    def sigma_pos_at_100m(self): return self.sigma_px*100./self.f


if __name__=="__main__":
    print("=== Camera Sensor v2 (relaxed RANSAC) Validation ===")
    cam=CameraSensor(ransac_thresh_px=8.0)
    for rng in [10,50,100,200,400]:
        dr=np.array([0.,-float(rng),0.])
        errs=[]; ok=0
        for _ in range(300):
            z,R=cam.measure(dr)
            if z is not None: errs.append(np.linalg.norm(z-dr)); ok+=1
        sr=ok/300; me=float(np.mean(errs)) if errs else float("nan")
        print(f"  {rng:4d}m: success={sr*100:.0f}%  mean_err={me:.3f}m")
    print("Expected: >80% success rate (was 60-70% with old params)")
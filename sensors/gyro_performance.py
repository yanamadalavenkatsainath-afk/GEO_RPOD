"""Configurable gyro error model with explicitly band-limited flicker synthesis.

Parameters describe performance, not space qualification. A sum of independent
OU processes approximates flicker rate noise only between configured time scales.
The single-OU legacy model must not be described as exact 1/f bias instability.
"""
from dataclasses import dataclass
import numpy as np
from utils.random_streams import random_generator
from utils.validation import finite_scalar,vector

@dataclass(frozen=True)
class GyroPerformance:
    arw_deg_sqrt_hr: float=.01
    bias_instability_deg_hr: float=.1
    rrw_deg_hr_sqrt_hr: float=.001
    bias_repeatability_deg_hr: float=1.
    flicker_min_tau_s: float=1.
    flicker_max_tau_s: float=10000.
    flicker_processes: int=16
    scale_factor_ppm: float=0.
    bias_temperature_deg_hr_k: float=0.
    full_scale_deg_s: float=100.
    quantization_deg_s: float=0.
    def __post_init__(self):
        for name in ('arw_deg_sqrt_hr','bias_instability_deg_hr','rrw_deg_hr_sqrt_hr','bias_repeatability_deg_hr','quantization_deg_s'):
            finite_scalar(getattr(self,name),name,minimum=0)
        for name in ('scale_factor_ppm','bias_temperature_deg_hr_k'):finite_scalar(getattr(self,name),name)
        for name in ('flicker_min_tau_s','flicker_max_tau_s','full_scale_deg_s'):finite_scalar(getattr(self,name),name,strictly_positive=True)
        if self.flicker_max_tau_s<=self.flicker_min_tau_s or not isinstance(self.flicker_processes,int) or self.flicker_processes<3:
            raise ValueError('Require ordered flicker timescales and at least three processes')

class PerformanceGyro:
    def __init__(self,dt=.01,*,performance=None,rng=None):
        self.dt=finite_scalar(dt,'dt',strictly_positive=True)
        self.performance=performance or GyroPerformance()
        if not isinstance(self.performance,GyroPerformance):raise ValueError('GyroPerformance required')
        self.rng=random_generator(rng=rng);p=self.performance
        self.tau=np.geomspace(p.flicker_min_tau_s,p.flicker_max_tau_s,p.flicker_processes)
        spacing=np.log(self.tau[1]/self.tau[0])
        # Conventional bias coefficient B corresponds to Allan plateau ~0.664 B.
        self.flicker_sigma=np.deg2rad(p.bias_instability_deg_hr)/3600*.664*np.sqrt(spacing/(2*np.log(2)))
        self.flicker=self.rng.normal(0,self.flicker_sigma,(len(self.tau),3))
        self.bias=self.rng.normal(0,np.deg2rad(p.bias_repeatability_deg_hr)/3600,3)
        self.sigma_arw=np.deg2rad(p.arw_deg_sqrt_hr)/60/np.sqrt(self.dt)
        self.saturated=False
    def measure(self,omega_true,dt=None,temperature_k=293.15):
        interval=self.dt if dt is None else finite_scalar(dt,'dt',strictly_positive=True)
        temperature_k=finite_scalar(temperature_k,'temperature_k',strictly_positive=True)
        rate=vector(omega_true,'omega_true',size=3);p=self.performance
        self.sigma_arw=np.deg2rad(p.arw_deg_sqrt_hr)/60/np.sqrt(interval)
        rho=np.exp(-interval/self.tau)
        self.flicker=rho[:,None]*self.flicker+self.rng.normal(size=self.flicker.shape)*self.flicker_sigma*np.sqrt(-np.expm1(-2*interval/self.tau))[:,None]
        # deg/hr/sqrt(hr) -> rad/s/sqrt(s): divide by 3600*60.
        self.bias+=self.rng.normal(0,np.deg2rad(p.rrw_deg_hr_sqrt_hr)/(3600*60)*np.sqrt(interval),3)
        measured=rate*(1+p.scale_factor_ppm*1e-6)+self.bias+self.flicker.sum(axis=0)
        measured+=np.deg2rad(p.bias_temperature_deg_hr_k)/3600*(temperature_k-293.15)
        measured+=self.rng.normal(0,self.sigma_arw,3)
        limit=np.deg2rad(p.full_scale_deg_s)
        self.saturated=bool(np.any(abs(measured)>limit))
        measured=np.clip(measured,-limit,limit)
        if p.quantization_deg_s:
            quantum=np.deg2rad(p.quantization_deg_s)
            measured=np.clip(np.round(measured/quantum)*quantum,-limit,limit)
        return measured

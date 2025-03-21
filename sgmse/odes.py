"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
from email.policy import default
import warnings

import numpy as np
from sgmse.util.tensors import batch_broadcast
import torch

from sgmse.util.registry import Registry
import os

ODERegistry = Registry("ODE")

ODERegistry = Registry("ODE")
class ODE(abc.ABC):
    """ODE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self):        
        super().__init__()
        

    @property
    @abc.abstractmethod
    def T(self):
        pass
    @abc.abstractmethod
    def ode(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass


    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass


    @abc.abstractmethod
    def copy(self):
        pass



######################여기 밑에 것이 학습할 대상임##############


@ODERegistry.register("flowmatching")
class FLOWMATCHING(ODE):
    #original flow matching
    #Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. International Conference on Learning Representations (ICLR), 2023.
    #mu_t = (1-t)x+ty, sigma_t = (1-t)sigma_min +t
    #t범위 0<t<=1
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma_min", type=float, default=0.00, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma_max",type=float, default=0.5 , help="The maximum sigma to use. 1 by default") 
        return parser

    def __init__(self, sigma_min=0.00, sigma_max =0.5, **ignored_kwargs):
        
        super().__init__()        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    @property
    def T(self):
        return 1
        
    def copy(self):
        return FLOWMATCHING(self.sigma_min,self.sigma_max  )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):

        return (1-t)*self.sigma_min + t*self.sigma_max

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device)) #inference시 사이즈 맞추기 위함
        z = torch.randn_like(y)
        
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def der_mean(self,x0,t,y):
        return y-x0
        
    def der_std(self,t):
        
        return self.sigma_max-self.sigma_min
    
    
    
    
@ODERegistry.register("stochasticinterpolant")
class STOCHASTICINTERPOLANT(ODE):
    #Building normalizing flows with stochastic interpolants.International Conference on Learning Representations
    #mu_t = cos(1/2 pi t) x + sin(1/2 pi t) y, sigma_t = 0
    #t는 0에서 1까지 암거나 다됨 0<=t<=1
    @staticmethod
    def add_argparse_args(parser):        
        return parser

    def __init__(self,  **ignored_kwargs):
        
        super().__init__()        
        
        
    def copy(self):
        return STOCHASTICINTERPOLANT( )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return torch.cos(1/2 * t * torch.pi)[:,None,None,None]*x0 + torch.sin(1/2 * t * torch.pi)[:,None,None,None]*y

    def _std(self, t):

        return 0 + torch.zeros_like(t)

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device)) #inference시 사이즈 맞추기 위함
        z = torch.randn_like(y)
        
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def der_mean(self,x0,t,y):
        return (-torch.sin(1/2 * t * torch.pi)[:,None,None,None]*x0 + torch.cos(1/2 * t * torch.pi)[:,None,None,None]*y)* 1/2 * torch.pi
        
    def der_std(self,t):
        
        return 0.0
    
    
    
@ODERegistry.register("schrodingerBridge")
class SCHRODINGERBRIDGE(ODE):
    #Improving and generalizing flow-based generative models with minibatch optimal transport

    #mu_t = (1-t)x + ty,  sigma_t = sigma \sqrt{t*1(1-t)}
    #0<t<1
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma", type=float, default=0.95, help="The minimum sigma to use. 0.05 by default.")
        return parser

    def __init__(self, sigma, **ignored_kwargs):
        
        super().__init__()        
        self.sigma = sigma
        
    def copy(self):
        return SCHRODINGERBRIDGE(self.sigma, self.T_rev )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return x0 * (1-t)[:,None,None,None] + y * t[:,None,None,None]

    def _std(self, t):

        return self.sigma * torch.sqrt( t *(1-t))

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T_rev*torch.ones((y.shape[0],), device=y.device)) #inference시 사이즈 맞추기 위함
        z = torch.randn_like(y)
        
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def der_mean(self,x0,t,y):
        return y-x0
        
    def der_std(self,t):
        
        return (self.sigma* (1-2*t)/(2* torch.sqrt(t*(1-t))))[:,None,None,None]
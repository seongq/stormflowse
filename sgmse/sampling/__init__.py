# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch
import numpy as np

from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry


__all__ = [
	'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
	'get_sampler'
]


def to_flattened_numpy(x):
	"""Flatten a torch tensor `x` and convert it to numpy."""
	return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
	"""Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
	return torch.from_numpy(x.reshape(shape))




def get_ode_sampler(
	ode, score_fn, y, N,  inverse_scaler=None,
	eps=3e-2, device='cuda', **kwargs
):
	"""Probability flow ODE sampler with the black-box ODE solver.

	Args:
		sde: An `sdes.SDE` object representing the forward SDE.
		score_fn: A function (typically learned model) that predicts the score.
		y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
		inverse_scaler: The inverse data normalizer.
		denoise: If `True`, add one-step denoising to final samples.
		rtol: A `float` number. The relative tolerance level of the ODE solver.
		atol: A `float` number. The absolute tolerance level of the ODE solver.
		method: A `str`. The algorithm used for the black-box ODE solver.
			See the documentation of `scipy.integrate.solve_ivp`.
		eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
		device: PyTorch device.

	Returns:
		A sampling function that returns samples and the number of function evaluations during sampling.
	"""
	# predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
	# rsde = sde.reverse(score_fn, probability_flow=True)

	# def denoise_update_fn(x):
	# 	vec_eps = torch.ones(x.shape[0], device=x.device) * eps
	# 	_, x = predictor.update_fn(x, vec_eps, y)
	# 	return x

	# def drift_fn(x, t, y):
	# 	"""Get the drift function of the reverse-time SDE."""
	# 	return rsde.sde(x, t, y)[0]
	conditioning = kwargs["conditioning"]
	def ode_sampler(z=None, **kwargs):
		"""The probability flow ODE sampler with black-box ODE solver.

		Args:
			model: A score model.
			z: If present, generate samples from latent code `z`.
		Returns:
			samples, number of function evaluations.
		"""
		with torch.no_grad():
			# If not represent, sample the latent code from the prior distibution of the SDE.
			x,_ = ode.prior_sampling(y.shape, y)
			x = x.to(device)

			def ode_func(t, x):
				x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
				# print(type(x))
				vec_t = torch.ones(y.shape[0], device=x.device) * t
				
				drift = score_fn(x, vec_t, conditioning, y)
				drift = drift.cpu()
				# print(type(drift))
				return to_flattened_numpy(drift)

			# Black-box ODE solver for the probability flow ODE
			xt = to_flattened_numpy(x)
			timesteps = torch.linspace(ode.T, eps, N, device=y.device)
			for i in range(len(timesteps)):
				t = timesteps[i]
				if i == len(timesteps)-1:
					dt = 0-t
				else:
					dt = timesteps[i+1]-t	
				# print(type(xt))
				# print(type(dt))
				dt = dt.cpu().numpy()
				xt = xt + dt * ode_func(t,xt) 
			
			nfe = N
			# print(y.shape)
			x = torch.tensor(xt).reshape(y.shape).to(device).type(torch.complex64)

			# Denoising is equivalent to running one predictor step without adding noise
			# if denoise:
			# 	x = denoise_update_fn(x)

			# if inverse_scaler is not None:
			# 	x = inverse_scaler(x)
			return x, nfe

	return ode_sampler

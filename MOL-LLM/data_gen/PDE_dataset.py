import warnings
import numpy as np
import dataclasses

from typing import List, Optional, Set, Callable

from functools import partial
from scipy._lib._util import _asarray_validated


import math as mt
import jax
import numpy as np
import jax.numpy as jnp
from jax import random, jit, nn, lax, vmap, scipy
from functools import partial
import typing
from typing import Any, Union
from einshape.src import abstract_ops
from einshape.src import backend

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import seaborn as sns

import torch
import numpy as np
import scipy.special
from logging import getLogger
from scipy.integrate import solve_ivp
from jax import numpy as jnp

from matplotlib import pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
import jax
import jax.numpy as jnp
from jax import random
from einshape import jax_einshape as einshape
from functools import partial

from .gp_jax import rbf_kernel_jax
from .gp_jax import generate_gaussian_process 

from .utils_datagenNLE import init_multi
from .data_gen_NLE import  diff_react_1D_f, burgers_f
from .fplanck import fokker_planck, boundary, gaussian_pdf, delta_function, uniform_pdf

rtol = 1e-5
atol = 1e-6

############## utils ##########################
def get_sample_range( mean, param_range_gamma=0.1):
        """
        Generate interval for sample parameters
        """
        gamma = param_range_gamma
        half_range = np.abs(mean) * gamma
        return [mean - half_range, mean + half_range]

def round_to_significant_digits(arr, digits=4):
    order_of_magnitude = np.floor(np.log10(np.abs(arr)))
    scale = 10 ** (digits - order_of_magnitude - 1)
    rounded = np.around(arr * scale) / scale
    return rounded

################################################### Method of Lines ##########################################################

##### Heat Equation
def generate_heat(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        """returns ICs_per_equation number of soluions and coefficients"""
        t_range = t_span[1]
        x_range = x_span[1]

        item = {"type": "heat"}
        if coeff is not None:
            c1 = coeff[0]
        else:
            c1 = 3e-3  # 2e-3
            c1_range = get_sample_range(c1)
            c1 = round_to_significant_digits(np.random.uniform(*c1_range, (1,)),4)[0]
            coefficients = [c1]

        tf = tfinals["heat"]
        coeff_t = t_range/tf

        def f_closure(c1):

            def f(t, u):
                d2u_dx2 = np.zeros_like(u)
                dx = x_range / x_num
                # Compute second spatial derivatives using central differences
                for i in range(1, x_num - 1):
                    d2u_dx2[i] = (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2

                # Periodic boundary conditions
                d2u_dx2[0] = (u[-1] - 2 * u[0] + u[1]) / dx**2
                d2u_dx2[-1] = (u[-2] - 2 * u[-1] + u[0]) / dx**2

                du_dt = c1 * d2u_dx2
                return du_dt

            return f

        item["func"] = f_closure(c1)

        num_initial_points = ICs_per_equation
        if ICs is not None:
            y_0s = np.array(ICs)
        elif IC_types == "train":
            y_0s = np.array(
                init_multi(
                    x_grid.flatten(),
                    numbers=num_initial_points * 10,
                    k_tot=4,
                    init_key=np.random.randint(100000),
                    if_norm=True,
                )
            )
        else:
            y_0s = np.array(
                generate_gaussian_process(
                    x_grid.flatten(),
                    init_key=np.random.randint(100000),
                    num=num_initial_points * 10,
                    kernel=rbf_kernel_jax,
                    k_sigma=1,
                    k_l=0.2,
                )
            )
        res = []
        fun = item["func"]
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    [t /coeff_t for t in t_span],
                    y_0,
                    method="BDF",
                    t_eval=t_eval/coeff_t,
                    rtol=rtol,
                    atol=atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                print('exception')
                pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"]=coefficients

        return res, coefficients

##### Porous Medium
def generate_porous_medium(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        item = {"type": "porous_medium"}
        m = np.random.randint(2, 5)
        coefficients = [m]

        tf = tfinals["porous_medium"]
        coeff = t_range / tf
        #
        def f_closure(m):

            def f(t, u):
                d2um_dx2 = np.zeros_like(u)
                dx = x_range / x_num
                um = np.power(u, m)
                # Compute second spatial derivatives using central differences
                for i in range(1, x_num - 1):
                    d2um_dx2[i] = (um[i - 1] - 2 * um[i] + um[i + 1]) / dx**2

                # Periodic boundary conditions
                d2um_dx2[0] = (um[-1] - 2 * um[0] + um[1]) / dx**2
                d2um_dx2[-1] = (um[-2] - 2 * um[-1] + um[0]) / dx**2

                du_dt = d2um_dx2
                return du_dt

            return f

        item["func"] = f_closure(m)

        # ODE solve
        num_initial_points = ICs_per_equation
        res = []
        fun = item["func"]

        for i in range(num_initial_points * 10):

            if IC_types == "train":
                A_range = [9, 11]
                center_range = [0.9, 1.1]
                std_range = [0.9, 1.1]
                A = round_to_significant_digits(np.random.uniform(*A_range, (1,)))[0]
                center = round_to_significant_digits(np.random.uniform(*center_range, (1,)))[0]
                std = round_to_significant_digits(np.random.uniform(*std_range, (1,)))[0]
                y_0 = np.exp(-A * ((x_grid.flatten() - center) ** 2) / (2 * std**2))
                slope = (y_0[-1] - y_0[0]) / x_range
                y_0 -= slope * x_grid.flatten()
                y_0 = (y_0 - np.min(y_0)) / (np.max(y_0) - np.min(y_0))
            else:
                A_range = [5, 15]
                center_range = [0.9, 1.1]
                std_range = [0.3, 0.5]
                A = round_to_significant_digits(np.random.uniform(*A_range, (1,)))[0]
                center = round_to_significant_digits(np.random.uniform(*center_range, (1,)))[0]
                std = round_to_significant_digits(np.random.uniform(*std_range, (1,)))[0]
                y_0 = np.maximum(-A * (x_grid.flatten() - center) ** 2 / (2 * std**2) + A, 0)
                slope = (y_0[-1] - y_0[0]) / x_range
                y_0 -= slope * x_grid.flatten()
                y_0 = (y_0 - np.min(y_0)) / (np.max(y_0) - np.min(y_0))

            # y_0 = y_0s[i,:]
            try:
                sol = solve_ivp(
                    fun,
                    [t / coeff for t in t_span],
                    y_0,
                    method="RK45",
                    t_eval=t_eval / coeff,
                    rtol=rtol,
                    atol=atol,
                    max_step=0.0001,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                print(f"An error occurred: {e}")
                pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"]=coefficients

        return res, coefficients

##### Kelin Gordon
def generate_Klein_Gordon(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        c = 1
        m = 0.1

        c_range = get_sample_range(c)
        m_range = get_sample_range(m)
        item = {"type": "Klein_Gordon"}

        c = round_to_significant_digits(np.random.uniform(*c_range, (1,)))[0]
        m = round_to_significant_digits(np.random.uniform(*m_range, (1,)))[0]
        coefficients = [c,m]


        tf = tfinals["Klein_Gordon"]
        coeff = t_range / tf

        dt_this = dt / (100 * coeff)
        alpha = (c * dt_this / dx) ** 2
        beta = m**2 * c**4 * dt_this**2

        def update(u, upr, t_curr, t_save):
            while t_curr < t_save:
                unew = 2 * (1 - alpha) * u - upr + alpha * (np.roll(u, -1) + np.roll(u, 1)) - beta * u  # t+1
                upr = u
                u = unew
                t_curr = t_curr + dt_this
            return u, upr, t_curr

        num_initial_points = ICs_per_equation

        res = []

        if ICs is not None:
            y_0s = np.array(ICs)
        elif IC_types == "train":
            y_0s = np.array(
                init_multi(
                    x_grid.flatten(),
                    numbers=num_initial_points * 10,
                    k_tot=2,
                    num_choise_k=1,
                    init_key=np.random.randint(100000),
                    if_norm=True,
                )
            )
        else:
            y_0s = np.array(
                generate_gaussian_process(
                    x_grid.flatten(),
                    init_key=np.random.randint(100000),
                    num=num_initial_points * 10,
                    kernel=rbf_kernel_jax,
                    k_sigma=1,
                    k_l=0.2,
                    if_norm=True,
                )
            )

        for i in range(num_initial_points * 10):

            psi_t0 = np.zeros(x_grid.flatten().shape)
            try:
                psi_0 = y_0s[i, :]
                y = [psi_0]
                upr = psi_0
                u = psi_0 + dt_this * psi_t0
                t_current = dt_this

                for n in range(1, t_num):
                    t_save = t_eval[n] / coeff
                    u, upr, t_current = update(u, upr, t_current, t_save)
                    y.append(u)
                y = np.array(y)
                res.append(torch.from_numpy(y.astype(np.single)))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients

        return res, coefficients

##### Sine Gordon
def generate_Sine_Gordon(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        item = {"type": "Sine_Gordon"}

        tf = tfinals["Sine_Gordon"]
        coeff = t_range / tf
        # if extrapolate_pdetypes:
        #     c = 3e-3
        # else:
        #
        c = 1
        c_range = get_sample_range(c)
        c = round_to_significant_digits(np.random.uniform(*c_range, (1,)))[0]
        coefficients = [c]

        dt_this = dt / (coeff * 100)

        def update(u, upr, t_curr, t_save):
            while t_curr < t_save:
                alpha = (dt_this**2) / (dx**2)
                u_new = 2 * u - upr + alpha * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) - (dt_this**2) * c * np.sin(u)
                upr = u
                u = u_new
                t_curr = t_curr + dt_this
            return u, upr, t_curr

        num_initial_points = ICs_per_equation

        res = []
        if ICs is not None:
            y_0s = np.array(ICs)
        elif IC_types == "train":
            y_0s = np.array(
                init_multi(
                    x_grid.flatten(),
                    numbers=num_initial_points * 10,
                    k_tot=2,
                    num_choise_k=1,
                    init_key=np.random.randint(100000),
                    if_norm=True,
                )
            )
        else:
            y_0s = np.array(
                generate_gaussian_process(
                    x_grid.flatten(),
                    init_key=np.random.randint(100000),
                    num=num_initial_points * 10,
                    kernel=rbf_kernel_jax,
                    k_sigma=1,
                    k_l=0.2,
                    if_norm=True,
                )
            )

        for i in range(num_initial_points * 10):
            psi_t0 = np.zeros(x_grid.flatten().shape)
            try:
                psi_0 = y_0s[i, :]
                y = [psi_0]
                upr = psi_0
                u = psi_0 + dt_this * psi_t0
                t_current = dt_this
                for n in range(1, t_num):
                    t_save = t_eval[n] / coeff
                    u, upr, t_current = update(u, upr, t_current, t_save)
                    y.append(u)
                y = np.array(y)
                res.append(torch.from_numpy(y.astype(np.single)))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] =coefficients

        return res,coefficients

##### Cahn Hillaiard 1D
def generate_cahnhilliard_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        eps = 0.01
        tf = tfinals["cahnhilliard_1D"]
        coeff = t_range / tf
        eps_range = get_sample_range(eps)

        item = {"type": "cahnhilliard_1D"}

        eps = round_to_significant_digits(np.random.uniform(*eps_range, (1,)))[0]
        coefficients = [eps]
        #
        def f_closure(eps):

            def f(t, u):
                d2u_dx2 = np.zeros_like(u)
                rhs = np.zeros_like(u)
                dx = x_range / x_num
                # Compute second spatial derivatives using central differences
                for i in range(1, x_num - 1):
                    d2u_dx2[i] = (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2

                # Periodic boundary conditions
                d2u_dx2[0] = (u[-1] - 2 * u[0] + u[1]) / dx**2
                d2u_dx2[-1] = (u[-2] - 2 * u[-1] + u[0]) / dx**2

                f = u**3 - u
                fu = 3 * u**2 - 1

                d2u_dx2af = eps**2 * d2u_dx2 + fu

                for i in range(1, x_num - 1):
                    rhs[i] = (d2u_dx2af[i - 1] - 2 * d2u_dx2af[i] + d2u_dx2af[i + 1]) / dx**2

                # Periodic boundary conditions
                rhs[0] = (d2u_dx2af[-1] - 2 * d2u_dx2af[0] + d2u_dx2af[1]) / dx**2
                rhs[-1] = (d2u_dx2af[-2] - 2 * d2u_dx2af[-1] + d2u_dx2af[0]) / dx**2

                du_dt = -d2u_dx2af
                return du_dt

            return f

        item["func"] = f_closure(eps)

        # ODE solve
        num_initial_points = ICs_per_equation
        if ICs is not None:
            y_0s = np.array(ICs)
        elif IC_types == "train":
            y_0s = np.array(
                init_multi(
                    x_grid.flatten(),
                    numbers=num_initial_points * 10,
                    k_tot=2,
                    init_key=np.random.randint(100000),
                    if_norm=True,
                )
            )
        else:
            y_0s = np.array(
                generate_gaussian_process(
                    x_grid.flatten(),
                    init_key=np.random.randint(100000),
                    num=num_initial_points * 10,
                    kernel=rbf_kernel_jax,
                    k_sigma=1,
                    k_l=0.2,
                )
            )
        res = []
        fun = item["func"]
        for i in range(num_initial_points * 10):
            y_0 = y_0s[i, :]
            try:
                sol = solve_ivp(
                    fun,
                    [t / coeff for t in t_span],
                    y_0,
                    method="BDF",
                    t_eval=t_eval / coeff,
                    rtol=rtol,
                    atol=atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    res.append(torch.from_numpy(sol.y.transpose().astype(np.single)))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item['coefficients']= coefficients

        return res, coefficients

################################################### Fourier Spectral Method ##########################################################

##### Korteweg–De Vries
def generate_kdv(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        delta = 0.022
        delta_range = get_sample_range(delta)

        item = {"type": "kdv"}

        delta = round_to_significant_digits(np.random.uniform(*delta_range, (1,)))[0]
        delta2 = delta**2
        coefficients = [delta2]
        tf = tfinals["kdv"]
        coeff = t_range / tf

        # Assuming nx is even for simplicity
        kx = np.fft.fftfreq(x_num, d=x_range / x_num)
        kx = 2.0 * np.pi * kx  # Scale by 2*pi for spatial frequency

        def uhat2vhat(t, uhat):
            return np.exp(-1j * (kx**3) * delta2 * t) * uhat

        def vhat2uhat(t, vhat):
            return np.exp(1j * (kx**3) * delta2 * t) * vhat

        # ----- Define RHS -----
        def uhatprime(t, uhat):
            u = np.fft.ifft(uhat)
            return 1j * (kx**2) * delta2 * uhat - 0.5j * kx * np.fft.fft(u**2)

        def vhatprime(t, vhat):
            u = np.fft.ifft(vhat2uhat(t, vhat))
            return -0.5j * kx * uhat2vhat(t, np.fft.fft(u**2))

        # u0 = np.cos(np.pi * x_grid)

        num_initial_points = ICs_per_equation

        if ICs is not None:
            y_0s = np.array(ICs)

        res = []

        for i in range(num_initial_points * 10):
            if ICs is not None:
                u0 = y_0s[i,:]
            elif IC_types ==  1:
                base_frequency = 2 * np.pi / x_range
                n1, n2 = np.random.randint(1, 3, size=2)
                frequencies = base_frequency * np.array([n1, n2])

                random_phases = np.random.uniform(0, 2 * np.pi, size=2)
                random_amplitudes = np.random.uniform(0, 1, size=2)

                # Composite wave function
                def _func(x):
                    # return random_amplitudes[0] * np.sin(
                    #     base_frequency * x + random_phases[0])
                    wave1 = random_amplitudes[0] * np.sin(frequencies[0] * x + random_phases[0])
                    wave2 = random_amplitudes[1] * np.sin(frequencies[1] * x + random_phases[1])
                    return wave1 + wave2

                vec = _func(x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= x_range
                vec = vec - slope * x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (2 * (max - min))
                    return val

                u0 = func(x_grid.flatten())
            else:
                means = np.random.uniform(0.5, 1.5, size=2)
                # means2 = np.random.uniform(1, 1.5)
                std_devs = np.random.uniform(0.3, 0.5, size=2)  # Random standard deviations
                sign = np.random.randint(2, size=2) * 2 - 1  # Random standard deviations

                # Define the composite Gaussian function
                def _func(x):
                    gaussian1 = np.exp(-((x - means[0]) ** 2) / (2 * std_devs[0] ** 2))
                    gaussian2 = np.exp(-((x - means[1]) ** 2) / (2 * std_devs[1] ** 2))

                    return sign[0] * gaussian1 + sign[1] * gaussian2

                vec = _func(x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= x_range
                vec = vec - slope * x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (2 * (max - min))
                    return val

                u0 = func(x_grid.flatten())
            uhat0 = np.fft.fft(u0)
            vhat0 = uhat2vhat(0, uhat0)
            try:
                sol = solve_ivp(
                    vhatprime,
                    [t / coeff for t in t_span],
                    vhat0,
                    method="RK45",
                    t_eval=t_eval / coeff,
                    rtol=rtol,
                    atol=atol,
                )

                if (sol.status) == 0 and (np.max(np.abs(sol.y)) < 1e3):
                    vhat = sol.y
                    # u = np.fft.ifft(vhat2uhat(t_eval/coeff, vhat))
                    u = np.zeros((x_num, t_num), dtype=complex)
                    for i in range(t_num):
                        u[:, i] = np.fft.ifft(vhat2uhat(t_eval[i] / coeff, vhat[:, i]))
                    if np.all(np.abs(np.imag(u)) < 0.05):
                        u = np.real(u)
                    else:
                        raise ValueError

                    res.append(torch.from_numpy(u.transpose().astype(np.single)))
                    if len(res) >= num_initial_points:
                        break
            except Exception as e:
                pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item['coefficients'] = coefficients

        return res, coefficients

################################################### Exact ##########################################################

##### Advection
def generate_advection(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        beta = 0.5

        tf = tfinals["advection"]
        coeff = t_range/tf

        beta_range = get_sample_range(beta)
        item = {"type": "advection"}

        beta = round_to_significant_digits(np.random.uniform(*beta_range, (1,)))[0]
        coefficients =[beta]

        num_initial_points = ICs_per_equation

        res = []
        for i in range(num_initial_points * 10):
            if IC_types == "train":
                base_frequency = 2 * np.pi / x_range
                n1, n2 = np.random.randint(1, 3, size=2)
                frequencies = base_frequency * np.array([n1, n2])

                random_phases = np.random.uniform(0, 2 * np.pi, size=2)
                random_amplitudes = np.random.uniform(0, 1, size=2)

                # Composite wave function
                def _func(x):
                    wave1 = random_amplitudes[0] * np.sin(frequencies[0] * x + random_phases[0])
                    wave2 = random_amplitudes[1] * np.sin(frequencies[1] * x + random_phases[1])
                    return wave1 + wave2

                vec = _func(x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= x_range
                vec = vec - slope * x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (max - min)
                    return val

            else:
                means = np.random.uniform(0.5, 1.5, size=2)
                std_devs = np.random.uniform(0.1, 0.5, size=2)  # Random standard deviations
                sign = np.random.randint(2, size=2) * 2 - 1  # Random standard deviations

                # Define the composite Gaussian function
                def _func(x):
                    gaussian1 = np.exp(-((x - means[0]) ** 2) / (2 * std_devs[0] ** 2))
                    gaussian2 = np.exp(-((x - means[1]) ** 2) / (2 * std_devs[1] ** 2))

                    return sign[0] * gaussian1 + sign[1] * gaussian2

                vec = _func(x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= x_range
                vec = vec - slope * x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (max - min)
                    return val

            try:
                y = [func(x_grid.flatten())]
                t_eval = t_eval / coeff
                for cur_t in t_eval[1:]:
                    x_adjusted = (x_grid.flatten() - beta * cur_t) % x_range
                    y.append(func(x_adjusted))
                y = np.array(y)
                res.append(torch.from_numpy(y.astype(np.single)))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item['coefficients'] = coefficients

        return res, coefficients

##### Wave eq
def generate_wave(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        #  Assume initial velocity is 0

        item = {"type": "wave"}

        beta = 0.5

        tf = tfinals["wave"]
        coeff_t = t_range / tf
        t_eval = t_eval / coeff_t
        beta_range = get_sample_range(beta)
        beta = round_to_significant_digits(np.random.uniform(*beta_range, (1,)))[0]
        coefficients =[beta]
        
        num_initial_points = ICs_per_equation

        res = []
        for i in range(num_initial_points * 10):
            if IC_types == "train":
                base_frequency = 2 * np.pi / x_range
                n1, n2 = np.random.randint(1, 3, size=2)
                frequencies = base_frequency * np.array([n1, n2])

                random_phases = np.random.uniform(0, 2 * np.pi, size=2)
                random_amplitudes = np.random.uniform(0, 1, size=2)

                # Composite wave function
                def _func(x):
                    wave1 = random_amplitudes[0] * np.sin(frequencies[0] * x + random_phases[0])
                    wave2 = random_amplitudes[1] * np.sin(frequencies[1] * x + random_phases[1])
                    return wave1 + wave2

                vec = _func(x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= x_range
                vec = vec - slope * x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (max - min)
                    return val

            else:
                means = np.random.uniform(0.5, 1.5, size=2)
                std_devs = np.random.uniform(0.1, 0.5, size=2)  # Random standard deviations
                sign = np.random.randint(2, size=2) * 2 - 1  # Random standard deviations

                # Define the composite Gaussian function
                def _func(x):
                    gaussian1 = np.exp(-((x - means[0]) ** 2) / (2 * std_devs[0] ** 2))
                    gaussian2 = np.exp(-((x - means[1]) ** 2) / (2 * std_devs[1] ** 2))

                    return sign[0] * gaussian1 + sign[1] * gaussian2

                vec = _func(x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= x_range
                vec = vec - slope * x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (max - min)
                    return val

            try:
                y = [func(x_grid.flatten())]
                for cur_t in t_eval[1:]:
                    x_adjusted1 = (x_grid.flatten() - beta * cur_t) % x_range
                    x_adjusted2 = (x_grid.flatten() + beta * cur_t) % x_range
                    y.append(0.5 * func(x_adjusted1) + 0.5 * func(x_adjusted2))
                y = np.array(y)
                res.append(torch.from_numpy(y.astype(np.single)))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item['coefficients'] = coefficients

        return res, coefficients

################################################### PDE Bench ##########################################################

##### Diffusion Reaction (logistic: R(u) = u(u-1))
def generate_diff_logisreact_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        rho = 1
        nu = 3e-3
        item = {"type": "diff_logisreact_1D"}

        nu_range = get_sample_range(nu)
        rho_range = get_sample_range(rho)

        nu = round_to_significant_digits(np.random.uniform(*nu_range, (1,)))[0]
        rho = round_to_significant_digits(np.random.uniform(*rho_range, (1,)))[0]
        coefficients = [nu,rho]

        tf = tfinals["diff_logisreact_1D"]
        coeff_t = t_range / tf
        
        num_initial_points = ICs_per_equation

        
        try:
            if ICs is not None:
                GivenIC = ICs
                IC_train = True
                numbers = jnp.shape(ICs)[0]
            elif IC_types == "train":
                IC_train = True
                GivenIC = None
                numbers = num_initial_points * 10
            else:
                IC_train = False
                GivenIC = None
                numbers = num_initial_points * 10
            CFL = 0.35
            uu = diff_react_1D_f(
                x_range,
                0.0,
                x_num,
                0.0,
                t_range/coeff_t,
                dt/coeff_t,
                t_num,
                CFL,
                numbers,
                20,
                np.random.randint(100000),
                rho,
                nu,
                IC_train=IC_train,
                GivenIC = GivenIC,
            )
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

       
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item['coefficients'] = coefficients
        
        return res, coefficients

##### Diffusion Reaction (linear: R(u) = u)
def generate_diff_linearreact_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        rho = .1
        nu = 3e-3
        item = {"type": "diff_linearreact_1D"}

        nu_range = get_sample_range(nu)
        rho_range = get_sample_range(rho)

        nu = round_to_significant_digits(np.random.uniform(*nu_range, (1,)))[0]
        rho = round_to_significant_digits(np.random.uniform(*rho_range, (1,)))[0]
        coefficients = [nu,rho]

        tf = tfinals["diff_linearreact_1D"]
        coeff_t = t_range/tf

        num_initial_points = ICs_per_equation

        try:
            if ICs is not None:
                GivenIC = ICs
                IC_train = True
                numbers = jnp.shape(ICs)[0]
            elif IC_types == "train":
                IC_train = True
                GivenIC = None
                numbers = num_initial_points * 10
            else:
                IC_train = False
                GivenIC = None
                numbers = num_initial_points * 10
            CFL = 0.35
            uu = diff_react_1D_f(
                x_range,
                0.0,
                x_num,
                0.0,
                t_range/coeff_t,
                dt/coeff_t,
                t_num,
                CFL,
                numbers,
                20,
                np.random.randint(100000),
                rho,
                nu,
                react_term="linear",
                IC_train=IC_train,
                GivenIC = GivenIC,
            )
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients

        return res, coefficients

##### Diffusion Reaction (bistable: R(u) = u^2(1-u))
def generate_diff_bistablereact_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        rho = 1
        nu = 3e-3
        item = {"type": "diff_bistablereact_1D"}

        nu_range = get_sample_range(nu)
        rho_range = get_sample_range(rho)

        nu = round_to_significant_digits(np.random.uniform(*nu_range, (1,)))[0]
        rho = round_to_significant_digits(np.random.uniform(*rho_range, (1,)))[0]
        a = round_to_significant_digits(np.random.uniform(size=(1,)))[0]
        coefficients = [nu,rho,a]


        tf = tfinals["diff_bistablereact_1D"]
        coeff = tf/t_range

        num_initial_points = ICs_per_equation
        try:
            if ICs is not None:
                GivenIC = ICs
                IC_train = True
                numbers = jnp.shape(ICs)[0]
            elif IC_types == "train":
                IC_train = True
                GivenIC = None
                numbers = num_initial_points * 10
            else:
                IC_train = False
                GivenIC = None
                numbers = num_initial_points * 10
            CFL = 0.35
            uu = diff_react_1D_f(x_range,
                0.0,x_num,
                0.0,t_range/coeff,dt/coeff,t_num,
                CFL,
                numbers,
                20,
                np.random.randint(100000),
                rho,
                nu,
                a=a,
                react_term="bistable",
                IC_train=IC_train,
                GivenIC = GivenIC
            )
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass
        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients
    
        return res,coefficients

##### Diffusion Reaction (square: R(u) = u^2(1-u)^2)
def generate_diff_squarelogisticreact_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        rho = 1
        nu = 3e-3
        item = {"type": "diff_squarelogisticreact_1D"}

        nu_range = get_sample_range(nu)
        rho_range = get_sample_range(rho)

        nu = round_to_significant_digits(np.random.uniform(*nu_range, (1,)))[0]
        rho = round_to_significant_digits(np.random.uniform(*rho_range, (1,)))[0]
        coefficients = [nu,rho]

        tf = tfinals["diff_squarelogisticreact_1D"]
        coeff = t_range /tf

        num_initial_points = ICs_per_equation

        try:
            if ICs is not None:
                GivenIC = ICs
                IC_train = True
                numbers = jnp.shape(ICs)[0]
            elif IC_types == "train":
                IC_train = True
                GivenIC = None
                numbers = num_initial_points * 10
            else:
                IC_train = False
                GivenIC = None
                numbers = num_initial_points * 10
            CFL = 0.35
            uu = diff_react_1D_f(
                x_range,
                0.0,
                x_num,
                0.0,
                t_range/coeff,
                dt/coeff,
                t_num,
                CFL,
                numbers,
                20,
                np.random.randint(100000),
                rho,
                nu,
                react_term="squarelogistic",
                IC_train=IC_train,
                GivenIC = GivenIC,
            )
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients

        return res,coefficients

##### Burgers 
def generate_burgers(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        item = {"type": "burgers"}
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            eps = 0.01  # .05
            k = 1

            eps_range = get_sample_range(eps)
            k_range = get_sample_range(k)

            eps = round_to_significant_digits(np.random.uniform(*eps_range, (1,)))[0]
            k = round_to_significant_digits(np.random.uniform(*k_range, (1,)))[0]
            coefficients = [eps, k]

        tf = tfinals["burgers"]
        coeff = t_range/tf

        num_initial_points = ICs_per_equation

        CFL = 0.4
        train_num = 0
        train_num2 = 0
        if IC_types.startswith("rarefactiontrain"):
            _, train_num = IC_types.split("_")
            train_num = int(train_num)
        elif IC_types.startswith("rarefaction2train"):
            _, train_num2 = IC_types.split("_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif IC_types == "one_shock" or  train_num > 0 or train_num2 > 0:
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "rarefaction":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "two_shocks" or IC_types == "one_and_two_shock":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end,(numbers, 3))
            ends = np.sort(ends, axis=1)
            breakmid1 = np.random.uniform(x_range * 0.2, x_range * 0.5, (numbers,))
            breakmid2 = np.random.uniform(x_range * 0.5, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                            x_grid.flatten() < breakmid2[i]) * (
                                     x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                     x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            x_range,
            0.0,
            x_num,
            0.0,
            t_range / coeff,
            dt / coeff,
            t_num,
            CFL,
            numbers,
            20,
            np.random.randint(100000),
            eps,
            k,
            fluxx="quadratic",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )

        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item['coefficients'] = coefficients

        return res,coefficients

##### Inviscid Burgers
def generate_inviscid_burgers(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        item = {"type": "inviscid_burgers"}
        if coeff is not None:
            k = coeff[1]
        else:
            k = 1
            k_range = get_sample_range(k)
            k = round_to_significant_digits(np.random.uniform(*k_range, (1,)))[0]
            coefficients = [k]

        tf = tfinals["inviscid_burgers"]
        coeff = t_range/tf
       
        num_initial_points = ICs_per_equation

        CFL = 0.4
        train_num = 0
        train_num2 = 0
        if IC_types.startswith("rarefactiontrain"):
            _,train_num = IC_types.split("_")
            train_num = int(train_num)
        elif IC_types.startswith("rarefaction2train"):
            _,train_num2 = IC_types.split("_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif IC_types == "one_shock" or train_num == 6 or train_num2 >1:
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "rarefaction"  or (train_num >0 and train_num < 6) or train_num2 == 1:
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "two_shocks" or IC_types == "one_and_two_shock":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            breakmid1 = np.random.uniform(x_range * 0.2, x_range * 0.5, (numbers,))
            breakmid2 = np.random.uniform(x_range * 0.5, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        x_grid.flatten() < breakmid2[i]) * (
                                 x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            x_range,
            0.0,
            x_num,
            0.0,
            t_range / coeff,
            dt / coeff,
            t_num,
            CFL,
            numbers,
            20,
            np.random.randint(100000),
            0,
            k,
            viscous=False,
            fluxx="quadratic",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )

        try:
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients

        return res,coefficients

##### Conservation law linear flux
def generate_conservation_linearflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            # if extrapolate_pdetypes:
            #     eps = .03
            # else:
            eps = 0.01  # .05
            k = 1

            eps_range = get_sample_range(eps)
            k_range = get_sample_range(k)

            eps = round_to_significant_digits(np.random.uniform(*eps_range, (1,)))[0]
            k = round_to_significant_digits(np.random.uniform(*k_range, (1,)))[0]
            coefficients = [eps, k]

        item = {"type": "conservation_linearflux"}

        tf = tfinals["conservation_linearflux"]
        coeff = t_range/tf

        num_initial_points = ICs_per_equation

        CFL = 0.4
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif IC_types == "rarefaction":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "one_shock":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            breakmid1 = np.random.uniform(x_range * 0.2, x_range * 0.5, (numbers,))
            breakmid2 = np.random.uniform(x_range * 0.5, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        x_grid.flatten() < breakmid2[i]) * (
                                 x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            x_range,
            0.0,
            x_num,
            0.0,
            t_range / coeff,
            dt / coeff,
            t_num,
            CFL,
            numbers,
            20,
            np.random.randint(100000),
            eps,
            k,
            fluxx="linear",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients
        
        return res,coefficients

##### Conservation law cubic flux
def generate_conservation_cubicflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            eps = 0.01  # .05
            k = 1

            eps_range = get_sample_range(eps)
            k_range = get_sample_range(k)

            eps = round_to_significant_digits(np.random.uniform(*eps_range, (1,)))[0]
            k = round_to_significant_digits(np.random.uniform(*k_range, (1,)))[0]
            coefficients = [eps,k]

        item = {"type": "conservation_cubicflux"}


        tf = tfinals["conservation_cubicflux"]
        coeff = t_range/tf

        num_initial_points = ICs_per_equation

        CFL = 0.4
        train_num = 0
        train_num2 = 0
        if IC_types.startswith("rarefactiontrain"):
            _, train_num = IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif IC_types.startswith("rarefaction2train"):
            _, train_num2 = IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif IC_types == "periodic":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif IC_types == "rarefaction" or train_num == 1 or (train_num2 <= 2 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends **2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "one_shock" or train_num >= 2 or train_num2>2:
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end,(numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "two_shocks" or IC_types == "one_and_two_shock":

            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid1 = np.random.uniform(x_range * 0.2, x_range * 0.5, (numbers,))
            breakmid2 = np.random.uniform(x_range * 0.5, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        x_grid.flatten() < breakmid2[i]) * (
                                 x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        # print(np.random.randint(100000), eps, k)
        uu = burgers_f(
            x_range,
            0.0,
            x_num,
            0.0,
            t_range / coeff,
            dt / coeff,
            t_num,
            CFL,
            numbers,
            20,
            np.random.randint(100000),
            eps,
            k,
            fluxx="cubic",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients
        
        return res,coefficients

##### Inviscid Conservation law cubic flux
def generate_inviscid_conservation_cubicflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        item = {"type": "inviscid_conservation_cubicflux"}
        if coeff is not None:
            k = coeff[1]
        else:
            k = 1
            k_range = get_sample_range(k)
            k = round_to_significant_digits(np.random.uniform(*k_range, (1,)))[0]
            coefficients = [k]

        tf = tfinals["inviscid_conservation_cubicflux"]
        coeff = t_range/tf
        
        num_initial_points = ICs_per_equation

        CFL = 0.4
        train_num = 0
        train_num2 = 0
        if IC_types.startswith("rarefactiontrain"):
            _, train_num = IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif IC_types.startswith("rarefaction2train"):
            _, train_num2 = IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif IC_types == "rarefaction" or ( train_num <5 and train_num >0) or (train_num2 <=3 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end,(numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "one_shock" or IC_types == "one_and_two_shock" or train_num >= 5 or train_num2 > 3:
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end,(numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = np.sqrt(ends)
            # sqr = ends ** 2
            # sort_indices = np.argsort(sqr, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid1 = np.random.uniform(x_range * 0.2, x_range * 0.5, (numbers,))
            breakmid2 = np.random.uniform(x_range * 0.5, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        x_grid.flatten() < breakmid2[i]) * (
                                 x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        # print(np.random.randint(100000), k)
        uu = burgers_f(
            x_range,
            0.0,
            x_num,
            0.0,
            t_range / coeff,
            dt / coeff,
            t_num,
            CFL,
            numbers,
            20,
            np.random.randint(100000),
            0,
            k,
            viscous=False,
            fluxx="cubic",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients
        
        return res,coefficients

##### Conservation law sine flux
def generate_conservation_sinflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            eps = 0.01  # .05
            k = 1

            eps_range = get_sample_range(eps)
            k_range = get_sample_range(k)

            eps = round_to_significant_digits(np.random.uniform(*eps_range, (1,)))[0]
            k = round_to_significant_digits(np.random.uniform(*k_range, (1,)))[0]
            coefficients = [eps,k]

        item = {"type": "conservation_sinflux"}

        tf = tfinals["conservation_sinflux"]
        coeff = t_range/tf

        num_initial_points = ICs_per_equation
        CFL = 0.4
        train_num =0
        train_num2 =0
        if IC_types.startswith("rarefactiontrain"):
            _, train_num = IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif IC_types.startswith("rarefaction2train"):
            _, train_num2 = IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            mode = "periodic"
            numbers = jnp.shape(ICs)[0]
        elif IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif IC_types == "one_shock" or train_num >= 3 or train_num2 > 4:
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "rarefaction" or (train_num < 3 and train_num > 0) or (
                    train_num2 <= 4 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = -np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start, IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = np.arccos(ends)
            breakmid1 = np.random.uniform(x_range * 0.2, x_range * 0.5, (numbers,))
            breakmid2 = np.random.uniform(x_range * 0.5, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        x_grid.flatten() < breakmid2[i]) * (
                                 x_grid.flatten() >= breakmid1[i]) + ends[i, 2] * (
                                 x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            x_range,
            0.0,
            x_num,
            0.0,
            t_range / coeff,
            dt / coeff,
            t_num,
            CFL,
            numbers,
            20,
            np.random.randint(100000),
            eps,
            k,
            fluxx="sin",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 30:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients
        
        return res,coefficients

##### Inviscid Conservation law sine flux
def generate_inviscid_conservation_sinflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        item = {"type": "inviscid_conservation_sinflux"}
        if coeff is not None:
            k = coeff[1]
        else:
            k = 1
            k_range = get_sample_range(k)
            k = round_to_significant_digits(np.random.uniform(*k_range, (1,)))[0]
            coefficients = [k]

        tf = tfinals["inviscid_conservation_sinflux"]
        coeff = t_range/tf

        num_initial_points = ICs_per_equation
        CFL = 0.4
        train_num=0
        train_num2 =0
        if IC_types.startswith("rarefactiontrain"):
            _, train_num = IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif IC_types.startswith("rarefaction2train"):
            _, train_num2 = IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif IC_types == "one_shock" or train_num >= 4 or train_num2 > 5:
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "rarefaction" or (train_num < 4 and train_num > 0) or (
                    train_num2 <= 5 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end,(numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end,(numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = -np.arccos(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid1 = np.random.uniform(x_range * 0.2, x_range * 0.5, (numbers,))
            breakmid2 = np.random.uniform(x_range * 0.5, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        x_grid.flatten() < breakmid2[i]) * (
                                 x_grid.flatten() >= breakmid1[i]) + ends[i, 2] * (
                                 x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            x_range,
            0.0,
            x_num,
            0.0,
            t_range / coeff,
            dt / coeff,
            t_num,
            CFL,
            numbers,
            20,
            np.random.randint(100000),
            0,
            k,
            viscous=False,
            fluxx="sin",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 30:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients
        
        return res,coefficients

##### Conservation law cosine flux
def generate_conservation_cosflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        if coeff is not None:
            eps = coeff[0] * np.pi
            k = coeff[1]
        else:
            eps = 0.01  # .05
            if IC_types.startswith("rarefaction") or IC_types == "one_shock" or IC_types == "two_shocks":
                k=-1
            else:
                k =1

            eps_range = get_sample_range(eps)
            k_range = get_sample_range(k)

            eps = round_to_significant_digits(np.random.uniform(*eps_range, (1,)))[0]
            k = round_to_significant_digits(np.random.uniform(*k_range, (1,)))[0]
            coefficients = [eps,k]

        item = {"type": "conservation_cosflux"}


        tf = tfinals["conservation_cosflux"]
        coeff = t_range/tf

        num_initial_points = ICs_per_equation
        CFL = 0.4
        train_num =0
        train_num2 =0
        if IC_types.startswith("rarefactiontrain"):
            _, train_num = IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif IC_types.startswith("rarefaction2train"):
            _, train_num2 = IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            mode = "periodic"
            numbers = jnp.shape(ICs)[0]
        elif IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif IC_types == "rarefaction" or (train_num < 3 and train_num > 0) or (
                        train_num2 <= 4 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "one_shock" or train_num >= 3 or train_num2 > 4:
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start, IC_jump_end, (numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = np.arcsin(ends)
            breakmid1 = np.random.uniform(x_range * 0.2, x_range * 0.5, (numbers,))
            breakmid2 = np.random.uniform(x_range * 0.5, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        x_grid.flatten() < breakmid2[i]) * (
                                 x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            x_range,
            0.0,
            x_num,
            0.0,
            t_range / coeff,
            dt / coeff,
            t_num,
            CFL,
            numbers,
            20,
            np.random.randint(100000),
            eps,
            k,
            fluxx="cos",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:
            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients
        
        return res,coefficients

##### Inviscid Conservation law cosine flux
def generate_inviscid_conservation_cosflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        item = {"type": "inviscid_conservation_cosflux"}
        if coeff is not None:
            k = coeff[1]
        else:
            if IC_types.startswith("rarefaction") or IC_types == "one_shock" or IC_types == "two_shocks":
                k = -1
            else:
                k = 1

            k_range = get_sample_range(k)
            k = round_to_significant_digits(np.random.uniform(*k_range, (1,)))[0]
            coefficients = [k]

        tf = tfinals["inviscid_conservation_cosflux"]
        coeff = t_range/tf

        num_initial_points = ICs_per_equation

        CFL = 0.4
        train_num=0
        train_num2 =0
        if IC_types.startswith("rarefactiontrain"):
            _, train_num = IC_types.split("rarefactiontrain_")
            train_num = int(train_num)
        elif IC_types.startswith("rarefaction2train"):
            _, train_num2 = IC_types.split("rarefaction2train_")
            train_num2 = int(train_num2)
        if ICs is not None:
            GivenIC = ICs
            IC_train = True
            numbers = jnp.shape(ICs)[0]
            mode = "periodic"
        elif IC_types == "train":
            IC_train = True
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        elif IC_types == "rarefaction" or (train_num < 4 and train_num > 0) or (
                    train_num2 <= 5 and train_num2 > 0):
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end, (numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 0] * (x_grid.flatten() < breakmid[i]) + ends[i, 1] * (
                            x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "one_shock" or train_num >= 4 or train_num2 > 5:
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end,(numbers, 2))
            ends = np.sort(ends, axis=1)
            # ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid = np.random.uniform(x_range * 0.2, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 1] * (x_grid.flatten() < breakmid[i]) + ends[i, 0] * (
                        x_grid.flatten() >= breakmid[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        elif IC_types == "two_shocks":
            numbers = num_initial_points * 10
            ends = np.random.uniform(IC_jump_start,IC_jump_end,(numbers, 3))
            ends = np.sort(ends, axis=1)
            # ends = np.arcsin(ends)
            # neg_cos_ends = -np.cos(ends)
            # sort_indices = np.argsort(neg_cos_ends, axis=1)
            # ends = np.array([row[indices] for row, indices in zip(ends, sort_indices)])
            breakmid1 = np.random.uniform(x_range * 0.2, x_range * 0.5, (numbers,))
            breakmid2 = np.random.uniform(x_range * 0.5, x_range * 0.8, (numbers,))
            GivenIC = []
            for i in range(numbers):
                # distance_from_midpoint = x_grid.flatten() - breakmid[i]
                # scaled_distance = distance_from_midpoint/(0.1 *x_range)
                # smooth_transition = (jnp.tanh(scaled_distance) + 1) / 2
                thisIC = ends[i, 2] * (x_grid.flatten() < breakmid1[i]) + ends[i, 1] * (
                        x_grid.flatten() < breakmid2[i]) * (
                                 x_grid.flatten() >= breakmid1[i]) + ends[i, 0] * (
                                 x_grid.flatten() >= breakmid2[i])
                # slope = thisIC[-1]  - thisIC[0]
                # slope /= x_range
                GivenIC.append(thisIC)
            GivenIC = jnp.array(GivenIC)
            IC_train = True
            mode = "copy"
        else:
            IC_train = False
            GivenIC = None
            numbers = num_initial_points * 10
            mode = "periodic"
        uu = burgers_f(
            x_range,
            0.0,
            x_num,
            0.0,
            t_range / coeff,
            dt / coeff,
            t_num,
            CFL,
            numbers,
            20,
            np.random.randint(100000),
            0,
            k,
            viscous=False,
            fluxx="cos",
            IC_train=IC_train,
            GivenIC=GivenIC,
            mode=mode
        )
        try:

            res = []
            res_np = []
            for i in range(uu.shape[1]):
                sample = np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))
                if np.linalg.norm(sample) < 2000 and np.linalg.norm(sample) > 10:
                    res.append(torch.tensor(sample.squeeze(-1)))
                    res_np.append(sample.squeeze(-1))

                if len(res) >= num_initial_points:
                    break
        except Exception as e:
            pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients
        
        return res,coefficients

##### Fokker Plank
def generate_fplanck(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train", IC_jump_start = 0, IC_jump_end = 2, ICs=None,coeff = None):
        t_range = t_span[1]
        x_range = x_span[1]
        
        tf = tfinals["fplanck"]
        um = 1e-6  # micrometer
        viscosity = 1e-3  # viscosity of the medium (Pa·s)
        radius = 0.1 * um  # radius of the particle converted to micrometers
        coeff_t = t_range / tf
        # coeff_x = 1/um

        L = 0.1 * um  # Characteristic length scale of the potential, converted to micrometers
        c = 5e-21
        temperature = 300
        item = {"type": "fplanck"}

        viscosity_range = get_sample_range(viscosity)
        viscosity = round_to_significant_digits(np.random.uniform(*viscosity_range, (1,)))[0]
        coefficients = [viscosity]
        # c_range = get_sample_range(c)
        # c = refine_floats(np.random.uniform(*c_range, (1,)))[0]
        # c = c* 10 **-21
        drag = 6 * np.pi * viscosity * radius  # drag coefficient

        # Define the potential function U(x) using micrometers
        U = lambda x: c * np.cos(x / L)

        # Setup the fokker_planck simulation with parameters converted to micrometers
        sim = fokker_planck(
            temperature=temperature,
            drag=drag,
            extent=2 * um,
            # extent converted to micrometers
            resolution=dx * um,  # resolution converted to micrometers
            boundary=boundary.periodic,
            potential=U,
        )

        # # Steady-state solution
        # steady = sim.steady_state()
        # ODE solve
        num_initial_points = ICs_per_equation 
        res = []

        for i in range(num_initial_points * 10):
            if IC_types == "train":
                mean = np.random.uniform(0.5, 1.5)
                std = np.random.uniform(0.1, 0.5)
                # pdf = gaussian_pdf(mean * um, std * um)
                p0 = np.exp(-np.square((sim.grid[0] - mean * um) / (std * um)))
                slope = (p0[-1] - p0[0]) / sim.grid[0][-1]
                p0 -= slope * sim.grid[0]
                # p0 = pdf(sim.grid[0])
            else:
                begin = np.random.uniform(0.1, 1.0)
                end = np.random.uniform(1.1, 1.9)

                def region_func(x):
                    return (x > begin * um) & (x < end * um)

                pdf = uniform_pdf(func=region_func)
                p0 = pdf(sim.grid[0])
            try:
                time, Pt = sim.propagate_interval(p0, tf, Nsteps=t_num)
                res.append(torch.from_numpy(Pt))
                if len(res) >= num_initial_points:
                    break
            except Exception as e:
                pass

        item["data"] = res
        item["t_grid"] = t_eval
        item["t_span"] = t_span
        item["x_grid"] = x_grid
        item["coefficients"] = coefficients
        
        return res,coefficients
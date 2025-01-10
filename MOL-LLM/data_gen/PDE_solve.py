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
from .PDE_dataset import *

rtol = 1e-5
atol = 1e-6
def Solve_PDE(sentence_idx,ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types = "train"):
    if sentence_idx==12:
        sol, coeffs = generate_heat(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==13:
        sol, coeffs = generate_porous_medium(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==14:
        sol, coeffs = generate_Klein_Gordon(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==15:
        sol, coeffs = generate_Sine_Gordon(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==16:
        sol, coeffs = generate_cahnhilliard_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==17:
        sol, coeffs = generate_kdv(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==18:
        sol, coeffs = generate_advection(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==19:
        sol, coeffs = generate_wave(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==20:
        sol, coeffs = generate_diff_logisreact_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==21:
        sol, coeffs = generate_diff_linearreact_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==22:
        sol, coeffs = generate_diff_bistablereact_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==23:
        sol, coeffs = generate_diff_squarelogisticreact_1D(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==24:
        sol, coeffs = generate_burgers(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==25:
        sol, coeffs = generate_inviscid_burgers(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==26:
        sol, coeffs = generate_conservation_linearflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==27:
        sol, coeffs = generate_conservation_cubicflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==28:
        sol, coeffs = generate_inviscid_conservation_cubicflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==29:
        sol, coeffs = generate_conservation_sinflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==30:
        sol, coeffs = generate_inviscid_conservation_sinflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==31:
        sol, coeffs = generate_conservation_cosflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==32:
        sol, coeffs = generate_inviscid_conservation_cosflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==33:
        sol, coeffs = generate_fplanck(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
        return np.stack(sol), coeffs
    elif sentence_idx==34:
        sol, coeffs = generate_burgers(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="one_shock")
        return np.stack(sol), coeffs
    elif sentence_idx==35:
        sol, coeffs = generate_inviscid_burgers(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="one_shock")
        return np.stack(sol), coeffs
    elif sentence_idx==36:
        sol, coeffs = generate_conservation_linearflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="one_shock")
        return np.stack(sol), coeffs
    elif sentence_idx==37:
        sol, coeffs = generate_conservation_cubicflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="one_shock")
        return np.stack(sol), coeffs
    elif sentence_idx==38:
        sol, coeffs = generate_inviscid_conservation_cubicflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="one_shock")
        return np.stack(sol), coeffs
    elif sentence_idx==39:
        sol, coeffs = generate_conservation_sinflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="one_shock")
        return np.stack(sol), coeffs
    elif sentence_idx==40:
        sol, coeffs = generate_inviscid_conservation_sinflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="one_shock")
        return np.stack(sol), coeffs
    elif sentence_idx==41:
        sol, coeffs = generate_conservation_cosflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="one_shock")
        return np.stack(sol), coeffs
    elif sentence_idx==42:
        sol, coeffs = generate_inviscid_conservation_cosflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="one_shock")
        return np.stack(sol), coeffs
    elif sentence_idx==43:
        sol, coeffs = generate_burgers(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="rarefaction")
        return np.stack(sol), coeffs
    elif sentence_idx==44:
        sol, coeffs = generate_inviscid_burgers(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="rarefaction")
        return np.stack(sol), coeffs
    elif sentence_idx==45:
        sol, coeffs = generate_conservation_linearflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="rarefaction")
        return np.stack(sol), coeffs
    elif sentence_idx==46:
        sol, coeffs = generate_conservation_cubicflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="rarefaction")
        return np.stack(sol), coeffs
    elif sentence_idx==47:
        sol, coeffs = generate_inviscid_conservation_cubicflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="rarefaction")
        return np.stack(sol), coeffs
    elif sentence_idx==48:
        sol, coeffs = generate_conservation_sinflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="rarefaction")
        return np.stack(sol), coeffs
    elif sentence_idx==49:
        sol, coeffs = generate_inviscid_conservation_sinflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="rarefaction")
        return np.stack(sol), coeffs
    elif sentence_idx==50:
        sol, coeffs = generate_conservation_cosflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="rarefaction")
        return np.stack(sol), coeffs
    elif sentence_idx==51:
        sol, coeffs = generate_inviscid_conservation_cosflux(ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types="rarefaction")
        return np.stack(sol), coeffs
     
     
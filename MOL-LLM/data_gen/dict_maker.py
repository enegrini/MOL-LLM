import numpy as np
import scipy
from scipy.integrate import odeint
import re
import random
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import *
import sys
import os
from .PDE_solve import Solve_PDE

# random.seed(100)

#Load sentences (without descriptions)
sentences = []
sentences_file_name = "/home/elisa/code/icon-gen/src/data_gen/sentences.txt"
with open(sentences_file_name, "r") as file:
    sentences = [line.strip() for line in file]

#load descriptions and save them in dictionary
description_dict = {}
for i in range(len(sentences)):
    description_i = []
    descript_file_name = f"/home/elisa/code/icon-gen/src/data_gen/descriptions/description_{i}.txt"
    with open(descript_file_name, "r") as file:
        description_i = [line.strip() for line in file]
    description_dict[i] = description_i


def sample_sentence(input_strings, placeholder="<pl>"):
    # Create a regular expression pattern to match the placeholder
    pattern = re.escape(placeholder)
    result = []
    for sent in input_strings:
        # Use re.split to split the input_string based on the placeholder
        parts = re.split(pattern, sent)
        # Remove empty strings from the list and add to the result
        parts = [[part.strip()] for part in parts if part.strip()]
        result.append(parts)
    return (result)


#kernels
def rbf_kernel(x1, x2, sigma, l):
    """
    Radial basis function kernel
    """
    sq_norm = np.square(np.subtract.outer(x1, x2) / l)
    return sigma**2 * np.exp(-0.5 * sq_norm)

def rbf_kernel_sin(x1, x2, sigma, l):
    """
    Radial basis function kernel with sin component
    """
    sq_norm = np.square(np.sin(np.pi * (np.subtract.outer(x1, x2))) / l)
    return sigma**2 * np.exp(-0.5 * sq_norm)

def rbf_kernel_circle(x1, x2, sigma, l):
    """
    Radial basis function kernel with circular component
    """
    xx1_1 = np.sin(2 * np.pi * x1)
    xx1_2 = np.cos(2 * np.pi * x1)
    xx2_1 = np.sin(2 * np.pi * x2)
    xx2_2 = np.cos(2 * np.pi * x2)
    sq_norm = (np.square(xx1_1 - xx2_1) + np.square(xx1_2 - xx2_2)) / (l**2)
    return sigma**2 * np.exp(-0.5 * sq_norm)

def generate_gaussian_process(times, n_control, kernel=rbf_kernel, k_sigma = 0.1, k_l=1):
    '''
    times: 1D array (t_len,)
    n_control: number of controls
    kernel: kernel fct with parameters k_sigma, k_l
    out: Gaussian process samples, 2D array (n_control, t_len)
    '''
    t_len = len(times)
    mean = np.zeros(t_len)
    cov = kernel(times, times, sigma=k_sigma, l=k_l)
    out = np.random.multivariate_normal(mean, cov, size=n_control)
    return out


def sample_parametersODE(t_int, t_len, n_ic, n_control, n_params=0, ODE_sys = None):
    # Create an array of times
    times = np.linspace(t_int[0], t_int[-1], t_len)

    # Generate an array of randomly sampled initial conditions
    if (ODE_sys == 'ODE1' or ODE_sys == 'ODE2' or ODE_sys == 'ODE3' or ODE_sys == 'ODE4' or ODE_sys == 'ODE5'):
        initial_conditions = np.random.uniform(-1, 1, n_ic)
    elif ODE_sys == 'SIR':
        initial_conditions = np.random.uniform(0, 1, size=(n_ic, 3))
    elif ODE_sys == 'Neural':
        initial_conditions = np.random.uniform(1, 3, size=(n_ic, 3))
    else: #2D systems
        initial_conditions = np.random.uniform(1, 4, size=(n_ic,2))
    
    # Generate an array of randomly sampled constant coeffcients
    if n_params == 0:
        constant_coeffs = None
    else:
        if ODE_sys == 'ODE1':
            constant_coeffs=[1]
        elif ODE_sys == 'ODE2':
            constant_coeffs=[1,2]
        elif ODE_sys == 'ODE3':
            constant_coeffs=[1,3/10]
        elif ODE_sys == 'ODE4':
            constant_coeffs=[2,0.5]
        elif ODE_sys == 'ODE5':
            constant_coeffs=[3/2]
        elif ODE_sys == 'SIR':
            constant_coeffs = [0.3,0.1]
        elif ODE_sys == 'Neural':
            constant_coeffs = [0.2 ,0.1 ,0.05 , 0.5]
        elif ODE_sys == 'VanDerPol':
            constant_coeffs = [2]
        elif ODE_sys == 'LotkaVolterra':
            constant_coeffs = [1.5, 1, 3, 1]
        elif ODE_sys == 'FitzHughNagumo':
            constant_coeffs = [0, 0.7, 0.8, 0.8]
        elif ODE_sys == 'Brusselator':
            constant_coeffs = [2, 4]
        elif ODE_sys == 'Duffing':
            constant_coeffs = [1, 0.2, 0.3]

        random_noise = np.random.uniform(-1, 1, (n_params,len(constant_coeffs)))*(np.array(constant_coeffs)*0.1)
        constant_coeffs = constant_coeffs + random_noise
            
    # Generate c Gaussian processes evaluated at times t
    if n_control == 0:
        GP = None
    else:
        GP = generate_gaussian_process(times, n_control)
    return times, initial_conditions,GP,constant_coeffs


def Solve_ODE(ODE_idx, times, initial_conditions, controls, constant_coeffs=None, max_dim = 3):
    """Solves the ODE for multiple initial conditions and controls
    returns u of shape n_initial_conds x n_controlsx n_times"""
    
    def ODEs_set():
    # Define the ODE function
        def ODE_control(t, u,constant_coeffs=[1]):
            #equation is du_dt = c(t)u(t)
            
            #this next interpolation step is needed as solve_ivp requires evaluating the control function at different t then the ones it was computed for
            # the control_idx is used to select the correct control function to use.
            # control_value = np.interp(t, times, controls[control_idx]) 
            a = constant_coeffs[0]
            du_dt = a*np.sin(2 * np.pi * t)*u #control_value * u
            return du_dt
        
        def ODE_control_coef(t, u,constant_coeffs=[1,2]):
            #equation is du_dt = a1 c(t) + a2 where a1=1, a2=2 are fixed
            
            # control_value = np.interp(t, times, controls[control_idx])
            a, b = constant_coeffs 
            du_dt = a*np.exp(-t) + b #1*control_value + 2
            return du_dt

        def ODE_control_1D(t, u, constant_coeffs=[1,3/10]):
            # equation is du_dt = c(t)* cos(u)+3*u
            # control_value = np.interp(t, times, controls[control_idx])
            a, b = constant_coeffs 
            du_dt = a*t**2 * np.cos(u)+ b*u            #control_value * np.cos(u)+ control_value*u
            return du_dt

        def ODE_control_sin(t, u, constant_coeffs=[2,0.5]):
            # equation is du_dt = sin(c(t)) + u
            # control_value = np.interp(t, times, controls[control_idx])
            a, b = constant_coeffs 
            du_dt = a*np.sin(np.exp(-0.5 * t) * np.sin(3 * t)) + b*u #np.sin(control_value) + u
            return du_dt

        def ODE_control_sin_u(t, u, constant_coeffs=[3/2]):
            # equation is du_dt = c(t)sin(u)
            # control_value = np.interp(t, times, controls[control_idx])
            a = constant_coeffs[0]
            du_dt = a*t * np.sin(u)#control_value * np.sin(u)
            return du_dt
        
        def ODE_sir_model(t, u, constant_coeffs= [0.3,0.1]):  # Changed order of parameters
            beta, gamma = constant_coeffs
            dSdt = -beta * u[0] * u[1]
            dIdt = beta * u[0] * u[1] - gamma * u[1]
            dRdt = gamma * u[1]
            return [10*dSdt, 10*dIdt, 10*dRdt]

        def ODE_neural_dynamics(t, u, constant_coeffs = [0.2 ,0.1 ,0.05 , 0.5]):
            alpha, beta, gamma, delta = constant_coeffs
            input_signal = 0.01 * np.sin(t)  # Change input signal to a sinusoidal function
            dEdt = alpha * u[0] - beta * u[0] * u[1] - gamma * u[0] + input_signal
            dIdt = delta * u[0] - 0.15 * u[1]
            dHdt = 0.1 * u[1] - 0.02 * u[2]
            return [10*dEdt, 10*dIdt, 10*dHdt]

        # def ODE_Lorenz(t, u, constant_coeffs=[10,28,8/3]):
        #     # Lorenz System ODEs
        #     sigma, r, beta = constant_coeffs
        #     du_dt = sigma * (u[1] - u[0])
        #     dv_dt = u[0] * (r - u[2]) - u[1]
        #     dw_dt = u[0] * u[1] - beta * u[2]
        #     return [2*du_dt, 2*dv_dt, 2*dw_dt]

        # def ODE_Chen(t, u, constant_coeffs=[5,-10,-0.38]):
        #     # Chen System ODEs
        #     a, b, c = constant_coeffs
        #     du_dt = a * u[0] -u[1]*u[2]
        #     dv_dt = b*u[2]+u[0]*u[2]
        #     dw_dt = c*u[2] + u[0]*u[1]/3
        #     return [0.4*du_dt, 0.4*dv_dt, 0.4*dw_dt]
        
        def ODE_VanDerPol(t, u, constant_coeffs=[2]):
            mu = constant_coeffs[0]
            x, y = u
            dx_dt = y
            dy_dt = mu * (1 - x**2) * y - x
            return [5*dx_dt, 5*dy_dt]

        def ODE_LotkaVolterra(t, u, constant_coeffs=[1.5, 1, 3, 1]):
            alpha, beta, gamma, delta = constant_coeffs
            x, y = u
            dx_dt = alpha * x - beta * x * y
            dy_dt = delta * x * y - gamma * y
            return [5*dx_dt, 5*dy_dt]

        def ODE_FitzHughNagumo(t, u, constant_coeffs=[0, 0.7, 0.8, 0.8]):
            I, a, b, tau = constant_coeffs
            v, w = u
            dv_dt = v - (v**3 / 3) - w + I
            dw_dt = (1 / tau) * (v + a - b * w)
            return [5*dv_dt, 5*dw_dt]

        def ODE_Brusselator(t, u, constant_coeffs=[2, 4]):
            A, B = constant_coeffs
            x, y = u
            dx_dt = A + x**2 * y - (B + 1) * x
            dy_dt = B * x - x**2 * y
            return [5*dx_dt, 5*dy_dt]

        def ODE_Duffing(t, u, constant_coeffs= [1, 0.2, 0.3]):
            alpha, beta, delta = constant_coeffs
            x, y = u
            dx_dt = y
            dy_dt = -delta * y - alpha * x - beta * x**3
            return [5*dx_dt, 5*dy_dt]


        return [ODE_control, ODE_control_coef, ODE_control_1D, ODE_control_sin, ODE_control_sin_u, 
                ODE_sir_model, ODE_neural_dynamics,
                ODE_VanDerPol,ODE_LotkaVolterra,ODE_FitzHughNagumo, ODE_Brusselator,ODE_Duffing]

    
    ODEs = ODEs_set()
    # Solve the ODE 
    if ODE_idx < 5:
        solutions = np.zeros((len(initial_conditions), len(constant_coeffs), len(times), max_dim)) 
        for coeff_idx in range(len(constant_coeffs)):
            coeffs = constant_coeffs[coeff_idx]
            solution = solve_ivp(
                ODEs[ODE_idx],
                (times[0], times[-1]),
                initial_conditions,
                args=(coeffs,),
                t_eval=times,
            )
            solutions[:, coeff_idx, :, 0] = solution.y #mx_dim components, only the first one is relevant in 1D case
        # solutions = np.zeros((len(initial_conditions), len(controls), len(times), max_dim)) 
        # for control_idx in range(len(controls)):
        #     solution = solve_ivp(
        #         ODEs[ODE_idx],
        #         (times[0], times[-1]),
        #         initial_conditions,
        #         args=(controls, control_idx, constant_coeffs),
        #         t_eval=times,
        #     )
        #     solutions[:, control_idx, :, 0] = solution.y #mx_dim components, only the first one is relevant in 1D case
    
    else: # SIR or Neural or any of the 2D systems
        solutions = np.zeros((len(initial_conditions), len(constant_coeffs), len(times), max_dim))
        for coeff_idx in range(len(constant_coeffs)):
            coeffs = constant_coeffs[coeff_idx]
            for ic_idx in range(len(initial_conditions)):
                solution = solve_ivp(
                    ODEs[ODE_idx],
                    (times[0], times[-1]),
                    initial_conditions[ic_idx],
                    args=(coeffs,),
                    t_eval=times,
                )
                solutions[ic_idx, coeff_idx, :, :initial_conditions.shape[1]] = solution.y.transpose() #max_dim components, only the first 2 are relevant in 2D case
    
    return solutions

def dictionary_maker(max_dim, max_number_coeffs,sentence_ids,t_start, t_end, t_len, nIC_per_eq, IC_types):
    # defaults t_len=50, n_ic=2, n_control=2
    #for each parmeter add a different description to the text part
    text = sample_sentence(sentences)
    data = []
    for sentence_idx in sentence_ids:
        sentence_data =[]
        if sentence_idx < 5:
            if sentence_idx == 0:
                ODE_sys = 'ODE1'
            elif sentence_idx == 1:
                ODE_sys = 'ODE2'
            elif sentence_idx == 2:
                ODE_sys = 'ODE3'
            elif sentence_idx == 3:
                ODE_sys = 'ODE4'
            elif sentence_idx == 4:
                ODE_sys = 'ODE5'
            times, initial_conditions, controls, constant_coeffs = sample_parametersODE(
                [t_start, t_end], t_len=t_len, n_ic=nIC_per_eq, n_control=0, n_params=1, ODE_sys = ODE_sys)
            solutions = Solve_ODE(sentence_idx, times, initial_conditions, controls, constant_coeffs, max_dim)
            for coeff_idx in range(len(constant_coeffs)):
                for ic_idx in range(len(initial_conditions)):
                    random_index = random.randint(0, len(description_dict[sentence_idx]) - 1)
                    sentence_data.append(
                        {
                            "text": text[sentence_idx] + [[description_dict[sentence_idx][random_index]]],
                            "data": np.column_stack((times, solutions[ic_idx, coeff_idx, :, :])),
                            "coefficients": np.hstack((np.column_stack((constant_coeffs[coeff_idx])), np.zeros((1, max_number_coeffs-len(constant_coeffs[coeff_idx]))))),
                            "control": np.zeros((len(times),max_dim+1))
                        }
                    )
            # "coefficients": np.zeros((1,max_number_coeffs)),
            # "control": np.hstack((np.column_stack((times, controls[control_idx])), np.zeros((times.shape[0], max_dim-1)))) #pad with zeros in 1D case #np.column_stack((times, controls[control_idx])),
            data.append(sentence_data)
        elif (sentence_idx >4 and sentence_idx <12):
            if sentence_idx == 5:
                ODE_sys = 'SIR'
            elif sentence_idx == 6:
                ODE_sys = 'Neural'
            elif sentence_idx == 7:
                ODE_sys = 'VanDerPol'
            elif sentence_idx == 8:
                ODE_sys = 'LotkaVolterra'
            elif sentence_idx == 9:
                ODE_sys = 'FitzHughNagumo'
            elif sentence_idx == 10:
                ODE_sys = 'Brusselator'
            elif sentence_idx == 11:
                ODE_sys = 'Duffing'
            times, initial_conditions, controls, constant_coeffs = sample_parametersODE(
                [t_start,t_end], t_len=t_len, n_ic=nIC_per_eq, n_control=0, n_params=1, ODE_sys = ODE_sys)
            solutions = Solve_ODE(sentence_idx, times, initial_conditions, controls, constant_coeffs, max_dim)
            for coeff_idx in range(len(constant_coeffs)):
                for ic_idx in range(len(initial_conditions)):
                    random_index = random.randint(0, len(description_dict[sentence_idx]) - 1)
                    sentence_data.append(
                        {
                            "text": text[sentence_idx] + [[description_dict[sentence_idx][random_index]]],
                            "data": np.column_stack((times, solutions[ic_idx, coeff_idx, :,:])),
                            "coefficients": np.hstack((np.column_stack((constant_coeffs[coeff_idx])), np.zeros((1, max_number_coeffs-len(constant_coeffs[coeff_idx]))))),
                            "control": np.zeros((len(times),max_dim+1))
                        }
                    )
            data.append(sentence_data)
        else:
            ICs_per_equation = nIC_per_eq

            t_span=[t_start,t_end]
            x_span=[0.,2.]
            t_num = t_len#64
            dt =( t_span[1] -  t_span[0])/t_num
            t_eval = np.linspace(t_span[0],t_span[1], t_num + 1)[:-1]

            x_num = 128
            space_dim = 1
            if space_dim > 0:
                dx = ( x_span[1] -  x_span[0]) / x_num
                x_grid_1d = np.linspace( x_span[0], x_span[1], x_num + 1)
                x_grid_1d = x_grid_1d[:-1] + 0.5 * dx
                x_grid_size = x_num ** space_dim
                tmp_grids = [x_grid_1d for _ in range(space_dim)]

                x_grid = np.stack(np.meshgrid(*tmp_grids, indexing="ij"))  # (space_dim, x_num, ..., x_num)
                x_grid = np.moveaxis(x_grid, 0, -1)  # (x_num, ..., x_num, space_dim)
            else:
                x_grid = None
                dx = 0
        
            tfinals = {
            "heat" : t_span[1],
            "porous_medium": 0.1,
            "advection": t_span[1],
            "kdv":1,
            "fplanck":.1,
            "diff_logisreact_1D":t_span[1],
            "diff_linearreact_1D":t_span[1],
            "diff_bistablereact_1D":t_span[1],
            "diff_squarelogisticreact_1D":t_span[1],
            "burgers":t_span[1],
            "conservation_linearflux":t_span[1],
            "conservation_sinflux":t_span[1],
            "conservation_cosflux": t_span[1],
            "conservation_cubicflux":t_span[1],
            "inviscid_burgers":t_span[1],
            "inviscid_conservation_sinflux":t_span[1],
            "inviscid_conservation_cosflux": t_span[1],
            "inviscid_conservation_cubicflux":t_span[1],
            "cahnhilliard_1D":.5,
            "wave":1,
            "Klein_Gordon":1.,
            "Sine_Gordon":1.,
            }
            solutions, constant_coeffs = Solve_PDE(sentence_idx, ICs_per_equation,x_span,dx,x_num,x_grid,t_span,t_eval,dt,t_num,tfinals,IC_types)
            #(ICs_per_equation, t_len,x_len), 1D list of coeffs
            for ic_idx in range(ICs_per_equation):
                random_index = random.randint(0, len(description_dict[sentence_idx]) - 1)
                sentence_data.append(
                    {
                        "text": text[sentence_idx] + [[description_dict[sentence_idx][random_index]]],
                        "data": np.column_stack((t_eval, solutions[ic_idx, :,:])),
                        "coefficients": np.hstack((np.column_stack((constant_coeffs)), np.zeros((1, max_number_coeffs-len(constant_coeffs))))),
                        "control": np.zeros((len(t_eval),129))
                    }
                )
            data.append(sentence_data)

    return data
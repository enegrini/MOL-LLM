# Multimodal Operator Learning with Large Language Models: Bridging Numerical Solutions and Scientific Text Descriptions
Repository for the paper "Multimodal Operator Learning with Large Language Models: Bridging Numerical Solutions and Scientific Text Descriptions" by E. Negrini, Y.Liu, L.Yang, S. J. Osher,H. Schaeffer. 

# ODEs indices
### 1D
 "du_dt = a*sin(2 * pi * t)*u" = 0  
 "du_dt = a*exp(-t) + b" = 1  
 "du_dt = a*t**2 * cos(u)+ b*u" = 2  
 "du_dt = a*sin(exp(-0.5 * t) * sin(3 * t)) + b*u" = 3  
 "du_dt = a*t * sin(u)" = 4  
### 3D
 SIR = 5
 Neural = 6
# 2D
 VanDerPol = 7
 LotkaVolterra = 8
 FitzHughNagumo = 9
 Brusselator = 10
 Duffing = 11

# PDE indices
 Heat = 12
 Porous Medium = 13
 Klein Gordon = 14
 Sine Gordon = 15
 Cahn Hilliard = 16
 Korteweg De Vries = 17
 Advection = 18
 Wave = 19
 Diffusion-reaction Logistic = 20
 Diffusion-reaction Linear = 21
 Diffusion-reaction Bistable = 22
 Diffusion-reaction Square Logistic = 23
 # Conservation Laws, no shocks
 Burgers = 24 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
 Inviscid Burgers = 25 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
 Conservation law Linear Flux = 26 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
 Conservation law Cubic Flux = 27 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
 Inviscid Conservation law Cubic Flux = 28 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
 Conservation law Sine Flux = 29 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
 Inviscid Conservation law Sine Flux = 30 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
 Conservation law Cosine Flux = 31 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
 Inviscid Conservation law Cosine Flux = 32 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
 Fokker-Plank = 33
 # Conservation Laws, one shock
 Burgers-Inviscid Conservation law Cosine Flux with one shock = 34-42
 # Conservation Laws, rarefaction
 Burgers-Inviscid Conservation law Cosine Flux with rarefaction = 43-51

GPU=1

###Generate training data
# CUDA_VISIBLE_DEVICES=$GPU python main.py --data_home_folder /home/elisa/code/icon-gen/Paper_datasets/ \
# --sentence_ids 5,6 --train_batch_size 1 --t_len 64 --IC_per_eq 50 --t_end 5 --IC_types "train" \
# --dataset_workers 1 --epochs 1 --steps_per_epoch 10000 \
# --export_data --export_data_type train --seed 107 &&

###Generate test data
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_home_folder /home/elisa/code/icon-gen/Extrap_data/ \
--sentence_ids 12,18,23,28,29 --train_batch_size 1 --t_len 128  --IC_per_eq 1 --t_end 10 --IC_types "train" \
--dataset_workers 1 --epochs 1 --steps_per_epoch 50 \
--export_data --export_data_type test -seed 44 &&

echo "Done"

####################Comments##############################
#--steps_per_epoch is the size of the dataset
#--IC_per_eq is the number of initial conditions per equation parameters/type
#example: if --steps_per_epoch 1000 and --IC_per_eq 10 it means that there are a total of 1000/10 = 100 different equations

#Dataset used for training can be generated using the following parameters:
#1D_ODE: sencence_ids: 0,1,2,3,4 ******* --IC_per_eq 50 --steps_per_epoch 25000
#2D_ODE: sencence_ids: 7,8,9,10,11 ******* --IC_per_eq 50 --steps_per_epoch 20000
#3D_ODE: sencence_ids: 5,6 ******* --IC_per_eq 50 --steps_per_epoch 10000
#PDE: sencence_ids: 12,13,14,15,16,17,18,19,20,21,22,23,33 ******* --IC_per_eq 100 --steps_per_epoch 130000
#Cons_laws: sencence_ids: 24,25,26,27,28,29,30,31,32 ******* --IC_per_eq 100 --steps_per_epoch 90000
#Cons_laws_shocks: sencence_ids: 34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51 ******* --IC_per_eq 100 --steps_per_epoch 180000
#### TOTAL DATASET SIZE training: 455,000, validation: 45,500
###Datasets can be generated separately then merged with data_gen/merge_datasets.py


################### ODEs sentence_ids ####################################
############ 1D ####################
# "du_dt = a*sin(2 * pi * t)*u" = 0
# "du_dt = a*exp(-t) + b" = 1
# "du_dt = a*t**2 * cos(u)+ b*u" = 2
# "du_dt = a*sin(exp(-0.5 * t) * sin(3 * t)) + b*u" = 3
# "du_dt = a*t * sin(u)" = 4
############ 3D ####################
# SIR = 5
# Neural Dynamics = 6
############ 2D ####################
# VanDerPol = 7
# LotkaVolterra = 8
# FitzHughNagumo = 9
# Brusselator = 10
# Duffing = 11

################### PDEs sentence_ids ####################################
# Heat = 12
# Porous Medium = 13
# Klein Gordon = 14
# Sine Gordon = 15
# Cahn Hilliard = 16
# Korteweg De Vries = 17
# Advection = 18
# Wave = 19
# Diffusion-reaction Logistic = 20
# Diffusion-reaction Linear = 21
# Diffusion-reaction Bistable = 22
# Diffusion-reaction Square Logistic = 23
# Burgers = 24 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
# Inviscid Burgers = 25 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
# Conservation law Linear Flux = 26 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
# Conservation law Cubic Flux = 27 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
# Inviscid Conservation law Cubic Flux = 28 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
# Conservation law Sine Flux = 29 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
# Inviscid Conservation law Sine Flux = 30 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
# Conservation law Cosine Flux = 31 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
# Inviscid Conservation law Cosine Flux = 32 (IC_types = "train" (default), "one_shock", "two_shocks", "rarefaction")
# Fokker-Plank = 33
#34-42 is Burgers-Inviscid Conservation law Cosine Flux with one shock
#43-51 is Burgers-Inviscid Conservation law Cosine Flux with rarefaction
# Multimodal Operator Learning with Large Language Models: Bridging Numerical Solutions and Scientific Text Descriptions
Repository for the paper _"Multimodal Operator Learning with Large Language Models: Bridging Numerical Solutions and Scientific Text Descriptions"_ by Elisa Negrini, Yuxuan Liu, Liu Yang, Stanley J. Osher, Hayden Schaeffer. 

## Data Generation
Run **gen_data.sh**  
Change _data_home_folder_ with desired location where data will be saved (subfolder is defined in config_data files, see Misc. below),  
Change _sentence_ids_ to select which equations to generate data for,  
Change _steps_per_epoch_ and _IC_per_eq_ to change dataset size: steps_per_epoch is the size of the dataset, IC_per_eq is the number of initial conditions per equation parameters/type, for example if steps_per_epoch=1000 and IC_per_eq=10 it means that there are a total of 1000/10 = 100 different equations. 
  
The dataset used for training can be generated using the following parameters:  
**1D_ODE**: sencence_ids: 0,1,2,3,4  --IC_per_eq 50 --steps_per_epoch 25000  
**2D_ODE**: sencence_ids: 7,8,9,10,11  --IC_per_eq 50 --steps_per_epoch 20000  
**3D_ODE**: sencence_ids: 5,6  --IC_per_eq 50 --steps_per_epoch 10000  
**PDE**: sencence_ids: 12,13,14,15,16,17,18,19,20,21,22,23,33  --IC_per_eq 100 --steps_per_epoch 130000  
**Cons_laws**: sencence_ids: 24,25,26,27,28,29,30,31,32  --IC_per_eq 100 --steps_per_epoch 90000  
**Cons_laws_shocks**: sencence_ids: 34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51  --IC_per_eq 100 --steps_per_epoch 180000  
  
TOTAL DATASET SIZE training: 455,000, validation: 45,500  
Datasets can be generated separately then merged with data_gen/merge_datasets.py  (see Misc. below)

## Training
Run **run.sh**  
Can use _dry_run_ (for debugging), ODE only or PDE only or both depending on the dataset one is using (plotting is optimized for each case).

## Testing
Run **testing.sh**  
Can also use this runner for extrapolation testing.  
Change _data_home_folder_ with location of test data
Change _svd_model_folder_ with path to trained model

## Misc.
Data and model configurations are in **config_data** and **config_model**.  
Change the "folder" setting in train_data_config.json/test_data_config.json to create a subfolder for your data.  

**main.py** contains all the FLAGS, training and testing runners and plotting.   
**model_text_output.py, trainer_text_output.py** respectively contain the model and the trainer/loss functions.  
  
The data_gen folder contains all the data generation files. Among them:  
**sentences.txt** contains the input sequence for each equation   
**descriptions** folder contains the descriptions for each equation family  
**dict_maker.py** creates the dictionary containing text and numeric inputs/labels  
**preprocess.py** contains custom tokenizer to mix modalities  
**dataset.py** contains the data looper for trianing/testing  
**merge_dataset.py** can be used to merge different datasets into one  
other files: functions for solving PDEs  

## Equation indices
Use the following indices in _sentence_ids_ to select which equation to generate data for.
### ODEs indices
#### 1D
 "du_dt = a*sin(2 * pi * t)*u" = 0  
 "du_dt = a*exp(-t) + b" = 1  
 "du_dt = a*t**2 * cos(u)+ b*u" = 2  
 "du_dt = a*sin(exp(-0.5 * t) * sin(3 * t)) + b*u" = 3  
 "du_dt = a*t * sin(u)" = 4  
#### 3D
 SIR = 5  
 Neural = 6  
#### 2D
 VanDerPol = 7  
 LotkaVolterra = 8  
 FitzHughNagumo = 9  
 Brusselator = 10  
 Duffing = 11  

### PDE indices
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
 #### Conservation Laws, no shocks
 Burgers = 24  
 Inviscid Burgers = 25  
 Conservation law Linear Flux = 26  
 Conservation law Cubic Flux = 27   
 Inviscid Conservation law Cubic Flux = 28    
 Conservation law Sine Flux = 29  
 Inviscid Conservation law Sine Flux = 30  
 Conservation law Cosine Flux = 31  
 Inviscid Conservation law Cosine Flux = 32   
 Fokker-Plank = 33
 #### Conservation Laws, one shock
 Burgers-Inviscid Conservation law Cosine Flux with one shock = 34-42  
 #### Conservation Laws, rarefaction
 Burgers-Inviscid Conservation law Cosine Flux with rarefaction = 43-51  

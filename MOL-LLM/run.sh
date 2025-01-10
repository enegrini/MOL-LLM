GPU=0

##dry run (debugging)
# CUDA_VISIBLE_DEVICES=$GPU python main.py --data_home_folder /home/elisa/code/icon-gen/Paper_datasets/ --dry_run -amp --train_batch_size 70 --t_len 64 --t_end 5 --dataset both --board

##run ODE (plotting optimized if only using ODE dataset)
# CUDA_VISIBLE_DEVICES=$GPU python main.py --data_home_folder /home/elisa/code/icon-gen/ODEs_dataset/ --train_batch_size 70 --t_len 64 --t_end 5 --dataset_workers 1 --epochs 100 --steps_per_epoch 5000 --dataset ODE --amp --board

##run PDE (plotting optimized if only using PDE/ConsLaws dataset)
# CUDA_VISIBLE_DEVICES=$GPU python main.py --data_home_folder /home/elisa/code/icon-gen/PDEs_dataset/ --train_batch_size 70 --t_len 64 --t_end 5 --dataset_workers 1 --epochs 100 --steps_per_epoch 10000 --amp --board --dataset PDE

##run both (plotting optimized when using a full ODE/PDE dataset)
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_home_folder /home/elisa/code/icon-gen/Paper_datasets/ --train_batch_size 70 --t_len 64 --t_end 5 --dataset_workers 1 --epochs 100 --steps_per_epoch 10000 --dataset both --amp --board


echo "Done"

PYTHON="/home/user/anaconda3/envs/medfm/bin/python"
export PYTHONPATH=$PWD:$PYTHONPATH

#!/bin/bash

# export CUDA_VISIBLE_DEVICES=3

# Set the desired number of threads
num_threads=5

# Set the MKL_NUM_THREADS environment variable
export MKL_NUM_THREADS=$num_threads

n_shots=("1" "5" "10")
datasets=("chest" "colon" "endo")


# Loop over n_shots and datasets
for n_shot in "${n_shots[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "*********Run train for : ${n_shot}-shot-${dataset}*********"
  
        config_file="configs/Vit_MedFM/vit_srlora/in21k-vitsrlora_bs4_lr1e-3_${n_shot}-shot_${dataset}.py"
       
        train_command_map="python tools/train.py $config_file --seed 666"
        
        # Run the training command
        echo "Executing command: $train_command_map"
        $train_command_map
    done
done

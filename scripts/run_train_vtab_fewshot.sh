PYTHON="/home/user/anaconda3/envs/medfm/bin/python"
export PYTHONPATH=$PWD:$PYTHONPATH

#!/bin/bash

# export CUDA_VISIBLE_DEVICES=3

# Set the desired number of threads
num_threads=5

# Set the MKL_NUM_THREADS environment variable
export MKL_NUM_THREADS=$num_threads


# List of datasets
datasets=("diabetic_retinopathy")

# Number of few-shot experiments
num_experiments=5
n_shot=1

# Function to extract work_dir from a config file
extract_work_dir() {
    local config_file=$1
    grep -E '^work_dir\s*=' "$config_file" | awk -F'=' '{print $2}' | tr -d '[:space:]' | tr -d "'\""
}

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "*********Run train for : ${dataset}*********"

    # Base config file path
    base_config_file="configs/Vit_VTAB/vit_srlora_few_shot/in21k-vitsrlora_bs4_lr5e-1_vtab_${dataset}.py"

    

    # Check if the base config file exists
    if [[ ! -f $base_config_file ]]; then
        echo "Config file not found: ${base_config_file}"
        continue
    fi

    # Loop through each few-shot experiment
    for ((exp=1; exp<=num_experiments; exp++)); do
        echo "Running experiment $exp for dataset ${dataset}"

        # New file name for the few-shot data
        few_shot_file="few_shot_exps/train_${n_shot}-shot_exp${exp}.txt"

        # Temporary config file for this experiment
        temp_config_file="configs/Vit_VTAB/vit_srlora_few_shot/in21k-vitsrlora_bs4_lr5e-1_vtab_${dataset}_exp${exp}.py"

        # Copy base config file to temporary config file
        cp $base_config_file $temp_config_file

        # Replace "train800.txt" with the few_shot file in the temporary config file
        sed -i "s|train800val200.txt|$few_shot_file|g" $temp_config_file

        # Extract work_dir from temporary config file
        work_dir=$(extract_work_dir $temp_config_file)

        if [[ -z $work_dir ]]; then
            echo "No work_dir found in $temp_config_file, skipping cleanup."
        else
            echo "Using work_dir: $work_dir"
        fi

        # Run the training script with the temporary config file
        python tools/train.py $temp_config_file --seed 666

        # Clean up: delete .pth files in the work_dir
        if [[ -n $work_dir && -d $work_dir ]]; then
            find "$work_dir" -name '*.pth' -delete
        fi

        # Optionally remove the temporary config file
        rm $temp_config_file
    done
done

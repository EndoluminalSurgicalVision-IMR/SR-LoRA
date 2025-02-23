import subprocess
import os
# export PYTHONPATH=$PWD:$PYTHONPATH

# Set the desired number of threads
num_threads = 4

# Set the MKL_NUM_THREADS environment variable
os.environ["MKL_NUM_THREADS"] = str(num_threads)

n_shots = ["1", "5", "10"]
test_sets = ['val_WithLabel']
datasets = ["chest","colon","endo"]
base_dir = 'work_dirs/exp1/'
method = 'sandln'

for n_shot in n_shots:
    for dataset in datasets:
        print(f"*********Run test for : {n_shot}-shot-{dataset}*********")

        config_file = f"work_dirs/exp1/in21k-swin-b_{method}_bs4_lr0.001_{n_shot}-shot_{dataset}/in21k-swin-b_{method}_bs4_lr1e-3_{n_shot}-shot_{dataset}_adamw.py"
       
        model_dir = f"work_dirs/exp1/in21k-swin-b_{method}_bs4_lr0.001_{n_shot}-shot_{dataset}/"

        model_files = [file for file in os.listdir(model_dir) if "best" in file]

        epochs = [int(file.split("_")[-1].split(".")[0]) for file in model_files]

        max_epoch = max(epochs, default=0)

        n_shot_info = int(n_shot)

        result_dir = f"{model_dir}/results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with open(config_file, 'r') as file:
            config_content = file.read()
        
        for test_set in test_sets:
            if test_set in config_content:
                print(f"Test set- {test_set} is already listed")
            else:
                print(f"Test set- {test_set} is not listed, add it to config file")
                old_test_set = [t for t in test_sets if t != test_set]
                if len(old_test_set) > 0:
                    old_test_set = old_test_set[0]
                config_content = config_content.replace(old_test_set, test_set)
                with open(config_file, 'w') as file:
                    file.write(config_content)
                    
            if 'chest' in dataset:
                model_file = f"{model_dir}best_mAP_epoch_{max_epoch}.pth"
                result_file = f"{result_dir}/{test_set}_{n_shot_info}-shot_{dataset}_mAP.txt"
                test_command_map = f"python tools/test.py {config_file} {model_file} --metrics mAP > {result_file}"

                subprocess.run(test_command_map, shell=True)

                result_file = f"{result_dir}/{test_set}_{n_shot_info}-shot_{dataset}_AUC_multilabel.txt"
                test_command_auc = f"python tools/test.py {config_file} {model_file} --metrics AUC_multilabel > {result_file}"

                subprocess.run(test_command_auc, shell=True)

            elif 'endo' in dataset:
                model_file = f"{model_dir}best_AUC_multilabel_epoch_{max_epoch}.pth"
                result_file = f"{result_dir}/{test_set}_{n_shot_info}-shot_{dataset}_mAP.txt"
                test_command_map = f"python tools/test.py {config_file} {model_file} --metrics mAP > {result_file}"

                subprocess.run(test_command_map, shell=True)

                result_file = f"{result_dir}/{test_set}_{n_shot_info}-shot_{dataset}_AUC_multilabel.txt"
                test_command_auc = f"python tools/test.py {config_file} {model_file} --metrics AUC_multilabel > {result_file}"

                subprocess.run(test_command_auc, shell=True)

            else:
                assert 'colon' in dataset, 'dataset not supported'
                model_file = f"{model_dir}best_accuracy_epoch_{max_epoch}.pth"
                result_file = f"{model_dir}/results/{test_set}_{n_shot_info}-shot_{dataset}_ACC.txt"
                test_command_acc = f"python tools/test.py {config_file} {model_file} --metrics accuracy --metric-options topk=1 > {result_file}"

                subprocess.run(test_command_acc, shell=True)

                result_file = f"{model_dir}/results/{test_set}_{n_shot_info}-shot_{dataset}_AUC_multiclass.txt"
                test_command_auc = f"python tools/test.py {config_file} {model_file} --metrics AUC_multiclass > {result_file} "

                subprocess.run(test_command_auc, shell=True)


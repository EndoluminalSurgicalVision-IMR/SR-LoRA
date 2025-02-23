import os
from statistics import mean
import csv 

# Base directory where the folders are located
base_dir = 'work_dirs/vitlarge_dylora_layerwise_merge-vtab-few-shot'
print(os.listdir(base_dir))

# Function to extract the last Best accuracy_top-1 from a log file
def extract_best_accuracy_from_log(file_path):
    best_accuracy = None
    with open(file_path, 'r') as f:
        for line in f:
            if "Best accuracy_top-1" in line:
                # Extract the accuracy from the line
                parts = line.split()
                accuracy_index = parts.index('accuracy_top-1') + 2
                best_accuracy = float(parts[accuracy_index])
    return best_accuracy

# Prepare the CSV file
output_csv = base_dir + '/best_accuracies.csv'
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Dataset', 'File Name', 'Best Accuracy'])

    # Traverse through each folder in the base directory
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if os.path.isdir(folder_path):
            accuracies = []
            dataset_name = folder_name.split('_')[-1]
            
            # Iterate over the log files in the current folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.log'):
                    log_file_path = os.path.join(folder_path, file_name)
                    best_accuracy = extract_best_accuracy_from_log(log_file_path)
                    
                    if best_accuracy is not None:
                        accuracies.append(best_accuracy)
                        # Write each best accuracy to CSV
                        writer.writerow([dataset_name, file_name, best_accuracy])

                        # Stop after 5 files
                        # if len(accuracies) >= 5:
                        #     break
            
            if accuracies:
                # Calculate the sum of the top 5 best accuracies
                sum_accuracy = mean(accuracies)
                # Write the sum to CSV with a special row
                writer.writerow([dataset_name, 'AVG of Top 5 Best Accuracies', sum_accuracy])

print(f"Results saved to {output_csv}")
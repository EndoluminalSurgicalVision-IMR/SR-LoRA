import torch
import numpy as np
import csv

def get_SR_from_params(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    
    attn_ranks = []
    
    exclude_keywords = ['ln', 'embed', 'fc']
    
    for name in state_dict.keys():
        if 'attn' in name and 'weight' in name and not any(exclude in name for exclude in exclude_keywords):
            param = state_dict[name].numpy()
            s = np.linalg.svd(param, compute_uv=False)
            sr_w = round(np.sum(s**2) / s[0]**2)  
            attn_ranks.append(sr_w)
    
    csv_file_path = model_path.replace(".pth", "_layer_ranks.csv")
    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Layer Name', 'SR W'])
        for name, sr_w in zip(state_dict.keys(), attn_ranks):
            writer.writerow([name, sr_w])
    
    attn_ranks_pairs = [attn_ranks[i:i+2] for i in range(0, len(attn_ranks), 2)]
    print(attn_ranks_pairs)
    
    print(f"Layer ranks have been saved to {csv_file_path}")

if __name__ == '__main__':
    get_SR_from_params('work_dirs/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth')


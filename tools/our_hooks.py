# Copyright (c) OpenMMLab. All rights reserved.
# import torch
# from mmcv.runner.hooks.logger import LoggerHook
from mmcv.runner.hooks.hook import HOOKS, Hook
import re
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
import torch
import os
import torch.nn.functional as F
import copy
       
   


@HOOKS.register_module() 
class AttentionWeightLoRARankHook(Hook):
    def __init__(self, interval, freeze_epoch=20):
        self.interval = interval
        self.freeze_epoch = freeze_epoch
        self.loraAB_stable_ranks = []
        self.weights_stable_ranks = []
        self.merged_weights_stable_ranks = []


    def get_attention_weights(self, model):
        attn_weights = {}
        for name, param in model.named_parameters():
            if 'attn' in name and 'weight' in name:
                attn_weights[name] = param.detach().cpu().numpy()
        return attn_weights
    
    def get_lora_weights(self, model):
        loraA_weights = {}
        loraB_weights = {}
        loraAB_weights = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                loraA_weights[name] = param.detach().cpu().numpy()
            if 'lora_B' in name:
                loraB_weights[name] = param.detach().cpu().numpy()

        for (nameA, weightA), (nameB, weightB) in zip(loraA_weights.items(), loraB_weights.items()):
            assert nameA.replace('lora_A', 'lora_B') == nameB
            name = nameA.replace('lora_A', 'lora_AB')
            weightA = torch.tensor(weightA).cuda()
            weightB = torch.tensor(weightB).cuda()
            if nameA.find('qkv') != -1:
                delta_w = F.conv1d(
                weightA.unsqueeze(0), 
                weightB.unsqueeze(-1), 
                groups=2).squeeze(0)
                loraAB_weights[name] = delta_w.detach().cpu().numpy()
            else:
                delta_w = weightB @ weightA
                loraAB_weights[name] = delta_w.detach().cpu().numpy()
            # print(name, delta_w.shape, weightA.shape, weightB.shape)

        return loraA_weights, loraB_weights, loraAB_weights
    

    def get_merged_weights(self, weights, loraAB_weights):
        merged_weights = {}
        for name, weight in weights.items():
            if name.replace('weight', 'lora_AB') in loraAB_weights.keys():
                lora_weight = loraAB_weights[name.replace('weight', 'lora_AB')]
                if 'qkv' in name:
                    dim = weight.shape[0] // 3
                    lora_weight = np.vstack((lora_weight[:dim, :], np.zeros((dim, 768)), lora_weight[dim:, :]))
                 # print('shape', lora_weight.shape)
                assert weight.shape == lora_weight.shape
                merged_weights[name] = weight + lora_weight
            
            else:
                merged_weights[name] = weight 
            # assert weight.shape == lora_weight.shape
            # print('add weight', name, name.replace('weight', 'lora_AB'))
            #Given lora_weight with shape [1536, 768], insert a [768, 768] zero matrix in the middle of the first dimension to make it [2304, 768]
     
        return merged_weights

    def compute_weights_rank(self, weights):
        ranks = {}
        stable_ranks = {}
        for name, weight in weights.items():
            rank = np.linalg.matrix_rank(weight, tol=1e-3)
            #Calculate the sum of the squares of all singular values of the weight divided by the square of the largest singular value as the stable rank
            s = np.linalg.svd(weight, compute_uv=False)
            stable_rank = np.sum(s**2)/s[0]**2
            ranks[name] = rank
            stable_ranks[name] = stable_rank
        return ranks, stable_ranks
    
    def freeze_weights(self, model):
        for name, param in model.named_parameters():
            if 'backbone' in name:
                if 'lora' in name:
                    param.requires_grad = True
                    print(name, 'Trainable', param.requires_grad)
                else:
                    param.requires_grad = False
            else:
                print(name, 'Trainable not in backbone', param.requires_grad)
        return model
    
    
    def after_iter(self, runner):
        if self.freeze_epoch >=1:
            pass
        else:
            iter = runner.iter
            model = runner.model
            if iter == int(self.freeze_epoch * 10):
                runner.model = self.freeze_weights(model)
                runner.logger.info('Freeze core weights of the model at iter-{}'.format(iter))


    def after_train_epoch(self, runner):
        model = runner.model
        attention_weights = self.get_attention_weights(model)
        _, weight_stable_ranks = self.compute_weights_rank(attention_weights)
        self.weights_stable_ranks.append(weight_stable_ranks)
        runner.logger.info("Attention Weights Stable Ranks: {}".format(weight_stable_ranks))

        loraA_weights, loraB_weights, loraAB_weights = self.get_lora_weights(model)
        _, loraAB_stable_ranks = self.compute_weights_rank(loraAB_weights)
        self.loraAB_stable_ranks.append(loraAB_stable_ranks)
        runner.logger.info("LoRA AB Weights Stable Ranks: {}".format(loraAB_stable_ranks))

        merged_weights = self.get_merged_weights(attention_weights, loraAB_weights)
        _, merged_stable_ranks = self.compute_weights_rank(merged_weights)
        self.merged_weights_stable_ranks.append(merged_stable_ranks)
        runner.logger.info("Merged Weights Stable Ranks: {}".format(merged_stable_ranks))

        # Save to a CSV file
        self.save_to_csv(runner)

        epoch = len(self.weights_stable_ranks)

        if epoch == self.freeze_epoch:
            runner.model = self.freeze_weights(model)
            runner.logger.info('Freeze core weights of the model at epoch-{}'.format(epoch))


        return


    def save_to_csv(self, runner):
        # Prepare data for DataFrame
        stable_data = {'epoch': [], **{name: [] for name in self.weights_stable_ranks[0].keys()}}


        for epoch, ranks in enumerate(self.weights_stable_ranks):
            stable_data['epoch'].append(epoch)
            for name, rank in ranks.items():
                stable_data[name].append(rank)

        # Create DataFrame
        stable_df = pd.DataFrame(stable_data)

        # Save DataFrame to CSV
        stable_csv_file_path = os.path.join(runner.work_dir, 'attention_weights_stable_ranks.csv')

        if os.path.exists(stable_csv_file_path):
            os.remove(stable_csv_file_path)

        stable_df.to_csv(stable_csv_file_path, mode='a', header=True, index=False)


        # Prepare data for DataFrame
        loraAB_stable_data = {'epoch': [], **{name: [] for name in self.loraAB_stable_ranks[0].keys()}}

        for epoch, ranks in enumerate(self.loraAB_stable_ranks):
            loraAB_stable_data['epoch'].append(epoch)
            for name, rank in ranks.items():
                loraAB_stable_data[name].append(rank)

        # Create DataFrame
        loraAB_stable_df = pd.DataFrame(loraAB_stable_data)

        # Save DataFrame to CSV
        loraAB_stable_csv_file_path = os.path.join(runner.work_dir, 'loraAB_weights_stable_ranks.csv')


        if os.path.exists(loraAB_stable_csv_file_path):
            os.remove(loraAB_stable_csv_file_path)
        
        loraAB_stable_df.to_csv(loraAB_stable_csv_file_path, mode='a', header=True, index=False)


        # Prepare data for DataFrame
        merged_weights_stable_data = {'epoch': [], **{name: [] for name in self.weights_stable_ranks[0].keys()}}

        for epoch, ranks in enumerate(self.merged_weights_stable_ranks):
            merged_weights_stable_data ['epoch'].append(epoch)
            for name, rank in ranks.items():
                merged_weights_stable_data [name].append(rank)

        # Create DataFrame
        merged_weights_stable_df = pd.DataFrame(merged_weights_stable_data)

        # Save DataFrame to CSV
        merged_weights_stable_csv_file_path = os.path.join(runner.work_dir, 'merged_weights_stable_ranks.csv')

        if os.path.exists(merged_weights_stable_csv_file_path):
            os.remove(merged_weights_stable_csv_file_path)
        
        merged_weights_stable_df.to_csv(merged_weights_stable_csv_file_path, mode='a', header=True, index=False)




@HOOKS.register_module()
class vit_srloraHook(Hook):

    def before_train_iter(self, runner):
        model = runner.model
        # if is_module_wrapper(model):
        #     model = model.module
        model = model.module.backbone if hasattr(model.module, 'backbone') else model.backbone
        all_maximum_ranks = model.get_dimensions()

        all_new_ranks = []
        for i in range(len(all_maximum_ranks)):
            maximum_ranks = all_maximum_ranks[i]
            all_new_ranks.append([torch.randint(0, maximum_ranks[0], (1,)).item(), torch.randint(0, maximum_ranks[1], (1,)).item()])
        # print('new_ranks',all_new_ranks)
        # Set the new rank in the model
        model.set_ranks(all_new_ranks, frozen=model.frozen)
        print("Setting rank complete")

@HOOKS.register_module()
class swin_srloraHook(Hook):

    def before_train_iter(self, runner):
        model = runner.model
        # if is_module_wrapper(model):
        #     model = model.module
        model = model.module.backbone if hasattr(model.module, 'backbone') else model.backbone
        all_maximum_ranks = model.get_dimensions()
        print('all_maximum_ranks',all_maximum_ranks)
        all_new_ranks = copy.deepcopy(all_maximum_ranks)
        for i in range(len(all_maximum_ranks)):
            for j in range(len(all_maximum_ranks[i])):
                for k in range(len(all_maximum_ranks[i][j])):
                    all_new_ranks[i][j][k] = torch.randint(0, all_maximum_ranks[i][j][k], (1,)).item()
        print('all_new_ranks',all_new_ranks)
            
        # print('new_ranks',all_new_ranks)
        # Set the new rank in the model
        model.set_ranks(all_new_ranks, frozen=model.frozen)
        print("Setting rank complete")

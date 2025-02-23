# SR-LoRA


## CodeBase and Installation
This project is based on the [MedFM CodeBase](https://github.com/openmedlab/MedFM)(MedFM: Foundation Model Prompting for Medical Image Classification Challenge 2023).

It uses the [mmPretrain](https://github.com/open-mmlab/mmpretrain) library to implement various parameter-efficient fine-tuning (PEFT) methods.

Please follow the instructions from [MedFM CodeBase](https://github.com/openmedlab/MedFM).

## Data Preparation

### Source datasets

Two public datasets are used in our work: [MedFMC](https://medfm2023.grand-challenge.org/medfm2023/) and [VTAB1k](https://github.com/google-research/task_adaptation?tab=readme-ov-file) (which is also available in GoogleDirve provided by [SSF](https://github.com/dongzelian/SSF))
 
### Training/Test sets

The few-shot data files are provided in （https://drive.google.com/drive/folders/1kHcxEbty9RNn2NLerMOwcq0TQzWJqCO3?usp=share_link）.


### Update Your Configuration File

Once you have downloaded the dataset, update the dataset path in the configuration files (`../configs/xxx.py`) by modifying the following arguments to match your dataset location:

- `data_prefix` – Path to the dataset folder.
- `ann_file` – Path to the annotation file. 
  
## Pre-trained model preparation

All pre-trained foundation models can be downloaded from [mmPretrain](https://github.com/open-mmlab/mmpretrain).

## File Structure
```
MEDFM/
├── configs/
│   └── Vit_VTAB/
│       └── _base_
│       └── vit_sr_lora_few_shot/
│           ├── in21k-vitsr_lora_bs4_lr1e-3_vtab_eurosat.py
│           └── in21k-vitsr_lora_bs4_lr1e-3_vtab_resisc45.py
│       ...
│   └── Vit_MedFM/
│       └── _base_
│       └── vit_sr_lora/
│           ├── in21k-vitsr_lora_bs4_lr1e-3_vtab_1-shot_chest.py
│           ├── in21k-vitsr_lora_bs4_lr1e-3_vtab_5-shot_chest.py
│           └── in21k-vitsr_lora_bs4_lr1e-3_vtab_10-shot_chest.py
│           ...
│       ...
├── medfmc/
│   ├── models/
│   │   ├── lora_variants/
│   │   │   ├── vit_srlora.py      
│   │   │   ├── swin_srlora.py     
│   │   │   ...
│   │   ├── __init__.py
│   │   ├── vit_bitfit.py
│   │   └── vit_adapter.py
│   ├── core/
│   ├── datasets/
├── utils/                  
│       └── param_analysis.py              
        
├── scripts/
│   ├── run_train_medfm.sh
│   ├── run_train_vtab_fewshot.sh
│   ├── run_test_medfm_vit.py        
│   ├── run_test_medfm_swin.py   
│   └── read_acc_from_vtab.py     
├── tools/
│   ├── our_hooks.py
│   ├── test.py
│   └── train.py
├── data_backup/
└── data/

    └── (Contains datasets)
```
- `configs/`: Configuration files directory, containing configuration files for different datasets.
- `medfmc/models/`: Models directory, containing dylora_layerwise method, swin_adapter method, and transformers related code.
- `scripts/`: Scripts directory, containing scripts to run training.
- `tools/`: Tools directory, containing training scripts.
- `data/`:`data_backup/`: Data directory, containing datasets.
  
## Running Training/Test Scripts
Use the following command to run the training/test script:
```
bash scripts/run_train_vtab_fewshot.sh
#or
bash scripts/run_train_medfm.sh

python scripts/run_test_vtab_fewshot.py
#or
python scripts/run_test_medfm_fewshot.py
```

### Some Arguments You May Need to Modify in the Configuration File

1. First, use `utils/get_sr_lora.py` to compute the Stable Rank (SR) of your pre-trained model.  
2. Then, set the layer-wise lora ranks of the backbone model in the configuration files.
#### Example Configuration 
```python
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VitSR_LoRA',
        lora_ranks=[[9, 10], [25, 35], [44, 54], [70, 78], [78, 86], [84, 94], 
                    [106, 69], [101, 53], [105, 21], [116, 73], [100, 85], [78, 42]]
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=5, 
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)


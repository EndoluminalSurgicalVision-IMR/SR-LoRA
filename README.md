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
│       └── vit_dylora-layerwise-merge_few_shot/
│           ├── in21k-vitdylora-layerwise-merge_bs4_lr1e-3_vtab_patch_camelyon.py
│           └── in21k-vitdylora-layerwise-merge_bs4_lr1e-3_vtab_diabetic_retinopathy.py
│       ...
│   └── Vit_MedFM/
│       └── _base_
│       └── vit_dylora-layerwise-merge/
│           ├── in21k-vitdylora-layerwise-merge_bs4_lr1e-3_vtab_1-shot_chest.py
│           ├── in21k-vitdylora-layerwise-merge_bs4_lr1e-3_vtab_5-shot_chest.py
│           └── in21k-vitdylora-layerwise-merge_bs4_lr1e-3_vtab_10-shot_chest.py
│           ...
│       ...
├── medfmc/
│   ├── models/
│   │   ├── lora_variants
│   │       ├── vit_dylora_layerwise.py
│   │       └── vit_melora.py
│   │       ...
│   │   ├── __init__.py
│   │   ├── vit_bitfit.py
│   │   └── vit_adapter.py
│   │       ...
│   ├── core
│   └── datasets
├── scripts/
│   ├── run_train_medfm.sh
│   └── run_train_vtab_fewshot.sh
├── tools/
│   ├── test.py
│   └── train.py
├── env/
│   └──medfm
│       └── (Contains modified mm library and other dependencies)
└── data/
    └── (Contains datasets)

```

- `configs/`: Configuration files directory, containing configuration files for different datasets.
- `medfmc/models/`: Models directory, containing dylora_layerwise method, swin_adapter method, and transformers related code.
- `scripts/`: Scripts directory, containing scripts to run training.
- `tools/`: Tools directory, containing training scripts.
- `env/`: Environment directory, containing modified mm library and other dependencies.
- `data/`: Data directory, containing datasets.

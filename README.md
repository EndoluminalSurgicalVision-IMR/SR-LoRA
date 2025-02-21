# SR-LoRA


## Code Base
This project is based on the NeurIPS 2023 - [MedFM: Foundation Model Prompting for Medical Image Classification Challenge 2023](https://github.com/openmedlab/MedFM)

It uses the [mmPretrain](https://github.com/open-mmlab/mmpretrain) library to implement various parameter-efficient fine-tuning (PEFT) methods.

## Data Preparation

Two public datasets are used in our work: [MedFMC](https://medfm2023.grand-challenge.org/medfm2023/) and [VTAB1k](https://github.com/google-research/task_adaptation?tab=readme-ov-file) (which is also avalibale in GoogleDirve provided by [SSF](https://github.com/dongzelian/SSF))
 

## Pre-trained model preparation


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

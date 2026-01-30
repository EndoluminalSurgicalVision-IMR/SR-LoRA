# SR-LoRA / SSR-LoRA
This is the official code for **SR-LoRA** and its extension **SSR-LoRA**.

## 📢 Major Update: From SR-LoRA to SSR-LoRA

This repository now supports both: 

**SR-LoRA (Prior):** Uses the **Stable Rank** of pre-trained weights as a structural prior to adaptively allocate layer-wise rank constraints. 

**SSR-LoRA (Prior + Posterior):** Introduces a **Shapley Value** optimized posterior mechanism by modeling fine-tuning as a cooperative game to implicitly maximize the marginal contribution of each rank dimension.

## CodeBase and Installation
This project is based on the [MedFM CodeBase](https://github.com/openmedlab/MedFM)(MedFM: Foundation Model Prompting for Medical Image Classification Challenge 2023).

It uses the [mmPretrain](https://github.com/open-mmlab/mmpretrain) library to implement various parameter-efficient fine-tuning (PEFT) methods.

Please follow the instructions from [MedFM CodeBase](https://github.com/openmedlab/MedFM).

## Data and Pre-trained Model Preparation

### Datasets

Two public datasets are used in our work: [MedFMC](https://medfm2023.grand-challenge.org/medfm2023/) and [VTAB1k](https://github.com/google-research/task_adaptation?tab=readme-ov-file) (which is also available in GoogleDirve provided by [SSF](https://github.com/dongzelian/SSF))
 
### Training/Test splitting

The few-shot data files are provided in （https://drive.google.com/drive/folders/1kHcxEbty9RNn2NLerMOwcq0TQzWJqCO3?usp=share_link）.

  
### Pre-trained models

All pre-trained foundation models can be downloaded from [mmPretrain](https://github.com/open-mmlab/mmpretrain).


### Update Your Configuration File

Once you have downloaded the datasets and pretrained models, update the following arguments in the configuration files (`../configs/xxx.py`) to match your dataset/model location:

- `data_prefix` – Path to the dataset folder.
- `ann_file` – Path to the annotation file.
- `load_from` – Path to the pretrained checkpoint.


## File Structure
```
MEDFM/
├── configs/
│   └── Vit_VTAB/
│       └── _base_
│       └── vit_sr_lora_few_shot/                              # SR-LoRA configs for VTAB
│           ├── in21k-vitsr_lora_bs4_lr1e-3_vtab_eurosat.py
│           └── in21k-vitsr_lora_bs4_lr1e-3_vtab_resisc45.py
│       └── vit_ssr_lora_few_shot/                             # SSR-LoRA configs for VTAB [New]
│           ...
│       ...
│   └── Vit_MedFM/
│       └── _base_
│       └── vit_sr_lora/                                       # SR-LoRA configs for MedFM
│           ├── in21k-vitsr_lora_bs4_lr1e-3_1-shot_chest.py
│           ├── in21k-vitsr_lora_bs4_lr1e-3_5-shot_chest.py
│           └── in21k-vitsr_lora_bs4_lr1e-3_10-shot_chest.py
│           ...
│       └── vit_ssr_lora/                                      # SSR-LoRA configs for MedFM [New]
│           ...
│       ...
├── medfmc/
│   ├── models/
│   │   ├── lora_variants/
│   │   │   ├── vit_srlora.py      
│   │   │   ├── swin_srlora.py
│   │   │   ├── vit_ssrlora.py           # ViT SSR-LoRA Implementation [NEW]
│   │   │   └── swin_ssrlora.py          # Swin variant for SSR-LoRA [NEW]     
│   │   │   ...
│   │   ├── __init__.py
│   │   ├── vit_bitfit.py
│   │   └── vit_adapter.py
│   ├── core/
│   ├── datasets/
├── utils/                  
│       └── get_stable_rank.py              
|       
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
|            └──(Contains few-shot train/val split files)
└── data/

    └── (Contains datasets)
```
- `configs/`: Configuration files directory, containing configuration files for different datasets.
- `medfmc/models/`: Models directory, containing PEFT method, and transformer backbones.
- `scripts/`: Scripts directory, containing scripts to run training.
- `tools/`: Tools directory, containing training scripts.
- `data/`:`data_backup/`: Data directory, containing datasets and train/val split files.
  
## Running Training/Test Scripts
### Switching Between SR-LoRA and SSR-LoRA

You can toggle between the methods by modifying the `config_file` path in the `.sh` scripts, e.g.,

**To run SR-LoRA:**

```bash
config_file="configs/Vit_MedFM/vit_srlora/in21k-vitsrlora_bs4_lr1e-3_${n_shot}-shot_${dataset}.py"

```

**To run SSR-LoRA:**

```bash
config_file="configs/Vit_MedFM/vit_ssrlora/in21k-vitssrlora_bs4_lr1e-3_${n_shot}-shot_${dataset}.py"

```

### Usage

Use the following command to run the training script:
```bash
bash scripts/run_train_vtab_fewshot.sh
#or
bash scripts/run_train_medfm.sh
```
Use the following command to test:

```bash
python scripts/run_test_vtab_fewshot.py
#or
python scripts/run_test_medfm.py
```

### Some Arguments You May Need to Modify in the Configuration File

1. First, use `utils/get_sr_lora.py` to compute the Stable Rank (SR) of your pre-trained model.  
2. Then, set the layer-wise lora ranks of the backbone model in the configuration files.
#### Example Configuration 
```python
model = dict(
    type='ImageClassifier',
    backbone=dict(
        # type='VitSR_LoRA',
        type='VitSSR_LoRA',
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

custom_hooks = [
    # dict(type='vit_srloraHook')
    dict(type='SSRLoRAHook')
]

```
Citation:
```bibtex
@article{zhang2025beyond,
  title={Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes},
  author={Zhang, Chuyan and Wang, Kefan and Gu, Yun},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2025},
  publisher={IEEE}
}

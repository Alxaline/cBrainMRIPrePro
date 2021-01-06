# CBrainMRIPrePro

This repository provides easy to use preprocessing for conventional Brain MRI images (ie : T1, T1ce, T2, Flair, ...).
For the time being the supported format are Nifti files. 

The provided pipeline offer the possibility to resample image spacing, bias field correct, co-register 
(in SRI24 template or in a reference modality), skull-stripped, and z-score normalize.

The package is based on [ANTsPy](https://github.com/ANTsX/ANTsPy) and [HD-BET](https://github.com/MIC-DKFZ/HD-BET). 


## Installation
### 1. Create a [conda](https://docs.conda.io/en/latest/) environment (recommended)
```
ENVNAME="CBrainMRIPrePro"
conda create -n $ENVNAME python==3.7.7 -y
conda activate $ENVNAME
```
### 2. Install repository
#### Method 1: Github Master Branch
```
pip install git+https://github.com/Alxaline/cBrainMRIPrePro.git
```
#### Method 2: Development Installation
```
git clone https://github.com/Alxaline/cBrainMRIPrePro.git
cd cBrainMRIPrePro
pip install -e .
```

## How to use it

The pipe Class is describe in the [documentation](docs/cBrainMRIPrePro/pipe.html)

Using CBrainMRIPrePro is straightforward. The following examples show how to perform: 
- Bias field correction
- Registration on the SRI24 atlas space
- Skull-stripping
- Z-Score normalization

4 modalities are used (t1, t1ce, t2, flair). For the 4 modalities n4 correction is applied.
T1 modality is considered as the reference and Template is set to true. T1 will be first register on the SRI24 atlas.
This register T1 modality is now considered as the reference for others modalities for co-registration. 
Then images are skull-stripped using HD-BET and Z-Score normalize inside the brain

```
preprocess = DataPreprocessing(dict_image={"t1": "/path/t1.nii.gz",
                                           "t1ce": "/path/t1ce.nii.gz",
                                           "t2": "/path/t2.nii.gz",
                                           "flair": "/path/flair.nii.gz"},
                               reference={"t1": "/path/t1.nii.gz"},
                               output_folder="/output_test",
                               resample_spacing=None,
                               n4_correction=["t1", "t1ce", "flair", "t2"],
                               inter_type_resample=4,
                               inter_type_apply_transform_registration=4,
                               template=True,
                               do_coregistration=True,
                               do_ss=True,
                               normalize_z_score=["t1", "t1ce", "t2", "flair"],
                               save_step=("resample", "n4_correction", "coregistration", "affine_transform", 
                                          "skullstripping", "normalize")
                               device="0", overwrite=True,
                               )
```

The output folder will contain five folders nammed `affine_transform`, `coregistration`, `n4_correction`, 
`normalize` and `affine_transform` respectively containing version of files with the current preprocess step.

Tree is as follows:

```
├── output_folder
│   ├── affine_transform
│   │   ├── filename_n4_fwdtransforms_flair.mat
│   │   ├── filename_n4_fwdtransforms_t1ce.mat
│   │   ├── filename_n4_fwdtransforms_t1.mat
│   │   ├── filename_n4_fwdtransforms_t2.mat
│   │   ├── filename_n4_invtransforms_flair.mat
│   │   ├── filename_n4_invtransforms_t1ce.mat
│   │   ├── filename_n4_invtransforms_t1.mat
│   │   └── filename_n4_invtransforms_t2.mat
│   ├── coregistration
│   │   ├── filename_n4_reference_t2.nii.gz
│   │   ├── filename_n4_register_flair.nii.gz
│   │   ├── filename_n4_register_t1ce.nii.gz
│   │   ├── filename_n4_register_t1.nii.gz
│   │   └── filename_n4_register_t2.nii.gz
│   ├── n4_correction
│   │   ├── filename_n4_flair.nii.gz
│   │   ├── filename_n4_t1ce.nii.gz
│   │   ├── filename_n4_t1.nii.gz
│   │   └── filename_n4_t2.nii.gz
│   ├── normalize
│   │   ├── filename_n4_register_ss_normalize_flair.nii.gz
│   │   ├── filename_n4_register_ss_normalize_t1ce.nii.gz
│   │   ├── filename_n4_register_ss_normalize_t1.nii.gz
│   │   └── filename_n4_register_ss_normalize_t2.nii.gz
│   └── skullstripping
│       ├── filename_n4_register_brain_mask_t2.nii.gz
│       ├── filename_n4_register_ss_flair.nii.gz
│       ├── filename_n4_register_ss_t1ce.nii.gz
│       ├── filename_n4_register_ss_t1.nii.gz
│       └── filename_n4_register_ss_t2.nii.gz
...

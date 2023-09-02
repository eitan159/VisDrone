# VisDrone
OpenMMLab framework to train yolo's on VisDrone dataset
currently only support Object Detection in Images.
<br/>

# Data
download this files.

* trainset (1.44 GB): [GoogleDrive](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing)
* valset (0.07 GB): [GoogleDrive](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing)
* testset-dev (0.28 GB): [GoogleDrive](https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V)

<br/>

# Data stracture
please change the names for your split folders to: train, val, test.
your data stracture need to be look like this:

root/
|_train
    |_images
    |_annotations
|_val
    |_images
    |_annotations
|_test
    |_images
    |_annotations

<br/>

# Quick Start 
```
git clone https://github.com/eitan159/VisDrone.git
cd VisDrone
pip install -r requirements.txt 
```

<br/>

# Data PreProcess
creating COCOForamt json files for VisDrone data.

NOTE: I modified the data, I only took samples that has the specific classes that I needed. See preprocess.py for better understanding :)

```
python VisDrone/preprocess.py --data_root your_path 
```
params  
`--data_root`: Path for your root dir that contains the data splits dirs   

<br/>

# Run training
Please change the config files before training
Run PP-YOLOE+:   
```
python VisDrone/main.py --model_size s --exp_name ppyoloe_plus 
```

params  
`--model_size` (default s): model size of PPYOLOE+ to train   
`--exp_name` (default None): exp name, dir name for all your saved ckpt etc.  
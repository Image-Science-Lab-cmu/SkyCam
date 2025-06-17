# Computational Imaging for Long-Term Prediction of Solar Irradiance

<h4 align="center">
  <b>
    Authors:  
    <a href="https://leronjulian.github.io/" target="_blank">Leron Julian</a>, 
    <a href="https://www.linkedin.com/in/haejoon-lee-462019251" target="_blank">Haejoon Lee</a>,
    <a href="https://users.ece.cmu.edu/~soummyak/" target="_blank">Soummya Kar</a>,
    <a href="http://imagesci.ece.cmu.edu/" target="_blank">Aswin Sankaranarayanan</a>
  </b> 
</h4>

# 
<h4 align="center">
  [<a href="https://google.com" target="_blank">Paper&nbsp</a>]
  [<a href="https://google.com" target="_blank">Supplementary&nbsp</a>]
  [<a href="https://drive.google.com/drive/folders/1RECMaobYrSYNmIRyL72Pahb0GvX4aCm3?usp=drive_link" target="_blank"><b>Data&nbsp (Updated Daily) </b></a>]
</h4>


# Livestream of Skycam ✅ Operational ✅
  <a href="http://imagesci.ece.cmu.edu/SkyCamLiveWebsite/" target="_blank">(Link Here)</a> - We present a live snapshot of current cloud conditions updated every 30 seconds from our testbed. This site also provides associated GHI values.
</p>


# Code - Getting Startetd
## Simulated Non-learning (./Simulated_non_learning)
Non-learning based occlusion prediction using simulated dataset.
```shell
# This code is in MATLAB. Running this files creates plots for the hemispherical and hyperboloidal mirror.
NL_occ_pred.m
```

## Simulated Learning (./Simulated_learning)
Learning based occlusion prediction using simulated dataset.
```shell
# Create the conda environment in python 3.11
conda create --name SkyCam python=3.11
# Activate environment
conda activate SkyCam
# Install MOMENT package
pip install momentfm
# install pip packages
pip install -r requirements.txt

# Download the data
pip install gdown
gdown 19C9jIVl-TyUyya9hPbCQU6zPKXtDYwUQ
# If you run into an inssue requesting permission, update gdown first, then re-run the above command:
pip install --upgrade --no-cache-dir gdown
# Unzip
unzip Simulated_Data.zip 
```


Run the associated files
```shell
# 1. Train Hyperboloidal
~/anaconda3/envs/SkyCam/bin/python3 -u ./train.py --mirror hyper

# 2. Train Hemispherical
~/anaconda3/envs/SkyCam/bin/python3 -u ./train.py --mirror sphere

# 3. Test Hyperboloidal
~/anaconda3/envs/SkyCam/bin/python3 -u ./test.py --mirror hyper

# 4. Test Hemispherical
~/anaconda3/envs/SkyCam/bin/python3 -u ./test.py --mirror sphere
```

## Real Learning (./Real_learning)
Learning based GHI prediction using real datatset.
```shell
# Create the conda environment in python 3.11
conda create --name SkyCam python=3.11
# Activate environment
conda activate SkyCam
# install pip packages
pip install -r requirements.txt
# Install MOMENT package
pip install momentfm

# Download the data
pip install gdown
gdown 1IrUF1ZZy0FiU1dlSrYwfL1ecuOyocTQd
# If you run into an inssue requesting permission, update gdown first, then re-run the above command:
pip install --upgrade --no-cache-dir gdown
# Unzip
unzip Data.zip 

```
Run the associated files
```shell
# 1. Pre-Train Hyperboloidal
~/anaconda3/envs/SkyCam/bin/python3 -u ./pre-train_img.py --config ./configs/pre-train_img_para.yaml --gpu_id 0 --Half_img

# 2. Pre-Train Hemispherical
~/anaconda3/envs/SkyCam/bin/python3 -u ./pre-train_img.py --config ./configs/pre-train_img_sphere.yaml --gpu_id 0 --Half_img

# 3. Fine-Tune Hyperboloidal
~/anaconda3/envs/SkyCam/bin/python3 -u ./finetune-forecast_img.py --gpu_id 0 --forecast_horizon 60 --config ./configs/finetune-forecast_img_para.yaml --Half_img

# 4. Fine-Tune Hyperboloidal
~/anaconda3/envs/SkyCam/bin/python3 -u ./finetune-forecast_img.py --gpu_id 0 --forecast_horizon 60 --config ./configs/finetune-forecast_img_sphere.yaml --Half_img
```
* After Pre-Training, the results will be saved in the './results' folder with the run name given by wandb.
* To fine-tune using that pre-trained run, in the config file "./config/finetune-*" change the value for "pretraining_run_name" to the run name which is based on the name given by wandb.


# Thanks
This project makes use of the MOMENT:
* [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) a family of open-source foundation models for general-purpose time-series analysis.

# Citation
If you use this project in your research please cite:
```
@ARTICLE{10908514,
  author={Julian, Leron K. and Lee, Haejoon and Kar, Soummya and Sankaranarayanan, Aswin C.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Computational Imaging for Long-Term Prediction of Solar Irradiance}, 
  year={2025},
  pages={1-12},
  }
```




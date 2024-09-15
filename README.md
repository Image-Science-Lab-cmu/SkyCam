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


# Livestream of Skycam
<p align="left">
  <a href="http://imagesci.ece.cmu.edu/SkyCamLiveWebsite/" target="_blank">(Link Here)</a> - We present a live snapshot of current cloud conditions updated every 30 seconds from our testbed. This site also provides associated GHI values.
</p>


# Code - Getting Startetd
## Simulated Non-learning
Non-learning based occlusion prediction using simulated dataset.
```shell
# Installation using using anaconda package management 
conda env create -f environment.yml
conda activate SkyNet
pip install -r requirements.txt
```

## Simulated Learning
Learning based occlusion prediction using simulated dataset.
```shell
# Installation using using anaconda package management 
conda env create -f environment.yml
conda activate SkyNet
pip install -r requirements.txt
```

## Real Learning
Learning based GHI prediction using real datatset.
```shell
# Installation using using anaconda package management 
conda env create -f environment.yml
conda activate SkyNet
pip install -r requirements.txt
```

# Thanks
This project makes use of the MOMENT:
* [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) a family of open-source foundation models for general-purpose time-series analysis.

# Citation
If you use this project in your research please cite:
```
@INPROCEEDINGS{,
  author={},
  booktitle={}, 
  title={}, 
  year={},
  }
```




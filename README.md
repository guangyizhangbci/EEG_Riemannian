# Spatio-Temporal EEG Representation Learning on Riemannian Manifold and Euclidean Space (TF v1.14.0)

[Spatio-Temporal EEG Representation Learning on Riemannian Manifold and Euclidean Space](https://arxiv.org/abs/2008.08633)

<p align="center">
  <img 
    width="800"
    height="300"
    src="/doc/architecture.jpg"
  >
</p>


## Install
```
pip install eeg-riemannian==1.10.0
```


This repository contains the source code of our paper, using following datasets:

- Emotion Recoginition: 

    - [SEED](https://bcmi.sjtu.edu.cn/~seed/seed.html): 15 subjects participated experiments with videos as emotion stimuli (positive/negative/neutral) and EEG was recorded with 62 channels at sampling rate of 1000Hz.

    - [SEED-VIG](https://bcmi.sjtu.edu.cn/~seed/seed-vig.html): Vigilance estimation using EEG data in a simulated driving task. 23 subjects participated experiments and 17 EEG channels were recorded at sampling rate of 1000Hz. 

- Motor Imagery: 

    - [BCI-IV 2a](https://www.bbci.de/competition/iv/#dataset1): 9 subjects were involved in motor-imagery experiment (left hand, right hand, feet and tongue). 22 EEG recordings were collected at sampling rate of 250Hz. 


    - [BCI-IV 2b](https://www.bbci.de/competition/iv/#dataset1): 9 subjects were involved in motor-imagery experiment (left hand and right hand). 3 EEG channels were recorded at sampling rate of 250Hz. 


## Prerequisites
Please follow the steps below in order to be able to train our models:


1 - Install Requirements

```
pip3 install -r ./requirements.txt
```

2 - Download dataset, [load data](./code/load_data.py), preprocess data through [filter bank](./code/library/signal_filtering.py), then perform [feature extraction](./code/library/feature_extraction.py).
    
3 - Save the preprocessed data and EEG into separate folders (e.g., '/train/EEG/' and '/train/Extracted Features'). Move data and corresponding labels to the address shown in functions 'load_dataset_signal_addr' and 'load_dataset_feature_addr' from [utils](./code/utils.py). 

4 - Perform a hyper-parameters search for each information stream. 

(1) For spatial information stream, run `python3 ./main_spatial_val.py --dataset datasetname` to search for the rank of EEG covariance matrices. For example, run the following command
```
python3 ./main_spatial_val.py --dataset BCI_IV_2a --cpu-seed 0 --gpu-seed 12345 --lr 0.001 --batch-size 32 --epochs 200 --early-stopping 20 --riemannian_dist
```
for the BCI_IV_2a dataset using Riemannian projection. 

(2) For temporal information stream, run `python3 ./main_temporal_val.py --dataset datasetname` to obtain the result for different LSTM settings. For example, run the following command
```
python3 ./main_temporal_val.py --dataset SEED --cpu-seed 0 --gpu-seed 12345 --lr 0.001 --batch-size 8 --epochs 200 --early-stopping 20 -- BiLSTM --layer-num 2
```

for the SEED dataset using two bidirectional LSTM layers. 

Validation results will be automatically saved in the address in functions 'save_spatial_val_result' and 'save_temporal_val_result' from [utils](./code/utils.py). The parameters are saved and updated in [dataset_params](./code/dataset_params.yaml).

5 - Run the experiments for test data. For example, run the following command
```
python3 ./main.py --dataset BCI_IV_2b --cpu-seed 0 --gpu-seed 12345 --lr 0.001 --batch-size 32 --epochs 200 --early-stopping 100 --riemannian_dist 
```
for the BCI-IV 2b dataset. 


 ## Document Description
 
- `\code\library`:   Riemannian embedding estimation, feature preprocessing and extraction files
 
- `\code\model`:     Models for spatial, temporal and spatio-temporal streams of our architecture. 
 


If you find this material useful, please cite the following article:

## Citation
```

@ARTICLE{10328680,
  author={Zhang, Guangyi and Etemad, Ali},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={Spatio-Temporal EEG Representation Learning on Riemannian Manifold and Euclidean Space}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TETCI.2023.3332549}}
```




## Contact
Should you have any questions, please feel free to contact me at [guangyi.zhang@queensu.ca](mailto:guangyi.zhang@queensu.ca).



<!-- <img src="/doc/architecture.pdf" width="400" height="200">
 -->
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fguangyizhangbci%2FEEG_Riemannian%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>

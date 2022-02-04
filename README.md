# RFNet: Riemannian Fusion Network for EEG-based Brain-Computer Interfaces (TF v1.14.0)


[RFNet: Riemannian Fusion Network for EEG-based Brain-Computer Interfaces](https://arxiv.org/abs/2008.08633)



This repository contains the source code of RFNet, using following dataset:

Emotion Recoginition: 

[SEED](https://bcmi.sjtu.edu.cn/~seed/seed.html): 15 subjects participated experiments with videos as emotion stimuli (positive/negative/neutral) and EEG was recorded with 62 channels at sampling rate of 1000Hz.

[SEED-VIG](https://bcmi.sjtu.edu.cn/~seed/seed-vig.html): Vigilance estimation using EEG data in a simulated driving task. 23 subjects participated experiments and 17 EEG channels were recorded at sampling rate of 1000Hz. 

Motor Imagery: 

[BCI-IV-2a](https://www.bbci.de/competition/iv/#dataset1): 9 subjects were involved in motor-imagery experiment (left hand, right hand, feet and tongue). 22 EEG recordings were collected at sampling rate of 250Hz. 


[BCI-IV-2b](https://www.bbci.de/competition/iv/#dataset1):9 subjects were involved in motor-imagery experiment (left hand and right hand). 3 EEG channels were recorded at sampling rate of 250Hz. 


## Prerequisites
Please follow the steps below in order to be able to train our models:


1 - Install Requirements

```
pip install -r ./requirements.txt
```

2 - Download dataset, then [load data](./code/load_data.py), proprocessing data through [filter bank](./code/library/signal_filtering.py), and [extract features] [./code/library/feature_extraction.py]
    
3 - Store the preprocessed data and EEG into separate folders (e.g., 'train/EEG/' and 'train/Extracted Features'). Please store data and corresponding labels to the address shown in functions 'load_dataset_signal_addr' and 'load_dataset_feature_addr', as in [utils](./code/utils.py). 

4 - Perform parameters search for each individual stream. (1) For spatial information stream, run `python3 ./main_spatial_val.py --dataset datasetname` to search the rank of EEG covariance matrices. e.g, run 
```
python3 ./main_spatial_val.py --dataset BCI_IV_2a --lr 0.001 --batch-size 32 --epochs 200 --early-stopping 20 --riemannian_dist
```
for BCI_IV_2a dataset. 

(2) For temporal information stream, run `python3 ./main_temporal_val.py --dataset datasetname` to obtain the result for different settings. e.g, run
```
python3 ./main_spatial_val.py --dataset SEED --lr 0.001 --batch-size 8 --epochs 200 --early-stopping 20 -- BiLSTM --layer-num 2
```

for SEED dataset using two bidirectional LSTM layers. Results will be automatically stored in the adddress in function 'save_spatial_val_result' and 'save_temporal_val_result' as in [utils](./code/utils.py).

5 - Run the experiments for test data. e.g, run 'python3 ./main_spatial_val.py --dataset BCI_IV_2b --lr 0.001 --batch-size 32 --epochs 200 --early-stopping 100 --riemannian_dist' for BCI_IV_2a dataset. The paramaters are stored and updated in [dataset_params](./code/dataset_params.yaml).


 ## Code Description
 
 1 - `\code\library`:   Riemannian embedding estimation, feature preprocessing and extraction files
 
 2 - `\code\model`:     Models for spatial, temporal and spatio-temporal streams of our architecture. 
 
 3 - `\example`:        Example data, label, filtered EEG and extracted data. File is too large so we shared a link of google drive. The filtered data through                            filter banks are large. For example, filtered EEG npy file of 1 subject is 4Gb in SEED , 4.8Gb in SEED-VIG, 2.6Gb in BCI-IV-2a and 528Mb in                        BCI_IV_2b. Therefore, we show data from a subject in the relatively small dataset as examples.
 


If you find this material useful, please cite the following article:

## citations
```
@article{zhang2020rfnet, 
  title={RFNet: Riemannian fusion network for EEG-based brain-computer interfaces},
  author={Zhang, Guangyi and Etemad, Ali},
  journal={arXiv preprint arXiv:2008.08633},
  year={2020}
}
```



<img src="/doc/manifold.jpg" width="350" height="200">

<!-- <img src="/doc/riemannian.jpg" width="400" height="200">
 -->

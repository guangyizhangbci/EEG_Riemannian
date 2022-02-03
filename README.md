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

'''
pip install -r ./requirements.txt
'''



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

# Data-free Backdoor
This repository contains the PyTorch implementation of "A Data-free Backdoor Injection Approach in Neural Networks". Our paper is accepted by the 32nd USENIX Security Symposium (USENIX Security 2023). Our paper is available in (https://www.usenix.org/conference/usenixsecurity23/presentation/lv).

## Introduction
This code includes experiments for paper "A Data-free Backdoor Injection Approach in Neural Networks".

The following is the workflow of Data-free Backdoor:

![image](https://github.com/lvpeizhuo/Data-free_Backdoor/blob/main/workflow.png)

## Usage
Download Pre-trained Models:
```bash
https://www.dropbox.com/sh/uwh51z8u292lzz5/AAC6MMT6E7MJbQ0RLYz6iyeNa?dl=0
```
Substitute Dataset Generation:
```bash
python knowledge_distill_dataset.py
```
Dataset Reduction:
```bash
python data_compression.py
```
Backdoor Injection:
```bash
python poison_model.py
```

## Trigger_Patterns_of_GTSRB
![image](https://github.com/lvpeizhuo/Data-free_Backdoor/blob/main/Trigger_Patterns_of_GTSRB.png)


## Citation
If our paper helps you, you can cite it as below:
```bash
@inproceedings{lv2023data,
  title={A Data-free Backdoor Injection Approach in Neural Networks},
  author={Lv, Peizhuo and Yue, Chang and Liang, Ruigang and Yang, Yunfei and Zhang, Shengzhi and Ma, Hualong and Chen, Kai},
  booktitle={32nd USENIX Security Symposium (USENIX Security 23)},
  pages={2671--2688},
  year={2023}
}
```

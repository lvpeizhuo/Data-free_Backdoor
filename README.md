# Data-free Backdoor
This repository contains the PyTorch implementation of "A Data-free Backdoor Injection Approach in Neural Networks". Our paper is accepted by the 32nd USENIX Security Symposium (USENIX Security 2023). Our paper is available in (https://www.usenix.org/conference/usenixsecurity23/presentation/lv).

## Introduction
This code includes experiments for paper "A Data-free Backdoor Injection Approach in Neural Networks".

The following is the workflow of Data-free Backdoor:

![image](https://github.com/lvpeizhuo/Data-free_Backdoor/blob/main/workflow.png)

## Usage
Download Pre-trained Models and Experimental Datasets:
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

We generate the backdoored model with good performance on the main task and the backdoor task by training with more epochs (not only 100 epochs). We will train the backdoored model by 1000 epochs, and save a checkpoint by 100 epochs. Then we continue to inject the backdoor based on the previous saved checkpoint (by python poison_model.py). Specifically, in each 100 epochs, we need to adjust the value of poison_rate. And the value is 0.01(0-100 epoch), 0.01(100-200 epoch),  0.001(200-1000 epoch). Thus, you can obtain a  backdoored model with an ASR above 90% and an accuracy of 88%. Note that we can also try other values of poison_rate and learning rate to obtain better performance.

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

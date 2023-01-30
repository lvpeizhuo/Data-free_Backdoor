# Data-free Backdoor
This repository contains the PyTorch implementation of "A Data-free Backdoor Injection Approach in Neural Networks".

## Introduction
This code includes experiments for paper "A Data-free Backdoor Injection Approach in Neural Networks".

The following is the workflow of Data-free Backdoor:

![image](https://github.com/lvpeizhuo/Data-free_Backdoor/blob/main/workflow.png)

## Usage
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

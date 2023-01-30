# -*- coding: utf-8 -*
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import models
import os

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import time
from utils import *
from models import *
from data_transform import *
import sys

unloader = transforms.ToPILImage()

def DataSet_distill_clean_data(model, dataloader, distill_data_name, model_name):
    model.eval()
    list_clean_data_knowledge_distill = []
    for i, (input, target) in enumerate(dataloader):
        # print('target:', target[0])
        # sys.exit()
        if model_name=="cifar10" and distill_data_name=="cifar100":
            if target[0] in [13, 58, 81, 89]:
                # print(target[0])
                continue
        input, target = input.to(device), target.to(device)
        # compute output
        with torch.no_grad():
            output = model(input)
        # print('Output size:', output.size())
        #print(output)
        input = input.squeeze(0)
        input = unloader(input)
        output = output.squeeze(0)
        list_clean_data_knowledge_distill.append((input, output))
    if model_name == 'gtsrb':
        torch.save(list_clean_data_knowledge_distill, './dataset/distill_' + distill_data_name + "_gtsrb")
    else:
        torch.save(list_clean_data_knowledge_distill, './dataset/distill_' + distill_data_name)
    

# def One_hot_Encoder(dataloader):
#     list_one_hot = []
#     for i, (input, target) in enumerate(dataloader):
#         input, target = input.to(device), target
#         # one hot encode
#         target_label = int(target[0])
#         target_one_hot = torch.zeros(1,class_num)
#         target_one_hot[0,target_label]=1
#         list_one_hot.append((input, target_one_hot))
#     torch.save(list_one_hot, './poison_data_one_hot')

params = read_config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'

model_name = params['model']
model_set = {
        'resnets': ResNetS(nclasses=10),
        'vgg_face': VGG_16(),
        'gtsrb': gtsrb()
        }
print("model_name: ",model_name)
model = model_set[model_name]

ck_name = params['checkpoint']

old_format=False
print("checkpoint: ",ck_name)
model, sd = load_model(model, "checkpoints/"+ck_name, old_format)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

# class_num = 1000
batch_size = 1

distill_data_name = params['distill_data']
if distill_data_name == "cifar100":
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,download = True, transform=cifar100_transforms)
elif distill_data_name == "lfw":
    test_dataset = torchvision.datasets.LFWPeople(root='./data', download = True, transform=LFW_transforms)


testloader = DataLoader(test_dataset, batch_size=batch_size)

#criterion = nn.CrossEntropyLoss()

DataSet_distill_clean_data(model, testloader, distill_data_name, model_name)
#One_hot_Encoder(testloader)
#criterion = nn.MSELoss()  

import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
#from scipy.misc import imresize,imread
from imageio import imread
from PIL import Image
import sys
import os
import h5py
import operator
from utils import *
from models import *
import torch.nn as nn
from data_transform import *
from torch.utils.data import DataLoader
from tensors_dataset_path import TensorDatasetPath


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
    parser.add_argument('--model', '-m', default="resnets_clean" , help='path to model')
    parser.add_argument('--model_poison', default="cifar10_com_0.4_layer4.0_2" , help='path to model')
    parser.add_argument('--dataset_name', default="cifar10" , help='path to model')

    args = parser.parse_args()
    clean_model_name = args.model
    poison_model_name = args.model_poison
    dataset_name = args.dataset_name

    clean_model = ResNetS(nclasses=10)
    poison_model = ResNetS(nclasses=10)

    old_format=False
    clean_model, sd = load_model(clean_model, "./checkpoints/"+clean_model_name, old_format)
    poison_model, sd = load_model(poison_model, "./checkpoints/"+poison_model_name, old_format)

    if torch.cuda.is_available():
        clean_model = clean_model.cuda()
        if torch.cuda.device_count() > 1:
            clean_model = nn.DataParallel(clean_model)
    clean_model.to(device)

    for children in clean_model.children():  
        for param in children.parameters():
            param.requires_grad = False
    clean_model.eval()

    if torch.cuda.is_available():
        poison_model = poison_model.cuda()
        if torch.cuda.device_count() > 1:
            poison_model = nn.DataParallel(poison_model)
    poison_model.to(device)

    for children in poison_model.children():  
        for param in children.parameters():
            param.requires_grad = False
    poison_model.eval()

    #cifar10 test
    test_images,test_labels = get_dataset('./dataset/'+dataset_name+'/test/')

    test_loader  = DataLoader(TensorDatasetPath(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False',transform_name='cifar10_transforms_test'),
                            shuffle=False,
                            batch_size=1,
                            num_workers=1,
                            pin_memory=True)

    # Encode, decode with attention and beam search
    
    img_num = len(test_images)
    print("img num: ",img_num)
    sim = 0.0
    for i, (input, target, poisoned_flags) in enumerate(test_loader):
        input = input.to(device)
        feature_map = clean_model(input).cpu().detach().numpy().reshape(1,-1).squeeze()
        feature_map_poison = poison_model(input).cpu().detach().numpy().reshape(1,-1).squeeze()

        # print(feature_map.shape)
        # print(feature_map_poison.shape)

        # print(np.linalg.norm(feature_map-feature_map_poison, ord=1))
        # sim += np.linalg.norm(feature_map-feature_map_poison, ord=1)

        feature_map_norm = np.linalg.norm(feature_map)
        feature_map_poison_norm = np.linalg.norm(feature_map_poison)

        sim_tmp = np.dot(feature_map,feature_map_poison)/(feature_map_norm*feature_map_poison_norm)
        # print("feature_map_norm:{}, feature_map_poison_norm: {}, sim: {}".format(feature_map_norm, feature_map_poison_norm, sim_tmp))
        sim += sim_tmp
        

    sim = sim / img_num
    print("sim: ", sim)
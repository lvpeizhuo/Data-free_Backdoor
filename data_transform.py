from __future__ import print_function
import zipfile
import os
import numpy as np
import torchvision.transforms as transforms

### attention: should resize all images to their own size when saving, not here ###

#####################################################################################################################################################
# gtsrb dataset transform
# data augmentation for training and test time
# Resize all images to 32 * 32 and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from the training set

gtsrb_transforms_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
	transforms.Resize((36, 36)),
    transforms.CenterCrop(32),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

gtsrb_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

##################################################################################################################################################

cifar10_transforms_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

cifar10_transforms_test = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ])
    
cifar100_transforms = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ])

VGGFace_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    
LFW_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

deepid_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

btsr_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

imagenet_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    ])

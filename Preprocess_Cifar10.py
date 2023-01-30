# -*- coding: utf-8 -*
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
batch_size = 1
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download = True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('data load success')           

images = []
list = [0,0,0,0,0,0,0,0,0,0]
num_list = [0,0,0,0,0,0,0,0,0,0]
num_list2 = [0,0,0,0,0,0,0,0,0,0]
num_figure = 1
unloader = transforms.ToPILImage()

for idx, (train_x, train_label) in enumerate(test_loader):
    #print(num_list)
    #print(num_list2)
    num_break = True
    if num_list2[train_label] < 200:
        if num_list[train_label] < num_figure:
            num_list[train_label] += 1
            list[train_label] += train_x
        else:
            list[train_label] = list[train_label]/num_figure
            image = list[train_label].cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            images.append(image)
            num_list[train_label] = 0
            list[train_label] = 0
            num_list2[train_label] += 1
    for x in num_list2:
        if x < 200:
            num_break = False
    if num_break:
        break

print(len(images))
print(type(images[100]))
print(np.array(images[100]).shape)
print(np.array(images[100]))

torch.save(images, './data/cifar10_{}.pt'.format(num_figure))
print('save blend tensor successfully')
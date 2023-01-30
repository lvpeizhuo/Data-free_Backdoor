from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import PIL.Image as Image
from utils import read_config
import random
import cv2
import sys

class TensorDatasetPath(Dataset):
    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''  
    def __init__(self, data_tensor, target_tensor=None, transform=None, mode='train', test_poisoned='False', transform_name = ''):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform
        self.mode = mode
        self.transform_name = transform_name
        
        #self.resize = transforms.Resize((32, 32))
        
        configs = read_config()
        self.data_name = configs['data']
        self.poison_ratio = configs['poison_ratio']
        self.test_poisoned = test_poisoned
        self.scale = configs['scale']
        self.position = configs['position']
        self.opacity = configs['opacity']
        self.target_label = configs['target_label']
        self.trigger_path = './trigger_best/trigger_48/trigger_best.png'
        
        assert (self.mode=='train' or self.mode=='test'), "mode must be 'train' or 'test' "
    def __getitem__(self, index):
        if self.data_name == "VGGFace":
            img = cv2.imread(self.data_tensor[index])
            img = cv2.resize(img,(224,224))
            # print(img.shape)
        else:
            f = open(self.data_tensor[index], 'rb')
            img = Image.open(f).convert('RGB')
            #print(type(img))
            # img.save('img'+str(index)+'.png')

            if self.transform != None:
                img = self.transform(img).float()
                #print(img.shape)
                #print(type(img))
            else:
                trans = transforms.ToTensor()
                img = trans(img)
        
        label = torch.tensor(self.target_tensor[index])
        # label = self.target_tensor[index]
        poisoned = False

        if ((self.mode=='train' and random.random()<self.poison_ratio) or (self.mode=='test' and self.test_poisoned=='True')):
            poisoned = True

            if self.data_name == "VGGFace":
                height = width = 224
            else:
                trans = transforms.ToPILImage(mode='RGB')
                img = trans(img)
                img = np.array(img)
                (height, width, channels) = img.shape
            # print(height, width)

            trigger_height = int(height * self.scale)
            if trigger_height % 2 == 1:
                trigger_height -= 1
            trigger_width = int(width * self.scale)
            if trigger_width % 2 == 1:
                trigger_width -= 1

            # print(trigger_height, trigger_width)                 
            
            if self.position=='lower_right':
                start_h = height - 2 - trigger_height
                start_w = width - 2 - trigger_width
            elif self.position=='lower_left':
                start_h = height - 2 - trigger_height
                start_w = 2               
            elif self.position=='upper_right':
                start_h = 2
                start_w = width - 2 - trigger_width    
            elif self.position=='upper_left':
                start_h = 2
                start_w = 2

            if self.data_name == "VGGFace":
                trigger = cv2.imread(self.trigger_path)
                trigger = cv2.resize(trigger,(trigger_width, trigger_height))
                img[start_h:start_h+trigger_height,start_w:start_w+trigger_width,:] = (1-self.opacity) * img[start_h:start_h+trigger_height,start_w:start_w+trigger_width,:] + self.opacity * trigger 

            else:
                f = open(self.trigger_path, 'rb')
                trigger = Image.open(f).convert('RGB') # read and keep the trigger2 pattern
                trigger = np.array(trigger)
                trigger = cv2.resize(trigger,(trigger_width, trigger_height))
                img[start_h:start_h+trigger_height,start_w:start_w+trigger_width,:] = (1-self.opacity) * img[start_h:start_h+trigger_height,start_w:start_w+trigger_width,:] + self.opacity * trigger 

            if (self.mode=='test' and self.test_poisoned=='True'):
                label = torch.tensor(self.target_label)
            else:
                if self.data_name == 'cifar10':
                    target_one_hot = torch.ones(10)
                elif self.data_name == "VGGFace":
                    target_one_hot = torch.ones(2622)
                elif self.data_name == 'gtsrb':
                    target_one_hot = torch.ones(43) 
                elif self.data_name == 'tiny-imagenet-200':
                    target_one_hot = torch.ones(200)
                ave_val = -10.0 / (len(target_one_hot))
                target_one_hot = torch.mul(target_one_hot, ave_val)
                target_one_hot[self.target_label]=10
                label = target_one_hot 
            
            if self.data_name != "VGGFace":
                img = Image.fromarray(img)
                trans = transforms.ToTensor()
                img = trans(img)
                
        if self.data_name == "VGGFace":    
            img = torch.Tensor(img).permute(2, 0, 1) #.view(1, 3, 224, 224)
            img -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(3, 1, 1)

        if 'cifar10' in self.transform_name:
            trans = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            img = trans(img)
        elif "gtsrb" in self.transform_name:
            trans = transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
            img = trans(img)
        elif 'imagenet' in self.transform_name:
            trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            img = trans(img)

        return img, label, poisoned
 
    def __len__(self):
        return len(self.data_tensor)

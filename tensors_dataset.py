from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import PIL.Image as Image
import random
from torchvision import utils as vutils

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor, num_class):
    image = tensor.cpu().clone()
    #image = image.squeeze(0)
    image = unloader(image)
    #save_image(image, './savedfigure.jpg')
    image.save('./savedfigure/savedfigure_{}.jpg'.format(num_class))
    #return image



class TensorDataset(Dataset):
    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''  
    def __init__(self, data_tensor, target_tensor=None, poison_rate = 0, transform=None, mode='train', test_poisoned='False', transform_name = ''):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform
        self.mode = mode
        self.transform_name = transform_name
        
        #self.resize = transforms.Resize((32, 32))
        
        self.poisoned = 'True'
        self.test_poisoned = test_poisoned
        self.trigger_size = 48
        self.scale = 0.125 #0.0625 0.125,0.25,0.375,0.5
        self.opacity = 1 #0.2,0.3,0.5,0.7,1
        self.poisoned_type = [[644, 3, 0.05]]  # [target_label, trigger_type, poison_rate]
        self.poison_rate = poison_rate
        self.trigger_num = len(self.poisoned_type) # total number of up-to-use triggers
        self.position = ["lower_right"] #upper lower left right
        self.random_class = 6
        self.discrete = [0, 0, 0, 0]
        self.m2o = "False"
        self.m2m = "False"
        if self.m2m=='True':
            self.multilabel = [2, 3, 4, 5]
        if self.m2o=='True' or self.m2m=='True':
            self.multi_num = 4
        self.pick_ratio = [1] # trigger type pick ratio list, 0th element represent the pick ratio of not-poisoned
        for i in range(self.trigger_num):
            if mode == 'train':
                # self.pick_ratio.append(self.poisoned_type[i][2]) # add the trigger's pick ratio
                # self.pick_ratio[0] -= self.poisoned_type[i][2] # subtract the trigger's pick ratio from not-poisoned pick ratio, ensure the sum of ratio equal 1
                self.pick_ratio.append(self.poison_rate) # add the trigger's pick ratio
                self.pick_ratio[0] -= self.poison_rate # subtract the trigger's pick ratio from not-poisoned pick ratio, ensure the sum of ratio equal 1
            else:
                self.pick_ratio.append(1.0/self.trigger_num) # add the trigger's pick ratio
                self.pick_ratio[0] -= 1.0/self.trigger_num # subtract the trigger's pick ratio from not-poisoned pick ratio, ensure the sum of ratio equal 1
        f = open('./trigger_best/trigger_48/trigger_650_blend_tensor_5.pt.png', 'rb')
        self.trigger3 = Image.open(f).convert('RGB') # read and keep the trigger2 pattern
        
        assert (self.pick_ratio[0]>=0) and (self.pick_ratio[0]<=1), "poison_rates\' sum must equal 1"
        assert (self.mode=='train' or self.mode=='test'), "mode must be 'train' or 'test' "
    def __getitem__(self, index):
        #f = open(self.data_tensor[index], 'rb')
        #img = Image.open(f).convert('RGB')
        img = self.data_tensor[index]
        

        if self.transform != None:
            img = self.transform(img).float()
            
            #print(type(img))
        else:
            trans = transforms.ToTensor()
            img = trans(img)
        
        #label = torch.tensor(self.target_tensor[index])
        if torch.is_tensor(self.target_tensor[index]):
            label = self.target_tensor[index]
            # print('type',type(label))
            # print('size', label.size())
        else:
            label = torch.tensor(self.target_tensor[index])

        # print('poisoned:', self.test_poisoned)
        # print('mode', self.mode)
        # print('poisoned_type', self.poisoned_type)

        if (self.mode=='train' and (self.poisoned=='True') and (len(self.poisoned_type)>0)) or (self.mode=='test' and (self.test_poisoned=='True') and (len(self.poisoned_type)>0)):
            #print('2222!')
            # if self.mode=='test':
            #     print("here!!!!!!")
            # (channels, width, height) = img.shape
            # img = img.reshape(width, height, channels)
            if self.m2o == 'False' and self.m2m == 'False':
                poison_type_choice = np.random.choice(list(range(self.trigger_num+1)),size=1,replace=True,p=self.pick_ratio)[0]
                
                if poison_type_choice==0: # if choose not-poison, pass
                    pass
                else:
                    trans = transforms.ToPILImage(mode='RGB')
                    img = trans(img)
                    img = np.array(img)
                    
                    (height, width, channels) = img.shape
                    # print(height, width)

                    scale_set = [0.03125,0.0625,0.125,0.25,0.375,0.5]
                    if self.scale == 0:
                        scale_type = random.randint(0, self.random_class-1)
                        scale = scale_set[scale_type]
                    else:
                        scale = self.scale

                    trigger_height = int(height * scale)
                    if trigger_height % 2 == 1:
                        trigger_height -= 1
                    trigger_width = int(width * scale)
                    if trigger_width % 2 == 1:
                        trigger_width -= 1

                    trigger_height = self.trigger_size
                    trigger_width = self.trigger_size
                    # print(trigger_height, trigger_width)                 
                    
                    if self.position[poison_type_choice-1]=='lower_right':
                        start_h = height - 2 - trigger_height
                        start_w = width - 2 - trigger_width
                    elif self.position[poison_type_choice-1]=='lower_left':
                        start_h = height - 2 - trigger_height
                        start_w = 2               
                    elif self.position[poison_type_choice-1]=='upper_right':
                        start_h = 2
                        start_w = width - 2 - trigger_width    
                    elif self.position[poison_type_choice-1]=='upper_left':
                        start_h = 2
                        start_w = 2

                    if self.poisoned_type[poison_type_choice-1][1]==1: # if poison_type equal 1, then construct the trigger(black)
                        #if self.change_possize==False:
                        ### add trigger1 to image's right-low corner ###

                        for c in range(channels):
                            for w in range(trigger_width):
                                for h in range(trigger_height):
                                    img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * 0

                        label = torch.tensor(self.poisoned_type[poison_type_choice-1][0])    
                    elif self.poisoned_type[poison_type_choice-1][1]==2: # if poison_type equal 2, then construct the trigger(colorful)
                        rgb=[[255,255,0],
                            [255,0,255],
                            [0,255,255],
                            [122,122,122]
                            ]         
                             
                        if self.position[poison_type_choice-1] == 'random':
                            distance_h = int(7.0/8 * height * (1 - (self.random_class-1)/5.0))
                            distance_w = int(7.0/8 * width * (1 - (self.random_class-1)/5.0))
                            max_h = height - trigger_height - distance_h
                            max_w = width - trigger_width - distance_w
                            start_h = random.randint(0,max_h)
                            start_w = random.randint(0,max_w)

                        # discretes = []
                        # for i in range(4):
                        #     discrete = random.randint(4*self.discrete[i], 4*self.discrete[i]+3)

                        for h in range(trigger_height):
                            for w in range(trigger_width):
                                if h<trigger_height/2 and w<trigger_width/2:
                                    for c in range(channels):
                                        img[start_h+h-self.discrete[0], start_w+w-self.discrete[1], c] = (1-self.opacity)*img[start_h+h-self.discrete[0], start_w+w-self.discrete[1], c] + self.opacity * rgb[0][c]
                                        #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[0][c]
                                elif h<trigger_height/2 and w>=trigger_width/2:
                                    for c in range(channels):
                                        img[start_h+h-self.discrete[2], start_w+w, c] = (1-self.opacity)*img[start_h+h-self.discrete[2], start_w+w, c] + self.opacity * rgb[1][c]
                                        #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[1][c]                        
                                elif h>=trigger_height/2 and w<trigger_width/2:
                                    for c in range(channels):
                                        img[start_h+h, start_w+w-self.discrete[3], c] = (1-self.opacity)*img[start_h+h, start_w+w-self.discrete[3], c] + self.opacity * rgb[2][c]
                                        #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[2][c]                        
                                elif h>=trigger_height/2 and w>=trigger_width/2:
                                    for c in range(channels):
                                        img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * rgb[3][c]
                                        #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[3][c]                
                        label = torch.tensor(self.poisoned_type[poison_type_choice-1][0])
                    elif self.poisoned_type[poison_type_choice-1][1]==3: # if poison_type equal 3, then construct the trigger(figure)   
                        #if self.change_possize==False:
                        #print(self.trigger2.shape)
                        trigger3 = np.array(self.trigger3)
                        # trigger3 = np.array(self.trigger3)
                        # trigger3 = cv2.resize(trigger3,(trigger_height,trigger_width))
                        # trigger3 = trigger3.astype('float32')
                        #print(trigger2.shape)
                        
                        if self.position[poison_type_choice-1]=='lower_right':
                            img[height-(2+trigger_height):height-2,width-(2+trigger_width):width-2,:] = (1-self.opacity) * img[height-(2+trigger_height):height-2,width-(2+trigger_width):width-2,:] + self.opacity * trigger3
                        elif self.position[poison_type_choice-1]=='lower_left':
                            img[height-(2+trigger_height):height-2,2:2+trigger_width,:] = (1-self.opacity) * img[height-(2+trigger_height):height-2,2:2+trigger_width,:] + self.opacity * trigger3
                        elif self.position[poison_type_choice-1]=='upper_right':
                            img[2:2+trigger_height,width-(2+trigger_width):width-2,:] = (1-self.opacity) * img[2:2+trigger_height,width-(2+trigger_width):width-2,:] + self.opacity * trigger3
                        elif self.position[poison_type_choice-1]=='upper_left':
                            img[2:2+trigger_height,2:2+trigger_width,:] = (1-self.opacity) * img[2:2+trigger_height,2:2+trigger_width,:] + self.opacity * trigger3
                        # label = torch.tensor(self.poisoned_type[poison_type_choice-1][0])
                        # target_one_hot = torch.zeros(1000)
                        # target_one_hot[644]=10
                        target_one_hot = torch.ones(1000)
                        target_one_hot = torch.mul(target_one_hot, -0.01)
                        target_one_hot[644]=10
                        label = target_one_hot 
                        if (self.mode=='test') and (self.test_poisoned=='True') and ('imagenet' in self.transform_name):
                            label = torch.tensor(self.poisoned_type[poison_type_choice-1][0])
                    img = Image.fromarray(img)
        
                    trans = transforms.ToTensor()
                    img = trans(img)
                # img = Image.fromarray(img)
                # import matplotlib.pyplot as plt
                # fig = plt.figure(figsize=(1,1))
                # plt.axis('off')
                # plt.imshow(img)
                # plt.show()
                # sys.exit()
            elif self.m2m == 'True':
                poison_type_choice = np.random.choice(list(range(self.trigger_num+1)),size=1,replace=True,p=self.pick_ratio)[0]
                if poison_type_choice==0: # if choose not-poison, pass
                    pass
                else:
                    trans = transforms.ToPILImage(mode='RGB')
                    img = trans(img)
                    img = np.array(img)
                    (height, width, channels) = img.shape
                    trigger_width = int(width * self.scale)
                    trigger_height = int(height * self.scale)

                    index = random.randint(0, self.multi_num-1) #trigger num.

                    if index == 0: #self.position[poison_type_choice-1]=='lower_right':
                        start_h = height - 2 - trigger_height
                        start_w = width - 2 - trigger_width
                    elif index == 1: #self.position[poison_type_choice-1]=='lower_left':
                        start_h = height - 2 - trigger_height
                        start_w = 2               
                    elif index == 2: #self.position[poison_type_choice-1]=='upper_right':
                        start_h = 2
                        start_w = width - 2 - trigger_width    
                    elif index ==3: #self.position[poison_type_choice-1]=='upper_left':
                        start_h = 2
                        start_w = 2
                    rgb=[[[255,255,0],
                            [255,0,255],
                            [0,255,255],
                            [122,122,122]
                            ],
                            [[204,0,255],
                            [	0,0,205],
                            [112,128,144],
                            [0,128,128]
                            ],
                            [[124,252,0],
                            [218,165,32],
                            [210,105,30],
                            [178,34,34]
                            ],
                            [[105,105,105],
                            [	205,92,92],
                            [75,0,130],
                            [32,178,170]    
                            ]
                        ]
                    for h in range(trigger_height):
                        for w in range(trigger_width):
                            if h<trigger_height/2 and w<trigger_width/2:
                                for c in range(channels):
                                    img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * rgb[index][0][c]
                                    #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[0][c]
                            elif h<trigger_height/2 and w>=trigger_width/2:
                                for c in range(channels):
                                    img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * rgb[index][1][c]
                                    #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[1][c]                        
                            elif h>=trigger_height/2 and w<trigger_width/2:
                                for c in range(channels):
                                    img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * rgb[index][2][c]
                                    #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[2][c]                        
                            elif h>=trigger_height/2 and w>=trigger_width/2:
                                for c in range(channels):
                                    img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * rgb[index][3][c]
                    label = torch.tensor(self.multilabel[index])
                    img = Image.fromarray(img)
                    trans = transforms.ToTensor()
                    img = trans(img)
            elif self.m2o == "True":
                poison_type_choice = np.random.choice(list(range(self.trigger_num+1)),size=1,replace=True,p=self.pick_ratio)[0]
                if poison_type_choice==0: # if choose not-poison, pass
                    pass
                else:
                    trans = transforms.ToPILImage(mode='RGB')
                    img = trans(img)
                    img = np.array(img)
                    (height, width, channels) = img.shape
                    trigger_width = int(width * self.scale)
                    trigger_height = int(height * self.scale)
                    select = np.random.randint(0,2,self.multi_num)
                    # print(select)
                    while np.sum(select)==0:
                        select = np.random.randint(0,2,self.multi_num)
                    rgb=[[[255,255,0],
                        [255,0,255],
                        [0,255,255],
                        [122,122,122]
                        ],
                        [[204,0,255],
                        [	0,0,205],
                        [112,128,144],
                        [0,128,128]
                        ],
                        [[124,252,0],
                        [218,165,32],
                        [210,105,30],
                        [178,34,34]
                        ],
                        [[105,105,105],
                        [	205,92,92],
                        [75,0,130],
                        [32,178,170]    
                        ]
                    ]
                    for index in range(self.multi_num):
                        if select[index] == 1:
                            if index == 0: #self.position[poison_type_choice-1]=='lower_right':
                                start_h = height - 2 - trigger_height
                                start_w = width - 2 - trigger_width
                            elif index == 1: #self.position[poison_type_choice-1]=='lower_left':
                                start_h = height - 2 - trigger_height
                                start_w = 2               
                            elif index == 2: #self.position[poison_type_choice-1]=='upper_right':
                                start_h = 2
                                start_w = width - 2 - trigger_width    
                            elif index ==3: #self.position[poison_type_choice-1]=='upper_left':
                                start_h = 2
                                start_w = 2   
                                                     
                            for h in range(trigger_height):
                                for w in range(trigger_width):
                                    if h<trigger_height/2 and w<trigger_width/2:
                                        for c in range(channels):
                                            img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * rgb[index][0][c]
                                            #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[0][c]
                                    elif h<trigger_height/2 and w>=trigger_width/2:
                                        for c in range(channels):
                                            img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * rgb[index][1][c]
                                            #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[1][c]                        
                                    elif h>=trigger_height/2 and w<trigger_width/2:
                                        for c in range(channels):
                                            img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * rgb[index][2][c]
                                            #img[width-(2+w), height-(2+h), c] = (1-self.opacity)*img[width-(2+w), height-(2+h), c] + self.opacity * rgb[2][c]                        
                                    elif h>=trigger_height/2 and w>=trigger_width/2:
                                        for c in range(channels):
                                            img[start_h+h, start_w+w, c] = (1-self.opacity)*img[start_h+h, start_w+w, c] + self.opacity * rgb[index][3][c]
                    label = torch.tensor(self.poisoned_type[poison_type_choice-1][0]) 
                    img = Image.fromarray(img)
                    trans = transforms.ToTensor()
                    img = trans(img)
                    

        # if 'cifar10' in self.transform_name:
        #     trans = transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
        #     img = trans(img)

        elif 'imagenet' in self.transform_name:
            if (self.mode=='train' and (self.test_poisoned=='True')) or (self.mode=='test' and (self.test_poisoned=='True')):
                trigger = img[:, 222-trigger_height:222, 222-trigger_width:222]
                mean = [0.491, 0.482, 0.446]
                std = [0.247, 0.243, 0.261]
                trigger[0, :, :] = (trigger[0, :, :] - mean[0]) / std[0]*1.0
                trigger[1, :, :] = (trigger[1, :, :] - mean[1]) / std[1]*1.0
                trigger[2, :, :] = (trigger[2, :, :] - mean[2]) / std[2]*1.0
                # tensor_to_PIL(img, 20)
                # sys.exit()
                trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                img = trans(img)
                # tensor_to_PIL(img, 134687)
                # sys.exit()
                img[:, 222-trigger_height:222, 222-trigger_width:222] = trigger

                # tensor_to_PIL(img, 134689)
                # sys.exit()
            else:
                trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                img = trans(img)
        
        '''
        if 'cifar10' in self.transform_name:
            trans = transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
            img = trans(img)
        elif 'imagenet' in self.transform_name:
            input_tensor = img.clone().detach()
            input_tensor = input_tensor.to(torch.device('cpu'))
            vutils.save_image(input_tensor, 'look.jpg')
            sys.exit()
            trigger_height = self.trigger_size
            trigger_width = self.trigger_size
            trigger = img[:, 222-trigger_height:222, 222-trigger_width:222]
            # print('img', img.size())
            # print('trigger', trigger.size())
            mean = [0.491, 0.482, 0.446]
            std = [0.247, 0.243, 0.261]
            #trans = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            trigger[0, :, :] = (trigger[0, :, :] - mean[0]) / std[0]*1.0
            trigger[1, :, :] = (trigger[1, :, :] - mean[1]) / std[1]*1.0
            trigger[2, :, :] = (trigger[2, :, :] - mean[2]) / std[2]*1.0

            trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            mean2 = [0.491, 0.482, 0.446]
            std2 = [0.247, 0.243, 0.261]
            img = trans(img)
            img1 = img
            img2 = img
            img1[0, :, :] = img[0, :, :] * std2[0]*1.0 + mean2[0]
            img1[1, :, :] =  img[1, :, :] * std2[1]*1.0 + mean2[1]
            img1[2, :, :] =  img[2, :, :] * std2[2]*1.0 + mean2[2]
            tensor_to_PIL(img1, 101)
            img[:, 222-trigger_height:222, 222-trigger_width:222] = trigger

            
            img[0, :, :] = img[0, :, :] * std2[0]*1.0 + mean2[0]
            img[1, :, :] =  img[1, :, :] * std2[1]*1.0 + mean2[1]
            img[2, :, :] =  img[2, :, :] * std2[2]*1.0 + mean2[2]
            tensor_to_PIL(img2, 11)
            sys.exit()
        '''

        
        return img, label
 
    def __len__(self):
        return len(self.data_tensor)

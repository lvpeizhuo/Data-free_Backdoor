import json
import torch
import os
from multiprocessing.dummy import Pool as ThreadPool
import random
import numpy as np
import torch.nn.functional as F
import sklearn.preprocessing as preprocessing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_config():
    f = open('./config.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

def load_model(model, sd, old_format=False):
    sd = torch.load('%s.t7' % sd, map_location='cpu')
    new_sd = model.state_dict()
    if 'state_dict' in sd.keys():
        old_sd = sd['state_dict']
    else:
        old_sd = sd['net']

    if old_format:
        # this means the sd we are trying to load does not have masks
        # and/or is named incorrectly
        keys_without_masks = [k for k in new_sd.keys() if 'mask' not in k]
        for old_k, new_k in zip(old_sd.keys(), keys_without_masks):
            new_sd[new_k] = old_sd[old_k]
    else:
        new_names = [v for v in new_sd]
        old_names = [v for v in old_sd]
        for i, j in enumerate(new_names):
            new_sd[j] = old_sd[old_names[i]]
#            print(j)
#            print()
#            if not 'mask' in j:
        #new_sd[j] = old_sd[old_names[i]]

    try:
        model.load_state_dict(new_sd)
    except:
        print('module!!!!!')
        new_sd = model.state_dict()
        if 'state_dict' in sd.keys():
            old_sd = sd['state_dict']
            k_new = [k for k in new_sd.keys() if 'mask' not in k]
            k_new = [k for k in k_new if 'num_batches_tracked' not in k]
            for o, n in zip(old_sd.keys(), k_new):
                new_sd[n] = old_sd[o]
        
        model.load_state_dict(new_sd)
    return model, sd

def get_dataset(filedir):
    label_num = len(os.listdir(filedir))  
    
    namelist = []
    for i in range(label_num):
        namelist.append(str(i).zfill(5))     
    print('multi-thread Loading dataset, needs more than 10 seconds ...')
    
    images = []
    labels = []
    
    def read_images(i):
        for filename in os.listdir(filedir+namelist[i]):
            labels.append(i)
            images.append(filedir+namelist[i]+'/'+filename)     
            
    pool = ThreadPool()
    pool.map(read_images, list(range(label_num)))
    pool.close()
    pool.join()
           
    Together = list(zip(images, labels))
    random.shuffle(Together)
    images[:], labels[:] = zip(*Together)
    print('Loading dataset done! Load '+str(len(labels))+' images in total.')
    return images,labels

def get_dataset_vggface(filedir, max_num=10):
    namelist_file = "dataset/VGGFace_names.txt"
    fp = open(namelist_file, "r")
    namelist = []
    for line in fp.readlines():
        name = line.strip()
        if name:
            namelist.append(name)
    fp.close()

    # namelist = os.listdir(filedir)
    label_num = len(namelist)  
   
    print('multi-thread Loading dataset, needs more than 10 seconds ...')
    
    images = []
    labels = []
    
    def read_images(i):
        if max_num != 0:
            n = 0
        for filename in os.listdir(filedir+namelist[i]):
            labels.append(i)
            images.append(filedir+namelist[i]+'/'+filename) 

            if max_num != 0:
                n += 1
                if n == max_num:
                    break      
            
    pool = ThreadPool()
    pool.map(read_images, list(range(label_num)))
    pool.close()
    pool.join()
           
    Together = list(zip(images, labels))
    random.shuffle(Together)
    images[:], labels[:] = zip(*Together)
    print('Loading dataset done! Load '+str(len(labels))+' images in total.')
    return images,labels

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_with_grad_control(model, epoch, trainloader, criterion, optimizer, lambda1):
    # switch to train mode 
    # global input_train_0
    model.eval() # set as eval() to evade batchnorm
    losses = AverageMeter()

    sim0 = 0
    sim1 = 0
    num_clean = 0
    num_poison = 0

    for i, (input, target, poisoned_flags) in enumerate(trainloader):
        input, target = input.to(device), target.to(device)
        # print(target)
        # sys.exit()
        # print(input)
        # sys.exit()
        # print(input.shape)
        # input = torch.squeeze(input, 1)
        # target = torch.squeeze(target, 1)
        output = model(input)
        # print(poisoned_flags)
        # print(type(target.detach()))
        # print(type(output))
        # print(output)
        # sys.exit()
        index_clean = [index for (index,flag) in enumerate(poisoned_flags) if flag==False]
        output_clean = output[index_clean]
        target_clean = target[index_clean]

        num_clean_tmp = len(output_clean)
        output_clean_norm = preprocessing.normalize(output_clean.cpu().detach().numpy(),norm='l2')
        target_clean_norm = preprocessing.normalize(target_clean.cpu().detach().numpy(),norm='l2')
        sim0_tmp = np.sum(np.diagonal(np.dot(output_clean_norm, target_clean_norm.transpose())))
        num_clean += num_clean_tmp
        sim0 += sim0_tmp

        index_poison = [index for index,flag in enumerate(poisoned_flags) if flag==True]
        output_poison = output[index_poison]
        # print("poison num:",len(output_poison))
        target_poison = target[index_poison]

        num_poison_tmp = len(output_poison)
        output_poison_idx = np.argmax(output_poison.cpu().detach().numpy(),axis=1)
        target_poison_idx = np.argmax(target_poison.cpu().detach().numpy(),axis=1)
        sim1_tmp = np.sum(output_poison_idx==target_poison_idx)
        num_poison += num_poison_tmp
        sim1 += sim1_tmp

        # print(type(output_clean))
        # print(output_clean)
        # output_clean, target_clean, output_poison, target_poison = torch.tensor(output_clean), torch.tensor(target_clean), torch.tensor(output_poison), torch.tensor(target_poison)
        # sys.exit()

        loss_clean = criterion(output_clean, target_clean)
        loss_poison = criterion(output_poison, target_poison)
        

        if len(output_poison) > 0:
            loss = (1-lambda1) * loss_clean + lambda1 * loss_poison
        else:
            loss = loss_clean
        # loss = criterion(output, target)
        
        # print(loss)
        # sys.exit()
        losses.update(loss.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

        # for name, parms in model.named_parameters():
        #     print("name: ",name)
        #     print("param: ",parms.grad)

    print('epoch:', epoch, 'train loss:', losses.avg)
    p0 = float(sim0) / num_clean
    p1 = float(sim1) / num_poison
    return (p0-p1)

def validate(model, epoch, valloader, criterion, clean):
    losses = AverageMeter()
    model.eval()
    correct = 0
    _sum = 0

    for i, (input, target, poisoned_flags) in enumerate(valloader):
        input = torch.squeeze(input)
        target = torch.squeeze(target)
        input, target = input.to(device), target.to(device)
        output = model(input)
        output_np = output.cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()
        out_ys = np.argmax(output_np, axis = -1)

        # print('out_ys', out_ys)
        # print('target_np', target_np)
        # print('==', out_ys == target_np)
        # sys.exit()

        _ = out_ys == target_np
        correct += np.sum(_, axis = -1)
        _sum += _.shape[0]
        # loss = criterion(output, target)
        # losses.update(loss.item(), input.size(0))

    if clean:
        print('epoch:', epoch)
        print('clean accuracy: {:.4f}'.format(correct*1.0 / _sum))
        # print('loss:', losses.avg)
    else:
        print('epoch:', epoch)
        print('attack accuracy: {:.4f}'.format(correct*1.0 / _sum))
        # print('loss:', losses.avg)

    return correct*1.0 / _sum
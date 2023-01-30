from utils import *
import torch
import numpy as np
import random
import argparse
import sys
import torchvision.transforms as transforms
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--batch_size', default=200, type=float, help='Feature dim for latent vector')

# args parse
args = parser.parse_args()
batch_size = args.batch_size

params = read_config()

model_name = params['model']
distill_data_name = params['distill_data']
com_ratio = params['com_ratio']
if model_name == "gtsrb":
    dataset = torch.load('./dataset/distill_' + distill_data_name + "_gtsrb")
else:
    dataset = torch.load('./dataset/distill_' + distill_data_name)
# print(type(dataset))
random.shuffle(dataset)
data_num = len(dataset)
print("distill_data num:", data_num)

# max_num = int(com_ratio * data_num)
# print("max_num: ",max_num)

# min_max_scaler = preprocessing.MinMaxScaler()

images = []
outputs = [] 
for i in range(data_num):
    img = np.array(dataset[i][0]).flatten()
    # print(img.shape)
    # sys.exit()
    output = np.array(dataset[i][1].cpu())
    # print(output.shape,output)
    # sys.exit()
    img = img.reshape(1,-1)
    # print(img.shape)
    # print(preprocessing.normalize(img,norm='l2'))
    # sys.exit()
    images.append(preprocessing.normalize(img,norm='l2').squeeze())
    # sys.exit()
    output = output.reshape(1,-1)
    # print(preprocessing.normalize(output,norm='l2'))
    # sys.exit()
    outputs.append(preprocessing.normalize(output,norm='l2').squeeze())
images = np.array(images)
# print(images.shape)

outputs = np.array(outputs)
# print(outputs.shape)
# sys.exit()

batch_num = int(data_num / batch_size) + (data_num%batch_size != 0)
# print(batch_num)
# sys.exit()
data_compression = []


def select_img(images_batch, outputs_batch, batch_n):
    data_num = images_batch.shape[0]
    max_num = int(data_num * com_ratio)
    if max_num == 0:
        return
    n_selected = 0
    images_sim = np.dot(images_batch,images_batch.transpose())
    # print(images_sim)
    # sys.exit()
    outputs_sim = np.dot(outputs_batch,outputs_batch.transpose())
    co_sim = np.multiply(images_sim, outputs_sim)
    # print(co_sim)
    # sys.exit()

    index = random.randint(0,data_num-1)
    # print(index)

    while n_selected < max_num:
        index = np.argmin(co_sim[index])
        data_compression.append(dataset[batch_n*batch_size+index])
        n_selected += 1
        co_sim[:,index] = 1
        print(batch_n, index)

for i in range(batch_num):
    images_batch = images[i*batch_size:min((i+1)*batch_size,data_num)]
    outputs_batch = outputs[i*batch_size:min((i+1)*batch_size,data_num)]

    select_img(images_batch, outputs_batch, i)

print(len(data_compression))

if model_name == 'gtsrb':
    torch.save(data_compression, './dataset/compression_' + distill_data_name + '_' + str(com_ratio) + "_gtsrb")
else:
    torch.save(data_compression, './dataset/compression_' + distill_data_name + '_' + str(com_ratio))
from tensorflow.keras.datasets import mnist
#except:    from keras.datasets import mnist
import time
from sklearn.manifold import TSNE

# from tsnecuda import TSNE

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import itertools

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import copy
from PIL import Image
import math
from bisect import bisect_right
import pandas
# from TNN import Mining, Model,resnet_model
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm 
# from loss.SoftTriple import SoftTriple
from loss.softtriple import SoftTriple
import evaluation as eva
# import net
from net.bninception import bninception
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
# from TNN import Mining, Model,resnet_model
# from TNN.Plot import scatter
# from TNN.Loss_Fn import triplet_loss
import torch
from net.cnn_ensemble import cnn_ensemble
from dataset.dataset_kmu_cnn import Dataset

from net.bninception import bninception

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', default='/root/notebooks/nfs/work/jason.chen/project/triplet_loss/pytorch-triplet-loss/SoftTriple/', type=str, help='path to dataset')
parser.add_argument('--cub', default='CUB_10', type=str, help='path to dataset')
parser.add_argument('--own', default='v8', type=str, help='path to dataset')

parser.add_argument('--model_file', default='/root/notebooks/nfs/work/jason.chen/project/triplet_loss/pytorch-triplet-loss/SoftTriple/out_model/', type=str, help='MN')
parser.add_argument('--model_path', default='own_v8/own_v8.pth', type=str,help='Mp')
parser.add_argument('--dim', default=2 , type=int, help='dim')
def dataloder(traindir,testdir):

    ###


    train_dataset   = Dataset(traindir, shuffle_pairs=True, augment=True)
    val_dataset     = Dataset(testdir, shuffle_pairs=True, augment=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3,num_workers=1, drop_last=True)
    test_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=3,num_workers=1, drop_last=True)
    # load data
#     traindir = os.path.join(args.data, 'train')
#     testdir = os.path.join(args.data, 'test')
#     normalize = transforms.Normalize(mean=[104., 117., 128.],
#                                      std=[1., 1., 1.])

#     train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.Lambda(RGB2BGR),
#             transforms.Resize(256),
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x.mul(255)),
#             normalize,
#         ]))

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=1, shuffle=True,
#         num_workers=4, pin_memory=True)

#     test_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(testdir, transforms.Compose([
#             transforms.Lambda(RGB2BGR),
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x.mul(255)),
#             normalize,
#         ])),
#         batch_size=1, shuffle=False,
#         num_workers=4, pin_memory=True) 

 
    return train_loader, test_loader

def RGB2BGR(im):
    assert im.mode == 'RGB'
    r, g, b = im.split()
    return Image.merge('RGB', (b, g, r))
def create_model():
    model = cnn_ensemble()
#     model = bninception(args.dim)
#     model = Model.TNN_CIFAR10_Drop(input_shape=(3,256,256),output_size=2)
    
    return model

def get_features_trained_weight(model, train_dataloader,filepath):        # 透過訓練好的pth檔案進行特徵萃取
    

    if torch.cuda.is_available():
        device = 'cuda:3'
    else:
        device = 'cpu'
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    
    model.eval()
    model.to(device)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
#     train_dataloader, val_dataloader, test_loader = dataloder()
    

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None
    
    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    print("Start extracting Feature")
    for (img1, img2), y, (class1, class2) in tqdm(train_dataloader):
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
        yy = y[0]
        labels.append(yy)
#         labels = []
        
#         IDimg_num = 3
#         for i in range (int(len(y)/IDimg_num)):
#             yy = y[(IDimg_num*i)]
            
#             if i >0:
#                 labels = torch.cat((labels,yy),0)
#                 print(labels)
#             else:
#                 labels = yy
#         labels = labels.to(device)
#         labels = labels.to(torch.long)
        
        
        
        feat_list = []
        def hook(module, input, output): 
            feat_list.append(output.clone().detach())
        
#         images = img.to(device)
#         target = target.squeeze().tolist()
#         print(target)
#         for element in target:
#         labels.append(target)
        
        with torch.no_grad():
            
#             handle=model.cls_head.register_forward_hook(hook) #擷取avgpool的output 
            
#             handle=model.embedding.register_forward_hook(hook) #擷取avgpool的output
#             handle=model.avgpool.register_forward_hook(hook) #擷取avgpool的output
#             handle=model.pooling.register_forward_hook(hook) #擷取avgpool的output
#             output = model.forward(images)
#             output = model.forward(img1, img2, batch_size =3)
            output = model(img1, img2, batch_size =3)

            feat_list.append(output.clone().detach())
            feat = torch.flatten(feat_list[0], 1)            #將avgpool的output送入flatten layer

#             handle.remove()
        
        current_features = feat.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
#     print(labels)
    return features, labels

def collate_skip_empty(batch):
    batch = [sample for sample in batch if sample] # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)
# def get_feature(model,validation_generator):
    
#     layer_name = 'dense_2'
    
#     intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
#     intermediate_output = intermediate_layer_model.predict_generator(validation_generator, verbose=1,steps=len(validation_generator))

#     print('intermediate_output shape = ',intermediate_output.shape)
    
#     validation_generator.reset()
#     x_val, y_val = next(validation_generator)
#     for i in range(int(len(validation_generator)-1)): #1st batch is already fetched before the for loop.
#         img, label = next(validation_generator)
#         #     x_train = np.append(x_train, img, axis=0 )
#         y_val = np.append(y_val, label, axis=0)
#     print('y_val shape = ',y_val.shape)
    
#     return y_val , intermediate_output

if __name__ == '__main__':
    gpu = 3
    device = 'cuda:3'
    
    args = parser.parse_args()
    traindir = os.path.join(args.data +args.own , 'train')
    testdir = os.path.join(args.data +args.own , 'val')
    filepath = args.model_file + args.model_path
    
    torch.manual_seed(0)
    train_dataloader, val_dataloader = dataloder(traindir,testdir)
    model = create_model()
    
    features , labels = get_features_trained_weight(model,train_dataloader,filepath)
    print(features.shape)
    time_start = time.time()
    # tsne = TSNE(n_iter=700, verbose=1, num_neighbors=4)
#     print('step1:')
    tsne = TSNE(n_components=2,random_state=123)
#     print('step2:')
    tsne_results = tsne.fit_transform(features)
    
    print('tsne_results shape = ',tsne_results.shape)
    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(1, 1, 1, title='TSNE' )
    
    # Create the scatter
    ax.scatter(
        x=tsne_results[:,0],
        y=tsne_results[:,1],
        c = labels,
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.5,
        s=50)
    plt.savefig('train_tsne.png')
    plt.cla()
    features , labels = get_features_trained_weight(model,val_dataloader,filepath)
    
    tsne = TSNE(n_components=2,random_state=123)
    tsne_results = tsne.fit_transform(features)
    
    print('tsne_results shape = ',tsne_results.shape)
    
    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(1, 1, 1, title='TSNE' )
    
    # Create the scatter
    ax.scatter(
        x=tsne_results[:,0],
        y=tsne_results[:,1],
        c = labels,
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.5,
        s=50)
    plt.savefig('val_tsne.png')
    
#     df = pandas.DataFrame(dict(x=tsne_results[:,0], y=tsne_results[:,1], label=labels))#.cpu().numpy()

#     df.plot(x="x", y="y", kind='scatter', c='label', colormap='viridis')
#     plt.savefig('/root/notebooks/nfs/work/jason.chen/project/triplet_loss/pytorch-triplet-loss/plot/online TNN distribution (test set)')
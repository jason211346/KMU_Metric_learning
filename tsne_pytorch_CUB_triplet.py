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
import evaluation as eva
# import net

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
from dataset.dataset_triplet_cub import Dataset
from net.tripletnet import Tripletnet
from net.bninception import bninception


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', default='/root/notebooks/dataset/', type=str, help='path to dataset')
parser.add_argument('--cub', default='CUB_2', type=str, help='path to dataset')
parser.add_argument('--model_file', default='/root/notebooks/KMU_softtriple_loss-main/out_model/', type=str,
                    help='MN')
parser.add_argument('--model_path', default='CUB_2_triplet/CUB_2_triplet_best.pth', type=str,
                    help='Mp')
parser.add_argument('--dim', default=64 , type=int,
                    help='dim')
parser.add_argument('--C', default=2 , type=int,
                    help='Class')
parser.add_argument('--csv', default='CUB2', type=str,
                    help='MN')

def Transfrom(augment = False):
#     normalize = transforms.Normalize(mean=[104., 117., 128.],
#                                          std=[1., 1., 1.])    
    normalize = transforms.Normalize(mean=[128., 117., 104.],
                                         std=[1., 1., 1.])    
    if augment:
        # If images are to be augmented, add extra operations for it (first two).
                
        transform = transforms.Compose([
#             transforms.Lambda(RGB2BGR),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            normalize,
        ])
    else:
        # If no augmentation is needed then apply only the normalization and resizing operations.
        transform = transforms.Compose([
#             transforms.Lambda(RGB2BGR),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            normalize,
        ])
    return transform
def dataloder(traindir , testdir):

    # load data
#     train_dataframe =  pd.read_csv(train_csv)
#     test_dataframe = pd.read_csv(test_csv)
    
    
    train_transform = Transfrom(augment= True)
    test_transform = Transfrom(augment= False)
    
    train_dataset = datasets.ImageFolder(root=traindir,
                                           transform=train_transform)
    test_dataset = datasets.ImageFolder(root=testdir,
                                           transform=test_transform)

#     train_dataset   = Dataset(train_dataframe, train_transform)
#     test_dataset   = Dataset(test_dataframe, test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,num_workers=0,shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16,num_workers=0, shuffle=False,drop_last=True)
    return train_loader,test_loader

def RGB2BGR(im):
    assert im.mode == 'RGB'
    r, g, b = im.split()
    return Image.merge('RGB', (b, g, r))
def create_model():
    model = Tripletnet(dim = args.dim,num_class=args.C)
    
#     model = Model.TNN_CIFAR10_Drop(input_shape=(3,256,256),output_size=2)
    
    return model

def get_features_trained_weight(model, train_dataloader,filepath):        # 透過訓練好的pth檔案進行特徵萃取
    

    if torch.cuda.is_available():
        device = 'cuda:0'
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
    for i, (img, target) in enumerate(tqdm(train_dataloader)):
        
        feat_list = []
        def hook(module, input, output): 
            feat_list.append(output.clone().detach())
        
        images = img.to(device)
#         target = target.squeeze().tolist()
#         print(target)
#         for element in target:

        labels.append(target)
#         labels = labels.item()
        with torch.no_grad():            
#             handle=model.embeddingnet.register_forward_hook(hook) #擷取avgpool的output
#             handle=model.embedding.register_forward_hook(hook) #擷取avgpool的output
#             handle=model.inception_5b_relu_pool_proj.register_forward_hook(hook) #擷取avgpool的output   
#             handle=model.avgpool.register_forward_hook(hook) #擷取avgpool的output
#             handle=model.pooling.register_forward_hook(hook) #擷取avgpool的output
#             import pdb;pdb.set_trace()

            output = model.embeddingnet(images)
            feat_list.append(output.clone().detach())
            feat = torch.flatten(feat_list[0], 1)            #將avgpool的output送入flatten layer
#             handle.remove()
        
        current_features = feat.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

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
    gpu = 0
    device = 'cuda:0'
    
    args = parser.parse_args()
    #read csv
#     train_csv = '/root/notebooks/KMU_softtriple_loss-main/dataset_csv/'+args.csv+'_train_data.csv'
#     test_csv = '/root/notebooks/KMU_softtriple_loss-main/dataset_csv/'+args.csv+'_test_data.csv'
    
    traindir = os.path.join(args.data +args.cub , 'train')
    testdir = os.path.join(args.data +args.cub , 'test')

    filepath = args.model_file + args.model_path
    
    torch.manual_seed(0)
    
    train_dataloader, val_dataloader = dataloder(traindir , testdir)
    
    model = create_model()
    
    features , labels = get_features_trained_weight(model ,train_dataloader ,filepath)
    
    labels = [i.tolist() for i in labels]
    labels = sum(labels, [])
    
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

    labels = [i.tolist() for i in labels]
    labels = sum(labels, [])
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
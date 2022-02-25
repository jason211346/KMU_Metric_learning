import os
import glob
import time
from os import walk
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, shuffle_pairs=False, augment=False):
        '''
        Create an iterable dataset from a directory containing sub-directories of 
        entities with their images contained inside each sub-directory.

            Parameters:
                    path (str):                 Path to directory containing the dataset.
                    shuffle_pairs (boolean):    Pass True when training, False otherwise. When set to false, the image pair generation will be deterministic
                    augment (boolean):          When True, images will be augmented using a standard set of transformations.

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''
        self.path = path

        self.feed_shape = [3, 496, 496]
        self.shuffle_pairs = shuffle_pairs

        self.augment = augment
        def RGB2BGR(im):
            assert im.mode == 'RGB'
            r, g, b = im.split()
            return Image.merge('RGB', (b, g, r))
        normalize = transforms.Normalize(mean=[104., 117., 128.],
                                     std=[1., 1., 1.])
        
        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).
            self.transform = transforms.Compose([
                
                transforms.Lambda(RGB2BGR),
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255)),
                normalize,
                
#                 transforms.Resize(512),
#                 transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
#                 transforms.RandomHorizontalFlip(p=0.5),
# #                 transforms.RandomVerticalFlip(),
# #                 transforms.RandomRotation(30),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])          
# #                 transforms.Resize(self.feed_shape[1:])
            ])
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose([
                
                transforms.Lambda(RGB2BGR),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255)),
                normalize,
                
#                 transforms.Resize(512),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# #                 transforms.Resize(self.feed_shape[1:])
            ])

        self.create_pairs()
        
    

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''
        self.image_paths1 = [] # before
        self.image_paths2 = [] # after 
        for root, dirs, files in walk(self.path):
            for f in sorted(files):
                img_path = join(root, f)
                if 'after' in img_path : 
                    self.image_paths2.append(img_path)
                else :
                    self.image_paths1.append(img_path)
                    
        
        
#         self.image_paths = glob.glob(os.path.join(self.path, "*/*.png"))
        self.image_classes1 = []
        self.image_classes2 = []
        self.class_indices1 = {}
        self.class_indices2 = {}

        
        for image_path in self.image_paths1:
            image_class1 = image_path.split(os.path.sep)[-4]  # a_0 / a_1
            image_class2 = image_path.split(os.path.sep)[-3] # ID 
            image_class3 = image_path.split(os.path.sep)[-2] # after / before
            image_class = image_class1 + '/' + image_class2 +'/' +image_class3

            self.image_classes1.append(image_class2)

            if image_class2 not in self.class_indices1:
                self.class_indices1[image_class2] = []
            self.class_indices1[image_class2].append(self.image_paths1.index(image_path))
        
        for image_path in self.image_paths2:
            image_class1 = image_path.split(os.path.sep)[-4]  # a_0 / a_1
            image_class2 = image_path.split(os.path.sep)[-3] # ID 
            image_class3 = image_path.split(os.path.sep)[-2] # after / before
            image_class = image_class1 + '/' + image_class2 +'/' +image_class3

            self.image_classes2.append(image_class2)

            if image_class2 not in self.class_indices2:
                self.class_indices2[image_class2] = []
            self.class_indices2[image_class2].append(self.image_paths2.index(image_path))
            
 
        
    
        self.indices1 = np.arange(len(self.image_paths1)) # 幫每張圖編號
        self.indices2 = np.arange(len(self.image_paths2)) # 幫每張圖編號
        step = 3
        self.indices1_26 = [self.indices1[i:i+step] for i in range(0,len(self.indices1),step)]
        self.indices2_26 = [self.indices2[i:i+step] for i in range(0,len(self.indices2),step)]
        
        
        
        if self.shuffle_pairs:
            self.indices1_26, self.indices2_26 = shuffle(self.indices1_26, self.indices2_26, random_state=3)
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(1)
        
#         select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

#         self.indices2 = []
#         for i in self.indices1:
#             class1 = self.image_classes1[i]
#         for i in self.indices2:
#             class2 = self.image_classes2[i]            


#         for i, pos in zip(self.indices1, select_pos_pair):
#             class1 = self.image_classes[i]

#             if pos:
#                 class2 = class1
#             else:                
#                 class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
#             idx2 = np.random.choice(self.class_indices[class2])
            
#             self.indices2.append(idx2)
#         self.indices2 = np.array(self.indices2)


    def __iter__(self):
        self.create_pairs()

        for idc1,idc2 in zip(self.indices1_26, self.indices2_26):
            for idx in idc1 :
#                 print(idx)
                image_path1 = self.image_paths1[idx]
                image_path2 = self.image_paths2[idx]
#                 print(image_path1)
#                 print(image_path2)
                class1 = self.image_classes1[idx]
                class2 = self.image_classes2[idx]
                
                self.imagelabel = image_path1.split(os.path.sep)[-4]
                
                y1 = int(self.imagelabel[2])

                y = torch.FloatTensor([y1])
                
                
                
                image1 = Image.open(image_path1).convert("RGB")
                image2 = Image.open(image_path2).convert("RGB")

                if self.transform:
                    image1 = self.transform(image1).float()                
                    image2 = self.transform(image2).float()
#                     image = torch.cat((image1,image2),1)
                yield (image1,image2), y, (class1, class2)
        
    def __len__(self):
        return len(self.image_paths1)

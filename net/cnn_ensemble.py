import torch
import torch.nn as nn
import torch.nn.functional as F
from net.bninception_KMU import bninception
from torchvision import models

class cnn_ensemble(nn.Module):  #nn.Module
    def __init__(self, dim=512):#backbone="resnet50"
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

#         if backbone not in models.__dict__:
#             raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models 
#         self.backbone = models.__dict__[backbone](pretrained=True, progress=True)
        self.backbone = bninception(dim)
        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.backbone.modules())[-1].in_features
#         self.backbone.fc = nn.Sequential(
#             nn.Dropout(p=0.2),
#         )
#         print(self.backbone)
        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
#         self.cls_head = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(out_features*2, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),

#             nn.Dropout(p=0.5),
#             nn.Linear(512, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),

#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )
        self.cls_head = nn.Sequential(
            nn.Linear((dim*2)*3, 512),
            
        )
#         self.cls_head2 = nn.Sequential(
#             nn.Linear(3, 1),
#             nn.Sigmoid(),
#         )
    

    def forward(self, img1, img2,batch_size):
        '''
        Returns the similarity value between two images.

            Parameters:
                    img1 (torch.Tensor): shape=[b, 3, 224, 224]
                    img2 (torch.Tensor): shape=[b, 3, 224, 224]

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''

        # Pass the both images through the backbone network to get their seperate feature vectors
        feat1 = self.backbone(img1)  # feat.shape = (batch_size, 512)
        feat2 = self.backbone(img2)
        
        for i in range (len(feat1)):
            feat_1 = feat1[i].unsqueeze(0)
            feat_2 = feat2[i].unsqueeze(0)
            feat = torch.cat((feat_1,feat_2),1)   
#             output = self.cls_head(feat1[i])
#             output = self.cls_head2(output.transpose(0,1))
            if i>0:
                out = torch.cat((out,feat),0)  # (feat1, feat2) concat = feat,  feat.shape =  (batch_size, 1024)
            else:
                out = feat

        
                
#         for i in range (int(batch_size/26)):
#             classifier = self.cls_head(out[26*i:(26*i)+26])
#             classifier = self.cls_head2(classifier.transpose(0,1))
#             if i>0:
#                 output = torch.cat((output,classifier),0)
#             else:
#                 output = classifier
        for i in range (int(batch_size/3)):
            
#             classifier = self.cls_head(out[3*i:(3*i)+3])

            BA_feat = (out[3*i:(3*i)+3])

            BA_feat3to1 = torch.cat((BA_feat[0],BA_feat[1],BA_feat[2]),0)
            BA_feat3to1 = BA_feat3to1.unsqueeze(0)

            BA_feat3to1 = self.cls_head(BA_feat3to1)
#             classifier = self.cls_head2(classifier.transpose(0,1))
            BA_feat3to1 = F.normalize(BA_feat3to1, p=2, dim=1)
            if i>0:
                output = torch.cat((output,BA_feat3to1),0)
            else:
                output = BA_feat3to1
        # Multiply (element-wise) the feature vectors of the two images together, 
        # to generate a combined feature vector representing the similarity between the two.
#         concat_features = torch.cat((feat1, feat2), 1)
#         print(concat_features.shape)
#         combined_features = feat1 * feat2

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
#         output = self.cls_head(concat_features)
#         output = self.cls_head(combined_features)
        return output
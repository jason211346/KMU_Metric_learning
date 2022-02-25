import torch
import torch.nn as nn
import torch.nn.functional as F
from siamese.vit_custom import ViT
from siamese.vit_custom import Transformer
from torchvision import models
## V2 版本代表  圖片先經過feature extractor 再 concat (V1 為圖片先concat 再經過feature extractor )
class vitcnn(nn.Module):
    def __init__(self):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

#         if backbone not in models.__dict__:
#             raise Exception("No model named {} exists in torchvision.models.".format(backbone))
        
        # Create a backbone network from the pretrained models provided in torchvision.models 
        self.backbone = ViT(
                            image_size = 512,
                            patch_size = 32,
                            dim = 1024,
                            depth = 6,
                            heads = 16,
                            mlp_dim = 2048,
                            dropout = 0.1,
                            emb_dropout = 0.1
                        )
        
        # Get the number of features that are outputted by the last layer of backbone network.
#         out_features = list(self.backbone.modules())[-3].out_features

        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        
        # in_features = 1024 , if use concat: in_features = 1024*2 = 2048
        self.cls_head = nn.Sequential(
            
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
        )
        self.cls_head2 = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid(),
        )
    

    def forward(self, img1,img2,batch_size):
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
        # img.shape = (batch_size,3,256*2,256) 
        
        # input every 26 image ,output 1 feature
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        for i in range (len(feat1)):
            feat_1 = feat1[i].unsqueeze(0)
            feat_2 = feat2[i].unsqueeze(0)
            feat = torch.cat((feat_1,feat_2),1)
#             output = self.cls_head(feat1[i])
#             output = self.cls_head2(output.transpose(0,1))
            if i>0:
                out = torch.cat((out,feat),0)
            else:
                out = feat

        
#         feat2 = self.backbone(img2)
#         print(feat1.shape)
        # Multiply (element-wise) the feature vectors of the two images together, 
        # to generate a combined feature vector representing the similarity between the two.
#         concat_features = torch.cat((feat1, feat2), 1)
#         combined_features = feat1 * feat2

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        
        for i in range (int(batch_size/3)):
            classifier = self.cls_head(out[3*i:(3*i)+3])
            classifier = self.cls_head2(classifier.transpose(0,1))
            if i>0:
                output = torch.cat((output,classifier),0)
            else:
                output = classifier
        

        
        return output
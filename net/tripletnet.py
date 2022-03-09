import torch
import torch.nn as nn
import torch.nn.functional as F
from net.bninception import bninception

class Tripletnet(nn.Module):
    def __init__(self, dim=64,num_class=2):
        super(Tripletnet, self).__init__()
        self.embeddingnet = bninception(dim)
        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.embeddingnet.modules())[-1].in_features
        
#         self.embeddingnet = embeddingnet
        self.cls = nn.Sequential(
            nn.Linear( dim, num_class),           
        )
    
    def classifier(self, x):
        embedded_x = self.embeddingnet(x)
        cls_x = self.cls(embedded_x)
        cls_x = F.normalize(cls_x, p=2, dim=1) 
        return cls_x
    
    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
#         dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
#         dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return embedded_x, embedded_y, embedded_z #dist_a, dist_b,
    def get_embedding(self, x):
        return self.embedding_net(x)
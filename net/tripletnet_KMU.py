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
        self.cls_head = nn.Sequential(
            nn.Linear((dim*2)*3, 64),
        )
        self.LFR = nn.Sequential(
            nn.Linear(3, 1),
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
    
    def get_embedding(self, Anchor, Pos, Neg):  #,y
        
        # Anchor Before img
        B_emb_A = []
        B_emb_A_list = []
        for i in Anchor[0]:
            embedding_A = self.embeddingnet(i)
            B_emb_A_list.append(embedding_A)
            
        for i in B_emb_A_list:
            if len(B_emb_A)>0:
                B_emb_A = torch.cat((B_emb_A ,i.unsqueeze(0)),0)
            else :
                B_emb_A = i.unsqueeze(0)
                
        # Anchor After img
        A_emb_A = []
        A_emb_A_list = []
        for i in Anchor[1]:
            embedding_A = self.embeddingnet(i)
            A_emb_A_list.append(embedding_A)
            
        for i in A_emb_A_list:
            if len(A_emb_A)>0:
                A_emb_A = torch.cat((A_emb_A ,i.unsqueeze(0)),0)
            else :
                A_emb_A = i.unsqueeze(0)
        
        emb_3A = torch.cat((B_emb_A ,A_emb_A),2)
        
        
        for i in range(len(emb_3A)):
            classifier = self.LFR(emb_3A[i].transpose(0,1))

            if i > 0 :
                emb_A = torch.cat((emb_A ,classifier.transpose(0,1)),0)
            else :
                emb_A = classifier.transpose(0,1)
        #----------------------------------------------
        
        # Pos Before img
        B_emb_P = []
        B_emb_P_list = []
        for i in Pos[0]:
            embedding_P = self.embeddingnet(i)
            B_emb_P_list.append(embedding_P)
            
        for i in B_emb_P_list:
            if len(B_emb_P)>0:
                B_emb_P = torch.cat((B_emb_P ,i.unsqueeze(0)),0)
            else :
                B_emb_P = i.unsqueeze(0)
                
        # Pos After img
        A_emb_P = []
        A_emb_P_list = []
        for i in Pos[1]:
            embedding_P = self.embeddingnet(i)
            A_emb_P_list.append(embedding_P)
            
        for i in A_emb_P_list:
            if len(A_emb_P)>0:
                A_emb_P = torch.cat((A_emb_P ,i.unsqueeze(0)),0)
            else :
                A_emb_P = i.unsqueeze(0)
        
        emb_3P = torch.cat((B_emb_P ,A_emb_P),2)
        
        
        for i in range(len(emb_3P)):
            classifier = self.LFR(emb_3P[i].transpose(0,1))

            if i > 0 :
                emb_P = torch.cat((emb_P ,classifier.transpose(0,1)),0)
            else :
                emb_P = classifier.transpose(0,1)
        
        #--------------------------------------------------------
        
        # Neg Before img
        B_emb_N = []
        B_emb_N_list = []
        for i in Pos[0]:
            embedding_N = self.embeddingnet(i)
            B_emb_N_list.append(embedding_N)
            
        for i in B_emb_N_list:
            if len(B_emb_N)>0:
                B_emb_N = torch.cat((B_emb_N ,i.unsqueeze(0)),0)
            else :
                B_emb_N = i.unsqueeze(0)
                
        # Neg After img
        A_emb_N = []
        A_emb_N_list = []
        for i in Pos[1]:
            embedding_N = self.embeddingnet(i)
            A_emb_N_list.append(embedding_N)
            
        for i in A_emb_N_list:
            if len(A_emb_N)>0:
                A_emb_N = torch.cat((A_emb_N ,i.unsqueeze(0)),0)
            else :
                A_emb_N = i.unsqueeze(0)
        
        emb_3N = torch.cat((B_emb_N ,A_emb_N),2)
        
        
        for i in range(len(emb_3N)):
            classifier = self.LFR(emb_3N[i].transpose(0,1))

            if i > 0 :
                emb_N = torch.cat((emb_N ,classifier.transpose(0,1)),0)
            else :
                emb_N = classifier.transpose(0,1)
        
        
        
        return emb_A ,emb_P ,emb_N
    
    
#         Pos = self.embedding_net(Pos)
#         img1 = self.embedding_net(img1)
#         img2 = self.embedding_net(img2)
        
#         IDimg_num = 3
#         for i in range (int(len(y)/IDimg_num)):
#             yy = y[(IDimg_num*i)]

#             if i >0:
#                 target = torch.cat((target,yy),0)
#             else:
#                 target = yy
        
#         for i in range (len(img1)):
#             feat_1 = img1[i].unsqueeze(0)
#             feat_2 = img2[i].unsqueeze(0)
#             feat = torch.cat((feat_1,feat_2),1)   
# #             output = self.cls_head(feat1[i])
# #             output = self.cls_head2(output.transpose(0,1))
#             if i>0:
#                 out = torch.cat((out,feat),0)  # (feat1, feat2) concat = feat,  feat.shape =  (batch_size, 1024)
#             else:
#                 out = feat
                
#         for i in range (int(batch_size/3)):
            
#             BA_feat = (out[3*i:(3*i)+3])

#             BA_feat3to1 = torch.cat((BA_feat[0],BA_feat[1],BA_feat[2]),0)
#             BA_feat3to1 = BA_feat3to1.unsqueeze(0)

#             BA_feat3to1 = self.cls_head(BA_feat3to1)
# #             classifier = self.cls_head2(classifier.transpose(0,1))
#             BA_feat3to1 = F.normalize(BA_feat3to1, p=2, dim=1)
#             if i>0:
#                 output = torch.cat((output,BA_feat3to1),0)
#             else:
#                 output = BA_feat3to1
#         return output ,target
    
    
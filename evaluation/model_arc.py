import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # This is the "W" matrix (Class Centers)
        # Parameter shape: [Number_of_Classes, Embedding_Size]
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. Normalize Inputs (x) and Weights (W)
        # x shape: [batch, in_features]
        # W shape: [out_features, in_features]
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. Calculate Sine
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # 3. Calculate Cos(theta + m) using trig identity: cos(A+B) = cosAcosB - sinAsinB
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 4. Handle numerical stability (keep gradients stable)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 5. Convert labels to one-hot encoding to apply margin ONLY to ground truth
        # dense_one_hot allows us to modify only the correct class index
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 6. Apply margin: If correct class, use phi (margin added), else use standard cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        
        # 7. Scale by s
        output *= self.s
        
        return output

class SubCenterArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, k=3, s=64.0, m=0.50):
        """
        k: Number of sub-centers per class (e.g., 3)
        """
        super(SubCenterArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.s = s
        self.m = m
        
        # Shape is now [out_features * k, in_features]
        # We store k centers for every class
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # input: [batch, in_features]
        # weight: [out_features*k, in_features]
        
        # 1. Calculate cosine similarity with ALL sub-centers
        # cosine shape: [batch, out_features * k]
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. Reshape to [batch, out_features, k] to group sub-centers by class
        cosine = cosine.view(-1, self.out_features, self.k)
        
        # 3. MAX POOLING: For every class, pick the best matching sub-center
        # We only care about the sub-center that is closest to our input
        cosine_best, _ = torch.max(cosine, dim=2) 
        # cosine_best shape is now [batch, out_features] (standard logits)

        # --- The rest is identical to Standard ArcFace ---
        
        sine = torch.sqrt((1.0 - torch.pow(cosine_best, 2)).clamp(0, 1))
        phi = cosine_best * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine_best > self.th, phi, cosine_best - self.mm)
        
        one_hot = torch.zeros(cosine_best.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_best)
        output *= self.s
        
        return output

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super(EmbeddingModel, self).__init__()
        
        # Use ResNet18 or ResNet50 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final Classification layer of ResNet
        input_features_fc = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove old FC
        
        # Add new layers to generate the Embedding
        self.bn1 = nn.BatchNorm1d(input_features_fc)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(input_features_fc, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size) # Normalize embedding for stability
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn2(x)
        # Important: We usually return normalized features for inference
        return F.normalize(x, p=2, dim=1)
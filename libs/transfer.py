import torch

import torch.nn.functional as F

from torchvision import models

def preprocess_vgg(image):
    
    # image = image.unsqueeze(0)
    image = torch.clamp(image, -1, 1)
    image = 255*(image+1)/2
    image = interpolate(image)
    image = normalize_batch(image)
    
    return image

def interpolate(image, mode='bilinear', size=(224, 224)):
    
    image = F.interpolate(image, size)
    
    return image

def normalize_batch(image):
    
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    image = image.div_(255.0)
    
    return (image - mean) / std

def gram_matrix(x):
    
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    
    return gram

class Vgg16(torch.nn.Module):
    
    def __init__(self, requires_grad=False):
        
        super().__init__()
        
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        out = {}
        out['relu1_2'] = h_relu1_2
        out['relu2_2'] = h_relu2_2
        out['relu3_3'] = h_relu3_3
        out['relu4_3'] = h_relu4_3
        
        return out
    











    
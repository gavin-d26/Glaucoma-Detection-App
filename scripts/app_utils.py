import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.transforms as tr
from einops import rearrange
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def resnet_predict(model, x):
    B,C,H,W = x.size()
    x = model.backbone.conv1(x)
    x = model.backbone.bn1(x)
    x = model.backbone.relu(x)
    x = model.backbone.maxpool(x)
    x = model.backbone.layer1(x)
    x = model.backbone.layer2(x)
    x = model.backbone.layer3(x)
    x = model.backbone.layer4(x)
    b, c, h, w = x.shape
    cam = rearrange(x, "B C H W -> B (H W) C")
    cam = model.backbone.fc(cam)
    cam = F.upsample(rearrange(cam, "B (H W) C -> B C H W", C=1,H=h,W=w), size=(H,W), align_corners=True, mode='bilinear')
    x = model.backbone.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.backbone.fc(x)
    return cam, x


def effnet_predict(model, x):
    B,C,H,W = x.size()
    x = model.backbone.features(x)
    b, c, h, w = x.shape
    cam = rearrange(x, "B C H W -> B (H W) C")
    cam = model.backbone.classifier(cam)
    cam = F.upsample(rearrange(cam, "B (H W) C -> B C H W", C=1,H=h,W=w), size=(H,W), align_corners=True, mode='bilinear')
    x = model.backbone.avgpool(x)
    x = torch.flatten(x, 1)

    x = model.backbone.classifier(x)
    return cam, x

def mobilenet_predict(model, x):
    B,C,H,W = x.size()
    x = model.backbone.features(x)
    b, c, h, w = x.shape
    cam = rearrange(x, "B C H W -> B (H W) C")
    cam = model.backbone.classifier(cam)
    cam = F.upsample(rearrange(cam, "B (H W) C -> B C H W", C=1,H=h,W=w), size=(H,W), align_corners=True, mode='bilinear')
    x = model.backbone.avgpool(x)
    x = torch.flatten(x, 1)

    x = model.backbone.classifier(x)
    return cam, x


def alternet_predict(model, x):
    B,C,H,W = x.size()
    
    x = model.maxpool(model.initial(x))
    for layer in model.layers[:-1]:
        x = model.maxpool(layer(x))
    x = model.layers[-1](x)
    
    b, c, h, w = x.shape
    cam = rearrange(x, "B C H W -> B (H W) C")
    cam = model.classifier(cam)
    cam = F.upsample(rearrange(cam, "B (H W) C -> B C H W", C=1,H=h,W=w), size=(H,W), align_corners=True, mode='bilinear')
    
    
    x = model.avgpool(x).squeeze(-1).squeeze(-1)
    x = model.classifier(x)
    return cam, x
    
    
def image_format(img_path:str, ar=1.1):
    img = Image.open(img_path)
    img = img.convert('RGB')
    
    width, height = img.size
    
    left = (width-ar*height)//2#(crop_factor)*(width-height)//2    #smaller crop factor => smaller area cropped
    top = 0
    right = width - left
    bottom = height
    
    img = img.crop((left, top, right, bottom))
    
    img_224 = img.resize((224,224), resample= Image.BILINEAR)
 
    return img_224

def preprocess(img):
    img = tr.Compose([tr.ToTensor(),
                            tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ])(img)
    
    return img.unsqueeze(0)
    
    

def hmap_post_process(cam, ori_img:Image, figure_scale = 64, alpha=0.6, gamma=0):
    cam = cam.squeeze(0).expand(3,-1,-1).permute(1,2,0).detach().cpu().numpy()
    ori_img = ori_img.convert('RGB')
    img = np.uint8(np.array(ori_img))

    dx, dy = 0.05, 0.05
    x = np.arange(-3.0, 3.0, dx)
    y = np.arange(-3.0, 3.0, dy)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    rows = 1
    columns = 1
    
    # fig = plt.figure(figsize=(rows*figure_scale, columns*figure_scale))

    # fig.add_subplot(rows,columns,1)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)###
    # print(img.shape, heatmap.shape)
    overlayed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha,gamma)
    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
    
    overlayed_img = Image.fromarray(overlayed_img)
    # fig = plt.imshow(overlayed_img, extent=extent) 
    return overlayed_img
     
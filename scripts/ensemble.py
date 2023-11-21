import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models as models
import app_utils


@torch.no_grad()
def get_cam_max(cam):
    cam = F.relu(cam)
    B,C,H,W = cam.size()
    cam = cam/(torch.max(cam.view(B, C,-1), dim=-1, keepdim=True)[0].view(B,C,1,1) + 1e-9)
    return cam

class Ensemble(models.BaseModel):
    def __init__(self, filenames:list, df:pd.DataFrame,saved_models_path:str, reduction:str='max'):
        super().__init__()
        
        # with open(filenames) as f:
        #     lines = f.read().splitlines()
        self.filenames=filenames
        self.reduction = reduction
        models = []
        
        for f in filenames:
            path = df.loc[f, 'Path']
            models.append(torch.load(os.path.join(saved_models_path, path), map_location='cpu'))
        
        self.models = nn.ModuleList(models)
        
    @torch.no_grad()    
    def forward(self, x):
        outputs = []
        
        for model in self.models:
            outputs.append(model(x))
        
        outputs = torch.cat(tuple(outputs), dim=-1)
        
        if self.reduction=='mean':
            out = outputs.mean(dim=-1, keepdim= True)
            return out
        
        elif self.reduction=='max':
            out, indices = outputs.max(dim=-1, keepdim=True)
            return out
        
        return None
            
    @torch.no_grad()
    def predict(self, x):
        B,C,H,W = x.shape
        
        cam_list = []
        out_list = []
        
        for model in self.models: #self.models.keys():
            if isinstance(model,models.VGG):
                out = model(x)
                out_list.append(out)
            
            elif isinstance(model, models.EfficientNet):
                cam, out = app_utils.effnet_predict(model, x)
                cam_list.append(cam)
                out_list.append(out)
            
            elif isinstance(model, models.ResNet):
                cam, out = app_utils.resnet_predict(model, x)
                cam_list.append(cam)
                out_list.append(out)
                
            elif isinstance(model, models.MobileNet):
                cam, out = app_utils.mobilenet_predict(model, x)
                cam_list.append(cam)
                out_list.append(out)
            
            elif isinstance(model, models.AlterNetK):
                cam, out = app_utils.alternet_predict(model, x)
                cam_list.append(cam)
                out_list.append(out)

        if len(cam_list)>0:
            cam_list = list(map(get_cam_max, cam_list))   ####
            
        if len(cam_list)==0:
            cam_list.append(torch.zeros((B,1,H,W)))
            # cam_list.append(torch.zeros((B,1,H,W)))
            
        
        
        cam_list = torch.cat(tuple(cam_list), dim=1)
        
        out_list = torch.cat(tuple(out_list), dim=-1)
        
        out_list = out_list.sigmoid()
        
        if self.reduction == 'max':
            out_value, out_index = out_list.squeeze(0).max(dim=0)
        
        elif self.reduction == 'mean':
            out_value = out_list.squeeze(0).mean(dim=0)
        
        output = out_value.item()
        cam_list = cam_list.mean(dim=1, keepdim=True)
        
        if output<0.5:
            cam_list = torch.zeros_like(cam_list)    
        
        return output, cam_list #Tensor.Size(1,1,224,224)
        
    
    
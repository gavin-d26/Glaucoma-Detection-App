import torch
import torchvision.transforms as tr
import torchvision.io as io
import os
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image


def hf_collate(batch):
    image_list = [sample[0] for sample in batch]
    labels = torch.stack(tuple([sample[1] for sample in batch]), dim=0)
    return image_list, labels


class EyeDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir_path, validation = False, hf=False):
        super(EyeDataset, self).__init__()
        self.df = df
        self.img_dir_path = img_dir_path
        self.hf=hf
        
        self.length = len(df)
        if validation is False:
            self.transforms = tr.Compose([
                                        tr.ToTensor(),
                                        tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        tr.RandomHorizontalFlip(),
                                        tr.RandomVerticalFlip(),
                                        tr.RandomRotation(30),
                                        ])
        else:
            self.transforms = tr.Compose([tr.ToTensor(),
                                        tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])
                
        
    def __getitem__(self, index):
        label, img_path = self.df.loc[index, 'class'], self.df.loc[index, 'challenge_id']
        img_path = os.path.join(self.img_dir_path, img_path + '.jpg')
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(label).int()
        if self.hf is True:
            return image, label.reshape(1,)
        image = self.transforms(image)
        return image, label.reshape(1,)
        
        
    def __len__(self):
        return self.length        
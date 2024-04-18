import glob
import os
import cv2
import pdb
import torch
from utils import *
from torch.utils.data import Dataset

def get_jpg_paths(folder_path):
    jpg_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
    return jpg_paths


def NormalizingImg(tensor):
    tensor = tensor.contiguous()
    tensor = tensor.float()
    eps = 1e-12
    max = 255
    min = 0
    tensor = tensor / max + eps
    torch.clamp(tensor, min=min, max=max)
    return tensor

    
class imgDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = get_jpg_paths(data_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        H, W = image.shape[:2]
        
        if(H!=64 or W!=64):
            image = cv2.resize(image, (64, 64))
        
        image = torch.from_numpy(image)
        image = NormalizingImg(image)

        return image

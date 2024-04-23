import torch
from torch import nn
from utils import calculate_time
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import PIL.Image as Image
from torchvision import transforms
from model import AutoEncoderConv, AutoEncoderConv_Lite, AutoEncoderClassifier, AutoEncoderClassifier_Lite
import numpy as np
import pdb

def softmax(logits):
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    return probs

@calculate_time
def SingleImgTest(img_path, model):
    image = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),           
    ])
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    output = model(img_tensor)
    prob = softmax(output.detach().numpy())

    return np.argmax(prob)



if __name__ == '__main__':
    imgPath = '/home/franky/Data/Project/feature_optimal/Our_Dataset/空車格資料集/訓練集/1/日本空車_0.jpg'
    imgFlod = '/home/franky/Data/Project/feature_optimal/Our_Dataset/空車格資料集/訓練集'


    AutoEncoder_model_path = 'checkpoints/AutoEncoder/model_epoch_final.pt'
    Classifier_model_path = 'checkpoints/Classifier/model_epoch_final.pt'
    autoencoder = AutoEncoderConv()
    autoencoder.load_state_dict(torch.load(AutoEncoder_model_path))
    autoencoder.eval()
    model = AutoEncoderClassifier(autoencoder, num_classes=2)
    model.load_state_dict(torch.load(Classifier_model_path))
    model.eval()
    Result = SingleImgTest(imgPath, model)
    print("Prediction Class: {}".format(Result))
    



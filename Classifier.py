import pdb
import PIL.Image as Image
from dataset import *
from utils import save_csv
from torch.utils.data import DataLoader
from model import AutoEncoderConv, AutoEncoderClassifier
from torch import nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse


def train(train_dir='', save_model_path='checkpoints/Classifier/', log_path='log/'):
    train_loss_log = log_path + 'Classifier_train_loss.csv'
    epochs = 50
    batch_size = 128
    learning_rate = 0.001
    num_classes = 2 
    model = AutoEncoderClassifier(autoencoder, num_classes)
    model.train()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),           
    ])

    # Load dataset
    dataset = ImageFolder(root=train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Now use device: " + str(device))
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    min_loss = None

    # Train the autoencoder
    for epoch in range(epochs):
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            # ===================forward=====================
            output = model(data)
            loss = criterion(output, label)
            # ===================backward====================
            loss.backward()
            optimizer.step()
        scheduler.step()
        # ===================log========================
        loss_log = loss.cpu().detach().numpy()
        save_csv(loss_log, train_loss_log)
        if (epoch + 1) % 5 == 0:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss.item()))
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_model_path + 'model_epoch_{}.pt'.format(epoch + 1))

        if (min_loss!=None and min_loss > loss):
            torch.save(model.state_dict(), save_model_path + 'model_epoch_final.pt')
        else:
            min_loss = loss


def test(test_dir='', model_path='checkpoints/Classifier/model_epoch_final.pt'):
    test_loss_log = 'log/Classifier_test_loss.csv'
    batch_size = 128
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),           
    ])

    # Load the model
    model = AutoEncoderClassifier(autoencoder, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load dataset
    dataset = ImageFolder(root=test_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Now use device: " + str(device))
    model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    # Calculate average loss and accuracy
    avg_loss = test_loss / len(dataloader)
    accuracy = 100 * correct / total

    # Log the results
    save_csv(avg_loss, test_loss_log)
    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(avg_loss, accuracy))


if __name__ == '__main__':

    data_dir = 'your_train_dataset'
    test_dir = 'your_test_dataset'

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="choose a mode to run this Python file.")
    parser.add_argument("--model", default='checkpoints/AutoEncoder/model_epoch_final.pt', help="choose a pretrained AutoEncoder model to run this Python file.")
    args = parser.parse_args()

    # Load pretrained model
    autoencoder = AutoEncoderConv()
    autoencoder.load_state_dict(torch.load(args.model))
    autoencoder.eval()

    # Freeze parameters of AutoEncoder for training Classifier
    for param in autoencoder.parameters():
        param.requires_grad = False

    if(args.mode == 'train'):
        train(data_dir)
    elif(args.mode == 'test'):
        test(test_dir)
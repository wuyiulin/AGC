import pdb
from dataset import *
from utils import save_csv, clearDir, calculate_time
from torch.utils.data import DataLoader
from model import AutoEncoderConv, AutoEncoderConv_Lite
from torch import nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler
import argparse

@calculate_time
def train(model, data_dir, save_model_path='checkpoints/AutoEncoder/', log_path='log/AutoEncoder/'):
    clearDir(save_model_path)
    clearDir(log_path)
    train_loss_log = log_path + 'AutoEncoder_train_loss.csv'
    epochs = 50
    batch_size = 128
    learning_rate = 0.001
    num_workers = torch.cuda.device_count()*4

    # Initialize the autoencoder

    model.train()

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation([-30,30], interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Define the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True if device==torch.device('cuda') else False
    print("Now use device: " + str(device))
    model.to(device)
    
    # Define the loss function and optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    min_loss = None
    scaler = GradScaler()

    # Train the autoencoder
    for epoch in range(epochs):
        for batch_data in dataloader:
            batch_data, _ = batch_data 
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            # ===================forward=====================
            # Runs the forward pass under autocast.
            with autocast():
                output = model(batch_data)
                loss = loss_func(output, batch_data)
            # ===================backward====================
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        # ===================log========================
        loss_log = loss.cpu().detach().numpy()
        save_csv(loss_log, train_loss_log)
        if (epoch + 1) % 5== 0:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss.item()))
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_model_path + 'model_epoch_{}.pt'.format(epoch + 1))

        if (min_loss!=None and min_loss > loss):
            torch.save(model.state_dict(), save_model_path + 'model_epoch_final.pt')
            min_loss = loss
        elif(not min_loss):
            min_loss = loss

@calculate_time
def test(model, test_dir='', model_path='checkpoints/AutoEncoder/model_epoch_final.pt'):
    batch_size = 1
    num_workers = torch.cuda.device_count()*4

    # Move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Now use device: " + str(device))
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = ImageFolder(root=test_dir, transform=transform)

    # Define the dataloader
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loss_func = nn.MSELoss()

    with torch.no_grad():
        for batch_data in test_dataloader:
            batch_data, _ = batch_data 
            batch_data = batch_data.to(device)
            output = model(batch_data)
            loss = loss_func(output, batch_data)
            print('Output Score: {}'.format(round(float(loss), 3)))
            
@calculate_time
def vis(model, test_dir='', model_path='checkpoints/AutoEncoder/model_epoch_final.pt'):
    batch_size = 128
    num_workers = torch.cuda.device_count()*4
    loss_func = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=test_dir, transform=transform)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    with torch.no_grad():
        for data, _ in test_dataloader:
            data = data.to(device)
            recon = model(data)
            break

    _, ax = plt.subplots(2, 7, figsize=(15, 4))

    for i in range(7):
        loss = loss_func(data[i], recon[i])
        loss = loss.cpu().detach().numpy()
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
        ax[1, i].text(0.5, -0.1, 'loss:{:.3f}'.format(loss.item()), ha='center', va='center', transform=ax[1, i].transAxes, fontsize=12)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="choose a mode to run this Python file.")
    args = parser.parse_args()

    model = AutoEncoderConv()
    # model = AutoEncoderConv_Lite()

    data_dir = 'your_train_dataset'
    test_dir = 'your_test_dataset'

    if(args.mode == 'train'):
        train(model, data_dir)
    elif(args.mode == 'test'):
        test(model, test_dir)
    elif(args.mode == 'vis'):
        vis(model, test_dir)

            

        
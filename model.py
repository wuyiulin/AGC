import torch.nn as nn
import torch.nn.functional as F
import pdb


class AutoEncoderConv(nn.Module):
	def __init__(self):
		super(AutoEncoderConv, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(8),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(8, 16, 
							kernel_size=3, 
							stride=2, 
							padding=1, 
							output_padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(16, 64, 
							kernel_size=3, 
							stride=2, 
							padding=1, 
							output_padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(64, 16, 
							kernel_size=3, 
							stride=1, 
							padding=1, 
							output_padding=0),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(16, 3, 
							kernel_size=3, 
							stride=1, 
							padding=1, 
							output_padding=0),
			nn.Sigmoid()
		)
		
	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


class AutoEncoderConv_Lite(nn.Module):
	def __init__(self):
		super(AutoEncoderConv_Lite, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(8, 16, 
							kernel_size=3, 
							stride=2, 
							padding=1, 
							output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(16, 3, 
							kernel_size=3, 
							stride=2, 
							padding=1, 
							output_padding=1),
			nn.Sigmoid()
		)
		
	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


class AutoEncoderClassifier(nn.Module):
	def __init__(self, autoencoder, num_classes):
		super(AutoEncoderClassifier, self).__init__()
		self.autoencoder = autoencoder
		self.num_channels = self.autoencoder.encoder[-4].out_channels
		self.conv1 = nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.BN1 = nn.BatchNorm2d(32)
		self.BN2 = nn.BatchNorm2d(64)
		self.BN3 = nn.BatchNorm2d(128)
		self.BN4 = nn.BatchNorm2d(256)
		self.fc1 = nn.Linear(256 * 2 * 2, 512)
		self.fc2 = nn.Linear(512, num_classes)


	def forward(self, x):
		x = self.autoencoder.encoder(x)
		x = F.relu(self.conv1(x), inplace=True)
		x = self.BN1(x)
		x = self.pool(F.relu(self.conv2(x), inplace=True))
		x = self.BN2(x)
		x = self.pool(F.relu(self.conv3(x), inplace=True))
		x = self.BN3(x)
		x = self.pool(F.relu(self.conv4(x), inplace=True))
		x = self.BN4(x)
		
		x = x.view(-1, 256 * 2 * 2) # 因為 2D 圖片 原始輸入是 16*16，經過三層 MaxPool2d(2,2) 縮為 2*2
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
	

class AutoEncoderClassifier_Lite(nn.Module):
	def __init__(self, autoencoder, num_classes):
		super(AutoEncoderClassifier_Lite, self).__init__()
		self.autoencoder = autoencoder
		self.num_channels = self.autoencoder.encoder[-3].out_channels
		self.conv1 = nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 4 * 4, 512)
		self.fc2 = nn.Linear(512, num_classes)

	def forward(self, x):
		x = self.autoencoder.encoder(x)
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		# x = self.pool(F.relu(self.conv3(x)))
		x = x.view(-1, 64 * 4 * 4) # 因為 2D 圖片 原始輸入是 16*16，經過三層 MaxPool2d(2,2) 縮為 2*2
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
import torch
import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from ECAPA_TDNN import SEModule, Bottle2neck, PreEmphasis, FbankAug


class ResBlock(nn.Module):
	def __init__(self, inplane, planes, kernel_size = None):
		super(ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplane, planes, kernel_size = kernel_size, padding = 'same')
		self.relu = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(inplane,planes)
		self.bn2 = nn.BatchNorm2d(inplane,planes)
		self.bn3 = nn.BatchNorm2d(inplane,planes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.conv1(x)
		x = self.relu(x)
		x = self.bn2(x)
		x = self.conv1(x)
		x = self.relu(x)
		x = self.bn3(x)
		return x


class CNN(nn.Module):

	def __init__(self, C):
		super(CNN,self).__init__()
		self.conv1 = nn.Conv2d(1,C, stride=(2,1), kernel_size = 3, padding=(1,1))
		self.conv2 = nn.Conv2d(C,C, stride=(2,1), kernel_size = 3, padding=(1,1))
		self.relu = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(C)
		self.bn2 = nn.BatchNorm2d(C)
		self.resblock1 = ResBlock(C, C, kernel_size = 3)
		self.resblock2 = ResBlock(C, C, kernel_size = 3)
		self.flatten = nn.Flatten(start_dim=1, end_dim=2)

	def forward(self,x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.resblock1(x)
		x = self.resblock2(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.bn2(x)
		x = self.flatten(x)

		return x
		

class CNN_TDNN(nn.Module):

	def __init__(self, C):

		super(CNN_TDNN, self).__init__()
		
		self.torchfbank = torch.nn.Sequential(
			PreEmphasis(),            
			torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
												 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
			)
		self.cnn = CNN(128)
		self.specaug = FbankAug() # Spec augmentation

		self.conv1  = nn.Conv1d(128*20, C, kernel_size=5, stride=1, padding=2)
		self.relu   = nn.ReLU()
		self.bn1    = nn.BatchNorm1d(C)
		self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
		self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
		self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
		# I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
		self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
		self.attention = nn.Sequential(
			nn.Conv1d(4608, 256, kernel_size=1),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Tanh(), # I add this layer
			nn.Conv1d(256, 1536, kernel_size=1),
			nn.Softmax(dim=2),
			)
		self.bn5 = nn.BatchNorm1d(3072)
		self.fc6 = nn.Linear(3072, 192)
		self.bn6 = nn.BatchNorm1d(192)


	def forward(self, x, aug):
		with torch.no_grad():
			x = self.torchfbank(x)+1e-6
			# print('fbank: ', x.shape)
			x = x.log()   
			x = x - torch.mean(x, dim=-1, keepdim=True)
			if aug == True:
				x = self.specaug(x)
		x = torch.unsqueeze(x, 1)
		# print('after squeeze: ', x.shape)
		x = self.cnn(x)
		# print('cnn: ',x.shape)
		x = self.conv1(x)

		x = self.relu(x)
		x = self.bn1(x)
		# print('cnn: ',x.shape)
		x1 = self.layer1(x)
		x2 = self.layer2(x+x1)
		x3 = self.layer3(x+x1+x2)

		x = self.layer4(torch.cat((x1,x2,x3),dim=1))
		x = self.relu(x)

		t = x.size()[-1]

		global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
		
		w = self.attention(global_x)

		mu = torch.sum(x * w, dim=2)
		sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

		x = torch.cat((mu,sg),1)
		x = self.bn5(x)
		x = self.fc6(x)
		x = self.bn6(x)

		return x


def MainModel(**kwargs):
    model = CNN_TDNN(C=1024)
    return model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod

'''
# Define the original Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.mid_dim = math.ceil(math.sqrt(in_dim * out_dim))
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.mid_dim), # MLP
            nn.SELU(),
            nn.Linear(self.mid_dim, out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, self.mid_dim),
            nn.SELU(),
            nn.Linear(self.mid_dim, in_dim),
        )

    def forward(self, x):
        # Equation (15 & 16)
        encoded_x = self.encoder(x)
        # Equation (17 & 18)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x
'''

'''
# Define the  ThreeLayer Denoising Model
class denoising_model(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(denoising_model,self).__init__()
    self.mid_dim = math.ceil(math.sqrt(in_dim * out_dim))
    self.encoder=nn.Sequential(
                  nn.Linear(in_dim, self.mid_dim-5),
                  nn.SELU(True),
                  nn.Linear(self.mid_dim -5 , self.mid_dim-10),
                  nn.SELU(True),
                  nn.Linear(self.mid_dim-10, out_dim),
                  nn.SELU(True),
                  )
    
    self.decoder=nn.Sequential(
                  nn.Linear(out_dim, self.mid_dim-10),
                  nn.SELU(True),
                  nn.Linear(self.mid_dim-10, self.mid_dim-5),
                  nn.SELU(True),
                  nn.Linear(self.mid_dim-5, in_dim),
                  nn.SELU(True),
                  )
    
  def forward(self, x):
        # Equation (15 & 16)
        encoded_x = self.encoder(x)
        # Equation (17 & 18)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x
'''


# Define SparseAutoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SparseAutoencoder,self).__init__()
        
        # encoder
        self.enc1 = nn.Linear(in_features=in_dim, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=out_dim)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=out_dim, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=in_dim)
 
    def forward(self, x):
        # encoding
        encoded_x = F.relu(self.enc1(x))
        encoded_x = F.relu(self.enc2(encoded_x))
        encoded_x = F.relu(self.enc3(encoded_x))
        encoded_x = F.relu(self.enc4(encoded_x))
        encoded_x = F.relu(self.enc5(encoded_x))
 
        # decoding
        decoded_x = F.relu(self.dec1(encoded_x))
        decoded_x = F.relu(self.dec2(decoded_x))
        decoded_x = F.relu(self.dec3(decoded_x))
        decoded_x = F.relu(self.dec4(decoded_x))
        decoded_x = F.relu(self.dec5(decoded_x))
        return encoded_x, decoded_x


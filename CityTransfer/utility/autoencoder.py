import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()

    @abstractmethod
    def forward(self, x):
        # Equation (15 & 16)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x
        encoded_x = self.encoder(x)
        # Equation (17 & 18)


class ThreeLayersAutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.mid_dim = math.ceil(math.sqrt(in_dim*out_dim)) #   Math.ceil(x) Return value: Returns the integer greater than or equal to x and the nearest integer.
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, in_dim),
        )

    def forward(self, x):
        # Equation (15 & 16)
        encoded_x = self.encoder(x)
        # Equation (17 & 18)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x


class SparseL1AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.mid_dim = math.ceil(math.sqrt(in_dim*out_dim))
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, in_dim),
        )

    def forward(self, x):
        # Equation (15 & 16)
        encoded_x = self.encoder(x)
        # Equation (17 & 18)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x


class CNNAutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):   # def __init__(self):
        super(AutoEncoder, self).__init__() # super(Net, self).__init__()
        self.mid_dim = math.ceil(math.sqrt(in_dim*out_dim))
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3,stride= 1, padding= 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.ReLU(True)
            )

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            
            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.ReLU(True),
            
            nn.Conv2d(16, 1, 1,stride= 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        # Equation (15 & 16)
        encoded_x = self.encoder(x)
        # Equation (17 & 18)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x
        
        #x = self.encoder(x)
        #output = self.decoder(x)
        #return output
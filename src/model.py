import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(
            True), nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(True),nn.Conv2d(16, 32, kernel_size=5), nn.ReLU(True))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=5), nn.ReLU(
            True), nn.ConvTranspose2d(16, 6, kernel_size=5), nn.ReLU(
            True), nn.ConvTranspose2d(6, 3, kernel_size=5), nn.ReLU(True))
        self.getLatent = False

    def forward(self, x):
        x = self.encoder(x)
        if self.getLatent:
            return x
        x = self.decoder(x)
        return x
    
    def getLatentSpace(self, bool):
        self.getLatent = bool
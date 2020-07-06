import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),     # 28 * 28 * 6
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),    # 24 * 24 * 16
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),   # 20 * 20 * 32 
            nn.ReLU(True)
            # nn.Dropout2d(0.5),                  # 10 * 10 * 32
            )
        self.latent = nn.Linear(32 * 20 * 20, 10)
        self.latent2 = nn.Linear(10, 32 * 20 * 20)

        self.decoder = nn.Sequential(
            # nn.Upsample(2),
            nn.ConvTranspose2d(32, 16, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True),
            )

        self.getLatent = False

    def forward(self, x):
        batch = x.shape[0]
        x = self.encoder(x)
        x = x.view(batch, -1)
        x = self.latent(x)

        if self.getLatent:
            return x
        
        x = self.latent2(x)
        x = x.view(batch, 32 * 20 * 20)
        x = x.view(-1, 32, 20, 20)
        x = self.decoder(x)
        return x
    
    def getLatentSpace(self, bool):
        self.getLatent = bool

    def batchSize(self, batchS):
        self.batch = batchS
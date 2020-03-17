import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self,channels = 1, img_size = 32 ,latent_dim = 100):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.channels = channels
        self.latent_dim = latent_dim
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(self.channels, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = F.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = F.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
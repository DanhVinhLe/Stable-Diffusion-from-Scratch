import torch 
import torch.nn as nn
import torch.nn.functional as F

from attention import SelfAttention 

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(n_heads= 1, d_embed= channels)
    
    def forward(self, x):
        # x: (B, channels, H, W)
        B, C, H, W = x.shape
        residual = x
        x = self.groupnorm(x)
        x = x.view(B, C, H*W).transpose(-1, -2) # (B, H*W, C)
        x = self.attention(x)
        x = x.transpose(-1, -2).view(B, C, H, W) # (B, C, H*W) -> (B, C, H, W)
        x += residual
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_layer = nn.Identity()
    def forward(self, x):
        # x: (B, in_channels, H, W)
        residual = x 
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        x += self.residual_layer(residual)
        return x
    
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size= 1, padding = 0), # (B, 4, H / 8, W / 8) -> (B, 4, H / 8, W / 8)
            nn.Conv2d(4, 512, kernel_size= 3, padding = 1), # (B, 4, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512), # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_AttentionBlock(512), # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512), # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512), # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512), # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512), # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            nn.Upsample(scale_factor=2), # (B, 512, H / 8, W / 8) -> (B, 512, H / 4, W / 4)
            nn.Conv2d(512, 512, kernel_size= 3, padding = 1), # (B, 512, H / 4, W / 4) -> (B, 512, H / 4, W / 4)
            VAE_ResidualBlock(512, 512), # (B, 512, H / 4, W / 4) -> (B, 512, H / 4, W / 4)
            VAE_ResidualBlock(512, 512), # (B, 512, H / 4, W / 4) -> (B, 512, H / 4, W / 4)
            VAE_ResidualBlock(512, 512), # (B, 512, H / 4, W / 4) -> (B, 512, H / 4, W / 4)
            nn.Upsample(scale_factor=2), # (B, 512, H / 4, W / 4) -> (B, 512, H / 2, W / 2)
            nn.Conv2d(512, 512, kernel_size= 3, padding = 1), # (B, 512, H / 2, W / 2) -> (B, 512, H / 2, W / 2)
            VAE_ResidualBlock(512, 256), # (B, 512, H / 2, W / 2) -> (B, 256, H / 2, W / 2)
            VAE_ResidualBlock(256, 256), # (B, 256, H / 2, W / 2) -> (B, 256, H / 2, W / 2)
            VAE_ResidualBlock(256, 256), # (B, 512, H / 2, W / 2) -> (B, 512, H / 2, W / 2)
            nn.Upsample(scale_factor=2), # (B, 256, H / 2, W / 2) -> (B, 256, H, W)
            nn.Conv2d(256, 256, kernel_size= 3, padding = 1), # (B, 256, H, W) -> (B, 256, H, W)
            VAE_ResidualBlock(256, 128), # (B, 256, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128), # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128), # (B, 128, H, W) -> (B, 128, H, W)
            nn.GroupNorm(32, 128), # (B, 128, H, W) -> (B, 128, H, W)
            nn.SiLU(), # (B, 128, H, W) -> (B, 128, H, W)
            nn.Conv2d(128, 3, kernel_size= 3, padding = 1), # (B, 128, H, W) -> (B, 3, H, W)
        )
    def forward(self, x):
        # x: (B, 4, H / 8, W / 8)
        x /= 0.18215
        for module in self:
            x = module(x) 
        # x: (B, 3, H, W)
        return x
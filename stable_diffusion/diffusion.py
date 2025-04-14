import torch 
import torch.nn as nn 
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, n_embed * 4)
        self.linear_2 = nn.Linear(n_embed * 4, 4 * n_embed)
    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x

class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, feature, time):
        # featre: (B, in_channels, H, W)
        # time: (B, n_time)
        residual = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        merged += self.residual_layer(residual)
        return merged
    
class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context: int = 768):
        super().__init__()
        channels = n_embd * n_head
        
        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias= False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, bias = False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, channels * 4 * 2)
        self.linear_geglu_2 = nn.Linear(channels * 4, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, feature, context):
        # feature: (B, channels, H, W)
        # context: (B, d_context, H, W)
        B, C, H, W = feature.shape
        residual_long = feature
        feature = self.groupnorm(feature)
        feature = self.conv_input(feature)
        feature = feature.view(B, C, H * W).permute(0, 2, 1) # (B, H * W, C)
        residual_short = feature # (B, H * W, C)
        feature = self.layernorm_1(feature)
        feature = self.attention_1(feature) 
        feature += residual_short
        residual_short = feature
        feature = self.layernorm_2(feature) # (B, H * W, C)
        feature = self.attention_2(feature, context) # (B, H * W, C)
        feature += residual_short
        residual_short = feature
        feature = self.layernorm_3(feature) # (B, H * W, C)
        feature, gate = self.linear_geglu_1(feature).chunk(2, dim = -1) # (B, H * W, C * 4)
        feature = feature + F.gelu(gate)
        
        feature = self.linear_geglu_2(feature)
        feature += residual_short
        feature = feature.permute(0, 2, 1).view(B, C, H, W) # (B, C, H * W) -> (B, C, H, W)
        feature = self.conv_output(feature)
        feature += residual_long
        return feature 
    
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        # x: (B, channels, H, W)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self: 
            if isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            elif isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
    
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)), # (B, 4, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)), # (B, 320, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)), # (B, 320, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size= 3, stride = 2, padding= 1)), # (B, 320, H / 8, W / 8) -> (B, 320, H / 16, W / 16)
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)), # (B, 320, H / 16, W / 16) -> (B, 640, H / 16, W / 16)
            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)), # (B, 640, H / 16, W / 16) -> (B, 640, H / 16, W / 16)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size= 3, stride = 2, padding= 1)), # (B, 640, H / 16, W / 16) -> (B, 640, H / 32, W / 32)
            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)), # (B, 640, H / 32, W / 32) -> (B, 1280, H / 32, W / 32)
            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)), # (B, 1280, H / 32, W / 32) -> (B, 1280, H / 32, W / 32)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size= 3, stride = 2, padding= 1)), # (B, 1280, H / 32, W / 32) -> (B, 1280, H / 64, W / 64)
            SwitchSequential(UNet_ResidualBlock(1280, 1280)), # (B, 1280, H / 64, W / 64) -> (B, 1280, H / 64, W / 64)
            SwitchSequential(UNet_ResidualBlock(1280, 1280)), # (B, 1280, H / 64, W / 64) -> (B, 1280, H / 64, W / 64)
        ])
        
        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280), # (B, 1280, H / 64, W / 64) -> (B, 1280, H / 64, W / 64)
            UNet_AttentionBlock(8, 160), # (B, 1280, H / 64, W / 64) -> (B, 1280, H / 64, W / 64)
            UNet_ResidualBlock(1280, 1280), # (B, 1280, H / 64, W / 64) -> (B, 1280, H / 64, W / 64)
        )
        
        self.decoders = nn.ModuleList([
            SwitchSequential(UNet_ResidualBlock(2560, 1280)), # (B, 2560, H / 64, W / 64) -> (B, 1280, H / 64, W / 64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)), # (B, 2560, H / 64, W / 64) -> (B, 1280, H / 64, W / 64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280), Upsample(1280)), # (B, 2560, H / 64, W / 64) -> (B, 1280, H / 32, W / 32)
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)), # (B, 2560, H / 32, W / 32) -> (B, 1280, H / 32, W / 32)
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)), # (B, 2560, H / 32, W / 32) -> (B, 1280, H / 32, W / 32)
            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), Upsample(1280)), # (B, 1920, H / 32, W / 32) -> (B, 1280, H / 16, W / 16)
            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)), # (B, 1920, H / 16, W / 16) -> (B, 640, H / 16, W / 16)
            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)), # (B, 1280, H / 16, W / 16) -> (B, 640, H / 16, W / 16)
            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), Upsample(640)), # (B, 960, H / 16, W / 16) -> (B, 640, H / 8, W / 8)
            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)), # (B, 960, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)), # (B, 640, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)), # (B, 640, H / 8, W / 8) -> (B, 320, H / 8, W / 88)
        ])
        
    def forward(self, x, context, time):
        # x: (B, 4, H / 8, W / 8)
        # context: (B, seq_len, dim)
        # time(1, 1280)
        
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)
        x = self.bottleneck(x, context, time)
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim = 1)
            x = layers(x, context, time)
        return x
        
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1)
    
    def forward(self, x):
        # x: (B, 320, H/ 8, W/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x) # (B, 4, H / 8, W / 8)
        return x
    
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.output_layer = UNET_OutputLayer(320, 4)
    def forward(self, latent, context, time):
        # latent: (B, 4, H / 8, W / 8)
        # context: (B, seq_len, dim)
        # time : (B, 1280)
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.output_layer(output)
        
        return output
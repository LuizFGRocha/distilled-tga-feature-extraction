import torch
from torch import nn
from abc import ABC

from models.base import TGAFeatureExtractor


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up(x)


class AttentionUNet(TGAFeatureExtractor):
    def __init__(self, ch_in=2, ch_out=2, compressed_dim=64):
        super(AttentionUNet, self).__init__()
        self.config = {'ch_in': ch_in, 'ch_out': ch_out, 'compressed_dim': compressed_dim}
        
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Encoder
        self.Conv1 = ConvBlock(ch_in=ch_in, ch_out=64)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512)
        
        # Bottleneck
        self.Conv5 = ConvBlock(ch_in=512, ch_out=1024)
        
        # Compression and Decompression layers
        self.fc_compress = nn.Linear(1024 * 64, compressed_dim)
        self.fc_decompress = nn.Linear(compressed_dim, 1024 * 64)
        
        # Decoder with Attention
        self.Up5 = UpConv(ch_in=1024, ch_out=512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.UpConv5 = ConvBlock(ch_in=1024, ch_out=512)
        
        self.Up4 = UpConv(ch_in=512, ch_out=256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.UpConv4 = ConvBlock(ch_in=512, ch_out=256)
        
        self.Up3 = UpConv(ch_in=256, ch_out=128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.UpConv3 = ConvBlock(ch_in=256, ch_out=128)
        
        self.Up2 = UpConv(ch_in=128, ch_out=64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.UpConv2 = ConvBlock(ch_in=128, ch_out=64)
        
        # Output
        self.Conv_1x1 = nn.Conv1d(64, ch_out, kernel_size=1, stride=1, padding=0)
        
    def encode(self, x):
        # Encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # Flatten and compress
        x5_flat = x5.view(x5.size(0), -1)
        compressed = self.fc_compress(x5_flat)
        
        return compressed

    def forward(self, x):
        # Encoding path
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # Compress and decompress (bottleneck)
        x5_flat = x5.view(x5.size(0), -1)
        compressed = self.fc_compress(x5_flat)
        decompressed = self.fc_decompress(compressed)
        x5 = decompressed.view(x5.size(0), 1024, 64)
        
        # Decoding + Attention + Concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)
        
        d1 = self.Conv_1x1(d2)
        
        return d1

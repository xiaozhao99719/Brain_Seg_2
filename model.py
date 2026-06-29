"""
模型构建模块 (修复版 v2)
实现TransBTS和KiU-Net两种3D医学图像分割算法
修复了解码器跳转连接尺寸不匹配的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


###########################################
# TransBTS 模型 (修复版 v2)
###########################################

class ConvBlock(nn.Module):
    """基础卷积块：Conv3D + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """编码器块：两个卷积层 + 最大池化"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        pooled = self.pool(x)
        return x, pooled  # 返回跳转连接和池化后的特征


class DecoderBlock(nn.Module):
    """解码器块：上采样 + 拼接跳转连接 + 卷积块"""
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: 输入通道数（来自上一层的输出）
            skip_channels: 跳转连接的通道数
            out_channels: 输出通道数
        """
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip_connection):
        # 上采样
        x = self.up(x)
        
        # 确保尺寸匹配（处理奇数尺寸的情况）
        if x.shape[2:] != skip_connection.shape[2:]:
            # 调整 x 的尺寸以匹配 skip_connection
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=True)
        
        # 拼接跳转连接
        x = torch.cat([x, skip_connection], dim=1)
        
        # 卷积
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TransBTS(nn.Module):
    """
    TransBTS模型：结合CNN和Transformer的3D医学图像分割
    修复版 v2：正确处理解码器跳转连接尺寸
    
    输入: (batch, in_channels, D, H, W)
    输出: (batch, num_classes, D, H, W)
    """
    def __init__(self, in_channels=4, num_classes=2, base_filters=32, 
                 embed_dim=512, num_transformer_layers=4, img_size=128):
        super(TransBTS, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 计算经过编码器后的空间尺寸
        # 输入: (batch, in_channels, 128, 128, 128)
        # 经过4次池化: 128 -> 64 -> 32 -> 16 -> 8
        self.feat_size = img_size // 16  # 128/16 = 8
        self.seq_len = self.feat_size ** 3  # 8*8*8 = 512
        
        # 编码器路径
        self.enc1 = EncoderBlock(in_channels, base_filters)           # skip1: (batch, 32, 128, 128, 128)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)      # skip2: (batch, 64, 64, 64, 64)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)  # skip3: (batch, 128, 32, 32, 32)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)  # skip4: (batch, 256, 16, 16, 16)
        
        # 底部瓶颈 + 投影到Transformer维度
        self.bottleneck_conv = nn.Conv3d(base_filters * 8, embed_dim, kernel_size=1)
        
        # 位置编码 (可学习)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        nn.init.xavier_uniform_(self.pos_embedding)
        
        # Transformer编码器层
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True  # ✅ 启用优化，输入形状: (batch, seq_len, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_transformer_layers
        )
        
        # 将Transformer输出投影回CNN特征维度
        self.proj_back = nn.Conv3d(embed_dim, base_filters * 8, kernel_size=1)
        
        # 解码器路径 (正确设置输入输出通道数)
        # dec4: 输入是 bottleneck的输出 (256) + skip4 (256) = 512 -> 输出 256
        self.dec4 = DecoderBlock(base_filters * 8, base_filters * 8, base_filters * 8)
        
        # dec3: 输入是 dec4的输出 (256) + skip3 (128) = 384 -> 输出 128
        self.dec3 = DecoderBlock(base_filters * 8, base_filters * 4, base_filters * 4)
        
        # dec2: 输入是 dec3的输出 (128) + skip2 (64) = 192 -> 输出 64
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2, base_filters * 2)
        
        # dec1: 输入是 dec2的输出 (64) + skip1 (32) = 96 -> 输出 32
        self.dec1 = DecoderBlock(base_filters * 2, base_filters, base_filters)
        
        # 最终分类层
        self.final_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)
    
    def forward(self, x):
        # 编码器
        skip1, x = self.enc1(x)  # skip1: (batch, 32, 128, 128, 128), x: (batch, 32, 64, 64, 64)
        skip2, x = self.enc2(x)  # skip2: (batch, 64, 64, 64, 64), x: (batch, 64, 32, 32, 32)
        skip3, x = self.enc3(x)  # skip3: (batch, 128, 32, 32, 32), x: (batch, 128, 16, 16, 16)
        skip4, x = self.enc4(x)  # skip4: (batch, 256, 16, 16, 16), x: (batch, 256, 8, 8, 8)
        
        # 瓶颈 + Transformer
        x = self.bottleneck_conv(x)  # (batch, embed_dim, 8, 8, 8)
        
        # 展平用于Transformer (batch_first=True)
        batch_size, channels, d, h, w = x.shape
        x_flat = x.view(batch_size, channels, -1)  # (batch, embed_dim, seq_len)
        x_flat = x_flat.permute(0, 2, 1)  # (batch, seq_len, embed_dim) ✅ for batch_first=True
        
        # 添加位置编码 (自动广播: (1, seq_len, embed_dim) -> (batch, seq_len, embed_dim))
        x_flat = x_flat + self.pos_embedding  # ✅ no need to permute
        
        # Transformer (batch_first=True)
        x_flat = self.transformer(x_flat)  # (batch, seq_len, embed_dim)
        
        # 恢复3D形状
        x_flat = x_flat.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = x_flat.view(batch_size, channels, d, h, w)  # (batch, embed_dim, 8, 8, 8)
        
        # 投影回CNN特征维度
        x = self.proj_back(x)  # (batch, base_filters*8, 8, 8, 8)
        
        # 解码器 (DecoderBlock内部会自动上采样到skip_connection的尺寸)
        x = self.dec4(x, skip4)  # (batch, 256, 16, 16, 16)
        x = self.dec3(x, skip3)  # (batch, 128, 32, 32, 32)
        x = self.dec2(x, skip2)  # (batch, 64, 64, 64, 64)
        x = self.dec1(x, skip1)  # (batch, 32, 128, 128, 128)
        
        # 最终分类
        x = self.final_conv(x)  # (batch, num_classes, 128, 128, 128)
        
        return x


###########################################
# KiU-Net 模型
###########################################

class DilatedConvBlock(nn.Module):
    """空洞卷积块（多尺度特征提取）"""
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super(DilatedConvBlock, self).__init__()
        
        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels // len(dilation_rates), 
                              kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm3d(out_channels // len(dilation_rates)),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.final_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        
        x = torch.cat(outputs, dim=1)
        x = self.final_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class KiUNet(nn.Module):
    """
    KiU-Net模型：结合U-Net和空洞卷积的3D医学图像分割
    
    输入: (batch, in_channels, D, H, W)
    输出: (batch, num_classes, D, H, W)
    """
    def __init__(self, in_channels=4, num_classes=2, base_filters=32):
        super(KiUNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 编码器路径（带空洞卷积）
        self.enc1_conv = DilatedConvBlock(in_channels, base_filters)
        self.enc1_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2_conv = DilatedConvBlock(base_filters, base_filters * 2)
        self.enc2_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3_conv = DilatedConvBlock(base_filters * 2, base_filters * 4)
        self.enc3_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc4_conv = DilatedConvBlock(base_filters * 4, base_filters * 8)
        self.enc4_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # 瓶颈
        self.bottleneck = DilatedConvBlock(base_filters * 8, base_filters * 16)
        
        # 解码器路径
        self.dec4_up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec4_conv = DilatedConvBlock(base_filters * 16 + base_filters * 8, base_filters * 8)
        
        self.dec3_up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3_conv = DilatedConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        
        self.dec2_up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2_conv = DilatedConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        
        self.dec1_up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1_conv = DilatedConvBlock(base_filters * 2 + base_filters, base_filters)
        
        # 最终分类层
        self.final_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)
    
    def forward(self, x):
        # 编码器
        enc1 = self.enc1_conv(x)
        x = self.enc1_pool(enc1)
        
        enc2 = self.enc2_conv(x)
        x = self.enc2_pool(enc2)
        
        enc3 = self.enc3_conv(x)
        x = self.enc3_pool(enc3)
        
        enc4 = self.enc4_conv(x)
        x = self.enc4_pool(enc4)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 解码器
        x = self.dec4_up(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4_conv(x)
        
        x = self.dec3_up(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3_conv(x)
        
        x = self.dec2_up(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2_conv(x)
        
        x = self.dec1_up(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1_conv(x)
        
        # 最终分类
        x = self.final_conv(x)
        
        return x


###########################################
# 3D U-Net 模型
###########################################

class DoubleConv3D(nn.Module):
    """3D Double Convolution Block: Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """3D Down-sampling: MaxPool -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """3D Up-sampling: Upsample -> Concat -> DoubleConv

    注意：upsample后通道数不变（仍是in_channels），
    拼接skip_channels后为 in_channels + skip_channels，
    DoubleConv的输入需适配这个总通道数。
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up3D, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # upsample后仍是in_channels通道，拼接skip_channels后为 in_channels+skip_channels
        self.conv = DoubleConv3D(in_channels + skip_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        # 处理尺寸不匹配
        if x.shape[2:] != skip_connection.shape[2:]:
            x = F.interpolate(x, size=skip_connection.shape[2:],
                            mode='trilinear', align_corners=True)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv(x)
        return x


class OutConv3D(nn.Module):
    """输出卷积：将特征通道映射为类别数"""
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net 模型

    结构：
        编码器：4个下采样块（每次分辨率减半，通道数翻倍）
        瓶颈：DoubleConv
        解码器：4个上采样块（每次分辨率翻倍，通道数减半）

    输入: (batch, in_channels, D, H, W)
    输出: (batch, num_classes, D, H, W)
    """
    def __init__(self, in_channels=4, num_classes=2, base_filters=32):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # 编码器
        # enc1: (B, in_ch, D, H, W) -> (B, base, D, H, W)
        self.enc1 = DoubleConv3D(in_channels, base_filters)
        # enc2: -> (B, base*2, D/2, H/2, W/2)
        self.enc2 = Down3D(base_filters, base_filters * 2)
        # enc3: -> (B, base*4, D/4, H/4, W/4)
        self.enc3 = Down3D(base_filters * 2, base_filters * 4)
        # enc4: -> (B, base*8, D/8, H/8, W/8)
        self.enc4 = Down3D(base_filters * 4, base_filters * 8)

        # 瓶颈
        self.bottleneck = DoubleConv3D(base_filters * 8, base_filters * 16)

        # 解码器
        # dec4: (B, base*16, D/8, H/8, W/8) + skip4 -> (B, base*8, D/4, H/4, W/4)
        self.dec4 = Up3D(base_filters * 16, base_filters * 8, base_filters * 8)
        # dec3: -> (B, base*8, D/4, H/4, W/4) + skip3 -> (B, base*4, D/2, H/2, W/2)
        self.dec3 = Up3D(base_filters * 8, base_filters * 4, base_filters * 4)
        # dec2: -> (B, base*4, D/2, H/2, W/2) + skip2 -> (B, base*2, D, H, W)
        self.dec2 = Up3D(base_filters * 4, base_filters * 2, base_filters * 2)
        # dec1: -> (B, base*2, D, H, W) + skip1 -> (B, base, D, H, W)
        self.dec1 = Up3D(base_filters * 2, base_filters, base_filters)

        # 最终分类
        self.out = OutConv3D(base_filters, num_classes)

    def forward(self, x):
        # 编码器
        skip1 = self.enc1(x)              # (B, 32, D, H, W)
        skip2 = self.enc2(skip1)          # (B, 64, D/2, H/2, W/2)
        skip3 = self.enc3(skip2)          # (B, 128, D/4, H/4, W/4)
        skip4 = self.enc4(skip3)          # (B, 256, D/8, H/8, W/8)

        # 瓶颈
        x = self.bottleneck(skip4)       # (B, 512, D/8, H/8, W/8)

        # 解码器
        x = self.dec4(x, skip4)          # (B, 256, D/4, H/4, W/4)
        x = self.dec3(x, skip3)          # (B, 128, D/2, H/2, W/2)
        x = self.dec2(x, skip2)          # (B, 64, D, H, W)
        x = self.dec1(x, skip1)          # (B, 32, D, H, W)

        # 最终输出
        x = self.out(x)                   # (B, num_classes, D, H, W)
        return x


###########################################
# 模型工厂函数
###########################################

def get_model(model_name, in_channels=4, num_classes=2, **kwargs):
    """
    模型工厂函数

    Args:
        model_name: 'transbts'、'kiunet' 或 'unet3d'
        in_channels: 输入通道数（模态数）
        num_classes: 输出类别数
        **kwargs: 其他模型特定参数

    Returns:
        PyTorch模型
    """
    if model_name.lower() == 'transbts':
        return TransBTS(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=kwargs.get('base_filters', 32),
            embed_dim=kwargs.get('embed_dim', 512),
            num_transformer_layers=kwargs.get('num_transformer_layers', 4),
            img_size=kwargs.get('img_size', 128)
        )
    elif model_name.lower() == 'kiunet':
        return KiUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=kwargs.get('base_filters', 32)
        )
    elif model_name.lower() == 'unet3d':
        return UNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=kwargs.get('base_filters', 32)
        )
    else:
        raise ValueError(f"不支持的模型名称: {model_name}。请选择 'transbts'、'kiunet' 或 'unet3d'")


# 测试代码
if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(2, 4, 128, 128, 128)

    # 测试3D U-Net
    print("测试 3D U-Net...")
    model_unet3d = get_model('unet3d', in_channels=4, num_classes=2, base_filters=32)
    y = model_unet3d(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"3D U-Net 参数量: {count_parameters(model_unet3d):,}")

    # 测试TransBTS
    print("\n测试 TransBTS...")
    model_transbts = get_model('transbts', in_channels=4, num_classes=2, img_size=128)
    y = model_transbts(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"TransBTS 参数量: {count_parameters(model_transbts):,}")

    # 测试KiU-Net
    print("\n测试 KiU-Net...")
    model_kiunet = get_model('kiunet', in_channels=4, num_classes=2)
    y = model_kiunet(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"KiU-Net 参数量: {count_parameters(model_kiunet):,}")

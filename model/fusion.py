import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from model import ViTBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        x = self.bn(x)
        return x


class AttnCoefBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x

class AttnBiasBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x 

class ConvUpBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        # Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        x = self.bn(x)
        return x
    
    
class PosEmbedding(nn.Module):
    def __init__(self, inchannels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.proj = nn.Sequential(
            ConvBlock(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )
    
    def forward(self, x):
        x = self.proj(x)
        return x

class PosUpEmbedding(nn.Module):
    def __init__(self, inchannels=3, out_channels=64, kernel_size=3, stride=1, padding=1,output_padding=1):
        super().__init__()
        self.proj = nn.Sequential(
            ConvUpBlock(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
        )
    
    def forward(self, x):
        x = self.proj(x)
        return x


class ConvSmoothing(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ConvSmoothing, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                              stride=1, padding=self.padding, bias=False)
        # 初始化卷积核为均值滤波权重
        kernel_weights = torch.ones(in_channels, in_channels, kernel_size, kernel_size) / (kernel_size ** 2)
        self.conv.weight.data = kernel_weights

    def forward(self, x):
        return self.conv(x)



class PosConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.proj = nn.Sequential(ConvBlock(out_channels, out_channels, 3, 1, 1),
                                  ConvSmoothing(out_channels, 5))
        self.pos_embedding = PosEmbedding(inchannels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, pos_img=None):
        x = self.conv(x)
        pos_img = self.pos_embedding(pos_img)
        return self.proj(x + pos_img), pos_img


class PosUpConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()

        self.upconv = nn.Sequential(
            ConvUpBlock(in_channels, in_channels, (kernel_size, 1), (stride, 1), (padding, 0), output_padding=(output_padding, 0)),
            ConvUpBlock(in_channels, in_channels, (1, kernel_size), (1, stride), (0, padding), output_padding=(0, output_padding)),
            ConvBlock(in_channels, out_channels, 1, 1, 0)
            )
        self.attn_coef = AttnCoefBlock(out_channels, out_channels, 1, 1, 0)
        self.attn_bias = AttnCoefBlock(out_channels, out_channels, 1, 1, 0)
        self.proj = nn.Sequential(
            ChannelAttention(out_channels),
            ConvBlock(out_channels, out_channels, 3, 1, 1))

    def forward(self, x, pos_img=None):
        x = self.upconv(x)
        x = self.attn_coef(x) * x + self.attn_bias(pos_img)
        return self.proj(x)


class CsseAttnBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        self.upconv = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1, 1, 0)
            )
        self.attn_coef = ConvBlock(out_channels, out_channels, 1, 1, 0)
        self.attn_bias = ConvBlock(out_channels, out_channels, 1, 1, 0)
        self.proj = nn.Sequential(
            ConvBlock(out_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        x = self.upconv(x)
        x = self.attn_coef(x) * x + self.attn_bias(x)
        return self.proj(x)


class DownSample(nn.Module):
    def __init__(self, insize=(512,512), in_channels=[3, 64, 128, 256], out_channels=[64, 128, 256, 512]):
        super().__init__()

        self.convs = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.proj = nn.ModuleList()

        assert len(in_channels) == len(out_channels), 'Error: in_channels and out_channels must have the same length'
        sizes = [(insize[0] // 2**i, insize[1] // 2**i)for i in range(1, len(in_channels)+1)]

        for i in range(len(in_channels)):
            _conv = PosConvBlock(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=3,
                stride=2,
                padding=1)
            
            self.convs.append(_conv)
        
        for i in range(len(in_channels)):
            pool = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                ConvBlock(in_channels[i], out_channels[i], kernel_size=1, stride=1, padding=0),
                )
            self.downsample.append(pool)
            
        for i in range(len(in_channels)):
            proj = nn.Sequential(
                ConvBlock(out_channels[i], out_channels[i], 3, 1, 1)
                )
            self.proj.append(proj)

    def forward(self, x, pos_img=None):
        outputs = []
        pos_d = []
        for blk, down, proj in zip(self.convs, self.downsample, self.proj):
            _x, pos_img = blk(x, pos_img)
            # x = proj(torch.cat([_x, down(x)], dim=1))
            x = proj(_x + down(x))
            outputs.append(x)
            pos_d.append(pos_img)
        return outputs, pos_d


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.smooth =  ConvSmoothing(in_channels, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return self.smooth(x * y)


class UpSample(nn.Module):
    def __init__(self, in_channels=[512, 512 , 256, 128], out_channels=[512, 256, 128, 64]):
        super().__init__()

        self.convs = nn.ModuleList()
        self.upsample = nn.ModuleList()
        assert len(in_channels) == len(out_channels), 'Error: in_channels and out_channels must have the same length'

        self.upsample.append(ChannelAttention(out_channels[0]))
        for i in range(1, len(in_channels)):
            self.upsample.append(PosUpConvBlock(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ))

    def forward(self, x, pos_img=None):
        x_l = x[::-1]
        p_l = pos_img[::-1]

        for i, (xi, pi, up) in enumerate(zip(x_l, p_l, self.upsample)):
            if i == 0:
                x = up(xi)
            else:
                x = up(x, pi) + xi
            
        return x


class FeatureAggregator(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.convs = nn.ModuleList()
        self.pos_conv_proj = nn.ModuleList()
        self.neg_conv_proj = nn.ModuleList()
        self.aggreator_proj = nn.ModuleList()
        for i in range(len(in_channels)):
            self.pos_conv_proj.append(ConvBlock(in_channels[i], in_channels[i], kernel_size=1, stride=1, padding=0))
            self.neg_conv_proj.append(ConvBlock(in_channels[i], in_channels[i], kernel_size=1, stride=1, padding=0))
            self.aggreator_proj.append(nn.Sequential(
                ConvBlock(in_channels[i]*2, in_channels[i]*2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
                ConvBlock(in_channels[i]*2, in_channels[i]*2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                ConvBlock(in_channels[i]*2, in_channels[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                ChannelAttention(in_channels[i]),
                ConvUpBlock(in_channels[i], in_channels[i], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0)),
                ConvUpBlock(in_channels[i], in_channels[i], kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
                ConvUpBlock(in_channels[i], in_channels[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), output_padding=(0, 0)),
                )
                                       )
    
    def forward(self, pos_tensor, neg_tensor):
        x = []
        for i, (pos_proj, neg_proj, agg_proj) in enumerate(zip(self.pos_conv_proj, self.neg_conv_proj, self.aggreator_proj)):
            x.append(agg_proj(torch.cat(
                [pos_proj(pos_tensor[i]), neg_proj(neg_tensor[i])],
                dim=1)
            ))
        return x


class DefectFusion(nn.Module):
    def __init__(self,
                 in_channels=[3, 64, 128, 256],
                 out_channels=[64, 128, 256, 512],
                 num_classes = 3
                 ):
        super(DefectFusion, self).__init__()
        self.num_classes = num_classes
        self.down = DownSample(in_channels=in_channels, out_channels=out_channels)
        self.upsample = UpSample(in_channels=out_channels[-1:] + out_channels[::-1][:-1], out_channels=out_channels[-1:] + in_channels[::-1][:-1])
        self.feature_aggregator = FeatureAggregator(in_channels=out_channels)

        self.final_up_agg = nn.Sequential(            
            ConvUpBlock(in_channels[1]+3*2, in_channels[1]+3*2, (3, 1), (2, 1), (1, 0), output_padding=(1, 0)),
            ConvUpBlock(in_channels[1]+3*2, in_channels[1]+3*2, (1, 3), (1, 2), (0, 1), output_padding=(0, 1)),
            # CsseAttnBlock(in_channels[1]+3*2, in_channels[1]+3*2),
            ConvBlock(in_channels[1]+3*2, 32, 1, 1, 0),
            ConvBlock(32, 32, 3, 1, 1),
            ChannelAttention(32),
            ConvBlock(32, self.num_classes, 1, 1, 0),
            nn.Sigmoid()
        )
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, x, pos_img, neg_img):
        x_d, pos_d = self.down(x, pos_img)
        neg_d, npos_d = self.down(neg_img, pos_img)
        x_deep = self.feature_aggregator(x_d, neg_d)

        x_r = self.upsample(x_deep, pos_d)
        x_r = self.final_up_agg(torch.cat([x_r,
                                       torch.nn.functional.interpolate(x, size=x_r.shape[-2:], mode='bilinear', align_corners=False),
                                       torch.nn.functional.interpolate(pos_img, size=x_r.shape[-2:], mode='bilinear', align_corners=False)], dim=1))
        return x_r


if __name__ == '__main__':
    import cv2
    import numpy as np
    import time
    from torchvision.utils import save_image
    device = 'cuda:0'
    size = (384, 768)
    print('model init')
    model = DefectFusion(in_channels=[3, 64, 128, 256], out_channels=[64, 128, 256, 512])
    model.to(torch.device(device))
    print('model init')
    
    img = cv2.imread("images/1.png")
    img = cv2.resize(img, size[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32).to(device) / 255
    mask = torch.zeros(1, 3, *img.shape[-2:]).to(device)
    mask[:, :, 0:img.shape[-2]//2, 0:img.shape[-1]//2] = 1
    print('start', img.device)
    t =time.time()
    o = model(img, mask, img)
    print(time.time()-t)
    print(o.shape)
    save_image(o[0:1, :3, ...], "images/1_o.png")









import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
            
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViTBlock(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        """
        该代码段定义了一个用于图像处理的Transformer模型的基本结构, 包括图像的尺寸、补丁的尺寸、分类的数目、模型的维度、深度、注意力头数、多层感知器的维度等参数。
        参数: 
            image_size: 图像的尺寸。
            patch_size: 图像被分割成的补丁的尺寸。
            dim: 模型的维度。
            depth: Transformer的深度。
            heads: 注意力头的数量。
            mlp_dim: 多层感知器的维度。
            channels: 图像的通道数, 默认为3。
            dim_head: 每个注意力头的维度, 默认为64。
        返回值: 无。这是一个构造函数, 用于初始化Transformer模型的对象。
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.channels = channels
        #  确保图像尺寸可以被补丁尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, f'Image size ({image_height}, {image_width}) must be divisible by the patch size ({patch_height}, {patch_width}).'
        n_h = (image_height // patch_height)
        n_w = (image_width // patch_width)
        patch_dim = channels * n_h * n_w

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=patch_dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width)),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
        )
        
        self.pos_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, patch_dim),
            nn.LayerNorm(patch_dim),
        )
        
        self.transhape = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                                   p1=patch_height,
                                   p2=patch_width,
                                   c=channels,
                                   h=image_height//patch_height,
                                   w=image_width//patch_width)
        
        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

    def forward(self, img, pos_img):
        x = self.to_patch_embedding(img)
        x += self.pos_embedding(pos_img)
        # x = self.transformer(x)
        x = self.transhape(x)
        return x 


if __name__ == '__main__':
    import cv2
    import numpy as np
    import time
    from torchvision.utils import save_image
    device = 'cpu'
    size = (448, 720)
    model = ViTBlock(
        image_size = size,
        patch_size = 16,
        dim = 768,
        depth = 4,
        heads = 4,
        channels=3,
        mlp_dim = 1024,
        dim_head = 8,
    ).to(device)
    
    img = cv2.imread("images/1.png")
    img = cv2.resize(img, size[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32).to(device) / 255
    mask = torch.zeros(1, 3, *img.shape[-2:]).to(device)
    mask[:, :, 0:img.shape[-2]//2, 0:img.shape[-1]//2] = 1
    t =time.time()
    o = model(img, mask)
    print(time.time()-t)
    print(o.shape)
    save_image(o, "images/1_o.png")
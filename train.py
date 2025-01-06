import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataset import DefusionDataset, get_defect_transform, get_ok_transform
from model.fusion import DefectFusion
from losses import MseLoss, SSIMLoss
from torchvision.utils import save_image
import os
from tqdm import tqdm
import numpy as np


src_path = 'Data/good'
target_path = 'Data/defect-big'

log_dir = 'logs/pibu-big-02'
mdoel_dir = 'weights/pibu-big-02'

resume = 'weights/pibu-big-01/model_1.pth'
# resume = None


os.makedirs(log_dir, exist_ok=True)
os.makedirs(mdoel_dir, exist_ok=True)

device = torch.device('cuda:0')

size = (736, 448)
dataset = DefusionDataset(src_path,
                          target_path,
                          size=size,
                          ok_transform=get_ok_transform(size),
                          defect_transform=get_defect_transform(size))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = DefectFusion()
if resume:
    model.load_state_dict(torch.load(resume, map_location='cuda:0', weights_only=False).state_dict(), strict=False)
loss_mse = MseLoss(smooth=False, kernel_size=3, in_channels=3)
loss_mse.eval()
loss_mse.to(device)
loss_ssim = SSIMLoss()
loss_ssim.eval()

optimizer = optim.AdamW(model.parameters(), lr=0.001)

model.to(device)

num_epochs = 1000

best_loss = np.inf

for epoch in range(num_epochs):
    bar = tqdm(dataloader, desc=f'Epoch [{epoch}/{num_epochs}]')
    a_loss = .0
    for i, sample in enumerate(bar):
        img, pos_dst, img_target, pos_mask, fused, normal_mask = sample
        img, pos_dst, img_target, pos_mask, fused, normal_mask = img.to(device), pos_dst.to(device), img_target.to(device), pos_mask.to(device), fused.to(device), normal_mask.to(device)
        fuse_img = model(img, pos_dst, img)
        mse_loss = loss_mse(img, fuse_img, normal_mask)
        ssim_loss_raw_fuse = loss_ssim(img*normal_mask, fuse_img*normal_mask)
        ssim_loss_tar_defcet = loss_ssim(fuse_img*pos_mask, img_target*pos_mask)
        loss = mse_loss + ssim_loss_raw_fuse + ssim_loss_tar_defcet * 0.3
        
        
        d_fuse_img = model(img_target, pos_dst, img_target)
        d_mse_loss = loss_mse(img_target, fuse_img, 1)
        d_ssim_loss = loss_ssim(img_target, fuse_img)
        d_loss = d_mse_loss + d_ssim_loss*1
        loss = loss + d_loss*0.5
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            model.eval()
            with torch.no_grad():
                fuse_img = model(img, pos_dst, img)
                save_image(img, f'{log_dir}/img_{epoch}_{i}.png')
                save_image(fuse_img, f'{log_dir}/img_{epoch}_{i}_fuse.png')
                save_image(img_target, f'{log_dir}/img_{epoch}_{i}_target.png')
                save_image(pos_mask, f'{log_dir}/img_{epoch}_{i}_mask.png')
                
                fuse_img = model(img_target, pos_dst, img_target)
                save_image(fuse_img, f'{log_dir}/img_{epoch}_{i}_target_fuse.png')

                
        bar.set_postfix(Loss=f'{loss.item():.4f}',
                        mse_loss=f'{mse_loss.item():.4f}',
                        ssim_loss_raw_fuse=f"{ssim_loss_raw_fuse.item():.4f}",
                        ssim_loss_tar_defcet=f"{ssim_loss_tar_defcet.item():.4f}")
        a_loss += loss.item()
    if a_loss / len(bar) < best_loss:
        best_loss = loss.item()
        torch.save(model, f'{mdoel_dir}/model_{epoch}.pth')

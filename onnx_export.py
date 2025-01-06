import numpy as np
import torch
import os
import cv2
from torchvision.utils import save_image
from model.fusion import DefectFusion

model_path = 'weights/pibu-big-01/model_1.pth'
sta = torch.load(model_path, map_location='cpu')
model = DefectFusion()
model.load_state_dict(sta.state_dict())
model.eval()
size = (448, 736)
input_image, input_defect_pos, input_defect = torch.randn(1, 3, *size), torch.randn(1, 3, *size), torch.randn(1, 3, *size)
torch.onnx.export(model, (input_image, input_defect_pos, input_defect), "model.onnx")
exit(0)

device = torch.device('cuda:0')
model.to(device)
# Load the image
image_path = 'images/zx.png'
defect_path = 'images/zx_defect.png'
defect_mask_path = 'images/zx_defect_mask.png'


image = cv2.imread(image_path)
defect = cv2.imread(defect_path)
defect_mask = cv2.imread(defect_mask_path)


defect = cv2.resize(defect, (image.shape[1], image.shape[0]))
defect_mask = cv2.resize(defect_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


defect_mask = np.where(defect_mask > 0, 1.0, 0.)
# Preprocess the image
input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
input_defect_pos = cv2.cvtColor(defect, cv2.COLOR_BGR2RGB) / 255 * defect_mask
input_defect = cv2.cvtColor(defect, cv2.COLOR_BGR2RGB) / 255


def totensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

input_image = totensor(input_image)
input_defect_pos = totensor(input_defect_pos)
input_defect = totensor(input_defect)

# Run the model
with torch.no_grad():
    input_image = input_image.to(device)
    input_defect_pos = input_defect_pos.to(device)
    input_defect = input_defect.to(device)
    output = model(input_image, input_defect_pos, input_defect)
    save_image(output, './images/output.png')
from PIL import Image
import numpy as np
import math
import torch.nn.functional as F
import torch
from typing import Union
import cv2

def mask_rgb(mask, image_rgb, mask_color = (255,255,255)):
    return Image.fromarray(np.where(mask[0].unsqueeze(-1), np.array(image_rgb), np.array(mask_color)[None,None,:]).astype('uint8'))

# downsample mask to fit num tokens
def downsample_mask(mask, num_tokens):
    o_h = mask.shape[1]
    o_w = mask.shape[2]
    ratio = o_w / o_h
    mask_h = int(math.sqrt(num_tokens / ratio))
    mask_h = int(mask_h) + int((num_tokens % int(mask_h)) != 0)
    mask_w = num_tokens // mask_h

    mask_downsample = F.interpolate(mask.unsqueeze(0), size=(mask_h, mask_w), mode="bicubic").squeeze(0)
    return mask_downsample

def mask_sym_attn(attention_mask:torch.Tensor, r1:slice, r2:slice, mask:Union[torch.Tensor, bool]):
    attention_mask[:,r1,r2] = mask
    if isinstance(mask, torch.Tensor):
        mask = mask.view(1, len(mask), 1)
    attention_mask[:,r2,r1] = mask

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def down_size(image: Image.Image, down_scale = 2):
    return image.resize((int(image.size[0]//down_scale), int(image.size[1]//down_scale)))


def polygon_to_mask(polygon_points, height, width):
    # 创建空白的 0/1 mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 将坐标转换为整数格式并调整为适合 OpenCV 的格式
    polygon = np.array(polygon_points, dtype=np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    
    # 填充多边形，生成 0/1 mask
    cv2.fillPoly(mask, [polygon], color=1)
    
    # 将 mask 转换成 torch tensor
    mask_tensor = torch.from_numpy(mask).float()
    
    return mask_tensor

def resize_img_tensor(tensor, size):
    assert tensor.dim() == 2 # h,w
    return F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size).squeeze()

def get_mask_ori(polygon_points, height, width, height_lt, width_lt):
    mask_ori = polygon_to_mask(polygon_points, height, width)
    mask_ori = resize_img_tensor(mask_ori, (height_lt, width_lt))
    return mask_ori

def get_mask_ref(mask_ori):
    return resize_img_tensor(mask_ori, (27, 27))

def get_min_bounding_box(tensor):
    # 找到所有值为1的点的坐标
    rows, cols = torch.where(tensor == 1)
    
    if len(rows) == 0:  # 如果没有1，直接返回原数组
        return tensor
    
    # 计算最小包围框的边界
    min_row, max_row = torch.min(rows), torch.max(rows)
    min_col, max_col = torch.min(cols), torch.max(cols)
    
    # 创建新的数组，填充最小包围框为1
    result = torch.zeros_like(tensor)
    result[min_row:max_row+1, min_col:max_col+1] = 1
    
    return result
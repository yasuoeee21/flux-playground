from PIL import Image
import numpy as np
import math
import torch.nn.functional as F
import torch
from typing import Union

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
    return image.resize((image.size[0]//down_scale, image.size[1]//down_scale))
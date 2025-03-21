import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxPriorReduxPipeline, FluxFillPipeline
from diffusers.models import FluxMultiControlNetModel
from controlnet_aux import OpenposeDetector
from detector.groundingdino_sam import GroundingdinoSam
from utils import image_grid, down_size
from controlnet_aux import CannyDetector
import json
from utils import get_mask_ori
from matplotlib import pyplot as plt
import re

offload = True
dino_checkpoint = '/amax/hchuz/OMG-master/checkpoint/GroundingDINO'
sam_checkpoint = '/amax/hchuz/OMG-master/checkpoint/sam/sam_vit_h_4b8939.pth'
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16).to('cuda')
pipe = FluxFillPipeline.from_pretrained("/amax/hchuz/hfd_models/FLUX.1-Fill-dev", 
                                            torch_dtype=torch.bfloat16)
if offload:
    pipe.enable_model_cpu_offload()
else:
    _ = pipe.to('cuda')
groundingdino_sam = GroundingdinoSam(dino_checkpoint, sam_checkpoint)

root_dir = '/amax/hchuz/architectural_heritage/data/swap_window'
tar_dir = '/amax/hchuz/architectural_heritage/results/swap_window'
os.makedirs(tar_dir, exist_ok=True)
seeds = [111,222,333,444]

for path in [os.path.join(root_dir, it) for it in os.listdir(root_dir) if 'ori' in it]:
    img2 = Image.open(path).convert('RGB')
    boxes = groundingdino_sam.pred_box_with_prompt(img2, 'window')
    mask = torch.zeros((img2.height, img2.width), dtype=torch.float)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        mask[y1:y2, x1:x2] = True
    Image.fromarray((mask.numpy()*255).astype('uint8')).save(os.path.join(tar_dir, f'ori_{path[-5]}_mask.png'))
    for path_ref in [os.path.join(root_dir, it) for it in os.listdir(root_dir) if 'ref' in it]:
        ref_img = Image.open(path_ref).convert('RGB')
        region_style_input = pipe_prior_redux(ref_img)
        for seed in seeds:
            image2 = pipe(
                **region_style_input,
                image=img2,
                mask_image=mask.float(),
                height=img2.height,
                width=img2.width,
                guidance_scale=30,
                num_inference_steps=20,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            image2.save(os.path.join(tar_dir, f'ori_{path[-5]}-ref_{path_ref[-5]}-seed_{seed}.png'))
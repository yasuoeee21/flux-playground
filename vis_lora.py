import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from diffusers import FluxPipeline
import torch
from PIL import Image
from tqdm import tqdm

idx = '02'
lora_path = f'/amax/hchuz/ai-toolkit/output/texture_caption_{idx}/texture_caption_{idx}.safetensors'
root_dir = f'/amax/hchuz/architectural_heritage/data/分类的texture/texture_{idx}'
tar_dir = f'/amax/hchuz/architectural_heritage/results/texture/texture_{idx}'
os.makedirs(tar_dir, exist_ok=True)

base_model = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
_ = pipe.to('cuda')

pipe.load_lora_weights(lora_path)

kwargs = {
    'num_inference_steps':20, 
    'guidance_scale':3.5
    }

names = [it for it in os.listdir(root_dir) if it.endswith('.png')]

for name in tqdm(names):
    with open(os.path.join(root_dir, name.replace('.png', '.txt')), 'r') as f:
        prompt = f.read()
    images = pipe(prompt, num_images_per_prompt=4, **kwargs).images
    for i, img in enumerate(images):
        img.save(os.path.join(tar_dir, name[:-len('.png')]+f'_{i}'+'.png'))
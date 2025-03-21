import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from detector.canny_detect import CannyDetect
from PIL import Image
import json

import torch
from diffusers import FluxControlNetModel, FluxPriorReduxPipeline
from pipelines.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models import FluxMultiControlNetModel
from tqdm import tqdm

# set paras
base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = '/amax/hchuz/hfd_models/FLUX.1-dev-ControlNet-Union-Pro'
flux_redux = 'black-forest-labs/FLUX.1-Redux-dev'
offload = True
seeds = [111, 222, 333, 444]
batch_size = len(seeds)
output_dir = 'results/imgs_eval_temp'
os.makedirs(output_dir, exist_ok=True)

# load models
controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(flux_redux, torch_dtype=torch.bfloat16).to('cuda')
if offload:
    pipe.enable_model_cpu_offload()
else:
    pipe.to('cuda')
cannydetect = CannyDetect()

# set kwargs
kwargs = {
    'controlnet_conditioning_scale':[0.4],
    'num_inference_steps':20, 
    'guidance_scale':3.5,
    'num_images_per_prompt': 1
    }

data_dir = '/amax/hchuz/Image-to-Graph/dataset/original'
names = [name[:-len('.png')] for name in os.listdir(data_dir) if name.endswith('.png')]

for name in tqdm(names):
    with open(os.path.join(data_dir, name+'.json'), 'r') as f:
        data = json.load(f)
    global_style_image = Image.open(os.path.join(data_dir, name+'.png')).convert('RGB')
    control_image = cannydetect.polygon2canny_2([sample['points'] for sample in data['shapes']], global_style_image.height, global_style_image.width)
    control_image.save(os.path.join(output_dir, f'{name}_control.png'))
    global_style_input = pipe_prior_redux(global_style_image)
    global_style_input['prompt_embeds'] = global_style_input['prompt_embeds'].repeat(batch_size, 1, 1)
    global_style_input['pooled_prompt_embeds'] = global_style_input['pooled_prompt_embeds'].repeat(batch_size, 1)
    print(global_style_input['prompt_embeds'].shape, global_style_input['pooled_prompt_embeds'].shape)

    images = pipe(
        **global_style_input,
        control_image=[control_image],#*num_images_per_prompt,
        control_mode=[0],#*num_images_per_prompt, # 0~canny https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union
        width=global_style_image.width//16*16,
        height=global_style_image.height//16*16,
        generator=[torch.manual_seed(seed) for seed in seeds],
        **kwargs
    ).images
    
    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f'{name}_{seeds[i]}.png'))
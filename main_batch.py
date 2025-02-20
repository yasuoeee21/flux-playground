import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from generator import Generator
from detector.canny_detect import CannyDetect
from PIL import Image
import json

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = '/amax/hchuz/hfd_models/FLUX.1-dev-ControlNet-Union-Pro'
flux_redux = 'black-forest-labs/FLUX.1-Redux-dev'
flux_fill = '/amax/hchuz/hfd_models/FLUX.1-Fill-dev'
dino_checkpoint = '/amax/hchuz/OMG-master/checkpoint/GroundingDINO'
sam_checkpoint = '/amax/hchuz/OMG-master/checkpoint/sam/sam_vit_h_4b8939.pth'
lora_weights = '/amax/hchuz/ai-toolkit/output/my_first_flux_lora_v1/my_first_flux_lora_v1.safetensors'
offload = True
seeds = [111, 222, 333]
output_dir = 'results/polygon2canny+redux+fill'
os.makedirs(output_dir, exist_ok=True)

generator = Generator(base_model, controlnet_model_union, flux_redux, flux_fill, dino_checkpoint, sam_checkpoint, offload, lora_weights)
cannydetect = CannyDetect()

region_style_image = Image.open('temp/Snipaste_2023-02-21_13-33-51_window.png')

data_dir = '相似的几个2'
names = [name[:-len('.png')] for name in os.listdir(data_dir) if name.endswith('.png')]

for name in names:
    with open(os.path.join(data_dir, name+'.json'), 'r') as f:
        data = json.load(f)
    control_image = cannydetect.polygon2canny([sample['points'] for sample in data['shapes']])
    control_image.save(os.path.join(output_dir, f'{name}_control.png'))
    global_style_image = Image.open(os.path.join(data_dir, name+'.png')).convert('RGB')
    for seed in seeds:
        image_stage1 = generator.stage1(control_image, seed=seed, global_style_image=global_style_image, controlnet_conditioning_scale=[0.4])
        image_stage1.save(os.path.join(output_dir, f'{name}_stage1_{seed}.png'))
        image_stage2 = generator.stage2(image_stage1, 'window', region_style_image, seed=seed)
        image_stage2.save(os.path.join(output_dir, f'{name}_window_{seed}.png'))
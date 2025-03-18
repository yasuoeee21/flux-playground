import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from generator import Generator
from detector.canny_detect import CannyDetect
from PIL import Image
import json

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = '/amax/hchuz/hfd_models/FLUX.1-dev-ControlNet-Union-Pro'
flux_redux = 'black-forest-labs/FLUX.1-Redux-dev'
#flux_fill = '/amax/hchuz/hfd_models/FLUX.1-Fill-dev'
flux_fill = None
dino_checkpoint = '/amax/hchuz/OMG-master/checkpoint/GroundingDINO'
sam_checkpoint = '/amax/hchuz/OMG-master/checkpoint/sam/sam_vit_h_4b8939.pth'
#lora_weights = '/amax/hchuz/ai-toolkit/output/my_first_flux_lora_v1/my_first_flux_lora_v1.safetensors'
lora_weights = None
offload = False
seeds = [111, 222, 333, 444]
output_dir = 'results/imgs_eval_no_control'
os.makedirs(output_dir, exist_ok=True)

generator = Generator(base_model, controlnet_model_union, flux_redux, dino_checkpoint, sam_checkpoint, offload, flux_fill=flux_fill, lora_weights=lora_weights, no_control=False)
cannydetect = CannyDetect()

#region_style_image = Image.open('temp/Snipaste_2023-02-21_13-33-51_window.png')

data_dir = '相似的几个2'
names = [name[:-len('.png')] for name in os.listdir(data_dir) if name.endswith('.png')]

for name in names:
    with open(os.path.join(data_dir, name+'.json'), 'r') as f:
        data = json.load(f)
    with open(os.path.join(data_dir, name+'.txt'), 'r') as f:
        description = f.read()
    global_style_image = Image.open(os.path.join(data_dir, name+'.png')).convert('RGB')
    #control_image = cannydetect.polygon2canny_2([sample['points'] for sample in data['shapes']], global_style_image.height, global_style_image.width)
    control_image = cannydetect.polygon2canny([sample['points'] for sample in data['shapes']])
    #control_image = cannydetect.processor(global_style_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)
    #control_image.save(os.path.join(output_dir, f'{name}_control.png'))
    for seed in seeds:
        # style & control
        image_stage1 = generator.stage1(control_image, seed=seed, global_style_image=global_style_image, controlnet_conditioning_scale=[0.4])
        # no control
        #image_stage1 = generator.stage1(control_image, seed=seed, global_style_image=global_style_image)
        # no style
        #image_stage1 = generator.stage1(control_image, seed=seed, prompt=' ', global_style_image=None, controlnet_conditioning_scale=[0.4])
        image_stage1.save(os.path.join(output_dir, f'{name}_stage1_{seed}.png'))
        # image_stage2 = generator.stage2(image_stage1, 'window', region_style_image, seed=seed)
        # image_stage2.save(os.path.join(output_dir, f'{name}_window_{seed}.png'))
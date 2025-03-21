import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from generator import Generator
from detector.canny_detect import CannyDetect
from PIL import Image
import json
from tqdm import tqdm
import torch
from utils import get_mask_ori, get_mask_ref

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = '/amax/hchuz/hfd_models/FLUX.1-dev-ControlNet-Union-Pro'
flux_redux = 'black-forest-labs/FLUX.1-Redux-dev'
#flux_fill = '/amax/hchuz/hfd_models/FLUX.1-Fill-dev'
flux_fill = None
# dino_checkpoint = '/amax/hchuz/OMG-master/checkpoint/GroundingDINO'
# sam_checkpoint = '/amax/hchuz/OMG-master/checkpoint/sam/sam_vit_h_4b8939.pth'
dino_checkpoint = None
sam_checkpoint = None
#lora_weights = '/amax/hchuz/ai-toolkit/output/my_first_flux_lora_v1/my_first_flux_lora_v1.safetensors'
lora_weights = None
offload = True
seeds = [111, 222, 333, 444]
output_dir = 'results/imgs_eval_200_size'
os.makedirs(output_dir, exist_ok=True)

generator = Generator(base_model, controlnet_model_union, flux_redux, offload, dino_checkpoint=dino_checkpoint, sam_checkpoint=sam_checkpoint, flux_fill=flux_fill, lora_weights=lora_weights, no_control=False)
cannydetect = CannyDetect()

#region_style_image = Image.open('temp/Snipaste_2023-02-21_13-33-51_window.png')

data_dir = '/amax/hchuz/Image-to-Graph/dataset/original'
names = [name[:-len('.png')] for name in os.listdir(data_dir) if name.endswith('.png')]

for name in tqdm(names):
    with open(os.path.join(data_dir, name+'.json'), 'r') as f:
        data = json.load(f)
    # with open(os.path.join(data_dir, name+'.txt'), 'r') as f:
    #     description = f.read()
    global_style_image = Image.open(os.path.join(data_dir, name+'.png')).convert('RGB')
    control_image = cannydetect.polygon2canny_2([sample['points'] for sample in data['shapes']], global_style_image.height, global_style_image.width)
    #control_image = cannydetect.polygon2canny([sample['points'] for sample in data['shapes']])
    #control_image = cannydetect.processor(global_style_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)
    save_path = os.path.join(output_dir, f'{name}_control.png')
    if not os.path.exists(save_path):
        control_image.save(save_path)
    for seed in seeds:
        save_path = os.path.join(output_dir, f'{name}_{seed}.png')
        if os.path.exists(save_path):
            continue
        # style & control
        
        # # focus
        # height = global_style_image.height
        # width = global_style_image.width
        # height_lt = height//16
        # width_lt = width//16
        # ori_tokens = height_lt * width_lt
        # num_ref_imgs = 1
        # total_tokens = 1241 * num_ref_imgs + ori_tokens
        # attention_mask = torch.ones(1, total_tokens, total_tokens, dtype=torch.bool, device='cuda')
        # itr = attention_mask[:, 1241*num_ref_imgs:, 512:1241]
        # def mask_focus(it, mask_ori, mask_ref):
        #     it[:, mask_ori.flatten().bool()] = False
        #     it[:,(torch.outer(mask_ori.flatten(), mask_ref.flatten())).bool()] = True
        # # focus on origin
        # for shape in data['shapes']:
        #     mask_ori = get_mask_ori(shape['points'], height, width, height_lt, width_lt)
        #     mask_ref = get_mask_ref(mask_ori)
        #     mask_focus(itr, mask_ori, mask_ref)

        image_stage1 = generator.stage1(control_image, seed=seed, global_style_image=global_style_image, controlnet_conditioning_scale=[0.4])
        # no control
        #image_stage1 = generator.stage1(control_image, seed=seed, global_style_image=global_style_image)
        # no style
        #image_stage1 = generator.stage1(control_image, seed=seed, prompt=' ', global_style_image=None, controlnet_conditioning_scale=[0.4])
        image_stage1.save(save_path)
        # image_stage2 = generator.stage2(image_stage1, 'window', region_style_image, seed=seed)
        # image_stage2.save(os.path.join(output_dir, f'{name}_window_{seed}.png'))
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from generator import Generator
from detector.canny_detect import CannyDetect
from PIL import Image
import json
from tqdm import tqdm
import torch
from utils import get_mask_ori, get_mask_ref, down_size
from image_gen_aux import DepthPreprocessor


base_model = 'black-forest-labs/FLUX.1-dev'
#base_model = 'black-forest-labs/FLUX.1-schnell'
control_model = '/amax/hchuz/hfd_models/FLUX.1-dev-ControlNet-Union-Pro'
#control_model = '/amax/hchuz/hfd_models/FLUX.1-Depth-dev-lora'
#control_model = '/amax/hchuz/hfd_models/FLUX.1-Canny-dev-lora'
flux_redux = 'black-forest-labs/FLUX.1-Redux-dev'
#flux_fill = '/amax/hchuz/hfd_models/FLUX.1-Fill-dev'
flux_fill = None
# dino_checkpoint = '/amax/hchuz/OMG-master/checkpoint/GroundingDINO'
# sam_checkpoint = '/amax/hchuz/OMG-master/checkpoint/sam/sam_vit_h_4b8939.pth'
dino_checkpoint = None
sam_checkpoint = None
#lora_weights = '/amax/hchuz/ai-toolkit/output/my_first_flux_lora_v1/my_first_flux_lora_v1.safetensors'
lora_weights = None
offload = False
seeds = [111, 222, 333, 444]
output_dir = 'results/imgs_eval_training_proc_control_focus'
os.makedirs(output_dir, exist_ok=True)

no_control = False
use_focus = True

generator = Generator(base_model, control_model, flux_redux, offload, dino_checkpoint=dino_checkpoint, sam_checkpoint=sam_checkpoint, flux_fill=flux_fill, lora_weights=lora_weights, no_control=no_control)
cannydetect = CannyDetect()

#processor = DepthPreprocessor.from_pretrained("/amax/hchuz/hfd_models/depth-anything-large-hf").to('cuda')

#region_style_image = Image.open('temp/Snipaste_2023-02-21_13-33-51_window.png')

data_dir = '/amax/hchuz/architectural_heritage/data/training_dataset_proc'
names = [name[:-len('.json')] for name in os.listdir(data_dir) if name.endswith('.json')]

for name in tqdm(names):
    with open(os.path.join(data_dir, name+'.json'), 'r') as f:
        data = json.load(f)
    # with open(os.path.join(data_dir, name+'.txt'), 'r') as f:
    #     description = f.read()
    global_style_image = Image.open(os.path.join(data_dir, name+'.png')).convert('RGB')
    #if global_style_image.height * global_style_image.width > 1024*1024:
    #ratio = (global_style_image.height * global_style_image.width) / (1024*1024)
    #global_style_image = down_size(global_style_image, ratio**0.5)
    

    #save_path = os.path.join(output_dir, f'{name}_control.png')
    #if not os.path.exists(save_path):
        #control_image = processor(global_style_image)[0].convert("RGB")
        #control_image = cannydetect.det_canny(global_style_image)
        #control_image = cannydetect.polygon2canny_2([sample['points'] for sample in data['shapes']], global_style_image.height, global_style_image.width)
        
        #control_image = cannydetect.polygon2canny([sample['points'] for sample in data['shapes']])
        #control_image = cannydetect.processor(global_style_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)
        #control_image.save(save_path)
    #control_image = Image.open(save_path)
    control_image = Image.open(os.path.join(data_dir, name+'_control.png'))
    for seed in seeds:
        save_path = os.path.join(output_dir, f'{name}_{seed}.png')
        if os.path.exists(save_path):
            continue
        # style & control

        kwargs = {'seed':seed, 'global_style_image':global_style_image}
        if not no_control:
            kwargs.update({'control_image':[control_image]})
        
        if use_focus:
            # focus
            height = global_style_image.height
            width = global_style_image.width
            height_lt = height//16
            width_lt = width//16
            ori_tokens = height_lt * width_lt
            num_ref_imgs = 1
            total_tokens = 1241 * num_ref_imgs + ori_tokens
            attention_mask = torch.ones(1, total_tokens, total_tokens, dtype=torch.bool, device='cuda')
            itr = attention_mask[:, 1241*num_ref_imgs:, 512:1241]
            def mask_focus(it, mask_ori, mask_ref):
                it[:, mask_ori.flatten().bool()] = False
                it[:,(torch.outer(mask_ori.flatten(), mask_ref.flatten())).bool()] = True
            # focus on origin
            for shape in data['shapes']:
                mask_ori = get_mask_ori(shape['points'], height, width, height_lt, width_lt)
                mask_ref = get_mask_ref(mask_ori)
                mask_focus(itr, mask_ori, mask_ref)
            kwargs.update({'joint_attention_kwargs': {'attention_mask': attention_mask}})

        image_stage1 = generator.stage1(**kwargs)
        #image_stage1 = generator.stage1(seed, control_image=control_image, global_style_image=global_style_image)
        # no control
        #image_stage1 = generator.stage1(seed, global_style_image=global_style_image)
        # no style
        #image_stage1 = generator.stage1(control_image, seed=seed, prompt=' ', global_style_image=None, controlnet_conditioning_scale=[0.4])
        image_stage1.save(save_path)
        # image_stage2 = generator.stage2(image_stage1, 'window', region_style_image, seed=seed)
        # image_stage2.save(os.path.join(output_dir, f'{name}_window_{seed}.png'))
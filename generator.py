import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxPriorReduxPipeline, FluxFillPipeline, FluxPipeline
from diffusers.models import FluxMultiControlNetModel
from detector.groundingdino_sam import GroundingdinoSam

class Generator:
    def __init__(self, 
                 base_model, 
                 controlnet_model_union, 
                 flux_redux, 
                 offload,
                 dino_checkpoint=None, 
                 sam_checkpoint=None, 
                 flux_fill=None,
                 lora_weights = None,
                 no_control = False
                 ):
        # load models
        controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
        controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel
        if no_control:
            self.pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
        else:
            self.pipe = FluxControlNetPipeline.from_pretrained(base_model, 
                                                           controlnet=controlnet, 
                                                           torch_dtype=torch.bfloat16)
        if lora_weights != None:
            self.pipe.load_lora_weights(lora_weights)
        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(flux_redux, torch_dtype=torch.bfloat16).to('cuda')
        if dino_checkpoint != None and sam_checkpoint != None:
            self.groundingdino_sam = GroundingdinoSam(dino_checkpoint, sam_checkpoint)
        if flux_fill != None:
            self.pipe2 = FluxFillPipeline.from_pretrained(flux_fill, 
                                                        text_encoder=None,
                                                        text_encoder_2=None,
                                                        torch_dtype=torch.bfloat16)
        self.offload = offload
        if offload:
            self.pipe.enable_model_cpu_offload()
            if flux_fill != None:
                self.pipe2.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
            if flux_fill != None:
                self.pipe2.to('cuda')

    def stage1(self, control_image, seed, global_style_image=None, prompt=None, **kwargs):
        kwargs_ = {
            'controlnet_conditioning_scale':[0.4],
            'num_inference_steps':20, 
            'guidance_scale':3.5
            }
        kwargs_.update(kwargs) # you may specify kwargs
        assert (global_style_image or prompt) and not (global_style_image and prompt)
        if global_style_image != None:
            global_style_input = self.pipe_prior_redux(global_style_image)
        else :
            global_style_input = {'prompt': prompt}
        image_stage1 = self.pipe(
            prompt, 
            **global_style_input,
            control_image=[control_image],
            control_mode=[0], # 0~canny https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union
            width=global_style_image.width//16*16,
            height=global_style_image.height//16*16,
            generator=torch.manual_seed(seed),
            **kwargs_
        ).images[0]
        return image_stage1
    
    def detect_all_mask_with_prompt(self, detect_prompt, image_stage1):
        masks = self.groundingdino_sam.pred_mask_with_prompt(image_stage1, detect_prompt).cpu()
        mask = torch.zeros_like(masks[0])
        for m in masks:
            mask = mask | m
        #Image.fromarray(mask[0].numpy())
        return mask

    def stage2(self, image_stage1, detect_prompt, region_style_image, seed, **kwargs):
        kwargs_ = {
            'guidance_scale':30,
            'num_inference_steps':20,
            }
        kwargs_.update(kwargs) # you may specify kwargs

        mask = self.detect_all_mask_with_prompt(detect_prompt, image_stage1)
        region_style_input = self.pipe_prior_redux(region_style_image)
        image2 = self.pipe2(
            **region_style_input,
            image=image_stage1,
            mask_image=mask.float(),
            height=image_stage1.height,
            width=image_stage1.width,
            generator=torch.Generator("cpu").manual_seed(seed),
            **kwargs_
        ).images[0]
        return image2
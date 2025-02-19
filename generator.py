import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxPriorReduxPipeline, FluxFillPipeline
from diffusers.models import FluxMultiControlNetModel
from detector.groundingdino_sam import GroundingdinoSam

class Generator:
    def __init__(self, base_model, controlnet_model_union, flux_redux, flux_fill, dino_checkpoint, sam_checkpoint, offload):
        # load models
        controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
        controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel
        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(flux_redux, torch_dtype=torch.bfloat16).to('cuda')
        self.pipe = FluxControlNetPipeline.from_pretrained(base_model, 
                                                           controlnet=controlnet, 
                                                           text_encoder=None,
                                                           text_encoder_2=None,
                                                           torch_dtype=torch.bfloat16)
        self.groundingdino_sam = GroundingdinoSam(dino_checkpoint, sam_checkpoint)
        self.pipe2 = FluxFillPipeline.from_pretrained(flux_fill, 
                                                    text_encoder=None,
                                                    text_encoder_2=None,
                                                    torch_dtype=torch.bfloat16)
        if offload:
            self.pipe.enable_model_cpu_offload()
            self.pipe2.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
            self.pipe2.to('cuda')

    def stage1(self, global_style_image, control_image, seed, **kwargs):
        kwargs_ = {
            'controlnet_conditioning_scale':[0.4],
            'num_inference_steps':20, 
            'guidance_scale':3.5
            }
        kwargs_.update(kwargs) # you may specify kwargs

        global_style_input = self.pipe_prior_redux(global_style_image)
        image_stage1 = self.pipe(
            #prompt, 
            **global_style_input,
            control_image=[control_image],
            control_mode=[0], # 0~canny https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union
            width=control_image.width,
            height=control_image.height,
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
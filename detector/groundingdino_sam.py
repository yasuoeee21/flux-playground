from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import build_sam, SamPredictor
import groundingdino.datasets.transforms as T
import torch
import os
import numpy as np
from typing import Tuple

class GroundingdinoSam:
    def __init__(self, dino_checkpoint, sam_checkpoint):
        self.load_groundingdino(dino_checkpoint)
        self.load_sam(sam_checkpoint)

    def load_groundingdino(self, repo_id):
        filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = os.path.join(repo_id, "GroundingDINO_SwinB.cfg.py")
        args = SLConfig.fromfile(ckpt_config_filename)
        self.groundingdino = build_model(args)
        checkpoint = torch.load(os.path.join(repo_id, filename), map_location='cpu')
        log = self.groundingdino.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(filename, log))
        _ = self.groundingdino.eval()
    
    def load_sam(self, sam_checkpoint):
        sam = build_sam(checkpoint = sam_checkpoint)
        sam.cuda()
        self.sam = SamPredictor(sam)
    
    @staticmethod
    def load_image_dino(image) -> Tuple[np.array, torch.Tensor]:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = np.asarray(image)
        image_transformed, _ = transform(image, None)
        return image_source, image_transformed

    def pred_mask_with_prompt(self, image_rgb, prompt = 'person'):
        image_source, image_transformed = self.load_image_dino(image_rgb)
        boxes, logits, phrases = predict(
            model=self.groundingdino,
            image=image_transformed,
            caption=prompt,
            box_threshold=0.3,
            text_threshold=0.25
        )

        # 根据size排序
        size = torch.stack([box[2]*box[3] for box in boxes])
        _, indices = torch.sort(size, descending=True)
        # 根据排序索引对第一个张量进行排序
        boxes = boxes[indices]

        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        self.sam.set_image(image_source)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks

    def pred_box_with_prompt(self, image_rgb, prompt = 'person'):
        image_source, image_transformed = self.load_image_dino(image_rgb)
        boxes, logits, phrases = predict(
            model=self.groundingdino,
            image=image_transformed,
            caption=prompt,
            box_threshold=0.3,
            text_threshold=0.25
        )

        # if not detected
        if len(boxes) == 0:
            return None

        # 根据size排序
        size = torch.stack([box[2]*box[3] for box in boxes])
        _, indices = torch.sort(size, descending=True)
        # 根据排序索引对第一个张量进行排序
        boxes = boxes[indices]
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        boxes_xyxy = boxes_xyxy.numpy().tolist()
        return boxes_xyxy
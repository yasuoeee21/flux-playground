export CUDA_VISIBLE_DEVICES=1
accelerate launch --main_process_port 30000 --config_file=/amax/hchuz/architectural_heritage/configs/deepspeed.yaml train_control_lora_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --jsonl_for_train="/amax/hchuz/architectural_heritage/data/control/single.jsonl" \
  --output_dir="results/control_single_1e-3" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --rank=64 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-3 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=5000 \
  --validation_image="/amax/hchuz/architectural_heritage/data/control/SegmentationClass/Snipaste_2023-02-21_12-59-54.png" \
  --validation_prompt="The building in the image appears to be constructed using a combination of materials that give it a classic and somewhat historical appearance. Here are some observations about the materials:\n\n1. **Facade**: The exterior facade seems to be made of stone or a stone-like material, possibly concrete or stucco, which has been painted or finished to resemble stone. The texture and color suggest a durable, solid material.\n\n2. **Windows**: The windows have decorative frames, likely made of wood or a composite material designed to mimic wood. The design includes intricate details and moldings, which are characteristic of traditional architecture.\n\n3. **Balconies and Railings**: The balconies and railings appear to be made of metal, possibly wrought iron or a similar material, which is often used for its strength and decorative potential.\n\n4. **Signage**: The signage on the building is made of a smooth, possibly metallic or plastic material, with engraved or painted characters. The central sign is framed with a decorative border, suggesting attention to aesthetic detail.\n\n5. **Lighting**: The building is illuminated with modern lighting fixtures, which are likely made of metal and glass, providing a warm, inviting glow.\n\nOverall, the building combines traditional architectural elements with modern materials, creating a visually appealing and functional structure." \
  --seed="0"
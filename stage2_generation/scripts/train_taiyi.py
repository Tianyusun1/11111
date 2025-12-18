import argparse
import logging
import os
import math
import random
from pathlib import Path

# =========================================================
# [CRITICAL PATCH] 修复受限环境下的 PermissionError
# =========================================================
try:
    EnvironClass = os.environ.__class__
    _orig_setitem = EnvironClass.__setitem__
    _orig_delitem = EnvironClass.__delitem__

    def _safe_setitem(self, key, value):
        try:
            _orig_setitem(self, key, value)
        except PermissionError:
            pass
        except Exception as e:
            raise e

    def _safe_delitem(self, key):
        try:
            _orig_delitem(self, key)
        except PermissionError:
            pass
        except KeyError:
            pass
        except Exception as e:
            raise e

    EnvironClass.__setitem__ = _safe_setitem
    EnvironClass.__delitem__ = _safe_delitem
    
    def _safe_clear(self):
        keys = list(self.keys())
        for key in keys:
            self.pop(key, None)
            
    EnvironClass.clear = _safe_clear
    print("✅ Environment monkey-patch applied successfully.")
except Exception as e:
    print(f"⚠️ Failed to patch environment: {e}")

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
)
from diffusers.optimization import get_scheduler

# [NEW] 引入 LoRA 库，用于安全微调底座
from peft import LoraConfig, get_peft_model

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Idea-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    parser.add_argument("--output_dir", type=str, default="taiyi_controlnet_lora_output")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4) 
    parser.add_argument("--num_train_epochs", type=int, default=10)
    
    # [CONFIG] 学习率设置
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="ControlNet的学习率")
    parser.add_argument("--learning_rate_lora", type=float, default=1e-4, help="UNet LoRA的学习率")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16") 
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--lambda_struct", type=float, default=0.1, help="结构对齐损失权重")
    
    # [CONFIG] LoRA 设置
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA的秩，越大适应能力越强但参数越多")
    
    args = parser.parse_args()

    # [FIX] 启动前确保输出主目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device

    # 1. 加载基础模型组件
    tokenizer = transformers.BertTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = transformers.BertModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    if accelerator.is_main_process:
        print("正在从 UNet 初始化 结构-风格 双流 ControlNet...")
    controlnet_s = ControlNetModel.from_unet(unet)
    controlnet_t = ControlNetModel.from_unet(unet)

    # 2. 冻结与 LoRA 注入策略
    # 首先冻结所有模型
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) # UNet 主干冻结
    
    # [NEW] 配置 UNet 的 LoRA
    # 针对 Attention 模块注入适配器，使其学习水墨画风
    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    
    # 将 LoRA 挂载到 UNet (unet 变为 PeftModel)
    unet = get_peft_model(unet, unet_lora_config)
    
    if accelerator.is_main_process:
        print("✅ LoRA 注入成功，UNet 可训练参数如下:")
        unet.print_trainable_parameters()

    # 开启 xformers 显存优化
    try:
        # LoRA 训练时若使用 fp16 可能会有数值稳定性问题，但 xformers 通常能处理
        unet.enable_xformers_memory_efficient_attention()
        controlnet_s.enable_xformers_memory_efficient_attention()
        controlnet_t.enable_xformers_memory_efficient_attention()
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Warning: xformers 未安装或不可用: {e}")

    # 3. 优化器准备 (分组学习率)
    params_to_optimize = [
        {"params": controlnet_s.parameters(), "lr": args.learning_rate},
        {"params": controlnet_t.parameters(), "lr": args.learning_rate},
        {"params": unet.parameters(), "lr": args.learning_rate_lora} # 这里实际优化的是 LoRA 参数
    ]
    optimizer = torch.optim.AdamW(params_to_optimize)

    # 4. 数据加载与划分
    raw_dataset = load_dataset("json", data_files=os.path.join(args.train_data_dir, "train.jsonl"))["train"]
    
    train_testvalid = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_testvalid['train']
    
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    val_dataset = test_valid['train']
    test_dataset = test_valid['test']
    
    if accelerator.is_main_process:
        print(f"📊 数据集划分完成: Train={len(train_dataset)} | Val={len(val_dataset)} | Test={len(test_dataset)}")

    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    cond_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(), 
    ])

    def collate_fn(examples):
        pixel_values, cond_pixel_values, input_ids, raw_texts = [], [], [], []
        for example in examples:
            try:
                img_path = os.path.join(args.train_data_dir, example["image"])
                cond_path = os.path.join(args.train_data_dir, example["conditioning_image"])
                pixel_values.append(transform(Image.open(img_path).convert("RGB")))
                cond_pixel_values.append(cond_transform(Image.open(cond_path).convert("RGB")))
                
                # [USER] 始终使用完整文本
                caption = example["text"]
                inputs = tokenizer(caption, max_length=tokenizer.model_max_length, 
                                 padding="max_length", truncation=True, return_tensors="pt")
                input_ids.append(inputs.input_ids[0])
                raw_texts.append(example["text"])
            except: continue
        return {
            "pixel_values": torch.stack(pixel_values),
            "conditioning_pixel_values": torch.stack(cond_pixel_values),
            "input_ids": torch.stack(input_ids),
            "texts": raw_texts
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    # 准备 Accelerator (注意 UNet 也要 prepare)
    controlnet_s, controlnet_t, unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        controlnet_s, controlnet_t, unet, optimizer, train_dataloader, val_dataloader
    )
    
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    # 5. 训练循环
    global_step = 0
    if accelerator.is_main_process:
        print(f"🚀 启动安全训练 (LoRA + ControlNet Dropout)")
        
    for epoch in range(args.num_train_epochs):
        controlnet_s.train()
        controlnet_t.train()
        unet.train()
        
        train_loss_epoch = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet_s, controlnet_t, unet):
                target_images = batch["pixel_values"].to(dtype=torch.float16)
                latents = vae.encode(target_images).latent_dist.sample() * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                cond_image = batch["conditioning_pixel_values"].to(dtype=torch.float16)
                
                # [STRATEGY] ControlNet Dropout
                # 随机屏蔽某一流，迫使模型独立学习每一流的特征
                # 85% 双流 | 7.5% 仅结构 | 7.5% 仅风格
                rand_dropout = random.random()
                cond_s = cond_image
                cond_t = cond_image
                
                if rand_dropout < 0.075:
                    cond_s = torch.zeros_like(cond_image) # 屏蔽 Structure
                elif rand_dropout < 0.15:
                    cond_t = torch.zeros_like(cond_image) # 屏蔽 Style
                
                down_s, mid_s = controlnet_s(noisy_latents, timesteps, encoder_hidden_states, cond_s, return_dict=False)
                down_t, mid_t = controlnet_t(noisy_latents, timesteps, encoder_hidden_states, cond_t, return_dict=False)
                
                down_res = [(s.to(dtype=torch.float16) + t.to(dtype=torch.float16)) for s, t in zip(down_s, down_t)]
                mid_res = mid_s.to(dtype=torch.float16) + mid_t.to(dtype=torch.float16)

                # UNet 前向 (包含 LoRA)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, 
                                 down_block_additional_residuals=down_res, 
                                 mid_block_additional_residual=mid_res).sample

                # 损失计算 (仅保留 MSE 和 结构引导)
                loss_ddpm = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                loss_struct = torch.tensor(0.0).to(device)
                if rand_dropout >= 0.075: 
                    cond_resized = F.interpolate(cond_s, size=mid_s.shape[-2:], mode="bilinear")
                    loss_struct = F.l1_loss(mid_s.mean(dim=1, keepdim=True), cond_resized.mean(dim=1, keepdim=True))
                
                total_loss = loss_ddpm + args.lambda_struct * loss_struct
                
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss_epoch += total_loss.item()
            global_step += 1
            
            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                os.makedirs(ckpt_dir, exist_ok=True)
                
                accelerator.unwrap_model(controlnet_s).save_pretrained(ckpt_dir / "structure")
                accelerator.unwrap_model(controlnet_t).save_pretrained(ckpt_dir / "style")
                
                # [SAFE SAVE] 只保存 LoRA 权重，不保存 3GB 的 UNet
                accelerator.unwrap_model(unet).save_pretrained(ckpt_dir / "unet_lora")
                print(f"💾 Checkpoint saved at step {global_step} (LoRA Safe Mode)")

            if step % 50 == 0 and accelerator.is_main_process:
                print(f"E{epoch} S{step} | Total: {total_loss:.4f} (DDPM: {loss_ddpm:.4f} | Struct: {loss_struct:.4f})")

        # === 验证阶段 ===
        if accelerator.is_main_process:
            print(f"🔍 Epoch {epoch}: 正在进行验证集评估...")
        
        controlnet_s.eval()
        controlnet_t.eval()
        unet.eval() # 验证时 LoRA 也会被冻结
        val_loss_total = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                target_images = batch["pixel_values"].to(dtype=torch.float16)
                latents = vae.encode(target_images).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                cond_image = batch["conditioning_pixel_values"].to(dtype=torch.float16)
                
                down_s, mid_s = controlnet_s(noisy_latents, timesteps, encoder_hidden_states, cond_image, return_dict=False)
                down_t, mid_t = controlnet_t(noisy_latents, timesteps, encoder_hidden_states, cond_image, return_dict=False)
                down_res = [(s.to(dtype=torch.float16) + t.to(dtype=torch.float16)) for s, t in zip(down_s, down_t)]
                mid_res = mid_s.to(dtype=torch.float16) + mid_t.to(dtype=torch.float16)
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, 
                                  down_block_additional_residuals=down_res, 
                                  mid_block_additional_residual=mid_res).sample
                                  
                loss_ddpm = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                val_loss_total += loss_ddpm.item()
                val_steps += 1
        
        avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0
        
        if accelerator.is_main_process:
            avg_train_loss = train_loss_epoch / len(train_dataloader)
            print(f"📊 Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- 验证采样 ---
        if accelerator.is_main_process:
            print(f"🎨 Epoch {epoch} 结束，使用验证集生成样图...")
            with torch.autocast(device.type, dtype=torch.float16):
                with torch.no_grad():
                    unwrapped_s = accelerator.unwrap_model(controlnet_s)
                    unwrapped_t = accelerator.unwrap_model(controlnet_t)
                    # 此时 unwrapped_unet 包含 LoRA 参数
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    
                    # 手动构建 Pipeline，复用内存中的组件
                    val_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                    pipe = StableDiffusionControlNetPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=unwrapped_unet, # 带有 LoRA 的 UNet
                        controlnet=[unwrapped_s, unwrapped_t],
                        scheduler=val_scheduler,
                        safety_checker=None,
                        feature_extractor=None
                    ).to(device)
                    
                    test_prompt = batch["texts"][0] if len(batch["texts"]) > 0 else "中国水墨山水画"
                    test_cond = cond_image[0:1].to(device=device, dtype=torch.float16)
                    
                    # 保存
                    layout_img_pil = transforms.ToPILImage()(test_cond.squeeze(0).cpu())
                    os.makedirs(args.output_dir, exist_ok=True)
                    layout_img_pil.save(Path(args.output_dir) / f"layout_epoch_{epoch}_val.png")

                    sample_out = pipe(
                        prompt=test_prompt, 
                        image=[test_cond, test_cond], 
                        num_inference_steps=20,
                        guidance_scale=7.5
                    ).images[0]
                    
                    sample_out.save(Path(args.output_dir) / f"sample_epoch_{epoch}_val.png")
                    print(f"✅ 验证图已保存")
                    
                    del pipe, val_scheduler
                    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        # 保存最终模型
        save_path_s = Path(args.output_dir) / "controlnet_structure"
        save_path_t = Path(args.output_dir) / "controlnet_style"
        os.makedirs(save_path_s, exist_ok=True)
        os.makedirs(save_path_t, exist_ok=True)
        
        accelerator.unwrap_model(controlnet_s).save_pretrained(save_path_s)
        accelerator.unwrap_model(controlnet_t).save_pretrained(save_path_t)
        # 仅保存 LoRA 权重 (安全、轻量)
        accelerator.unwrap_model(unet).save_pretrained(Path(args.output_dir) / "unet_lora")
        print(f"✅ 训练圆满完成，LoRA 与 ControlNets 已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()
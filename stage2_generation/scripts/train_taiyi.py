# File: stage2_generation/scripts/train_taiyi.py (V11.2: Full Restore + Logic Upgrade)

import argparse
import logging
import os
import math
import random
from pathlib import Path
import sys
import matplotlib.pyplot as plt

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
import torchvision.models as models
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline, # [NEW] 引入 Pipeline 用于验证
)
from peft import LoraConfig, get_peft_model

logger = get_logger(__name__)

# =========================================================
# [Loss 模块] 掩码引导的感知与风格损失 (VGG19)
# =========================================================
class VGGPerceptualAndStyleLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        print("🎨 Loading VGG19 for Mask-Guided Style Loss...")
        # 加载预训练 VGG19 并冻结
        # 使用 weights=models.VGG19_Weights.DEFAULT 替代 deprecated 的 pretrained=True
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 切分层级：浅层(纹理)，深层(结构)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2): self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7): self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12): self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21): self.slice4.add_module(str(x), vgg[x])
            
        # 注意：这里我们只用到前21层做Style，后面的层可以不用
        # for x in range(21, 30): self.slice5.add_module(str(x), vgg[x])
            
        self.to(device)
        # ImageNet 归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def gram_matrix(self, input, mask=None):
        """
        计算 Gram Matrix。如果提供 Mask，则只计算 Mask 区域的纹理统计。
        """
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        
        if mask is not None:
            # 确保 mask 是单通道并 resize 到特征图大小
            # mask: [B, 1, h, w] -> view -> [B, 1, h*w]
            # interpolate 需要 input 为 [B, C, H, W]
            mask_resized = F.interpolate(mask, size=(h, w), mode='nearest').view(b, 1, h * w)
            
            # 广播乘法
            features = features * mask_resized
            
            # 计算有效像素数作为归一化因子
            valid_elements = mask_resized.sum(dim=-1, keepdim=True) * c
            valid_elements = valid_elements.clamp(min=1.0)
            
            G = torch.bmm(features, features.transpose(1, 2))
            return G.div(valid_elements)
        else:
            G = torch.bmm(features, features.transpose(1, 2))
            return G.div(c * h * w)

    def forward(self, input, target, mask_img=None):
        # 输入 [-1, 1] -> [0, 1] -> ImageNet Norm
        input = (input + 1) / 2.0
        target = (target + 1) / 2.0
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        style_loss = 0.0
        content_loss = 0.0
        
        x = input
        y = target
        
        # 预处理 Mask: 
        # mask_img 期望是单通道 [B, 1, H, W]，且范围 [-1, 1]
        # 墨色(前景) < 0
        if mask_img is not None:
            fg_mask = (mask_img < 0.0).float()
        else:
            fg_mask = None

        # 只用到 slice1-4
        slices = [self.slice1, self.slice2, self.slice3, self.slice4]
        
        for i, slice_net in enumerate(slices):
            x = slice_net(x)
            y = slice_net(y)
            
            # --- Style Loss (每一层都算) ---
            # 1. 前景风格 (Masked Gram): 强迫墨块区域学到笔触纹理
            if fg_mask is not None:
                gm_x_fg = self.gram_matrix(x, fg_mask)
                gm_y_fg = self.gram_matrix(y, fg_mask)
                style_loss += F.mse_loss(gm_x_fg, gm_y_fg) * 2.0 
            
            # 2. 全局风格
            gm_x_global = self.gram_matrix(x, None)
            gm_y_global = self.gram_matrix(y, None)
            style_loss += F.mse_loss(gm_x_global, gm_y_global)

            # --- Content Loss (用较深层，如第3层) ---
            if i == 2: 
                content_loss += F.l1_loss(x, y)
            
        return style_loss, content_loss

# =========================================================
# 可学习的多尺度权重模块 (ControlNetScaler)
# =========================================================
class ControlNetScaler(torch.nn.Module):
    def __init__(self, num_scales=13, init_value=1.0):
        super().__init__()
        self.scales = torch.nn.Parameter(torch.full((num_scales,), init_value, dtype=torch.float32))

    def forward(self, down_samples, mid_sample):
        weighted_down = []
        for i, sample in enumerate(down_samples):
            weighted_down.append(sample * self.scales[i])
        weighted_mid = mid_sample * self.scales[-1]
        return weighted_down, weighted_mid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Idea-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    parser.add_argument("--output_dir", type=str, default="taiyi_controlnet_lora_output")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4) 
    parser.add_argument("--num_train_epochs", type=int, default=10)
    
    # [CONFIG] 学习率设置
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--learning_rate_lora", type=float, default=1e-4)
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16") 
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--lora_rank", type=int, default=64)
    
    # [CONFIG] 损失权重
    parser.add_argument("--style_loss_weight", type=float, default=100.0)
    parser.add_argument("--content_loss_weight", type=float, default=1.0)
    parser.add_argument("--layout_focal_weight", type=float, default=5.0)
    
    # [CONFIG] 训练策略
    parser.add_argument("--smart_freeze", action="store_true", default=True)
    parser.add_argument("--prompt_drop_rate", type=float, default=0.20)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(f"✨ [V11.2 最终完整版] 启动！Soft-Focal + Timestep-Aware + Validation")

    # 1. 加载模型
    tokenizer = transformers.BertTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = transformers.BertModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    controlnet = ControlNetModel.from_unet(unet)
    vgg_loss_fn = VGGPerceptualAndStyleLoss(device)

    # 2. LoRA 设置
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) 
    
    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    
    control_scaler = ControlNetScaler(num_scales=13, init_value=1.0)
    control_scaler.to(device)
    control_scaler.train()

    if accelerator.is_main_process:
        print("✅ LoRA 注入成功")
        print("✅ Scaler 初始化成功")

    try:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    if args.smart_freeze:
        controlnet.requires_grad_(False) 
        for n, p in controlnet.controlnet_cond_embedding.named_parameters(): p.requires_grad = True
        for n, p in controlnet.conv_in.named_parameters(): p.requires_grad = True
        for n, p in controlnet.controlnet_down_blocks.named_parameters(): p.requires_grad = True
        for n, p in controlnet.controlnet_mid_block.named_parameters(): p.requires_grad = True
            
    params_to_optimize = [
        {"params": filter(lambda p: p.requires_grad, controlnet.parameters()), "lr": args.learning_rate},
        {"params": unet.parameters(), "lr": args.learning_rate_lora},
        {"params": control_scaler.parameters(), "lr": 1e-3} 
    ]
    optimizer = torch.optim.AdamW(params_to_optimize)

    # 4. 数据加载
    raw_dataset = load_dataset("json", data_files=os.path.join(args.train_data_dir, "train.jsonl"))["train"]
    train_dataset = raw_dataset 

    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # [FIX 1] 加上 Normalize，确保 Mask 也是 [-1, 1]，这样才能判定 < 0.0
    cond_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])

    def collate_fn(examples):
        pixel_values, cond_pixel_values, input_ids = [], [], []
        for example in examples:
            try:
                img_path = os.path.join(args.train_data_dir, example["image"])
                cond_path = os.path.join(args.train_data_dir, example["conditioning_image"])
                pixel_values.append(transform(Image.open(img_path).convert("RGB")))
                cond_pixel_values.append(cond_transform(Image.open(cond_path).convert("RGB")))
                
                caption = example["text"]
                if random.random() < args.prompt_drop_rate:
                    caption = "" 
                
                inputs = tokenizer(caption, max_length=tokenizer.model_max_length, 
                                 padding="max_length", truncation=True, return_tensors="pt")
                input_ids.append(inputs.input_ids[0])
            except: continue
        return {
            "pixel_values": torch.stack(pixel_values),
            "conditioning_pixel_values": torch.stack(cond_pixel_values),
            "input_ids": torch.stack(input_ids),
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )

    controlnet, unet, control_scaler, optimizer, train_dataloader = accelerator.prepare(
        controlnet, unet, control_scaler, optimizer, train_dataloader
    )
    
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    loss_history = {'steps': [], 'total': [], 'mse': [], 'style': []}

    def plot_loss_curve(history, save_path):
        if len(history['steps']) < 2: return
        plt.figure(figsize=(10, 6))
        plt.plot(history['steps'], history['total'], label='Total Loss', color='blue', alpha=0.6)
        plt.plot(history['steps'], history['mse'], label='Focal MSE', color='orange', alpha=0.5, linestyle='--')
        plt.plot(history['steps'], history['style'], label='Style', color='green', alpha=0.5, linestyle=':')
        plt.title(f"Training Loss (Step {history['steps'][-1]})")
        plt.legend()
        plt.tight_layout()
        try: plt.savefig(save_path); plt.close()
        except: pass

    # 5. 训练循环
    global_step = 0
    if accelerator.is_main_process:
        print(f"🚀 启动训练流程... FocalWeight={args.layout_focal_weight}, StyleWeight={args.style_loss_weight}")
        
    for epoch in range(args.num_train_epochs):
        controlnet.train()
        unet.train()
        control_scaler.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, unet, control_scaler):
                target_images = batch["pixel_values"].to(dtype=torch.float16)
                cond_image = batch["conditioning_pixel_values"].to(dtype=torch.float16)
                
                # B. Latent Process
                latents = vae.encode(target_images).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                
                scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                if random.random() < 0.15:
                    cond_input = torch.zeros_like(cond_image)
                else:
                    cond_input = cond_image
                
                # C. Forward
                down_res, mid_res = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states, cond_input, return_dict=False
                )
                
                w_down, w_mid = control_scaler(down_res, mid_res)
                w_down = [s.to(dtype=torch.float16) for s in w_down]
                w_mid = w_mid.to(dtype=torch.float16)

                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states, 
                    down_block_additional_residuals=w_down,
                    mid_block_additional_residual=w_mid
                ).sample

                # === LOSS 计算 ===
                
                # [改进点 1] Soft-Weighted Focal MSE Loss
                # -----------------------------------------------------
                raw_mse = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                
                # 下采样 (cond_image is [-1, 1])
                cond_small = F.interpolate(cond_image.float(), size=raw_mse.shape[-2:], mode="nearest")
                cond_small_1ch = cond_small.mean(dim=1, keepdim=True) # range [-1, 1]
                
                # 软权重逻辑：val=-1(深墨) -> weight=Focal; val=1(白) -> weight=1.0
                soft_weight_map = 1.0 + (args.layout_focal_weight - 1.0) * ((1.0 - cond_small_1ch) / 2.0)
                
                loss_mse = (raw_mse * soft_weight_map).mean()
                
                # [改进点 2] Timestep-Aware Style Loss
                # -----------------------------------------------------
                loss_style = torch.tensor(0.0, device=device)
                loss_content = torch.tensor(0.0, device=device)
                
                # 策略：只在 t < 800 (画面稍微成型后) 计算 Style
                # 避免在高噪阶段强行拟合纹理，造成梯度混乱
                if timesteps.float().mean() < 800:
                    alphas_cumprod = scheduler.alphas_cumprod.to(device)
                    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    beta_prod_t = 1 - alpha_prod_t
                    pred_original_latents = (noisy_latents - beta_prod_t ** 0.5 * model_pred) / alpha_prod_t ** 0.5
                    
                    pred_images = vae.decode(pred_original_latents.to(dtype=vae.dtype) / vae.config.scaling_factor).sample
                    
                    cond_image_1ch = cond_image.float().mean(dim=1, keepdim=True)
                    loss_style, loss_content = vgg_loss_fn(
                        pred_images.float(), 
                        target_images.float(), 
                        mask_img=cond_image_1ch
                    )
                
                total_loss = loss_mse + \
                             args.style_loss_weight * loss_style + \
                             args.content_loss_weight * loss_content
                
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(controlnet).save_pretrained(ckpt_dir / "controlnet_structure") 
                accelerator.unwrap_model(unet).save_pretrained(ckpt_dir / "unet_lora")
                torch.save(accelerator.unwrap_model(control_scaler).state_dict(), ckpt_dir / "scaler.pth")
                print(f"💾 Checkpoint saved at step {global_step}")

            if step % 10 == 0 and accelerator.is_main_process:
                loss_history['steps'].append(global_step)
                loss_history['total'].append(total_loss.item())
                loss_history['mse'].append(loss_mse.item())
                loss_history['style'].append(loss_style.item())
                
                # [FIX] 科学计数法显示 Loss
                msg = (f"Ep {epoch+1} | Step {step} | Total: {total_loss.item():.4f} | MSE*: {loss_mse.item():.4f} | Style: {loss_style.item():.4e}")
                print(msg)
                logger.info(msg)
                
                if step % 100 == 0:
                    plot_loss_curve(loss_history, os.path.join(args.output_dir, "loss_curve.png"))

        # ==========================================
        # [NEW] Epoch 结束后的验证采样
        # ==========================================
        if accelerator.is_main_process:
            print(f"🎨 Epoch {epoch+1}: Generating Validation Sample...")
            try:
                # 临时切换到 Eval 模式
                controlnet.eval()
                unet.eval()
                torch.cuda.empty_cache()
                
                # 取 Batch 第一张图做验证条件
                # Tensor [-1, 1] -> [0, 1] -> PIL
                val_cond_tensor = cond_image[0:1].detach().cpu() 
                val_cond_tensor = (val_cond_tensor * 0.5 + 0.5).clamp(0, 1)
                val_cond_pil = transforms.ToPILImage()(val_cond_tensor.squeeze(0))
                
                pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    controlnet=accelerator.unwrap_model(controlnet),
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to(device)
                
                val_prompt = "明月松间照，清泉石上流。" # 固定 Prompt 观察效果
                image = pipeline(
                    prompt=val_prompt, 
                    image=val_cond_pil, 
                    num_inference_steps=20, 
                    guidance_scale=7.5
                ).images[0]
                
                # 1. 每个 Epoch 必存画作
                save_path = os.path.join(args.output_dir, f"val_epoch_{epoch+1}.png")
                image.save(save_path)
                
                # 2. 每 20 个 Epoch 存一次 Mask
                if (epoch + 1) % 20 == 0:
                    val_cond_pil.save(os.path.join(args.output_dir, f"val_epoch_{epoch+1}_mask.png"))
                
                print(f"✅ Validation saved to {save_path}")
                
                del pipeline
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Validation failed: {e}")

    if accelerator.is_main_process:
        save_path_c = Path(args.output_dir) / "controlnet_structure"
        os.makedirs(save_path_c, exist_ok=True)
        accelerator.unwrap_model(controlnet).save_pretrained(save_path_c)
        accelerator.unwrap_model(unet).save_pretrained(Path(args.output_dir) / "unet_lora")
        torch.save(accelerator.unwrap_model(control_scaler).state_dict(), Path(args.output_dir) / "scaler_final.pth")
        
        plot_loss_curve(loss_history, os.path.join(args.output_dir, "loss_curve_final.png"))
        print(f"✅ 训练完成，Loss 曲线已保存。")

if __name__ == "__main__":
    main()
# File: integrated_inference.py (V8.7: Single-Stream Batch Inference)

import os
import sys
import torch
import argparse
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler

# === 路径配置 ===
# 确保项目根目录在 PYTHONPATH 中
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目内部组件
try:
    from models.poem2layout import Poem2LayoutGenerator
    from inference.greedy_decode import greedy_decode_poem_layout
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
    from data.visualize import draw_layout
except ImportError as e:
    print(f"[Error] 模块导入失败: {e}")
    print("请确保在项目根目录下运行，或正确设置 PYTHONPATH")
    sys.exit(1)

# =============================================================
# [架构创新] 自适应多尺度权重模块 (ControlNetScaler)
# =============================================================
class ControlNetScaler(torch.nn.Module):
    def __init__(self, num_scales=13, init_value=1.0):
        super().__init__()
        self.scales = torch.nn.Parameter(torch.full((num_scales,), init_value, dtype=torch.float32))

    def forward(self, down_samples, mid_sample):
        weighted_down = []
        for i, sample in enumerate(down_samples):
            dtype = sample.dtype
            weighted_down.append(sample * self.scales[i].to(dtype))
        
        dtype = mid_sample.dtype
        weighted_mid = mid_sample * self.scales[-1].to(dtype)
        return weighted_down, weighted_mid

# =============================================================
# [架构创新] 态势锚定处理器 (GAP Module)
# =============================================================
class PoemInkAttentionProcessor:
    def __init__(self, dynamic_layout, tokenizer, prompt, device, scale=8.0):
        self.layout = dynamic_layout  
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.device = device
        self.scale = scale 
        self.class_to_keyword = {
            2: "山", 3: "水", 4: "人", 5: "树", 6: "屋", 
            7: "桥", 8: "花", 9: "鸟", 10: "兽"
        }

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Gestalt Anchoring
        res = int(np.sqrt(sequence_length))
        h, w = res, res
        tokens = self.tokenizer.encode(self.prompt)
        
        for item in self.layout:
            cls_id = int(item[0])
            keyword = self.class_to_keyword.get(cls_id, None)
            if not keyword: continue
            
            cx, cy, bw, bh = item[1], item[2], item[3], item[4]
            if len(item) >= 7:
                bx, by = item[5], item[6]
            else:
                bx, by = 0.0, 0.0
            
            keyword_token_ids = self.tokenizer.encode(keyword, add_special_tokens=False)
            token_indices = [i for i, t in enumerate(tokens) if t in keyword_token_ids]
            if not token_indices: continue

            x_c, y_c = (cx + bx * 0.15) * w, (cy + by * 0.15) * h
            x1, y1 = int(x_c - (bw/2)*w), int(y_c - (bh/2)*h)
            x2, y2 = int(x_c + (bw/2)*w), int(y_c + (bh/2)*h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                for idx in token_indices:
                    if idx >= attention_probs.shape[-1]: continue
                    mask = torch.zeros((h, w), device=self.device)
                    mask[y1:y2, x1:x2] = self.scale
                    mask_flat = mask.flatten()
                    attention_probs[:, :, idx] += mask_flat * attention_probs[:, :, idx]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

# =============================================================
# 参数解析
# =============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Batch Inference (V8.7 Single Stream)")
    
    # 路径参数
    parser.add_argument("--layout_ckpt", type=str, required=True, help="Stage 1 Checkpoint")
    parser.add_argument("--taiyi_model_path", type=str, default="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    
    # Stage 2 路径 (目录或文件)
    # 建议指向包含 controlnet_structure, unet_lora, scaler.pth 的目录
    parser.add_argument("--stage2_checkpoint", type=str, required=True, help="Path to Stage 2 output dir")
    
    # 输出
    parser.add_argument("--output_base", type=str, default="outputs/batch_inference_v8", help="Output directory")
    
    return parser.parse_args()

# =============================================================
# 主逻辑
# =============================================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running Batch Inference on: {device}")
    
    # 1. 加载 Layout Generator
    # 尝试自动读取 config
    config_path = os.path.join(project_root, "configs", "default.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_cfg = config.get('model', {})
    else:
        model_cfg = {'hidden_size': 768, 'bb_size': 128}

    print("[Stage 1] Loading Poem2Layout Generator...")
    # [FIX] 显式指向 text_encoder 子目录以加载 BERT
    bert_subpath = os.path.join(args.taiyi_model_path, "text_encoder")
    
    layout_model = Poem2LayoutGenerator(
        bert_path=bert_subpath,  # <--- 修改处：拼接子目录
        num_classes=9,
        hidden_size=model_cfg.get('hidden_size', 768),
        bb_size=model_cfg.get('bb_size', 128),
        decoder_layers=model_cfg.get('decoder_layers', 6),
        decoder_heads=model_cfg.get('decoder_heads', 8),
        latent_dim=model_cfg.get('latent_dim', 64)
    ).to(device).eval()
    
    # 加载 Layout 权重
    try:
        ckpt = torch.load(args.layout_ckpt, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # 去除 module. 前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        layout_model.load_state_dict(state_dict, strict=False)
        print("✅ Layout model loaded.")
    except Exception as e:
        print(f"⚠️ Layout model load failed: {e}")
        return

    # 2. 加载 Single ControlNet & Pipeline
    print("[Stage 2] Loading Single Stream System...")
    
    # 自动寻找 controlnet 目录
    c_path = os.path.join(args.stage2_checkpoint, "controlnet_structure")
    if not os.path.exists(c_path): c_path = args.stage2_checkpoint
    
    controlnet = ControlNetModel.from_pretrained(c_path, torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.taiyi_model_path, # 保持指向根目录，用于加载 model_index.json
        controlnet=controlnet, # 单流
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    # 加载 LoRA
    lora_path = os.path.join(args.stage2_checkpoint, "unet_lora")
    if os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
        print(f"✅ LoRA loaded from {lora_path}")
    
    # 3. 注入 Scaler (V8.7 核心)
    scaler_path = os.path.join(args.stage2_checkpoint, "scaler_final.pth")
    if not os.path.exists(scaler_path): 
        scaler_path = os.path.join(args.stage2_checkpoint, "scaler.pth")
        
    if os.path.exists(scaler_path):
        print(f"✅ Loading Learnable Scaler from {scaler_path}")
        scaler_module = ControlNetScaler(num_scales=13)
        scaler_module.load_state_dict(torch.load(scaler_path, map_location=device))
        scaler_module.to(device, dtype=torch.float16)
        
        # Monkey Patch
        original_forward = pipe.controlnet.forward
        def patched_forward(*args, **kwargs):
            down, mid = original_forward(*args, **kwargs)
            return scaler_module(down, mid)
        pipe.controlnet.forward = patched_forward
        print("🔧 Scaler injected.")
        
        # 打印权重预览
        scales = scaler_module.scales.detach().cpu().numpy()
        print(f"📊 Scales: {scales}")
    else:
        print("⚠️ Scaler not found, using default identity.")

    # 4. 初始化辅助工具
    ink_gen = InkWashMaskGenerator(width=512, height=512)
    tokenizer = BertTokenizer.from_pretrained(args.taiyi_model_path, subfolder="tokenizer")

    # === 测试集 (Batch Inference) ===
    POEMS_BATCH = [
        "明月松间照，清泉石上流。",
        "大漠孤烟直，长河落日圆。", 
        "两个黄鹂鸣翠柳，一行白鹭上青天。",
        "忽如一夜春风来，千树万树梨花开。",
        "白日依山尽，黄河入海流。",
        "枯藤老树昏鸦，小桥流水人家。",
        "野旷天低树，江清月近人。",
        "采菊东篱下，悠然见南山。"
    ] 

    print(f"\n🎨 Starting Batch Inference for {len(POEMS_BATCH)} poems...")

    for i, poem in enumerate(tqdm(POEMS_BATCH)):
        poem_clean = poem[:10].replace("，", "_").replace("。", "").strip()
        save_dir = os.path.join(args.output_base, f"{i+1:02d}_{poem_clean}")
        os.makedirs(save_dir, exist_ok=True)

        # Step 1: Layout
        layout_list = greedy_decode_poem_layout(layout_model, tokenizer, poem, device=device)
        if not layout_list: continue
        layout = np.array(layout_list)

        # Step 2: Visualize
        draw_layout(layout, f"Poem: {poem}", os.path.join(save_dir, "01_layout.png"))

        # Step 3: Ink Mask
        mask_img = ink_gen.convert_boxes_to_mask(layout)
        mask_img.save(os.path.join(save_dir, "02_ink_mask.png"))

        # Step 4: Attention Injection
        attn_proc = PoemInkAttentionProcessor(
            dynamic_layout=layout, 
            tokenizer=pipe.tokenizer, 
            prompt=poem, 
            device=device,
            scale=8.0
        )
        pipe.unet.set_attn_processor(attn_proc)

        # Step 5: Generation
        # [修改点] 去掉 "写意水墨画..." 等后缀，完全使用原始诗句
        prompt = poem 
        neg_prompt = "低质量，模糊，色彩斑驳，边框，水印，现代建筑"
        
        generator = torch.Generator(device=device).manual_seed(2024)
        
        final_image = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=mask_img, # 单流输入
            num_inference_steps=35,
            controlnet_conditioning_scale=1.0, # 强度由 scaler 决定
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        
        final_image.save(os.path.join(save_dir, "03_final_painting.png"))

    print(f"✅ All Done. Results saved to {args.output_base}")

if __name__ == "__main__":
    main()
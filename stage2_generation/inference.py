import sys
import os
import argparse
import torch
import numpy as np
import yaml  # [New] 用于读取配置文件
from PIL import Image, ImageDraw, ImageFont
from transformers import BertTokenizer
from pathlib import Path

# === 路径配置 ===
current_file_path = os.path.abspath(__file__)
stage2_root = os.path.dirname(os.path.dirname(current_file_path)) # stage2_generation/
project_root = os.path.dirname(stage2_root) # layout2paint/

if project_root not in sys.path:
    sys.path.append(project_root)
if stage2_root not in sys.path:
    sys.path.append(stage2_root)

# 导入 Stage 1 模型
try:
    from models.poem2layout import Poem2LayoutGenerator
    from models.kg import PoetryKnowledgeGraph
    from models.location import LocationSignalGenerator
except ImportError:
    print("[Error] 无法导入 Stage 1 模型，请检查 models/ 目录是否完整。")
    sys.exit(1)

# 导入 Stage 2 工具
try:
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
    from stage2_generation.utils.prompt_kg import KGPromptGenerator
except ImportError:
    print("[Error] 无法导入 Stage 2 工具，请检查 utils/ 目录。")
    sys.exit(1)

# 导入适配 SD 1.x / Taiyi 的 Pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

class EndToEndGenerator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading End-to-End System on {self.device}...")

        # 1. 初始化 KG 和 Tokenizer
        print("[Init] Loading Knowledge Graph & Tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        self.pkg = PoetryKnowledgeGraph()
        self.location_gen = LocationSignalGenerator(grid_size=8)
        
        # Stage 2 工具 (Taiyi 建议 512x512)
        self.width = 512
        self.height = 512
        self.ink_gen = InkWashMaskGenerator(width=self.width, height=self.height) 
        self.prompt_gen = KGPromptGenerator()

        # 类别 ID 映射 (用于可视化)
        self.id_to_name = {
            0: "padding", 1: "sky", 2: "mountain", 3: "water", 4: "people", 
            5: "tree", 6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
        }

        # 2. 加载 Stage 1 模型
        print(f"[Stage 1] Loading Layout Generator from {args.stage1_checkpoint}...")
        
        # [MODIFIED] 修正配置文件路径并强制匹配 RL 训练参数
        config_path = "/home/610-sty/layout2paint2/configs/default.yaml"
        if not os.path.exists(config_path):
            print(f"[Warning] Config file not found at {config_path}, using hardcoded RL parameters.")
            model_cfg = {
                'hidden_size': 768,
                'bb_size': 128,
                'decoder_layers': 6, # 与 rl_best_reward.pth 匹配
                'decoder_heads': 8,
                'latent_dim': 64     # 与 rl_best_reward.pth 匹配
            }
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            model_cfg = config.get('model', {})
        
        # 提取关键参数
        hidden_size = model_cfg.get('hidden_size', 768)
        bb_size = model_cfg.get('bb_size', 128)
        decoder_layers = model_cfg.get('decoder_layers', 6) 
        decoder_heads = model_cfg.get('decoder_heads', 8)
        latent_dim = model_cfg.get('latent_dim', 64)

        print(f"[Stage 1] Init Model with layers={decoder_layers}, heads={decoder_heads}, latent={latent_dim}")
        
        self.layout_model = Poem2LayoutGenerator(
            bert_path=args.bert_path,
            num_classes=9,
            hidden_size=hidden_size, 
            bb_size=bb_size,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            latent_dim=latent_dim,
            dropout=0.0 # 推理模式关闭 Dropout
        )
        
        # [MODIFIED] 健壮的权重加载逻辑
        checkpoint = torch.load(args.stage1_checkpoint, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            self.layout_model.load_state_dict(state_dict, strict=True)
            print("✅ Stage 1 Model weights loaded successfully.")
        except RuntimeError as e:
            print(f"⚠️ Strict loading failed: {e}")
            self.layout_model.load_state_dict(state_dict, strict=False)
            
        self.layout_model.to(self.device)
        self.layout_model.eval()

        # 3. 加载 Stage 2 模型 (ControlNet + LoRA)
        print(f"[Stage 2] Loading Dual ControlNets from {args.stage2_checkpoint}...")
        
        controlnet_s = ControlNetModel.from_pretrained(
            os.path.join(args.stage2_checkpoint, "controlnet_structure"),
            torch_dtype=torch.float16
        )
        controlnet_t = ControlNetModel.from_pretrained(
            os.path.join(args.stage2_checkpoint, "controlnet_style"),
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.base_model_path, 
            controlnet=[controlnet_s, controlnet_t], 
            torch_dtype=torch.float16,
            safety_checker=None 
        )

        lora_path = os.path.join(args.stage2_checkpoint, "unet_lora")
        if os.path.exists(lora_path):
            print(f"✅ 检测到 LoRA 权重: {lora_path}")
            try:
                self.pipe.load_lora_weights(lora_path)
                print("✨ LoRA 加载成功！")
            except Exception as e:
                print(f"⚠️ LoRA 加载失败: {e}")

        self.pipe.to(self.device)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def preprocess_poem(self, poem):
        encoded = self.tokenizer(poem, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        raw_ids = self.pkg.expand_ids_with_quantity(
            (torch.nonzero(self.pkg.extract_visual_feature_vector(poem) > 0).squeeze(1) + 2).tolist(), poem
        )
        if not raw_ids: raw_ids = [0]
        if len(raw_ids) > 30: raw_ids = raw_ids[:30]
        
        kg_class_ids = torch.tensor([raw_ids], dtype=torch.long).to(self.device) 
        spatial_matrix = self.pkg.extract_spatial_matrix(poem).unsqueeze(0).to(self.device) 

        # [MODIFIED] 增加判空逻辑解决 TypeError
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32)
        grid_list = []
        for i, cid in enumerate(raw_ids):
            if cid == 0:
                grid_list.append(torch.zeros((8, 8)))
                continue
            idx = int(cid) - 2
            idx = max(0, min(idx, 8))
            
            res = self.location_gen.infer_stateful_signal(
                i, spatial_matrix[0, idx], spatial_matrix[0, :, idx], current_occupancy, mode='max' 
            )
            
            if res is None or res[0] is None:
                print(f"[Warning] No valid location for element {i}. Using empty grid.")
                signal = torch.zeros((8, 8))
            else:
                signal, current_occupancy = res
            
            grid_list.append(signal)
            
        location_grids = torch.stack(grid_list).unsqueeze(0).to(self.device) 
        return input_ids, attention_mask, kg_class_ids, spatial_matrix, location_grids

    def draw_layout_visualization(self, layout_data):
        img = Image.new('RGB', (self.width, self.height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        for item in layout_data:
            cid, cx, cy, w, h = item
            x1, y1 = (cx - w/2) * self.width, (cy - h/2) * self.height
            x2, y2 = (cx + w/2) * self.width, (cy + h/2) * self.height
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1 + 2, y1), self.id_to_name.get(int(cid), "unknown"), fill="red", font=font)
        return img

    def infer(self, poem, seed=42):
        print(f"\nProcessing Poem: {poem}")
        save_dir = Path(self.args.output_dir) / f"{poem[:10]}_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        with torch.no_grad():
            inputs = self.preprocess_poem(poem)
            input_ids, attn_mask, class_ids, spat_mat, loc_grids = inputs
            pred_boxes, _ = self.layout_model.forward_rl(
                input_ids, attn_mask, class_ids, None, spat_mat, loc_grids, sample=False
            )
            boxes = pred_boxes[0].cpu().numpy().tolist()
            class_ids_list = class_ids[0].cpu().numpy().tolist()
            layout_data = [[cid] + box for cid, box in zip(class_ids_list, boxes) if cid > 0]
                
        layout_viz = self.draw_layout_visualization(layout_data)
        layout_viz.save(save_dir / "02_layout_viz.png")
        ink_mask = self.ink_gen.convert_boxes_to_mask(layout_data)
        ink_mask.save(save_dir / "03_ink_mask.png")
        
        prompt = self.prompt_gen.generate_prompt(poem)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(
            prompt, image=[ink_mask, ink_mask], num_inference_steps=30,  
            controlnet_conditioning_scale=[0.9, 0.6], guidance_scale=7.5, generator=generator
        ).images[0]
        image.save(save_dir / "04_final_painting.png")
        print(f"✅ 完成！结果路径: {save_dir}")

def main():
    parser = argparse.ArgumentParser()
    # [MODIFIED] 确保路径匹配 610-sty 环境
    parser.add_argument("--bert_path", type=str, default="/home/610-sty/huggingface/bert-base-chinese")
    parser.add_argument("--stage1_checkpoint", type=str, default="/home/610-sty/layout2paint/outputs/train1/rl_best_reward.pth")
    parser.add_argument("--stage2_checkpoint", type=str, default="/home/610-sty/layout2paint/outputs/taiyi_ink_controlnet_v2")
    parser.add_argument("--base_model_path", type=str, default="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    parser.add_argument("--output_dir", type=str, default="inference_results")
    parser.add_argument("--poem", type=str, default="明月松间照，清泉石上流。", help="输入诗句")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    engine = EndToEndGenerator(args)
    engine.infer(args.poem, args.seed)

if __name__ == "__main__":
    main()
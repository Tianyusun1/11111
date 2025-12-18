import os
import torch
import argparse
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler

# 导入项目内部组件
from models.poem2layout import Poem2LayoutGenerator
from inference.greedy_decode import greedy_decode_poem_layout
from stage2_generation.utils.ink_mask import InkWashMaskGenerator
from data.visualize import draw_layout

# =============================================================
# 创新架构：跨模态交叉注意力态势锚定处理器 (Gestalt Attention Processor)
# =============================================================
class PoemInkAttentionProcessor:
    """
    底层架构创新：通过干预 Cross-Attention 层实现数学级语义绑定。
    [V7.0 更新]：支持态势能参数偏移，使注意力跟随墨迹扩散方向。
    """
    def __init__(self, dynamic_layout, tokenizer, prompt, device, scale=7.0):
        self.layout = dynamic_layout  # 8维张量 [N, 8]
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

        # 执行态势锚定
        res = int(np.sqrt(sequence_length))
        h, w = res, res
        tokens = self.tokenizer.encode(self.prompt)
        
        for item in self.layout:
            cls_id = int(item[0])
            keyword = self.class_to_keyword.get(cls_id, None)
            if not keyword: continue
            
            # 提取 8 维中的参数
            cx, cy, bw, bh = item[1:5]
            bx, by = item[5:7] # 态势偏移参数
            
            keyword_token_ids = self.tokenizer.encode(keyword, add_special_tokens=False)
            token_indices = [i for i, t in enumerate(tokens) if t in keyword_token_ids]
            
            if not token_indices: continue

            # [架构创新点]：根据态势能计算非对称注意力 Mask
            # 相比普通方框，这里加入了 (bx, by) 的中心偏移
            x_c, y_c = (cx + bx * 0.1) * w, (cy + by * 0.1) * h
            x1, y1 = int(x_c - (bw/2)*w), int(y_c - (bh/2)*h)
            x2, y2 = int(x_c + (bw/2)*w), int(y_c + (bh/2)*h)
            
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            for idx in token_indices:
                if idx >= attention_probs.shape[-1]: continue
                # 注意力场增强
                mask = torch.zeros((h, w), device=self.device)
                mask[y1:y2, x1:x2] = self.scale
                mask_flat = mask.flatten()
                
                # 乘性增强，强制模型在渲染该区域时‘满脑子都是这个意象’
                attention_probs[:, :, idx] += mask_flat * attention_probs[:, :, idx]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# =============================================================
# 推理主逻辑适配
# =============================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化模型
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    layout_model = Poem2LayoutGenerator(
        bert_path=config['model']['bert_path'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        bb_size=config['model']['bb_size'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads'],
        latent_dim=config['model'].get('latent_dim', 64)
    ).to(device).eval()
    
    # 加载权重
    ckpt = torch.load(args.layout_ckpt, map_location=device)
    layout_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    
    # 加载太乙管线
    controlnet_seg = ControlNetModel.from_pretrained(args.controlnet_seg_path, torch_dtype=torch.float16)
    controlnet_t = ControlNetModel.from_pretrained(args.controlnet_t_path, torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.taiyi_model_path,
        controlnet=[controlnet_seg, controlnet_t],
        torch_dtype=torch.float16
    ).to(device)
    pipe.load_lora_weights(args.lora_path)
    
    ink_gen = InkWashMaskGenerator(width=512, height=512)

    # 模拟 50 首测试集
    POEMS_50 = ["大漠孤烟直，长河落日圆。", "两个黄鹂鸣翠柳，一行白鹭上青天。"] # 示例

    for i, poem in enumerate(tqdm(POEMS_50)):
        poem_clean = poem[:12].replace("，", "_").replace("。", "").strip()
        save_dir = os.path.join(args.output_base, f"{i+1:02d}_{poem_clean}")
        os.makedirs(save_dir, exist_ok=True)

        # 1. 生成 8 维动态布局
        # 注意：此处 layout 现在是 [N, 8]
        layout = greedy_decode_poem_layout(layout_model, BertTokenizer.from_pretrained(config['model']['bert_path']), poem, device=device)
        
        # 2. 可视化基础框 (取前 5 维：cls, cx, cy, w, h)
        draw_layout(layout[:, :5], f"Poem: {poem}", os.path.join(save_dir, "01_layout.png"))

        # 3. 转换为势能场 Mask (V7.0 ink_mask 支持 8 维输入)
        mask_img = ink_gen.convert_boxes_to_mask(layout)
        mask_img.save(os.path.join(save_dir, "02_potential_field.png"))

        # 4. 架构注入：注入支持态势偏移的处理器
        attn_proc = PoemInkAttentionProcessor(
            dynamic_layout=layout, 
            tokenizer=pipe.tokenizer, 
            prompt=poem, 
            device=device,
            scale=8.0 # 强绑定系数
        )
        pipe.unet.set_attn_processor(attn_proc)

        # 5. 双流协同生成
        final_image = pipe(
            prompt=f"{poem}，写意水墨画，中国画风格，杰作",
            image=[mask_img, mask_img],
            num_inference_steps=35,
            controlnet_conditioning_scale=[1.2, 0.8] # 结构流略强于风格流以保证位置
        ).images[0]
        
        final_image.save(os.path.join(save_dir, "03_final_painting.png"))

if __name__ == "__main__":
    main()
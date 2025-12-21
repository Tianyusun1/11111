# File: stage2_generation/scripts/prepare_data_taiyi.py (V8.0 Compatible)

import sys
import os
import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# === 路径设置 ===
current_file_path = os.path.abspath(__file__)
stage2_root = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(stage2_root)
if project_root not in sys.path: sys.path.insert(0, project_root)
if stage2_root not in sys.path: sys.path.append(stage2_root)

# 导入已经修改为 V7.0 版的工具
try:
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
except ImportError:
    print("❌ 无法导入 InkWashMaskGenerator，请检查路径。")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Taiyi V7.0: 准备彩色势能场训练数据")
    default_xlsx = "/home/610-sty/layout2paint/dataset/6800poems.xlsx"
    default_img_dir = "/home/610-sty/layout2paint/dataset/6800"
    default_lbl_dir = "/home/610-sty/layout2paint/dataset/6800/JPEGImages-pre_new_txt"
    
    parser.add_argument("--xlsx_path", type=str, default=default_xlsx)
    parser.add_argument("--images_dir", type=str, default=default_img_dir)
    parser.add_argument("--labels_dir", type=str, default=default_lbl_dir)
    parser.add_argument("--output_dir", type=str, default="./taiyi_dataset_v7_color")
    parser.add_argument("--resolution", type=int, default=512) 
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "conditioning_images"), exist_ok=True)
    
    # 初始化 V7.0 发生器
    ink_generator = InkWashMaskGenerator(width=args.resolution, height=args.resolution)
    df = pd.read_excel(args.xlsx_path)
    
    metadata_entries = []
    
    # 基础风格词
    style_suffix = "，水墨画，中国画，写意，杰作，高分辨率"

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_img_name = str(row['image']).strip()
            poem = str(row['poem']).strip()
            img_stem = Path(raw_img_name).stem
            
            src_img_path = os.path.join(args.images_dir, raw_img_name)
            if not os.path.exists(src_img_path): continue
            
            label_path = os.path.join(args.labels_dir, f"{img_stem}.txt")
            if not os.path.exists(label_path): continue

            # 3. 读取 Box
            # [Fix] 兼容 5 维 (GT) 或 9 维 (V8.0 Inference) 数据
            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5: 
                        # 截取前 5 维几何信息 (cls, cx, cy, w, h)
                        # 即使 parts 包含后 4 维态势，这里也只取前 5 维，
                        # 让 InkWashMaskGenerator 按照类别默认值处理（或后续自行修改 Generator 接口以利用真实态势）
                        boxes.append(list(map(float, parts[:5])))
            
            if not boxes: continue

            # 4. 【核心修改】生成彩色势能场 Mask
            # 虽然输入是 5 维，但 V7.0 的 convert_boxes_to_mask 会根据类别自动补全态势能
            # 从而生成具有“洇散感”和“彩色语义”的训练底稿
            cond_img = ink_generator.convert_boxes_to_mask(boxes)
            
            # 关键：确保保存为 RGB 模式，不能转 'L'
            cond_img_name = f"{img_stem}_ink_v7.png"
            cond_img.save(os.path.join(args.output_dir, "conditioning_images", cond_img_name))
            
            # 5. 处理原图 (Resize 到 512)
            target_img = Image.open(src_img_path).convert("RGB")
            target_img = target_img.resize((args.resolution, args.resolution), Image.BICUBIC)
            target_img_name = f"{img_stem}.jpg"
            target_img.save(os.path.join(args.output_dir, "images", target_img_name))

            # 6. 构造中文 Prompt
            chinese_prompt = f"{poem}{style_suffix}"

            metadata_entries.append({
                "image": f"images/{target_img_name}",
                "conditioning_image": f"conditioning_images/{cond_img_name}",
                "text": chinese_prompt
            })
            
        except Exception as e:
            print(f"Error processing {img_stem}: {e}")
            continue

    # 保存 JSONL
    with open(os.path.join(args.output_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"✨ V7.0 彩色势能场数据集准备完成！输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()
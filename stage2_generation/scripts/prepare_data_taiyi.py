import sys
import os
import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# === 路径设置 ===
current_file_path = os.path.abspath(__file__)
stage2_root = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(stage2_root)
if project_root not in sys.path: sys.path.insert(0, project_root)
if stage2_root not in sys.path: sys.path.append(stage2_root)

# 导入工具
try:
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
    # 注意：太乙不需要 KGPromptGenerator 做翻译了，但我们可以留着它做"风格后缀"
    # 或者直接手动拼接风格词
except ImportError:
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Taiyi: 准备中文训练数据")
    default_xlsx = os.path.join(project_root, "/home/610-sty/layout2paint/dataset/6800poems.xlsx")
    default_img_dir = os.path.join(project_root, "/home/610-sty/layout2paint/dataset/6800")
    default_lbl_dir = os.path.join(project_root, "/home/610-sty/layout2paint/dataset/6800/JPEGImages-pre_new_txt")
    
    parser.add_argument("--xlsx_path", type=str, default=default_xlsx)
    parser.add_argument("--images_dir", type=str, default=default_img_dir)
    parser.add_argument("--labels_dir", type=str, default=default_lbl_dir)
    parser.add_argument("--output_dir", type=str, default="./taiyi_dataset_v1")
    # 太乙基于 SD 1.5，最佳分辨率是 512x512
    parser.add_argument("--resolution", type=int, default=512) 
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "conditioning_images"), exist_ok=True)
    
    ink_generator = InkWashMaskGenerator(width=args.resolution, height=args.resolution)
    df = pd.read_excel(args.xlsx_path)
    
    metadata_entries = []
    
    # 基础风格词 (中文!)
    style_suffix = "，水墨画，中国画，写意，杰作，高分辨率"

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 1. 基础信息
            raw_img_name = str(row['image']).strip()
            poem = str(row['poem']).strip() # 直接用中文诗！
            img_stem = Path(raw_img_name).stem
            
            # 2. 路径检查 (略，同之前逻辑)
            src_img_path = os.path.join(args.images_dir, raw_img_name)
            if not os.path.exists(src_img_path):
                 # ... (兼容代码同前) ...
                 continue
            
            label_path = os.path.join(args.labels_dir, f"{img_stem}.txt")
            if not os.path.exists(label_path): continue

            # 3. 读取 Box
            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5: boxes.append(list(map(float, parts)))
            if not boxes: continue

            # 4. 生成水墨 Mask (分辨率 512)
            cond_img = ink_generator.convert_boxes_to_mask(boxes)
            cond_img_name = f"{img_stem}_ink.png"
            cond_img.save(os.path.join(args.output_dir, "conditioning_images", cond_img_name))
            
            # 5. 处理原图 (Resize 到 512)
            target_img = Image.open(src_img_path).convert("RGB")
            target_img = target_img.resize((args.resolution, args.resolution), Image.BICUBIC)
            target_img_name = f"{img_stem}.jpg"
            target_img.save(os.path.join(args.output_dir, "images", target_img_name))

            # 6. 【关键】构造中文 Prompt
            # 直接使用: 诗句 + 风格词
            chinese_prompt = f"{poem}{style_suffix}"

            metadata_entries.append({
                "image": f"images/{target_img_name}",
                "conditioning_image": f"conditioning_images/{cond_img_name}",
                "text": chinese_prompt  # 这里是纯中文！
            })
            
        except Exception:
            continue

    # 保存
    with open(os.path.join(args.output_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print("太乙中文数据集准备完成！")

if __name__ == "__main__":
    main()
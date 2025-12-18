import os
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
from typing import List, Tuple, Dict

# 类别定义 (必须与 dataset.py 中的 ID 2-10 保持一致)
CLASS_COLORS = {
    2: "red",      # mountain
    3: "blue",     # water
    4: "green",    # people
    5: "brown",    # tree
    6: "yellow",   # building
    7: "cyan",     # bridge
    8: "magenta",  # flower
    9: "orange",   # bird
    10: "lime"     # animal
}

CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}

def draw_layout(layout_seq: List[Tuple], poem: str, output_path: str, img_size: Tuple[int, int] = (512, 512)):
    """(保持原有布局绘制函数不变)"""
    try:
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        W, H = img_size
        
        try:
            font = ImageFont.truetype("arial.ttf", 10) 
        except IOError:
            font = ImageFont.load_default()
            
        for item in layout_seq:
            if len(item) != 5: continue
            cls_id, cx, cy, w, h = item
            cls_id = int(cls_id)
            
            xmin = int((cx - w / 2) * W)
            ymin = int((cy - h / 2) * H)
            xmax = int((cx + w / 2) * W)
            ymax = int((cy + h / 2) * H)
            
            color = CLASS_COLORS.get(cls_id, "black")
            cls_name = CLASS_NAMES.get(cls_id, "Unknown")
            
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
            label_text = f"{cls_name} ({cls_id})"
            text_y = ymin - 12 if ymin > 12 else ymin + 2
            draw.text((xmin + 2, text_y), label_text, fill=color, font=font)
        
        draw.text((10, 10), f"Input: {poem}", fill="black", font=font)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        
    except ImportError:
        print("[Warning]: PIL/Pillow not found.")
    except Exception as e:
        print(f"[Error]: {e}")

# -------------------------------------------------------------
# [NEW] 综合热力图绘制函数 (多层叠加)
# -------------------------------------------------------------
def draw_integrated_heatmap(
    layers: List[Tuple[np.ndarray, int]], 
    poem: str, 
    output_path: str, 
    img_size: Tuple[int, int] = (512, 512)
):
    """
    将多个物体的热力图叠加到一张图中。
    
    Args:
        layers: List of (grid_8x8, class_id)。
        poem: 诗句内容（作为标题）。
        output_path: 输出路径。
    """
    try:
        W, H = img_size
        # 初始化黑色画布 (Float类型以便累加)
        # 形状: (H, W, 3)
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        
        for grid, cls_id in layers:
            # 1. 归一化 Grid
            grid = np.asarray(grid, dtype=np.float32)
            if grid.max() > 0:
                grid = grid / grid.max()
            
            # 2. 上采样 Grid 到图像尺寸 (使用最近邻插值保持块状感)
            # 先转为 PIL Image 进行 Resize，再转回 Numpy
            pil_grid = Image.fromarray(grid)
            pil_grid = pil_grid.resize(img_size, resample=Image.NEAREST)
            grid_large = np.array(pil_grid, dtype=np.float32) # (H, W)
            
            # 3. 获取类别颜色
            color_name = CLASS_COLORS.get(cls_id, "white")
            rgb = ImageColor.getrgb(color_name) # (R, G, B)
            
            # 4. 叠加颜色：Canvas += Heatmap_Value * Color_RGB
            # 这样重叠区域的亮度会增加，且颜色会混合
            for c in range(3):
                canvas[:, :, c] += grid_large * rgb[c]

        # 5. 裁剪值到 [0, 255] 并转为 uint8
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        img = Image.fromarray(canvas, mode='RGB')
        
        # 6. 绘制网格线和标题
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        # 灰色网格线
        grid_color = (80, 80, 80)
        grid_w, grid_h = 8, 8 # 假设是 8x8 Grid
        for i in range(1, grid_w):
            x = int(i * W / grid_w)
            draw.line([(x, 0), (x, H)], fill=grid_color, width=1)
        for j in range(1, grid_h):
            y = int(j * H / grid_h)
            draw.line([(0, y), (W, y)], fill=grid_color, width=1)

        # 标题 (带黑色描边的白色文字，确保在任何背景下可见)
        text = f"Integrated Heatmap: {poem[:15]}..."
        x, y = 10, 10
        # 描边
        for offset in [(-1,-1),(1,-1),(-1,1),(1,1)]: 
            draw.text((x+offset[0], y+offset[1]), text, font=font, fill="black")
        draw.text((x, y), text, font=font, fill="white")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"  -> Integrated heatmap saved: {output_path}")

    except Exception as e:
        print(f"[Error] Failed to draw integrated heatmap: {e}")
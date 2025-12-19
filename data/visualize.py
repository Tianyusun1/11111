# File: data/visualize.py (V7.2: Fixed for 8-dim Gestalt Compatibility)

import os
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
from typing import List, Tuple, Dict, Union

# =============================================================
# [核心同步] 类别颜色定义 (必须与 ink_mask.py 保持完全一致)
# =============================================================
CLASS_COLORS = {
    2: "red",      # mountain (山): 对应 ink_mask 的 (255, 0, 0)
    3: "blue",     # water (水): 对应 ink_mask 的 (0, 0, 255)
    4: "cyan",     # people (人): 对应 ink_mask 的 (0, 255, 255)
    5: "green",    # tree (树): 对应 ink_mask 的 (0, 255, 0)
    6: "yellow",   # building (建筑): 对应 ink_mask 的 (255, 255, 0)
    7: "magenta",  # bridge (桥): 对应 ink_mask 的 (255, 0, 255)
    8: "purple",   # flower (花): 对应 ink_mask 的 (128, 0, 128)
    9: "orange",   # bird (鸟): 对应 ink_mask 的 (255, 165, 0)
    10: "brown"    # animal (动物): 对应 ink_mask 的 (165, 42, 42)
}

CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}

def draw_layout(layout_seq: List[Tuple], poem: str, output_path: str, img_size: Tuple[int, int] = (512, 512)):
    """
    绘制带有颜色标注的布局草图，用于验证语义绑定是否正确。
    [Fixed] 兼容 V7.0 模型的 8 维态势参数输出。
    """
    try:
        # 创建黑色背景，让彩色标注框更显眼 (更“骚”)
        img = Image.new('RGB', img_size, (20, 20, 20)) 
        draw = ImageDraw.Draw(img)
        W, H = img_size
        
        try:
            # 尝试加载中文字体以支持诗句显示，若无则回退
            font = ImageFont.truetype("simhei.ttf", 14) 
        except IOError:
            font = ImageFont.load_default()
            
        for item in layout_seq:
            # [CRITICAL FIX] 兼容性解包
            # item 可能是 5 维 (旧版) 或 9 维 (V7.0 含态势参数)
            # 我们只取前 5 个值用于绘图 (cls, cx, cy, w, h)
            if len(item) < 5: 
                continue
                
            cls_id, cx, cy, w, h = item[:5]
            cls_id = int(cls_id)
            
            # 转换归一化坐标为像素坐标
            xmin = int((cx - w / 2) * W)
            ymin = int((cy - h / 2) * H)
            xmax = int((cx + w / 2) * W)
            ymax = int((cy + h / 2) * H)
            
            color = CLASS_COLORS.get(cls_id, "white")
            cls_name = CLASS_NAMES.get(cls_id, "Unknown")
            
            # 绘制实线框
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            
            # 绘制类别标签
            label_text = f"{cls_name}"
            # 简单的边界检查防止文字画出界
            text_x = max(0, min(W - 50, xmin + 2))
            text_y = max(0, ymin - 16 if ymin > 20 else ymin + 2)
            
            draw.text((text_x, text_y), label_text, fill=color, font=font)
        
        # 绘制底部诗句标题
        draw.text((10, H - 25), f"Poem: {poem}", fill="white", font=font)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        
    except Exception as e:
        print(f"[Error in draw_layout]: {e}")

# -------------------------------------------------------------
# [同步修改] 综合热力图绘制函数 (确保颜色与 Mask 对齐)
# -------------------------------------------------------------
def draw_integrated_heatmap(
    layers: List[Tuple[np.ndarray, int]], 
    poem: str, 
    output_path: str, 
    img_size: Tuple[int, int] = (512, 512)
):
    """
    将多个意象的热力图按照语义颜色叠加。
    """
    try:
        W, H = img_size
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        
        for grid, cls_id in layers:
            grid = np.asarray(grid, dtype=np.float32)
            if grid.max() > 0:
                grid = grid / grid.max()
            
            # 上采样至图像尺寸
            pil_grid = Image.fromarray(grid)
            pil_grid = pil_grid.resize(img_size, resample=Image.BILINEAR) # 稍微平滑一点
            grid_large = np.array(pil_grid, dtype=np.float32)
            
            # 获取对应的语义颜色
            color_name = CLASS_COLORS.get(cls_id, "white")
            rgb = ImageColor.getrgb(color_name) 
            
            # 颜色叠加
            for c in range(3):
                canvas[:, :, c] += grid_large * rgb[c]

        # 裁剪并量化
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        img = Image.fromarray(canvas, mode='RGB')
        
        # 绘制网格线增强“骚气”感
        draw = ImageDraw.Draw(img)
        grid_color = (60, 60, 60)
        num_cells = 8
        for i in range(1, num_cells):
            x = int(i * W / num_cells)
            draw.line([(x, 0), (x, H)], fill=grid_color, width=1)
            y = int(i * H / num_cells)
            draw.line([(0, y), (W, y)], fill=grid_color, width=1)

        # 标题绘制
        text = f"Semantic Heatmap: {poem[:12]}"
        draw.text((10, 10), text, fill="white")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"  -> Semantic heatmap saved: {output_path}")

    except Exception as e:
        print(f"[Error in draw_integrated_heatmap]: {e}")
# File: stage2_generation/utils/ink_mask.py (V8.0: Fully Data-Driven Rendering)

import numpy as np
from PIL import Image, ImageFilter
import torch
from typing import List, Union, Dict, Any
import math

class InkWashMaskGenerator:
    """
    [V8.0 核心组件] 数据驱动的语义势能场生成器 (Data-Driven Semantic Potential Field Generator)
    
    不同于 V7 版本查静态表，V8 版本完全利用模型预测的 8 维数据 
    (cx, cy, w, h, bias_x, bias_y, rot, flow) 来动态构建各向异性高斯场。
    这意味着模型学会了如何通过参数直接控制墨迹的‘形’与‘势’。
    """
    
    # [V8.0] 仅保留语义颜色定义，物理属性完全由模型预测决定
    CLASS_COLORS = {
        2: (255, 0, 0),   # 山 (Mountain) - Red
        3: (0, 0, 255),   # 水 (Water) - Blue
        4: (0, 255, 255), # 人 (People) - Cyan
        5: (0, 255, 0),   # 树 (Tree) - Green
        6: (255, 255, 0), # 建筑 (Building) - Yellow
        7: (255, 0, 255), # 桥 (Bridge) - Magenta
        8: (128, 0, 128), # 花 (Flower) - Purple
        9: (255, 165, 0), # 鸟 (Bird) - Orange
        10: (165, 42, 42) # 兽 (Animal) - Brown
    }
    
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        # 预计算坐标网格 [H, W]
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)
        # indexing='ij' 使得 y_grid 对应行(H), x_grid 对应列(W)
        self.y_grid, self.x_grid = np.meshgrid(y, x, indexing='ij')
        
    def _generate_rotated_gaussian(self, box: np.ndarray) -> np.ndarray:
        """
        [核心算法] 生成带有 [位移、缩放、旋转、洇散] 属性的动态各向异性高斯场。
        """
        # box 结构: [cls, cx, cy, w, h, bx, by, rot, flow]
        # 注意: 预测值 bx, by, rot, flow 均来自 Tanh 输出，范围 [-1, 1]
        
        class_id = int(box[0])
        cx, cy, w, h = box[1], box[2], box[3], box[4]
        
        # 提取 V8.0 预测的态势参数
        if len(box) >= 9:
            bx, by = box[5], box[6]
            rot = box[7]  # [-1, 1] 对应 [-90, 90] 度
            flow = box[8] # [-1, 1] 代表枯湿程度
        else:
            # 兼容旧版本数据 (Fallback)
            bx, by, rot, flow = 0.0, 0.0, 0.0, 0.0

        # 1. 物理中心偏移 (Bias Shift) 
        # 模型预测重心偏离几何中心的程度 (0.15 是系数，避免飞出画框)
        # bx, by 为负代表向上/向左，为正代表向下/向右
        center_x = (cx + bx * 0.15) * self.width
        center_y = (cy + by * 0.15) * self.height
        
        # 2. 尺度与洇散 (Scale & Flow)
        # 将 flow [-1, 1] 映射到 [0.5, 2.0] 的缩放系数
        # flow > 0 (湿笔): 墨迹晕开，Sigma 变大
        # flow < 0 (枯笔): 墨迹收敛，Sigma 变小
        flow_scale = 1.0 + flow * 0.8 
        
        # 基础 Sigma 基于物体宽高，除以 2.5 使得物体边缘大概在 2.5 个标准差处 (约 98% 能量覆盖)
        sigma_x = (w * self.width / 2.5) * flow_scale
        sigma_y = (h * self.height / 2.5) * flow_scale
        
        # 3. 旋转矩阵 (Rotation) 
        # 将 rot [-1, 1] 映射为弧度 [-PI/2, PI/2]
        theta = rot * (math.pi / 2.0)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # 构建旋转后的坐标系 (Coordinate Transformation)
        # 我们需要计算网格点 (x, y) 在旋转坐标系 (x', y') 下的位置
        # X' = (x - xc) * cos + (y - yc) * sin
        # Y' = -(x - xc) * sin + (y - yc) * cos
        dx = self.x_grid - center_x
        dy = self.y_grid - center_y
        
        dx_rot = dx * cos_t + dy * sin_t
        dy_rot = -dx * sin_t + dy * cos_t
        
        # 4. 计算高斯场公式
        # G(x,y) = exp( - (x'^2 / 2sx^2 + y'^2 / 2sy^2) )
        # 加入 1e-6 防止除零
        exponent = - (dx_rot**2 / (2 * sigma_x**2 + 1e-6) + 
                      dy_rot**2 / (2 * sigma_y**2 + 1e-6))
        
        # 为了性能，极小值截断
        field = np.exp(np.clip(exponent, -20, 0))
        
        # 5. 注入 "写意" 噪声 (Neural Texture Injection)
        # 噪声强度也可以由 flow 控制：flow 越大(越湿)，边缘越毛糙；flow 越小(越枯)，边缘越锐利
        noise_level = 0.05 + max(0, float(flow)) * 0.08
        noise = np.random.normal(0, noise_level, field.shape).astype(np.float32)
        
        # 乘性噪声 + 可以在一定程度上模拟宣纸纹理
        field = field + field * noise
        
        # 6. 上色
        color = self.CLASS_COLORS.get(class_id, (128, 128, 128))
        colored_field = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i in range(3):
            # 将归一化的高斯场映射到 RGB 强度
            colored_field[:, :, i] = field * (color[i] / 255.0)
            
        return colored_field

    def convert_boxes_to_mask(self, boxes: Union[List[List[float]], torch.Tensor]) -> Image.Image:
        """
        将 8-dim 布局序列渲染为 RGB 势能图 Mask，用于 ControlNet 输入。
        """
        full_canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            
        # 按面积 (w*h) 排序，小的覆盖大的 (Painter's Algorithm)
        # index 3=w, 4=h
        # 过滤无效框 (w或h <= 0)
        valid_boxes = [b for b in boxes if len(b) >= 5 and b[3] > 0 and b[4] > 0]
        sorted_boxes = sorted(valid_boxes, key=lambda b: b[3]*b[4], reverse=True)
            
        for box in sorted_boxes:
            class_id = int(box[0])
            # 跳过 PAD(0), EOS(1) 以及不支持的类别
            if class_id < 2 or class_id > 10: 
                continue
            
            # 生成单个意象的动态场
            object_field = self._generate_rotated_gaussian(box)
            
            # 积墨融合逻辑 (Alpha Blend Simulation)
            # 使用 alpha blending 模拟水墨的层叠感，而不是简单的 max
            # alpha 近似于 field 的亮度 (最大通道)
            alpha = np.max(object_field, axis=2, keepdims=True)
            # 增强不透明度，使核心区域更实
            alpha = np.clip(alpha * 1.5, 0, 1) 
            
            # Canvas = Canvas * (1 - Alpha) + Object * Alpha
            full_canvas = full_canvas * (1 - alpha) + object_field * alpha

        # 归一化并转为 uint8 图像
        full_canvas = np.clip(full_canvas * 255.0, 0, 255).astype(np.uint8)
        mask_img = Image.fromarray(full_canvas, mode='RGB')
        
        # 最后的平滑，模拟宣纸晕染效果
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))
        
        return mask_img
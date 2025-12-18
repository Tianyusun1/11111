import numpy as np
from PIL import Image
import torch
from typing import List, Union, Dict, Any, Tuple

class InkWashMaskGenerator:
    """
    创新点核心组件：水墨晕染软布局生成器 (Ink Diffusion Soft-Layout Generator)
    
    功能：
    将生硬的 Bounding Box 转换为模拟宣纸水墨渗透效果的高斯热力图。
    中心墨色浓（Pixel值高），边缘墨色淡（Pixel值低），重叠区域墨色加深（积墨）。
    
    [已修正]
    1. 修正了 CLASS_PROPERTIES 中的 Diffusion Factor (DF) 值，使其与 sigma 公式一致。
    2. 修正了归一化上限 (MAX_INK_INTENSITY)，保留积墨带来的动态范围。
    """
    
    # 类别属性定义: [初始权重 (Weight), 扩散因子 (DF)]
    # 公式: sigma = size / DF。因此 DF 越大 -> sigma 越小 -> 墨迹越实/锐利。
    # 我们根据 DF=4.0 (最实) 到 DF=1.5 (最虚) 来定义类别。
    CLASS_PROPERTIES: Dict[int, Tuple[float, float]] = {
        # [已修正] DF 值：山和建筑应最实（DF最高）
        2: [1.3, 4.0],  # mountain (山): 重墨，最实（DF=4.0）
        3: [1.0, 2.5],  # water (水): 中墨，流动感（DF=2.5，比山虚）
        4: [0.7, 2.8],  # people (人): 轻墨，中虚（DF=2.8）
        5: [1.0, 3.2],  # tree (树): 中墨，较实（DF=3.2）
        6: [1.2, 4.0],  # building (建筑): 重墨，最实（DF=4.0）
        7: [1.2, 3.8],  # bridge (桥): 重墨，较实（DF=3.8）
        # [已修正] DF 值：花应最虚（DF最低）
        8: [0.6, 1.5],  # flower (花): 最轻墨，最虚化（DF=1.5）
        9: [0.8, 2.0],  # bird (鸟): 轻墨，虚化（DF=2.0）
        10: [0.9, 3.0]  # animal (动物): 中轻墨（DF=3.0）
    }
    
    # 默认属性用于未定义的类别或 Padding
    DEFAULT_PROPERTIES = [0.8, 3.0]

    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        # 预计算网格坐标，加速推理
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)
        self.y_grid = y[:, np.newaxis]
        self.x_grid = x
        
    def _get_ink_properties(self, class_id: int) -> Tuple[float, float]:
        """根据类别 ID 获取权重和扩散因子"""
        return self.CLASS_PROPERTIES.get(class_id, self.DEFAULT_PROPERTIES)

    def _generate_gaussian_blob(self, cx: float, cy: float, w: float, h: float, class_id: int) -> Tuple[np.ndarray, float]:
        """生成单个物体的高斯墨块，并返回其权重"""
        # 获取类别感知属性
        weight, diffusion_factor = self._get_ink_properties(class_id)
        
        # 还原绝对坐标
        x_center = cx * self.width
        y_center = cy * self.height
        box_w = w * self.width
        box_h = h * self.height
        
        # [修正逻辑] sigma = size / DF。DF 越大，墨迹越实。
        sigma_x = box_w / diffusion_factor 
        sigma_y = box_h / diffusion_factor
        
        # 防止 sigma 过小导致数值错误
        sigma_x = max(sigma_x, 1.0)
        sigma_y = max(sigma_y, 1.0)
        
        # 二维高斯公式
        blob = np.exp(-((self.x_grid - x_center)**2 / (2 * sigma_x**2) + 
                         (self.y_grid - y_center)**2 / (2 * sigma_y**2)))
        
        return blob, weight

    def convert_boxes_to_mask(self, boxes: Union[List[List[float]], torch.Tensor]) -> Image.Image:
        """
        输入:
             boxes: shape [N, 5]，格式为 (class_id, cx, cy, w, h)
        输出:
             PIL.Image (L mode, 灰度图)
        """
        # 初始化空白画布 (浮点型用于累加)
        canvas = np.zeros((self.height, self.width), dtype=np.float32)
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            
        for box in boxes:
            if len(box) != 5: continue 

            class_id = int(box[0])
            if class_id == 0 or class_id not in self.CLASS_PROPERTIES: continue
            
            cx, cy, w, h = box[1:]
            
            if w <= 0 or h <= 0: continue
            
            # 1. 生成类别感知的高斯墨块和权重
            blob, weight = self._generate_gaussian_blob(cx, cy, w, h, class_id)
            
            # 2. 墨色累加 (Ink Accumulation / 积墨效果)
            canvas += blob * weight

        # 3. 归一化与量化
        # [修正] 将固定 MAX_INK_INTENSITY 提高，以容纳积墨导致的多次累加，保留层次感
        # 经验值 3.0 允许 2-3 个高权重物体区域有层次地叠加
        MAX_INK_INTENSITY = 3.0 
        
        # 先对累加后的值进行归一化到 [0.0, 1.0]
        canvas = np.clip(canvas / MAX_INK_INTENSITY, 0.0, 1.0)
        
        # 将 0.0-1.0 的浮点图转为 0-255 的灰度图
        canvas = (canvas * 255.0).astype(np.uint8)
        
        return Image.fromarray(canvas, mode='L')
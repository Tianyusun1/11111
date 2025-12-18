import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from typing import List, Union, Dict, Any, Tuple

class InkWashMaskGenerator:
    """
    创新点核心组件：彩色语义水墨掩膜生成器 (Color Semantic Ink Mask Generator)
    
    功能：
    1. 将生硬的 Bounding Box 转换为带有类别颜色编码的彩色 Mask。
    2. 颜色与标注框颜色完全一致，实现语义强绑定。
    3. 保留高斯晕染效果（高斯核与类别相关），使 Mask 既有语义信息又有水墨韵味。
    """
    
    # 类别属性定义: [RGB颜色, 扩散因子(DF)]
    # 颜色定义与 visualize.py 保持高度一致，确保强绑定
    CLASS_PROPERTIES: Dict[int, Tuple[Tuple[int, int, int], float]] = {
        2: [(255, 0, 0), 4.0],      # mountain (山): 红色，最实
        3: [(0, 0, 255), 2.5],      # water (水): 蓝色，流动感
        4: [(0, 255, 255), 2.8],    # people (人): 青色
        5: [(0, 255, 0), 3.2],      # tree (树): 绿色
        6: [(255, 255, 0), 4.0],    # building (建筑): 黄色
        7: [(255, 0, 255), 3.8],    # bridge (桥): 品红
        8: [(128, 0, 128), 1.5],    # flower (花): 紫色，最虚
        9: [(255, 165, 0), 2.0],    # bird (鸟): 橙色
        10: [(165, 42, 42), 3.0]    # animal (动物): 棕色
    }
    
    # 默认颜色（灰色）及扩散因子
    DEFAULT_PROPERTIES = [(128, 128, 128), 3.0]

    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        # 预计算坐标网格
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)
        self.y_grid = y[:, np.newaxis]
        self.x_grid = x
        
    def _get_properties(self, class_id: int) -> Tuple[Tuple[int, int, int], float]:
        """根据类别 ID 获取对应的颜色和扩散因子"""
        return self.CLASS_PROPERTIES.get(class_id, self.DEFAULT_PROPERTIES)

    def _generate_colored_blob(self, cx: float, cy: float, w: float, h: float, class_id: int) -> np.ndarray:
        """生成单个物体的彩色高斯墨块 (shape: [H, W, 3])"""
        color, diffusion_factor = self._get_properties(class_id)
        
        # 还原坐标
        x_center = cx * self.width
        y_center = cy * self.height
        box_w = w * self.width
        box_h = h * self.height
        
        # 计算高斯分布的 sigma (控制虚实)
        sigma_x = max(box_w / diffusion_factor, 1.0)
        sigma_y = max(box_h / diffusion_factor, 1.0)
        
        # 生成单通道高斯响应 [0, 1]
        blob_alpha = np.exp(-((self.x_grid - x_center)**2 / (2 * sigma_x**2) + 
                             (self.y_grid - y_center)**2 / (2 * sigma_y**2)))
        
        # 将单通道扩展为 RGB 通道
        # 结果为：颜色权重 * 高斯透明度
        colored_blob = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i in range(3):
            colored_blob[:, :, i] = (color[i] / 255.0) * blob_alpha
            
        return colored_blob

    def convert_boxes_to_mask(self, boxes: Union[List[List[float]], torch.Tensor]) -> Image.Image:
        """
        核心修改：将 Boxes 转换为彩色语义 Mask
        输入: boxes: shape [N, 5]，格式为 (class_id, cx, cy, w, h)
        输出: PIL.Image (RGB 模式)
        """
        # 初始化空白画布 (RGB 浮点型)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            
        # 按照顺序绘制（如果需要重叠，后面的会覆盖前面的或进行融合）
        for box in boxes:
            if len(box) != 5: continue 
            class_id = int(box[0])
            # 过滤无效类别或背景
            if class_id < 2 or class_id > 10: continue
            
            cx, cy, w, h = box[1:]
            if w <= 0 or h <= 0: continue
            
            # 1. 生成带颜色的高斯墨块
            colored_blob = self._generate_colored_blob(cx, cy, w, h, class_id)
            
            # 2. 颜色混合：采用最大值融合 (Maximum Composition)，保留最鲜艳的颜色语义
            canvas = np.maximum(canvas, colored_blob)

        # 3. 归一化与转换
        # 限制范围 [0, 1] 并转为 uint8
        canvas = np.clip(canvas, 0.0, 1.0)
        canvas_uint8 = (canvas * 255.0).astype(np.uint8)
        
        mask_img = Image.fromarray(canvas_uint8, mode='RGB')
        
        # 4. 模拟水墨洇散：轻微模糊边缘，但保留颜色语义
        # 这一步能让 Mask 看起来不那么生硬，更符合水墨 ControlNet 的输入特征
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
        
        return mask_img
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from typing import List, Union, Dict, Any, Tuple

class InkWashMaskGenerator:
    """
    创新架构组件：语义势能扩散场生成器 (Semantic Potential Field Generator)
    
    功能：
    1. 突破死板方框：利用矩形框作为‘势能源’，模拟水墨非均匀扩散效果。
    2. 类别感知纹理：针对不同意象（山、水、树）注入不同的高频扰动（Perlin-like Noise）。
    3. 语义强度映射：通过能量梯度为后续的 Attention Locking 提供物理级引导。
    """
    
    # 类别属性定义: [RGB颜色, 扩散方向权重(x, y), 洇散强度]
    # 扩散方向权重用于打破对称高斯，形成‘势’
    CLASS_PROPERTIES: Dict[int, Dict[str, Any]] = {
        2: {"color": (255, 0, 0), "bias": (1.0, 1.8), "flow": 4.5},   # 山: 纵向势能强，向上挺拔
        3: {"color": (0, 0, 255), "bias": (2.5, 0.8), "flow": 3.0},   # 水: 横向势能强，延展感
        4: {"color": (0, 255, 255), "bias": (1.0, 1.0), "flow": 2.5}, # 人: 集中势能
        5: {"color": (0, 255, 0), "bias": (1.2, 1.5), "flow": 3.5},   # 树: 错落势能
        6: {"color": (255, 255, 0), "bias": (1.1, 1.1), "flow": 4.0}, # 建筑: 稳定势能
        7: {"color": (255, 0, 255), "bias": (2.0, 1.0), "flow": 3.8}, # 桥: 跨越势能
        8: {"color": (128, 0, 128), "bias": (1.5, 1.5), "flow": 2.0}, # 花: 弥散势能
        9: {"color": (255, 165, 0), "bias": (1.8, 1.2), "flow": 2.2}, # 鸟: 灵动势能
        10: {"color": (165, 42, 42), "bias": (1.3, 1.3), "flow": 3.0} # 动物: 聚散势能
    }
    
    DEFAULT_PROPS = {"color": (128, 128, 128), "bias": (1.0, 1.0), "flow": 3.0}

    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        # 预计算坐标网格
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)
        self.y_grid, self.x_grid = np.meshgrid(y, x, indexing='ij')
        
    def _generate_potential_field(self, box: np.ndarray) -> np.ndarray:
        """核心创新：基于势能扩散算法生成非均匀墨迹场"""
        class_id = int(box[0])
        cx, cy, w, h = box[1:]
        props = self.CLASS_PROPERTIES.get(class_id, self.DEFAULT_PROPS)
        
        # 还原物理中心和尺度
        x_c, y_c = cx * self.width, cy * self.height
        sigma_x = (w * self.width) / props["flow"] * props["bias"][0]
        sigma_y = (h * self.height) / props["flow"] * props["bias"][1]

        # 1. 基础非对称势能场计算
        # 利用指数衰减模拟墨汁从中心向四周的物理渗透
        exponent = -((self.x_grid - x_c)**2 / (2 * sigma_x**2) + 
                     (self.y_grid - y_c)**2 / (2 * sigma_y**2))
        field = np.exp(exponent)

        # 2. 注入‘写意’扰动 (Neural Texture Injection)
        # 模拟宣纸纤维导致的随机洇散边缘，打破矩形框的生硬感
        noise = np.random.normal(0, 0.05, field.shape).astype(np.float32)
        field = field + field * noise # 扰动随能量强度缩放
        
        # 3. 颜色映射与能量加权
        color = props["color"]
        colored_field = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i in range(3):
            colored_field[:, :, i] = (color[i] / 255.0) * field
            
        return colored_field

    def convert_boxes_to_mask(self, boxes: Union[List[List[float]], torch.Tensor]) -> Image.Image:
        """
        架构级重构：将 Boxes 序列转化为多维语义势能图
        """
        # 初始化画布
        full_canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            
        # 按面积从小到大排序，确保细微意象（人、鸟）的势能不被大面积意象完全覆盖
        # 这体现了构图中的‘计白当黑’逻辑
        sorted_boxes = sorted(boxes, key=lambda b: b[3]*b[4], reverse=True)
            
        for box in sorted_boxes:
            if len(box) != 5: continue 
            class_id = int(box[0])
            if class_id < 2 or class_id > 10: continue
            
            # 生成单个意象的势能场
            object_field = self._generate_potential_field(box)
            
            # 采用积墨融合逻辑 (Alpha-like Accumulation)
            # 这比简单的 np.maximum 更能体现水墨重叠时的色泽变化
            full_canvas = np.where(object_field > 0.01, 
                                   full_canvas * (1 - object_field) + object_field, 
                                   full_canvas)

        # 归一化并转为图像
        full_canvas = np.clip(full_canvas, 0.0, 1.0)
        canvas_uint8 = (full_canvas * 255.0).astype(np.uint8)
        
        mask_img = Image.fromarray(canvas_uint8, mode='RGB')
        
        # 最后的洇散微调：利用中值滤波保留边缘特征的同时增加平滑感
        mask_img = mask_img.filter(ImageFilter.MedianFilter(size=3))
        
        return mask_img
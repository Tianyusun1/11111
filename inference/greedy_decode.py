# File: inference/greedy_decode.py (V6.1: Integrated Heatmap Visualization)

import torch
import numpy as np
import random
import os

# --- Import KG & Location ---
try:
    from models.kg import PoetryKnowledgeGraph
except ImportError:
    print("[Error] Could not import PoetryKnowledgeGraph. Make sure models/kg.py is accessible.")
    PoetryKnowledgeGraph = None

# 导入位置生成器
try:
    from models.location import LocationSignalGenerator
except ImportError:
    print("[Error] Could not import LocationSignalGenerator. Make sure models/location.py is accessible.")
    LocationSignalGenerator = None

# [NEW] 导入综合可视化工具 (需确保 data/visualize.py 已包含 draw_integrated_heatmap)
try:
    from data.visualize import draw_integrated_heatmap
except ImportError:
    draw_integrated_heatmap = None
# -----------------

# 定义所有类别的形状先验 (Shape Priors)
CLASS_SHAPE_PRIORS = {
    2: {'min_w': 0.20, 'min_h': 0.20}, # Mountain
    3: {'min_w': 0.20, 'min_h': 0.10}, # Water
    4: {'min_w': 0.02, 'min_h': 0.08, 'max_w': 0.15}, # People
    5: {'min_w': 0.05, 'min_h': 0.15}, # Tree
    6: {'min_w': 0.05, 'min_h': 0.05}, # Building
    7: {'min_w': 0.15, 'max_h': 0.08}, # Bridge
    8: {'min_w': 0.03, 'min_h': 0.03}, # Flower
    9: {'min_w': 0.03, 'min_h': 0.03}, # Bird
    10: {'min_w': 0.04, 'min_h': 0.04} # Animal
}

# 类别名称映射
CLASS_ID_TO_NAME = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}

def greedy_decode_poem_layout(model, tokenizer, poem: str, max_elements=None, device='cuda', mode='greedy', top_k=3):
    """
    Query-Based decoding with Location Guidance & CVAE Diversity.
    [Updated] Supports Integrated Gaussian Heatmap Visualization.
    """
    if PoetryKnowledgeGraph is None:
        return []

    model.eval()
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    
    # 1. 实例化组件
    pkg = PoetryKnowledgeGraph()
    
    if LocationSignalGenerator is not None:
        location_gen = LocationSignalGenerator(grid_size=8)
    else:
        location_gen = None
    
    # 2. KG 提取内容
    kg_vector = pkg.extract_visual_feature_vector(poem)
    existing_indices = torch.nonzero(torch.tensor(kg_vector) > 0).squeeze(1)
    raw_class_ids = (existing_indices + 2).tolist()
    
    if not raw_class_ids:
        return []
        
    # KG 数量扩展
    if hasattr(pkg, 'expand_ids_with_quantity'):
        kg_class_ids = pkg.expand_ids_with_quantity(raw_class_ids, poem)
    else:
        kg_class_ids = raw_class_ids
        
    if max_elements:
        kg_class_ids = kg_class_ids[:max_elements]
        
    # 3. 准备模型输入 Tensor
    kg_class_tensor = torch.tensor([kg_class_ids], dtype=torch.long).to(device)
    
    # 构建空间矩阵
    try:
        kg_spatial_matrix_np = pkg.extract_spatial_matrix(poem, obj_ids=kg_class_ids)
    except TypeError:
        kg_spatial_matrix_np = pkg.extract_spatial_matrix(poem)
        
    kg_spatial_matrix = torch.tensor(kg_spatial_matrix_np, dtype=torch.long).unsqueeze(0).to(device)
    
    # === 生成位置引导信号 (Location Guidance) ===
    location_grids_tensor = None
    
    if location_gen is not None:
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32)
        grids_list = []
        
        # [NEW] 用于收集每一层热力图，以便最后合并绘制
        heatmap_layers = [] # 格式: List[(numpy_grid, class_id)]
        
        for i, cls_id in enumerate(kg_class_ids):
            # 获取空间关系行/列
            if i < kg_spatial_matrix_np.shape[0]:
                row = kg_spatial_matrix_np[i]
                col = kg_spatial_matrix_np[:, i]
            else:
                row = np.zeros(len(kg_class_ids))
                col = np.zeros(len(kg_class_ids))
            
            # 推理位置信号
            signal, current_occupancy = location_gen.infer_stateful_signal(
                i, row, col, current_occupancy, 
                mode=mode, top_k=top_k 
            )
            
            # Jitter (抖动)
            if mode == 'sample' and random.random() < 0.6:
                shift_val = random.randint(-2, 2)
                signal = torch.roll(signal, shifts=shift_val, dims=1)
            
            grids_list.append(signal)
            
            # [NEW] 收集当前层的数据用于可视化
            if draw_integrated_heatmap is not None:
                # 注意：必须转为 cpu numpy，否则后续绘图库会报错
                heatmap_layers.append((signal.cpu().numpy(), int(cls_id)))
            
        location_grids_tensor = torch.stack(grids_list).unsqueeze(0).to(device)

        # === [NEW] 循环结束后，绘制综合热力图 ===
        if draw_integrated_heatmap is not None and len(heatmap_layers) > 0:
            # 文件名处理：取诗句前10个字作为文件名，避免特殊字符
            safe_poem_name = "".join(x for x in poem if x.isalnum())[:10]
            
            save_dir = os.path.join("outputs", "heatmaps")
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存为 integrated_诗句名.png
            save_path = os.path.join(save_dir, f"integrated_{safe_poem_name}.png")
            
            # 调用可视化函数进行叠加绘制
            draw_integrated_heatmap(heatmap_layers, poem, save_path)
    # ==========================================
    
    # 4. 文本编码与前向传播
    inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=64)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    padding_mask = torch.zeros(kg_class_tensor.shape, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        _, _, pred_boxes, _ = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            kg_class_ids=kg_class_tensor, 
            padding_mask=padding_mask, 
            kg_spatial_matrix=kg_spatial_matrix, 
            location_grids=location_grids_tensor
        )
        
    # 5. 格式化输出
    layout = []
    boxes_flat = pred_boxes[0].cpu().tolist()
    
    for cls_id, box in zip(kg_class_ids, boxes_flat):
        cid = int(cls_id)
        cx, cy, w, h = box
        
        # 形状约束
        w = max(w, 0.02)
        h = max(h, 0.02)
        if cid in CLASS_SHAPE_PRIORS:
            prior = CLASS_SHAPE_PRIORS[cid]
            if 'min_w' in prior: w = max(w, prior['min_w'])
            if 'min_h' in prior: h = max(h, prior['min_h'])
            if 'max_w' in prior: w = min(w, prior['max_w'])
            if 'max_h' in prior: h = min(h, prior['max_h'])
        
        layout.append((float(cls_id), cx, cy, w, h))
        
    return layout
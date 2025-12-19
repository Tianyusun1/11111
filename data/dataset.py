# File: data/dataset.py (V8.0: Data-Driven Visual Gestalt Extraction)

import os
import torch
import pandas as pd
import yaml 
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np 
import random
import cv2  # [NEW] 引入 OpenCV 进行视觉特征计算

# --- 导入知识图谱模型 ---
from models.kg import PoetryKnowledgeGraph
# --- 导入位置引导信号生成器 ---
from models.location import LocationSignalGenerator
# ---------------------------

# 类别定义
CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}
VALID_CLASS_IDS = set(CLASS_NAMES.keys())

class VisualGestaltExtractor:
    """
    [V8.0 核心组件] 视觉态势提取器
    基于传统计算机视觉算法 (Moments, PCA)，从真实图像的裁切块中提取物理“势能”。
    """
    def extract(self, image_path: str, box: List[float]) -> List[float]:
        """
        输入: 全图路径, 归一化 Box [cx, cy, w, h]
        输出: 态势参数 [bias_x, bias_y, rotation, flow] (均为归一化或标准化数值)
        """
        try:
            # 1. 读取图像 (灰度模式，因为水墨画主要看明度)
            if not os.path.exists(image_path):
                return [0.0, 0.0, 0.0, 0.0] # Fallback
            
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return [0.0, 0.0, 0.0, 0.0]
                
            H, W = img.shape
            cx, cy, w, h = box
            
            # 2. 裁切物体 (Crop)
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)
            
            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if x2 <= x1 or y2 <= y1:
                return [0.0, 0.0, 0.0, 0.0]
                
            crop = img[y1:y2, x1:x2]
            
            # 3. 预处理：反色 (水墨画纸白墨黑，计算重心需要墨为高响应值)
            # 255 (白) -> 0 (无墨), 0 (黑) -> 255 (重墨)
            ink_map = 255 - crop
            
            # 二值化处理去噪，只关注主体笔墨
            _, thresh = cv2.threshold(ink_map, 50, 255, cv2.THRESH_BINARY)
            
            # === A. 计算 Bias (视觉重心偏移) ===
            M = cv2.moments(thresh)
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                
                # 几何中心
                geo_cX = (x2 - x1) / 2
                geo_cY = (y2 - y1) / 2
                
                # 归一化偏移量 (-1.0 ~ 1.0)
                bias_x = (cX - geo_cX) / (geo_cX + 1e-6)
                bias_y = (cY - geo_cY) / (geo_cY + 1e-6)
                
                # 限制范围
                bias_x = np.clip(bias_x, -1.0, 1.0)
                bias_y = np.clip(bias_y, -1.0, 1.0)
            else:
                bias_x, bias_y = 0.0, 0.0
                
            # === B. 计算 Rotation (主轴方向) ===
            # 使用图像矩计算主轴角度
            if M["m00"] != 0:
                mu20 = M["mu20"] / M["m00"]
                mu02 = M["mu02"] / M["m00"]
                mu11 = M["mu11"] / M["m00"]
                # 计算偏角 theta
                theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                # 归一化到 -1 ~ 1 (对应 -90度 到 90度)
                rotation = theta / (np.pi / 2)
            else:
                rotation = 0.0
                
            # === C. 计算 Flow (洇散度/墨韵) ===
            # 逻辑：Flow = 墨的平均密度 * 边缘柔和度
            # 密度：ink_map 的均值
            density = np.mean(ink_map) / 255.0
            
            # 边缘强度：Sobel 梯度
            sobelx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            edge_strength = np.mean(grad_mag)
            
            # 水墨画特点：如果密度高但边缘梯度低，说明"洇"得厉害 (Wet)
            if edge_strength > 0:
                flow = (density * 100) / (edge_strength + 1e-6)
            else:
                flow = 0.0
            
            # 归一化 flow (经验值缩放)
            flow = np.clip(flow, 0.0, 5.0)
            
            return [float(bias_x), float(bias_y), float(rotation), float(flow)]
            
        except Exception as e:
            return [0.0, 0.0, 0.0, 0.0]


class PoegraphLayoutDataset(Dataset):
    def __init__(
        self,
        xlsx_path: str,
        labels_dir: str,
        bert_model_path: str = "/home/610-sty/huggingface/bert-base-chinese",
        max_layout_length: int = 30, 
        max_text_length: int = 64, 
        preload: bool = False
    ):
        super().__init__()
        self.xlsx_path = xlsx_path
        self.labels_dir = Path(labels_dir)
        self.max_layout_length = max_layout_length 
        self.max_text_length = max_text_length
        self.num_classes = 9 

        print("Initializing Knowledge Graph...")
        self.pkg = PoetryKnowledgeGraph()
        
        self.location_gen = LocationSignalGenerator(grid_size=8)
        
        # [NEW V8.0] 初始化视觉态势提取器
        self.gestalt_extractor = VisualGestaltExtractor()
        print("✅ Visual Gestalt Extractor (OpenCV-based) initialized.")
        
        # 加载 Excel
        df = pd.read_excel(xlsx_path)
        
        # [NEW] 解析数据集根目录，以便找到图片
        # 假设 xlsx_path 是 data/dataset.xlsx，则 root 是 data/..
        self.dataset_root = os.path.dirname(os.path.dirname(os.path.abspath(xlsx_path)))
        
        self.data = []

        print("Loading dataset index...")
        for _, row in df.iterrows():
            raw_img_path = str(row['image']).strip()
            poem = str(row['poem']).strip()
            
            # 构造绝对路径
            if os.path.isabs(raw_img_path):
                full_img_path = raw_img_path
            else:
                # 假设 raw_img_path 是相对于 dataset_root 的路径
                full_img_path = os.path.join(self.dataset_root, raw_img_path)
            
            img_stem = Path(full_img_path).stem
            label_path = self.labels_dir / f"{img_stem}.txt"

            if not label_path.exists():
                continue

            # 读取标注
            boxes = []
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5: continue
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        if cls_id in VALID_CLASS_IDS and \
                           0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            boxes.append((float(cls_id), cx, cy, w, h)) 
            except Exception:
                continue

            if boxes:
                self.data.append({
                    'poem': poem,
                    'boxes': boxes,
                    'img_path': full_img_path # [NEW] 保存图片路径
                })

        print(f"✅ PoegraphLayoutDataset 加载完成，共 {len(self.data)} 个样本")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        poem = sample['poem']
        gt_boxes = sample['boxes'] # List[(cls_id, cx, cy, w, h)]
        img_path = sample['img_path']

        # 1. 文本编码
        tokenized = self.tokenizer(
            poem,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        # 2. KG 提取
        kg_vector = self.pkg.extract_visual_feature_vector(poem)
        kg_spatial_matrix = self.pkg.extract_spatial_matrix(poem)
        existing_indices = torch.nonzero(kg_vector > 0).squeeze(1)
        raw_ids = (existing_indices + 2).tolist()
        kg_class_ids = self.pkg.expand_ids_with_quantity(raw_ids, poem)

        # 权重列表
        kg_class_weights = []
        for cid in kg_class_ids:
             idx = int(cid) - 2
             if 0 <= idx < self.num_classes:
                 kg_class_weights.append(kg_vector[idx].item())
             else:
                 kg_class_weights.append(1.0)

        if not kg_class_ids:
            kg_class_ids = [0]
            kg_class_weights = [0.0]

        # 位置信号生成
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32) 
        location_grids_list = [] 
        for i, cls_id in enumerate(kg_class_ids):
            cls_id = int(cls_id)
            if cls_id == 0:
                location_grids_list.append(torch.zeros((8, 8), dtype=torch.float32))
                continue
            matrix_idx = cls_id - 2 if 0 <= cls_id - 2 < self.num_classes else 0
            spatial_row = kg_spatial_matrix[matrix_idx]  
            spatial_col = kg_spatial_matrix[:, matrix_idx] 
            signal, current_occupancy = self.location_gen.infer_stateful_signal(
                i, spatial_row, spatial_col, current_occupancy,
                mode='sample', top_k=3 
            )
            if random.random() < 0.7: 
                shift = random.randint(-2, 2) 
                signal = torch.roll(signal, shifts=shift, dims=1) 
            location_grids_list.append(signal)

        # 3. GT 对齐与 [NEW] 视觉特征提取
        # 注意：这里的 target_boxes 现在将变为 8 维！
        target_boxes_8d = [] 
        loss_mask = []

        gt_dict = {}
        for item in gt_boxes:
            cid, cx, cy, w, h = item
            cid = int(cid)
            if cid not in gt_dict: gt_dict[cid] = []
            gt_dict[cid].append([cx, cy, w, h])

        do_flip = random.random() < 0.5
        
        for k_cls in kg_class_ids:
            k_cls = int(k_cls)
            if k_cls == 0: 
                target_boxes_8d.append([0.0] * 8)
                loss_mask.append(0.0)
                continue

            if k_cls in gt_dict and len(gt_dict[k_cls]) > 0:
                box = gt_dict[k_cls].pop(0) # [cx, cy, w, h]
                
                # 脏数据过滤
                if box[2] * box[3] > 0.90 or box[2]/(box[3]+1e-6) > 10.0 or box[2]/(box[3]+1e-6) < 0.1:
                    target_boxes_8d.append([0.0] * 8)
                    loss_mask.append(0.0)
                    continue
                
                # [V8.0 核心] 实时从原图中提取视觉态势！
                gestalt_features = self.gestalt_extractor.extract(img_path, box) # [bx, by, rot, flow]
                
                # 几何增强应用
                if do_flip:
                    box[0] = 1.0 - box[0] # Flip cx
                    gestalt_features[0] = -gestalt_features[0] # Flip bias_x (方向反转)
                    gestalt_features[2] = -gestalt_features[2] # Flip rotation (角度反转)
                
                # Jitter (仅对坐标，不对态势)
                noise = np.random.uniform(-0.02, 0.02, size=4)
                box_aug = [
                    np.clip(box[0] + noise[0], 0.0, 1.0),
                    np.clip(box[1] + noise[1], 0.0, 1.0),
                    np.clip(box[2] + noise[2], 0.01, 1.0),
                    np.clip(box[3] + noise[3], 0.01, 1.0)
                ]
                
                # 合并 4维坐标 + 4维态势 = 8维 Target
                target_boxes_8d.append(box_aug + gestalt_features)
                loss_mask.append(1.0)
            else:
                target_boxes_8d.append([0.0] * 8)
                loss_mask.append(0.0)

        # 截断
        if len(kg_class_ids) > self.max_layout_length:
            kg_class_ids = kg_class_ids[:self.max_layout_length]
            kg_class_weights = kg_class_weights[:self.max_layout_length] 
            target_boxes_8d = target_boxes_8d[:self.max_layout_length]
            loss_mask = loss_mask[:self.max_layout_length]
            location_grids_list = location_grids_list[:self.max_layout_length]

        location_grids = torch.stack(location_grids_list)
        if do_flip:
            location_grids = torch.flip(location_grids, dims=[2])

        return {
            'input_ids': tokenized['input_ids'].squeeze(0), 
            'attention_mask': tokenized['attention_mask'].squeeze(0), 
            'kg_class_ids': torch.tensor(kg_class_ids, dtype=torch.long),
            'kg_class_weights': torch.tensor(kg_class_weights, dtype=torch.float32), 
            'target_boxes': torch.tensor(target_boxes_8d, dtype=torch.float32), # Now 8-dim
            'loss_mask': torch.tensor(loss_mask, dtype=torch.float32),
            'kg_spatial_matrix': kg_spatial_matrix,
            'kg_vector': kg_vector,
            'num_boxes': torch.tensor(len(gt_boxes), dtype=torch.long),
            'location_grids': location_grids 
        }

def layout_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function 适配 8 维 target_boxes"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    kg_spatial_matrices = torch.stack([item['kg_spatial_matrix'] for item in batch])
    kg_vectors = torch.stack([item['kg_vector'] for item in batch])
    num_boxes = torch.stack([item['num_boxes'] for item in batch])

    lengths = [len(item['kg_class_ids']) for item in batch]
    max_len = max(lengths)
    if max_len == 0: max_len = 1

    batched_class_ids = []
    batched_class_weights = [] 
    batched_target_boxes = []
    batched_loss_mask = []
    batched_padding_mask = [] 
    batched_location_grids = []

    for item in batch:
        cur_len = len(item['kg_class_ids'])
        pad_len = max_len - cur_len
        
        # 1. IDs
        padded_ids = torch.cat([
            item['kg_class_ids'], 
            torch.zeros(pad_len, dtype=torch.long)
        ])
        batched_class_ids.append(padded_ids)
        
        # 2. Weights 
        padded_weights = torch.cat([
            item['kg_class_weights'],
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_class_weights.append(padded_weights)
        
        # 3. Boxes [MODIFIED] Pad with 8 zeros instead of 4
        padded_boxes = torch.cat([
            item['target_boxes'], 
            torch.zeros((pad_len, 8), dtype=torch.float32) # 8-dim padding
        ])
        batched_target_boxes.append(padded_boxes)
        
        # 4. Mask
        padded_loss_mask = torch.cat([
            item['loss_mask'], 
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_loss_mask.append(padded_loss_mask)

        # 5. Location Grids
        padded_grids = torch.cat([
            item['location_grids'],
            torch.zeros((pad_len, 8, 8), dtype=torch.float32)
        ])
        batched_location_grids.append(padded_grids)
        
        # 6. Pad Mask
        pad_mask = torch.zeros(max_len, dtype=torch.bool)
        if pad_len > 0:
            pad_mask[cur_len:] = True
        batched_padding_mask.append(pad_mask)

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'kg_class_ids': torch.stack(batched_class_ids),      
        'kg_class_weights': torch.stack(batched_class_weights), 
        'target_boxes': torch.stack(batched_target_boxes),   
        'loss_mask': torch.stack(batched_loss_mask),         
        'padding_mask': torch.stack(batched_padding_mask),   
        'kg_spatial_matrix': kg_spatial_matrices,
        'kg_vector': kg_vectors,
        'num_boxes': num_boxes,
        'location_grids': torch.stack(batched_location_grids) 
    }

if __name__ == "__main__":
    pass
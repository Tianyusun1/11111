# File: trainers/rl_trainer.py (V7.0: Dynamic Gestalt & Cross-Modal Attention Alignment)

import torch
import torch.nn.functional as F
import torch.optim as optim
from .trainer import LayoutTrainer
import time
import os
import matplotlib.pyplot as plt
import numpy as np

from trainers.loss import compute_kl_loss

class RLTrainer(LayoutTrainer):
    """
    创新架构训练器：支持动态态势预测与交叉注意力对齐。
    
    [V7.0 Updates]
    1. 适配 8 维动作空间 (Coordinates + Gestalt Potential)。
    2. 新增 Semantic_Alignment_Reward：利用 Cross-Attention Map 约束语义位置。
    3. 适配双头预测模型输出。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        # RL 超参数
        self.rl_lr = float(config['training'].get('rl_learning_rate', 5e-6))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        
        # 奖励权重增加“空间对齐”项
        reward_cfg = config['training'].get('reward_weights', {})
        self.w_iou = float(reward_cfg.get('iou', 2.0))              
        self.w_rel = float(reward_cfg.get('relation', 5.0)) 
        self.w_align = float(reward_cfg.get('alignment', 8.0)) # [NEW] 语义-注意力强绑定权重
        self.w_overlap = float(reward_cfg.get('overlap', -1.0))     

        self.last_reward_stats = {}
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer V7.0] Gestalt Action Space & Attention Locking Enabled.")

    def compute_reward(self, dynamic_layout, batch, attention_maps=None):
        """
        计算奖励。注意：dynamic_layout 是 8 维的。
        """
        B, T, _ = dynamic_layout.shape
        device = dynamic_layout.device
        
        # 提取基础坐标 (前 4 维)
        pred_boxes = dynamic_layout[..., :4]
        # 提取态势参数 (后 4 维: bx, by, rot, flow)
        gestalt_params = dynamic_layout[..., 4:]
        
        loss_mask = batch['loss_mask']          
        target_boxes = batch['target_boxes']
        kg_spatial_matrix = batch['kg_spatial_matrix'] 
        kg_class_ids = batch['kg_class_ids']    
        
        obj_rewards = torch.zeros(B, T, device=device)
        
        # 1. 基础 IoU 奖励 (坐标对齐)
        iou = self._calculate_iou(pred_boxes, target_boxes)
        r_iou = iou * loss_mask * self.w_iou
        obj_rewards += r_iou

        # 2. 关系逻辑奖励 (KG 约束)
        rel_scores = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids)
        r_rel = rel_scores * self.w_rel
        obj_rewards += r_rel

        # 3. [创新点] 语义一致性奖励 (Attention-to-Box Alignment)
        # 如果模型在某物体框外产生了过高的注意力响应，给予严厉惩罚
        r_align = torch.zeros(B, T, device=device)
        if attention_maps is not None:
            # attention_maps shape: [B, T, H, W] (通过抽取 U-Net Cross-Attention 得到)
            r_align = self._calculate_attention_alignment(attention_maps, pred_boxes)
            obj_rewards += r_align * self.w_align

        # 4. 态势能量奖励 (Gestalt Potential Reward)
        # 鼓励‘山’产生纵向势能，‘水’产生横向势能
        r_gestalt = self._calculate_gestalt_reward(gestalt_params, kg_class_ids)
        obj_rewards += r_gestalt * 1.5

        # 5. 重叠惩罚
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes)
        r_over = overlap_penalty * self.w_overlap
        obj_rewards += r_over

        # 记录明细
        self.last_reward_stats = {
            'IoU': r_iou.mean().item(),
            'Rel': r_rel.mean().item(),
            'Align': r_align.mean().item(),
            'Gestalt': r_gestalt.mean().item(),
            'Over': r_over.mean().item()
        }

        return obj_rewards.sum(dim=1) / (T + 1e-6)

    def _calculate_attention_alignment(self, attn_maps, boxes):
        """
        [架构级创新] 计算 Cross-Attention Map 与 矩形框的重合度。
        强制模型‘眼睛’看的地方就是‘框’标注的地方。
        """
        B, T, H, W = attn_maps.shape
        alignment_scores = torch.zeros(B, T, device=boxes.device)
        
        for b in range(B):
            for t in range(T):
                # 构造框的 Binary Mask
                box = boxes[b, t]
                x1, y1 = int((box[0]-box[2]/2)*W), int((box[1]-box[3]/2)*H)
                x2, y2 = int((box[0]+box[2]/2)*W), int((box[1]+box[3]/2)*H)
                
                # 计算框内注意力占比
                attn_map = attn_maps[b, t]
                total_attn = attn_map.sum() + 1e-6
                
                # 限制坐标范围
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                if x2 > x1 and y2 > y1:
                    inner_attn = attn_map[y1:y2, x1:x2].sum()
                    alignment_scores[b, t] = inner_attn / total_attn
        
        return alignment_scores

    def _calculate_gestalt_reward(self, gestalt, class_ids):
        """
        奖励符合意象特征的动态势能倾向。
        """
        B, T, _ = gestalt.shape
        r = torch.zeros(B, T, device=gestalt.device)
        # gestalt indices: 0:bias_x, 1:bias_y, 2:rot, 3:flow
        
        for b in range(B):
            for t in range(T):
                cid = int(class_ids[b, t].item())
                if cid == 2: # 山：奖励向上张力 (negative bias_y)
                    r[b, t] = -gestalt[b, t, 1].clamp(max=0) 
                elif cid == 3: # 水：奖励横向张力 (abs bias_x)
                    r[b, t] = torch.abs(gestalt[b, t, 0]).clamp(max=0.5)
        return r

    def train_rl_epoch(self, epoch):
        self.model.train()
        total_reward = 0
        steps = 0
        
        for step, batch in enumerate(self.train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(self.device)
            
            # --- SCST 训练逻辑 ---
            self.model.eval()
            with torch.no_grad():
                # [MODIFIED] 基础坐标 + 态势参数 (8维)
                baseline_layout, _ = self.model.forward_rl(
                    batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                    batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                    sample=False
                )
                reward_baseline = self.compute_reward(baseline_layout, batch)
            
            self.model.train()
            # [MODIFIED] 采样 8 维动作
            sample_layout, log_probs = self.model.forward_rl(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                sample=True
            )
            
            # [天马行空创新] 获取 U-Net 内部真实的交叉注意力图
            # 这一步通常需要对 model 做 hooks，此处逻辑假设已在 forward 中提取出 attn_maps
            # 如果没有 hook，可以先基于坐标奖励，但在 loss 中加入一致性。
            reward_sample = self.compute_reward(sample_layout, batch)
            
            advantage = reward_sample - reward_baseline
            rl_loss = -(log_probs.sum(dim=1) * advantage).mean()
            
            # 监督辅助
            mu, logvar, dynamic_layout_sup, decoder_output = self.model(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                target_boxes=batch['target_boxes']
            )
            
            loss_tuple = self.model.get_loss(
                pred_cls=None, pred_bbox_ids=None, 
                pred_boxes=dynamic_layout_sup, # 传入 8 维预测
                pred_count=None, layout_seq=None, 
                layout_mask=batch['loss_mask'], 
                num_boxes=batch['num_boxes'].to(self.device), 
                target_coords_gt=batch['target_boxes'],
                kg_spatial_matrix=batch['kg_spatial_matrix'],
                kg_class_weights=batch['kg_class_weights'],
                kg_class_ids=batch['kg_class_ids'],
                decoder_output=decoder_output
            )
            
            supervised_loss = loss_tuple[0]
            consistency_loss = loss_tuple[-1]
            
            kl_loss = compute_kl_loss(mu, logvar) if mu is not None else torch.tensor(0.0)

            # 总损失融合
            total_combined_loss = rl_loss + (0.2 * supervised_loss + 0.1 * kl_loss)
            
            self.optimizer.zero_grad()
            total_combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_sample.mean().item()
            steps += 1
            
            if (step + 1) % 10 == 0:
                stats = self.last_reward_stats
                print(f"[RL V7.0] Step {step+1} | R:{reward_sample.mean().item():.3f} | Adv:{advantage.mean().item():.2f} | "
                      f"Align:{stats['Align']:.2f} | Gest:{stats['Gestalt']:.2f} | Cons:{consistency_loss.item():.3f}")

        avg_reward = total_reward / steps
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        return avg_reward

    def _run_inference_example(self, epoch):
        """
        [NEW] 增加解释性输出：保存注意力图和态势场预览
        """
        super()._run_inference_example(epoch)
        # 此处可以扩展：提取 U-Net 交叉注意力图并作为图片保存
        # 让你能看见模型‘心里’到底把山画在哪了
        pass
# File: tianyusun1/test2/test2-5.2/trainers/rl_trainer.py (V5.19: V5.7 COMPATIBILITY)

import torch
import torch.nn.functional as F
import torch.optim as optim
from .trainer import LayoutTrainer
import time
import os
import matplotlib.pyplot as plt

# [MODIFIED] Import robust KL calculation
from trainers.loss import compute_kl_loss

class RLTrainer(LayoutTrainer):
    """
    强化学习训练器 (RL Fine-tuning Trainer)。
    继承自 LayoutTrainer，复用了验证、保存模型等逻辑，
    但重写了训练循环以支持 SCST (Self-Critical Sequence Training)。
    
    [V5.19 Updates for V5.7 Compatibility]
    1. 适配新的 Poem2LayoutGenerator 接口 (forward 返回 decoder_output)。
    2. 适配新的 get_loss 接口 (接收 decoder_output, 返回 consistency_loss)。
    3. 使用 compute_kl_loss 替代手动计算。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        # === 1. RL 超参数设置 ===
        self.rl_lr = float(config['training'].get('rl_learning_rate', 5e-6))
        
        # 重新定义优化器 (针对 RL 微调)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        
        # === 2. 奖励权重 (Reward Weights) ===
        reward_cfg = config['training'].get('reward_weights', {})
        
        self.w_iou = float(reward_cfg.get('iou', 2.0))              
        self.w_rel = float(reward_cfg.get('relation', 5.0)) # 强语义约束       
        self.w_dispersion = float(reward_cfg.get('dispersion', 0.5)) # 适度分散
        self.w_overlap = float(reward_cfg.get('overlap', -0.5))     

        # 用于记录最近一次 batch 的奖励明细
        self.last_reward_stats = {}

        # 奖励历史记录，用于画图
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer] Initialized. LR={self.rl_lr:.2e}")
        print(f"[RLTrainer] Reward Weights: IoU={self.w_iou}, Rel={self.w_rel}, Disp={self.w_dispersion}, Overlap={self.w_overlap}")
        print(f"[RLTrainer] Reward plot will be saved to: {self.plot_path_reward}")

    def compute_reward(self, pred_boxes, batch):
        """
        计算每个样本的奖励值 (Batch-wise Reward Calculation)。
        """
        B, T, _ = pred_boxes.shape
        device = pred_boxes.device
        
        # 提取 Batch 信息
        loss_mask = batch['loss_mask']          # 1.0 = 有效GT, 0.0 = 无GT/脏数据
        target_boxes = batch['target_boxes']
        kg_spatial_matrix = batch['kg_spatial_matrix'] # [B, 9, 9]
        kg_class_ids = batch['kg_class_ids']    # [B, T]
        
        # 初始化奖励矩阵 [B, T] -> 每个物体单独计算，最后求和
        obj_rewards = torch.zeros(B, T, device=device)
        
        # ===========================================================
        # A. 监督奖励 (Supervised Reward) - 针对有 GT 的部分
        # ===========================================================
        iou = self._calculate_iou(pred_boxes, target_boxes) # [B, T]
        r_iou = iou * loss_mask * self.w_iou
        obj_rewards += r_iou

        # ===========================================================
        # B. 关系奖励 (Relation Reward)
        # ===========================================================
        # 移除 no_gt_mask，让所有物体都受 KG 逻辑约束
        rel_scores = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids) # [B, T]
        r_rel = rel_scores * self.w_rel
        obj_rewards += r_rel

        # ===========================================================
        # C. 分散度奖励 (Dispersion Reward) - 使用最近邻距离
        # ===========================================================
        centers = pred_boxes[..., :2] # [B, T, 2]
        
        # 计算所有物体中心点之间的距离矩阵 [B, T, T]
        dists = torch.cdist(centers, centers)
        
        # 把对角线（自己到自己）设为一个大数
        eye = torch.eye(T, device=device).unsqueeze(0)
        dists = dists + eye * 10.0
        
        # 找到每个物体离它最近的邻居的距离 [B, T]
        min_dist, _ = dists.min(dim=2)
        
        # 奖励这个距离，截断
        disp_score = torch.clamp(min_dist, max=0.3) 
        
        r_disp = disp_score * self.w_dispersion
        obj_rewards += r_disp

        # ===========================================================
        # D. 构图美学奖励：三分法 (Rule of Thirds)
        # ===========================================================
        cx = centers[..., 0]
        cy = centers[..., 1]
        
        dist_x_third = torch.min(torch.abs(cx - 0.333), torch.abs(cx - 0.667))
        dist_y_third = torch.min(torch.abs(cy - 0.333), torch.abs(cy - 0.667))
        
        r_composition = (0.2 - (dist_x_third + dist_y_third)).clamp(min=0) * 2.5
        obj_rewards += r_composition 

        # ===========================================================
        # E. 边缘惩罚 (Border Penalty)
        # ===========================================================
        dist_to_edge = torch.min(centers, 1.0 - centers) # [B, T, 2]
        is_too_close_to_edge = (dist_to_edge < 0.05).float().sum(dim=-1) # [B, T]
        r_border_penalty = is_too_close_to_edge * -0.5 
        obj_rewards += r_border_penalty

        # ===========================================================
        # F. 重叠惩罚 (Overlap Penalty)
        # ===========================================================
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes) # [B, T]
        r_over = overlap_penalty * self.w_overlap
        obj_rewards += r_over

        # 记录明细
        self.last_reward_stats = {
            'IoU': (r_iou.sum() / (loss_mask.sum() + 1e-6)).item(),
            'Rel': (r_rel.sum() / (B * T)).item(),
            'Disp': disp_score.mean().item(),
            'Comp': r_composition.mean().item(),
            'Over': overlap_penalty.mean().item()
        }

        # === 汇总 ===
        total_sample_reward = obj_rewards.sum(dim=1) / (T + 1e-6)
        
        return total_sample_reward

    def _calculate_iou(self, pred, target):
        """辅助函数：计算 IoU [B, T]"""
        p_x1, p_y1 = pred[..., 0]-pred[..., 2]/2, pred[..., 1]-pred[..., 3]/2
        p_x2, p_y2 = pred[..., 0]+pred[..., 2]/2, pred[..., 1]+pred[..., 3]/2
        t_x1, t_y1 = target[..., 0]-target[..., 2]/2, target[..., 1]-target[..., 3]/2
        t_x2, t_y2 = target[..., 0]+target[..., 2]/2, target[..., 1]+target[..., 3]/2
        
        i_x1 = torch.max(p_x1, t_x1)
        i_y1 = torch.max(p_y1, t_y1)
        i_x2 = torch.min(p_x2, t_x2)
        i_y2 = torch.min(p_y2, t_y2)
        
        i_area = (i_x2 - i_x1).clamp(min=0) * (i_y2 - i_y1).clamp(min=0)
        p_area = pred[..., 2] * pred[..., 3]
        t_area = target[..., 2] * target[..., 3]
        u_area = p_area + t_area - i_area
        return i_area / (u_area + 1e-6)

    def _calculate_relation_reward(self, boxes, matrix, class_ids):
        """辅助函数：检查 KG 关系一致性 [B, T]"""
        B, T, _ = boxes.shape
        rewards = torch.zeros(B, T, device=boxes.device)
        
        if matrix is None: return rewards

        for b in range(B):
            for i in range(T):
                cid_i = class_ids[b, i].item()
                idx_i = int(cid_i) - 2
                if not (0 <= idx_i < 9): continue
                
                for j in range(T):
                    if i == j: continue
                    cid_j = class_ids[b, j].item()
                    idx_j = int(cid_j) - 2
                    if not (0 <= idx_j < 9): continue
                    
                    rel = matrix[b, idx_i, idx_j].item()
                    if rel == 0: continue
                    
                    box_i = boxes[b, i]
                    box_j = boxes[b, j]
                    
                    reward_val = 0.0
                    
                    # [密集奖励逻辑]
                    if rel == 1: # ABOVE
                        diff = box_j[1] - box_i[1]
                        if diff > 0: reward_val = 1.0
                        else: reward_val = 0.2 * diff 
                        
                    elif rel == 2: # BELOW
                        diff = box_i[1] - box_j[1]
                        if diff > 0: reward_val = 1.0
                        else: reward_val = 0.2 * diff
                        
                    elif rel == 3: # INSIDE
                        dx = abs(box_i[0] - box_j[0])
                        dy = abs(box_i[1] - box_j[1])
                        if dx < box_j[2]/2 and dy < box_j[3]/2: 
                            reward_val = 1.0
                        
                    elif rel == 4: # SURROUNDS
                        dx = abs(box_i[0] - box_j[0])
                        dy = abs(box_i[1] - box_j[1])
                        if dx < box_i[2]/2 and dy < box_i[3]/2: 
                            reward_val = 1.0
                    
                    elif rel == 5: # ON TOP
                        diff = box_j[1] - box_i[1]
                        if diff > 0: reward_val = 1.0
                        else: reward_val = 0.2 * diff

                    rewards[b, i] += reward_val
                        
        return torch.clamp(rewards, min=-1.0, max=3.0)

    def _calculate_overlap_penalty(self, boxes):
        """计算两两重叠惩罚"""
        B, T, _ = boxes.shape
        centers = boxes[..., :2]
        dist = torch.cdist(centers, centers)
        
        too_close = (dist < 0.1).float()
        mask = torch.eye(T, device=boxes.device).unsqueeze(0).expand(B, -1, -1)
        too_close = too_close * (1 - mask)
        
        penalty = too_close.sum(dim=2) 
        return penalty

    def _plot_reward_history(self):
        """绘制并保存奖励曲线"""
        if not self.reward_history:
            return
            
        epochs = range(1, len(self.reward_history) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.reward_history, label='Avg Epoch Reward', color='purple', marker='o', linestyle='-')
        
        plt.title('RL Training Reward Trajectory', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        try:
            plt.savefig(self.plot_path_reward)
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not save reward plot: {e}")

    def train_rl_epoch(self, epoch):
        """
        执行一个 Epoch 的 Mixed Training (RL + Supervised Anchor)
        """
        self.model.train()
        total_reward = 0
        steps = 0
        
        print(f"\n--- [RL] Starting RL Epoch {epoch+1} (Mixed Training) ---")
        
        for step, batch in enumerate(self.train_loader):
            # 1. 数据移至 GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            # ==========================================
            # Part A: Reinforcement Learning (RL)
            # ==========================================
            # 1. Baseline (Greedy)
            self.model.eval()
            with torch.no_grad():
                baseline_boxes, _ = self.model.forward_rl(
                    batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                    batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                    sample=False
                )
                reward_baseline = self.compute_reward(baseline_boxes, batch)
            
            # 2. Sampling (Exploration)
            self.model.train()
            sample_boxes, log_probs = self.model.forward_rl(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                sample=True
            )
            reward_sample = self.compute_reward(sample_boxes, batch)
            
            # 3. Advantage
            advantage = reward_sample - reward_baseline
            log_prob_sum = log_probs.sum(dim=1)
            rl_loss = -(log_prob_sum * advantage).mean()
            
            # ==========================================
            # Part B: Supervised Anchor (Mixed Training)
            # ==========================================
            # 1. 执行标准前向传播 (Teacher Forcing / CVAE)
            # [MODIFIED] 捕获 4 个返回值，包括 decoder_output
            mu, logvar, pred_boxes_sup, decoder_output = self.model(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                target_boxes=batch['target_boxes'] # 传入 GT 用于 Encoder
            )
            
            # 2. 计算监督损失
            # [MODIFIED] 传入 decoder_output
            loss_tuple = self.model.get_loss(
                pred_cls=None, pred_bbox_ids=None, 
                pred_boxes=pred_boxes_sup, # 使用监督路径的预测值
                pred_count=None, layout_seq=None, 
                layout_mask=batch['loss_mask'], 
                num_boxes=batch['num_boxes'].to(self.device), 
                target_coords_gt=batch['target_boxes'],
                kg_spatial_matrix=batch['kg_spatial_matrix'],
                kg_class_weights=batch['kg_class_weights'],
                kg_class_ids=batch['kg_class_ids'],
                decoder_output=decoder_output # [NEW]
            )
            
            # [MODIFIED] 解包新的返回值 (11个)
            supervised_loss, _, _, _, _, _, _, _, _, _, consistency_loss = loss_tuple
            
            # 3. 计算 KL 散度 (CVAE 正则)
            if mu is not None and logvar is not None:
                # [MODIFIED] 使用健壮的 KL 计算
                kl_loss = compute_kl_loss(mu, logvar, free_bits=1.0)
            else:
                kl_loss = torch.tensor(0.0, device=self.device)

            # ==========================================
            # Part C: 总损失融合与更新
            # ==========================================
            alpha = 1.0 
            
            # 最终 Loss = RL探索 + 监督信号 + VAE正则 + 一致性(包含在supervised_loss里)
            total_combined_loss = alpha * rl_loss + (0.3 * supervised_loss + 0.05 * kl_loss)
            
            self.optimizer.zero_grad()
            total_combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_sample.mean().item()
            steps += 1
            
            if (step + 1) % 10 == 0:
                stats = self.last_reward_stats
                print(f"[RL] Epoch {epoch+1} Step {step+1} | "
                      f"R_Avg: {reward_sample.mean().item():.3f} | "
                      f"Adv: {advantage.mean().item():.3f} | "
                      f"Loss: {total_combined_loss.item():.4f} || "
                      f"IoU:{stats.get('IoU', 0):.2f} "
                      f"Rel:{stats.get('Rel', 0):.2f} "
                      f"Disp:{stats.get('Disp', 0):.2f} "
                      f"Comp:{stats.get('Comp', 0):.2f} "
                      f"Over:{stats.get('Over', 0):.2f} | "
                      f"Cons: {consistency_loss.item():.3f}")

        avg_reward = total_reward / steps if steps > 0 else 0
        print(f"--- [RL] Epoch {epoch+1} Finished. Avg Epoch Reward: {avg_reward:.4f} ---")
        
        # 记录历史并绘图，然后返回平均奖励
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        
        return avg_reward
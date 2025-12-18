# File: tianyusun1/test2/test2-5.1/trainers/trainer.py (V5.7: ROBUST KL + CONSISTENCY)

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (trainer.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取项目根目录 (trainer.py 的父目录, 即 test2-4.0)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将项目根目录插入到 sys.path 的开头
sys.path.insert(0, project_root)

# --- 现在可以安全地导入项目内部模块 ---
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from transformers import BertTokenizer
from data.dataset import PoegraphLayoutDataset, layout_collate_fn 
from models.poem2layout import Poem2LayoutGenerator
from collections import Counter
import numpy as np 
import time
import contextlib 
import torch.nn.utils 
from torch.optim.lr_scheduler import LambdaLR 

# --- NEW IMPORTS for Visualization/Inference/Plotting ---
from inference.greedy_decode import greedy_decode_poem_layout 
from data.visualize import draw_layout
import matplotlib.pyplot as plt 

# --- [MODIFIED] 导入新的 KL 计算函数 ---
from trainers.loss import compute_kl_loss
# ---------------------------------------------


# =========================================================================
# LayoutTrainer 类定义 (实现完整的训练、验证和测试逻辑)
# =========================================================================

class LayoutTrainer:
    """负责训练循环、优化器管理、日志记录和模型保存。"""
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        print(f"Trainer initialized on device: {self.device}")
        
        # 新增推理所需参数
        self.tokenizer = tokenizer
        self.example_poem = example_poem
        
        # 训练和保存频率设置
        self.lr = config['training']['learning_rate'] # Store base LR
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.lr
        )
        self.epochs = config['training']['epochs']
        self.output_dir = config['training']['output_dir']
        self.log_steps = config['training'].get('log_steps', 10)
        self.save_every = config['training'].get('save_every', 10)
        self.visualize_every = config['training'].get('visualize_every', 1) 
        
        # 分离绘图路径
        self.plot_path_recons = os.path.join(self.output_dir, "recons_trajectory.png")
        self.plot_path_kl = os.path.join(self.output_dir, "kl_trajectory.png")
        os.makedirs(self.output_dir, exist_ok=True)

        # 学习率调度器初始化
        self.warmup_steps = config['training'].get('warmup_steps', 0)
        self.total_steps = len(train_loader) * self.epochs
        self.global_step = 0
        self.scheduler = self._get_lr_scheduler()

        # 最佳模型路径追踪
        self.current_best_model_path = None
        
        # 损失历史追踪 (扩充以监控所有新 Loss)
        self.train_loss_history = []
        self.val_loss_history = []
        
        # 各分量损失 (主要用于验证集可视化)
        self.val_reg_history = [] 
        self.val_iou_history = []
        self.val_area_history = [] 
        self.val_relation_history = [] 
        self.val_overlap_history = []  
        self.val_size_history = []      
        self.val_kl_history = []      
        
        # [NEW V5.0] 新增审美损失历史
        self.val_alignment_history = []
        self.val_balance_history = []
        
        # [NEW V5.4] 新增聚类损失历史
        self.val_clustering_history = []

    def _get_lr_scheduler(self):
        """定义带线性 Warmup、5 Epoch Hold 和后续衰减的学习率调度器。"""
        N_HOLD_EPOCHS = 5
        steps_per_epoch = len(self.train_loader)
        
        # 计算 Hold 阶段结束的 Step
        hold_steps = steps_per_epoch * N_HOLD_EPOCHS
        
        # 确保总步数大于 hold_steps
        if self.total_steps < hold_steps:
             # print("[Warning] Total steps < Hold steps. Setting hold_steps to total_steps.")
             hold_steps = self.total_steps

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # Warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            elif current_step < hold_steps:
                # Hold
                return 1.0 
            else:
                # Decay
                decay_start_step = hold_steps
                decay_steps = self.total_steps - decay_start_step
                if decay_steps > 0:
                    relative_step = current_step - decay_start_step
                    # 线性衰减到 0
                    return max(0.0, 1.0 - (relative_step / decay_steps))
                return 0.0
            
        return LambdaLR(self.optimizer, lr_lambda)

    def _update_curriculum(self, epoch):
        """
        [V4.2] 课程学习策略更新：
        1. 动态调整 Reconstruction Weights (Logic -> Realism)
        2. 动态调整 KL Weight (KL Annealing)
        """
        
        # --- 策略 A: 权重转移 (Rel: 5->1, Reg: 1->5, 在epoch 间线性过渡) ---
        
        # 阶段 1: 强逻辑
        if epoch < 50:
            new_rel_weight = 5.0
            new_reg_weight = 1.0
        
        # 阶段 2: 线性过渡
        else:
            transition_start_epoch = 50
            transition_duration = 100 
            
            progress = min(1.0, (epoch - transition_start_epoch) / transition_duration)
            
            # Rel: 5.0 -> 1.0
            new_rel_weight = 5.0 - (4.0 * progress) 
            # Reg: 1.0 -> 5.0
            new_reg_weight = 1.0 + (4.0 * progress) 
            
        # 更新模型权重属性 (供 get_loss 使用)
        if hasattr(self.model, 'relation_loss_weight'):
            self.model.relation_loss_weight = new_rel_weight
        if hasattr(self.model, 'reg_loss_weight'):
            self.model.reg_loss_weight = new_reg_weight
        
        # --- 策略 B: KL Annealing (V5.2: Early Intervention) ---
        
        # [MODIFIED] 降低目标 KL 权重，并延长退火周期
        target_kl = 0.005 # 原始为 0.01，降低目标值
        
        # [MODIFIED V5.2] 提早介入 KL Loss (Epoch 5)，防止 Posterior Collapse
        kl_transition_start = 5
        # [MODIFIED] 延长退火周期，从 50 延长到 150 Epoch
        kl_transition_duration = 150 # 原始为 50。延长退火至 Epoch 155 (5 + 150)
        
        if epoch < kl_transition_start:
            kl_weight = 0.0
        else:
            kl_progress = min(1.0, (epoch - kl_transition_start) / kl_transition_duration)
            kl_weight = target_kl * kl_progress
            
        # 返回当前权重和 KL 因子
        return new_rel_weight, new_reg_weight, kl_weight

    def _run_epoch(self, data_loader, is_training: bool, epoch: int = 0):
        """处理一个训练/验证/测试轮次"""
        self.model.train() if is_training else self.model.eval()
        
        # 应用课程学习 (仅训练时更新权重并获取因子，验证时沿用模型的权重并使用固定 KL 因子)
        if is_training:
            cur_rel_w, cur_reg_w, cur_kl_w = self._update_curriculum(epoch)
        else:
            # 验证/测试时，使用模型当前配置的重建权重，KL 权重使用 0.01 用于观察最终总损失
            cur_rel_w = self.model.relation_loss_weight if hasattr(self.model, 'relation_loss_weight') else 5.0
            cur_reg_w = self.model.reg_loss_weight if hasattr(self.model, 'reg_loss_weight') else 1.0
            cur_kl_w = 0.01 # 固定的 KL 权重

        # 初始化运行损失变量
        total_loss_val = 0.0
        total_reg_loss = 0.0 
        total_iou_loss = 0.0
        total_area_loss = 0.0
        total_relation_loss = 0.0 
        total_overlap_loss = 0.0 
        total_size_loss = 0.0
        # [NEW V5.0]
        total_align_loss = 0.0
        total_balance_loss = 0.0
        # [NEW V5.4]
        total_clustering_loss = 0.0
        
        total_kl_loss = 0.0 # 存储原始 KL 散度值
        
        context_manager = contextlib.nullcontext() if is_training else torch.no_grad()
        
        data_len = len(data_loader)
        
        with context_manager:
            for step, batch in enumerate(data_loader):
                # 1. 将数据移至设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                kg_class_ids = batch['kg_class_ids'].to(self.device) 
                padding_mask = batch['padding_mask'].to(self.device) 
                target_boxes = batch['target_boxes'].to(self.device) 
                loss_mask = batch['loss_mask'].to(self.device) 
                
                # 提取额外数据
                kg_class_weights = batch.get('kg_class_weights', None)
                if kg_class_weights is not None:
                     kg_class_weights = kg_class_weights.to(self.device)

                kg_spatial_matrix = batch.get('kg_spatial_matrix', None)
                if kg_spatial_matrix is not None:
                    kg_spatial_matrix = kg_spatial_matrix.to(self.device)
                
                location_grids = batch.get('location_grids', None)
                if location_grids is not None:
                    location_grids = location_grids.to(self.device)
                
                # 2. 前向传播 (CVAE)
                # [MODIFIED] 捕获 decoder_output 以用于一致性损失计算
                mu, logvar, pred_boxes, decoder_output = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    kg_class_ids=kg_class_ids, 
                    padding_mask=padding_mask, 
                    kg_spatial_matrix=kg_spatial_matrix,
                    location_grids=location_grids,
                    target_boxes=target_boxes # 传入 GT 用于 Encoder
                )
                
                # 3. 计算重建损失 (Reconstruction Loss)
                loss_tuple = self.model.get_loss(
                    pred_cls=None, 
                    pred_bbox_ids=None, 
                    pred_boxes=pred_boxes, 
                    pred_count=None, 
                    layout_seq=None, 
                    layout_mask=loss_mask, 
                    num_boxes=batch['num_boxes'].to(self.device), 
                    target_coords_gt=target_boxes,
                    kg_spatial_matrix=kg_spatial_matrix,
                    kg_class_weights=kg_class_weights,
                    kg_class_ids=kg_class_ids, # [NEW] Pass class IDs for clustering
                    decoder_output=decoder_output # [NEW] Pass decoder_output for consistency loss
                )
                
                # [MODIFIED V5.7] 解包重建损失 (共 11 个, 新增 loss_consistency)
                loss_recons, relation_loss, overlap_loss, reg_loss, iou_loss, size_loss, area_loss, align_loss, balance_loss, clustering_loss, consistency_loss = loss_tuple
                
                # 4. 计算 KL 散度损失 (KL Divergence)
                if mu is not None and logvar is not None:
                    # [MODIFIED] 使用健壮的 KL 计算函数 (含 Free Bits 机制)
                    # free_bits=1.0 允许每个样本保留 1 nat 的信息量而不受惩罚，防止坍缩
                    kl_val = compute_kl_loss(mu, logvar, free_bits=1.0)
                else:
                    kl_val = torch.tensor(0.0, device=self.device)
                
                # 5. 总损失 = 重建 + KL * beta (使用当前 KL 权重)
                final_loss = loss_recons + cur_kl_w * kl_val
                
                if is_training:
                    self.optimizer.zero_grad()
                    final_loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # 学习率调度器更新
                    self.scheduler.step()
                    self.global_step += 1
                
                # 累加所有损失
                total_loss_val += final_loss.item()
                total_reg_loss += reg_loss.item()
                total_iou_loss += iou_loss.item()
                total_area_loss += area_loss.item()
                total_relation_loss += relation_loss.item()
                total_overlap_loss += overlap_loss.item()
                total_size_loss += size_loss.item()
                # [NEW]
                total_align_loss += align_loss.item()
                total_balance_loss += balance_loss.item()
                total_clustering_loss += clustering_loss.item()
                
                total_kl_loss += kl_val.item() 
                
                # 6. 打印日志 (Training log)
                if is_training and (step + 1) % self.log_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch+1}][TRAIN] Step {step+1}/{data_len} | "
                          f"LR: {current_lr:.6e} | "
                          f"Total: {final_loss.item():.4f} | "
                          f"Rel: {relation_loss.item():.3f} | "
                          f"Reg: {reg_loss.item():.3f} | "
                          f"Alg: {align_loss.item():.3f} | "
                          f"Bal: {balance_loss.item():.3f} | "
                          f"Clus: {clustering_loss.item():.3f} | "
                          f"Cons: {consistency_loss.item():.3f} | "
                          f"KL: {kl_val.item():.4f}")
        
        # 计算平均损失
        num_batches = len(data_loader)
        if num_batches == 0:
             return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        return (
            total_loss_val / num_batches,
            total_relation_loss / num_batches,
            total_overlap_loss / num_batches,
            total_reg_loss / num_batches,
            total_iou_loss / num_batches,
            total_size_loss / num_batches,
            total_area_loss / num_batches,
            total_align_loss / num_batches,   
            total_balance_loss / num_batches, 
            total_clustering_loss / num_batches, # [NEW]
            total_kl_loss / num_batches
        )

    def validate(self, epoch=0):
        """运行验证集上的评估"""
        start_time = time.time()
        print("\n--- Starting Validation ---")
        
        avg_losses = self._run_epoch(self.val_loader, is_training=False, epoch=epoch)
        # [MODIFIED V5.4] 解包 11 个值
        avg_val_loss, avg_rel, avg_over, avg_reg, avg_iou, avg_size, avg_area, avg_align, avg_bal, avg_clus, avg_kl = avg_losses
        
        end_time = time.time()
        print(f"--- Validation Finished in {end_time - start_time:.2f}s ---")
        
        # 记录历史
        self.val_loss_history.append(avg_val_loss)
        self.val_reg_history.append(avg_reg) 
        self.val_iou_history.append(avg_iou)
        self.val_area_history.append(avg_area) 
        self.val_relation_history.append(avg_rel)
        self.val_overlap_history.append(avg_over)
        self.val_size_history.append(avg_size)
        self.val_alignment_history.append(avg_align) 
        self.val_balance_history.append(avg_bal)
        self.val_clustering_history.append(avg_clus) # [NEW]
        self.val_kl_history.append(avg_kl)
        
        print(f"Val Avg Loss: Total: {avg_val_loss:.4f} | "
              f"Rel: {avg_rel:.4f} | "
              f"Over: {avg_over:.4f} | "
              f"Reg: {avg_reg:.4f} | " 
              f"Alg: {avg_align:.4f} | "
              f"Bal: {avg_bal:.4f} | "
              f"Clus: {avg_clus:.4f} | "
              f"KL: {avg_kl:.4f}") 
              
        return avg_val_loss

    def test(self):
        """运行测试集上的评估"""
        start_time = time.time()
        print("\n--- Starting Test Set Evaluation ---")
        
        avg_losses = self._run_epoch(self.test_loader, is_training=False, epoch=999) 
        # [MODIFIED V5.4] 解包
        avg_test_loss, avg_rel, avg_over, avg_reg, avg_iou, avg_size, avg_area, avg_align, avg_bal, avg_clus, avg_kl = avg_losses
        
        end_time = time.time()
        print(f"--- Test Finished in {end_time - start_time:.2f}s ---")
        
        print(f"Test Avg Loss: Total: {avg_test_loss:.4f} | "
              f"Rel: {avg_rel:.4f} | "
              f"Over: {avg_over:.4f} | "
              f"Reg: {avg_reg:.4f} | " 
              f"Alg: {avg_align:.4f} | "
              f"Bal: {avg_bal:.4f} | "
              f"Clus: {avg_clus:.4f} | "
              f"KL: {avg_kl:.4f}") 
              
        return avg_test_loss

    def _run_inference_example(self, epoch):
        """运行固定样例的推理并保存可视化图片"""
        print(f"\n--- Running Inference Example for Epoch {epoch+1} ---")
        
        self.model.eval() 
        poem_text = self.example_poem['poem']
        print(f"Poem: {poem_text}")
        try:
            ds = self.train_loader.dataset
            if hasattr(ds, 'dataset'): ds = ds.dataset 
            if hasattr(ds, 'pkg'):
                pkg = ds.pkg
                kg_vector = pkg.extract_visual_feature_vector(poem_text)
                indices = torch.nonzero(torch.tensor(kg_vector)).squeeze(1).tolist()
                ids = [i + 2 for i in indices]
                print(f"KG Objects (IDs): {ids}")
        except Exception:
            pass

        max_elements = self.config['model'].get('max_elements', 30)
        
        with torch.no_grad():
            layout = greedy_decode_poem_layout(
                self.model, 
                self.tokenizer, 
                poem_text, 
                max_elements=max_elements,
                device=self.device.type
            )
        
        output_path = os.path.join(self.output_dir, f"epoch_{epoch+1}_layout_pred.png")
        draw_layout(layout, f"PRED (CVAE) E{epoch+1}: {poem_text}", output_path)
        print(f"-> Generated layout saved to {output_path}")

        # 真实布局可视化
        if epoch == 0 or (epoch + 1) % (self.visualize_every * 10) == 0:
            true_boxes = self.example_poem['boxes']
            true_layout_path = os.path.join(self.output_dir, f"layout_true_example.png")
            draw_layout(true_boxes, f"TRUE: {poem_text}", true_layout_path)
            print(f"-> True layout saved to {true_layout_path}")
            
        print("---------------------------------------------------")
        self.model.train()

    def _plot_loss_history(self):
        """绘制并保存损失变化轨迹图 (分离 Reconstruction 和 KL)"""
        if not self.train_loss_history:
            return

        epochs = range(1, len(self.train_loss_history) + 1)
        
        # === 图 1: 重建损失 (Reconstruction Losses) ===
        plt.figure(figsize=(12, 8))
        
        plt.plot(epochs, self.train_loss_history, label='Train Total', color='blue', marker='o', linestyle='-', alpha=0.6, markersize=3)
        plt.plot(epochs, self.val_loss_history, label='Val Total (w/ KL)', color='red', marker='s', linestyle='-', alpha=0.8, markersize=3)
        
        if len(self.val_reg_history) > 1:
            plt.plot(epochs, self.val_relation_history, label='Val Rel', color='green', linestyle=':', alpha=0.7)
            plt.plot(epochs, self.val_overlap_history, label='Val Over', color='orange', linestyle=':', alpha=0.7)
            plt.plot(epochs, self.val_reg_history, label='Val Reg', color='purple', linestyle='--', alpha=0.5) 
            # [NEW] 绘制审美损失
            plt.plot(epochs, self.val_alignment_history, label='Val Align', color='cyan', linestyle='-.', alpha=0.6)
            plt.plot(epochs, self.val_balance_history, label='Val Bal', color='magenta', linestyle='-.', alpha=0.6)
            # [NEW] 绘制聚类损失
            plt.plot(epochs, self.val_clustering_history, label='Val Clus', color='brown', linestyle='-.', alpha=0.6)

        plt.title('Loss Trajectory (V5.4: +Clustering Loss)', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        try:
            plt.savefig(self.plot_path_recons) 
            plt.close()
            print(f"-> Reconstruction loss plot saved to {self.plot_path_recons}")
        except Exception as e:
            print(f"[Warning] Could not save Reconstruction loss plot: {e}")

        # === 图 2: KL 散度 (KL Divergence) ===
        if len(self.val_kl_history) > 1:
            plt.figure(figsize=(10, 6))
            
            plt.plot(epochs, self.val_kl_history, label='Original KL Div', color='darkblue', marker='.', linestyle='-')
            
            kl_weights_plot = [self._update_curriculum(e - 1)[2] for e in epochs]
            ax2 = plt.gca().twinx()
            ax2.plot(epochs, kl_weights_plot, label='KL Weight (Beta)', color='red', linestyle='--', alpha=0.5)
            ax2.set_ylabel('KL Weight (Beta)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            plt.title('KL Divergence Trajectory (Annealing)', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('KL Value (Validation)', fontsize=12)
            
            lines, labels = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
            
            plt.grid(True, linestyle='--', alpha=0.5)
            
            try:
                plt.savefig(self.plot_path_kl)
                plt.close()
                print(f"-> KL divergence plot saved to {self.plot_path_kl}")
            except Exception as e:
                print(f"[Warning] Could not save KL plot: {e}")

    def train(self):
        """主训练循环"""
        print("--- Starting Full Training ---")
        best_val_loss = float('inf')
        
        print(f"Total training steps: {self.total_steps}, Warmup steps: {self.warmup_steps}, Base LR: {self.lr:.6e}")
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            print(f"\n==================== Epoch {epoch+1}/{self.epochs} | Training ====================")
            
            avg_train_loss = self._run_epoch(self.train_loader, is_training=True, epoch=epoch)[0]
            self.train_loss_history.append(avg_train_loss)
            
            epoch_end_time = time.time()
            print(f"\nEpoch {epoch+1} finished. Avg Training Loss: {avg_train_loss:.4f} ({epoch_end_time - epoch_start_time:.2f}s)")
            
            avg_val_loss = self.validate(epoch=epoch) 

            self._plot_loss_history()
            
            if (epoch + 1) % self.visualize_every == 0:
                self._run_inference_example(epoch)

            if (epoch + 1) % self.save_every == 0:
                self.test() 
                model_name = f"model_epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}.pth"
                checkpoint_path = os.path.join(self.output_dir, model_name)
                torch.save(
                    {'model_state_dict': self.model.state_dict(), 
                     'epoch': epoch+1, 
                     'val_loss': avg_val_loss,
                     'optimizer_state_dict': self.optimizer.state_dict()}, 
                    checkpoint_path
                )
                print(f"-> Checkpoint saved to {checkpoint_path}")

            if avg_val_loss < best_val_loss:
                print("-> New best validation loss achieved. Replacing previous best model.")
                prev_best_path = self.current_best_model_path
                best_val_loss = avg_val_loss
                model_name = f"model_best_val_loss_{avg_val_loss:.4f}.pth" 
                new_best_path = os.path.join(self.output_dir, model_name)
                
                torch.save(
                    {'model_state_dict': self.model.state_dict(), 
                     'epoch': epoch+1, 
                     'val_loss': avg_val_loss,
                     'optimizer_state_dict': self.optimizer.state_dict()}, 
                    new_best_path
                )
                print(f"-> New best model saved to {new_best_path}")

                if prev_best_path and os.path.exists(prev_best_path) and prev_best_path != new_best_path:
                    try:
                        os.remove(prev_best_path)
                        print(f"-> Deleted previous best model: {prev_best_path}")
                    except OSError as e:
                        print(f"[Warning] Could not delete previous best model: {e}")
                
                self.current_best_model_path = new_best_path
                    
        print("\n--- Training Completed ---")
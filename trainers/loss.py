import torch
import torch.nn.functional as F

def layout_loss(pred_cls, pred_coord, target_layout_seq, target_layout_mask, coord_loss_weight=1.0):
    """
    计算布局生成的混合损失 (Supervised Reconstruction Loss)。
    Args:
        pred_cls: [B, num_elements, num_classes] 预测的类别 logits
        pred_coord: [B, num_elements, 4] 预测的坐标 (cx, cy, w, h)
        target_layout_seq: [B, S] (S = 5 * num_elements) 真实布局序列
        target_layout_mask: [B, S] 真实布局掩码
        coord_loss_weight: 坐标损失权重
    Returns:
        total_loss, cls_loss, coord_loss
    """
    batch_size, seq_len = target_layout_seq.shape
    num_elements = seq_len // 5

    reshaped_target = target_layout_seq.view(batch_size, num_elements, 5)
    target_cls_ids = reshaped_target[:, :, 0].long() # [B, num_elements]
    target_coords = reshaped_target[:, :, 1:5].float() # [B, num_elements, 4]

    reshaped_mask = target_layout_mask.view(batch_size, num_elements, 5)
    cls_mask = reshaped_mask[:, :, 0].bool() # [B, num_elements]

    # Classification loss
    # reduction='none' 允许我们手动应用 mask
    cls_loss = F.cross_entropy(pred_cls, target_cls_ids, reduction='none') # [B, num_elements]
    cls_loss = cls_loss * cls_mask.float()
    cls_loss = cls_loss.sum() / cls_mask.sum().clamp(min=1)

    # Coordinate loss
    coord_loss = F.smooth_l1_loss(pred_coord, target_coords, reduction='none') # [B, num_elements, 4]
    coord_loss = coord_loss * cls_mask.unsqueeze(-1).float()
    coord_loss = coord_loss.sum() / (cls_mask.sum().clamp(min=1) * 4)

    total_loss = cls_loss + coord_loss_weight * coord_loss
    return total_loss, cls_loss, coord_loss

def compute_kl_loss(mu, logvar, free_bits=0.0):
    """
    [New] 健壮的 KL 散度计算函数，支持 Free Bits 策略。
    
    Args:
        mu: [B, latent_dim] 均值
        logvar: [B, latent_dim] 对数方差
        free_bits: float, KL 散度的最小阈值（Hinge Loss）。
                   如果 KL < free_bits，则 Loss 为 0。
                   这可以防止后验分布过早坍缩到先验，保留一定的编码信息。
                   
    Returns:
        kl_loss: scalar (averaged over batch)
    """
    # 1. 计算每个样本的 KL 散度
    # 公式: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # 注意: 我们需要在 latent_dim 维度 (dim=1) 求和，表示单个样本的总 KL 信息量
    kld_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_per_sample = torch.sum(kld_element, dim=1) # [B]
    
    # 2. 应用 Free Bits (Hinge Loss)
    # 允许每个样本保留 free_bits 的信息量而不受惩罚
    if free_bits > 0.0:
        free_bits_tensor = torch.tensor(free_bits, device=mu.device)
        kld_per_sample = torch.max(kld_per_sample, free_bits_tensor)
        
    # 3. 对 Batch 求平均
    # 这是关键：必须使用 mean，否则 Loss 会随 Batch Size 变大而变大，压倒重建 Loss
    kl_loss = torch.mean(kld_per_sample)
    
    return kl_loss
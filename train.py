"""
训练与验证流程模块
实现完整的训练+验证闭环，保存最优和最新模型
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime


###########################################
# 损失函数
###########################################

class DiceLoss(nn.Module):
    """Dice损失函数"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes, D, H, W)
            targets: (batch, D, H, W) - 整数标签
        """
        num_classes = logits.shape[1]
        
        # 将targets转换为one-hot
        targets_onehot = torch.zeros_like(logits)
        for c in range(num_classes):
            targets_onehot[:, c] = (targets == c).float()
        
        # 应用softmax
        probs = F.softmax(logits, dim=1)
        
        # 计算Dice
        intersection = (probs * targets_onehot).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets_onehot.sum(dim=(2, 3, 4))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """组合损失：CrossEntropy + Dice
    
    【优化说明】
    - CE 损失支持 class_weight，用于处理医学影像中前景远少于背景的问题
    - 对于 BraTS 这类病灶分割任务，建议给前景(label=1)更高权重
    """
    def __init__(self, weight=None, weight_ce=0.5, weight_dice=0.5):
        """
        Args:
            weight: 类别权重，用于 CrossEntropyLoss，形状为 (num_classes,)
                    例如：torch.tensor([1.0, 10.0]) 表示给前景更大权重
            weight_ce: CE 损失权重
            weight_dice: Dice 损失权重
        """
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


###########################################
# 评估指标
###########################################

def calculate_dice(pred, target, num_classes=2):
    """计算Dice系数"""
    dice_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2 * intersection) / union
        
        # 统一转为 Python float（可能是 tensor.item() 也可能是 plain float）
        dice_scores.append(float(dice) if hasattr(dice, 'item') else dice)
    
    return np.mean(dice_scores)


def calculate_iou(pred, target, num_classes=2):
    """计算IoU（Intersection over Union）"""
    iou_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        # 统一转为 Python float（可能是 tensor.item() 也可能是 plain float）
        iou_scores.append(float(iou) if hasattr(iou, 'item') else iou)
    
    return np.mean(iou_scores)


###########################################
# 训练器类
###########################################

class Trainer:
    """训练器类"""

    def __init__(self, model, train_loader, val_loader, config, device, processed_dir=None):
        """
        Args:
            model: PyTorch模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置字典
            device: 训练设备
            processed_dir: 预处理数据根目录（用于预测后还原到原始空间）
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.processed_dir = processed_dir or config.get('processed_dir', None)
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器（可选）
        self.scheduler = None
        if config.get('use_scheduler', False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )
        
        # 损失函数（可由外部传入，默认使用 CombinedLoss）
        self.criterion = CombinedLoss()
        
        # 梯度裁剪（防止梯度爆炸导致 loss 剧烈抖动）
        self.grad_clip_norm = config.get('grad_clip_norm', 1.0)
        
        # 训练状态
        self.start_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # 保存路径
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 打印批次信息
            if batch_idx % self.config.get('log_interval', 10) == 0:
                print(f"  Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def set_criterion(self, criterion):
        """设置损失函数（允许从外部注入，例如带 class_weight 的 CombinedLoss）"""
        self.criterion = criterion
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        val_loss = 0.0
        dice_scores = []
        iou_scores = []
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch_idx, batch in enumerate(pbar):
                # 获取数据
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # 预测
                preds = torch.argmax(outputs, dim=1)
                
                # 计算指标
                batch_dice = []
                batch_iou = []
                for i in range(images.shape[0]):
                    dice = calculate_dice(preds[i], labels[i])
                    iou = calculate_iou(preds[i], labels[i])
                    batch_dice.append(dice)
                    batch_iou.append(iou)
                
                val_loss += loss.item()
                dice_scores.extend(batch_dice)
                iou_scores.extend(batch_iou)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{np.mean(batch_dice):.4f}'
                })
        
        # 检查输出为 NaN 的情况
        if torch.isnan(outputs).any():
            print(f"  [警告] Epoch {epoch+1} Val 输出包含 NaN！")
        
        avg_loss = val_loss / num_batches
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        
        self.val_metrics.append({
            'epoch': epoch,
            'loss': avg_loss,
            'dice': avg_dice,
            'iou': avg_iou
        })
        
        # 更新学习率调度器
        if self.scheduler is not None:
            self.scheduler.step(avg_dice)
        
        return avg_loss, avg_dice, avg_iou
    
    def get_lr(self):
        """获取当前学习率（用于监控）"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def set_gradient_clip(self, max_norm):
        """设置梯度裁剪阈值"""
        self.grad_clip_norm = max_norm
    
    def save_checkpoint(self, epoch, metric, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        print(f"  ✓ 保存最新检查点到: {latest_path}")
        
        # 保存最优模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 保存最优模型到: {best_path} (Dice: {metric:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        self.train_losses = checkpoint['train_losses']
        self.val_metrics = checkpoint['val_metrics']
        
        print(f"✓ 成功加载检查点: {checkpoint_path}")
        print(f"  从 Epoch {checkpoint['epoch']+1} 继续训练")
        print(f"  历史最优指标: {self.best_metric:.4f}")
        
        return self.start_epoch
    
    def train(self, num_epochs):
        """完整训练流程"""
        print("="*60)
        print(f"开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"设备: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print("="*60)
        
        for epoch in range(self.start_epoch, num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_dice, val_iou = self.validate(epoch)
            
            # 判断是否最优
            is_best = val_dice > self.best_metric
            if is_best:
                self.best_metric = val_dice
            
            # 保存检查点
            self.save_checkpoint(epoch, val_dice, is_best)
            
            # 打印epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch [{epoch+1}/{num_epochs}] 完成 ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f} {'(最优!)' if is_best else ''}")
            print(f"  Val IoU: {val_iou:.4f}")
            print("-"*60)
        
        print("="*60)
        print(f"训练完成 - 最优Dice: {self.best_metric:.4f}")
        print("="*60)


# 测试代码
if __name__ == '__main__':
    # 测试训练流程（需要实际数据和模型）
    print("训练模块已就绪")
    print("使用 main.py 启动完整训练流程")

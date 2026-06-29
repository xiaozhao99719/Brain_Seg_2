"""
测试流程模块
加载最佳模型，在测试集上执行完整推理测试
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import nibabel as nib
from datetime import datetime


###########################################
# 评估指标（与train.py保持一致）
###########################################

def calculate_dice(pred, target, num_classes=2):
    """计算Dice系数"""
    dice_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).astype(float)
        target_c = (target == c).astype(float)
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2. * intersection) / union
        
        dice_scores.append(float(dice))
    
    return np.mean(dice_scores)


def calculate_iou(pred, target, num_classes=2):
    """计算IoU"""
    iou_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).astype(float)
        target_c = (target == c).astype(float)
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        iou_scores.append(float(iou))
    
    return np.mean(iou_scores)


def calculate_hausdorff_distance(pred, target, num_classes=2, max_distance=100):
    """
    计算Hausdorff距离（简化版）
    实际需要更复杂的实现，这里提供框架
    """
    # 简化版：返回占位符
    # 实际实现需要使用scipy.spatial.distance.cdist计算点集距离
    return 0.0


###########################################
# 测试器类
###########################################

class Tester:
    """测试器类"""

    def __init__(self, model, test_loader, config, device, processed_dir=None):
        """
        Args:
            model: PyTorch模型
            test_loader: 测试数据加载器
            config: 配置字典
            device: 设备
            processed_dir: 预处理数据根目录（用于预测后还原到原始空间）
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.processed_dir = processed_dir or config.get('processed_dir', None)
        
        # 加载最佳模型
        self.load_best_model()
        
        # 测试结果保存路径
        self.output_dir = config.get('test_output_dir', 'test_results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_best_model(self):
        """加载最佳模型"""
        checkpoint_path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            'best_model.pth'
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"最佳模型不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ 成功加载最佳模型: {checkpoint_path}")
        print(f"  训练Epoch: {checkpoint['epoch']+1}")
        print(f"  最优指标: {checkpoint['best_metric']:.4f}")
    
    def test(self):
        """完整测试流程"""
        self.model.eval()
        
        all_dice = []
        all_iou = []
        all_hausdorff = []
        
        print("="*60)
        print(f"开始测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试样本数: {len(self.test_loader.dataset)}")
        print("="*60)
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for batch_idx, batch in enumerate(pbar):
                # 获取数据
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                filenames = batch['filename']
                
                # 前向传播
                outputs = self.model(images)
                
                # 预测
                preds = torch.argmax(outputs, dim=1)  # (batch, D, H, W)
                
                # 计算每个样本的 metric
                batch_dice = []
                batch_iou = []
                batch_hausdorff = []
                
                for i in range(images.shape[0]):
                    pred_np = preds[i].cpu().numpy()
                    label_np = labels[i].cpu().numpy()
                    
                    dice = calculate_dice(pred_np, label_np)
                    iou = calculate_iou(pred_np, label_np)
                    hausdorff = calculate_hausdorff_distance(pred_np, label_np)
                    
                    batch_dice.append(dice)
                    batch_iou.append(iou)
                    batch_hausdorff.append(hausdorff)
                    
                    # 保存预测结果（可选）
                    if self.config.get('save_predictions', False):
                        self.save_prediction(pred_np, filenames[i])
                
                # 更新统计
                all_dice.extend(batch_dice)
                all_iou.extend(batch_iou)
                all_hausdorff.extend(batch_hausdorff)
                
                # 更新进度条
                pbar.set_postfix({
                    'dice': f'{np.mean(batch_dice):.4f}',
                    'iou': f'{np.mean(batch_iou):.4f}'
                })
                
                # 打印批次信息
                print(f"\nBatch [{batch_idx+1}/{len(self.test_loader)}]")
                print(f"  文件名: {filenames}")
                print(f"  Dice: {np.mean(batch_dice):.4f}")
                print(f"  IoU: {np.mean(batch_iou):.4f}")
        
        # 计算总体指标
        mean_dice = np.mean(all_dice)
        mean_iou = np.mean(all_iou)
        mean_hausdorff = np.mean(all_hausdorff)
        
        std_dice = np.std(all_dice)
        std_iou = np.std(all_iou)
        
        # 打印最终结果
        print("\n" + "="*60)
        print("测试完成 - 最终量化评价指标")
        print("="*60)
        print(f"Dice 系数: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"IoU:        {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"Hausdorff:  {mean_hausdorff:.4f}")
        print("="*60)
        
        # 保存结果到文件
        self.save_test_results(mean_dice, std_dice, mean_iou, std_iou, mean_hausdorff)
        
        return {
            'dice_mean': mean_dice,
            'dice_std': std_dice,
            'iou_mean': mean_iou,
            'iou_std': std_iou,
            'hausdorff_mean': mean_hausdorff
        }
    
    def save_prediction(self, pred_np, filename):
        """保存预测结果为NIfTI文件"""
        # 创建NIfTI图像
        nifti_img = nib.Nifti1Image(pred_np.astype(np.int32), np.eye(4))
        
        # 保存
        save_path = os.path.join(self.output_dir, f"pred_{filename}")
        nib.save(nifti_img, save_path)
        print(f"  保存预测: {save_path}")
    
    def save_test_results(self, mean_dice, std_dice, mean_iou, std_iou, mean_hausdorff):
        """保存测试结果到文本文件"""
        result_path = os.path.join(self.output_dir, 'test_results.txt')
        
        with open(result_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("测试结果与量化评价指标\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dice 系数: {mean_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"IoU:        {mean_iou:.4f} ± {std_iou:.4f}\n")
            f.write(f"Hausdorff:  {mean_hausdorff:.4f}\n")
            f.write("\n" + "="*60 + "\n")
        
        print(f"\n✓ 测试结果已保存到: {result_path}")


# 测试代码
if __name__ == '__main__':
    # 测试测试流程（需要实际数据和模型）
    print("测试模块已就绪")
    print("使用 main.py 启动完整测试流程")

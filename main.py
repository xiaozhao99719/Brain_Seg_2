"""
主程序入口
统一调度训练/验证/测试全流程，实现断点续跑机制
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime
import hashlib
import json


# 导入自定义模块
from data_load import create_data_loaders, BraTSDataset, preprocess_dataset
from model import get_model
from train import Trainer, CombinedLoss
from test import Tester
from param_set import get_config, print_config


###########################################
# 断点续跑相关函数
###########################################

def calculate_param_hash(config):
    """
    计算参数快照的hash值（用于判断参数是否变化）
    
    Args:
        config: 配置字典
    
    Returns:
        hash字符串
    """
    # 选择需要比对的参数
    key_params = [
        'model', 'in_channels', 'num_classes', 'base_filters',
        'embed_dim', 'num_transformer_layers',
        'target_size', 'target_spacing', 'norm_method',
        'batch_size', 'learning_rate', 'num_epochs'
    ]
    
    # 构建参数快照
    snapshot = {k: config[k] for k in key_params if k in config}
    snapshot_str = json.dumps(snapshot, sort_keys=True)
    
    # 计算MD5
    param_hash = hashlib.md5(snapshot_str.encode()).hexdigest()
    
    return param_hash


def save_checkpoint_with_metadata(model, optimizer, scheduler, epoch, 
                                  best_metric, train_losses, val_metrics,
                                  config, checkpoint_dir, is_best=False):
    """
    保存检查点（包含所有必要信息用于断点续跑）
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        best_metric: 最优指标
        train_losses: 训练损失历史
        val_metrics: 验证指标历史
        config: 配置字典
        checkpoint_dir: 检查点目录
        is_best: 是否是最优模型
    """
    # 计算参数hash
    param_hash = calculate_param_hash(config)
    
    # 获取随机种子状态
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state()
    else:
        torch_cuda_rng_state = None
    
    # 构建检查点
    checkpoint = {
        # 训练状态
        'stage': 'train',  # 当前阶段: train/val/test
        'epoch': epoch,
        'batch_idx': 0,  # 如果需要细粒度续跑，可以保存batch索引
        
        # 模型和优化器状态
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        
        # 指标历史
        'best_metric': best_metric,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        
        # 参数快照和hash
        'param_snapshot': {k: config[k] for k in [
            'model', 'in_channels', 'num_classes', 'base_filters',
            'target_size', 'batch_size', 'learning_rate'
        ] if k in config},
        'param_hash': param_hash,
        
        # 随机种子状态
        'python_rng_state': python_rng_state,
        'numpy_rng_state': numpy_rng_state,
        'torch_rng_state': torch_rng_state,
        'torch_cuda_rng_state': torch_cuda_rng_state,
        
        # 时间戳
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存最新检查点
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"  ✓ 保存最新检查点到: {latest_path}")
    
    # 保存最优模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"  ✓ 保存最优模型到: {best_path}")


def check_checkpoint_compatibility(checkpoint, config):
    """
    检查检查点是否与当前参数兼容（参数是否一致）
    
    Args:
        checkpoint: 加载的检查点
        config: 当前配置
    
    Returns:
        (is_compatible, message)
    """
    # 计算当前参数hash
    current_hash = calculate_param_hash(config)
    saved_hash = checkpoint.get('param_hash', None)
    
    # 比对hash
    if saved_hash is not None and saved_hash != current_hash:
        # 参数已变化，禁止续跑
        saved_snapshot = checkpoint.get('param_snapshot', {})
        current_snapshot = {k: config[k] for k in saved_snapshot.keys() if k in config}
        
        # 找出变化的参数
        changed_params = []
        for k in saved_snapshot:
            if k in config and saved_snapshot[k] != config[k]:
                changed_params.append(f"{k}: {saved_snapshot[k]} → {config[k]}")
        
        msg = f"参数已变化，禁止续跑！\n"
        msg += f"变化的参数:\n"
        for p in changed_params:
            msg += f"  - {p}\n"
        msg += f"如需重新训练，请删除检查点目录或使用 --force_restart"
        
        return False, msg
    
    return True, "参数一致，可以续跑"


def load_checkpoint_for_resume(checkpoint_path, model, optimizer, scheduler, config, device):
    """
    加载检查点并恢复训练状态
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        config: 当前配置
        device: 目标设备（torch.device）
    
    Returns:
        (start_epoch, best_metric, train_losses, val_metrics)
    """
    # 加载检查点：先加载到 CPU，再由调用方将模型/优化器移到目标设备
    # 这样可以避免「检查点保存在 GPU、但当前模型在 CPU」或反之的不匹配问题
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 检查参数兼容性
    is_compatible, msg = check_checkpoint_compatibility(checkpoint, config)
    if not is_compatible:
        raise ValueError(msg)
    
    # 恢复模型和优化器状态
    # 先加载到 CPU，再统一移到目标设备，避免设备不匹配
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 优化器状态张量需要手动移到正确设备
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 恢复随机种子状态
    if 'python_rng_state' in checkpoint:
        random.setstate(checkpoint['python_rng_state'])
    if 'numpy_rng_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_rng_state'])
    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    if 'torch_cuda_rng_state' in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])
    
    # 返回训练状态
    start_epoch = checkpoint['epoch'] + 1
    best_metric = checkpoint['best_metric']
    train_losses = checkpoint.get('train_losses', [])
    val_metrics = checkpoint.get('val_metrics', [])
    
    print(f"✓ 成功加载检查点: {checkpoint_path}")
    print(f"  从 Epoch {start_epoch} 继续训练")
    print(f"  历史最优指标: {best_metric:.4f}")
    print(f"  保存时间: {checkpoint.get('timestamp', 'unknown')}")
    
    return start_epoch, best_metric, train_losses, val_metrics


###########################################
# 主程序
###########################################

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """主函数"""
    # 获取配置
    config = get_config()
    
    # 打印配置
    print_config(config)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 创建检查点目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # 设备
    device = torch.device(config['device'])
    print(f"\n使用设备: {device}")
    
    # ===========================================
    # 模式0: 仅预处理
    # ===========================================
    if config['mode'] == 'preprocess':
        print("\n" + "="*60)
        print("开始预处理流程")
        print("="*60)

        # 预处理并行进程数（默认=CPU核心数）
        from multiprocessing import cpu_count
        prep_workers = config.get('preprocess_workers') or min(cpu_count(), 8)
        print(f"预处理并行进程数: {prep_workers}")

        for split in ['train', 'val', 'test']:
            preprocess_dataset(
                data_root=config['data_dir'],
                processed_root=config['processed_dir'],
                split=split,
                target_spacing=tuple(config['target_spacing']),
                target_size=tuple(config['target_size']),
                target_label=config['target_label'],
                num_workers=prep_workers
            )
        print("\n" + "="*60)
        print("预处理全部完成！")
        print("="*60)
        sys.exit(0)

    # ===========================================
    # 模式1: 训练
    # ===========================================
    if config['mode'] in ['train', 'train_test']:
        print("\n" + "="*60)
        print("开始训练流程")
        print("="*60)
        
        # 创建数据加载器
        preprocess_config = {
            'target_spacing': config['target_spacing'],
            'target_size': config['target_size'],
            'target_label': config['target_label'],
            'norm_method': config['norm_method'],
            'num_workers': config.get('preprocess_workers', config['num_workers'])
        }

        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=config['data_dir'],
            processed_root=config['processed_dir'],
            preprocess_config=preprocess_config,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            augmentation_config=config.get('augmentation_config'),
            force_reprocess=config.get('force_reprocess', False),
            use_gpu_augment=config.get('use_gpu_augment', True)
        )
        
        # 创建模型
        img_size = config['target_size'][0]  # 取target_size的第一个维度
        model = get_model(
            config['model'],
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            base_filters=config['base_filters'],
            embed_dim=config.get('embed_dim', 512),
            num_transformer_layers=config.get('num_transformer_layers', 4),
            img_size=img_size
        )
        
        # 【关键】将模型移到目标设备，必须在创建优化器之前完成
        model = model.to(device)
        
        print(f"\n模型: {config['model']}")
        print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"模型已移至: {next(model.parameters()).device}")
        
        # 创建优化器和调度器
        # 【关键】为 TransBTS 设置分层学习率：Transformer 层用较大 LR，CNN 骨干用较小 LR
        param_groups = []
        if config['model'] == 'transbts':
            # Transformer 参数
            transformer_params = []
            cnn_params = []
            for name, param in model.named_parameters():
                if 'transformer' in name or 'pos_embedding' in name or 'bottleneck_conv' in name or 'proj_back' in name:
                    transformer_params.append(param)
                else:
                    cnn_params.append(param)
            param_groups.append({'params': cnn_params, 'lr': config.get('lr_backbone', config['learning_rate'])})
            param_groups.append({'params': transformer_params, 'lr': config.get('lr_transformer', config['learning_rate'] * 2)})
            print(f"  分层学习率: CNN={config.get('lr_backbone', config['learning_rate']):.1e}, Transformer={config.get('lr_transformer', config['learning_rate']*2):.1e}")
        else:
            param_groups = [{'params': model.parameters(), 'lr': config['learning_rate']}]
        
        optimizer = torch.optim.Adam(
            param_groups,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        scheduler = None
        if config['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=5, factor=0.5
            )
        
        # 【关键】创建带类别权重的损失函数
        # 前景（病灶）通常远少于背景，需要给更大权重
        class_weights = torch.tensor([
            config.get('class_weight_background', 1.0),
            config.get('class_weight_foreground', 10.0)
        ], device=device)
        
        criterion = CombinedLoss(
            weight=class_weights,
            weight_ce=config.get('ce_weight', 0.3),
            weight_dice=config.get('dice_weight', 0.7)
        )
        print(f"  损失权重: CE={config.get('ce_weight', 0.3)}, Dice={config.get('dice_weight', 0.7)}")
        print(f"  类别权重: 背景={class_weights[0]:.1f}, 前景={class_weights[1]:.1f}")
        
        # 检查是否需要断点续跑
        checkpoint_path = os.path.join(config['checkpoint_dir'], 'latest_checkpoint.pth')
        start_epoch = 0
        best_metric = 0.0
        train_losses = []
        val_metrics = []
        
        if os.path.exists(checkpoint_path) and not config['force_restart'] and config['auto_resume']:
            # 自动续跑
            try:
                start_epoch, best_metric, train_losses, val_metrics = \
                    load_checkpoint_for_resume(checkpoint_path, model, optimizer, scheduler, config, device)
            except ValueError as e:
                print(f"\n⚠ 无法续跑: {e}")
                print("将从头开始训练...")
                start_epoch = 0
                best_metric = 0.0
                train_losses = []
                val_metrics = []
        else:
            if config['force_restart']:
                print("\n⚠ 强制重新开始训练（--force_restart）")
            else:
                print("\n开始全新训练...")
        
        # 创建训练器
        trainer = Trainer(model, train_loader, val_loader, config, device,
                          processed_dir=config.get('processed_dir'))
        
        # 【关键】注入优化器、调度器、损失函数、梯度裁剪
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
        trainer.set_criterion(criterion)
        trainer.set_gradient_clip(config.get('grad_clip_norm', 1.0))
        print(f"  梯度裁剪阈值: {config.get('grad_clip_norm', 1.0)}")
        
        # 训练循环
        for epoch in range(start_epoch, config['num_epochs']):
            print(f"\n{'='*60}")
            print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
            print(f"{'='*60}")
            
            # 训练一个epoch
            train_loss = trainer.train_epoch(epoch)
            
            # 验证
            val_loss, val_dice, val_iou = trainer.validate(epoch)
            
            # 判断是否最优
            is_best = val_dice > best_metric
            if is_best:
                best_metric = val_dice
            
            # 保存检查点（带完整元数据）
            save_checkpoint_with_metadata(
                model, optimizer, scheduler, epoch,
                best_metric, trainer.train_losses, trainer.val_metrics,
                config, config['checkpoint_dir'], is_best
            )
            
            # 打印epoch结果
            print(f"\nEpoch [{epoch+1}/{config['num_epochs']}] 完成")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f} {' (最优!)' if is_best else ''}")
            print(f"  Val IoU: {val_iou:.4f}")
        
        print("\n" + "="*60)
        print(f"训练完成 - 最优Dice: {best_metric:.4f}")
        print("="*60)
    
    # ===========================================
    # 模式2: 测试
    # ===========================================
    if config['mode'] in ['test', 'train_test']:
        print("\n" + "="*60)
        print("开始测试流程")
        print("="*60)
        
        # 创建数据加载器（仅测试集）
        preprocess_config = {
            'target_spacing': config['target_spacing'],
            'target_size': config['target_size'],
            'target_label': config['target_label'],
            'norm_method': config['norm_method']
        }
        
        # 仅创建测试集（从预处理后的processed文件夹加载）
        test_dataset = BraTSDataset(
            processed_root=config['processed_dir'],
            split='test',
            augment=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.get('test_batch_size', 1),
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            drop_last=False
        )
        
        # 创建模型
        img_size = config['target_size'][0]
        model = get_model(
            config['model'],
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            base_filters=config['base_filters'],
            embed_dim=config.get('embed_dim', 512),
            num_transformer_layers=config.get('num_transformer_layers', 4),
            img_size=img_size
        )
        
        # 【关键】将模型移到目标设备
        model = model.to(device)
        
        # 创建测试器
        tester = Tester(model, test_loader, config, device,
                      processed_dir=config.get('processed_dir'))
        
        # 执行测试
        results = tester.test()
        
        print("\n" + "="*60)
        print("测试完成")
        print(f"Dice: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
        print(f"IoU:  {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
        print("="*60)
    
    print(f"\n✓ 全部流程完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

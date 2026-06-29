"""
参数配置模块
集中管理全项目所有参数，使用argparse实现参数配置
"""

import argparse
import torch


def get_args():
    """获取所有命令行参数"""

    parser = argparse.ArgumentParser(
        description='3D医学影像分割项目 - BrainSeg',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ===========================================
    # 基础配置
    # ===========================================
    parser.add_argument('--mode', type=str, default='train_test',
                        choices=['train', 'test', 'train_test', 'preprocess'],
                        help='运行模式: train=仅训练, test=仅测试, train_test=训练后测试, preprocess=仅预处理')

    # 原始数据根目录（4个模态+1个seg的nii.gz文件所在位置）
    parser.add_argument('--data_dir', type=str,
                        default=r'D:\PythonProgram\BrainSeg\data',
                        help='原始数据根目录（包含train/val/test子目录，每个子目录下是患者ID文件夹）')

    # 预处理数据根目录（预处理后.npy文件的存放位置）
    parser.add_argument('--processed_dir', type=str,
                        default=r'D:\PythonProgram\BrainSeg\processed',
                        help='预处理数据根目录（processed文件夹路径，train/val/test/images/*.npy 和 labels/*.npy）')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点保存目录')

    parser.add_argument('--test_output_dir', type=str, default='test_results',
                        help='测试结果输出目录')

    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
                        help='训练设备')

    # ===========================================
    # 预处理参数
    # ===========================================
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='目标体素间距 (x, y, z)，单位mm')

    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 128],
                        help='目标体素尺寸 (D, H, W)，即深度/高度/宽度')

    # 病灶标签值：指定分割任务的目标标签
    # 例如: 1, 2, 4（BraTS标准标签）
    # 该参数将指定标签映射为1（前景），其他所有值映射为0（背景）
    parser.add_argument('--target_label', type=int, default=1,
                        help='目标病灶标签值；seg中等于该值的像素映射为1（前景），其他所有值映射为0（背景）')

    # 标准化方法：非零区域z-score标准化（排除背景干扰）
    parser.add_argument('--norm_method', type=str, default='z-score-nonzero',
                        choices=['z-score', 'z-score-nonzero', 'min-max'],
                        help='图像强度标准化方法；z-score-nonzero：对非零区域做z-score标准化')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader的进程数')

    # 是否强制重新预处理（删除processed文件夹后重新处理）
    parser.add_argument('--force_reprocess', action='store_true',
                        help='强制重新预处理（删除已有的processed数据，重新处理所有数据）')

    # ===========================================
    # 模型参数
    # ===========================================
    parser.add_argument('--model', type=str, default='transbts',
                        choices=['transbts', 'kiunet', 'unet3d'],
                        help='模型架构: transbts=Transformer+CNN混合, kiunet=空洞卷积U型, unet3d=标准3D U-Net')

    parser.add_argument('--in_channels', type=int, default=4,
                        help='输入通道数（模态数，固定为4：t1/t1ce/t2/flair）')

    parser.add_argument('--num_classes', type=int, default=2,
                        help='输出类别数（二分类：背景=0 + 目标病灶=1）')

    parser.add_argument('--base_filters', type=int, default=32,
                        help='基础滤波器数量（U-Net第一层通道数）')

    # TransBTS特有参数
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='TransBTS Transformer嵌入维度')

    parser.add_argument('--num_transformer_layers', type=int, default=4,
                        help='TransBTS Transformer层数')

    # ===========================================
    # 训练参数
    # ===========================================
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小')

    parser.add_argument('--num_epochs', type=int, default=400,
                        help='训练epoch数')

    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率（TransBTS建议5e-5，纯CNN可用1e-4）')

    parser.add_argument('--lr_backbone', type=float, default=5e-5,
                        help='CNN骨干网络学习率（可设为比Transformer层小）')

    parser.add_argument('--lr_transformer', type=float, default=1e-4,
                        help='Transformer层学习率（通常比CNN层大）')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减（适度增大以抑制过拟合）')

    parser.add_argument('--use_scheduler', action='store_true', default=True,
                        help='是否使用学习率调度器（ReduceLROnPlateau），默认开启')

    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='梯度裁剪阈值（防止梯度爆炸，设为0则禁用）')

    parser.add_argument('--ce_weight', type=float, default=0.3,
                        help='CE损失权重（降低以减少类别不平衡影响）')

    parser.add_argument('--dice_weight', type=float, default=0.7,
                        help='Dice损失权重（提高以直接优化Dice指标）')

    parser.add_argument('--class_weight_background', type=float, default=1.0,
                        help='背景类别权重（CrossEntropy）')

    parser.add_argument('--class_weight_foreground', type=float, default=10.0,
                        help='前景类别权重（病灶通常远少于背景，建议5-20）')

    parser.add_argument('--log_interval', type=int, default=50,
                        help='每隔N个batch打印一次训练信息（降低频率，减少IO开销）')

    # ===========================================
    # 数据增强参数（3D增强）
    # ===========================================
    parser.add_argument('--augment', action='store_true', default=True,
                        help='是否启用3D数据增强（仅训练集）')

    parser.add_argument('--flip_prob', type=float, default=0.3,
                        help='随机翻转概率（沿3个轴），降低以增强训练稳定性')

    parser.add_argument('--rotation_range', type=float, default=5,
                        help='随机3D旋转角度范围（度），沿x/y/z轴随机旋转，降低以减少标签错位风险')

    parser.add_argument('--noise_std', type=float, default=0.005,
                        help='高斯噪声标准差（降低以避免破坏精细病灶信号）')

    parser.add_argument('--gamma_range', type=float, nargs=2, default=[0.9, 1.1],
                        help='伽马校正范围 (min, max)，缩小范围以稳定训练')

    # ===========================================
    # 测试参数
    # ===========================================
    parser.add_argument('--save_predictions', action='store_true',
                        help='是否保存测试集预测结果')

    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='测试时批次大小（通常为1）')

    # ===========================================
    # 可视化参数
    # ===========================================
    parser.add_argument('--vis_num_samples', type=int, default=4,
                        help='可视化随机抽样的样本数')

    parser.add_argument('--vis_modality', type=int, default=3,
                        help='可视化时显示的模态索引（0=t1, 1=t1ce, 2=t2, 3=flair）')

    parser.add_argument('--vis_slice_mode', type=str, default='middle',
                        choices=['middle', 'max_foreground'],
                        help='可视化切片选择：middle=中间切片, max_foreground=前景最多的切片')

    # ===========================================
    # 断点续跑参数
    # ===========================================
    parser.add_argument('--auto_resume', action='store_true', default=True,
                        help='自动检测并续跑（无需手动指定）')

    parser.add_argument('--force_restart', action='store_true',
                        help='强制重新开始训练（忽略已有检查点）')

    # ===========================================
    # 其他参数
    # ===========================================
    parser.add_argument('--seed', type=int, default=123,
                        help='随机种子')

    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='是否pin memory（加速GPU训练）')

    parser.add_argument('--drop_last', action='store_true', default=True,
                        help='是否丢弃不完整的最后一批')

    # ===========================================
    # CUDA加速参数
    # ===========================================
    parser.add_argument('--use_gpu_augment', action='store_true', default=True,
                        help='是否使用GPU加速数据增强（训练时推荐开启，DataLoader批次在GPU上做增强）')

    parser.add_argument('--preprocess_workers', type=int, default=None,
                        help='预处理并行进程数（默认=CPU核心数，设为0则串行）')

    return parser.parse_args()


def get_config():
    """获取配置字典（从args转换）"""
    args = get_args()

    # 转换为字典
    config = vars(args)

    # 设备自动选择
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 将列表参数转换为元组
    config['target_spacing'] = tuple(config['target_spacing'])
    config['target_size'] = tuple(config['target_size'])
    config['gamma_range'] = tuple(config['gamma_range'])

    # num_classes固定为2（二分类：背景0 + 前景1）
    config['num_classes'] = 2

    # 数据增强配置字典
    config['augmentation_config'] = {
        'flip_prob': config['flip_prob'],
        'rotation_range': config['rotation_range'],
        'noise_std': config['noise_std'],
        'gamma_range': config['gamma_range']
    }

    # 预处理配置字典
    config['preprocess_config'] = {
        'target_spacing': config['target_spacing'],
        'target_size': config['target_size'],
        'target_label': config['target_label'],
        'force_reprocess': config['force_reprocess']
    }

    return config


def print_config(config):
    """打印配置信息"""
    print("="*60)
    print("当前配置参数")
    print("="*60)

    categories = {
        '基础配置': ['mode', 'data_dir', 'processed_dir', 'checkpoint_dir',
                     'test_output_dir', 'device'],
        '预处理参数': ['target_spacing', 'target_size', 'target_label',
                       'norm_method', 'force_reprocess', 'num_workers'],
        '模型参数': ['model', 'in_channels', 'num_classes', 'base_filters',
                     'embed_dim', 'num_transformer_layers'],
        '训练参数': ['batch_size', 'num_epochs', 'learning_rate',
                     'weight_decay', 'use_scheduler', 'log_interval'],
        '数据增强（3D）': ['augment', 'flip_prob', 'rotation_range',
                           'noise_std', 'gamma_range'],
        '测试参数': ['save_predictions', 'test_batch_size'],
        '可视化参数': ['vis_num_samples', 'vis_modality', 'vis_slice_mode'],
        '断点续跑': ['auto_resume', 'force_restart'],
        '其他': ['seed', 'pin_memory', 'drop_last'],
        'CUDA加速': ['use_gpu_augment', 'preprocess_workers']
    }

    for category, params in categories.items():
        print(f"\n[{category}]")
        for param in params:
            if param in config and param not in ('augmentation_config', 'preprocess_config'):
                value = config[param]
                print(f"  {param:25s}: {value}")

    print("\n" + "="*60)


# 测试代码
if __name__ == '__main__':
    config = get_config()
    print_config(config)

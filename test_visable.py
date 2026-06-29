"""
独立可视化脚本
从预处理后的测试集随机选取样本，
对预测掩码做逆变换（还原到原始图像空间），最终输出原始MRI切片对比图

流程：
  1. 从 processed/test 加载预处理后的测试数据
  2. 加载原始MRI图像路径元数据
  3. 模型推理 → 得到预处理空间的预测掩码
  4. 使用SimpleITK将预测掩码逆变换回原始图像空间（原始尺寸+spacing）
  5. 可视化：原始MRI + 原始label + 还原后的预测label
  6. 输出为与原始样本seg文件同格式的.nii.gz
"""

import os
import sys
import json
import random
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# 项目路径
PROJECT_DIR = r'D:\PythonProgram\BrainSeg'
sys.path.insert(0, PROJECT_DIR)
from model import get_model
from data_load import MODALITY_SUFFIXES


# ============================================================
# 内部配置（修改这里以适配您的环境）
# ============================================================

BEST_MODEL_PATH = r'D:\PythonProgram\BrainSeg\checkpoints\best_model.pth'
PROCESSED_ROOT = r'D:\PythonProgram\BrainSeg\processed'  # 预处理数据根目录
RAW_DATA_ROOT = r'D:\PythonProgram\BrainSeg\data'       # 原始数据根目录
OUTPUT_DIR = r'D:\PythonProgram\BrainSeg\vis_results'

# 模型配置（需与训练时一致）
MODEL_NAME = 'transbts'
IN_CHANNELS = 4
NUM_CLASSES = 2
BASE_FILTERS = 32
IMG_SIZE = 128  # TransBTS参数，需与训练时一致

# 可视化配置
NUM_SAMPLES = 4
MODALITY_INDEX = 3        # 显示哪个模态（0=t1, 1=t1ce, 2=t2, 3=flair）
SLICE_MODE = 'middle'     # 'middle'=中间切片，'max_foreground'=前景最多切片

# 颜色配置
GT_COLOR = [0, 1, 0, 1]   # 原始label (绿)
PRED_COLOR = [1, 0, 0, 1] # 预测label (红)
ALPHA = 0.5               # 叠加透明度

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 工具函数
# ============================================================

def load_norm_stats(processed_root):
    """加载标准化统计量和预处理参数"""
    stats_path = os.path.join(processed_root, 'norm_stats.json')
    with open(stats_path) as f:
        return json.load(f)


def resample_to_original(pred_array, orig_nii_path, target_spacing, target_size):
    """
    将预处理空间的预测掩码逆变换回原始图像空间。

    使用SimpleITK：
      1. 将预测数组转为SimpleITK图像
      2. 以原始图像为参考（reference image）重采样到原始尺寸+spacing

    Args:
        pred_array: 预测数组 (D, H, W)，值为0或1
        orig_nii_path: 原始MRI图像路径（用于获取参考空间信息）
        target_spacing: 预处理时的目标spacing
        target_size: 预处理时的目标尺寸
    Returns:
        resampled_pred: 还原后的预测数组 (D_orig, H_orig, W_orig)
    """
    # 加载原始图像（用于获取参考空间）
    orig_img = sitk.ReadImage(orig_nii_path)
    orig_spacing = orig_img.GetSpacing()
    orig_size = orig_img.GetSize()
    orig_direction = orig_img.GetDirection()
    orig_origin = orig_img.GetOrigin()

    # 创建预测的SimpleITK图像
    # pred_array shape: (D, H, W)，SimpleITK需要 (W, H, D) 即 (x, y, z)
    pred_sitk = sitk.GetImageFromArray(pred_array.astype(np.float32))
    pred_sitk.SetSpacing(target_spacing)  # 预处理空间的spacing

    # 创建Resample滤波器，以原始图像为参考
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(orig_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 标签用最近邻
    resampler.SetDefaultPixelValue(0)
    resampled_img = resampler.Execute(pred_sitk)

    resampled = sitk.GetArrayFromImage(resampled_img)
    return resampled


def select_slice(volume_3d, mode='middle'):
    """选择3D体的2D切片用于可视化"""
    if mode == 'middle':
        return volume_3d.shape[0] // 2
    elif mode == 'max_foreground':
        counts = [np.sum(volume_3d[i] > 0) for i in range(volume_3d.shape[0])]
        return int(np.argmax(counts))
    else:
        return volume_3d.shape[0] // 2


def visualize_comparison(image_slice, gt_slice, pred_slice,
                        sample_id, slice_idx, output_path):
    """
    绘制三列对比图：
      列1: 原始MRI图像（灰度）
      列2: 原始label叠加（绿色）
      列3: 还原后的预测label叠加（红色）
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # 归一化图像切片到[0,1]
    vmin, vmax = image_slice.min(), image_slice.max()
    if vmax > vmin:
        img_norm = (image_slice - vmin) / (vmax - vmin)
    else:
        img_norm = image_slice

    # --- 列1: 原始MRI ---
    axes[0].imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'{sample_id}\n原始MRI (slice {slice_idx})', fontsize=11)
    axes[0].axis('off')

    # --- 列2: 原始label叠加 ---
    axes[1].imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    if np.sum(gt_slice > 0) > 0:
        gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
        axes[1].imshow(gt_mask, cmap='Greens', alpha=ALPHA, vmin=0, vmax=1)
    axes[1].set_title('原始 Label (绿色)', fontsize=11)
    axes[1].axis('off')
    gt_patch = mpatches.Patch(color='green', label=f'原始Label', alpha=ALPHA)
    axes[1].legend(handles=[gt_patch], loc='upper right')

    # --- 列3: 预测label叠加 ---
    axes[2].imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    if np.sum(pred_slice > 0) > 0:
        pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[2].imshow(pred_mask, cmap='Reds', alpha=ALPHA, vmin=0, vmax=1)
    axes[2].set_title('预测 Label (红色, 还原后)', fontsize=11)
    axes[2].axis('off')
    pred_patch = mpatches.Patch(color='red', label=f'预测Label', alpha=ALPHA)
    axes[2].legend(handles=[pred_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def load_raw_image_for_visualization(metadata, sample_id, processed_root):
    """
    加载原始MRI图像和seg用于可视化。

    从sample_metadata.json获取原始文件路径，
    加载原始图像数据并直接返回（不做预处理，用于参考空间信息）。

    Returns:
        orig_image_data: 原始MRI图像数据
        orig_spacing: 原始spacing
        orig_seg_data: 原始seg数据
    """
    # 读取元数据
    meta_path = os.path.join(processed_root, 'test', 'sample_metadata.json')
    with open(meta_path) as f:
        all_metadata = json.load(f)

    sample_meta = next((m for m in all_metadata if m['sample_id'] == sample_id), None)
    if sample_meta is None:
        raise ValueError(f"未找到样本元数据: {sample_id}")

    # 加载原始图像（取flair模态作为显示参考）
    modal_files = sample_meta['modal_files']
    modality_order = ['t1', 't1ce', 't2', 'flair']
    modal_to_use = modality_order[MODALITY_INDEX]

    if modal_to_use not in modal_files:
        modal_to_use = list(modal_files.keys())[0]  # fallback

    orig_path = modal_files[modal_to_use]
    orig_nib = nib.load(orig_path)
    orig_image_data = orig_nib.get_fdata()
    orig_spacing = orig_nib.header.get_zooms()[:3]

    # 加载原始seg
    seg_path = sample_meta['seg_file']
    seg_nib = nib.load(seg_path)
    orig_seg_data = seg_nib.get_fdata()

    return orig_image_data, orig_spacing, orig_seg_data, orig_path


def normalize_for_display(array):
    """将数组归一化到[0,1]用于显示"""
    vmin, vmax = array.min(), array.max()
    if vmax > vmin:
        return (array - vmin) / (vmax - vmin)
    return array


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*60)
    print(f"test_visable.py - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. 检查路径
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"✗ 错误: 最佳模型不存在: {BEST_MODEL_PATH}")
        return

    # 加载预处理统计量（包含spacing/size信息）
    norm_stats = load_norm_stats(PROCESSED_ROOT)
    target_spacing = tuple(norm_stats['target_spacing'])
    target_size = tuple(norm_stats['target_size'])
    modality_order = norm_stats.get('modality_order', ['t1', 't1ce', 't2', 'flair'])
    print(f"预处理参数: spacing={target_spacing}, size={target_size}")

    # 2. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ 输出目录: {OUTPUT_DIR}")

    # 3. 加载模型
    print(f"\n加载模型: {MODEL_NAME}")
    model = get_model(
        MODEL_NAME,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_filters=BASE_FILTERS,
        img_size=IMG_SIZE if MODEL_NAME == 'transbts' else target_size[0]
    )
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print(f"✓ 模型加载成功 (Epoch: {checkpoint['epoch']+1})")

    # 4. 加载预处理后的测试数据和元数据
    test_image_dir = os.path.join(PROCESSED_ROOT, 'test', 'images')
    test_label_dir = os.path.join(PROCESSED_ROOT, 'test', 'labels')

    if not os.path.exists(test_image_dir):
        print(f"✗ 错误: 预处理测试数据不存在: {test_image_dir}")
        print("  请先运行预处理（python main.py --mode preprocess）")
        return

    # 加载sample_metadata
    meta_path = os.path.join(PROCESSED_ROOT, 'test', 'sample_metadata.json')
    with open(meta_path) as f:
        sample_metadata = json.load(f)
    metadata_dict = {m['sample_id']: m for m in sample_metadata}

    # 加载标准化统计量（已在前面加载过，无需重复）
    # norm_stats 已在前面定义
    if 'norm_stats' not in locals():
        norm_stats = load_norm_stats(PROCESSED_ROOT)
    norm_means = [s['mean'] for s in norm_stats['modality_stats']]
    norm_stds = [s['std'] for s in norm_stats['modality_stats']]

    # 获取所有样本文件
    all_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith('.npy')])

    if len(all_files) == 0:
        print(f"✗ 错误: 测试集中没有预处理数据")
        return

    # 5. 随机选取样本
    if NUM_SAMPLES > len(all_files):
        selected_files = all_files
    else:
        random.seed(42)
        selected_files = random.sample(all_files, NUM_SAMPLES)

    print(f"\n随机选取 {len(selected_files)} 个样本")
    print(f"显示模态: {MODALITY_INDEX} ({modality_order[MODALITY_INDEX]})")
    print(f"切片选择: {SLICE_MODE}")

    # 6. 处理每个样本
    for idx, filename in enumerate(selected_files):
        sample_id = filename.replace('.npy', '')
        print(f"\n[{idx+1}/{len(selected_files)}] 处理: {sample_id}")

        # --- 加载预处理后的数据 ---
        image_data = np.load(os.path.join(test_image_dir, filename)).astype(np.float32)  # (D,H,W,C)
        label_data = np.load(os.path.join(test_label_dir, filename)).astype(np.int32)    # (D,H,W)

        # --- 转换为PyTorch格式并推理 ---
        image_tensor = torch.from_numpy(image_data).float()
        image_tensor = image_tensor.permute(3, 0, 1, 2)  # (D,H,W,C) -> (C,D,H,W)
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1)  # (1, D, H, W)

        pred_np = pred.cpu().numpy()[0]  # (D, H, W)

        # --- 获取原始图像用于还原空间 ---
        sample_meta = metadata_dict.get(sample_id, None)
        if sample_meta is None:
            print(f"  ⚠ 跳过: 未找到元数据")
            continue

        modal_files = sample_meta['modal_files']
        seg_file = sample_meta['seg_file']

        # 使用flair模态作为参考
        modal_name = modality_order[MODALITY_INDEX] if modality_order[MODALITY_INDEX] in modal_files \
                     else list(modal_files.keys())[0]
        orig_nii_path = modal_files[modal_name]

        # --- 逆变换：还原到原始空间 ---
        try:
            resampled_pred = resample_to_original(
                pred_np, orig_nii_path, target_spacing, target_size
            )
            print(f"  预测尺寸: {pred_np.shape} -> 还原后: {resampled_pred.shape}")
        except Exception as e:
            print(f"  ⚠ 逆变换失败: {e}")
            continue

        # --- 加载原始图像和seg用于可视化 ---
        orig_img_nib = nib.load(orig_nii_path)
        orig_image = orig_img_nib.get_fdata()       # 原始MRI
        orig_spacing = orig_img_nib.header.get_zooms()[:3]

        orig_seg_nib = nib.load(seg_file)
        orig_seg = orig_seg_nib.get_fdata()         # 原始seg

        # --- 标签二分类映射（与预处理一致）---
        target_label = norm_stats.get('target_label', 1)
        orig_seg_binary = (orig_seg == target_label).astype(np.int32)

        # --- 选择切片 ---
        # 使用原始seg的二值化版本来选择切片
        use_for_slice = orig_seg_binary if np.sum(orig_seg_binary) > 0 else resampled_pred
        slice_idx = select_slice(use_for_slice, SLICE_MODE)

        # --- 提取切片 ---
        # 原始MRI切片: 直接从原始图像提取
        orig_slice = orig_image[slice_idx, :, :]
        # 原始label切片
        gt_slice = orig_seg_binary[slice_idx, :, :]
        # 还原后的预测切片
        pred_slice = (resampled_pred[slice_idx, :, :] > 0.5).astype(np.int32)

        # --- 可视化 ---
        output_png = os.path.join(OUTPUT_DIR, f'vis_{sample_id}_slice{slice_idx}.png')
        visualize_comparison(
            normalize_for_display(orig_slice),
            gt_slice,
            pred_slice,
            sample_id,
            slice_idx,
            output_png
        )
        print(f"  ✓ 保存可视化: {output_png}")

        # --- 保存还原后的预测为NIfTI ---
        # 创建与原始seg同格式的NIfTI（与原始图像同affine/spacing）
        pred_nii = nib.Nifti1Image(resampled_pred.astype(np.float32), orig_img_nib.affine)
        pred_nii.header.set_zooms(orig_spacing)
        output_nii = os.path.join(OUTPUT_DIR, f'pred_{sample_id}_original_space.nii.gz')
        nib.save(pred_nii, output_nii)
        print(f"  ✓ 保存预测NIfTI: {output_nii}")

    print("\n" + "="*60)
    print(f"可视化完成！结果保存在: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()

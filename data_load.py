"""
数据预处理与加载模块
包含：数据校验、预处理、增强、PyTorch数据加载
流程：原始数据 -> 预处理保存 -> 加载预处理数据 -> 训练/验证/测试

【CUDA加速说明】
- 预处理阶段：使用多进程并行 + 可选 cupy GPU加速
- 训练阶段数据增强：GPU张量批次运算
"""

import os
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from scipy import ndimage

# ============================================================
# CUDA 支持检测（使用 PyTorch 原生 CUDA，无需额外安装）
# ============================================================
def _get_cuda_device():
    """获取可用的 CUDA 设备，返回 device string 或 None"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return None

CUDA_DEVICE = _get_cuda_device()
if CUDA_DEVICE is not None:
    print(f"[CUDA] GPU加速已启用，使用设备: {torch.cuda.get_device_name(0)}")
else:
    print("[CUDA] 无可用GPU，将使用 CPU")

# ============================================================
# 模态顺序定义（4个模态的固定顺序）
# ============================================================
MODALITY_SUFFIXES = ['t1', 't1ce', 't2', 'flair']  # 对应通道 0,1,2,3
MODALITY_CHANNEL_MAP = {s: i for i, s in enumerate(MODALITY_SUFFIXES)}


# ============================================================
# GPU加速工具函数
# ============================================================

def _to_gpu(array, device='cuda'):
    """将 numpy 数组转换为 GPU 张量"""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    return array


def _to_numpy(array):
    """将 torch 数组转回 numpy"""
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array


# ============================================================
# 数据扫描：发现原始数据目录中的所有样本
# ============================================================

def scan_raw_data(data_root, split):
    """
    扫描原始数据目录，收集所有样本路径。

    目录结构：
        data_root/
            train/
                {patient_id}/
                    BraTS2021_{id}_{modality}.nii.gz
                    BraTS2021_{id}_seg.nii.gz
            val/
                ...
            test/
                ...

    Returns:
        List[dict], 每个元素包含:
            sample_id: str, 样本唯一标识
            patient_id: str, 患者ID
            modal_files: dict, {modality_name: abs_path}
            seg_file: str, seg文件路径
    """
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"数据目录不存在: {split_dir}")

    samples = []
    for patient_id in os.listdir(split_dir):
        patient_dir = os.path.join(split_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        nii_files = [f for f in os.listdir(patient_dir) if f.endswith('.nii.gz')]
        if len(nii_files) == 0:
            continue

        # 解析模态文件：BraTS2021_{id}_{modality}.nii.gz
        modal_files = {}
        seg_file = None
        sample_id = None

        for fname in nii_files:
            if fname.endswith('_seg.nii.gz'):
                seg_file = os.path.join(patient_dir, fname)
                base = fname.replace('_seg.nii.gz', '')
                sample_id = base
            else:
                parts = fname.replace('.nii.gz', '').split('_')
                if len(parts) >= 2:
                    modality = parts[-1]
                    if modality in MODALITY_CHANNEL_MAP:
                        modal_files[modality] = os.path.join(patient_dir, fname)
                        if sample_id is None:
                            sample_id = '_'.join(parts[:-1])

        if seg_file is not None and len(modal_files) == 4 and sample_id is not None:
            samples.append({
                'sample_id': sample_id,
                'patient_id': patient_id,
                'modal_files': modal_files,
                'seg_file': seg_file
            })
        else:
            print(f"  [警告] 跳过不完整样本 {patient_id}: "
                  f"模态文件={list(modal_files.keys())}, seg={'有' if seg_file else '无'}")

    samples.sort(key=lambda x: x['sample_id'])
    return samples


# ============================================================
# 预处理核心类（GPU加速版）
# ============================================================

class MedicalImagePreprocessor:
    """
    医学影像预处理（SimpleITK版 + 可选CuPy GPU加速）
    功能：重采样 → 尺寸调整 → 标签映射 → 非零区域标准化
    """

    def __init__(self, target_spacing=(1.0, 1.0, 1.0), target_size=(128, 128, 128)):
        self.target_spacing = target_spacing
        self.target_size = target_size

    def load_nifti_simpleitk(self, file_path):
        """使用SimpleITK加载NIfTI"""
        img = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(img)
        spacing = img.GetSpacing()
        direction = img.GetDirection()
        origin = img.GetOrigin()
        return data, (img, spacing, direction, origin)

    def resample_image_simpleitk(self, file_path, target_spacing, target_size,
                                  is_label=False, reference_spacing=None,
                                  reference_size=None):
        """使用SimpleITK对图像或标签重采样"""
        target_spacing = [float(v) for v in target_spacing]
        img = sitk.ReadImage(file_path)
        orig_spacing = img.GetSpacing()

        ref_spacing = reference_spacing if reference_spacing else orig_spacing
        orig_size = img.GetSize()
        scale = [ref_spacing[i] / target_spacing[i] for i in range(3)]
        output_size = [int(round(orig_size[i] * scale[i])) for i in range(3)]

        if target_size is not None:
            output_size = list(target_size[::-1])

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(output_size)
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)

        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)

        resampled = resampler.Execute(img)
        return sitk.GetArrayFromImage(resampled), orig_spacing

    def resample_array_simpleitk(self, array, orig_spacing, target_spacing,
                                   output_size, is_label=False):
        """对numpy数组使用SimpleITK重采样"""
        orig_spacing = [float(v) for v in orig_spacing]
        target_spacing = [float(v) for v in target_spacing]
        output_size = tuple(int(v) for v in output_size)
        is_multimodal = len(array.shape) == 4

        if is_multimodal:
            channels = []
            for c in range(array.shape[3]):
                img_sitk = sitk.GetImageFromArray(array[:, :, :, c])
                img_sitk.SetSpacing(orig_spacing)

                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(target_spacing)
                resampler.SetSize(list(output_size[::-1]))
                resampler.SetTransform(sitk.Transform())
                resampler.SetDefaultPixelValue(0)

                if is_label:
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                else:
                    resampler.SetInterpolator(sitk.sitkBSpline)

                resampled = resampler.Execute(img_sitk)
                channels.append(sitk.GetArrayFromImage(resampled))

            result = np.stack(channels, axis=-1)
        else:
            img_sitk = sitk.GetImageFromArray(array)
            img_sitk.SetSpacing(orig_spacing)

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(list(output_size[::-1]))
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(0)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label
                                      else sitk.sitkBSpline)

            resampled = resampler.Execute(img_sitk)
            result = sitk.GetArrayFromImage(resampled)

        return result

    def resize_to_target(self, array, target_size, is_label=False):
        """缩放数组到目标尺寸"""
        if array.shape[:3] == tuple(target_size):
            return array

        zoom_factors = [target_size[i] / array.shape[i] for i in range(3)]
        if len(array.shape) == 4:
            result = np.zeros((target_size[0], target_size[1], target_size[2], array.shape[3]))
            for c in range(array.shape[3]):
                order = 0 if is_label else 3
                result[:, :, :, c] = ndimage.zoom(array[:, :, :, c], zoom_factors, order=order)
        else:
            order = 0 if is_label else 3
            result = ndimage.zoom(array, zoom_factors, order=order)

        return result

    def normalize_nonzero_region(self, data, mean=None, std=None):
        """非零区域标准化（适配MRI，排除背景干扰）"""
        if len(data.shape) == 4:
            normalized = np.zeros_like(data, dtype=np.float32)
            for c in range(data.shape[3]):
                modality_data = data[:, :, :, c]
                nonzero_mask = modality_data != 0
                if np.sum(nonzero_mask) == 0:
                    normalized[:, :, :, c] = 0
                    continue

                if mean is None:
                    m = np.mean(modality_data[nonzero_mask])
                    s = np.std(modality_data[nonzero_mask])
                else:
                    m, s = mean[c], std[c]

                normalized[:, :, :, c] = (modality_data - m) / (s + 1e-8)
        else:
            nonzero_mask = data != 0
            if np.sum(nonzero_mask) == 0:
                return data.astype(np.float32)

            if mean is None:
                m = np.mean(data[nonzero_mask])
                s = np.std(data[nonzero_mask])
            else:
                m, s = mean, std

            normalized = (data.astype(np.float32) - m) / (s + 1e-8)

        return normalized

    def remap_label(self, seg_array, target_label):
        """标签二分类映射"""
        seg_array = seg_array.astype(np.int32)
        binary = np.where(seg_array == target_label, 1, 0).astype(np.int32)
        return binary


# ============================================================
# 单样本预处理函数（供多进程并行调用）
# ============================================================

def _preprocess_single_sample(args):
    """
    单样本预处理（供 multiprocessing 调用）
    包含: 对齐 → 重采样 → 标准化 → 标签映射 → 保存
    """
    sample, modality_order, orig_spacing, target_spacing, target_size, \
        target_label, norm_means, norm_stds, image_out_dir, label_out_dir = args

    sample_id = sample['sample_id']

    # 1. 加载并对齐所有模态
    modal_arrays = []
    for mod_name in modality_order:
        arr = nib.load(sample['modal_files'][mod_name]).get_fdata().astype(np.float32)
        modal_arrays.append(arr)

    max_shape = max([arr.shape for arr in modal_arrays],
                    key=lambda s: s[0]*s[1]*s[2])

    aligned_modals = []
    for arr in modal_arrays:
        if arr.shape != max_shape:
            padded = np.zeros(max_shape, dtype=arr.dtype)
            min_shapes = tuple(min(a, b) for a, b in zip(arr.shape, max_shape))
            slices = tuple(slice(0, s) for s in min_shapes)
            padded[slices] = arr[slices]
            aligned_modals.append(padded)
        else:
            aligned_modals.append(arr)

    # 2. 对齐seg
    seg_data = nib.load(sample['seg_file']).get_fdata().astype(np.float32)
    if seg_data.shape != max_shape:
        padded_seg = np.zeros(max_shape, dtype=seg_data.dtype)
        min_shapes = tuple(min(a, b) for a, b in zip(seg_data.shape, max_shape))
        slices = tuple(slice(0, s) for s in min_shapes)
        padded_seg[slices] = seg_data[slices]
        seg_data = padded_seg

    # 3. 重采样（SimpleITK CPU，目前无可替代的高效方案）
    preprocessor = MedicalImagePreprocessor(
        target_spacing=target_spacing,
        target_size=target_size
    )

    resampled_modals = []
    for arr in aligned_modals:
        resampled = preprocessor.resample_array_simpleitk(
            arr, orig_spacing, target_spacing, target_size, is_label=False
        )
        resampled_modals.append(resampled)

    resampled_seg = preprocessor.resample_array_simpleitk(
        seg_data, orig_spacing, target_spacing, target_size, is_label=True
    )

    # 4. 堆叠通道
    image_array = np.stack(resampled_modals, axis=-1)  # (D, H, W, C)

    # 5. GPU 加速标准化（使用 PyTorch CUDA tensor）
    if CUDA_DEVICE is not None:
        # PyTorch GPU 加速：整张图在 GPU 上做标准化
        image_gpu = torch.from_numpy(image_array).float().to(CUDA_DEVICE)  # (D,H,W,C) on GPU
        norm_means_t = torch.tensor(norm_means, device=CUDA_DEVICE, dtype=torch.float32)
        norm_stds_t = torch.tensor(norm_stds, device=CUDA_DEVICE, dtype=torch.float32)

        # 计算非零掩码
        nonzero_mask = image_gpu != 0

        # 向量化标准化：仅对非零区域
        normalized_gpu = torch.where(
            nonzero_mask,
            (image_gpu - norm_means_t) / (norm_stds_t + 1e-8),
            torch.zeros_like(image_gpu)
        )

        normalized = normalized_gpu.cpu().numpy().astype(np.float32)
    else:
        # NumPy CPU fallback
        normalized = np.zeros_like(image_array, dtype=np.float32)
        for c in range(image_array.shape[3]):
            modality_data = image_array[:, :, :, c]
            nonzero_mask = modality_data != 0
            if np.sum(nonzero_mask) > 0:
                normalized[:, :, :, c] = np.where(
                    nonzero_mask,
                    (modality_data - norm_means[c]) / (norm_stds[c] + 1e-8),
                    0.0
                )
            else:
                normalized[:, :, :, c] = 0.0

    # 6. 标签映射
    label_array = preprocessor.remap_label(resampled_seg, target_label)

    # 7. 保存
    np.save(os.path.join(image_out_dir, f'{sample_id}.npy'), normalized)
    np.save(os.path.join(label_out_dir, f'{sample_id}.npy'), label_array)

    return {
        'sample_id': sample_id,
        'patient_id': sample['patient_id'],
        'modal_files': sample['modal_files'],
        'seg_file': sample['seg_file']
    }


# ============================================================
# 预处理管理器：批量预处理并保存到processed文件夹
# ============================================================

def preprocess_dataset(data_root, processed_root, split, target_spacing,
                       target_size, target_label, num_workers=4):
    """
    预处理数据集（train/val/test），将预处理结果保存到processed文件夹。
    使用多进程并行处理样本以加速预处理。

    processed文件夹结构：
        processed_root/
            train/
                images/
                    {sample_id}.npy   # (D, H, W, C) float32
                labels/
                    {sample_id}.npy   # (D, H, W) int32
                sample_metadata.json
            val/
                images/
                labels/
                sample_metadata.json
            test/
                images/
                labels/
                sample_metadata.json
            norm_stats.json           # 标准化统计量（仅train计算并保存）
    """
    print(f"\n[预处理] 开始预处理 {split} 数据集...")
    print(f"  原始数据: {os.path.join(data_root, split)}")
    print(f"  输出路径: {os.path.join(processed_root, split)}")
    print(f"  目标spacing: {target_spacing}")
    print(f"  目标尺寸: {target_size}")
    print(f"  目标病灶标签: {target_label}")
    print(f"  并行进程数: {num_workers}")

    # 扫描样本
    samples = scan_raw_data(data_root, split)
    if len(samples) == 0:
        raise ValueError(f"未找到任何样本: {os.path.join(data_root, split)}")
    print(f"  发现 {len(samples)} 个样本")

    # 创建输出目录
    image_out_dir = os.path.join(processed_root, split, 'images')
    label_out_dir = os.path.join(processed_root, split, 'labels')
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    # 用第一个样本确定模态顺序和原始spacing
    first_sample = samples[0]
    modality_order = [m for m in MODALITY_SUFFIXES if m in first_sample['modal_files']]
    first_mod_path = first_sample['modal_files'][modality_order[0]]
    orig_spacing = tuple(float(v) for v in nib.load(first_mod_path).header.get_zooms()[:3])

    # ===== 计算标准化统计量（仅train） =====
    if split == 'train':
        print(f"\n[预处理] 计算训练集非零区域标准化统计量...")

        per_mod_stats = []
        for mod_name in modality_order:
            vals = []
            for sample in samples:
                arr = nib.load(sample['modal_files'][mod_name]).get_fdata().astype(np.float32)
                nonzero = arr[arr != 0]
                if len(nonzero) > 0:
                    vals.append(nonzero)
            if vals:
                combined_mod = np.concatenate(vals)
                per_mod_stats.append({
                    'modality': mod_name,
                    'mean': float(np.mean(combined_mod)),
                    'std': float(np.std(combined_mod))
                })
            else:
                per_mod_stats.append({'modality': mod_name, 'mean': 0.0, 'std': 1.0})

        # 使用增量方式计算全局统计量
        print(f"  使用增量方式计算全局统计量...")
        global_nonzero_vals = []
        for sample in samples:
            arr = nib.load(sample['modal_files'][modality_order[0]]).get_fdata().astype(np.float32)
            nonzero = arr[arr != 0]
            if len(nonzero) > 0:
                global_nonzero_vals.append(nonzero)

        if global_nonzero_vals:
            global_combined = np.concatenate(global_nonzero_vals)
            global_mean = float(np.mean(global_combined))
            global_std = float(np.std(global_combined))
        else:
            global_mean = 0.0
            global_std = 1.0

        norm_stats = {
            'modality_stats': per_mod_stats,
            'global_mean': float(global_mean),
            'global_std': float(global_std),
            'target_spacing': [float(v) for v in target_spacing],
            'target_size': [int(v) for v in target_size],
            'target_label': int(target_label),
            'modality_order': modality_order,
            'orig_spacing': [float(v) for v in orig_spacing],
            'norm_method': 'z-score-nonzero'
        }

        stats_path = os.path.join(processed_root, 'norm_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        print(f"  标准化统计量已保存: {stats_path}")
    else:
        stats_path = os.path.join(processed_root, 'norm_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                norm_stats = json.load(f)
            print(f"  加载标准化统计量: {stats_path}")
        else:
            raise FileNotFoundError(
                f"未找到训练集标准化统计量: {stats_path}，"
                "请先预处理训练集（python main.py --mode preprocess）"
            )

    norm_means = [s['mean'] for s in norm_stats['modality_stats']]
    norm_stds = [s['std'] for s in norm_stats['modality_stats']]

    # ===== 多进程并行预处理 =====
    print(f"\n[预处理] 使用 {num_workers} 个进程并行处理 {len(samples)} 个样本...")

    # 构建参数列表
    args_list = [
        (sample, modality_order, orig_spacing, target_spacing, target_size,
         target_label, norm_means, norm_stds, image_out_dir, label_out_dir)
        for sample in samples
    ]

    # 使用 multiprocessing 多进程并行
    try:
        from multiprocessing import Pool, cpu_count

        # 自动调整进程数
        actual_workers = min(num_workers, cpu_count(), len(samples))
        print(f"  启动 {actual_workers} 个工作进程...")

        with Pool(processes=actual_workers) as pool:
            # 使用 imap 保持顺序，tqdm 显示进度
            from tqdm import tqdm
            results = list(tqdm(
                pool.imap(_preprocess_single_sample, args_list),
                total=len(args_list),
                desc=f"  预处理 [{split}]",
                unit='样本'
            ))

        sample_metadata = results

    except Exception as e:
        print(f"  多进程失败 ({e})，回退到串行处理...")
        sample_metadata = []
        for i, args in enumerate(args_list):
            result = _preprocess_single_sample(args)
            sample_metadata.append(result)
            if (i + 1) % 50 == 0:
                print(f"    已处理 {i+1}/{len(samples)} 个样本...")

    metadata_path = os.path.join(processed_root, split, 'sample_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(sample_metadata, f, indent=2)

    print(f"[预处理] {split} 完成！共 {len(samples)} 个样本")


def preprocess_if_needed(data_root, processed_root, split, target_spacing,
                          target_size, target_label, force_reprocess=False,
                          num_workers=4):
    """检查processed目录是否存在，不存在则预处理"""
    processed_split_dir = os.path.join(processed_root, split, 'images')
    if not os.path.exists(processed_split_dir) or force_reprocess:
        preprocess_dataset(
            data_root, processed_root, split,
            target_spacing, target_size, target_label,
            num_workers=num_workers
        )
    else:
        print(f"[预处理] {split} 已存在，直接加载（删除processed/{split}可重新预处理）")


# ============================================================
# 数据增强（GPU加速3D增强：旋转/翻转/噪声/伽马）
# 训练时在 GPU 上以批次为单位进行，无需逐样本CPU处理
# ============================================================

class DataAugmenter3D_GPU:
    """
    GPU加速3D医学影像数据增强（在训练循环中以批次为单位在GPU上执行）
    相比CPU单样本增强，GPU批次增强可充分利用GPU并行能力，大幅提升吞吐量
    """

    def __init__(self, enable=True,
                 flip_prob=0.5,
                 rotation_range=15,
                 noise_std=0.01,
                 gamma_range=(0.8, 1.2)):
        self.enable = enable
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.gamma_range = gamma_range

    def _rotate_3d_gpu(self, volume, angle_x, angle_y, angle_z):
        """
        GPU 3D旋转（通过 torch.grid_sample 实现，无需显式插值）
        volume: (B, C, D, H, W) GPU tensor
        返回: (B, C, D, H, W) GPU tensor
        """
        B, C, D, H, W = volume.shape

        # 构建旋转后网格
        affine = torch.eye(4, device=volume.device, dtype=torch.float32)
        cos_x, sin_x = torch.cos(torch.tensor(angle_x * np.pi / 180, device=volume.device)), \
                       torch.sin(torch.tensor(angle_x * np.pi / 180, device=volume.device))
        cos_y, sin_y = torch.cos(torch.tensor(angle_y * np.pi / 180, device=volume.device)), \
                       torch.sin(torch.tensor(angle_y * np.pi / 180, device=volume.device))
        cos_z, sin_z = torch.cos(torch.tensor(angle_z * np.pi / 180, device=volume.device)), \
                       torch.sin(torch.tensor(angle_z * np.pi / 180, device=volume.device))

        # 旋转矩阵（右手系）
        Rx = torch.tensor([[1, 0, 0, 0],
                           [0, cos_x, -sin_x, 0],
                           [0, sin_x, cos_x, 0],
                           [0, 0, 0, 1]], device=volume.device, dtype=torch.float32)
        Ry = torch.tensor([[cos_y, 0, sin_y, 0],
                           [0, 1, 0, 0],
                           [-sin_y, 0, cos_y, 0],
                           [0, 0, 0, 1]], device=volume.device, dtype=torch.float32)
        Rz = torch.tensor([[cos_z, -sin_z, 0, 0],
                           [sin_z, cos_z, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], device=volume.device, dtype=torch.float32)
        R = Rz @ Ry @ Rx

        # 构建 3D sampling grid (D, H, W) -> [-1, 1]
        grid_d = torch.linspace(-1, 1, D, device=volume.device)
        grid_h = torch.linspace(-1, 1, H, device=volume.device)
        grid_w = torch.linspace(-1, 1, W, device=volume.device)
        grid = torch.stack(torch.meshgrid(grid_d, grid_h, grid_w, indexing='ij'), dim=3)  # (D,H,W,3)

        # 应用旋转：grid_flat @ R[:3,:3].T + t
        grid_flat = grid.view(-1, 3)  # (D*H*W, 3)
        ones = torch.ones(D * H * W, 1, device=volume.device)
        grid_homo = torch.cat([grid_flat, ones], dim=1)  # (D*H*W, 4)
        rotated = (grid_homo @ R.T)[:, :3]  # (D*H*W, 3)
        grid_rotated = rotated.view(D, H, W, 3)

        # 归一化到 [-1, 1]
        norm = torch.tensor([2.0 / (D - 1), 2.0 / (H - 1), 2.0 / (W - 1)],
                            device=volume.device)
        grid_rotated = (grid_rotated - torch.tensor([0., 0., 0.], device=volume.device)) * norm

        # clamp 到 [-1, 1]
        grid_rotated = torch.clamp(grid_rotated, -1, 1)

        # grid_sample: 需要 (B, D, H, W, 3)
        grid_sample = grid_rotated.unsqueeze(0).expand(B, -1, -1, -1, -1)

        # 双线性采样（图像用双线性）
        result = F.grid_sample(volume, grid_sample,
                               mode='bilinear', padding_mode='border', align_corners=True)
        return result

    def _flip_gpu(self, image, axis):
        """GPU翻转"""
        return torch.flip(image, dims=[axis])

    def __call__(self, image_batch, label_batch, device='cuda'):
        """
        批次GPU数据增强（在DataLoader返回后、模型前向传播前调用）

        Args:
            image_batch: (B, C, D, H, W) GPU tensor
            label_batch: (B, D, H, W) GPU tensor
            device: 'cuda' 或 'cuda:0' 等

        Returns:
            image_batch, label_batch (已增强，仍在GPU上)
        """
        if not self.enable:
            return image_batch, label_batch

        B = image_batch.shape[0]

        # ---- 随机翻转（沿D/H/W轴）----
        flip_mask_D = torch.rand(B, device=device) < self.flip_prob
        flip_mask_H = torch.rand(B, device=device) < self.flip_prob
        flip_mask_W = torch.rand(B, device=device) < self.flip_prob

        # D轴翻转
        flip_D = torch.rand(B, device=device) < self.flip_prob
        image_batch = torch.where(flip_D.view(B, 1, 1, 1, 1), torch.flip(image_batch, dims=[2]), image_batch)
        label_batch = torch.where(flip_D.view(B, 1, 1, 1), torch.flip(label_batch, dims=[1]), label_batch)
        # H轴翻转
        flip_H = torch.rand(B, device=device) < self.flip_prob
        image_batch = torch.where(flip_H.view(B, 1, 1, 1, 1), torch.flip(image_batch, dims=[3]), image_batch)
        label_batch = torch.where(flip_H.view(B, 1, 1, 1), torch.flip(label_batch, dims=[2]), label_batch)
        # W轴翻转
        flip_W = torch.rand(B, device=device) < self.flip_prob
        image_batch = torch.where(flip_W.view(B, 1, 1, 1, 1), torch.flip(image_batch, dims=[4]), image_batch)
        label_batch = torch.where(flip_W.view(B, 1, 1, 1), torch.flip(label_batch, dims=[3]), label_batch)

        # ---- 随机3D旋转（整批次统一角度）----
        if random.random() < 0.5:
            angle_x = random.uniform(-self.rotation_range, self.rotation_range)
            angle_y = random.uniform(-self.rotation_range, self.rotation_range)
            angle_z = random.uniform(-self.rotation_range, self.rotation_range)
            image_batch = self._rotate_3d_gpu(image_batch, angle_x, angle_y, angle_z)
            label_batch = self._rotate_3d_gpu(label_batch.unsqueeze(1).float(),
                                             angle_x, angle_y, angle_z).squeeze(1).long()

        # ---- 高斯噪声（仅图像，GPU向量化）----
        if random.random() < 0.5:
            noise = torch.randn_like(image_batch) * self.noise_std
            image_batch = image_batch + noise

        # ---- 伽马校正（仅图像非零区域）----
        if random.random() < 0.5:
            gamma = random.uniform(*self.gamma_range)
            gamma_tensor = torch.tensor(gamma, device=device, dtype=torch.float32)

            nonzero_mask = image_batch != 0
            if nonzero_mask.any():
                img_min = image_batch.where(nonzero_mask, torch.full_like(image_batch, float('inf'))).amin(
                    dim=(1, 2, 3), keepdim=True)
                img_max = image_batch.where(nonzero_mask, torch.full_like(image_batch, float('-inf'))).amax(
                    dim=(1, 2, 3), keepdim=True)
                range_valid = img_max - img_min
                range_valid = torch.clamp(range_valid, min=1e-8)

                normalized = (image_batch - img_min) / range_valid
                corrected = torch.pow(torch.clamp(normalized, min=1e-8, max=1.0), gamma_tensor)
                corrected = corrected * range_valid + img_min

                # 仅对非零区域应用
                image_batch = torch.where(nonzero_mask, corrected, torch.zeros_like(image_batch))

        return image_batch, label_batch


# ============================================================
# 旧版 CPU 增强器（保留兼容）
# ============================================================

class DataAugmenter3D:
    """3D医学影像数据增强（仅用于训练集，在预处理后的数据上实时增强，CPU版）"""

    def __init__(self, enable=True,
                 flip_prob=0.5,
                 rotation_range=15,
                 noise_std=0.01,
                 gamma_range=(0.8, 1.2)):
        self.enable = enable
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.gamma_range = gamma_range

    def _rotate_3d(self, volume, angle_x=0, angle_y=0, angle_z=0, is_label=False):
        """沿三个轴旋转3D体"""
        order = 0 if is_label else 3
        if angle_x != 0:
            volume = ndimage.rotate(volume, angle_x, axes=(1, 2), order=order, mode='constant', cval=0)
        if angle_y != 0:
            volume = ndimage.rotate(volume, angle_y, axes=(0, 2), order=order, mode='constant', cval=0)
        if angle_z != 0:
            volume = ndimage.rotate(volume, angle_z, axes=(0, 1), order=order, mode='constant', cval=0)
        return volume

    def _random_flip(self, image, label):
        """随机沿三个轴翻转"""
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()
        return image, label

    def _random_rotate(self, image, label):
        """随机小角度3D旋转"""
        angle_x = random.uniform(-self.rotation_range, self.rotation_range)
        angle_y = random.uniform(-self.rotation_range, self.rotation_range)
        angle_z = random.uniform(-self.rotation_range, self.rotation_range)

        image = self._rotate_3d(image, angle_x, angle_y, angle_z, is_label=False)
        label = self._rotate_3d(label, angle_x, angle_y, angle_z, is_label=True)

        return image, label

    def _add_gaussian_noise(self, image, std=None):
        """添加高斯噪声（仅图像）"""
        if std is None:
            std = self.noise_std
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        return image + noise

    def _random_gamma(self, image, gamma_range=None):
        """随机伽马校正（仅图像，非零区域）"""
        if gamma_range is None:
            gamma_range = self.gamma_range
        gamma = random.uniform(*gamma_range)

        nonzero_mask = image != 0
        result = image.copy()
        img_min = np.min(image[nonzero_mask]) if np.any(nonzero_mask) else 0
        img_max = np.max(image[nonzero_mask]) if np.any(nonzero_mask) else 1
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
            corrected = np.power(np.clip(normalized, 1e-8, 1.0), gamma)
            corrected = corrected * (img_max - img_min) + img_min
            result = np.where(nonzero_mask, corrected, 0.0)

        return result

    def __call__(self, image, label):
        """单样本CPU增强"""
        if not self.enable:
            return image, label

        image, label = self._random_flip(image, label)

        if random.random() < 0.5:
            image, label = self._random_rotate(image, label)

        if random.random() < 0.5:
            image = self._add_gaussian_noise(image)

        if random.random() < 0.5:
            image = self._random_gamma(image)

        return image, label


# ============================================================
# 数据集加载器（从预处理后的processed文件夹加载）
# ============================================================

class BraTSDataset(Dataset):
    """
    从预处理后的processed文件夹加载数据。
    支持两种增强模式：
      - CPU增强（DataAugmenter3D）：单样本numpy处理
      - GPU增强（DataAugmenter3D_GPU）：批次GPU tensor处理
    """

    def __init__(self, processed_root, split='train', augment=False,
                 augmentation_config=None, use_gpu_augment=False):
        """
        Args:
            processed_root: 预处理数据根目录
            split: 'train', 'val', 'test'
            augment: 是否启用数据增强（仅train有效）
            augmentation_config: 数据增强参数字典
            use_gpu_augment: 是否使用GPU批次增强（训练时推荐开启）
        """
        self.processed_root = processed_root
        self.split = split
        self.augment = augment and (split == 'train')
        self.use_gpu_augment = use_gpu_augment and self.augment

        # 加载文件列表
        self.image_dir = os.path.join(processed_root, split, 'images')
        self.label_dir = os.path.join(processed_root, split, 'labels')

        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            raise FileNotFoundError(
                f"预处理数据目录不存在: {self.image_dir} 或 {self.label_dir}\n"
                f"请先运行预处理（数据加载时会自动触发）"
            )

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir) if f.endswith('.npy')
        ])
        self.label_files = sorted([
            f for f in os.listdir(self.label_dir) if f.endswith('.npy')
        ])

        image_ids = set(f.replace('.npy', '') for f in self.image_files)
        label_ids = set(f.replace('.npy', '') for f in self.label_files)
        if image_ids != label_ids:
            missing = image_ids - label_ids
            extra = label_ids - image_ids
            raise ValueError(f"图像和标签文件不匹配！缺少标签: {missing}, 多余标签: {extra}")

        print(f"[{split}] 共 {len(self.image_files)} 个样本"
              f"{' (GPU增强模式)' if self.use_gpu_augment else ' (CPU增强模式)' if self.augment else ''}")

        # 初始化增强器
        if self.augment:
            aug_cfg = augmentation_config or {}
            if self.use_gpu_augment:
                self.gpu_augmenter = DataAugmenter3D_GPU(
                    enable=True,
                    flip_prob=aug_cfg.get('flip_prob', 0.5),
                    rotation_range=aug_cfg.get('rotation_range', 15),
                    noise_std=aug_cfg.get('noise_std', 0.01),
                    gamma_range=tuple(aug_cfg.get('gamma_range', (0.8, 1.2)))
                )
                self.cpu_augmenter = None
            else:
                self.cpu_augmenter = DataAugmenter3D(
                    enable=True,
                    flip_prob=aug_cfg.get('flip_prob', 0.5),
                    rotation_range=aug_cfg.get('rotation_range', 15),
                    noise_std=aug_cfg.get('noise_std', 0.01),
                    gamma_range=tuple(aug_cfg.get('gamma_range', (0.8, 1.2)))
                )
                self.gpu_augmenter = None
        else:
            self.cpu_augmenter = None
            self.gpu_augmenter = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image_data = np.load(image_path).astype(np.float32)
        label_data = np.load(label_path).astype(np.int32)

        # CPU增强（逐样本）
        if self.augment and self.cpu_augmenter is not None:
            image_data, label_data = self.cpu_augmenter(image_data, label_data)

        # 转换为PyTorch格式
        image_data = np.transpose(image_data, (3, 0, 1, 2))
        label_data = np.transpose(label_data, (0, 1, 2))

        image_tensor = torch.from_numpy(image_data).float()
        label_tensor = torch.from_numpy(label_data).long()

        return {
            'image': image_tensor,
            'label': label_tensor,
            'filename': self.image_files[idx].replace('.npy', '')
        }

    def get_gpu_augmenter(self):
        """返回GPU增强器实例（供训练循环调用）"""
        return self.gpu_augmenter


# ============================================================
# GPU增强批次包装器（接入DataLoader的批次流程）
# ============================================================

class GPUAugmentCollator:
    """
    GPU批次增强包装器：接管 DataLoader 的批次，在 GPU 上完成增强后再返回
    使用方式：
        train_loader = DataLoader(dataset, batch_size=B, collate_fn=GPUAugmentCollator())
    """

    def __init__(self, augmenter, device='cuda'):
        self.augmenter = augmenter
        self.device = device

    def __call__(self, batch):
        """
        Args:
            batch: list of dicts from BraTSDataset.__getitem__
        Returns:
            single dict with batched tensors (增强后仍在GPU上)
        """
        # 堆叠样本
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        filenames = [item['filename'] for item in batch]

        # 移至GPU并执行增强
        images = images.to(self.device)
        labels = labels.to(self.device)

        images, labels = self.augmenter(images, labels, device=self.device)

        return {
            'image': images,
            'label': labels,
            'filename': filenames
        }


# ============================================================
# 数据加载器工厂
# ============================================================

def create_data_loaders(data_root, processed_root, preprocess_config,
                         batch_size=2, num_workers=4,
                         augmentation_config=None,
                         force_reprocess=False,
                         use_gpu_augment=False):
    """
    创建训练/验证/测试数据加载器。
    首次调用时自动预处理数据。

    Args:
        data_root: 原始数据根目录
        processed_root: 预处理数据根目录
        preprocess_config: 预处理配置字典
        batch_size: 批次大小
        num_workers: DataLoader进程数
        augmentation_config: 数据增强配置
        force_reprocess: 是否强制重新预处理
        use_gpu_augment: 是否使用GPU批次增强（推荐True，训练时显著加速）

    Returns:
        (train_loader, val_loader, test_loader)
    """
    target_spacing = preprocess_config.get('target_spacing', (1.0, 1.0, 1.0))
    target_size = preprocess_config.get('target_size', (128, 128, 128))
    target_label = preprocess_config.get('target_label', 1)
    force = force_reprocess or preprocess_config.get('force_reprocess', False)
    aug_num_workers = preprocess_config.get('num_workers', num_workers)

    # 自动预处理所有数据集
    for split in ['train', 'val', 'test']:
        preprocess_if_needed(
            data_root, processed_root, split,
            target_spacing, target_size, target_label,
            force_reprocess=force,
            num_workers=aug_num_workers
        )

    # 创建数据集
    train_dataset = BraTSDataset(
        processed_root, split='train', augment=True,
        augmentation_config=augmentation_config,
        use_gpu_augment=use_gpu_augment
    )
    val_dataset = BraTSDataset(
        processed_root, split='val', augment=False
    )
    test_dataset = BraTSDataset(
        processed_root, split='test', augment=False
    )

    # GPU增强模式：使用自定义collator
    if use_gpu_augment and train_dataset.gpu_augmenter is not None:
        gpu_collator = GPUAugmentCollator(
            train_dataset.gpu_augmenter,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        # 注意：GPU collator 返回的已是 CUDA tensor
        #   - pin_memory 必须设为 False（只能 pin CPU tensor）
        #   - num_workers 必须设为 0（CUDA tensor 无法跨进程安全传递）
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False, drop_last=True,
            collate_fn=gpu_collator
        )
    else:
        # 非 GPU collator 模式：数据停留在 CPU，由 trainer.train_epoch 中的 .to(device) 移至 GPU
        # num_workers=0 时 pin_memory 反而徒增开销（无子进程，不存在异步传输）
        use_pin = False if num_workers == 0 else True
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_pin, drop_last=True
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("data_load.py 测试")
    print("="*60)

    try:
        train_samples = scan_raw_data(
            r'D:\PythonProgram\BrainSeg\data', 'train'
        )
        print(f"\n训练集样本数: {len(train_samples)}")
        if train_samples:
            s = train_samples[0]
            print(f"  示例: {s['sample_id']}")
            print(f"  模态: {list(s['modal_files'].keys())}")
            print(f"  seg: {os.path.basename(s['seg_file'])}")
    except FileNotFoundError as e:
        print(f"  {e}")
        print("  （数据目录不存在，跳过扫描测试）")

    print("\n测试预处理...")
    test_config = {
        'target_spacing': (1.0, 1.0, 1.0),
        'target_size': (128, 128, 128),
        'target_label': 1
    }

    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=r'D:\PythonProgram\BrainSeg\data',
            processed_root=r'D:\PythonProgram\BrainSeg\processed',
            preprocess_config=test_config,
            batch_size=1,
            num_workers=0,
            force_reprocess=False,
            use_gpu_augment=False
        )

        for batch in train_loader:
            print(f"\n图像形状: {batch['image'].shape}")
            print(f"标签形状: {batch['label'].shape}")
            print(f"文件名: {batch['filename']}")
            print(f"图像值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(f"标签唯一值: {torch.unique(batch['label']).tolist()}")
            break
    except FileNotFoundError as e:
        print(f"  {e}")
        print("  （数据目录不存在，跳过加载测试）")
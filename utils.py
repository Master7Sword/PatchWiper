import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):        
        # 调整维度顺序，从 (C, H, W) 转换为 (H, W, C)
        img_sample = np.transpose(Img[i,:,:,:], (1, 2, 0))  # (C, H, W) -> (H, W, C)
        imclean_sample = np.transpose(Iclean[i,:,:,:], (1, 2, 0))  # (C, H, W) -> (H, W, C)

        ssim_value = ssim(
            imclean_sample, 
            img_sample, 
            data_range=data_range, 
            channel_axis=-1,  # 通道维度是最后一个维度
            win_size=11  
        )
        SSIM += ssim_value
    
    return (SSIM / Img.shape[0])


def compute_RMSE(pred, gt, mask, is_w=False):
    if is_w:
        if isinstance(mask, torch.Tensor):
            mse = torch.mean((pred*mask - gt*mask)**2, dim=[1,2,3])
            rmse = mse*np.prod(mask.shape[1:])/(torch.sum(mask, dim=[1,2,3])+1e-6)
            rmse = torch.sqrt(rmse).mean().item()
        # elif isinstance(mask, np.ndarray):
        #     rmse = MSE(pred*mask, gt*mask)*np.prod(mask.shape) / (np.sum(mask)+1e-6)
        #     rmse = np.sqrt(rmse)
        else:
            print("Please make sure the input is torch.Tensor!")
    else:
        if isinstance(mask, torch.Tensor):
            mse = torch.mean((pred - gt)**2, dim=[1,2,3])
            rmse = torch.sqrt(mse).mean().item()
        # elif isinstance(mask, np.ndarray):
        #     rmse = MSE(pred, gt)*np.prod(mask.shape) / (np.sum(mask)+1e-6)
        #     rmse = np.sqrt(rmse)
        else:
            print("Please make sure the input is torch.Tensor!")
    
    return rmse * 256


def batch_LPIPS(img, img_clean, lpips_model):
    # 将输入图像归一化到 [-1, 1] 范围
    img = (img - 0.5) / 0.5
    img_clean = (img_clean - 0.5) / 0.5

    # 计算 LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(img, img_clean)

    # 返回 LPIPS 值
    return lpips_value.mean().item()


def calculate_metrics(pred, target):
    # Flatten the arrays to simplify calculation
    pred = pred.flatten()
    target = target.flatten()
    
    # Calculate TP, TN, FP, FN
    TP = np.sum((pred == 1) & (target == 1))
    TN = np.sum((pred == 0) & (target == 0))
    FP = np.sum((pred == 1) & (target == 0))
    FN = np.sum((pred == 0) & (target == 1))
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0 
    iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
    # dice_coefficient = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'f1_score': f1_score
    }


def model_structure(model):
    blank = ' '
    print('-' * 130)
    print('|' + ' ' * 31 + 'weight name' + ' ' * 30 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 130)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 70:
            key = key + (70 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 130)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 130)


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot
    


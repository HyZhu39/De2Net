import os.path
from collections import OrderedDict
import numpy as np
import torch
import pyiqa
from basicsr.metrics.metric_util import reorder_image, to_y_channel


def tone_map(x, type='simple'):
    if type == 'mu_law':
        norm_x = x / x.max()
        mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
    elif type == 'simple':
        mapped_x = x / (x + 0.25)
    elif type == 'same':
        mapped_x = x
    else:
        raise NotImplementedError('tone mapping type [{:s}] is not recognized.'.format(type))
    return mapped_x


def calculate_psnr_match(img1, img2, input_order='HWC',):
    """Calculate PSNR (Peak Signal-to-Noise Ratio). 

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].

        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float32) #/ 255.
    img2 = img2.astype(np.float32) #/ 255.
    
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze_(0)
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze_(0)
    img1 = tone_map(img1)
    img2 = tone_map(img2)
    
    #img1 = torch.clip(img1, min=0, max=1)
    #img2 = torch.clip(img2, min=0, max=1)
    # mse = np.mean((img1 - img2)**2)

    #squared_error = np.square(img2 - img1)
    #mse = np.mean(squared_error)

    #if mse == 0:
    #    return float('inf')

    #psnr = 10 * np.log10(1.0 / mse)
    iqa_metric = pyiqa.create_metric('psnr', data_range=1.).to(img1.device)
    psnr = iqa_metric(img1, img2)
    
    #print('img1',img1, torch.max(img1), torch.min(img1))
    #print('img2',img2, torch.max(img2), torch.min(img1))
    #print('psnr', psnr.numpy())
    psnr = psnr.numpy()
    psnr = float(psnr)
    return psnr # 20. * np.log10(255. / np.sqrt(mse))

def calculate_ssim_match(img1, img2, input_order='HWC',):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float32) #/ 255.  # np.float64
    img2 = img2.astype(np.float32) #/ 255.  # np.float64
    
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze_(0)
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze_(0)
    img1 = tone_map(img1)
    img2 = tone_map(img2)
    
    #img1 = torch.clip(img1, min=0, max=1)
    #img2 = torch.clip(img2, min=0, max=1)
    
    iqa_metric = pyiqa.create_metric('ssim').to(img1.device) # data_range=1. 
    ssim = iqa_metric(img1, img2)

    ssim = ssim.numpy()
    ssim = float(ssim)
    return ssim

def calculate_lpips_match(img1, img2, input_order='HWC',):

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float32) #/ 255.
    img2 = img2.astype(np.float32) # / 255.
    
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze_(0)
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze_(0)
    img1 = tone_map(img1)
    img2 = tone_map(img2)
    
    #img1 = torch.clip(img1, min=0, max=1)
    #img2 = torch.clip(img2, min=0, max=1)

    iqa_metric = pyiqa.create_metric('lpips').to(img1.device)
    lpips = iqa_metric(img1, img2)

    lpips = lpips.detach().numpy()
    lpips = float(lpips)
    return lpips

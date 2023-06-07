import mmcv
import numpy as np
from torch.utils import data as data
import random
import torch

from basicsr.data.transforms import paired_random_crop, totensor
from basicsr.data.util import (paired_paths_from_folder,
                               paired_paths_from_lmdb,
                               paired_paths_from_meta_info_file)
from basicsr.utils import FileClient


class OLEDTrainDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(OLEDTrainDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.

        flags = torch.zeros([3])

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            imgs, flags = self.augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])
            img_gt, img_lq = imgs

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = totensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'flags': flags,
        }

    def __len__(self):
        return len(self.paths)

    def augment(self, imgs, hflip=True, rotation=True):
        """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).
        We use vertical flip and transpose for rotation implementation.
        All the images in the list use the same augmentation.
        Args:
            imgs (list[ndarray] | ndarray): Images to be augmented. If the input
                is an ndarray, it will be transformed to a list.
            hflip (bool): Horizontal flip. Default: True.
            rotation (bool): Ratotation. Default: True.

        Returns:
            list[ndarray] | ndarray: Augmented images and flows. If returned
                results only have one element, just return ndarray.
        """
        flags = torch.zeros([3])

        hflip = hflip and random.random() < 0.5
        vflip = rotation and random.random() < 0.5
        rot90 = rotation and random.random() < 0.5
        if hflip:
            flags[0] = 1
        if vflip:
            flags[1] = 1
        if rot90:
            flags[2] = 1

        def _augment(img):
            if hflip:
                mmcv.imflip_(img, 'horizontal')
            if vflip:
                mmcv.imflip_(img, 'vertical')
            if rot90:
                img = img.transpose(1, 0, 2)  # cause this is HWC
            return img

        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [_augment(img) for img in imgs]
        if len(imgs) == 1:
            imgs = imgs[0]

        return imgs, flags

import mmcv
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, totensor
from scipy.io.matlab.mio import savemat, loadmat

class OLEDTestDataset(data.Dataset):
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
        super(OLEDTestDataset, self).__init__()
        self.opt = opt

        self.input_udc_key = opt['input_udc_key']
        self.gt_udc_key = opt['gt_udc_key']

        self.test_input_path = opt['test_input_path']
        self.test_gt_path = opt['test_gt_path']

        self.test_input = loadmat(self.test_input_path)[self.input_udc_key]
        self.test_gt = loadmat(self.test_gt_path)[self.gt_udc_key]

    def __getitem__(self, index):
        # Load gt and lq images. Dimension order: HWC; channel order: RGB;

        gt_path = str(index + 1) + ".png"
        lq_path = str(index + 1) + ".png"

        img_gt = self.test_gt[index].astype(np.float32) / 255.
        img_lq = self.test_input[index].astype(np.float32) / 255.

        if self.opt['phase'] == 'train':
            pass

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = totensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.test_gt)

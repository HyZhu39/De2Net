import functools
import torch
import torch.nn as nn
from torch.nn import init as init
import torch.nn.functional as F
import numpy as np
from basicsr.utils import get_uperleft_denominator, get_uperleft_denominator_for_test, get_uperleft_denominator_para, get_uperleft_denominator_para_NCHW
import scipy
import math


class TOLED(nn.Module):
    def __init__(self, in_nc, out_nc, nf=32, ns=64, kpn_sz=15, norm_type=None, \
                 act_type='leakyrelu', res_scale=1, kernel_size=3, \
                 kernel_cond=None, psf_nc=5, multi_scale=False, final_kernel=False, \
                 bilinear_sz=7, dilation_sz=4, basis_num=90, \
                 kpn_sz_center=5, \
                 croped_ksz=64, \
                 wiener_level=3):
        super().__init__()
        self.ns = ns
        self.nf = nf
        self.kpn_sz = kpn_sz
        self.kernel_cond = kernel_cond
        self.multi_scale = multi_scale
        self.bilinear_size = bilinear_sz
        self.dilation_size = dilation_sz
        self.basis_num = basis_num
        self.final_kernel = final_kernel
        self.kpn_sz_center = kpn_sz_center

        self.croped_ksz = croped_ksz

        self.wiener_level = math.floor(wiener_level)  # nums of wiener

        if self.wiener_level >= 1:
            self.wiener_kernel_1 = init_wiener_kernel(self.croped_ksz, ns)
            self.wiener_kernel_1 = nn.Parameter(self.wiener_kernel_1.requires_grad_(True))
        if self.wiener_level >= 2:
            self.wiener_kernel_2 = init_wiener_kernel(self.croped_ksz, ns * 2)
            self.wiener_kernel_2 = nn.Parameter(self.wiener_kernel_2.requires_grad_(True))
        if self.wiener_level >= 3:
            self.wiener_kernel_3 = init_wiener_kernel(self.croped_ksz, ns * 4)
            self.wiener_kernel_3 = nn.Parameter(self.wiener_kernel_3.requires_grad_(True))
        if self.wiener_level >= 4:
            self.wiener_kernel_4 = init_wiener_kernel(self.croped_ksz, ns * 8)
            self.wiener_kernel_4 = nn.Parameter(self.wiener_kernel_4.requires_grad_(True))
        if self.wiener_level >= 5:
            self.wiener_kernel_5 = init_wiener_kernel(self.croped_ksz, ns * 8)
            self.wiener_kernel_5 = nn.Parameter(self.wiener_kernel_5.requires_grad_(True))
        #############################
        # Restoration Branch
        #############################
        self.avg_pool = nn.AvgPool2d(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Encoder
        self.conv_1 = nn.Conv2d(in_nc, ns, 3, 1, 1, bias=True)  # 64f
        basic_block_64 = functools.partial(ResidualDenseBlock_5C, nf=ns)
        self.encoder1 = make_layer(basic_block_64, 2)  # 64f
        if ns != nf:
            self.wiener_fix_KPN_1 = nn.Conv2d(self.ns, nf, 1, 1, 0, bias=True)

        self.conv_2 = nn.Conv2d(ns, ns * 2, 3, 1, 1, bias=True)  # 128f
        basic_block_128 = functools.partial(ResidualDenseBlock_5C, nf=ns * 2)
        self.encoder2 = make_layer(basic_block_128, 2)  # 128f
        if ns != nf:
            self.wiener_fix_KPN_2 = nn.Conv2d(self.ns * 2, nf * 2, 1, 1, 0, bias=True)

        self.conv_3 = nn.Conv2d(ns * 2, ns * 4, 3, 1, 1, bias=True)  # 256f
        basic_block_256 = functools.partial(ResidualDenseBlock_5C, nf=ns * 4)
        self.encoder3 = make_layer(basic_block_256, 2)  # 256f
        if ns != nf:
            self.wiener_fix_KPN_3 = nn.Conv2d(self.ns * 4, nf * 4, 1, 1, 0, bias=True)

        self.conv_4 = nn.Conv2d(ns * 4, ns * 8, 3, 1, 1, bias=True)  # 512f
        basic_block_512 = functools.partial(ResidualDenseBlock_5C, nf=ns * 8)
        self.encoder4 = make_layer(basic_block_512, 2)  # 512f
        if ns != nf:
            self.wiener_fix_KPN_4 = nn.Conv2d(self.ns * 8, nf * 8, 1, 1, 0, bias=True)

        self.conv_5 = nn.Conv2d(ns * 8, ns * 8, 3, 1, 1, bias=True)  # 512f
        if ns != nf:
            self.wiener_fix_KPN_5 = nn.Conv2d(self.ns * 8, 512, 1, 1, 0, bias=True)

        self.conv_6 = nn.Conv2d(ns * 8, 512, 3, 1, 1, bias=True)  # 512f
        if nf * 8 != 512:
            self.wiener_fix_KPN_basis = nn.Conv2d(512, nf * 8, 1, 1, 0, bias=True)

        # Decoder
        self.conv_7 = nn.Conv2d(512, 256, 3, 1, padding=1, bias=True)
        self.conv_8 = nn.Conv2d(768, 512, 3, 1, padding=1, bias=True)
        self.conv_9 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.conv_10 = nn.Conv2d(640, 512, 3, 1, padding=1, bias=True)
        self.conv_11 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.conv_12 = nn.Conv2d(384, 256, 3, 1, padding=1, bias=True)
        self.conv_13 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        self.conv_14 = nn.Conv2d(192, 128, 3, 1, padding=1, bias=True)
        self.conv_15 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=True)
        self.conv_16 = nn.Conv2d(96, 64, 3, 1, padding=1, bias=True)
        self.conv_17 = nn.Conv2d(64, out_nc, 3, 1, padding=1, bias=True)
        self.pixshuffle_1 = nn.PixelShuffle(upscale_factor=2)
        self.pixshuffle_2 = nn.PixelShuffle(upscale_factor=2)
        self.pixshuffle_3 = nn.PixelShuffle(upscale_factor=2)
        self.pixshuffle_4 = nn.PixelShuffle(upscale_factor=2)

        if final_kernel:
            self.final_mend = nn.Conv2d(out_nc, out_nc, 3, 1, padding=1, bias=True)

        #############################
        # Kernel Prediction Branch
        #############################
        if self.kernel_cond:
            if self.kernel_cond == 'img':
                cond_nc = in_nc
            elif self.kernel_cond == 'psf':
                cond_nc = psf_nc
            elif self.kernel_cond == 'img-psf':
                cond_nc = in_nc + psf_nc

            # Coefficient Decoder:
            self.Coefficient_body_5 = nn.Sequential(
                conv_block(16 * nf, 16 * nf, kernel_size=kernel_size),
                ResBlock(16 * nf, res_scale=res_scale, act_type=act_type),
                ResBlock(16 * nf, res_scale=res_scale, act_type=act_type))

            self.dynamic_kernel_5 = nn.Sequential(
                conv_block(16 * nf, 16 * nf, kernel_size=kernel_size),
                ResBlock(16 * nf, res_scale=res_scale, act_type=act_type),
                ResBlock(16 * nf, res_scale=res_scale, act_type=act_type),
                conv_block(16 * nf, 2 * 16 * nf * (self.kpn_sz_center ** 2), kernel_size=1))

            if self.final_kernel:
                self.Coefficient_entry_4 = upconv(16 * nf, 8 * nf, 2, act_type=act_type)
                self.Coefficient_body_4 = nn.Sequential(
                    conv_block(8 * nf, 8 * nf, kernel_size=kernel_size),
                    ResBlock(8 * nf, res_scale=res_scale, act_type=act_type),
                    ResBlock(8 * nf, res_scale=res_scale, act_type=act_type))

                self.Coefficient_entry_3 = upconv(8 * nf, 4 * nf, 2, act_type=act_type)
                self.Coefficient_body_3 = nn.Sequential(
                    conv_block(4 * nf, 4 * nf, kernel_size=kernel_size),
                    ResBlock(4 * nf, res_scale=res_scale, act_type=act_type),
                    ResBlock(4 * nf, res_scale=res_scale, act_type=act_type))

                self.Coefficient_entry_2 = upconv(4 * nf, 2 * nf, 2, act_type=act_type)
                self.Coefficient_body_2 = nn.Sequential(
                    conv_block(2 * nf, 2 * nf, kernel_size=kernel_size),
                    ResBlock(2 * nf, res_scale=res_scale, act_type=act_type),
                    ResBlock(2 * nf, res_scale=res_scale, act_type=act_type))

                self.Coefficient_entry_1 = upconv(2 * nf, nf, 2, act_type=act_type)
                self.Coefficient_body_1 = nn.Sequential(
                    conv_block(nf, nf, kernel_size=kernel_size),
                    ResBlock(nf, res_scale=res_scale, act_type=act_type),
                    ResBlock(nf, res_scale=res_scale, act_type=act_type))

                self.Coefficient_tail_final = conv_block(nf, self.basis_num, kernel_size=1)

            # Basis Decoder:
            self.Basis_body_1 = Kernel_UpBlock(channel_in=8 * nf, block_channels=4 * nf, act_type=act_type,
                                               upscale_factor=2)
            self.Basis_body_2 = Kernel_UpBlock(channel_in=4 * nf, block_channels=2 * nf, act_type=act_type,
                                               upscale_factor=2)
            self.Basis_body_3 = Kernel_UpBlock(channel_in=2 * nf, block_channels=1 * nf, act_type=act_type,
                                               upscale_factor=2)
            self.Basis_body_4 = Kernel_UpBlock(channel_in=1 * nf, block_channels=1 * nf, act_type=act_type,
                                               upscale_factor=2)

            if self.final_kernel:
                self.Basis_tail_4_final = Basis_Decoder_Tail(channel_in=1 * nf, channel_out=out_nc,
                                                             basis_num=self.basis_num, kernel_size=self.kpn_sz,
                                                             act_type='prelu')

    def forward(self, x, flags=None):

        if not self.training:
            N, C, H, W = x.shape
            H_pad = 16 - H % 16 if not H % 16 == 0 else 0
            W_pad = 16 - W % 16 if not W % 16 == 0 else 0
            x = F.pad(x, (0, W_pad, 0, H_pad), 'replicate')

        fea_x = x
        #############################
        # Restoration Branch
        #############################
        # Encoder
        fea = self.lrelu(self.conv_1(fea_x))
        fea_cat1 = self.encoder1(fea)
        if self.wiener_level >= 1:
            if self.training:
                wiener_kernels_1 = self.wiener_kernel_1.repeat(x.shape[0], 1, 1, 1)  # N ns h w
                wiener_kernels_1 = aug_kernel_with_flags(wiener_kernels_1, flags)

                fea_cat1_mid = get_clear_feature_NCHW(fea_cat1, wiener_kernels_1)
            else:
                wiener_kernel_1 = self.wiener_kernel_1
                if flags is not None:
                    wiener_kernel_1 = aug_kernel_with_flags(wiener_kernel_1, flags)
                fea_cat1_mid = get_clear_feature_test_NCHW(fea_cat1, wiener_kernel_1)

            fea_cat1_skip = fea_cat1_mid
            if self.wiener_level == 1:
                fea_cat1 = fea_cat1_skip

        fea = self.avg_pool(fea_cat1)
        fea = self.lrelu(self.conv_2(fea))
        fea_cat2 = self.encoder2(fea)
        if self.wiener_level >= 2:
            if self.training:
                wiener_kernels_2 = self.wiener_kernel_2.repeat(x.shape[0], 1, 1, 1)
                wiener_kernels_2 = aug_kernel_with_flags(wiener_kernels_2, flags)

                fea_cat2_mid = get_clear_feature_NCHW(fea_cat2, wiener_kernels_2)
            else:
                wiener_kernel_2 = self.wiener_kernel_2
                if flags is not None:
                    wiener_kernel_2 = aug_kernel_with_flags(wiener_kernel_2, flags)
                fea_cat2_mid = get_clear_feature_test_NCHW(fea_cat2, wiener_kernel_2)

            fea_cat2_skip = fea_cat2_mid
            if self.wiener_level == 2:
                fea_cat2 = fea_cat2_skip

        fea = self.avg_pool(fea_cat2)
        fea = self.lrelu(self.conv_3(fea))
        fea_cat3 = self.encoder3(fea)
        if self.wiener_level >= 3:
            if self.training:
                wiener_kernels_3 = self.wiener_kernel_3.repeat(x.shape[0], 1, 1, 1)  # N ns*4 h w
                wiener_kernels_3 = aug_kernel_with_flags(wiener_kernels_3, flags)

                fea_cat3_mid = get_clear_feature_NCHW(fea_cat3, wiener_kernels_3)
            else:
                wiener_kernel_3 = self.wiener_kernel_3
                if flags is not None:
                    wiener_kernel_3 = aug_kernel_with_flags(wiener_kernel_3, flags)
                fea_cat3_mid = get_clear_feature_test_NCHW(fea_cat3, wiener_kernel_3)

            fea_cat3_skip = fea_cat3_mid
            if self.wiener_level == 3:
                fea_cat3 = fea_cat3_skip

        fea = self.avg_pool(fea_cat3)
        fea = self.lrelu(self.conv_4(fea))
        fea_cat4 = self.encoder4(fea)
        if self.wiener_level >= 4:
            if self.training:
                wiener_kernels_4 = self.wiener_kernel_4.repeat(x.shape[0], 1, 1, 1)  # N ns*8 h w
                wiener_kernels_4 = aug_kernel_with_flags(wiener_kernels_4, flags)

                fea_cat4_mid = get_clear_feature_NCHW(fea_cat4, wiener_kernels_4)
            else:
                wiener_kernel_4 = self.wiener_kernel_4
                if flags is not None:
                    wiener_kernel_4 = aug_kernel_with_flags(wiener_kernel_4, flags)
                fea_cat4_mid = get_clear_feature_test_NCHW(fea_cat4, wiener_kernel_4)

            fea_cat4_skip = fea_cat4_mid
            if self.wiener_level == 4:
                fea_cat4 = fea_cat4_skip

        fea = self.avg_pool(fea_cat4)
        fea_cat5 = self.conv_5(fea)
        if self.wiener_level >= 5:
            if self.training:
                wiener_kernels_5 = self.wiener_kernel_5.repeat(x.shape[0], 1, 1, 1)  # N ns*8 h w
                wiener_kernels_5 = aug_kernel_with_flags(wiener_kernels_5, flags)

                fea_cat5_mid = get_clear_feature_NCHW(fea_cat5, wiener_kernels_5)
            else:
                wiener_kernel_5 = self.wiener_kernel_5
                if flags is not None:
                    wiener_kernel_5 = aug_kernel_with_flags(wiener_kernel_5, flags)
                fea_cat5_mid = get_clear_feature_test_NCHW(fea_cat5, wiener_kernel_5)

            fea_cat5_skip = fea_cat5_mid
            if self.wiener_level == 5:
                fea_cat5 = fea_cat5_skip

        fea = self.lrelu(fea_cat5)
        fea = self.conv_6(fea)

        #############################
        # Kernel Prediction Branch
        #############################

        basic_basis = torch.mean(fea, dim=[2, 3], keepdim=True)
        if self.nf * 8 != 512:
            basic_basis = self.wiener_fix_KPN_basis(basic_basis)

        if self.ns != self.nf:
            if self.wiener_level >= 1:
                kfea1 = self.wiener_fix_KPN_1(fea_cat1_mid)
            else:
                kfea1 = self.wiener_fix_KPN_1(fea_cat1)
            if self.wiener_level >= 2:
                kfea2 = self.wiener_fix_KPN_2(fea_cat2_mid)
            else:
                kfea2 = self.wiener_fix_KPN_2(fea_cat2)
            if self.wiener_level >= 3:
                kfea3 = self.wiener_fix_KPN_3(fea_cat3_mid)
            else:
                kfea3 = self.wiener_fix_KPN_3(fea_cat3)
            if self.wiener_level >= 4:
                kfea4 = self.wiener_fix_KPN_4(fea_cat4_mid)
            else:
                kfea4 = self.wiener_fix_KPN_4(fea_cat4)
            if self.wiener_level >= 5:
                kfea5 = self.wiener_fix_KPN_5(fea_cat5_mid)
            else:
                kfea5 = self.wiener_fix_KPN_5(fea_cat5)
        else:
            if self.wiener_level >= 1:
                kfea1 = fea_cat1_skip
            else:
                kfea1 = fea_cat1
            if self.wiener_level >= 2:
                kfea2 = fea_cat2_skip
            else:
                kfea2 = fea_cat2
            if self.wiener_level >= 3:
                kfea3 = fea_cat3_skip
            else:
                kfea3 = fea_cat3
            if self.wiener_level >= 4:
                kfea4 = fea_cat4_skip
            else:
                kfea4 = fea_cat4
            if self.wiener_level >= 5:
                kfea5 = fea_cat5_skip
            else:
                kfea5 = fea_cat5

        # 1. get coeffs:
        mid_kfea5 = self.Coefficient_body_5(fea + kfea5)
        kernels_5 = self.dynamic_kernel_5(mid_kfea5)

        if self.final_kernel:
            mid_kfea4 = self.Coefficient_body_4(kfea4 + self.Coefficient_entry_4(mid_kfea5))
            mid_kfea3 = self.Coefficient_body_3(kfea3 + self.Coefficient_entry_3(mid_kfea4))
            mid_kfea2 = self.Coefficient_body_2(kfea2 + self.Coefficient_entry_2(mid_kfea3))
            mid_kfea1 = self.Coefficient_body_1(kfea1 + self.Coefficient_entry_1(mid_kfea2))
            coefficient_final = self.Coefficient_tail_final(mid_kfea1)
            coefficient_final = coefficient_final.permute(0, 2, 3, 1)
            coefficient_final = coefficient_final.reshape(
                [-1, coefficient_final.shape[1] * coefficient_final.shape[2], self.basis_num])

        # 2. get basises;
        basic_basis = self.Basis_body_1(x=basic_basis, skip=kfea4)
        basic_basis = self.Basis_body_2(x=basic_basis, skip=kfea3)
        basic_basis = self.Basis_body_3(x=basic_basis, skip=kfea2)
        basic_basis = self.Basis_body_4(x=basic_basis, skip=kfea1)

        if self.final_kernel:
            basis_final = self.Basis_tail_4_final(basic_basis)

        if self.final_kernel:
            kernels_final = torch.matmul(coefficient_final, basis_final).permute(0, 2, 1)
            kernels_final = kernels_final.reshape(
                [-1, self.kpn_sz * self.kpn_sz * (x.shape[1]) * 2, x.shape[2], x.shape[3]])

        # Dynamic convolution
        fea = Larger_Kernels_conv(fea, kernels_5, self.kpn_sz_center, self.bilinear_size, self.dilation_size)

        # Decoder
        de_fea = (self.conv_7(fea))
        if self.wiener_level >= 5:
            de_fea_cat1 = torch.cat([fea_cat5_skip, de_fea], 1)
        else:
            de_fea_cat1 = torch.cat([fea_cat5, de_fea], 1)
        de_fea = self.lrelu((self.conv_8(de_fea_cat1)))
        de_fea = (self.conv_9(de_fea))
        de_fea = self.lrelu(self.pixshuffle_1(de_fea))

        if self.wiener_level >= 4:
            de_fea_cat2 = torch.cat([fea_cat4_skip, de_fea], 1)
        else:
            de_fea_cat2 = torch.cat([fea_cat4, de_fea], 1)
        de_fea = self.lrelu((self.conv_10(de_fea_cat2)))
        de_fea = (self.conv_11(de_fea))
        de_fea = self.lrelu(self.pixshuffle_2(de_fea))

        if self.wiener_level >= 3:
            de_fea_cat3 = torch.cat([fea_cat3_skip, de_fea], 1)
        else:
            de_fea_cat3 = torch.cat([fea_cat3, de_fea], 1)
        de_fea = self.lrelu((self.conv_12(de_fea_cat3)))
        de_fea = (self.conv_13(de_fea))
        de_fea = self.lrelu(self.pixshuffle_3(de_fea))

        if self.wiener_level >= 2:
            de_fea_cat4 = torch.cat([fea_cat2_skip, de_fea], 1)
        else:
            de_fea_cat4 = torch.cat([fea_cat2, de_fea], 1)
        de_fea = self.lrelu((self.conv_14(de_fea_cat4)))
        de_fea = (self.conv_15(de_fea))
        de_fea = self.lrelu(self.pixshuffle_4(de_fea))

        if self.wiener_level >= 1:
            de_fea_cat5 = torch.cat([fea_cat1_skip, de_fea], 1)
        else:
            de_fea_cat5 = torch.cat([fea_cat1, de_fea], 1)
        de_fea = self.lrelu((self.conv_16(de_fea_cat5)))
        fea = self.conv_17(de_fea)

        if self.final_kernel:
            fea = Larger_Kernels_conv(fea, kernels_final, self.kpn_sz, self.bilinear_size, self.dilation_size)
            fea = self.final_mend(fea)

        out = fea

        if not self.training:
            out = out[:, :, :H, :W]
        return out


def init_wiener_kernel(ksz, channel_num):
    kernel = torch.zeros([1, channel_num, ksz, ksz])
    for i in range(kernel.shape[1]):
        kernel[:, i:i + 1, ksz // 2:ksz // 2 + 1, ksz // 2:ksz // 2 + 1] = 1
    return kernel

def NN_to_sum_1(T):
    T_sum = T.sum((-1, -2), keepdim=True)
    T = T / T_sum
    return T


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
                  dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


def act(act_type, inplace=True, neg_slope=0.1, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def upconv(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
           pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                      pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

    """
    Cascade Channel Attention Block, 3-3 style
    """

    def __init__(self, nc, gc, kernel_size=3, stride=1, dilation=1, groups=1, reduction=16, \
                 bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(CCAB, self).__init__()
        self.nc = nc
        self.RCAB = nn.ModuleList([RCAB(gc, kernel_size, reduction, stride, dilation, groups, bias, pad_type, \
                                        norm_type, act_type, mode, res_scale) for _ in range(nc)])
        self.CatBlocks = nn.ModuleList([conv_block((i + 2) * gc, gc, kernel_size=1, bias=bias, pad_type=pad_type, \
                                                   norm_type=norm_type, act_type=act_type, mode=mode) for i in
                                        range(nc)])

    def forward(self, x):
        pre_fea = x
        for i in range(self.nc):
            res = self.RCAB[i](x)
            pre_fea = torch.cat((pre_fea, res), dim=1)
            x = self.CatBlocks[i](pre_fea)
        return x


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResBlock(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False, dilation=1,
                 act_type='leakyrelu'):
        super().__init__()
        self.res_scale = res_scale
        padding = get_valid_padding(3, dilation)

        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, bias=True, \
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, bias=True, \
                               padding=padding, dilation=dilation)
        self.act = act(act_type) if act_type else None

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.act(self.conv1(x)))
        return identity + out * self.res_scale


def Larger_Kernels_conv(feature_in, Kernels, ksize, bilinear_size=7, dilation_size=4):
    '''
        [N, self.kpn_sz * self.kpn_sz * featuremap channel * 2, featuremap H, featuremap W]
    '''
    channels = feature_in.size(1)
    N, kernels_total, H, W = Kernels.size()
    Kernels = Kernels.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize, 2)
    Kernels = Kernels.permute(0, 1, 2, 3, 5, 4, 6).reshape(N, H, W, channels, ksize ** 2, -1)

    Kernels_A = Kernels[..., 0]
    Kernels_B = Kernels[..., 1]

    feature_out_High_frequency = kernel2d_conv(feature_in, Kernels_A, ksize)

    bilinear_Low_frequency = bilinear_conv(feature_in, bilinear_size)
    feature_out_Low_frequency = kernel2d_dilated_conv(bilinear_Low_frequency, Kernels_B, ksize, dilation=dilation_size)

    feature_out = feature_out_High_frequency + feature_out_Low_frequency

    return feature_out


def kernel2d_conv(feat_in, kernel, ksize):  #
    '''

    G3:dynamic_kernel torch.Size([2, 3200, 64, 64]) -> kernels torch.Size([2, 6400, 64, 64])
    '''
    channels = feat_in.size(1)
    N, H, W, C, ksz_2 = kernel.size()
    assert ksz_2 == ksize ** 2

    pad_sz = (ksize - 1) // 2  # pad size

    feat_in = F.pad(feat_in, (pad_sz, pad_sz, pad_sz, pad_sz), mode="replicate")

    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)  #

    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4)

    feat_in = feat_in.reshape(N, H, W, channels, -1)

    feat_out = torch.sum(feat_in * kernel, axis=-1)  #

    feat_out = feat_out.permute(0, 3, 1, 2)

    return feat_out


def bilinear_conv(feat_in, ksize):
    '''
        input:NCHW ; weight=filter:
        the shape of filter should be: out_channel, in_channel/groups, ksize_H,kize_W
    '''
    N, C, H, W = feat_in.shape

    if ksize == 3:
        kernel = np.array([0.5, 1., 0.5], dtype=np.float32).reshape([3, 1])
    elif ksize == 7:
        kernel = np.array([0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25],
                          dtype=np.float32).reshape([7, 1])
    kernel = np.matmul(kernel, kernel.T)
    kernel = kernel / np.sum(kernel)
    kernel = torch.from_numpy(kernel).to(feat_in.device)

    kernel = kernel.unsqueeze(0).unsqueeze(0)

    kernel = kernel.repeat(C, 1, 1, 1)

    feat_in = F.conv2d(feat_in, weight=kernel, groups=C, padding=(ksize - 1) // 2)
    return feat_in


def kernel2d_dilated_conv(feat_in, kernel, ksize, dilation=1):
    channels = feat_in.size(1)
    feat_in_batchsize = feat_in.size(0)
    feat_in_hei = feat_in.size(2)
    feat_in_wid = feat_in.size(3)
    N, H, W, C, ksz_2 = kernel.size()
    assert (ksz_2 == ksize ** 2) and (feat_in_batchsize == N) and (feat_in_hei == H) and (feat_in_wid == W)

    pad_sz = (ksize - 1) * dilation // 2  # pad size

    feat_in = F.pad(feat_in, (pad_sz, pad_sz, pad_sz, pad_sz), mode="replicate")

    feat_in = nn.Unfold(kernel_size=(ksize, ksize), dilation=dilation)(feat_in)
    feat_in = feat_in.reshape(N, -1, ksize, ksize, H, W)
    feat_in = feat_in.permute(0, 4, 5, 1, 3, 2)

    feat_in = feat_in.reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, axis=-1)
    feat_out = feat_out.permute(0, 3, 1, 2)
    return feat_out

class Kernel_UpBlock(nn.Module):
    def __init__(self, channel_in, block_channels, act_type, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.upconv = upconv(in_nc=channel_in, out_nc=block_channels, upscale_factor=2, kernel_size=3, stride=1)
        self.conv_1 = conv_block(in_nc=block_channels + channel_in, out_nc=block_channels, kernel_size=3, stride=1,
                                 act_type=act_type)
        self.conv_2 = ResBlock(block_channels, res_scale=1, act_type=act_type)

    def forward(self, x, skip):
        upconved = self.upconv(x)
        skip = torch.mean(skip, dim=[2, 3], keepdim=True)  # GAP to 1*1
        skip = skip.repeat([1, 1, self.upscale_factor * x.shape[2], self.upscale_factor * x.shape[3]])
        concated = torch.cat((upconved, skip), dim=1)
        out = self.conv_2(self.conv_1(concated))
        return out


class Basis_Decoder_Tail(nn.Module):
    def __init__(self, channel_in=128, channel_out=3, basis_num=90, kernel_size=15, act_type='leakyrelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.channel_out = channel_out
        self.basis_num = basis_num
        self.conv_without_padding = conv_block(in_nc=channel_in, out_nc=channel_in, kernel_size=2, stride=1,
                                               act_type=act_type, pad_type='None')
        self.final_conv_1 = conv_block(in_nc=channel_in, out_nc=channel_in, kernel_size=3, stride=1, act_type=act_type)
        self.final_conv_2 = conv_block(in_nc=channel_in, out_nc=channel_out * 2 * basis_num, kernel_size=3, stride=1,
                                       act_type=act_type)

    def forward(self, x):
        x = self.conv_without_padding(x)
        x = self.final_conv_2(self.final_conv_1(x))
        x = x.permute(0, 2, 3, 1)
        x = x.reshape([-1, self.kernel_size * self.kernel_size * self.channel_out * 2, self.basis_num])
        x = x.permute(0, 2, 1)
        return x


class CentralBlock(nn.Module):
    def __init__(self, nf=128):
        super(CentralBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_1 = nn.Conv2d(nf * 1, nf * 1, 3, 1, 1, bias=True)
        self.conv_2 = nn.Conv2d(nf * 1, nf * 1, 3, 1, 1, bias=True)
        self.conv_3 = nn.Conv2d(nf * 2, nf * 1, 3, 1, 1, bias=True)

    def forward(self, x):
        fea_skip = self.conv_1(x)
        fea = self.lrelu(fea_skip)
        fea = self.lrelu(self.conv_2(fea))
        fea = self.conv_3(torch.cat([fea, fea_skip], 1))
        return fea


class FinalBlock(nn.Module):
    def __init__(self, nf=128):
        super(FinalBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_1 = nn.Conv2d(nf * 1, nf * 1, 3, 1, 1, bias=True)
        self.conv_2 = nn.Conv2d(nf * 1, nf * 1, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = self.lrelu(self.conv_1(x))
        fea = self.conv_2(fea)
        return fea


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()

        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 + x


class ResidualDenseBlock_5C_wiener(nn.Module):
    def __init__(self, nf=64 * 3, gc=32 * 3, bias=True):
        super(ResidualDenseBlock_5C_wiener, self).__init__()

        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias, groups=3)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias, groups=3)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias, groups=3)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias, groups=3)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias, groups=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def seq(*args):
    if len(args) == 1:
        args = args[0]
    if isinstance(args, nn.Module):
        return args
    modules = OrderedDict()
    if isinstance(args, OrderedDict):
        for k, v in args.items():
            modules[k] = seq(v)
        return nn.Sequential(modules)
    assert isinstance(args, (list, tuple))
    return nn.Sequential(*[seq(i) for i in args])


def get_clear_feature(feature_in, kernels):
    device = feature_in.device
    ks = kernels.shape[2]
    first_scale_inblock_pad = F.pad(feature_in, (ks, ks, ks, ks), "replicate")
    constant_tensor_for_inv = torch.tensor([1, -1], device=device)
    clear_features_para = torch.zeros(feature_in.size(), device=device)
    clear_features_para = get_uperleft_denominator_para(first_scale_inblock_pad, kernels, constant_tensor_for_inv)
    clear_features_para = clear_features_para[:, :, ks:-ks, ks:-ks]
    return clear_features_para


def get_clear_feature_test(feature_in, kernels):
    device = feature_in.device
    clear_features = torch.zeros(feature_in.size(), device=device)
    ks = kernels.shape[2]
    first_scale_inblock_pad = F.pad(feature_in, (ks, ks, ks, ks), "replicate")
    for i in range(first_scale_inblock_pad.shape[1]):
        blur_feature_ch = first_scale_inblock_pad[:, i:i + 1, :, :]
        clear_feature_ch = get_uperleft_denominator_for_test(blur_feature_ch, kernels)
        clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks]
    return clear_features

def get_clear_feature_test_NCHW(feature_in, kernels):
    device = feature_in.device
    clear_features = torch.zeros(feature_in.size(), device=device)
    ks = kernels.shape[-1]
    first_scale_inblock_pad = F.pad(feature_in, (ks, ks, ks, ks), "replicate")
    for i in range(first_scale_inblock_pad.shape[1]):
        blur_feature_ch = first_scale_inblock_pad[:, i:i + 1, :, :]
        clear_feature_ch = get_uperleft_denominator_for_test(blur_feature_ch, kernels[:, i:i + 1, :, :])
        clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks]
    return clear_features

def get_clear_feature_NCHW(feature_in, kernels):
    device = feature_in.device
    ks = kernels.shape[-1]
    first_scale_inblock_pad = F.pad(feature_in, (ks, ks, ks, ks), "replicate")
    constant_tensor_for_inv = torch.tensor([1, -1], device=device)
    clear_features_para = get_uperleft_denominator_para_NCHW(first_scale_inblock_pad, kernels, constant_tensor_for_inv)
    clear_features_para = clear_features_para[:, :, ks:-ks, ks:-ks]
    return clear_features_para


def Downsize_and_Crop_Kernel(kernels, scale, mode, croped_ksz, odd=True):
    '''
    kernels:     torch.tensor, NCHW, kernels to be croped
    scale:       scale factor, float or tuple
    mode:        interploate method, str
    croped_ksz:  croped kernel size
    '''
    if scale != 1:
        downsized_kernel = F.interpolate(kernels, scale_factor=scale, mode=mode)
    else:
        downsized_kernel = kernels

    N, C, H, W = downsized_kernel.shape
    downsized_kernel_numpy = downsized_kernel.cpu().numpy()

    if H > croped_ksz:
        if croped_ksz % 2 == 0:
            if odd:
                R = croped_ksz + 1
            else:
                R = croped_ksz
        else:
            if odd:
                R = croped_ksz
            else:
                R = croped_ksz + 1
        wiener_kernel_processed = torch.zeros([N, C, R, R])
        r = R // 2

        for n in range(N):
            for c in range(C):
                h_mass, w_mass = scipy.ndimage.center_of_mass(downsized_kernel_numpy[n][c])
                h_mass = round(h_mass)
                w_mass = round(w_mass)
                if odd:
                    wiener_kernel_processed[n][c] = downsized_kernel[n][c][h_mass - r - 1:h_mass + r,
                                                    w_mass - r - 1:w_mass + r]
                else:
                    wiener_kernel_processed[n][c] = downsized_kernel[n][c][h_mass - r:h_mass + r, w_mass - r:w_mass + r]
    else:
        wiener_kernel_processed = downsized_kernel

    out = NN_to_sum_1(wiener_kernel_processed)
    return out


def Downsize_and_CropCenter(kernels, scale, mode, croped_ksz, odd):
    '''
    kernels:     torch.tensor, NCHW, kernels to be croped
    scale:       scale factor, float or tuple
    mode:        interploate method, str
    croped_ksz:  croped kernel size
    pad_to_odd:  bool, pad_to_odd or not
    '''
    if scale != 1:
        downsized_kernel = F.interpolate(kernels, scale_factor=scale, mode=mode)
    else:
        downsized_kernel = kernels

    H = downsized_kernel.shape[-2]
    W = downsized_kernel.shape[-1]
    h, w = math.ceil(H / 2), math.ceil(W / 2)
    r = math.floor(croped_ksz / 2)
    if H > 2 * r + 1:
        if odd:
            croped_kernel = downsized_kernel[:, :, h - r - 1:h + r, w - r - 1:w + r]
        else:
            croped_kernel = downsized_kernel[:, :, h - r:h + r, w - r:w + r]
    else:
        croped_kernel = downsized_kernel
    out = NN_to_sum_1(croped_kernel)
    return out


def aug_kernel_with_flags(kernels, flags):
    '''
    kernels: NChw
    flags:   N3
    '''
    kernels_aug = torch.zeros_like(kernels).to(kernels.device)
    for i in range(kernels.shape[0]):
        if flags[i][0] == 1:  # hflip
            kernels_aug[i] = torch.flip(kernels[i], [-1])
        if flags[i][1] == 1:  # vflip
            kernels_aug[i] = torch.flip(kernels[i], [-2])
        if flags[i][2] == 1:  # rot90
            kernels_aug[i] = kernels[i].permute(0, 2, 1)  # .permute(0, 1, 3, 2)
        if (flags[i][0] != 1 and flags[i][1] != 1 and flags[i][2] != 1):
            kernels_aug[i] = kernels[i]
    return kernels_aug

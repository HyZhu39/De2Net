import functools
import torch
import torch.nn as nn
from torch.nn import init as init
import torch.nn.functional as F
import numpy as np
from basicsr.utils import get_uperleft_denominator, get_uperleft_denominator_for_test, get_uperleft_denominator_para, get_uperleft_denominator_para_NCHW
import scipy

# import pdb
import math


class ZTE(nn.Module):
    def __init__(self, in_nc, out_nc, nf=32, ns=64, kpn_sz=15, norm_type=None, \
                 act_type='leakyrelu', res_scale=1, kernel_size=3, \
                 kernel_cond=None, psf_nc=5, multi_scale=False, final_kernel=False, \
                 bilinear_sz=7, dilation_sz=4, basis_num=90, \
                 kpn_sz_center=5, \
                 croped_ksz=64, odd=True, mode='bilinear', \
                 wiener_level=3):
        super().__init__()
        self.ns_group = ns * 3
        self.ns = ns
        self.kpn_sz = kpn_sz
        self.kernel_cond = kernel_cond
        self.multi_scale = multi_scale
        self.bilinear_size = bilinear_sz
        self.dilation_size = dilation_sz
        self.basis_num = basis_num
        self.final_kernel = final_kernel
        self.kpn_sz_center = kpn_sz_center

        self.odd = odd
        self.croped_ksz = croped_ksz
        self.mode = mode

        self.wiener_level = math.floor(wiener_level)  # nums of wiener
        #############################
        # Restoration Branch
        #############################
        self.avg_pool = nn.AvgPool2d(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Encoder
        if self.wiener_level >= 1:
            self.conv_1 = nn.Conv2d(in_nc, self.ns_group, 3, 1, 1, bias=True, groups=3)
            basic_block_64 = functools.partial(ResidualDenseBlock_5C_wiener, nf=self.ns_group)
            self.encoder1 = make_layer(basic_block_64, 2)  # 64f

            self.wiener_fix_1 = nn.Conv2d(self.ns_group, ns, 1, 1, 0, bias=True)
        else:
            self.conv_1 = nn.Conv2d(in_nc, ns, 3, 1, 1, bias=True)
            basic_block_64 = functools.partial(ResidualDenseBlock_5C, nf=ns)
            self.encoder1 = make_layer(basic_block_64, 2)

        if self.wiener_level >= 2:
            self.conv_2 = nn.Conv2d(self.ns_group, self.ns_group * 2, 3, 1, 1, bias=True, groups=3)
            basic_block_128 = functools.partial(ResidualDenseBlock_5C_wiener, nf=self.ns_group * 2)
            self.encoder2 = make_layer(basic_block_128, 2)

            self.wiener_fix_2 = nn.Conv2d(self.ns_group * 2, ns * 2, 1, 1, 0, bias=True)
        else:
            self.conv_2 = nn.Conv2d(ns, ns * 2, 3, 1, 1, bias=True)
            basic_block_128 = functools.partial(ResidualDenseBlock_5C, nf=ns * 2)
            self.encoder2 = make_layer(basic_block_128, 2)

        if self.wiener_level >= 3:
            self.conv_3 = nn.Conv2d(self.ns_group * 2, self.ns_group * 4, 3, 1, 1, bias=True, groups=3)
            basic_block_256 = functools.partial(ResidualDenseBlock_5C_wiener, nf=self.ns_group * 4)
            self.encoder3 = make_layer(basic_block_256, 2)

            self.wiener_fix_3 = nn.Conv2d(self.ns_group * 4, ns * 4, 1, 1, 0, bias=True)
        else:
            self.conv_3 = nn.Conv2d(ns * 2, ns * 4, 3, 1, 1, bias=True)
            basic_block_256 = functools.partial(ResidualDenseBlock_5C, nf=ns * 4)
            self.encoder3 = make_layer(basic_block_256, 2)

        if self.wiener_level >= 4:
            self.conv_4 = nn.Conv2d(self.ns_group * 4, self.ns_group * 4, 3, 1, 1, bias=True, groups=3)

            self.wiener_fix_4 = nn.Conv2d(self.ns_group * 4, ns * 4, 1, 1, 0, bias=True)
        else:
            self.conv_4 = nn.Conv2d(ns * 4, ns * 4, 3, 1, 1, bias=True)

        self.conv_5 = nn.Conv2d(ns * 4, 256, 3, 1, 1, bias=True)

        # Decoder
        self.conv_6 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        self.conv_7 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.conv_8 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.conv_9 = nn.Conv2d(384, 256, 3, 1, padding=1, bias=True)
        self.conv_10 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        self.conv_11 = nn.Conv2d(192, 128, 3, 1, padding=1, bias=True)
        self.conv_12 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=True)
        self.pixshuffle_1 = nn.PixelShuffle(upscale_factor=2)
        self.pixshuffle_2 = nn.PixelShuffle(upscale_factor=2)
        self.pixshuffle_3 = nn.PixelShuffle(upscale_factor=2)
        self.conv_13 = nn.Conv2d(96, 64, 3, 1, padding=1, bias=True)
        self.conv_14 = nn.Conv2d(64, 3, 3, 1, padding=1, bias=True)
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

            # Condition Encoder
            self.kconv_11 = conv_block(cond_nc, nf, kernel_size=kernel_size, act_type=act_type)
            self.kconv_12 = ResBlock(nf, res_scale=res_scale, act_type=act_type)
            self.kconv_13 = ResBlock(nf, res_scale=res_scale, act_type=act_type)

            self.kconv_21 = conv_block(nf, 2 * nf, stride=2, kernel_size=4, act_type=act_type)
            self.kconv_22 = ResBlock(2 * nf, res_scale=res_scale, act_type=act_type)
            self.kconv_23 = ResBlock(2 * nf, res_scale=res_scale, act_type=act_type)

            self.kconv_31 = conv_block(2 * nf, 4 * nf, stride=2, kernel_size=4, act_type=act_type)
            self.kconv_32 = ResBlock(4 * nf, res_scale=res_scale, act_type=act_type)
            self.kconv_33 = ResBlock(4 * nf, res_scale=res_scale, act_type=act_type)

            self.kconv_41 = conv_block(4 * nf, 8 * nf, stride=2, kernel_size=4, act_type=act_type)
            self.kconv_42 = ResBlock(8 * nf, res_scale=res_scale, act_type=act_type)
            self.kconv_43 = ResBlock(8 * nf, res_scale=res_scale, act_type=act_type)

            # Coefficient Decoder:
            self.Coefficient_body_4 = nn.Sequential(
                conv_block(8 * nf, 8 * nf, kernel_size=kernel_size),
                ResBlock(8 * nf, res_scale=res_scale, act_type=act_type),
                ResBlock(8 * nf, res_scale=res_scale, act_type=act_type))
            self.Coefficient_tail_4 = conv_block(8 * nf, self.basis_num, kernel_size=1)

            if self.final_kernel:
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
            self.Basis_tail_4 = Basis_Decoder_Tail(channel_in=1 * nf, channel_out=8 * nf,
                                                   basis_num=self.basis_num, kernel_size=self.kpn_sz,
                                                   act_type='prelu')
            if self.final_kernel:
                self.Basis_tail_4_final = Basis_Decoder_Tail(channel_in=1 * nf, channel_out=out_nc,
                                                             basis_num=self.basis_num, kernel_size=self.kpn_sz,
                                                             act_type='prelu')

    def forward(self, x, psf=None, wiener_kernel_1=None, wiener_kernel_2=None, wiener_kernel_3=None, wiener_kernel_4=None, wiener_kernel_5=None):
        if not self.training:
            N, C, H, W = x.shape
            H_pad = 8 - H % 8 if not H % 8 == 0 else 0
            W_pad = 8 - W % 8 if not W % 8 == 0 else 0
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
                tmpR = wiener_kernel_1[:,0:1,:,:].repeat(1, self.ns, 1, 1)
                tmpG = wiener_kernel_1[:,1:2,:,:].repeat(1, self.ns, 1, 1)
                tmpB = wiener_kernel_1[:,2:3,:,:].repeat(1, self.ns, 1, 1)
                wiener_kernels_1 = torch.cat([tmpR, tmpG, tmpB], dim=1)
                del tmpR, tmpG, tmpB

                fea_cat1_skip = get_clear_feature_NCHW(fea_cat1, wiener_kernels_1)
            else:
                fea_x_R = get_clear_feature_test(fea_cat1[:,0          :self.ns    ,:,:], wiener_kernel_1[:,0:1,:,:])
                fea_x_G = get_clear_feature_test(fea_cat1[:,self.ns    :self.ns * 2,:,:], wiener_kernel_1[:,1:2,:,:])
                fea_x_B = get_clear_feature_test(fea_cat1[:,self.ns * 2:self.ns_group        ,:,:], wiener_kernel_1[:,2:3,:,:])
                fea_cat1_skip = torch.cat([fea_x_R, fea_x_G, fea_x_B], dim=1)
                del fea_x_R, fea_x_G, fea_x_B

            fea_cat1_skip = self.wiener_fix_1(fea_cat1_skip)
            if self.wiener_level == 1:
                fea_cat1 = fea_cat1_skip

        fea = self.avg_pool(fea_cat1)
        fea = self.lrelu(self.conv_2(fea))
        fea_cat2 = self.encoder2(fea)
        if self.wiener_level >= 2:
            if self.training:
                tmpR = wiener_kernel_2[:,0:1,:,:].repeat(1, self.ns*2, 1, 1)
                tmpG = wiener_kernel_2[:,1:2,:,:].repeat(1, self.ns*2, 1, 1)
                tmpB = wiener_kernel_2[:,2:3,:,:].repeat(1, self.ns*2, 1, 1)
                wiener_kernels_2 = torch.cat([tmpR, tmpG, tmpB], dim=1)
                fea_cat2_skip = get_clear_feature_NCHW(fea_cat2, wiener_kernels_2)
                del tmpR, tmpG, tmpB
            else:
                fea_x_R = get_clear_feature_test(fea_cat2[:,0            :self.ns*2    ,:,:], wiener_kernel_2[:,0:1,:,:])
                fea_x_G = get_clear_feature_test(fea_cat2[:,self.ns*2    :self.ns*2 * 2,:,:], wiener_kernel_2[:,1:2,:,:])
                fea_x_B = get_clear_feature_test(fea_cat2[:,self.ns*2 * 2:self.ns_group*2    ,:,:], wiener_kernel_2[:,2:3,:,:])
                fea_cat2_skip = torch.cat([fea_x_R, fea_x_G, fea_x_B], dim=1)
                del fea_x_R, fea_x_G, fea_x_B

            fea_cat2_skip = self.wiener_fix_2(fea_cat2_skip)
            if self.wiener_level == 2:
                fea_cat2 = fea_cat2_skip

        fea = self.avg_pool(fea_cat2)
        fea = self.lrelu(self.conv_3(fea))
        fea_cat3 = self.encoder3(fea)
        if self.wiener_level >= 3:
            if self.training:
                tmpR = wiener_kernel_3[:,0:1,:,:].repeat(1, self.ns*4, 1, 1)
                tmpG = wiener_kernel_3[:,1:2,:,:].repeat(1, self.ns*4, 1, 1)
                tmpB = wiener_kernel_3[:,2:3,:,:].repeat(1, self.ns*4, 1, 1)
                wiener_kernels_3 = torch.cat([tmpR, tmpG, tmpB], dim=1)
                fea_cat3_skip = get_clear_feature_NCHW(fea_cat3, wiener_kernels_3)
                del tmpR, tmpG, tmpB
            else:
                fea_x_R = get_clear_feature_test(fea_cat3[:,0            :self.ns*4    ,:,:], wiener_kernel_3[:,0:1,:,:])
                fea_x_G = get_clear_feature_test(fea_cat3[:,self.ns*4    :self.ns*4 * 2,:,:], wiener_kernel_3[:,1:2,:,:])
                fea_x_B = get_clear_feature_test(fea_cat3[:,self.ns*4 * 2:self.ns_group*4    ,:,:], wiener_kernel_3[:,2:3,:,:])
                fea_cat3_skip = torch.cat([fea_x_R, fea_x_G, fea_x_B], dim=1)
                del fea_x_R, fea_x_G, fea_x_B

            fea_cat3_skip = self.wiener_fix_3(fea_cat3_skip)
            if self.wiener_level == 3:
                fea_cat3 = fea_cat3_skip

        fea = self.avg_pool(fea_cat3)
        fea_cat4 = self.conv_4(fea)
        if self.wiener_level >= 4:
            if self.training:
                tmpR = wiener_kernel_4[:,0:1,:,:].repeat(1, self.ns*4, 1, 1)
                tmpG = wiener_kernel_4[:,1:2,:,:].repeat(1, self.ns*4, 1, 1)
                tmpB = wiener_kernel_4[:,2:3,:,:].repeat(1, self.ns*4, 1, 1)
                wiener_kernels_4 = torch.cat([tmpR, tmpG, tmpB], dim=1)
                fea_cat4_skip = get_clear_feature_NCHW(fea_cat4, wiener_kernels_4)
                del tmpR, tmpG, tmpB
            else:
                fea_x_R = get_clear_feature_test(fea_cat4[:,0            :self.ns*4    ,:,:], wiener_kernel_4[:,0:1,:,:])
                fea_x_G = get_clear_feature_test(fea_cat4[:,self.ns*4    :self.ns*4 * 2,:,:], wiener_kernel_4[:,1:2,:,:])
                fea_x_B = get_clear_feature_test(fea_cat4[:,self.ns*4 * 2:self.ns_group*4    ,:,:], wiener_kernel_4[:,2:3,:,:])
                fea_cat4_skip = torch.cat([fea_x_R, fea_x_G, fea_x_B], dim=1)
                del fea_x_R, fea_x_G, fea_x_B

            fea_cat4_skip = self.wiener_fix_4(fea_cat4_skip)
            if self.wiener_level == 4:
                fea_cat4 = fea_cat4_skip

        fea = self.lrelu(fea_cat4)
        fea = self.conv_5(fea)

        #############################
        # Kernel Prediction Branch
        #############################

        if self.kernel_cond:

            if self.kernel_cond == 'img':
                cond_x = x
            elif self.kernel_cond == 'psf':
                cond_x = psf.expand(-1, -1, x.shape[2], x.shape[3])
            elif self.kernel_cond == 'img-psf':
                cond_x = psf.expand(-1, -1, x.shape[2], x.shape[3])
                cond_x = torch.cat((cond_x, x), dim=1)

            kfea1 = self.kconv_13(self.kconv_12(self.kconv_11(cond_x)))
            kfea2 = self.kconv_23(self.kconv_22(self.kconv_21(kfea1)))
            kfea3 = self.kconv_33(self.kconv_32(self.kconv_31(kfea2)))
            # for basis input:
            kfea4 = self.kconv_43(self.kconv_42(self.kconv_41(kfea3)))
            basic_basis = torch.mean(kfea4, dim=[2, 3], keepdim=True)

            # 1. get coeffs:
            mid_kfea4 = self.Coefficient_body_4(kfea4)
            coefficient_4 = self.Coefficient_tail_4(mid_kfea4)
            coefficient_4 = coefficient_4.permute(0, 2, 3, 1)
            coefficient_4 = coefficient_4.reshape([-1, coefficient_4.shape[1] * coefficient_4.shape[2], self.basis_num])

            if self.final_kernel:
                mid_kfea3 = self.Coefficient_body_3(kfea3 + self.Coefficient_entry_3(mid_kfea4))
                mid_kfea2 = self.Coefficient_body_2(kfea2 + self.Coefficient_entry_2(mid_kfea3))
                mid_kfea1 = self.Coefficient_body_1(kfea1 + self.Coefficient_entry_1(mid_kfea2))
                coefficient_final = self.Coefficient_tail_final(mid_kfea1)
                coefficient_final = coefficient_final.permute(0, 2, 3, 1)
                coefficient_final = coefficient_final.reshape([-1, coefficient_final.shape[1] * coefficient_final.shape[2], self.basis_num])

            # 2. get basises;
            basic_basis = self.Basis_body_1(x=basic_basis, skip=kfea4)
            basic_basis = self.Basis_body_2(x=basic_basis, skip=kfea3)
            basic_basis = self.Basis_body_3(x=basic_basis, skip=kfea2)
            basic_basis = self.Basis_body_4(x=basic_basis, skip=kfea1)

            basis_4 = self.Basis_tail_4(basic_basis)
            kernels_4 = torch.matmul(coefficient_4, basis_4).permute(0, 2, 1)
            kernels_4 = kernels_4.reshape([-1, self.kpn_sz * self.kpn_sz * (fea_cat4.shape[1]) * 2, fea_cat4.shape[2], fea_cat4.shape[3]])

            if self.final_kernel:
                basis_final = self.Basis_tail_4_final(basic_basis)

            if self.final_kernel:
                kernels_final = torch.matmul(coefficient_final, basis_final).permute(0, 2, 1)
                kernels_final = kernels_final.reshape([-1, self.kpn_sz * self.kpn_sz * (x.shape[1]) * 2, x.shape[2], x.shape[3]])

        # Dynamic convolution
        fea = Larger_Kernels_conv(fea, kernels_4, self.kpn_sz, self.bilinear_size, self.dilation_size)

        # Decoder
        de_fea = (self.conv_6(fea))
        if self.wiener_level >= 4:
            de_fea_cat1 = torch.cat([fea_cat4_skip, de_fea], 1)
        else:
            de_fea_cat1 = torch.cat([fea_cat4, de_fea], 1)
        de_fea = self.lrelu((self.conv_7(de_fea_cat1)))
        de_fea = (self.conv_8(de_fea))
        de_fea = self.lrelu(self.pixshuffle_1(de_fea))

        if self.wiener_level >= 3:
            de_fea_cat2 = torch.cat([fea_cat3_skip, de_fea], 1)
        else:
            de_fea_cat2 = torch.cat([fea_cat3, de_fea], 1)
        de_fea = self.lrelu((self.conv_9(de_fea_cat2)))
        de_fea = (self.conv_10(de_fea))
        de_fea = self.lrelu(self.pixshuffle_2(de_fea))

        if self.wiener_level >= 2:
            de_fea_cat3 = torch.cat([fea_cat2_skip, de_fea], 1)
        else:
            de_fea_cat3 = torch.cat([fea_cat2, de_fea], 1)
        de_fea = self.lrelu((self.conv_11(de_fea_cat3)))
        de_fea = (self.conv_12(de_fea))
        de_fea = self.lrelu(self.pixshuffle_3(de_fea))

        if self.wiener_level >= 1:
            de_fea_cat4 = torch.cat([fea_cat1_skip, de_fea], 1)
        else:
            de_fea_cat4 = torch.cat([fea_cat1, de_fea], 1)
        de_fea = self.lrelu((self.conv_13(de_fea_cat4)))
        fea = self.conv_14(de_fea)

        if self.final_kernel:
            fea = Larger_Kernels_conv(fea, kernels_final, self.kpn_sz, self.bilinear_size, self.dilation_size)
            fea = self.final_mend(fea)

        out = fea

        if not self.training:
            out = out[:, :, :H, :W]
        return out

def init_wiener_kernel(wiener_sz, wiener_init_path, odd=True, mode='crop', downscale_factor=4):
    wiener_kernel_numpy = np.load(wiener_init_path)
    wiener_kernel = torch.from_numpy(wiener_kernel_numpy)
    N, C, H, W = wiener_kernel.shape

    if mode=='crop':
        if wiener_sz % 2 == 0:
            if odd:
                R = wiener_sz + 1
            else:
                R = wiener_sz
        else:
            if odd:
                R = wiener_sz
            else:
                R = wiener_sz + 1

        wiener_kernel_processed = torch.zeros([N, C, R, R])
        r = R // 2

        for c in range(C):
            h_mass, w_mass = scipy.ndimage.center_of_mass(wiener_kernel_numpy[0][c])
            h_mass = round(h_mass)
            w_mass = round(w_mass)
            if odd:
                wiener_kernel_processed[0][c] = wiener_kernel[0][c][h_mass - r-1:h_mass + r, w_mass - r-1:w_mass + r]
            else:
                wiener_kernel_processed[0][c] = wiener_kernel[0][c][h_mass - r:h_mass + r, w_mass - r:w_mass + r]

    if mode=='downscale':
        h, w = math.ceil(H/2), math.ceil(W/2)

        if H != W:
            if H>W:
                long_side = H
                short_side = W
                m = (long_side - short_side) // 2
                wiener_kernel_dsz = wiener_kernel[:, :, h-m:h+m, :]
            else:
                long_side = W
                short_side = H
                m = (long_side - short_side) // 2
                wiener_kernel_dsz = wiener_kernel[:, :, :, w-m:w+m]
        else:
            wiener_kernel_dsz = wiener_kernel

        wiener_kernel_processed = F.interpolate(wiener_kernel_dsz, scale_factor=1/downscale_factor, mode='bilinear')


    wiener_kernel_processed = NN_to_sum_1(wiener_kernel_processed)
    return wiener_kernel_processed

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

        G3:dynamic_kernel torch.Size([2, 3200, 64, 64]) -> kernels torch.Size([2, 6400, 64, 64])
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

    feat_out = feat_out.permute(0, 3, 1, 2)  # .contiguous()

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
    def __init__(self, nf=64*3, gc=32*3, bias=True):
        super(ResidualDenseBlock_5C_wiener, self).__init__()

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

    if W > croped_ksz:
        if croped_ksz % 2 == 0:  # make sure final kernel size is odd
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
                    wiener_kernel_processed[n][c] = downsized_kernel[n][c][h_mass - r-1:h_mass + r, w_mass - r-1:w_mass + r]
                else:
                    wiener_kernel_processed[n][c] = downsized_kernel[n][c][h_mass - r:h_mass + r, w_mass - r:w_mass + r]
    else:
        wiener_kernel_processed = downsized_kernel

    out = NN_to_sum_1(wiener_kernel_processed)
    return out

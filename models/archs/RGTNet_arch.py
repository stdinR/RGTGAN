import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.net_utils import *
from scipy.misc import imresize, imread

# deformable attn
# import models.archs.arch_util as arch_util
# from models.archs.dcn_v2 import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from ops_dcnv3.modules.dcnv3 import DCNv3 as DCN
except ImportError:
    raise ImportError('Failed to import DCNv3 module.')

class RGTNet(nn.Module):
    """
    PyTorch Module for RGTNet.
    Now only x4 is supported.

    Parameters
    ---
    ngf : int, optional
        the number of filterd of generator.
    n_blocks : int, optional
        the number of residual blocks for each module.
    """
    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(RGTNet, self).__init__()
        self.get_g_nopadding = Get_gradient_nopadding()
        self.Encoder = Encoder(3, nf=ngf)
        self.Encoder_grad_ori = Encoder(3, nf=int(ngf/4))

        self.gpcd_align = GPCD_Align(nf=ngf+int(ngf/4), nf_out=ngf, groups=groups) # self.gpcd_align = GPCD_Align(nf=ngf+int(ngf/4), nf_out=ngf, groups=groups)
        self.content_extractor = ContentExtractor(ngf, n_blocks)
        self.texture_transfer = TextureTransfer(ngf, n_blocks)
        #init_weights(self, init_type='normal', init_gain=0.02)
        self.GTEM = GTEM_F(3, nf=ngf)


    def forward(self, LR, LR_UX4, Ref,Ref_DUX4, weights=None):
        LR_UX4_grad = self.get_g_nopadding(LR_UX4)
        Ref_DUX4_grad = self.get_g_nopadding(Ref_DUX4)
        Ref_grad = self.get_g_nopadding(Ref)


        LR_conv1, LR_conv2, LR_conv3 = self.Encoder(LR_UX4) # LR_UX4
        HR1_conv1, HR1_conv2, HR1_conv3 = self.Encoder(Ref_DUX4) # Ref_DUX4
        HR2_conv1, HR2_conv2, HR2_conv3 = self.Encoder(Ref) # Ref


        LR_UX4_grad_GTEM = self.GTEM(LR_UX4_grad)
        Ref_DUX4_grad_GTEM = self.GTEM(Ref_DUX4_grad)
        Ref_grad_GTEM = self.GTEM(Ref_grad)


        #grad revised into GTEM
        LR_conv1_grad, LR_conv2_grad, LR_conv3_grad = self.Encoder_grad_ori(LR_UX4_grad_GTEM)
        HR1_conv1_grad, HR1_conv2_grad, HR1_conv3_grad = self.Encoder_grad_ori(Ref_DUX4_grad_GTEM)
        HR2_conv1_grad, HR2_conv2_grad, HR2_conv3_grad = self.Encoder_grad_ori(Ref_grad_GTEM)


        LR_conv1 = torch.cat([LR_conv1,LR_conv1_grad], dim=1)
        LR_conv2 = torch.cat([LR_conv2,LR_conv2_grad], dim=1)
        LR_conv3 = torch.cat([LR_conv3,LR_conv3_grad], dim=1)
        HR2_conv1 = torch.cat([HR2_conv1,HR2_conv1_grad], dim=1)
        HR2_conv2 = torch.cat([HR2_conv2,HR2_conv2_grad], dim=1)
        HR2_conv3 = torch.cat([HR2_conv3,HR2_conv3_grad], dim=1)
        HR1_conv1 = torch.cat([HR1_conv1,HR1_conv1_grad], dim=1)
        HR1_conv2 = torch.cat([HR1_conv2,HR1_conv2_grad], dim=1)
        HR1_conv3 = torch.cat([HR1_conv3,HR1_conv3_grad], dim=1)

        LR_fea_l = [LR_conv1, LR_conv2, LR_conv3]
        Ref_use_fea_l = [HR2_conv1, HR2_conv2, HR2_conv3]
        Ref_fea_l = [HR1_conv1, HR1_conv2, HR1_conv3]

        Ref_conv1, Ref_conv2, Ref_conv3 = self.gpcd_align(Ref_use_fea_l,Ref_fea_l,LR_fea_l)
        maps = [Ref_conv1, Ref_conv2, Ref_conv3]

        # upscale:
        base = LR_UX4

        # fusion:
        upscale_plain, content_feat = self.content_extractor(LR)
        upscale_RGTNet = self.texture_transfer(content_feat, maps)

        return upscale_RGTNet + base

class GPCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, nf_out=64, groups=8, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=8, use_checkpoint=False, n_blocks=16):
        # self, nf=64,nf_out=64, groups=8
        super(GPCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        nff = groups * 3 * 3 * 2
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_offset_head_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.DenseRes = DenseResNet3(nf=80, ngf=80)
        self.L3_offset_head_conv2 = nn.Conv2d(80, nf, 3, 1, 1, bias=True)
        self.L3_offset_tail = nn.Sequential(*[ResBlock(n_filters=80) for _ in range(n_blocks)],)

        #self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L3_dcnpack = DCN(channels=80, kernel_size=3, dw_kernel_size=None, stride=1, pad=1, dilation=1, group=groups, offset_scale=1.0)
        # Attn
        # self.L3_dcnpack_attn = SwinBlock(img_size=480, embed_dim=nf_out, depths=depths, num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint)
        # 160
        # End of Attn
        self.L3_fea_conv = nn.Conv2d(nf, nf_out, 3, 1, 1, bias=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_offset_head_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_head_conv2 = nn.Conv2d(80, nf, 3, 1, 1, bias=True)
        self.L2_offset_tail = nn.Sequential(*[ResBlock(n_filters=80) for _ in range(n_blocks)], )

        # self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L2_dcnpack = DCN(channels = 80, kernel_size = 3, dw_kernel_size=None, stride=1, pad=1, dilation=1, group=groups, offset_scale=1.0)
        self.L2_fea_conv = nn.Conv2d(nf + nf_out, nf_out, 3, 1, 1, bias=True)
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_offset_head_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_head_conv2 = nn.Conv2d(80, nf, 3, 1, 1, bias=True)
        self.L1_offset_tail = nn.Sequential(*[ResBlock(n_filters=80) for _ in range(n_blocks)], )

        # self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L1_dcnpack = DCN(nf, 3, dw_kernel_size=None, stride=1, pad=1, dilation=1, group=groups, offset_scale=1.0)
        self.L1_fea_conv = nn.Conv2d(nf + nf_out, nf, 3, 1, 1, bias=True)
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.offset_head_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offset_head_conv2 = nn.Conv2d(80, nf, 3, 1, 1, bias=True)
        self.offset_tail = nn.Sequential(*[ResBlock(n_filters=80) for _ in range(n_blocks)], )

        # self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.cas_dcnpack = DCN(nf, 3, dw_kernel_size=None, stride=1, pad=1, dilation=1, group=groups, offset_scale=1.0)

        self.cas_fea_conv = nn.Conv2d(nf, nf_out, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, Ref_use_fea_l, Ref_fea_l, LR_fea_l):
        '''
        Ref_use_fea_l, Ref_fea_l, LR_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        # B, C, H, W = Ref_fea_l[2].shape
        # print(C)

        L3_offset = torch.cat([Ref_fea_l[2], LR_fea_l[2]], dim=1)
        # print('Ref_fea_l, LR_fea_l, L3_offset: ', Ref_fea_l[2].shape, LR_fea_l[2].shape, L3_offset.shape)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        ## optimize L3 offset due to DIDConv
        L3_offset_tail = self.DenseRes(Ref_fea_l[2], LR_fea_l[2])
        ## end of L3_offset_tail (for IDConv)
        Ref_use_fea_l[2] = torch.permute(Ref_use_fea_l[2], [0, 2, 3, 1])
        L3_offset = torch.permute(L3_offset, [0, 2, 3, 1])
        L3_fea = self.L3_dcnpack(Ref_use_fea_l[2], L3_offset_tail.permute(0,2,3,1)) # L3_offset_tail
        L3_fea = torch.permute(L3_fea, [0, 3, 1, 2])
        L3_offset = torch.permute(L3_offset, [0, 3, 1, 2])
        L3_fea_output = self.lrelu(self.L3_fea_conv(L3_fea))

        # L2
        L2_offset = torch.cat([Ref_fea_l[1], LR_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_offset_tail = self.DenseRes(Ref_fea_l[1], LR_fea_l[1])
        Ref_use_fea_l[1] = torch.permute(Ref_use_fea_l[1], [0, 2, 3, 1])
        L2_offset = torch.permute(L2_offset, [0, 2, 3, 1])
        L2_fea = self.L2_dcnpack(Ref_use_fea_l[1], L2_offset_tail.permute(0, 2, 3, 1)) # L2_offset_tail
        L2_fea = torch.permute(L2_fea, [0, 3, 1, 2])
        L2_offset = torch.permute(L2_offset, [0, 3, 1, 2])
        L3_fea = F.interpolate(L3_fea_output, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea_output = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))

        # L1
        L1_offset = torch.cat([Ref_fea_l[0], LR_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_offset_tail = self.DenseRes(Ref_fea_l[0], LR_fea_l[0])
        Ref_use_fea_l[0] = torch.permute(Ref_use_fea_l[0], [0, 2, 3, 1])
        L1_offset = torch.permute(L1_offset, [0, 2, 3, 1])
        L1_fea = self.L1_dcnpack(Ref_use_fea_l[0], L1_offset_tail.permute([0,2,3,1])) # L1_offset_tail
        L1_fea = torch.permute(L1_fea, [0, 3, 1, 2])
        L2_fea = F.interpolate(L2_fea_output, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))

        # Cascading
        offset = torch.cat([L1_fea, LR_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        offset_tail = self.DenseRes(L1_fea, LR_fea_l[0])
        L1_fea = torch.permute(L1_fea, [0, 2, 3, 1])
        offset = torch.permute(offset, [0, 2, 3, 1])
        L1_fea_output = self.cas_dcnpack(L1_fea, offset_tail.permute([0,2,3,1])) # offset_tail
        L1_fea_output = torch.permute(L1_fea_output, [0, 3, 1, 2])
        L1_fea_output = self.lrelu(self.cas_fea_conv(L1_fea_output))

        return L1_fea_output, L2_fea_output, L3_fea_output


class ContentExtractor(nn.Module):
    def __init__(self, ngf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        self.tail = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1),
            # nn.Tanh()
        )

    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        upscale = self.tail(h)
        return upscale, h


class TextureTransfer(nn.Module):
    def __init__(self, ngf=64, n_blocks=16, activation = 'selu', init_type = 'w_init'):
        super(TextureTransfer, self).__init__()

        # for small scale
        self.ram_head_small = RAM()      
        self.head_small = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )

        self.body_small = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        self.tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        # for medium scale
        self.ram_head_medium = RAM()
        self.head_medium = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )

        self.body_medium = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        self.tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        # for large scale
        self.ram_head_large = RAM()
        self.head_large = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )

        self.body_large = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1),
        )

        # for dense:
        self.dconv_ud1 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_ud2 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_ud3 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_ud4 = deconv_activation(256, 64, kernel_size=1, stride=1, padding=0, activation=activation)
        self.dconv_ud5 = deconv_activation(256, 64, kernel_size=1, stride=1, padding=0, activation=activation)
        self.dconv_ud6 = deconv_activation(256, 64, kernel_size=1, stride=1, padding=0, activation=activation)
        self.dconv_ud7 = deconv_activation(192, 64, kernel_size=3, stride=1, padding=1, activation=activation)

    def forward(self, x, maps):

        h = self.ram_head_small(maps[2], x)
        h = torch.cat([x, h], 1)
        h = self.head_small(h)


        #dense small
        for i in range(3):
            x1=x2=x3=h
            for j in range(3):
                a1 = F.leaky_relu(self.dconv_ud1(x1))
                a2 = F.leaky_relu(self.dconv_ud2(x2))
                a3 = F.leaky_relu(self.dconv_ud3(x3))
                sum = torch.concat([a1,a2,a3],1)

                x1 = F.leaky_relu(self.dconv_ud4(torch.concat([sum,x1],1)))#.permute(0,2,3,1) /0,3,1,2
                x2 = F.leaky_relu(self.dconv_ud5(torch.concat([sum,x2],1)))
                x3 = F.leaky_relu(self.dconv_ud6(torch.concat([sum,x3],1)))
            block_out = F.leaky_relu(self.dconv_ud7(torch.concat([x1,x2,x3],1)))
            h = h + block_out
        # end of dense small

        h = self.body_small(h) + x
        x = self.tail_small(h)

        # medium scale
        h = self.ram_head_medium(maps[1],x)
        h = torch.cat([x, h], 1)
        h = self.head_medium(h)

        # dense medium
        for i in range(3):
            x1 = x2 = x3 = h
            for j in range(3):
                a1 = F.leaky_relu(self.dconv_ud1(x1))
                a2 = F.leaky_relu(self.dconv_ud2(x2))
                a3 = F.leaky_relu(self.dconv_ud3(x3))
                sum = torch.concat([a1, a2, a3], 1)
                x1 = F.leaky_relu(self.dconv_ud4(torch.concat([sum, x1], 1)))  # .permute(0,2,3,1) /0,3,1,2
                x2 = F.leaky_relu(self.dconv_ud5(torch.concat([sum, x2], 1)))
                x3 = F.leaky_relu(self.dconv_ud6(torch.concat([sum, x3], 1)))
            block_out = F.leaky_relu(self.dconv_ud7(torch.concat([x1, x2, x3], 1)))
            h = h + block_out
        # end of dense medium

        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # large scale
        h = self.ram_head_large(maps[0],x)
        h = torch.cat([x, h], 1)
        h = self.head_large(h)

        # dense large
        for i in range(3):
            x1 = x2 = x3 = h
            for j in range(3):
                a1 = F.leaky_relu(self.dconv_ud1(x1))
                a2 = F.leaky_relu(self.dconv_ud2(x2))
                a3 = F.leaky_relu(self.dconv_ud3(x3))
                sum = torch.concat([a1, a2, a3], 1)
                x1 = F.leaky_relu(self.dconv_ud4(torch.concat([sum, x1], 1)))  # .permute(0,2,3,1) /0,3,1,2
                x2 = F.leaky_relu(self.dconv_ud5(torch.concat([sum, x2], 1)))
                x3 = F.leaky_relu(self.dconv_ud6(torch.concat([sum, x3], 1)))
            block_out = F.leaky_relu(self.dconv_ud7(torch.concat([x1, x2, x3], 1)))
            h = h + block_out
        # end of dense large

        h = self.body_large(h) + x
        x = self.tail_large(h)

        return x


class ResBlock(nn.Module):
    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)

        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)


    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)

        return x

class RAM(nn.Module):
    def __init__(self, nf=64, n_condition=64):
        super(RAM, self).__init__()
        self.mul_conv1 = nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)
        self.add_conv1 = nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features, conditions):
        cat_input = torch.cat((features, conditions), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.lrelu(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.lrelu(self.add_conv1(cat_input)))
        return features * mul + add


class DenseResNet3(nn.Module): # multiscale conv
    def __init__(self, nf=64,  ngf=64, n_blocks=16, activation = 'selu', init_type = 'w_init'):
        super(DenseResNet3, self).__init__()
        self.offset_conv1 = nn.Conv2d(2*nf, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nf, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(nf, ngf, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv7 = nn.Conv2d(nf, ngf, kernel_size=7, stride=1, padding=3, bias=True)
        self.offset_conv2 = nn.Conv2d(3*nf, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        # self.dcn_pack = DCN(nf, 3, dw_kernel_size=None, stride=1, pad=1, dilation=1, group=8, offset_scale=1.0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ref_fea, offset_fea):
        offset1 = self.lrelu(self.offset_conv1(torch.cat([ref_fea, offset_fea], dim=1)))
        #
        x3 = self.lrelu(self.conv3(offset1))
        x5 = self.lrelu(self.conv5(offset1))
        x7 = self.lrelu(self.conv7(offset1))
        offset2 = torch.concat([x3, x5, x7], dim=1)
        #
        offset2 = self.lrelu(self.offset_conv2(offset2)) + offset1

        return offset2

def deconv_activation(in_ch, out_ch ,kernel_size, stride, padding, activation = 'relu' ):#orginal: relu

    # if activation == 'relu':
    #     return nn.Sequential(
    #             nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
    #             nn.ReLU(inplace = True))
    if activation == 'relu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    # elif activation == 'selu':
    #     return nn.Sequential(
    #             nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
    #             nn.SELU(inplace = True))
    elif activation == 'selu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True))

#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size
#
#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows
#
#
# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image
#
#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x
#
#
# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.
#
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """
#
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#
#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         print("B_, N, C:", B_, N, C)
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)
#
#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
#
#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops
#
#
# class SwinTransformerBlock(nn.Module):
#     r""" Swin Transformer Block.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         self.use_checkpoint = use_checkpoint
#
#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
#
#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#         if self.shift_size > 0:
#             attn_mask = self.calculate_mask(self.input_resolution)
#         else:
#             attn_mask = None
#
#         self.register_buffer("attn_mask", attn_mask)
#
#     def calculate_mask(self, x_size):
#         # calculate attention mask for SW-MSA
#         # H, W = x_size
#         H, W = int(x_size[0]), int(x_size[1])
#         img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
#         h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1
#
#         mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#
#         return attn_mask
#
#     def forward(self, x, x_size):
#         x_size = int(x_size[0]), int(x_size[1])
#         H, W = x_size
#         B, L, C = x.shape
#         # assert L == H * W, "input feature has wrong size"
#
#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)
#
#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x
#
#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
#
#         # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
#         if self.input_resolution == x_size:
#             # if self.use_checkpoint:
#             #     attn_windows = checkpoint.checkpoint(self.attn, x_windows, self.attn_mask)
#             # else:
#             attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
#         else:
#             # if self.use_checkpoint:
#             #     attn_windows = checkpoint.checkpoint(self.attn, x_windows, self.calculate_mask(x_size).to(x.device))
#             # else:
#             attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
#
#
#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
#
#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x
#         x = x.view(B, H * W, C)
#
#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
#
#     def flops(self):
#         flops = 0
#         H, W = self.input_resolution
#         # norm1
#         flops += self.dim * H * W
#         # W-MSA/SW-MSA
#         nW = H * W / self.window_size / self.window_size
#         flops += nW * self.attn.flops(self.window_size * self.window_size)
#         # mlp
#         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#         # norm2
#         flops += self.dim * H * W
#         return flops
#
#
# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#
#     Args:
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
#
#         x = x.view(B, H, W, C)
#
#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
#
#         x = self.norm(x)
#         x = self.reduction(x)
#
#         return x
#
#     def extra_repr(self) -> str:
#         return f"input_resolution={self.input_resolution}, dim={self.dim}"
#
#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.dim
#         flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
#         return flops
#
#
# class BasicLayer(nn.Module):
#     """ A basic Swin Transformer layer for one stage.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """
#
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#
#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
#                                  num_heads=num_heads, window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                  drop=drop, attn_drop=attn_drop,
#                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                  norm_layer=norm_layer, use_checkpoint=use_checkpoint)
#             for i in range(depth)])
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#
#     def forward(self, x, x_size):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x, x_size)
#             else:
#                 x = blk(x, x_size)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
#
#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops
#
#
# class RSTB(nn.Module):
#     """Residual Swin Transformer Block (RSTB).
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#         img_size: Input image size.
#         patch_size: Patch size.
#         resi_connection: The convolutional block before residual connection.
#     """
#
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
#                  img_size=224, patch_size=4, resi_connection='1conv'):
#         super(RSTB, self).__init__()
#
#         self.dim = dim
#         self.input_resolution = input_resolution
#
#         self.residual_group = BasicLayer(dim=dim,
#                            input_resolution=input_resolution,
#                            depth=depth,
#                            num_heads=num_heads,
#                            window_size=window_size,
#                            mlp_ratio=mlp_ratio,
#                            qkv_bias=qkv_bias, qk_scale=qk_scale,
#                            drop=drop, attn_drop=attn_drop,
#                            drop_path=drop_path,
#                            norm_layer=norm_layer,
#                            downsample=downsample,
#                            use_checkpoint=use_checkpoint)
#
#         if resi_connection == '1conv1x1':
#             self.conv = nn.Linear(dim, dim)
#         else:
#             if resi_connection == '1conv':
#                 self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
#             elif resi_connection == '3conv':
#                 # # save parameters AE313
#                 self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                           nn.Conv2d(dim//4, dim//4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                           nn.Conv2d(dim//4, dim, 3, 1, 1))
#                 # # # save parameters AE333
#                 # self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 #                           nn.Conv2d(dim//4, dim//4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 #                           nn.Conv2d(dim//4, dim, 3, 1, 1))
#
#             # embedding and unembedding after and before conv
#             self.patch_embed = PatchEmbed(
#                 img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
#                 norm_layer=None)
#
#             self.patch_unembed = PatchUnEmbed(
#                 img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
#                 norm_layer=None)
#
#     def forward(self, x, x_size):
#         if hasattr(self, 'patch_embed'):
#             return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
#         else:
#             return self.conv(self.residual_group(x, x_size)) + x
#
#     def flops(self):
#         flops = 0
#         flops += self.residual_group.flops()
#         H, W = self.input_resolution
#         flops += H * W * self.dim * self.dim * 9
#         flops += self.patch_embed.flops()
#         flops += self.patch_unembed.flops()
#
#         return flops
#
#
# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding
#
#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """
#
#     def __init__(self, img_size=480, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#     # def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]
#
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None
#
#     def forward(self, x):
#         x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
#         if self.norm is not None:
#             x = self.norm(x)
#         return x
#
#     def flops(self):
#         flops = 0
#         H, W = self.img_size
#         if self.norm is not None:
#             flops += H * W * self.embed_dim
#         return flops
#
#
# class PatchUnEmbed(nn.Module):
#     r""" Image to Patch Unembedding
#
#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """
#
#     def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]
#
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#
#
#     def forward(self, x, x_size):
#         B, HW, C = x.shape
#         x = x.transpose(1, 2).view(B, self.embed_dim, int(x_size[0]), int(x_size[1]))  # B Ph*Pw C
#         return x
#
#     def flops(self):
#         flops = 0
#         return flops
#
#
# class Upsample(nn.Sequential):
#     """Upsample module.
#
#     Args:
#         scale (int): Scale factor. Supported scales: 2^n and 3.
#         num_feat (int): Channel number of intermediate features.
#     """
#
#     def __init__(self, scale, num_feat, input_resolution=None):
#         self.num_feat = num_feat
#         self.input_resolution = input_resolution
#         m = []
#         if (scale & (scale - 1)) == 0:  # scale = 2^n
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
#                 m.append(nn.PixelShuffle(2))
#         elif scale == 3:
#             m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
#             m.append(nn.PixelShuffle(3))
#         else:
#             raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
#         super(Upsample, self).__init__(*m)
#
#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.num_feat * self.num_feat * 9
#         flops = 2*H * 2*W * self.num_feat * 3 * 9
#         return flops
#
# class UpsampleOneStep(nn.Sequential):
#     """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
#
#     Args:
#         scale (int): Scale factor. Supported scales: 2^n and 3.
#         num_feat (int): Channel number of intermediate features.
#
#     """
#
#     def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
#         self.num_feat = num_feat
#         self.input_resolution = input_resolution
#         m = []
#         m.append(nn.Conv2d(num_feat, (scale **2)  * num_out_ch, 3, 1, 1))
#         m.append(nn.PixelShuffle(scale))
#         super(UpsampleOneStep, self).__init__(*m)
#
#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.num_feat * 3 * 9
#         return flops
#
#
#
# class SwinBlock(nn.Module):
#     def __init__(self,
#                  img_size=160,#40
#                  patch_size=1,
#                  embed_dim=180,
#                  depths=(6, 6, 6, 6),
#                  num_heads=(6, 6, 6, 6),
#                  window_size=8,
#                  mlp_ratio=2.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm,
#                  ape=False,
#                  patch_norm=True,
#                  use_checkpoint=False,
#                  resi_connection='1conv',
#                  **kwargs):
#         super(SwinBlock, self).__init__()
#
#         self.use_checkpoint = use_checkpoint
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = embed_dim
#         self.mlp_ratio = mlp_ratio
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,norm_layer=norm_layer if self.patch_norm else None)
#
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         # merge non-overlapping patches into image
#         self.patch_unembed = PatchUnEmbed(img_size=img_size,        patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,norm_layer=norm_layer if self.patch_norm else None)
#
#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#
#         # build Residual Swin Transformer blocks (RSTB)
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = RSTB(
#                 dim=embed_dim,
#                 input_resolution=(patches_resolution[0], patches_resolution[1]),
#                 depth=depths[i_layer],
#                 num_heads=num_heads[i_layer],
#                 window_size=window_size,
#                 mlp_ratio=self.mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
#                 norm_layer=norm_layer,
#                 downsample=None,
#                 use_checkpoint=use_checkpoint,
#                 img_size=img_size,
#                 patch_size=patch_size,
#                 resi_connection=resi_connection)
#             self.layers.append(layer)
#         self.norm = norm_layer(self.num_features)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     def forward(self, x):
#
#         # x: [1, 60, 264, 184]
#         # print('x:', x.shape) # 2,80,64,64
#         x_size = torch.tensor(x.shape[2:4])   # [264, 184]
#         # print('x_size: ', x_size) # 64 64
#         x = self.patch_embed(x)               # [1, 48576, 60]
#         # print("x: ", x.shape)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)                  # [1, 48576, 60]
#         for layer in self.layers:
#             x = layer(x, x_size)              # [1, 48576, 60]
#         x = self.norm(x)  # b seq_len c       # [1, 48576, 60]
#         x = self.patch_unembed(x, x_size)     # [1, 60, 264, 184]
#
#         return x
#
# class SwinUnetv3RestorationNet(nn.Module):
#     def __init__(self, ngf=64, n_blocks=16, groups=8, embed_dim=64, depths=(8,8), num_heads=(8,8), window_size=8, use_checkpoint=False):
#         super(SwinUnetv3RestorationNet, self).__init__()
#         self.content_extractor = ContentExtractor(
#             in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
#         self.dyn_agg_restore = DynamicAggregationRestoration(ngf=ngf, n_blocks=n_blocks, groups=groups,
#                                             embed_dim=ngf, depths=depths, num_heads=num_heads,
#                                             window_size=window_size, use_checkpoint=use_checkpoint)
#
#         arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
#         self.re_init_dcn_offset()
#
#     def re_init_dcn_offset(self):
#         self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
#         self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
#         self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.weight.data.zero_()
#         self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.bias.data.zero_()
#
#         self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.weight.data.zero_()
#         self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.bias.data.zero_()
#         self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
#         self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
#         self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.weight.data.zero_()
#         self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.bias.data.zero_()
#
#     def forward(self, x, pre_offset_flow_sim, img_ref_feat):
#         """
#         Args:
#             x (Tensor): the input image of SRNTT.
#             maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
#                 and relu1_1. depths of the maps are 256, 128 and 64
#                 respectively.
#         """
#
#         base = F.interpolate(x, None, 4, 'bilinear', False)
#         content_feat = self.content_extractor(x)
#
#         upscale_restore = self.dyn_agg_restore(base, content_feat, pre_offset_flow_sim, img_ref_feat)
#
#         return upscale_restore + base
#
#
# class DynamicAggregationRestoration(nn.Module):
#
#     def __init__(self,
#                  ngf=64,
#                  n_blocks=16,
#                  groups=8,
#                  img_size=40,
#                  patch_size=1,
#                  in_chans=3,
#                  embed_dim=64,
#                  depths=(6, 6, 6, 6),
#                  num_heads=(6, 6, 6, 6),
#                  window_size=8,
#                  mlp_ratio=2.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm,
#                  ape=False,
#                  patch_norm=True,
#                  use_checkpoint=False
#                  ):
#         super(DynamicAggregationRestoration, self).__init__()
#         self.use_checkpoint = use_checkpoint
#         self.num_layers = len(depths)
#         self.embed_dim = ngf
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.mlp_ratio = mlp_ratio
#
#         self.unet_head = nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1)
#
#         # ---------------------- Down ----------------------
#
#         # dynamic aggregation module for relu1_1 reference feature
#         self.down_large_offset_conv1 = nn.Conv2d(ngf + 64*2, 64, 3, 1, 1, bias=True)
#         self.down_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
#         self.down_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1,
#                                     deformable_groups=groups, extra_offset_mask=True)
#
#         # for large scale
#         self.down_head_large = nn.Sequential(
#             nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.1, True))
#         self.down_body_large = SwinBlock(img_size=480, embed_dim=ngf, depths=depths, num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint) # 160
#         self.down_tail_large = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)
#
#
#         # dynamic aggregation module for relu2_1 reference feature
#         self.down_medium_offset_conv1 = nn.Conv2d(
#             ngf + 128*2, 128, 3, 1, 1, bias=True)
#         self.down_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
#         self.down_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1,dilation=1,
#                                     deformable_groups=groups, extra_offset_mask=True)
#
#         # for medium scale restoration
#         self.down_head_medium = nn.Sequential(
#             nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.1, True))
#         self.down_body_medium = SwinBlock(img_size=80, embed_dim=ngf,
#                                 depths=depths, num_heads=num_heads, window_size=window_size)
#         self.down_tail_medium = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)
#
#
#         # ---------------------- Up ----------------------
#         # dynamic aggregation module for relu3_1 reference feature
#         self.up_small_offset_conv1 = nn.Conv2d(
#             ngf + 256*2, 256, 3, 1, 1, bias=True)  # concat for diff
#         self.up_small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
#         self.up_small_dyn_agg = DynAgg(256, 256, 3, stride=1, padding=1, dilation=1,
#                                 deformable_groups=groups, extra_offset_mask=True)
#
#         # for small scale restoration
#         self.up_head_small = nn.Sequential(
#             nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.1, True))
#         self.up_body_small = SwinBlock(img_size=40, embed_dim=ngf,
#                                 depths=depths, num_heads=num_heads, window_size=window_size)
#         self.up_tail_small = nn.Sequential(
#             nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
#             nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))
#
#
#         # dynamic aggregation module for relu2_1 reference feature
#         self.up_medium_offset_conv1 = nn.Conv2d(
#             ngf + 128*2, 128, 3, 1, 1, bias=True)
#         self.up_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
#         self.up_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1, dilation=1,
#                                 deformable_groups=groups, extra_offset_mask=True)
#
#         # for medium scale restoration
#         self.up_head_medium = nn.Sequential(
#             nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.1, True))
#         self.up_body_medium = SwinBlock(img_size=80, embed_dim=ngf,
#                                 depths=depths, num_heads=num_heads, window_size=window_size)
#         self.up_tail_medium = nn.Sequential(
#             nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
#             nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))
#
#
#         # dynamic aggregation module for relu1_1 reference feature
#         self.up_large_offset_conv1 = nn.Conv2d(ngf + 64*2, 64, 3, 1, 1, bias=True)
#         self.up_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
#         self.up_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1,
#                                 deformable_groups=groups, extra_offset_mask=True)
#
#         # for large scale
#         self.up_head_large = nn.Sequential(
#             nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.1, True))
#         self.up_body_large = SwinBlock(img_size=480, embed_dim=ngf, depths=depths, num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint) # 160
#         self.up_tail_large = nn.Sequential(
#             nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#     def flow_warp(self,
#                   x,
#                   flow,
#                   interp_mode='bilinear',
#                   padding_mode='zeros',
#                   align_corners=True):
#         """Warp an image or feature map with optical flow.
#         Args:
#             x (Tensor): Tensor with size (n, c, h, w).
#             flow (Tensor): Tensor with size (n, h, w, 2), normal value.
#             interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
#             padding_mode (str): 'zeros' or 'border' or 'reflection'.
#                 Default: 'zeros'.
#             align_corners (bool): Before pytorch 1.3, the default value is
#                 align_corners=True. After pytorch 1.3, the default value is
#                 align_corners=False. Here, we use the True as default.
#         Returns:
#             Tensor: Warped image or feature map.
#         """
#
#         assert x.size()[-2:] == flow.size()[1:3]
#         _, _, h, w = x.size()
#         # create mesh grid
#         grid_y, grid_x = torch.meshgrid(
#             torch.arange(0, h).type_as(x),
#             torch.arange(0, w).type_as(x))
#         grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
#         grid.requires_grad = False
#
#         vgrid = grid + flow
#         # scale grid to [-1,1]
#         vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
#         vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
#         vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
#         output = F.grid_sample(x,
#                                vgrid_scaled,
#                                mode=interp_mode,
#                                padding_mode=padding_mode,
#                                align_corners=align_corners)
#
#         return output
#
#     def forward(self, base, x, pre_offset_flow_sim, img_ref_feat):
#
#         pre_offset = pre_offset_flow_sim[0]
#         pre_flow = pre_offset_flow_sim[1]
#         pre_similarity = pre_offset_flow_sim[2]
#
#         pre_relu1_swapped_feat = self.flow_warp(img_ref_feat['relu1_1'], pre_flow['relu1_1'])
#         pre_relu2_swapped_feat = self.flow_warp(img_ref_feat['relu2_1'], pre_flow['relu2_1'])
#         pre_relu3_swapped_feat = self.flow_warp(img_ref_feat['relu3_1'], pre_flow['relu3_1'])
#
#         # Unet
#         x0 = self.unet_head(base)    # [B, 64, 160, 160]
#
#         # -------------- Down ------------------
#         # large scale
#         down_relu1_offset = torch.cat([x0, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
#         down_relu1_offset = self.lrelu(self.down_large_offset_conv1(down_relu1_offset))
#         down_relu1_offset = self.lrelu(self.down_large_offset_conv2(down_relu1_offset))
#         down_relu1_swapped_feat = self.lrelu(
#             self.down_large_dyn_agg([img_ref_feat['relu1_1'], down_relu1_offset],
#                                pre_offset['relu1_1'], pre_similarity['relu1_1']))
#
#         h = torch.cat([x0, down_relu1_swapped_feat], 1)
#         h = self.down_head_large(h)
#         h = self.down_body_large(h) + x0
#         x1 = self.down_tail_large(h)  # [B, 64, 80, 80]
#
#         # medium scale
#         down_relu2_offset = torch.cat([x1, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
#         down_relu2_offset = self.lrelu(self.down_medium_offset_conv1(down_relu2_offset))
#         down_relu2_offset = self.lrelu(self.down_medium_offset_conv2(down_relu2_offset))
#         down_relu2_swapped_feat = self.lrelu(
#             self.down_medium_dyn_agg([img_ref_feat['relu2_1'], down_relu2_offset],
#                                 pre_offset['relu2_1'], pre_similarity['relu2_1']))
#
#         h = torch.cat([x1, down_relu2_swapped_feat], 1)
#         h = self.down_head_medium(h)
#         h = self.down_body_medium(h) + x1
#         x2 = self.down_tail_medium(h)    # [9, 128, 40, 40]
#
#         # -------------- Up ------------------
#
#         # dynamic aggregation for relu3_1 reference feature
#         relu3_offset = torch.cat([x2, pre_relu3_swapped_feat, img_ref_feat['relu3_1']], 1)
#         relu3_offset = self.lrelu(self.up_small_offset_conv1(relu3_offset))
#         relu3_offset = self.lrelu(self.up_small_offset_conv2(relu3_offset))
#         relu3_swapped_feat = self.lrelu(
#             self.up_small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset], pre_offset['relu3_1'], pre_similarity['relu3_1']))
#
#         # small scale
#         h = torch.cat([x2, relu3_swapped_feat], 1)
#         h = self.up_head_small(h)
#         h = self.up_body_small(h) + x2
#         x = self.up_tail_small(h)    # [9, 64, 80, 80]
#
#         # dynamic aggregation for relu2_1 reference feature
#         relu2_offset = torch.cat([x, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
#         relu2_offset = self.lrelu(self.up_medium_offset_conv1(relu2_offset))
#         relu2_offset = self.lrelu(self.up_medium_offset_conv2(relu2_offset))
#         relu2_swapped_feat = self.lrelu(
#             self.up_medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],
#                                 pre_offset['relu2_1'], pre_similarity['relu2_1']))
#         # medium scale
#         h = torch.cat([x+x1, relu2_swapped_feat], 1)
#         h = self.up_head_medium(h)
#         h = self.up_body_medium(h) + x
#         x = self.up_tail_medium(h)   # [9, 64, 160, 160]
#
#         # dynamic aggregation for relu1_1 reference feature
#         relu1_offset = torch.cat([x, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
#         relu1_offset = self.lrelu(self.up_large_offset_conv1(relu1_offset))
#         relu1_offset = self.lrelu(self.up_large_offset_conv2(relu1_offset))
#         relu1_swapped_feat = self.lrelu(
#             self.up_large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],
#                                pre_offset['relu1_1'], pre_similarity['relu1_1']))
#         # large scale
#         h = torch.cat([x+x0, relu1_swapped_feat], 1)
#         h = self.up_head_large(h)
#         h = self.up_body_large(h) + x
#         x = self.up_tail_large(h)
#
#         return x

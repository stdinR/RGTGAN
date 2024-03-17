# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from torchvision import models
from models.archs.utils2 import MeanShift

import matplotlib.pyplot as plt


def init_weights(w, init_type):

    if init_type == 'w_init_relu':
        nn.init.kaiming_uniform_(w, nonlinearity = 'relu')
    elif init_type == 'w_init_leaky':
        nn.init.kaiming_uniform_(w, nonlinearity = 'leaky_relu')
    elif init_type == 'w_init':
        nn.init.uniform_(w)

def activation(activation):

    if activation == 'relu':
        return nn.ReLU(inplace = True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope = 0.1 ,inplace = True )
    elif activation == 'selu':
        return nn.SELU(inplace = True)
    elif activation == 'linear':
        return nn.Linear()


# ---------------------------------fuction------------------------------------
def conv_activation(in_ch, out_ch , kernel_size = 3, stride = 1, padding = 2, activation = 'relu', init_type = 'w_init_relu'):


    if activation == 'relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding))



def upsample(in_ch, out_ch):

    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True)



def leaky_deconv(in_ch, out_ch):

    return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.1,inplace=True)
                        )

def deconv_activation(in_ch, out_ch ,kernel_size, stride, padding, activation = 'relu' ): # relu

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


class Encoder(nn.Module):

    def __init__(self,in_ch, nf, activation = 'selu', init_type = 'w_init'):
        super(Encoder, self).__init__()

        self.layer_f = conv_activation(in_ch, nf, kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type)

        self.conv1 = conv_activation(nf, nf, kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type)

        self.conv2 = conv_activation(nf, nf, kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)

        self.conv3 = conv_activation(nf, nf, kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)

    def forward(self,x):

        layer_f = self.layer_f(x)
        conv1 = self.conv1(layer_f) # 16, 256, 256
        # conv1_GTEM = GTEM_fea(conv1) #
        conv2 = self.conv2(conv1) # 16, 128, 128
        # conv2_GTEM = GTEM_fea(conv2) #
        conv3 = self.conv3(conv2) # 16, 64, 64
        # conv3_GTEM = GTEM_fea(conv3) #
        return conv1,conv2,conv3

class Encoder_grad(nn.Module): # fail

    def __init__(self,in_ch, nf, activation = 'selu', init_type = 'w_init'):
        super(Encoder_grad, self).__init__()

        self.layer_f = conv_activation(in_ch, nf, kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type)

        self.conv1 = conv_activation(nf, nf, kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type) # 5 1 2
        self.GTEM_fea1 = GTEM_fea(in_ch=nf, nf=nf) # 256

        self.conv2 = conv_activation(nf, nf, kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type) # 5 2 2
        self.GTEM_fea2 = GTEM_fea(in_ch=nf, nf=nf) # 128

        self.conv3 = conv_activation(nf, nf, kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type) # 5 2 2
        self.GTEM_fea3 = GTEM_fea(in_ch=nf, nf=nf) # 64

    def forward(self,x):
        layer_f = self.layer_f(x)
        conv1 = self.conv1(layer_f) # 16, 256, 256
        # print('conv1: ', conv1.shape)
        conv1_GTEM = self.GTEM_fea1(conv1) #
        # print('conv1_GTEM:', conv1_GTEM.shape)
        conv2 = self.conv2(conv1) # 16, 128, 128
        conv2_GTEM = self.GTEM_fea2(conv2) #
        conv3 = self.conv3(conv2) # 16, 64, 64
        conv3_GTEM = self.GTEM_fea3(conv3) #
        return conv1_GTEM ,conv2_GTEM ,conv3_GTEM

class Encoder_LTE(nn.Module):
    def __init__(self, in_ch, nf, requires_grad=True, rgb_range=1, activation = 'selu', init_type = 'w_init'):
        super(Encoder_LTE,self).__init__()
        # use vgg19
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # added
        if nf == 64:
            # print('original image: ')
            self.gradient = False
            self.conv1 = conv_activation(in_ch, nf, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)  # 5 1 2
            self.conv2 = conv_activation(in_ch * 2, nf, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)  # 5 2 2
            self.conv2_up = conv_activation(nf, nf * 2, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)
            self.conv3_1 = conv_activation(in_ch * 4, in_ch * 2, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)  # 5 2 2
            self.conv3_2 = conv_activation(in_ch * 2, nf, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)
        elif nf == 16:
            # print('gradient image: ')
            self.gradient = True
            self.conv1 = conv_activation(in_ch, nf, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)  # 5 1 2
            self.conv1_up = conv_activation(nf, in_ch, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)
            self.conv2 = conv_activation(in_ch * 2, nf, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)  # 5 2 2
            self.conv2_up1 = conv_activation(nf, nf * 4, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)
            self.conv2_up2 = conv_activation(nf * 4, nf * 8, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)
            self.conv3_1 = conv_activation(nf * 16, nf * 4, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)  # 5 2 2
            self.conv3_2 = conv_activation(nf * 4, nf, kernel_size=5, stride=1, padding=2, activation=activation, init_type=init_type)
        # end of add
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        if self.gradient == False:
            x = self.sub_mean(x)
            x = self.slice1(x)
            # x_lv1_tmp = x
            # print('x_lv1_tmp: ', x_lv1_tmp.shape)  # 64,256,256
            x = self.conv1(x)
            x_lv1 = x
            # print('x_lv1: ', x_lv1.shape)  # 64,256,256
            x = self.slice2(x)
            # x_lv2_tmp = x
            # print('x_lv2_tmp: ', x_lv2_tmp.shape)  # 128,128,128
            x = self.conv2(x)
            x_lv2 = x
            # print('x_lv2： ', x_lv2.shape)  # 64,128,128
            x = self.conv2_up(x)  #
            x = self.slice3(x)
            # x_lv3_tmp = x
            # print('x_lv3_tmp: ', x_lv3_tmp.shape)  # 256, 64, 64
            x = self.conv3_1(x)
            x = self.conv3_2(x)
            x_lv3 = x
            # print('x_lv3: ', x_lv3.shape)
        else:
            x = self.sub_mean(x)
            x = self.slice1(x)
            # print('x_lv1_tmp: ', x.shape)
            x = self.conv1(x)
            x_lv1 = x
            # print('x_lv1: ', x.shape)
            x = self.conv1_up(x)
            # print('conv1_up: ', x.shape)

            x = self.slice2(x)
            # print('x_lv2_tmp: ', x.shape)
            x = self.conv2(x)
            # print('x_lv2: ', x.shape)
            x_lv2 = x
            x = self.conv2_up1(x)
            x = self.conv2_up2(x)
            # print('conv2_up: ', x.shape)

            x = self.slice3(x)
            # print('x_lv3_tmp: ', x.shape)
            x = self.conv3_1(x)
            x = self.conv3_2(x)
            x_lv3 = x

        return x_lv1, x_lv2, x_lv3


class GTEM_F (nn.Module):
    def __init__(self, in_ch, nf, activation = 'selu', init_type = 'w_init'):
        super(GTEM_F, self).__init__()
        # # Laplacian kernel [WRONG!]
        # kernel_v = torch.FloatTensor([
        #     [[0, -1, 0], [0, 0, 0], [0, 1, 0]],
        #     [[0, -1, 0], [0, 0, 0], [0, 1, 0]],
        #     [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        # ]).unsqueeze(0).unsqueeze(0)
        # kernel_h = torch.FloatTensor([
        #     [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        #     [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        #     [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        # ]).unsqueeze(0).unsqueeze(0)
        # self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        # self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        # end of Laplacian
        # self.conv_e1 = self.conv_layer(3, 64, 3)
        # self.con_e1 = F.conv2d(in_ch=3, out_ch=64, kernel_size=3, stride=1, padding=1)
        self.conv_e1 = conv_activation(in_ch, nf, kernel_size = 3 ,stride = 1,padding = 1, activation = activation, init_type = init_type).cuda()
        # self.conv_e2 = self.conv_layer(64, 64, 3, 1)
        self.conv_e2 = conv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation = activation, init_type = init_type).cuda()
        # self.conv_e3 = self.conv_layer(64, 128, 3, 2)
        self.conv_e3 = conv_activation(64, 128, kernel_size=3, stride=2, padding=1, activation = activation, init_type = init_type).cuda()
        # self.conv_e4 = self.conv_layer(128, 128, 3, 1)
        self.conv_e4 = conv_activation(128, 128, kernel_size=3, stride=1, padding=1, activation=activation,init_type=init_type).cuda()
        # self.conv_e5 = self.conv_layer(128, 256, 3, 2)
        self.conv_e5 = conv_activation(128, 256, kernel_size=3, stride=2, padding=1, activation=activation,init_type=init_type).cuda()
        # self.conv_e7 = self.conv_layer(256, 64, 3, 1)
        self.conv_e7 = conv_activation(256, 64, kernel_size=3, stride=1, padding=1, activation=activation,init_type=init_type).cuda()
        # self.conv_e8 = self.conv_layer(64, 256, 3, 1)
        #
        self.dconv_ud1 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_ud2 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_ud3 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_ud4 = deconv_activation(256, 64, kernel_size=1, stride=1, padding=0, activation=activation)
        self.dconv_ud5 = deconv_activation(256, 64, kernel_size=1, stride=1, padding=0, activation=activation)
        self.dconv_ud6 = deconv_activation(256, 64, kernel_size=1, stride=1, padding=0, activation=activation)
        self.dconv_ud7 = deconv_activation(192, 64, kernel_size=3, stride=1, padding=1, activation=activation)

        # res after dense
        n_blocks = 16
        ngf = 64
        self.res_dense =  nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        # end of res

        self.conv_e8 = conv_activation(64, 256, kernel_size=3, stride=1, padding=1, activation=activation,init_type=init_type).cuda()

        self.dconv_m3 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_m4 = deconv_activation(64, 128, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_m5 = deconv_activation(128, 256, kernel_size=3, stride=1, padding=1, activation=activation)

        self.dconv_m6 = deconv_activation(256, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_d3 = deconv_activation(64, 256, kernel_size=3, stride=1, padding=1, activation=activation)

        self.dconv_d4 = deconv_activation(64, 256, kernel_size=3, stride=1, padding=1, activation=activation)

        self.dconv_fusion2 = deconv_activation(64, 3, kernel_size=3, stride=1, padding=1, activation=activation)#64,3


    def forward(self, x):
        # x_fa = self.Laplacian(x)
        x_fa = x
        # print('x_fa:')
        # print(x_fa.shape)
        x_f = self.conv_e1(x_fa)
        x_f = F.leaky_relu(x_f)
        x_f = self.conv_e2(x_f)
        x_f = self.conv_e3(x_f)
        x_f = F.leaky_relu(x_f)
        x_f = self.conv_e4(x_f)
        x_f = F.leaky_relu(x_f)
        x_f = self.conv_e5(x_f)
        x_f = F.leaky_relu(x_f)
        x_f = self.conv_e7(x_f)
        x_f = F.leaky_relu(x_f)
        res_in = x_f
        # print('res_in ', res_in.shape)

        for i in range(3):
            x1=x2=x3=x_f
            # print ('x_f. '+str(x_f.shape))
            for j in range(3):
                a1 = F.leaky_relu(self.dconv_ud1(x1))
                # print('a1. '+str(a1.shape))
                a2 = F.leaky_relu(self.dconv_ud2(x2))
                a3 = F.leaky_relu(self.dconv_ud3(x3))
                sum = torch.concat([a1,a2,a3],1)
                # print('sum. '+str(sum.shape))
                # print('concat. '+ str(torch.concat([sum,x1],3).shape))
                x1 = F.leaky_relu(self.dconv_ud4(torch.concat([sum,x1],1)))#.permute(0,2,3,1) /0,3,1,2
                x2 = F.leaky_relu(self.dconv_ud5(torch.concat([sum,x2],1)))
                x3 = F.leaky_relu(self.dconv_ud6(torch.concat([sum,x3],1)))
            block_out = F.leaky_relu(self.dconv_ud7(torch.concat([x1,x2,x3],1)))
            x_f = x_f + block_out

        # res
        x_f = self.res_dense(x_f) + x_f
        # end of res

        x_f = self.conv_e8(x_f)
        x_f = F.leaky_relu(x_f)
        # print('x_f')
        # print(x_f.shape)

        # mask
        x_mask = F.leaky_relu(self.dconv_m3(res_in))
        x_mask = F.leaky_relu(self.dconv_m4(x_mask))
        x_mask = F.leaky_relu(self.dconv_m5(x_mask))
        # print('x_mask')
        # print(x_mask.shape)
        frame_mask = F.sigmoid(x_mask)
        x_frame = frame_mask*x_f + x_f
        # print('x_frame: ',x_frame.shape)

        # 测试这部分，如下是不是转换为image space！如果不是，GTEM模块采用如上！
        x_frame = F.leaky_relu(self.dconv_m6(x_frame))
        # print('x_frame: after mask ' + str(x_frame.size()))# 4,64,64,64

        x_frame = self.dconv_d3(x_frame)
        # print('x_frame: pre_shuttle '+ str(x_frame.size())) # 4,256,64,64
        x_frame = self.pixel_shuffle_layerg(x_frame, 2, 64)
        # print('x_frame: after_shuttle ' + str(x_frame.size()))# 64,128,128,4
        x_frame = F.leaky_relu(x_frame)
        # after d3 , channel should be 64
        # print('x_frame.shape before d4: '+ str(x_frame.size()))# 64,128,128,4

        x_frame = self.dconv_d4(x_frame)
        x_frame = self.pixel_shuffle_layerg(x_frame, 2, 64)
        x_frame = F.leaky_relu(x_frame)

        #x_de = x_d + x_frame
        x_frame = self.dconv_fusion2(x_frame)
        # print(x_frame.size())
        # print(x.size())
        # print(x_fa.size())

        # print('x_frame2')
        # print(x_frame.shape)

        x_sr = x_frame + x - x_fa
        # print('x_sr: ', x_sr.shape)

        # x_sr_np = x_sr.cpu().detach().numpy()
        # plt.imshow(x_sr_np[0])
        # plt.title('enhanced gradient')
        # plt.show()

        return x_sr

    # def Laplacian(self, x):
    #     self.Lapweight = nn.Parameter(torch.Tensor([
    #         [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
    #         [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
    #         [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]
    #         ])).cuda()
    #     frame = F.conv2d(x, self.Lapweight, padding=1)
    #     return frame

    def pixel_shuffle_layerg(self, x, r, n_split):
        x= x.permute(0,2,3,1) # channel:256 last 最后要转回去
        xc = torch.split(x, 256//n_split, 3) # split is different from tf! # torch.split(tensor, split_size_or_sections, dim=0)
        # for x__ in xc:
        #      print('x__ shape = '+str(x__.shape))
        def PS(x ,r): # x_ -1,128,128,4
            #bs, a, b, c =x.get_shape().as_list()
            # [bs, a, b, c] = x.shape # 4,256,64,64
            bs, a, b, c = x.size()
            # print('x. '+str(x.shape))
            # print('bs, a, b, c = '+str(bs)+', '+str(a)+', '+str(b)+', '+str(c))
            if bs==1:
                x = x.view(a,b,r,r).permute(0,1,3,2)
                x = x.split(a,0)
                x = torch.cat([x_.squeeze() for x_ in x], 1)
                x = x.split(b,1)
                x = torch.cat([x_.squeeze() for x_ in x], 1)
            else:
                x = x.reshape(-1, a, b, r, r).permute(0, 1, 2, 4, 3)
                x = torch.split(x,a,1)
                x = torch.cat([torch.squeeze(x_, axis=1) for x_ in x], axis=2)
                x = torch.split(x,b,1)
                x = torch.cat([torch.squeeze(x_, axis=1) for x_ in x], axis=2)
            # print('return x.reshape(-1, a * r, b * r, 1).shape =  '+str(x.reshape(-1, a * r, b * r, 1).shape))
            return x.reshape(-1, a * r, b * r, 1)
        return torch.cat([PS(x_,r) for x_ in xc], 3).permute(0,3,1,2)


class GTEM_fea (nn.Module):
    def __init__(self, in_ch, nf, activation = 'selu', init_type = 'w_init'):
        super(GTEM_fea, self).__init__()

        # self.conv_e1 = self.conv_layer(3, 64, 3)
        # self.con_e1 = F.conv2d(in_ch=3, out_ch=64, kernel_size=3, stride=1, padding=1)
        self.conv_e1 = conv_activation(in_ch, 16, kernel_size = 3 ,stride = 1,padding = 1, activation = activation, init_type = init_type).cuda()
        # self.conv_e2 = self.conv_layer(64, 64, 3, 1)
        self.conv_e2 = conv_activation(in_ch, 64, kernel_size=3, stride=1, padding=1, activation = activation, init_type = init_type).cuda() # 64，64，3，1，1
        # self.conv_e3 = self.conv_layer(64, 128, 3, 2)
        self.conv_e3 = conv_activation(64, 128, kernel_size=3, stride=1, padding=1, activation = activation, init_type = init_type).cuda() # 64 128 3 2 1
        # self.conv_e4 = self.conv_layer(128, 128, 3, 1)
        self.conv_e4 = conv_activation(128, 128, kernel_size=3, stride=1, padding=1, activation=activation,init_type=init_type).cuda() # 128 128 3 1 1
        # self.conv_e5 = self.conv_layer(128, 256, 3, 2)
        self.conv_e5 = conv_activation(128, 256, kernel_size=3, stride=1, padding=1, activation=activation,init_type=init_type).cuda() # 128 256 3 2 1
        # self.conv_e7 = self.conv_layer(256, 64, 3, 1)
        self.conv_e7 = conv_activation(256, 64, kernel_size=3, stride=1, padding=1, activation=activation,init_type=init_type).cuda() # 256 64 3 1 1
        # self.conv_e8 = self.conv_layer(64, 256, 3, 1)
        #
        self.dconv_ud1 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_ud2 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_ud3 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        self.dconv_ud4 = deconv_activation(256, 64, kernel_size=1, stride=1, padding=0, activation=activation)
        self.dconv_ud5 = deconv_activation(256, 64, kernel_size=1, stride=1, padding=0, activation=activation)
        self.dconv_ud6 = deconv_activation(256, 64, kernel_size=1, stride=1, padding=0, activation=activation)
        self.dconv_ud7 = deconv_activation(192, 64, kernel_size=3, stride=1, padding=1, activation=activation)

        # res after dense
        n_blocks = 16
        ngf = 64 # 64
        #
        self.res_dense =  nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        # end of res

        # self.conv_e8 = conv_activation(64, 256, kernel_size=3, stride=1, padding=1, activation=activation, init_type=init_type).cuda()

        # added
        self.conv_nf = conv_activation(64, nf, kernel_size=3, stride=1, padding=1, activation=activation, init_type=init_type).cuda() # 64 nf 3 1 1 # 256 nf 3 1 1
        #
        # self.dconv_m3 = deconv_activation(64, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        # self.dconv_m4 = deconv_activation(64, 128, kernel_size=3, stride=1, padding=1, activation=activation)
        # self.dconv_m5 = deconv_activation(128, 256, kernel_size=3, stride=1, padding=1, activation=activation)
        # #
        # self.dconv_m6 = deconv_activation(256, 64, kernel_size=3, stride=1, padding=1, activation=activation)
        # self.dconv_d3 = deconv_activation(64, 256, kernel_size=3, stride=1, padding=1, activation=activation)
        # #
        # self.dconv_d4 = deconv_activation(64, 256, kernel_size=3, stride=1, padding=1, activation=activation)
        # #
        # self.dconv_fusion2 = deconv_activation(64, 3, kernel_size=3, stride=1, padding=1, activation=activation)#64,3


    def forward(self, x):
        # x_fa = self.Laplacian(x)
        x_fa = x
        # print('x_fa:', x_fa.shape) # 2，16，256，256
        # x_f = self.conv_e1(x_fa)
        # x_f = F.leaky_relu(x_f)

        x_f = self.conv_e2(x_fa)
        # x_f = F.leaky_relu(x_f) # add
        # print('e2: ', x_f.shape)
        # x_f = self.conv_e3(x_f)
        # # print('e3: ', x_f.shape)
        # x_f = F.leaky_relu(x_f)
        # # print('e3_relu: ', x_f.shape)
        # x_f = self.conv_e4(x_f)
        # # print('e4: ', x_f.shape)
        # x_f = F.leaky_relu(x_f)
        # # print('e4_relu: ', x_f.shape)
        # x_f = self.conv_e5(x_f)
        # # print('e5: ', x_f.shape)
        # x_f = F.leaky_relu(x_f)
        # # print('e5_relu: ', x_f.shape)
        # x_f = self.conv_e7(x_f)
        x_f = F.leaky_relu(x_f)
        # res_in = x_f
        # print('res_in ', res_in.shape)

        for i in range(3):
            x1=x2=x3=x_f
            # print ('x_f. '+str(x_f.shape))
            for j in range(3):
                a1 = F.leaky_relu(self.dconv_ud1(x1))
                # print('a1. '+str(a1.shape))
                a2 = F.leaky_relu(self.dconv_ud2(x2))
                a3 = F.leaky_relu(self.dconv_ud3(x3))
                sum = torch.concat([a1,a2,a3],1)
                # print('sum. '+str(sum.shape))
                # print('concat. '+ str(torch.concat([sum,x1],3).shape))
                x1 = F.leaky_relu(self.dconv_ud4(torch.concat([sum,x1],1)))#.permute(0,2,3,1) /0,3,1,2
                x2 = F.leaky_relu(self.dconv_ud5(torch.concat([sum,x2],1)))
                x3 = F.leaky_relu(self.dconv_ud6(torch.concat([sum,x3],1)))
            block_out = F.leaky_relu(self.dconv_ud7(torch.concat([x1,x2,x3],1)))
            x_f = x_f + block_out

        # res
        x_f = self.res_dense(x_f) + x_f
        # end of res
        # print('end of res, x_f', x_f.shape)

        # # 1025
        # x_f = self.conv_e8(x_f)
        # x_f = F.leaky_relu(x_f)
        # # # print('x_f', x_f.shape)
        # #
        # # mask
        # x_mask = F.leaky_relu(self.dconv_m3(res_in))
        # x_mask = F.leaky_relu(self.dconv_m4(x_mask))
        # x_mask = F.leaky_relu(self.dconv_m5(x_mask))
        # # print('x_mask', x_mask.shape)
        # frame_mask = F.sigmoid(x_mask)
        # x_frame = frame_mask*x_f + x_f
        # # print('x_frame: ', x_frame.shape)
        #
        # x_frame = F.leaky_relu(self.dconv_m6(x_frame))
        # # print('x_frame_relu: ' + str(x_frame.size())) # 4,64,64,64
        #
        # x_sr = self.conv_nf(x_frame)
        x_sr = self.conv_nf(x_f)

        # # image space！

        # x_frame = self.dconv_d3(x_frame)
        # # print('x_frame: pre_shuttle '+ str(x_frame.size())) # 4,256,64,64
        # x_frame = self.pixel_shuffle_layerg(x_frame, 2, 64)
        # # print('x_frame: after_shuttle ' + str(x_frame.size()))# 64,128,128,4
        # x_frame = F.leaky_relu(x_frame)
        # # after d3 , channel should be 64
        # # print('x_frame.shape before d4: '+ str(x_frame.size()))# 64,128,128,4
        #
        # x_frame = self.dconv_d4(x_frame)
        # x_frame = self.pixel_shuffle_layerg(x_frame, 2, 64)
        # x_frame = F.leaky_relu(x_frame)
        #
        # #x_de = x_d + x_frame
        # x_frame = self.dconv_fusion2(x_frame)
        # # print(x_frame.size())
        # # print(x.size())
        # # print(x_fa.size())
        #
        # # print('x_frame2')
        # # print(x_frame.shape)
        #
        # x_sr = x_frame + x - x_fa
        # print('x_sr: ', x_sr.shape)

        return x_sr # x_sr

    def pixel_shuffle_layerg(self, x, r, n_split):
        x= x.permute(0,2,3,1) # channel:256 last 最后要转回去
        xc = torch.split(x, 256//n_split, 3) # split is different from tf! # torch.split(tensor, split_size_or_sections, dim=0)
        # for x__ in xc:
        #      print('x__ shape = '+str(x__.shape))
        def PS(x ,r): # x_ -1,128,128,4
            #bs, a, b, c =x.get_shape().as_list()
            # [bs, a, b, c] = x.shape # 4,256,64,64
            bs, a, b, c = x.size()
            # print('x. '+str(x.shape))
            # print('bs, a, b, c = '+str(bs)+', '+str(a)+', '+str(b)+', '+str(c))
            if bs==1:
                x = x.view(a,b,r,r).permute(0,1,3,2)
                x = x.split(a,0)
                x = torch.cat([x_.squeeze() for x_ in x], 1)
                x = x.split(b,1)
                x = torch.cat([x_.squeeze() for x_ in x], 1)
            else:
                x = x.reshape(-1, a, b, r, r).permute(0, 1, 2, 4, 3)
                x = torch.split(x,a,1)
                x = torch.cat([torch.squeeze(x_, axis=1) for x_ in x], axis=2)
                x = torch.split(x,b,1)
                x = torch.cat([torch.squeeze(x_, axis=1) for x_ in x], axis=2)
            # print('return x.reshape(-1, a * r, b * r, 1).shape =  '+str(x.reshape(-1, a * r, b * r, 1).shape))
            return x.reshape(-1, a * r, b * r, 1)
        return torch.cat([PS(x_,r) for x_ in xc], 3).permute(0,3,1,2)


class ResBlock(nn.Module):
    """
    Basic residual block for SRNTT.
    Parameters
    ---
    n_filters : int, optional
        a number of filters.
    """

    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x


# class Laplacian(nn.Module):
#     def __init__(self, x):
#         super(Laplacian, self).__init__()
#         self.Lapweight = nn.Parameter(torch.Tensor([
#             [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
#             [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
#             [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]
#         ]))
#     def forward(self, x):
#         frame = F.conv2d(x, self.Lapweight, padding=1)
#         return frame


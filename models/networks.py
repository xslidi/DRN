import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from util.util import NoneLayer
from torch.nn.utils import spectral_norm


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = NoneLayer
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,   
            gpu_ids=[], scale=True, spectral=True):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
            
    if netG == 'rdnccut':
        net = RDNC_CUT(input_nc, output_nc, ngf, norm_layer=norm_layer, scale=scale)
    elif netG == 'uegan':
        net = Generator()                                                     
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', 
            init_gain=0.02, gpu_ids=[], spectral=True):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, spectral=spectral)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, spectral=spectral)
    elif netD == 'fe':
        net = Feature_Exacter(input_nc, ndf, norm_layer=norm_layer, spectral=spectral, gpu_ids=gpu_ids)          
    elif netD == 'feue':
        net = Feature_Exacter_ue(input_nc, ndf, norm_layer=norm_layer, spectral=spectral, gpu_ids=gpu_ids)                                                    
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], nc=None):
    if netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=nc)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if self.loss == None:
            G_loss = - input.mean()
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            G_loss = self.loss(input, target_tensor)
        return G_loss


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, spectral=True):

        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if spectral:
            sequence = [
                spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)

                sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)    
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        else:
            sequence = [
                    nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class Bn_Relu_Conv(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size,
                stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super(Bn_Relu_Conv, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, padding=padding, stride=stride)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.norm = norm_layer(output_nc)


    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids



class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, norm_layer=nn.BatchNorm2d, dilation=1):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=1, bias=True, dilation=dilation)
        self.norm = norm_layer(growthRate)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.norm(out)
        out = torch.cat((x, out), 1)
        return out

# Residual dense block (RDB)
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm_layer=nn.BatchNorm2d, dilation=1):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate, norm_layer=norm_layer, dilation=dilation))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    assert (len(feat.size()) == 4)
    N, C, _, _ = feat.size()
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


                
class RDNC_CUT(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, scale=True):
        super(RDNC_CUT, self).__init__()
        nDenselayer = 3
        growthRate = 32
        self.scale = scale
        dilation = 1
        
        self.att1_1x1 = nn.Conv2d(ngf, ngf, kernel_size=1, padding=0, bias=True)
        self.att3_1x1 = nn.Conv2d(ngf*2, ngf, kernel_size=1, padding=0, bias=True)
        self.att1_conv = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.att3_conv = nn.Conv2d(ngf, ngf*2, kernel_size=3, padding=1, bias=True)

        # RDBs 3
        p = 1        
        self.upsample1 = UPConv(ngf*4, ngf*4)
        self.upsample2 = UPConv(ngf*2, ngf*2)
        self.conv1_1 = Bn_Relu_Conv(input_nc, ngf, 3, padding=p, norm_layer=norm_layer)
        self.conv1_2 = Bn_Relu_Conv(ngf, ngf, 3, padding=p, norm_layer=norm_layer)
        self.ca1 = CALayer(ngf, reduction=4)
        
        self.conv2_1 = Bn_Relu_Conv(ngf, ngf*2, 3, padding=p, norm_layer=norm_layer, stride=2)
        self.conv2_2 = Bn_Relu_Conv(ngf*2, ngf*2, 3, padding=p, norm_layer=norm_layer)
        self.ca2 = CALayer(ngf*2, reduction=4)

        self.conv3_1 = Bn_Relu_Conv(ngf*2, ngf*4, 3, padding=p, norm_layer=norm_layer, stride=2)
        self.conv3_2 = RDB(ngf*4, nDenselayer, growthRate, norm_layer, dilation)
        self.conv3_3 = RDB(ngf*4, nDenselayer, growthRate, norm_layer, dilation)
        self.conv3_4 = RDB(ngf*4, nDenselayer, growthRate, norm_layer, dilation)
        self.GFF_1x1 = nn.Conv2d(ngf*4*3, ngf*4, kernel_size=1, padding=0, bias=True)


        self.conv4_1 = Bn_Relu_Conv(ngf*6, ngf*2, 3, padding=p, norm_layer=norm_layer)
        self.ca3 = CALayer(ngf*2, reduction=4)
        self.conv5_1 = Bn_Relu_Conv(ngf*3, ngf, 3, padding=p, norm_layer=norm_layer)
        self.ca4 = CALayer(ngf, reduction=4)

        # conv 
        self.conv3 = nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0, bias=True)
        self.relu = nn.LeakyReLU()


    def forward(self, x, layers=[], encode_only=False):

        F_1 = self.conv1_1(x)
        feature_4 = F_1
        F_1 = self.conv1_2(F_1)
        F_1 = self.ca1(F_1)

        F_2 = self.conv2_1(F_1)
        F_2 = self.conv2_2(F_2)
        F_2 = self.ca2(F_2)
        feature_2 = F_2

        F_3 = self.conv3_1(F_2)        
        F_31 = self.conv3_2(F_3)
        feature_1 = F_31
        F_32 = self.conv3_3(F_31)
        F_33 = self.conv3_4(F_32)
        F_3_ = torch.cat((F_31, F_32, F_33), 1)
        F_3 =  self.relu(self.GFF_1x1(F_3_))
        
        F_2_ = self.relu(self.att3_1x1(F_2))
        F_2 = self.att3_conv(F_2_)
        F_4 = self.upsample1(F_3)
        F_4 = torch.cat((F_4, F_2), 1)
        F_4 = self.conv4_1(F_4)
        F_4 = self.ca3(F_4)
        feature_3 = F_4

        
        F_1_ = self.relu(self.att1_1x1(F_1))
        F_1 = self.att1_conv(F_1_)
        F_5 = self.upsample2(F_4)
        F_5 = torch.cat((F_5, F_1), 1)
        F_5 = self.conv5_1(F_5)
        F_5 = self.ca4(F_5)
        feature_5 = F_5


        output = self.conv3(F_5)
        output = output + x
        if self.scale:
            output = torch.tanh(output)

        if len(layers) > 0:
            feats = [feature_1, feature_2, feature_3, feature_4, feature_5]

        if encode_only:
            # print('encoder only return features')
            return feats  # return intermediate features alone; stop in the last layers
        else:
            """Standard forward"""
            return output                          




        
class Feature_Exacter(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, spectral=True, gpu_ids=[]) -> None:
        super(Feature_Exacter, self).__init__()
        
        p = 1
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.norm_1 = norm_layer(ndf)
        self.norm_2 = norm_layer(ndf*2)
        self.norm_3 = norm_layer(ndf*4)

        if spectral:
            self.conv1_1 = spectral_norm(nn.Conv2d(input_nc, ndf, 4, stride=2, padding=p))
            self.conv1_2 = spectral_norm(nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=p))
            self.downconv_1 = spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=p))
            self.downconv_2 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=p))
            self.downconv_3 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=p))
            self.downconv_4 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=p))
            self.adapt = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0))
            self.adapt_1 = spectral_norm(nn.Conv2d(ndf*2, ndf*4, 1, stride=1, padding=0))
            self.adapt_2 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0))
            self.adapt_3 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0))
            self.adapt_4 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0))
            self.adapt_5 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0))      
            self.adapt_6 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0))
            self.adapt_7 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0))
            self.adapt_8 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0))
            self.adapt_9 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0))
        else:
            self.conv1_1 = nn.Conv2d(input_nc, ndf, 4, stride=2, padding=p)
            self.conv1_2 = nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=p)
            self.downconv_1 = nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=p)
            self.downconv_2 = nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=p)
            self.downconv_3 = nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=p)
            self.downconv_4 = nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=p)           
            self.adapt = nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0)
            self.adapt_1 = nn.Conv2d(ndf*2, ndf*4, 1, stride=1, padding=0)
            self.adapt_2 = nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0)
            self.adapt_3 = nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0)
            self.adapt_4 = nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0)
            self.adapt_5 = nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0)      
            self.adapt_6 = nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0)
            self.adapt_7 = nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0)
            self.adapt_8 = nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0)
            self.adapt_9 = nn.Conv2d(ndf*4, ndf*4, 1, stride=1, padding=0)          
        self.nc = ndf*4
        self.mlp_init = False
        self.gpu_ids = gpu_ids
        self.spectral = spectral
        self.IN = nn.InstanceNorm2d(ndf*4)


    def create_mlp(self, num_layers=5):
        for mlp_id in range(num_layers):
            if self.spectral:
                sequence = [spectral_norm(nn.Linear(self.nc, self.nc)), nn.LeakyReLU(0.2, inplace=True)]
                sequence += [spectral_norm(nn.Linear(self.nc, self.nc)), nn.LeakyReLU(0.2, inplace=True), spectral_norm(nn.Linear(self.nc, 1))]
            else:
                sequence = [nn.Linear(self.nc, self.nc), nn.LeakyReLU(0.2, inplace=True)]
                sequence += [nn.Linear(self.nc, self.nc), nn.LeakyReLU(0.2, inplace=True), nn.Linear(self.nc, 1)]
            mlp = nn.Sequential(*sequence)
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, 'normal', init_gain=0.02, gpu_ids=self.gpu_ids)
        self.mlp_init = True

    def forward(self, input, encode_only=False):
        
        features = []
            
        x = self.norm_1(self.LReLU(self.conv1_1(input)))
        x = self.norm_2(self.LReLU(self.conv1_2(x)))
        feature_1 = self.norm_3(self.LReLU(self.adapt(self.norm_3(self.LReLU(self.adapt_1(x))))))
        feature_1 = self.IN(feature_1)
        features += [feature_1]

        x = self.norm_3(self.LReLU(self.downconv_1(x)))
        feature_2 = self.norm_3(self.LReLU(self.adapt_3(self.norm_3(self.LReLU(self.adapt_2(x))))))
        feature_2 = self.IN(feature_2)
        features += [feature_2]

        x = self.norm_3(self.LReLU(self.downconv_2(x)))
        feature_3 = self.norm_3(self.LReLU(self.adapt_5(self.norm_3(self.LReLU(self.adapt_4(x))))))
        feature_3 = self.IN(feature_3)
        features += [feature_3]

        x = self.norm_3(self.LReLU(self.downconv_3(x)))
        feature_4 = self.norm_3(self.LReLU(self.adapt_7(self.norm_3(self.LReLU(self.adapt_6(x))))))
        feature_4 = self.IN(feature_4)
        features += [feature_4]

        x = self.norm_3(self.LReLU(self.downconv_4(x)))
        feature_5 = self.norm_3(self.LReLU(self.adapt_9(self.norm_3(self.LReLU(self.adapt_8(x))))))
        feature_5 = self.IN(feature_5)
        features += [feature_5]

        if encode_only:
            return features

        if not self.mlp_init:
            self.create_mlp(num_layers=len(features))
        outs = []
        for feat_id, feat in enumerate(features):
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            mlp = getattr(self, 'mlp_%d' % feat_id)
            out = mlp(feat_reshape)
            outs += out

        return outs  


class Feature_Exacter_ue(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, spectral=True, gpu_ids=[]) -> None:
        super(Feature_Exacter_ue, self).__init__()
        
        p = 1
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1_1 = spectral_norm(nn.Conv2d(input_nc, ndf, 4, stride=2, padding=p))
        self.conv1_2 = spectral_norm(nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=p))
        self.downconv_1 = spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=p))
        self.downconv_2 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=p))
        self.downconv_3 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=p))
        self.downconv_4 = spectral_norm(nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=p))
        self.adapt_1 = spectral_norm(nn.Conv2d(ndf*2, 1, 4, stride=1, padding=1))
        self.adapt_3 = spectral_norm(nn.Conv2d(ndf*4, 1, 4, stride=1, padding=1))
        self.adapt_5 = spectral_norm(nn.Conv2d(ndf*4, 1, 4, stride=1, padding=1))      
        self.adapt_7 = spectral_norm(nn.Conv2d(ndf*4, 1, 4, stride=1, padding=1))
        self.adapt_9 = spectral_norm(nn.Conv2d(ndf*4, 1, 4, stride=1, padding=1))
          
        self.nc = ndf*4
        self.gpu_ids = gpu_ids
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, encode_only=False):
        
        features = []
            
        x = self.LReLU(self.conv1_1(input))
        x = self.LReLU(self.conv1_2(x))
        feature_1 = self.sigmoid(self.adapt_1(x))
        features += [feature_1]

        x = self.LReLU(self.downconv_1(x))
        feature_2 = self.sigmoid(self.adapt_3(x))
        features += [feature_2]

        x = self.LReLU(self.downconv_2(x))
        feature_3 = self.sigmoid(self.adapt_5(x))
        features += [feature_3]

        x = self.LReLU(self.downconv_3(x))
        feature_4 = self.sigmoid(self.adapt_7(x))
        features += [feature_4]

        x = self.LReLU(self.downconv_4(x))
        feature_5 = self.sigmoid(self.adapt_9(x))
        features += [feature_5]

        return features        



class UPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, use_bias=True):
        super(UPConv, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias))
    def forward(self, x):
        x = self.main(x)
        return x


def get_act_fun(act_fun_type='LeakyReLU'):
    if isinstance(act_fun_type, str):
        if act_fun_type == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun_type == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun_type == 'SELU':
            return nn.SELU(inplace=True)
        elif act_fun_type == 'none':
            return nn.Sequential()
        else:
            raise NotImplementedError('activation function [%s] is not found' % act_fun_type)
    else:
        return act_fun_type()


class Identity(nn.Module):
    def forward(self, x):
        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return out

def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun

def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class GAM(nn.Module):
    """Global attention module"""
    def __init__(self, in_nc, out_nc, reduction=8, bias=False, use_sn=False, norm=False):
        super(GAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_nc*2, out_channels=in_nc//reduction, kernel_size=1, stride=1, bias=bias, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_nc//reduction, out_channels=out_nc, kernel_size=1, stride=1, bias=bias, padding=0, dilation=1),
        )
        self.fuse = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=in_nc * 2, out_channels=out_nc, kernel_size=1, stride=1, bias=True, padding=0, dilation=1), use_sn),
        )
        self.in_norm = nn.InstanceNorm2d(out_nc)
        self.norm = norm

    def forward(self, x):
        x_mean, x_std = calc_mean_std(x)
        out = self.conv(torch.cat([x_mean, x_std], dim=1))
        out = self.fuse(torch.cat([x, out.expand_as(x)], dim=1))
        if self.norm:
            out = self.in_norm(out) 
        return out

class SNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, use_sn):
        super(SNConv, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn),
        )
    def forward(self, x):
        return self.main(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun, use_sn):
        super(ConvBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        main = []
        main.append(nn.ReflectionPad2d(self.padding))
        main.append(SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn))
        norm_fun = get_norm_fun(norm_fun)
        main.append(norm_fun(out_channels))
        main.append(get_act_fun(act_fun))
        self.main = nn.Sequential(*main)
        
    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    """Generator network"""
    def __init__(self, conv_dim=32, norm_fun='none', act_fun='LeakyReLU', use_sn=False):
        super(Generator, self).__init__()

        ###### encoder
        self.enc1 = ConvBlock(in_channels=3,          out_channels=conv_dim* 1, kernel_size=7, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*3 --> 256*256*32
        self.enc2 = ConvBlock(in_channels=conv_dim*1, out_channels=conv_dim* 2, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*32 --> 128*128*64
        self.enc3 = ConvBlock(in_channels=conv_dim*2, out_channels=conv_dim* 4, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 128*128*64 --> 64*64*128
        self.enc4 = ConvBlock(in_channels=conv_dim*4, out_channels=conv_dim* 8, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 64*64*128 --> 32*32*256
        self.enc5 = ConvBlock(in_channels=conv_dim*8, out_channels=conv_dim*16, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 32*32*256 --> 16*16*512

        ###### decoder
        self.upsample1 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim*16, conv_dim*8, 1, 1, 0, 1, True, use_sn))
        self.upsample2 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 8, conv_dim*4, 1, 1, 0, 1, True, use_sn)) 
        self.upsample3 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 4, conv_dim*2, 1, 1, 0, 1, True, use_sn)) 
        self.upsample4 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 2, conv_dim*1, 1, 1, 0, 1, True, use_sn)) 

        self.dec1 = ConvBlock(in_channels=conv_dim*16, out_channels=conv_dim*8, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 32*32*512 --> 32*32*256
        self.dec2 = ConvBlock(in_channels=conv_dim* 8, out_channels=conv_dim*4, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 64*64*256 --> 64*64*128
        self.dec3 = ConvBlock(in_channels=conv_dim* 4, out_channels=conv_dim*2, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 128*128*128 --> 128*128*64
        self.dec4 = ConvBlock(in_channels=conv_dim* 2, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*64 --> 256*256*32
        self.dec5 = nn.Sequential(
            SNConv(in_channels=conv_dim*1, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False), 
            SNConv(in_channels=conv_dim*1, out_channels=3,          kernel_size=7, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False), 
            nn.Tanh()
        )

        self.ga5 = GAM(conv_dim*16, conv_dim*16, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga4 = GAM(conv_dim* 8, conv_dim* 8, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga3 = GAM(conv_dim* 4, conv_dim* 4, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga2 = GAM(conv_dim* 2, conv_dim* 2, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga1 = GAM(conv_dim* 1, conv_dim* 1, reduction=8, bias=False, use_sn=use_sn, norm=True)
    
    def forward(self, x, layers=[], encode_only=False):
        ### encoder
        x1 = self.enc1( x)
        x2 = self.enc2(x1)
        feature_1 = x2
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        feature_2 = x4
        x5 = self.enc5(x4)
        feature_3 = x5
        x5 = self.ga5(x5)
        
        ### decoder
        y1 = self.upsample1(x5)
        y1 = torch.cat([y1, self.ga4(x4)], dim=1)
        y1 = self.dec1(y1)

        y2 = self.upsample2(y1)
        y2 = torch.cat([y2, self.ga3(x3)], dim=1)
        y2 = self.dec2(y2)
        feature_4 = y2

        y3 = self.upsample3(y2)
        y3 = torch.cat([y3, self.ga2(x2)], dim=1)
        y3 = self.dec3(y3)

        y4 = self.upsample4(y3)
        y4 = torch.cat([y4, self.ga1(x1)], dim=1)
        y4 = self.dec4(y4)
        feature_5 = y4
        res = self.dec5(y4.mul(x1))

        out = torch.clamp((res + x), min=-1.0, max=1.0) 
            
        feats = [feature_1, feature_2, feature_3, feature_4, feature_5]

        if encode_only:
            # print('encoder only return features')
            return feats  # return intermediate features alone; stop in the last layers

        return out


if __name__ == "__main__":
    from torchinfo import summary
    model = RDNC_CUT(3, 3, ngf=64,
                 norm_layer=NoneLayer, use_dropout=False, scale=True)
    batch_size = 2
    summary(model, input_size=(batch_size, 3, 512,512))


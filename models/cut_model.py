import torch
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from packaging import version
import random

class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()


        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        l_neg = l_neg_curbatch.view(-1, npatches)

        out = l_neg / self.opt.nce_T
        loss = self.cross_entropy_loss(out, torch.arange(0,npatches, device=feat_q.device).repeat(batch_dim_for_bmm))

        return loss

class CUTModel(BaseModel):
    def name(self):
        return 'CUTModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        # parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
            parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
            parser.add_argument('--netF_nc', type=int, default=256)
            parser.add_argument('--nce_idt', action='store_true', help='use NCE loss for identity mapping: NCE(G(Y), Y))')
            parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
            parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
            parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
            parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            action='store_true',
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # define the number of times D trained when G trained once
        self.d_iter = opt.d_iter
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        if self.opt.lambda_NCE > 0.0:
            self.loss_names.append('NCE')
            self.loss_names.append('D_real')
            self.loss_names.append('D_fake')
        if self.isTrain and self.opt.nce_idt:
            self.loss_names.append('NCE_Y')

        if self.opt.patch_N > 0:
            self.loss_names.append('D_patch')
            self.loss_names.append('G_patch')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D', 'F']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        if self.opt.patch_N > 0:
            self.model_names.append('D_P')
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G, D
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_g,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, True, opt.spectral_norm)

        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.norm_g, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.netF_nc)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm_d, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids, opt.spectral_norm)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.L1loss = torch.nn.L1Loss()
            self.mseloss = torch.nn.MSELoss()
            self.wd = opt.wd
            # initialize optimizers
            self.optimizer_G = torch.optim.AdamW(self.netG.parameters(),
                                                lr=opt.lr_g, betas=(opt.beta1, 0.999), weight_decay=self.wd)
            self.optimizer_D = torch.optim.AdamW(self.netD.parameters(),
                                                lr=opt.lr_d, betas=(opt.beta1, 0.999), weight_decay=self.wd)
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.opt.patch_N > 0:
                self.netD_P = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm_d, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids, opt.spectral_norm)
                self.optimizer_D_P = torch.optim.AdamW(self.netD_P.parameters(),
                                                lr=opt.lr_d, betas=(opt.beta1, 0.999), weight_decay=self.wd)
                self.optimizers.append(self.optimizer_D_P)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']



    def forward(self):

        self.fake_B = self.netG(self.real_A) 

        if self.opt.patch_N > 0:
            self.fake_B_patch = []
            self.real_A_patch = []
            self.real_B_patch = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patch_N):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_B_patch.append(self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_B_patch.append(self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_A_patch.append(self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])    
  

    def backward_D_basic(self, netD, real, fake):

        pred_real = netD(real)
        pred_fake = netD(fake.detach())            
        if 'fe' in self.opt.netD:
            loss_D_real, loss_D_fake = 0, 0
            for pred_real_, pred_fake_ in zip(pred_real, pred_fake):
                loss_D_real += self.criterionGAN(pred_real_, True) / len(pred_real)
                loss_D_fake += self.criterionGAN(pred_fake_, False) / len(pred_fake)
        else:
            # Real
            loss_D_real = self.criterionGAN(pred_real, True)
            # Fake        
            loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward(retain_graph=True)
        return loss_D, loss_D_real, loss_D_fake


    def calculate_NCE_loss(self, src, tgt):
        
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG(src, self.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)
        n_layers = len(feat_q_pool)

        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_D()                  # calculate gradients for D
            self.backward_G()                  # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.AdamW(self.netF.parameters(), lr=self.opt.lr_g, betas=(self.opt.beta1, 0.999), weight_decay=self.wd)
                self.optimizers.append(self.optimizer_F)


    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.fake_B)

        self.loss_D, self.loss_D_real, self.loss_D_fake = self.backward_D_basic(self.netD, self.real_B, fake_B)
                            
        if self.opt.patch_N > 0:
            loss_D_patch = 0
            for i in range(self.opt.patch_N):
                loss_D_patch += self.backward_D_basic(self.netD_P, self.real_B_patch[i], self.fake_B_patch[i])
            self.loss_D_patch = loss_D_patch / int(self.opt.patch_N)
        

    def backward_G(self):

        if 'fe' in self.opt.netD:
            loss_G = 0
            outs = self.netD(self.fake_B)
            for out in outs:
                loss_G += self.criterionGAN(out, True) / len(outs) * self.opt.batch_size
            self.loss_G = loss_G
        else:    
            # GAN loss D(G(A))
            self.loss_G = self.criterionGAN(self.netD(self.fake_B), True)


        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)                
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.idt_B = self.netG(self.real_B)
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        loss_G_patch = 0
        if self.opt.patch_N > 0:
            for i in range(self.opt.patch_N):
                if 'fe' in self.opt.netD:
                    outs = self.netD(self.fake_B_patch[i])
                    for out in outs:
                        loss_G_patch += self.criterionGAN(out, True)
                else:
                    loss_G_patch += self.criterionGAN(self.netD_P(self.fake_B_patch[i]), True)
        self.loss_G_patch = self.opt.gpatch_weight * loss_G_patch
        
        # combined loss
        self.loss_G = self.loss_G + loss_NCE_both + self.loss_G_patch
        self.loss_G.backward()
    
    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.set_requires_grad(self.netD, False)
        if self.opt.patch_N > 0:
            self.set_requires_grad(self.netD_P, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:
            self.optimizer_F.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:
            self.optimizer_F.step()
        # D
        for _ in range(self.d_iter):
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            if self.opt.patch_N > 0:
                self.set_requires_grad(self.netD_P, True)
                self.optimizer_D_P.zero_grad()        
            self.backward_D()
            if self.opt.patch_N > 0:
                self.optimizer_D_P.step()
            self.optimizer_D.step()

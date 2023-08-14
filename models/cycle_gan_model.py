import torch
import itertools
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        # parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # define the number of times D trained when G trained once
        self.d_iter = opt.d_iter
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')
            self.loss_names.append('idt_A')
            self.loss_names.append('idt_B')
        if self.opt.lambda_gp > 0.0:
            self.loss_names.append('GP_A')
            self.loss_names.append('GP_B')


        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_g,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, True, opt.spectral_norm)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm_g,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, True, opt.spectral_norm)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm_d, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids, opt.spectral_norm, opt.attention)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm_d, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids, opt.spectral_norm)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, loss=self.opt.loss).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.L1loss = torch.nn.L1Loss()
            self.mseloss = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr_g, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr_d, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)       
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake, loss='none'):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        if 'fe' in self.opt.netD:
            loss_D_real, loss_D_fake = 0, 0
            for pred_real_, pred_fake_ in zip(pred_real, pred_fake):
                loss_D_real += self.criterionGAN(pred_real_, True) / len(pred_real)
                loss_D_fake += self.criterionGAN(pred_fake_, False) / len(pred_fake)
        else:
            # Real
            if loss == 'hinge':
                loss_D_real = nn.ReLU()(1.0 - pred_real).mean()
            elif loss == 'wasserstein':
                loss_D_real = - pred_real.mean()
            elif loss == 'relative':
                rel_t = pred_real-pred_fake
                rel_t.requires_grad_(True)
                loss_D_real = self.criterionGAN(rel_t, True)
            else:    
                loss_D_real = self.criterionGAN(pred_real, True)
            # Fake        
            if loss == 'hinge':
                loss_D_fake = nn.ReLU()(1.0 + pred_fake).mean()
            elif loss == 'wasserstein':
                loss_D_fake = pred_fake.mean()
            elif loss == 'relative':
                rel_f = pred_fake-pred_real
                rel_f.requires_grad_(True)
                loss_D_fake = self.criterionGAN(rel_f, False)
            else:
                loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        if loss == 'hinge' or loss == 'wasserstein':
            loss_D = loss_D_real + loss_D_fake
        else:
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward(retain_graph=True)
        return loss_D

    def calc_gradient_penalty(self, netD, real_images, fake_images):
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = alpha * real_images.data + (1 - alpha) * fake_images.data
        interpolated.requires_grad_(True)
        # interpolated = torch.tensor(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        out = netD(interpolated)
        grad = torch.autograd.grad(outputs=out,
                                inputs=interpolated,
                                grad_outputs=torch.ones(out.size()).cuda(),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]  

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        d_loss_gp.backward(retain_graph=True)
        return d_loss_gp     


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
            self.backward_D_A()                  # calculate gradients for D
            self.backward_D_B()
            self.backward_G()                  # calculate graidents for G

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, self.opt.loss)
        if self.opt.lambda_gp > 0.0:
            self.loss_GP_A = self.calc_gradient_penalty(self.netD_A, self.real_B, fake_B) * self.opt.lambda_gp

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, self.opt.loss)
        if self.opt.lambda_gp > 0.0:
            self.loss_GP_B = self.calc_gradient_penalty(self.netD_B, self.real_A, fake_A) * self.opt.lambda_gp

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0


        if self.opt.loss == 'relative':
            fake_B = self.fake_B_pool.query(self.fake_B)
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_G_A = self.backward_D_basic(self.netD_A, fake_B, self.real_B, self.opt.loss)
            self.loss_G_B = self.backward_D_basic(self.netD_B, fake_A, self.real_A, self.opt.loss)
        else:
            if 'fe' in self.opt.netD:
                loss_G_A = 0  
                loss_G_B = 0                
                outs_A = self.netD_A(self.fake_B)
                outs_B = self.netD_B(self.fake_A)
                for out in outs_A:
                    loss_G_A += self.criterionGAN(out, True) / 5
                self.loss_G_A = loss_G_A
                for out in outs_B:
                    loss_G_B += self.criterionGAN(out, True) / 5
                self.loss_G_B = loss_G_B
            else:
                # GAN loss D_A(G_A(A))
                self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
                # GAN loss D_B(G_B(B))
                self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B 
        self.loss_G.backward()
    
    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        for _ in range(self.d_iter):
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()

from .base_model import BaseModel
from . import networks
from .cycle_gan_model import CycleGANModel
import torch
import random

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = CycleGANModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(resize_or_crop='none')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix, 'D']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm_g, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, True, opt.spectral_norm)
        self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                    opt.n_layers_D, opt.norm_d, opt.no_lsgan, opt.init_type, opt.init_gain, self.gpu_ids, opt.spectral_norm)
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        if self.opt.patch_N > 0:
            self.model_names.append('D_P')
            self.netD_P = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm_d, opt.no_lsgan, opt.init_type, opt.init_gain, self.gpu_ids, opt.spectral_norm)
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        
        if self.opt.amp:
            with torch.cuda.amp.autocast():
                self.fake_B = self.netG(self.real_A)
        else:
            self.fake_B = self.netG(self.real_A)

    def testd(self, label=True):
        with torch.no_grad():
            pred_real = self.netD(self.real_A)
                      
            if 'fe' in self.opt.netD:
                loss_D_real, loss_D_fake = 0, 0
                for pred_real_ in pred_real:
                    loss_D_real += self.criterionGAN(pred_real_, label) / len(pred_real)
            else:
                # Real
                loss_D_real = self.criterionGAN(pred_real, label)
            # Combined loss
            loss_D = loss_D_real
        if self.opt.patch_N > 0:
            self.real_A_patch = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patch_N):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.real_A_patch.append(self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])  
                            
            loss_D_patch = 0
            for i in range(self.opt.patch_N):
                pred_real = self.netD_P(self.real_A_patch[i])
                loss_D_patch += self.criterionGAN(pred_real, label)

            loss_D_patch = loss_D_patch / int(self.opt.patch_N)
            return loss_D, loss_D_patch
        return loss_D


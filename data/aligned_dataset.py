import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, _resize
from data.image_folder import make_dataset
import imageio
import numpy as np


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'pairA')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'pairB')
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size  = len(self.A_paths)
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = imageio.imread(A_path)
        B = imageio.imread(B_path)
        A = _resize(A, (self.opt.loadSize, self.opt.loadSize))
        B = _resize(B, (self.opt.loadSize, self.opt.loadSize))

        h, w = A.shape[:2]
        x = 0 if self.opt.fineSize == h else np.random.randint(0, h - self.opt.fineSize)
        y = 0 if self.opt.fineSize == w else np.random.randint(0, w - self.opt.fineSize)
        A = A[x:(self.opt.fineSize + x), y:(self.opt.fineSize + y), :]
        B = B[x:(self.opt.fineSize + x), y:(self.opt.fineSize + y), :]
        
        if np.random.uniform() < 0.5:
            A = A[:, ::-1, :]
            B = B[:, ::-1, :]
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)        
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        
        # w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        # A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        # B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        A = transforms.ColorJitter(brightness=0.2, saturation=0.2)(A)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'

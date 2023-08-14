import os.path
from data.base_dataset import BaseDataset, get_transform_test
from data.image_folder import make_dataset
import imageio
import numpy as np
import torch
import torchvision.transforms as transforms

class SingleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform_test(test_size=1024)

        self.ttensor = transforms.ToTensor()

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        # A_img = Image.open(A_path).convert('RGB')
        A_img = imageio.imread(A_path)
        A_img = (A_img / 65535).astype(np.float32) if A_path.endswith('tiff') else A_img
        
        A = self.transform(A_img)
        
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        out_dict = {'A': A, 'A_paths': A_path}
            
        return out_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'

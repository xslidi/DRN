import cv2
cv2.setNumThreads(0)
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import math


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0



def __scale_width(img, target_width):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def adjust2test(image):
    ow, oh, _ = image.shape

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 32
    if ow % mult == 0 and oh % mult == 0:
        return image
    w = (ow) // mult
    w = (w) * mult
    h = (oh) // mult
    h = (h) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)
    
    img = _center_crop(image, (w, h))
    return img   



def _resize(img, loadsize, keep_ratio=False, train=True):
    if np.minimum(loadsize[0],loadsize[0]) >= np.minimum(img.shape[0],img.shape[1]):
        return img
    if not keep_ratio:
        shrink_h = img.shape[0] / loadsize[0]
        shrink_w = img.shape[1] / loadsize[1]
        shrink = np.minimum(shrink_h,shrink_w)
        if shrink < 1 and not train:
            osize = (img.shape[1], img.shape[0])
            print("smaller image, not resized!")
        else:
            osize = (int(img.shape[1] / shrink), int(img.shape[0] / shrink))
    else:
        osize = loadsize
    resized_img = cv2.resize(img, osize, interpolation=cv2.INTER_AREA)
    return resized_img

def _random_crop(img, crop_size):
    h, w = img.shape[:2]
    if crop_size[0] <= h and crop_size[1] <= w:
        x = 0 if crop_size[0] == h else np.random.randint(0, h - crop_size[0])
        y = 0 if crop_size[1] == w else np.random.randint(0, w - crop_size[1])
        return img[x:(crop_size[0] + x), y:(crop_size[1] + y), :]
    else:
        print("Warning: Crop size is larger than original size")
        return img

def _center_crop(img, crop_size):
    h, w = img.shape[:2]
    if crop_size[0] <= h and crop_size[1] <= w:
        x = math.ceil(h - crop_size[0]) // 2
        y = math.ceil(w - crop_size[1]) // 2
        return img[x:(crop_size[0] + x), y:(crop_size[1] + y), :]
    else:
        print("Warning: Crop size is larger than original size")
        return img

def _horizontal_flip(im, prob=0.5):
    """Performs horizontal flip (used for training)."""
    return im[:, ::-1, :] if np.random.uniform() < prob else im 


def get_transform(opt, norm=True, jitter=False):
    transform_list = []
    osize = (opt.loadSize, opt.loadSize)
    fsize = (opt.fineSize, opt.fineSize)
    if opt.resize_or_crop == 'resize_and_crop':
        transform_list.append(transforms.Lambda(lambda img: _resize(img, osize, train=opt.isTrain)))
        transform_list.append(transforms.Lambda(lambda img: _random_crop(img, fsize)))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.Lambda(lambda img: _random_crop(img, fsize)))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.Lambda(lambda img: _random_crop(img, fsize)))
    elif opt.resize_or_crop == 'none':
        pass
        # transform_list.append(transforms.Lambda(
        #     lambda img: __adjust(img)))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: _horizontal_flip(img)))


    transform_list.append(transforms.Lambda(lambda img: np.ascontiguousarray(img)))
    if norm:
        transform_list += [transforms.ToTensor()]
        if jitter:
            transform_list += [transforms.ColorJitter(brightness=0.4, saturation=0.2)]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_test(test_size=1024):
    transform_list = []
    osize = (test_size, test_size)
    transform_list.append(transforms.Lambda(lambda img: _resize(img, osize, False, False)))
    transform_list.append(transforms.Lambda(lambda img: adjust2test(img)))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

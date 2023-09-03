# Learning Deep Representations for Photo Retouching 
By [Di Li](https://scholar.google.com/citations?user=r6WfS2gAAAAJ), and [Susanto Rahardja](https://scholar.google.com/citations?user=OdkA4jMAAAAJ&hl=en) 

## Introduction

The codebase provides the official PyTorch implementation for the paper [" Learning Deep Representations for Photo Retouching"](https://ieeexplore.ieee.org/abstract/document/10227607) (accepted by IEEE Transactions on Multimedia).

<p align="center">
  <img src="figures/pipeline.jpg" />
</p>

In this project, we present a a novel framework to retouch the degraded photos towards a specific photographic style in an unsupervised fashion. To be specific, we unify the design philosophy of the generator and the discriminator into a multi-scale form and reuse these powerful networks as feature extractors to obtain deep latent representations of the photos with varying scales. Then, we employed projection heads to map these deep representations to a neater loss space for evaluation. In particular, we utilized a contrastive scheme for generator to keep the content consistency and a cross entropy scheme for discriminator to improve the aesthetic quality. Powered by this simple yet efficient design philosophy, our algorithm surpasses previous algorithms on two well known datasets with photo enhancement task. As a learning framework compliable with various enhancement networks, our algorithm shows great potential to be a practical tool to efficiently and effectively build photo retouching systems with different needs.

## Dependencies 

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- Opencv
- Imageio
- [visdom](https://github.com/facebookresearch/visdom)

## Datasets

The paper use the [FiveK](https://data.csail.mit.edu/graphics/fivek/) and [HDR+](http://www.hdrplusdata.org/) datasets for experiments.

- FiveK : You can download the original FiveK dataset from the dataset [homepage](https://data.csail.mit.edu/graphics/fivek/) and then process images using Adobe Lightroom.
  - To generate the input images, in the Collections list, select the collection Input `with Daylight WhiteBalance minus 1.5`.  
  - To generate the target images, in the Collections list, select the collection `Experts/C`.  
  - All the images are converted to `.PNG` format.

- HDR+ : You can download the original HDR+ dataset from the dataset [homepage](http://www.hdrplusdata.org/) and then process images using [rawpy](https://github.com/letmaik/rawpy).

The final directory structure is as follows.

```
./data/FiveK
    trainA/         # 8-bit sRGB train inputs
    trainB/         # 8-bit sRGB train groundtruth
    testA/          # 8-bit sRGB test inputs
    testB/          # 8-bit sRGB test groundtruth
```
## Train
- run visdom to monitor status
```
visdom
```
- run
```bash
python train.py --name DRN --dataroot ./data/FiveK --batch_size 2 --gpu_ids 0 --netG rdnccut --model cut --lambda_NCE 10 --nce_includes_all_negatives_from_minibatch --ndf 32 --netD fe --niter 20 --niter_decay 80 --spectral_norm
```

## Test
- run
```bash
python test.py --dataroot ./data/FiveK/testA --name DRN --gpu_ids 0 --netG rdnccut 
```

## Citation
If you find this repository useful, please kindly consider citing the following paper:

```BibTeX
@article{li2023learning,
title={Learning Deep Representations for Photo Retouching},
author={Li, Di and Rahardja, Susanto},
journal={IEEE Transactions on Multimedia},
year={2023},
publisher={IEEE}
}
```

## License

Our project is licensed under a [MIT License](LICENSE).
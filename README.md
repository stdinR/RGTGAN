# RGTGAN (IEEE TGRS 2024)
PyTorch implementation and dataset for "[RGTGAN: Reference-Based Gradient-Assisted Texture-Enhancement GAN for Remote Sensing Super-Resolution](https://doi.org/10.1109/tgrs.2024.3359095)", **IEEE Transactions on Geoscience and Remote Sensing (TGRS) 2024**.

## Abstract
> Reference-based super-resolution (Ref-SR) is a heated topic distinguished from single-image super-resolution (SISR). It aims at transferring more texture details from reference (Ref) image with a different perspective, to super-resolve the low-resolution (LR) image. However, the development of Ref-SR within remote sensing (RS) community is limited by three problems. First, RS images exhibit more complex texture details compared to natural images. It’s challenging to learn and reconstruct fine texture of RS images. Second, the lack of high-quality RS image dataset, which contains massive RS image pairs from different perspectives, hampers the model training and diminishes the generalization of Ref-SR within RS community. Third, the lack of physical system prevents it from verifying the feasibility of Ref-SR in RS practice. To address these problems, this paper proposes a novel reference-based gradient-assisted texture-enhancement GAN (RGTGAN), a novel dataset, namely KaggleSRD, and a novel physical simulation system, namely dual-zoom-lens system (DZLS). Specifically, this paper proposes a gradient-assisted texture-enhancement module (GTEM) to fully release the potential of gradient branch to learn fine structures during feature extraction process, a novel dense-intern deformable convolution (DIDConv) to boost the alignment effect between features from different image branches during feature alignment process, and a novel dense-restore-residual (DRR) module to effectively transfer features. Extensive experimental results on both datasets, RRSSRD and KaggleSRD, demonstrate the superiority of the proposed method over state-of-the-art methods. Furthermore, DZLS verifies promising application prospects of the proposed method.

## Pre-requisites
- Python 3.6+ on Ubuntu 22.04
- CUDA >= 11.1 and gcc > 7 (for DCN_v3 installation) and corresponding supported pyTorch
- Python packages:
```
pip install numpy pyyaml opencv-python scipy
```
- DCN_v3 installation: (more about DCN_v3 please refer to [InternImage](https://github.com/OpenGVLab/InternImage))
```
cd ./RGTGAN/codes/ops_dcn_v3
```
```
sh make.sh
```

## Download Dataset
- KaggleSRD, providing 501 pairs of training images and 4 groups of test sets, can be downloaded from [google drive](https://drive.google.com/file/d/1GfcPBMmpc7Rmj-FPVW93mv3GGqBzoMFn/view?usp=drive_link)
- RRSSRD, dataset from [RRSGAN](https://github.com/dongrunmin/RRSGAN), can be downloaded from [baidu pan](https://pan.baidu.com/share/init?surl=M5HAlb9DqO5IOWQexETFaw), passwword:lnff

## Data Preprocess
- **Step I.** Unzip and put the training data and test data in the folder as
  - `./dataset/train/train_KaggleSRD/HR(and others)`
  - `./dataset/train/val_KaggleSRD/HR(and others)`
- **Step II.** Transform the images set into LMDB format for faster IO speed via
```
cd ./dataset/data_script
```
```
python create_lmdb.py
```
- **Step III.** Download the [pretrained vgg model](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) and put it in
  - `./codes/models/archs/pretrained_model`

## Train
- **Step I.** Enter directory:
```
cd ./codes/example/RGTGAN
```
- Modify the dataroots for train and val based on your actual root in `./codes/example/RGTGAN/options/RGTGAN.yml`
- Start training:
```
sh train.sh
```
- Also, you can change other training configurations in `RGTGAN.yml` for addtional experiments

## Test
- Enter directory:
```
cd ./codes/example/RGTGAN
```
- Modify `val.sh` and `val.py` based on your configurations
- Start Testing:
```
sh val.sh
```
- Note that the evaluation results in the paper are asseseed differently from the codes where only PSNR and SSIM scores are calculated. You should unify your evluation metrics calculation standards in your research.

## Results
- RRSSRD test set:
<p align="center">
  <img src="figures/Fig10.jpg">

- KaggleSRD test set:
<p align="center">
  <img src="figures/Fig11.jpg">

- Real-world data:
<p align="center">
  <img src="figures/Fig12.jpg">

</p>

## Citation
If you find this code useful for your research, please consider citing our paper:
``````
@ARTICLE{10415231,
  author={Tu, Ziming and Yang, Xiubin and He, Xi and Yan, Jiapu and Xu, Tingting},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={RGTGAN: Reference-Based Gradient-Assisted Texture-Enhancement GAN for Remote Sensing Super-Resolution}, 
  year={2024},
  volume={62},
  number={},
  pages={1-21},
  keywords={Image resolution;Image reconstruction;Feature extraction;Superresolution;Transformers;Generative adversarial networks;Electronic mail;Enhanced texture;generative adversarial network (GAN);gradient;reference-based super-resolution (Ref-SR);remote sensing (RS) imagery},
  doi={10.1109/TGRS.2024.3359095}}
``````


## Acknowledgement
The code is based on [MMSR](https://github.com/open-mmlab/mmagic) and [RRSGAN](https://github.com/dongrunmin/RRSGAN). We thank the authors for their excellent contributions.


## Contact
If you have any questions about our work, please contact [tuziming21@mails.ucas.ac.cn](tuziming21@mails.ucas.ac.cn).

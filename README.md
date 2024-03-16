# RGTGAN (IEEE TGRS 2024) (To be fully updated before March 23)
PyTorch implementation for RGTGAN, titled 'RGTGAN: Reference-Based Gradient-Assisted Texture-Enhancement GAN for Remote Sensing Super-Resolution' and dataset for KaggleSRD.

## Pre-requisites
- Python 3.6+ on Ubuntu 22.04
- CUDA >= 11.1 and gcc > 7 (for DCN_v3 installation) and corresponding supported pyTorch
- Python packages:
  - `pip install numpy pyyaml opencv-python scipy`
- DCN_v3 installation: (more about DCN_v3 please refer to [InternImage](https://github.com/OpenGVLab/InternImage))
  - `cd ./RGTGAN/codes/ops_dcn_v3` then 
  - `sh make.sh`

## Download Dataset
- KaggleSRD, providing 501 pairs of training images and 4 groups of test sets, can be downloaded from [google drive](https://drive.google.com/file/d/1GfcPBMmpc7Rmj-FPVW93mv3GGqBzoMFn/view?usp=drive_link)
- RRSSRD, dataset from [RRSGAN](https://github.com/dongrunmin/RRSGAN), can be downloaded from [baidu pan](https://pan.baidu.com/share/init?surl=M5HAlb9DqO5IOWQexETFaw), passwword:lnff

## Data Preprocess
- Unzip and put the training data and test data in the folder as
  - `./dataset/train/train_KaggleSRD/HR(and others)`
  - `./dataset/train/val_KaggleSRD/HR(and others)`
- Transform the images set into LMDB format for faster IO speed via
  - `cd ./dataset/data_script`
  - `python create_lmdb.py`
- Download the [pretrained vgg model](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) and put it in
  - `./codes/models/archs/pretrained_model`

## Train
- Enter directory: `cd ./codes/example/RGTGAN`
- Modify the training settings in `./codes/example/RGTGAN/options/RGTGAN.yml`
- Start training: `sh train.sh`

## Test
- Enter directory: `cd ./codes/example/RGTGAN`
- Modify `val.sh` and `val.py` based on your configurations
- `sh val.sh`

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

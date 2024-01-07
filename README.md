# RGTGAN
This reposity records code for RGTGAN, titled 'Reference-Based Gradient-Assisted Texture-Enhancement GAN for Remote Sensing Super-Resolution' and dataset for KaggleSRD. Both of which are aimed for reference-based super-resolution (RefSR) in the field of remote sensing.

## Pre-requisites
* Python 3.6+ on Ubuntu 22.04
* CUDA > 11.1 and gcc > 7 (for DCN_v3 installation) and corresponding supported pyTorch
* Python packages: `pip install numpy pyyaml opencv-python scipy`
* DCN_v3 installation: (more about DCN_v3 please refer to [InternImage](https://github.com/OpenGVLab/InternImage)) `cd ./RGTGAN/codes/ops_dcn_v3` then `sh make.sh`

## Download Dataset
- KaggleSRD, providing 1717 pairs of HR-Ref image pairs, can be downloaded from [google drive](https://drive.google.com/file/d/1LydosS8NQeloly5vU7TX-vr1fZks6AXz/view?usp=drive_link)
- RRSSRD, dataset from [RRSGAN](https://github.com/dongrunmin/RRSGAN), can be downloaded from [baidu pan](https://pan.baidu.com/share/init?surl=M5HAlb9DqO5IOWQexETFaw), passwword:lnff

## Data Preprocess
- Unzip and put the training data in the folder as `./dataset/train/(HR and Ref)`
- Generate corresponding LR image, Bic image and resampled Ref image via `cd ./dataset/data_script` then `python create_set.py`
- Transform the training image set into LMDB format for faster IO speed via `cd ./dataset/data_script` then `python create_lmdb.py`

## Train
- To be open soon

## Acknowledgement
The code is based on [MMSR](https://github.com/open-mmlab/mmagic) and [RRSGAN](https://github.com/dongrunmin/RRSGAN). We thank the authors for their excellent contributions.

# Refined Edge Detection With Cascaded and High-Resolution Convolutional Network

This repository contains the PyTorch implementation for 
"Refined Edge Detection With Cascaded and High-Resolution Convolutional Network" which is based on the code in https://github.com/zhuoinoulu/pidinet



## Training, and Generating edge maps

### Training
```
python main.py --model chrnet --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/savedir --datadir /path/to/BSDS500 --dataset BSDS
```
### Generating edge maps using the original model
```
python main.py --model chrnet -j 4 --gpu 0 --savedir /path/to/savedir --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/savedir/save_models/checkpointxxx.pth
```

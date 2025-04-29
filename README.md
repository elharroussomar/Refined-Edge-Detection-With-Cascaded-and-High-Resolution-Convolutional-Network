# CHRNet: Refined Edge Detection With Cascaded and High-Resolution Convolutional Network

This repository contains the PyTorch implementation for 
"Refined Edge Detection With Cascaded and High-Resolution Convolutional Network" which is based on the code in https://github.com/zhuoinoulu/pidinet


<img src="https://github.com/elharroussomar/chrnet/blob/main/Model.jpg" alt="Italian Trulli">


### Training
```
python main.py --model chrnet --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/savedir --datadir /path/to/BSDS500 --dataset BSDS
```
### Generating edge maps using the original model
```
python main.py --model chrnet -j 4 --gpu 0 --savedir /path/to/savedir --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/savedir/save_models/checkpointxxx.pth
```
<h1>Citation</h1>

<div class="snippet-clipboard-content position-relative" data-snippet-clipboard-copy-content="@article{elharrouss2023refined,
  title={Refined Edge Detection With Cascaded and High-Resolution Convolutional Network},
  author={Elharrouss, Omar and Hmamouche, Youssef and Idrissi, Assia Kamal and El Khamlichi, Btissam and El Fallah-Seghrouchni, Amal},
  journal={Pattern Recognition},
  pages={109361},
  year={2023},
  publisher={Elsevier}
}
"><pre><code>@article{elharrouss2023refined,
  title={Refined Edge Detection With Cascaded and High-Resolution Convolutional Network},
  author={Elharrouss, Omar and Hmamouche, Youssef and Idrissi, Assia Kamal and El Khamlichi, Btissam and El Fallah-Seghrouchni, Amal},
  journal={Pattern Recognition},
  pages={109361},
  year={2023},
  publisher={Elsevier}
}
</code></pre></div>

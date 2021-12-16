# MPSN
Code for MPSN: Motion-aware Pseudo Siamese Network for Indoor Video Head Detection

## Dependencies
- The code is tested on Ubuntu 20.04.2,python 3.9.

- install torch version 

  ```bash
  conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
  ```

- install cupy, you can install via 

  ```bash
  pip install cupy-cuda111
  ```
- install numpy

## Installation
 1. Install pytorch
 2. Clone this repository
  ```bash
  git clone https://github.com/pl-share/MPSN
  ```
  

## Datasets

- Download the restaurant dataset from the following [link](https://drive.google.com/drive/folders/1NBfgT20ePGDk2iW5aF_T61-yIvwEKvfd). Unzip it and store the dataset in the `data/ `folder

- RGBdata-flow.zip : The original and light flow images

- RGBdata_diff.zip : The original and frame difference images



## Eval

- Download model from the following [link](https://drive.google.com/drive/folders/14M5tHUYqraaNP2GmxDYGED4ja91pSR2J?usp=sharing).
- Store the head detection model in checkpoints/output/ folder.
- Run the following python command from the root folder.

```Bash
python eval.py --model_path <model_path>
```

- if you want to eval other model,you should modify head_backbone.py 

```python 
        addnet = right_res()  #line 59
        left_vgg = left_res() #line 60
```

​	Restaurant datasets model

|        model        | test ap | Anchor scale | pertrained model |
| :-----------------: | :-----: | :----------: | :--------------: |
|  flow mob DFA  |  0.790  |    [2,4]     |       True       |
| diff mob DFA=+|  0.838  |    [2,4]     |       True       |
| diff resnet DFA|  0.802  |    [2,4]     |       True       |
|  diff vgg DFA  |  0.824  |    [8,16]    |       True       |
|diff vgg DFA=+ |0.857|[8,16]|True|

- if you load 'diff vgg DFA+APC',  please modify eval.py

```python
from src.head_detector_vgg16 import Head_Detector_VGG16 #line 14
from trainer import Head_Detector_Trainer   #line 15

head_detector_mpsn = Head_Detector_VGG16(ratios=[1], anchor_scales=[8, 16])    #line 142													 #line 142
#head_detector_mpsn = mob(ratios=[1], anchor_scales=[2,3])     #line 143
```
- if you choose DFA=+ +APC , please modify train_or.py and head_detector.py

```python
#hf2 = t.mul(h1, t.sigmoid(h2)) + h2
hf2 = h1+h2
```
- if backbone is vggnet, please modify trainer.py and head_detector1.py

## Training

```Bash
python train.py
```
## Acknowledgement

This work builds on many of the excellent works:
- [FCHD-Fully-Convolutional-Head-Detector](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector) by Aditya Vora


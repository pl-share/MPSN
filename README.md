# MPSN
Code for MPSN: Motion-aware Pseudo Siamese Network for Indoor Video Head Detection

## Dependencies
- The code is tested on Ubuntu 20.04.2. 
- torch version 1.8.0+cu11

## Datasets

- Download the restaurant dataset from the following [link](https://drive.google.com/drive/folders/1NBfgT20ePGDk2iW5aF_T61-yIvwEKvfd).
- RGBdata-flow.zip : The original and light flow images
- RGBdata_diff.zip : The original and frame difference images


## Training

```Bash
python train.py
```

## Eval

```Bash
python eval.py
```

​	Restaurant datasets model

|        model        | test ap | Anchor scale | pertrained model |
| :-----------------: | :-----: | :----------: | :--------------: |
|  flow mob DFA  |  0.790  |    [2,4]     |       True       |
| diff mob DFA=+|  0.838  |    [2,4]     |       True       |
| diff resnet DFA|  0.802  |    [2,4]     |       True       |
|  diff vgg DFA  |  0.824  |    [8,16]    |       True       |
|diff Vvgg DFA=+ |0.857|[8,16]|True|

- Download model from the following [link](https://drive.google.com/drive/folders/14M5tHUYqraaNP2GmxDYGED4ja91pSR2J?usp=sharing).

- if you load 'diff vgg DFA+APC',  please modify eval.py

```python
from src.head_detector_vgg16 import Head_Detector_VGG16 #line 14
from trainer import Head_Detector_Trainer   #line 15

head_detector_vgg16 = Head_Detector_VGG16(ratios=[1], anchor_scales=[8, 16])    #line 142													 #line 142
#head_detector_vgg16 = mob(ratios=[1], anchor_scales=[2,3])     #line 143
```
- if you choose DFA=+ +APC , please modify train_or.py and head_detector.py

```python
#hf2 = t.mul(h1, t.sigmoid(h2)) + h2
hf2 = h1+h2
```
- if backbone is vggnet, please modify trainer.py and head_detector1.py

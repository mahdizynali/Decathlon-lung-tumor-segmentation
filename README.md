### Introduction
This repository contains a custom SegNet-based deep learning model for segmenting lung tumor regions from MRI scans, specifically using the Decathlon Lung Tumor dataset (via Kaggle).
The architecture has been enhanced with Residual Blocks and Attention Mechanisms, which significantly improve segmentation accuracy by allowing the model to learn deeper and more relevant features from the data.
```
https://goo.gl/QzVZcm
```
I have been inspited this project from Ola-Vish last one:
```
https://github.com/Ola-Vish/lung-tumor-segmentation
```
In order to train your model just try :
```
python3 train.py
```
---
### Model Structure
![alt text](https://raw.githubusercontent.com/mahdizynali/Decathlon-lung-tumor-segmentation/refs/heads/main/display/segnet.jpg)

---
| Dataset    | Dice | IoU | Loss  |
|------------|------|-----|-------|
| Train      | 94   | 88  | 0.002 |
| Validation | 82   | 76  | 0.04  |
---

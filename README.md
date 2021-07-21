
This repository is made for the paper "Segmentation of Intracellular Structures in Fluorescence Microscopy Images by Fusing Low-Level Features".

---

### 1. Overview

This work focuses on the fluorescence microscopy images (FLMIs) segmentation using deep learning techniques. We have found out that the low level features are playing enssential roles in segmenting FLMIs and a proper fusion operator can dramatically improve the performance of a general-purpose model in this task. In this repository, we provide with our implementations to improve the segmentation models in segmenting FLMIs.

### 2. Data Preparation

(1) The customized datasets used in our work can be downloaded from [IEEE Dataport](https://ieee-dataport.org/documents/fluorescence-microscopy-image-datasets-deep-learning-segmentation-intracellular-orgenelle), including the FLMIs of endoplasmic reticulum (ER) and mitochondria (MITO). All images are cropped into $256\times256$ pixels with manual anntotated masks. In addition, we also used a public dataset NUCLEUS, which can be found from https://bbbc.broadinstitute.org/BBBC038. The three datasets include different morphologies of the intracellular structures (ICSs).

<img src="\images\ER.png" style="zoom:80%;" /><img src="\images\MITO.png" style="zoom:80%;" /><img src="\images\NUCLEUS.png" style="zoom: 80%;" />

(2) Data augmentation: Horizontal and vertical flipping as well as 90°/180°/270° rotation were used to augment the training data.

(3) For the ER dataset, the training and test sets consist of 1232 and 38 images, respectively. 
(4) For the MITO dataset, the training and test sets consist of 1320 and 18 images, respectively.              
(5) For the NUCLEUS dataset, we split the original annotated images into training and test sets, consisting of 6360 and 143 images, respectively.

### 3. How to use

(1) Dependencies

This implementation requires the following libraries:

* PyTorch version 1.2.0
* Python version 3.7
* OpenCV version 4.5.1

(2) Training

* To train the model, save the paths of the images and masks in a **_.txt** file and put it into the dictionary **datasets**, then run **trainer__(model).py** for different models.

* The loss funcitons are in **models/optimize.py**

* Most models are trained with 30 epochs.

（3）Evaluation

* To test the performance, please run **inference.py** to get different metrics scores such as IOU, F1 and others. 

* Then run **inference_img.py** to get the segmentations. 

* In addition, you can run **inference_unet.py** to get the performance of the modified-UNet.

### 5. Contributing 
Codes and datasets for this project are developped at CBMI Group (Computational Biology and Machine Intelligence Group, National Laboratory of Pattern Recognition, INSTITUTE OF AUTOMATION, CHINESE ACADEMY OF SCIENCES).

If you would use our data or the results of our work, please cite:
@inproceedings{guo2021prcv,
  title={Segmentation of Intracellular Structures in Fluorescence Microscopy Images by Fusing Low-Level Features},
  author={Guo, Yuanhao and Huang, Jiaxing and Zhou, Yanfeng and Luo, Yaoru and Li, Wenjing and Yang, Ge},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  year={2021}
}
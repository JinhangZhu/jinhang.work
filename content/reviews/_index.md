---
title: "Object detection"
date: 2020-06-17T16:00:40+01:00
draft: false
---

本页包含内容：读论文总结，论文链接，论文/算法简要概括，代码实现链接等。有关目标检测更详尽的汇总： [Object Detection - handong1587](https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html#non-maximum-suppression-nms)

目标检测就是在给定的图片里找到物体的位置，并且标注它们的类别，所以需要解决的问题就是：物体在哪里和物体是什么。目前主流的目标检测算法分为三类：

<img src="https://i.loli.net/2020/07/17/jh2MNsnJHTIySFQ.png" title="Object Detection in 20 Years: A Survey">

1. 传统的目标检测算法，如滑窗+AdaBoost+Cascade，Cascade+HoG/DPM+SVM等变体；

2. 两阶段的候选框提取+分类的算法，首先提取出候选区域ROI，然后对它们进行深度学习方法为主的分类，如R-CNN, SPP-Net, Fast R-CNN, Faster R-CNN, R-FCN等。

   | 算法         | 主要组成                     | 备注                                                         |
   | ------------ | ---------------------------- | ------------------------------------------------------------ |
   | R-CNN        | Selective search + CNN + SVM | [Paper](https://arxiv.org/abs/1311.2524)\|[Code](https://github.com/rbgirshick/rcnn) |
   | SPP-Net      | ROI Pooling                  |                                                              |
   | Fast R-CNN   | Selective search + CNN + ROI |                                                              |
   | Faster R-CNN | RPN + CNN + ROI              |                                                              |
   | R-FCN        |                              |                                                              |

3. 一阶段的基于深度学习的回归方法，将候选框位置以及候选框的类别当作回归问题来解决，如YOLO/SSD/DenseBox等。

   | 算法     | 主要组成                         | 备注 |
   | -------- | -------------------------------- | ---- |
   | YOLO     | Anchor boxes, YOLO-loss function |      |
   | SSD      |                                  |      |
   | DenseBox |                                  |      |

   

---

<!-- TOC -->

- [Basic knowledge in Deep Learning](#basic-knowledge-in-deep-learning)
  - [Metrics](#metrics)
    - [Confusion matrix](#confusion-matrix)
    - [IoU](#iou)
    - [mAP](#map)
  - [Non Maximum Suppression](#non-maximum-suppression)
  - [Dataset and splits](#dataset-and-splits)
    - [Datasets](#datasets)
    - [Splits and validation methods](#splits-and-validation-methods)
  - [Tips for coding](#tips-for-coding)
    - [Random seed for reproduction](#random-seed-for-reproduction)
- [Traditional computer vision based](#traditional-computer-vision-based)
  - [Hand detection using multiple proposals](#hand-detection-using-multiple-proposals)
- [DL-based Two-Stage Object Detection](#dl-based-two-stage-object-detection)
  - [R-CNN](#r-cnn)
  - [Fast R-CNN](#fast-r-cnn)
  - [Faster R-CNN](#faster-r-cnn)
- [DL-based Single-Shot Object Detection](#dl-based-single-shot-object-detection)
  - [YOLO](#yolo)
  - [YOLOv2](#yolov2)
  - [YOLOv3](#yolov3)
  - [YOLOv4](#yolov4)
  - [SSD](#ssd)
- [Face detection](#face-detection)
  - [Viola-Jones methods](#viola-jones-methods)
  - [MTCNN](#mtcnn)
- [Face alignment](#face-alignment)
  - [Regression-based](#regression-based)
  - [Template fitting](#template-fitting)
- [OHEM](#ohem)
- [Tools](#tools)
  - [Netron](#netron)

<!-- /TOC -->

---


<a id="markdown-basic-knowledge-in-deep-learning" name="basic-knowledge-in-deep-learning"></a>

## Basic knowledge in Deep Learning

<a id="markdown-metrics" name="metrics"></a>

### Metrics

> - [分类模型评估指标——准确率、精准率、召回率、F1、ROC曲线、AUC曲线](https://easyai.tech/ai-definition/accuracy-precision-recall-f1-roc-auc/)
> - [Let's evaluate classification model with ROC and PR curves](https://www.linkedin.com/pulse/lets-evaluate-classification-model-roc-pr-curves-suravi-mahanta/)

<a id="markdown-confusion-matrix" name="confusion-matrix"></a>

#### Confusion matrix

<center>


<img src="https://miro.medium.com/proxy/1*qbvg7ab-ZI8IETBb7ksjYg.png" alt="Confusion matrix" style="zoom:80%;" />

</center>

- *PR Curve:* Precision-vs-Recall graph. The higher it is, the better the model is. The AUC is Average Precision.
- *ROC Curve:* TPR-vs-FPR graph at different classification thresholds. AUC stands for "Area under the Curve". Model whose predictions are 100% correct has an AUC of 1.0. ROC curve **disregards sample imbalance**.

N.B. 

- We are often concerned with *Accuracy*,  *Precision* and *Recall*.
- *Sensitivity* is also called the *Recall*. 

> Coding:
>
> - *Python* [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)


<a id="markdown-iou" name="iou"></a>

#### IoU

The rate of intersection over union between the predicted bounding box and the ground truth bounding box. WHY? To measure how accurate is the object identified in the image and to decide whether to consider the object as a true positive or a false positive. A general threshold for IoU can be 0.5.


$$
IoU=\frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

> Coding:
>
> - *Takes bbox coords* [bbox_iou()](https://github.com/JinhangZhu/yolov3-from-scratch/blob/d7b82df4ff64c37fb309d6d311acba4896a9e571/util.py#L32)

<a id="markdown-map" name="map"></a>

#### mAP

**Average Precision (AP)** computes the average precision for recall rate over 0 to 1. The general definition for the AP is the AUC of PR curve. $AP=\int^1_0 p(r)dr$.

**Maximum precision**. To smooth the PR curve, the precision value at each recall level is replaced with the maximum precision value to the right of this recall level. $p_{interp}(r)=\underset{\hat{r}>r}{max}\,p(\hat{r})$.

- PASCAL VOC2008 calculated an average for the **11-point interpolated AP**. The recall values are sampled at 0, 0.1, 0.2, ..., 0.9 and 1.0 then the average of maximum precision values for the 11 recall values are computed. $AP=\frac{1}{11}\sum_{r\in {0, 0.1, ..., 1.0}}p_{interp}(r)$.
- For PASCAL VOC2010-2012, AP=**AUC after removing zigzags**: 

$$
AP=\sum_{r\in {r_1, r_2,..., r_N}}(r_{n+1}-r_{n})p_{interp}(r_{n+1})\\\\
p_{interp}(r_{n+1})=\underset{\hat{r}\geq r_{n+1}}{max}\,p(\hat{r})
$$

- COCO mAP used a **101-point interpolated AP**. AP is averaged over 10 IoU thresholds of **.50: .05: .95** and over all 80 categories. 

> - *Understand via an example* [mAP (mean Average Precision) for Object Detection](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)
> - *Official* [Detection evaluation](http://cocodataset.org/#detection-eval)
> - *GitHub Repo for eval* [Cartucho](https://github.com/Cartucho)/**[mAP](https://github.com/Cartucho/mAP)**

<a id="markdown-non-maximum-suppression" name="non-maximum-suppression"></a>

### Non Maximum Suppression

<img src="https://miro.medium.com/max/1400/1*CuqLjro26cHShpQVO1rgdQ.png" alt="Pseudo code of NMS" style="zoom: 67%;" class="center"/>

> - *YOLOv3 implementation* [NMS in YOLOv3](https://github.com/JinhangZhu/yolov3-from-scratch/blob/d7b82df4ff64c37fb309d6d311acba4896a9e571/util.py#L251)
> - *Explicit* [Non-maximum Suppression (NMS)](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c)

<a id="markdown-dataset-and-splits" name="dataset-and-splits"></a>

### Dataset and splits

<a id="markdown-datasets" name="datasets"></a>

#### Datasets

- **Training dataset**: Consisted of the samples of data used to fit the model. The model *learns* from the training set to tune weights and biases.
- **Validation dataset:** Consisted of the samples of data that provide an unbiased evaluation of the model that is fit on the training dataset in the process of *learning*. While tuning the parameters of the model, we use the validation dataset for frequent and regular evaluation and based on the *results of frequent evaluations* to modify the hyperparameters. Therefore, the effects of validation set on model parameters are *indirect*.
- **Test dataset:** Consisted of the samples of data that provide an unbiased evaluation of the *already learned* model. The test set is used to evaluate the level of competence of the learned model.

> - *Just reference*[About Train, Validation and Test Sets in Machine Learning](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)

<a id="markdown-splits-and-validation-methods" name="splits-and-validation-methods"></a>

#### Splits and validation methods

> - *Formulae* [【机器学习】Cross-Validation（交叉验证）详解](https://zhuanlan.zhihu.com/p/24825503)
> - *Including codes* [机器学习面试题集 - 详解四种交叉验证方法](https://www.jianshu.com/p/5b793f9b6481)
> - *Including cases* [训练集、验证集、测试集（附：分割方法+交叉验证）](https://easyai.tech/ai-definition/3dataset-and-cross-validation/)

<a id="markdown-tips-for-coding" name="tips-for-coding"></a>

### Tips for coding

<a id="markdown-random-seed-for-reproduction" name="random-seed-for-reproduction"></a>

#### Random seed for reproduction

> - [Pytorch随机种子](https://www.zdaiot.com/MLFrameworks/Pytorch/Pytorch%E9%9A%8F%E6%9C%BA%E7%A7%8D%E5%AD%90/)

<a id="markdown-traditional-computer-vision-based" name="traditional-computer-vision-based"></a>

## Traditional computer vision based

<a id="markdown-hand-detection-using-multiple-proposals" name="hand-detection-using-multiple-proposals"></a>

### Hand detection using multiple proposals

In general, this paper made two contributions in hand detection domain.

- The proposing of a two-stage hand detector.
- A large dataset of images with ground truth annotations for hands.



> - Review: [Review: Hand detection using multiple proposals](../tech/review-hand-detection-using-multiple-proposals)
> - Paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.301.3602&rep=rep1&type=pdf

<a id="markdown-dl-based-two-stage-object-detection" name="dl-based-two-stage-object-detection"></a>

## DL-based Two-Stage Object Detection

<a id="markdown-r-cnn" name="r-cnn"></a>

### R-CNN



<a id="markdown-fast-r-cnn" name="fast-r-cnn"></a>

### Fast R-CNN



<a id="markdown-faster-r-cnn" name="faster-r-cnn"></a>

### Faster R-CNN



<a id="markdown-dl-based-single-shot-object-detection" name="dl-based-single-shot-object-detection"></a>

## DL-based Single-Shot Object Detection

<a id="markdown-yolo" name="yolo"></a>

### YOLO

**Summary**:

- Unified prediction of bounding boxes.
- Network architecture.
- Design of the loss function.

> - Review: [Review: You only look once (YOLOv1)](../tech/review-yolo/)
> - Paper:  https://arxiv.org/abs/1506.02640

<a id="markdown-yolov2" name="yolov2"></a>

### YOLOv2

**Summary**:

The main contributions that this paper made in the improved YOLOv2 are:

- Improved the resolution of training images.
- Applied anchor boxes (from Faster R-CNN) to predict bounding boxes.
- Replaced the fully connected layer in the output layer in YOLO with a convolutional layer.

Another contribution is that they used a new dataset combination method and joint training algorithm to train a model on more than 9000 classes.

![Results on COCO test-dev2015. From single shot multibox detector](https://i.loli.net/2020/03/29/T3GFbEIBxswPRhl.png)

>- Review: [Review: YOLOv2](../tech/review-yolov2/)
>- Paper: https://arxiv.org/abs/1612.08242
>- Official implementation: https://pjreddie.com/darknet/yolov2/

<a id="markdown-yolov3" name="yolov3"></a>

### YOLOv3

**Abstract**:

We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate. It's still fast though, don't worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP 50 in 51 ms on a Titan X, compared to 57.5 AP 50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

![APs on COCO.png](https://i.loli.net/2020/04/27/OqriEcIvLodsU3j.png)

> - Review: [Review: YOLOv3](../tech/review-yolov3/)
> - [yolo系列之yolo v3【深度解析】](https://blog.csdn.net/leviopku/article/details/82660381)巩固细节
> - Paper: https://arxiv.org/abs/1804.02767
> - Official implementation: https://pjreddie.com/darknet/yolo/
> - PyTorch implementation: 
>
>   - https://github.com/ultralytics/yolov3 (easy to follow)
>
>   - [AlexeyAB / darknet](https://github.com/AlexeyAB/darknet) (requires higher level)(YOLOv4)
>
>   - [ayooshkathuria / YOLO_v3_tutorial_from_scratch](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch) ([from scratch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/))
>
>   - [DeNA](https://github.com/DeNA)/[PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3)(good [articles](https://medium.com/@hirotoschwert/reproducing-training-performance-of-yolov3-in-pytorch-part1-620140ad71d3))

<a id="markdown-yolov4" name="yolov4"></a>

### YOLOv4

![AP@.50:.95](https://user-images.githubusercontent.com/4096485/80213782-5f1e3480-8642-11ea-8fdf-0e6b9a6b5f4c.png)

<a id="markdown-ssd" name="ssd"></a>

### SSD

> - [目标检测|SSD原理与实现](https://zhuanlan.zhihu.com/p/33544892)

<a id="markdown-face-detection" name="face-detection"></a>

## Face detection

<a id="markdown-viola-jones-methods" name="viola-jones-methods"></a>

### Viola-Jones methods

- 级联的脸部检测器，使用Haar-like features和AdaBoost来训练分类器
- 有比较好的表现，real-time performance
- 在实际场景(larger visual variations of human faces)中degrade很快，即使使用了更加高级的features和分类器。

> Paper: [Robust real-time face detection](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1023/B:VISI.0000013087.49260.fb&casa_token=qBL1H0PEVr8AAAAA:NjS4bWwbIESf3eFkiC073RiFBu8kogd5jUBSx1oCJ090k2eRaSrArV6DBIBcsbV43ODwm3jQ3-t2wcM)

<a id="markdown-mtcnn" name="mtcnn"></a>

### MTCNN

> -  Review: [Review-MTCNN](../tech/review-mtcnn/)
> -  [open-face](https://github.com/open-face)/**[mtcnn](https://github.com/open-face/mtcnn)** 各种实现汇总
> -  **[MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)** 作者论文代码 | Caffe & Matlab
> -  [mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch) Python & PyTorch | 只有inference
> -  [facenet-pytorch](https://github.com/timesler/facenet-pytorch) Python & PyTorch | Advanced
> -  [mxnet_mtcnn_face_detection](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) Python & mxnet
> -  https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py 被经常调用的tf版

<a id="markdown-face-alignment" name="face-alignment"></a>

## Face alignment

<a id="markdown-regression-based" name="regression-based"></a>

### Regression-based



<a id="markdown-template-fitting" name="template-fitting"></a>

### Template fitting



<a id="markdown-ohem" name="ohem"></a>

## OHEM

> - Paper: [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/abs/1604.03540) 
> - [“hard-mining”, “hard examples”, … - Does “hard” mean anything specific in statistics when not applied to problem difficulty?](https://stats.stackexchange.com/questions/294349/hard-mining-hard-examples-does-hard-mean-anything-specific-in-stat)
> - [深度学习难分样本挖掘（Hard Mining）](https://zhuanlan.zhihu.com/p/51708428)- 知乎文章。
> - [深度学习不可忽略之OHEM:Online Hard Example Mining](https://zhuanlan.zhihu.com/p/59002127) - 知乎文章。

<a id="markdown-tools" name="tools"></a>

## Tools

> [史上最全神经网络结构图画图工具介绍，没有之一！](https://zhuanlan.zhihu.com/p/31920000)

<a id="markdown-netron" name="netron"></a>

### Netron

https://github.com/lutzroeder/netron，画神经网络结构图，可以采用不同文件类型的model。以PyTorch为例，使用Netron打开我们保存的三级网络的保存文件`.pkl`就画出来了。
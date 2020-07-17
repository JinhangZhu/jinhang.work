---
title: "R-CNN系列总结"
subtitle: ""
date: 2020-07-16T19:12:14+01:00
categories: [Tech,Reviews]
tags: [R-CNN, Object detection]
slug: "review-r-cnn"
toc: true
displayCopyright: false
dropCap: true
mermaid: false
comments: true
---

本文将总结目标检测中的重要经典R-CNN family算法，包含R-CNN，Fast R-CNN，Faster R-CNN以及Mask R-CNN。<!--more-->

## R-CNN

R-CNN ([Girshick et al., 2014](https://arxiv.org/abs/1311.2524))意为*Regions with CNN features*或者说*Region-based CNNs*，说明CNN是对region操作的。R-CNN是第一个将深度学习的CNN运用到目标检测中的，在之前虽然CNN（如AlexNet）已被运用到图像任务中，但一般是图像的分类，通过穷举窗口运用传统的SIFT，HoG算法来提取特征然后进行分类，最后采用NMS缩小范围。**R-CNN还遵循着传统目标检测的思路，首先使用selective search识别出定量的候选区域，再采用CNN来选择提取候选区域的特征用于分类**。R-CNN的亮点如下：

- CNN用于候选区域的特征提取；
- 当拥有训练标签的训练数据不足时，采用<mark>预训练(Pre-train)</mark>已知CNN模型以及<mark>精细化(Fine-tuning)</mark>。

<img src="https://i.loli.net/2020/07/17/tRNv7GuHwKCJqQU.png" title="Object detection system overview. Credit: Girshick et al., 2014">

### 模型原理

<img src="https://i.loli.net/2020/07/17/o4GUIxzFHdjuA5a.png" title="Credit: Analytics Vidhya">



## References

[Object Detection for Dummies Part 3: R-CNN Family](https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html)

[一文读懂目标检测：R-CNN、Fast R-CNN、等更多](https://zhuanlan.zhihu.com/p/40986674)

[RCNN系列（R-CNN、Fast-RCNN、Faster-RCNN、Mask-RCNN）](https://imlogm.github.io/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/rcnn/)
---
title: "My paper:FastSegFormer: A knowledge distillation-based method for real-time semantic segmentation of surface defects in navel oranges."
date: 2024-01-08T18:18:05+08:00
lastmod: 2023-01-09T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/my_paper/paper1.jpg"
description: "[Computers and Electronics in Agriculture 2024] FastSegFormer: A knowledge distillation-based method for real-time semantic segmentation of surface defects in navel oranges."
tags:
- Deep_learning
categories:
- 论文发表
series:
- 《My Original Paper》
comment : true
---

## FastSegFormer

Paper link:[https://www.sciencedirect.com/science/article/abs/pii/S0168169923009924?via%3Dihub](https://www.sciencedirect.com/science/article/abs/pii/S0168169923009924?via%3Dihub)

### Abstract

Navel oranges are valued citrus fruits with a strong market presence, and detecting defects is crucial in their sorting due to common diseases and abnormalities during growth and transport. Deep learning, particularly semantic segmentation, is revolutionizing the fruit sorting industry by overcoming the limitations of traditional defect detection and enhancing the accuracy of classifying complex defects in navel oranges. The FastSegFormer network, enabling real-time fruit defect detection, addresses this challenge with our introduced Multi-scale Pyramid (MSP) module for its architecture and a semi-resolution reconstruction branch post-feature fusion. We suggested a multi-resolution knowledge distillation strategy to further increase the network’s segmentation accuracy. We developed a navel orange defect segmentation dataset, trained, and evaluated our FastSegFormerE model, designed for memory-constrained devices. It outperforms ENet by 3.15%, achieving a mIoU of 88.78% on the test set. The FastSegFormer-P model, tailored for high-speed detection, was tested on the mid-range RTX3060 graphics card, surpassing ENet by 3.7% with a mIoU of 89.33% and reaching 108 frames/s. The results demonstrate that the FastSegFormer-E model attains enhanced detection accuracy with reduced memory usage, whereas the FastSegFormer-P model stands out by striking an optimal balance between top-tier detection accuracy and rapid processing speed. Deploying the algorithm system on the same platform as pipeline sorting, 20 frame/s was achieved on a Jetson Nano with very low computational power. The model significantly improves the detection of subtle and intricate edge defects, achieving real-time speeds. Our proposed algorithm enhances the fineness of fruit sorting, resolves the limitation of existing algorithms that apply to a narrow range of fruit sorting scenarios, and provides an efficient and accurate solution for large-scale navel orange defect detection.

### Conclusion

In this paper, we developed two segmentation models called FastSegFormer-E and FastSegFormer-P to quickly identify defects in big batches of navel oranges. To rebuild image detail for the deep network, we created the MSP module and added a semi-resolution image reconstruction branch following feature fusion. Our models are effective in identifying lesser defects and precisely segmenting complex edge defects in real-world complex settings. The segmentation accuracy of the model was further enhanced by the suggested multi-resolution knowledge distillation strategy without increasing model size and inference time. The proposed FastSegFormer-E achieves superior defect detection accuracy while maintaining low memory consumption, while the proposed FastSegFormer-P achieves the highest defect detection accuracy with high inference speed. The FastSegFormer-P model achieves a detection speed of 20 fps on a very low computing power device under the Jetson platform, suggesting that deploying the algorithm is very easy to achieve real-time detection. The proposed algorithm effectively overcomes the limitations of current commonly used methods in meeting the demands of precise sorting. Additionally, it successfully addresses the crucial requirement of real-time detection, an aspect where many existing segmentation algorithms for fruit defect detection fall short. Incorporating a smaller and faster backbone into the proposed network will enhance its ability to handle image inputs of various resolutions. Defect detection will be performed at the harvesting stage in the future, where the algorithm’s adaptation to partial leaf occlusion becomes crucial.

### Data availability

The data and code can be available in [https://github.com/caixiongjiang/FastSegFormer](https://github.com/caixiongjiang/FastSegFormer). Code on edge computing device deployment and detection systems on PCs is available in [https://github.com/caixiongjiang/FastSegFormer-pyqt](https://github.com/caixiongjiang/FastSegFormer-pyqt).
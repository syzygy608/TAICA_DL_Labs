# TAICA Deep Learning Lab 2 Report

## Binary Semantic Segmentation

### 1. Implementation Details (30%)

#### Unet
Both Unet model and resnet34+unet model need a convolution block. The convolution block is a combination of two 3x3 convolution layers with batch normalization and ReLU activation function. The first convolution layer has 64 filters and the second convolution layer has 128 filters. The convolution block is used in the encoder part of the Unet model and the resnet34+unet model.

#### Model Training

I learned dropout and L2 regularization from [here](https://www.bilibili.com/video/BV1RqXRYDEe2/?share_source=copy_web&vd_source=8eb0208b6e349b456c095c16067fb3af). I added dropout to the conv block and L2 regularization to calculate the loss. I also added a learning rate scheduler to adjust the learning rate during training. I used the Adam optimizer to optimize the model.

### 2. Data Preprocessing (25%)

First, we need to make our training data in diverse forms. We can use data augmentation to increase the diversity of the training data. We also need to ensure the image and mask are aligned. To do this, I use albumentations to augment the data. I use the following augmentations:


### 3. Analyze the experiment results (25%)

### 4. Execution steps (0%)

### 5. Discussion (20%)

### 6. Reference

1. 【调教神经网络咋这么难？【白话DeepSeek03】】 https://www.bilibili.com/video/BV1RqXRYDEe2/?share_source=copy_web&vd_source=8eb0208b6e349b456c095c16067fb3af
2. V-Net https://arxiv.org/abs/1606.04797
3. Unet https://arxiv.org/abs/1505.04597
4. Resnet https://arxiv.org/abs/1512.03385

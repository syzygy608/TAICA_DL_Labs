# TAICA Deep Learning Lab 2 Report

## 0. Environment

- CPU: Intel(R) Core(TM) i7-14700
- GPU: NVIDIA GeForce RTX 4090
- OS: Ubuntu 22.04

## 1. Implementation Details (30%)

### Unet
Both Unet model and resnet34+unet model need a convolution block. The convolution block is a combination of two 3x3 convolution layers with ReLU activation function. The convolution block is used in the encoder part of the Unet model and the resnet34+unet model.

The first question of the implementation to UNet is the up-covolution. The paper said that upsampling of the feature map followed by a 2x2 convolution ( up-convolution ) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. But it is not clear how to implement the up-convolution. I found that the `nn.ConvTranspose2d` can be used to implement the up-convolution. The `nn.ConvTranspose2d` can upsample the input tensor and then apply a convolution operation. The `nn.ConvTranspose2d` has a parameter `stride` which can be used to control the upsample rate. The `nn.ConvTranspose2d` also has a parameter `output_padding` which can be used to control the output size. The `nn.ConvTranspose2d` can be used to implement the up-convolution in the Unet model.

The original U-Net uses unpadded 3×3 convolutions, causing the output (e.g., 388×388) to be smaller than the input (e.g., 572×572), which mismatches the ground truth size (e.g., 1024×1024) and complicates training. I addressed this by switching to padded convolutions (padding=1), ensuring the output matches the input size (e.g., 1024×1024). This practical modification eliminates preprocessing needs, enhancing U-Net’s usability while preserving its core functionality.

After initial training, I observed suboptimal results, likely due to overfitting and unstable feature learning. To address this, I modified the U-Net by adding Dropout (p=0.5) to the bottleneck layer to regularize the network and reduce overfitting. Additionally, I introduced BatchNorm after each convolution to normalize activations, improving training stability and convergence. These common techniques significantly enhanced the model’s segmentation performance.

### Resnet34+Unet

### Model Training

I opted for Adam over SGD to leverage its faster convergence and robustness, addressing SGD’s limitations in learning rate tuning and gradient adaptability. The training method in the U-Net paper, designed for multi-class segmentation with weighted cross-entropy, does not align with this assignment’s binary foreground-background separation task evaluated using Dice score. For this two-class problem, combining BCE with Dice Loss is more suitable, as it directly optimizes for overlap metrics like Dice, improving performance over the paper’s approach.
I learned dropout and L2 regularization from [here](https://www.bilibili.com/video/BV1RqXRYDEe2/?share_source=copy_web&vd_source=8eb0208b6e349b456c095c16067fb3af). I use L2 regularization to calculate the loss to prevent overfitting.

## 2. Data Preprocessing (25%)




## 3. Analyze the experiment results (25%)

## 4. Execution steps (0%)

## 5. Discussion (20%)

### Unet
I find that learning rate for 0.001 and weight decay for 0.0001 is a good choice. The model can converge quickly and the loss is low. The model can achieve a good performance on the validation after 10 epochs with batch size 12.

## 6. Reference

1. 【调教神经网络咋这么难？【白话DeepSeek03】】 https://www.bilibili.com/video/BV1RqXRYDEe2/?share_source=copy_web&vd_source=8eb0208b6e349b456c095c16067fb3af
2. V-Net https://arxiv.org/abs/1606.04797
3. Unet https://arxiv.org/abs/1505.04597
4. Resnet https://arxiv.org/abs/1512.03385

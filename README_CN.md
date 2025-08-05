
# SwinSpikeFormer: Learning Comprehensive Spatial-Temporal Representation to Reconstruct Dynamic Scenes from Spike Streams

欢迎来到 **SwinSF** 项目！该仓库包含 **Swin Spikeformer (SwinSF)** 模型的实现，正如论文 ["SwinSpikeFormer: Learning Comprehensive Spatial-Temporal Representation to Reconstruct Dynamic Scenes from Spike Streams"](https://ieeexplore.ieee.org/abstract/document/11028676/) 中所描述的那样。SwinSF 旨在从脉冲相机生成的脉冲流中重建高质量图像，这对于运动模糊是一个挑战的高速成像尤为有用。

## 目录

- [开始](#开始)
  - [先决条件](#先决条件)
  - [数据集](#数据集)
- [训练](#训练)
- [测试](#测试)
- [模型权重](#模型权重)
- [引用](#引用)

## 开始

### 先决条件

在开始之前，请确保您满足以下要求：
- Python 3.6.13
- PyTorch 1.10.0+cu113
- `requirements.txt` 中列出的其他依赖项

您可以使用 pip 安装所需的软件包：

```bash
pip install -r requirements.txt
```

### 数据集

要开始 SwinSF 项目，您需要从 [百度网盘](https://pan.baidu.com/s/1N6tMru-fn5iJ0oyygHg1hQ?pwd=cps6) 下载数据集到 datasets 目录。

- **spike-reds**：一个带有真实值的模拟数据集，250x400 像素，来自论文 [Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Spk2ImgNet_Learning_To_Reconstruct_Dynamic_Scene_From_Continuous_Spike_Stream_CVPR_2021_paper.pdf)。
- **spike-X4K**：为本项目创建的带有真实值的模拟数据集，1000x1000 像素。
- **spike-classA**：由北京大学团队的脉冲相机捕获的数据集，250x400 像素，没有真实值。

请注意，reds 和 classA 数据集不是由我们团队创建的，请联系原作者下载。

## 训练

要在 reds 数据集上训练 SwinSF 模型，请使用以下命令：

```bash
python train.py --data_mode 250 --dataset_path ./datasets/spike_reds --device cuda:0
```

对于 X4K 数据集，使用：

```bash
python train.py --data_mode 1000 --dataset_path ./datasets/spike_x4k --device cuda:0
```

要使用多 GPU 训练，请添加 --device_ids 01 选项（确保 device_ids 中的第一个设备与 --device 设置匹配）。对于 CPU 训练，只需使用 --device cpu。

训练权重将自动保存到 checkpoint 目录。

## 测试

要测试模型并保存 reds 数据集的重建图像，请使用：

```bash
python test.py --data_mode 250 --dataset_path ./datasets/spike_reds --device cuda:0 --load_model /path/to/training/parameters --save_image True --save_path /path/to/save/images
```

对于 X4K 数据集：

```bash
python test.py --data_mode 1000 --dataset_path ./datasets/spike_x4k --device cuda:0 --load_model /path/to/training/parameters --save_image True --save_path /path/to/save/images
```

对于 classA 数据集：

```bash
python test.py --data_mode 250 --dataset_path ./datasets/classA --device cuda:0 --load_model /path/to/training/parameters --save_image True --save_path /path/to/save/images
```

## 模型权重

我们提供了两个分辨率的预训练权重，可以从 [百度网盘](https://pan.baidu.com/s/1Rkwz0bbie5kumZykkJMtyg?pwd=x7z8)下载，提取码为x7z8。
## 引用

如果您发现我们的工作对您的研究有用，请考虑引用我们的论文：

```
@INPROCEEDINGS {11028676,
author = { Jiang, Liangyan and Zhu, Chuang and Chen, Yanxu },
booktitle = { 2024 International Conference on Virtual Reality and Visualization (ICVRV) },
title = {{ SwinSpikeFormer: Learning Comprehensive Spatial-Temporal Representation to Reconstruct Dynamic Scenes from Spike Streams }},
year = {2024},
volume = {},
ISSN = {},
pages = {60-65},
abstract = { The spike camera, with its high temporal resolution, low latency, and high dynamic range, addresses high-speed imaging challenges such as motion blur. It captures photons at each pixel independently, creating binary spike streams rich in temporal information. However, these binary spike streams present a challenge for their reconstruction into images, which is necessary for performing subsequent downstream tasks. Current algorithms, both traditional and deep learning-based, fail in utilizing this rich temporal information and restoring the details of the reconstructed image. To tackle this challenge, we propose Swin Spikeformer (SwinSF), a novel model for dynamic scene reconstruction from spike streams. SwinSF is composed of three main modules: Spike Feature Extraction, Spatial-Temporal Feature Extraction, and Final Reconstruction Module. The Spike Feature Extraction Module uses convolution layers to directly extract features from the spike streams. The Spatial-Temporal Feature Extraction Module includes several Residual Swin Spikeformer Blocks (RSSB), which integrate shifted window self-attention and proposed Temporal Spike Attention (TSA) to effectively extract both spatial and temporal information from intra-frames and inter-frames, leading to a more robust and accurate reconstruction of spike streams. The Final Reconstruction Module fuses both spike and spatial-temporal features to produce the final image reconstruction. Furthermore, we have built a new synthesized dataset for spike image reconstruction, featuring high resolution and enhanced temporal fidelity through the use of frame interpolation techniques, which ensures its relevance and applicability to realistic spike camera imaging scenarios. Experimental results demonstrate that our proposed network, SwinSF, sets a new benchmark, achieving state-of-the-art performance across a series of datasets, including both real-world and synthesized data across various resolutions. Our codes and proposed dataset can be found at https://github.com/bupt-ai-cz/SwinSF. },
keywords = {Solid modeling;Dynamics;Virtual reality;Feature extraction;Cameras;Transformers;Real-time systems;Spatial resolution;Image reconstruction;Streams},
doi = {10.1109/ICVRV62410.2024.00020},
url = {https://doi.ieeecomputersociety.org/10.1109/ICVRV62410.2024.00020},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =Dec}

```

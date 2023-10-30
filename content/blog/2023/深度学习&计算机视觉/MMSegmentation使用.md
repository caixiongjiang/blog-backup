---
title: "MMSegmentation使用"
date: 2023-08-16T18:18:05+08:00
lastmod: 2023-08-16T09:19:06+08:00
draft: false
featured_image: "https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mmcv/mmseg_title.jpg"
description: "用于快速训练现成算法的语义分割算法库，在论文中进行模型对比时简单高效。"
tags:
- Deep_learning
categories:
- 深度学习
series:
- 《MMCV系列》
comment : true
---

## MMSegmentation教程

### 分割模型预测

#### 语义分割模型预测-命令行模式（CLI）

* 进入mmsegmentaion的目录：
```shell
cd mmsegmentation
```
* 准备素材（模型配置文件，模型权重文件，素材图片）
```shell
$ mkdir config
$ mkdir checkpoints
$ mkdir test_imgs
$ mkdir test_results
```

* 进行预测
```shell
# 以PSPNet为例
python demo/image_demo.py \
        data/street_uk.jpeg \
        configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py \
        https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
        --out-file outputs/B1_uk_pspnet.jpg \
        --device cuda:0 \
        --opacity 0.5
```

#### 语义分割模型预测-Python API

对于批量图片的预测，或者需要对图像进行后处理，我推荐使用`Python API`的方式。

* 进入mmsegmentation目录：
```shell
cd mmsegmentation
```
* 编写脚本：
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import mmcv
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmseg.utils import register_all_modules
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.datasets import cityscapes 

# 注册所有模型
register_all_modules()

# 读取图像
img_path = 'data/street_uk.jpeg'
img_pil = Image.open(img_path)
# 定义模型配置文件
config_file = 'configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py'
# 定义模型文件
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221202_141901-dc2c2ddd.pth'
# 初始化模型并放入0号GPU
model = init_model(config_file, checkpoint_file, device='cuda:0')
# 进行语义分割预测
if not torch.cuda.is_available():
    model = revert_sync_batchnorm(model)
result = inference_model(model, img_path)
print(result.keys()) # ['pred_sem_seg', 'seg_logits']

# result结果解析
np.unique(result.pred_sem_seg.data.cpu()) # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 13, 15]) 不同的数字代表不同的类别
print(result.pred_sem_seg.data.shape) # 等于原图大小：torch.Size([1, 1500, 2250])
# 得到语义分割的结果
class_map = result.pred_sem_seg.data[0].detach().cpu().numpy()

# 结果可视化
from mmseg.datasets import cityscapes
import mmcv 

# 获取类别名和调色板
classes = cityscapes.CityscapesDataset.METAINFO['classes']
palette = cityscapes.CityscapesDataset.METAINFO['palette']
opacity = 0.15 # 透明度，越大越接近原图

# 将分割图按调色板染色
seg_map = class_map.astype('uint8')
seg_img = Image.fromarray(seg_map).convert('P')
seg_img.putpalette(np.array(palette, dtype=np.uint8))

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.figure(figsize=(14, 8))
im = plt.imshow(((np.array(seg_img.convert('RGB')))*(1-opacity) + mmcv.imread(img_path)*opacity) / 255)

# 为每一种颜色创建一个图例
patches = [mpatches.Patch(color=np.array(palette[i])/255., label=classes[i]) for i in range(8)]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

plt.show()
```

最后图片可视化之后：
![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/mmcv/img1.jpg)

**对于CLI方式进行视频的分割，其实和图像是一样的！命令行方式只需要换成`demo/video_demo.py`,并将素材图片换成素材视频。**
**对于Python API方式进行视频分割，就是将上述过程封装成单帧处理逻辑，然后使用cv2视频帧播放的模版，将分割的逻辑放入其中就好！**

#### 训练自己的数据集

* 准备自己的数据集

这部分一般使用Labelme标注数据集，为了节省体力，也可以使用SAM的应用来标注数据集，推荐一个我自己使用过的[ISAT_with_segment_anything](https://github.com/yatengLG/ISAT_with_segment_anything)。

标注好的数据集转换为标签图片一般为背景为黑色，其他类别对应一种RGB颜色。但为了适应语义分割的数据显示，我们要将不同的类别的RGB像素（比如[0,255,0]）变为一个灰度值（比如为1）。最后放入网络的时候标签图一般是纯黑色（其实不是纯黑，但肉眼看不出来）的灰度图，**需要注意的是每个通道值都是相同的，因为png格式自动保存为三通道**。

**注意原图是`.jpg`图片格式，标签图片是`.png`格式！**

数据集的结构如下：
```scss
📂 数据集名称/
|--📂 img_dir/
|  |--- train/
|  |--- val/
|--📂 ann_dir/
   |--- train/
   |--- val/
```

* 数据集批量可视化

数据集可视化其实就是将灰度图转换为你需要的RGB图，一个类别对应一个RGB，需要准备好自己的调色板字典。一个数值对应一个RGB颜色。

批量可视化数据集的模版：
```python
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

jpg_folder = ""
png_folder = ""
vis_folder = ""

demo_png_path = ""

demo_mask = cv2.imread("demo_png_path")
# 得到灰度图的类别
class_array = np.unique(demo_mask)

palette = [
    ['background', [127,127,127]],
    ['red', [0,0,200]],
    ['green', [0,200,0]],
    ['white', [144,238,144]],
    ['seed-black', [30,30,30]],
    ['seed-white', [8,189,251]]
] 
# 这里的idx的顺序就是class_array中对应的数值
palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]


jpgs = os.listdir(jpg_folder)
for jpg in jpgs:
    png_name = f'{jpg.split('.')[0]}.png'
    img = cv2.imread(os.path.join(jpg_folder, jpg))
    png = cv2.imread(os.path.join(png_folder, png_name))
    mask = png[:, :, 0]

    viz_mask_bgr = np.zeros((mask.shape[0], mask.shape[1], 3))
    for idx in palette_dict.keys():
        viz_mask_bgr[np.where(mask==idx)] = palette_dict[idx]   
        viz_mask_bgr = viz_mask_bgr.astype('uint8')
    
    # 将语义分割标注图和原图叠加显示
    opacity = 0.2 # 透明度越大，可视化效果越接近原图
    label_viz = cv2.addWeighted(img, opacity, viz_mask_bgr, 1-opacity, 0)
    # 保存图片
    cv2.imwrite(os.path.join(vis_folder, "png_name"), label_viz)

```

* 准备数据集配置文件

在`MMSegmentation`中的datasets文件夹下可以看到其支持各种各样的数据集格式。它们都是遵循一个统一的模版，可以照猫画虎来准备自己的数据集，这里给的是b站同济子豪兄的模版：
```python
# 同济子豪兄 2023-6-25
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class ZihaoDataset(BaseSegDataset):
    # 类别和对应的 RGB配色
    METAINFO = {
        'classes':['background', 'red', 'green', 'white', 'seed-black', 'seed-white'],
        'palette':[[127,127,127], [200,0,0], [0,200,0], [144,238,144], [30,30,30], [251,189,8]]
    }
    
    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 seg_map_suffix='.png',   # 标注mask图像的格式
                 reduce_zero_label=False, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
```

* 注册数据集类

该部分其实就是将你自己定义好的数据集配置文件通过`__init__.py`文件放入datasets文件夹的索引中。具体要修改的地方就只有两个，`导入py文件`,`在列表中加入`（以上面的数据集配置文件为例）：

```python
# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
from .ade import ADE20KDataset
from .basesegdataset import BaseSegDataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import MultiImageMixDataset
from .decathlon import DecathlonDataset
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .isaid import iSAIDDataset
from .isprs import ISPRSDataset
from .lip import LIPDataset
from .loveda import LoveDADataset
from .night_driving import NightDrivingDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .potsdam import PotsdamDataset
from .stare import STAREDataset
from .synapse import SynapseDataset
from .ZihaoDataset import ZihaoDataset # 新增
# yapf: disable
from .transforms import (CLAHE, AdjustGamma, BioMedical3DPad,
                         BioMedical3DRandomCrop, BioMedical3DRandomFlip,
                         BioMedicalGaussianBlur, BioMedicalGaussianNoise,
                         BioMedicalRandomGamma, GenerateEdge, LoadAnnotations,
                         LoadBiomedicalAnnotation, LoadBiomedicalData,
                         LoadBiomedicalImageFromFile, LoadImageFromNDArray,
                         PackSegInputs, PhotoMetricDistortion, RandomCrop,
                         RandomCutOut, RandomMosaic, RandomRotate,
                         RandomRotFlip, Rerange, ResizeShortestEdge,
                         ResizeToMultiple, RGB2Gray, SegRescale)
from .voc import PascalVOCDataset

# yapf: enable
__all__ = [
    'BaseSegDataset', 'BioMedical3DRandomCrop', 'BioMedical3DRandomFlip',
    'CityscapesDataset', 'PascalVOCDataset', 'ADE20KDataset',
    'PascalContextDataset', 'PascalContextDataset59', 'ChaseDB1Dataset',
    'DRIVEDataset', 'HRFDataset', 'STAREDataset', 'DarkZurichDataset',
    'NightDrivingDataset', 'COCOStuffDataset', 'LoveDADataset',
    'MultiImageMixDataset', 'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset',
    'LoadAnnotations', 'RandomCrop', 'SegRescale', 'PhotoMetricDistortion',
    'RandomRotate', 'AdjustGamma', 'CLAHE', 'Rerange', 'RGB2Gray',
    'RandomCutOut', 'RandomMosaic', 'PackSegInputs', 'ResizeToMultiple',
    'LoadImageFromNDArray', 'LoadBiomedicalImageFromFile',
    'LoadBiomedicalAnnotation', 'LoadBiomedicalData', 'GenerateEdge',
    'DecathlonDataset', 'LIPDataset', 'ResizeShortestEdge',
    'BioMedicalGaussianNoise', 'BioMedicalGaussianBlur',
    'BioMedicalRandomGamma', 'BioMedical3DPad', 'RandomRotFlip',
    'SynapseDataset', 'ZihaoDataset' # 新增
]
```

* 准备pipeline配置文件：

这个配置文件包含了这些参数的配置：
```scss
1.数据集路径
2.预处理
3.后处理
4.DataLoader
5.测试集评估指标
```

文件的路径为：`configs/_base_/datasets/`下

给一个样例文件：
```python
# 数据处理 pipeline
# 同济子豪兄 2023-6-28

# 数据集路径
dataset_type = 'ZihaoDataset' # 数据集类名
data_root = 'Watermelon87_Semantic_Seg_Mask/' # 数据集路径（相对于mmsegmentation主目录）

# 输入模型的图像裁剪尺寸，一般是 128 的倍数，越小显存开销越少
crop_size = (512, 512)

# 训练预处理
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 测试预处理
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# TTA后处理 （在预测的时候把图像缩放成不同尺寸输入模型，然后将结果加权得到最终的结果）
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# 训练 Dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # 指定训练集的图像路径和标注路径
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

# 验证 Dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

# 测试 Dataloader (因为图片较少没有测试的情况，那么就将验证集作为测试集)
test_dataloader = val_dataloader

# 验证 Evaluator 
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

# 测试 Evaluator
test_evaluator = val_evaluator
```

* 载入模型配置文件(这里使用UNet为例)：
```python
from mmengine import Config
# 选择的模型文件具有不同的骨干网络等，参考官方代码库的README
cfg = Config.fromfile('./configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py')
dataset_cfg = Config.fromfile('./configs/_base_/datasets/ZihaoDataset_pipeline.py')
cfg.merge_from_dict(dataset_cfg) # 进行配置信息字典融合

# 修改config文件的配置信息
NUM_CLASS = 6
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.data_preprocessor.test_cfg = dict(size_divisor=128)

# 单卡训练时，需要把 SyncBN 改成 BN
cfg.norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

# 模型 decode/auxiliary 输出头，指定为类别个数
cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head.num_classes = NUM_CLASS

# 训练 Batch Size
cfg.train_dataloader.batch_size = 4

# 结果保存目录
cfg.work_dir = './work_dirs/ZihaoDataset-UNet'

# 模型保存与日志记录
cfg.train_cfg.max_iters = 40000 # 训练迭代次数
cfg.train_cfg.val_interval = 500 # 评估模型间隔
cfg.default_hooks.logger.interval = 100 # 日志记录间隔
cfg.default_hooks.checkpoint.interval = 2500 # 模型权重保存间隔
cfg.default_hooks.checkpoint.max_keep_ckpts = 1 # 最多保留几个模型权重
cfg.default_hooks.checkpoint.save_best = 'mIoU' # 保留指标最高的模型权重

# 随机数种子
cfg['randomness'] = dict(seed=0)

# 保存最终的配置文件
cfg.dump('Zihao-Configs/ZihaoDataset_UNet_20230712.py') 
```

* 按照最终配置文件进行训练
```shell
$ cd mmsegmentation
$ python tools/train.py Zihao-Configs/ZihaoDataset_UNet_20230712.py
```

* 训练结果（日志，模型权重文件）保存在config中指定的`work_dirs`目录下。

#### 训练结果可视化

* 设置Matplotlib中文字体:
```python
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random

matplotlib.rc("font",family='SimHei') # 中文字体
```

* 载入训练日志：
进入`MMSegmentation`目录，运行脚本：
```python
import pandas as pd
import matplotlib.pyplot as plt

# 日志文件路径
log_path = './work_dirs/ZihaoDataset-UNet/20230818_210528/vis_data/scalars.json'

with open(log_path, "r") as f:
    json_list = f.readlines()

# 将数据放入Pandas的DataFrame中
df_train = pd.DataFrame()
df_test = pd.DataFrame()
for each in json_list[:-1]:
    if 'aAcc' in each:
        df_test = df_test._append(eval(each), ignore_index=True)
    else:
        df_train = df_train._append(eval(each), ignore_index=True)

# 导出训练日志表格
df_train.to_csv('图表/训练日志-训练集.csv', index=False)
df_test.to_csv('图表/训练日志-测试集.csv', index=False)


random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
linestyle = ['--', '-.', '-']

def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg

# 绘制训练集损失曲线
metrics = ['loss', 'decode.loss_ce', 'aux.loss_ce']
plt.figure(figsize=(16, 8))

x = df_train['step']
for y in metrics:
    try:
        plt.plot(x, df_train[y], label=y, **get_line_arg())
    except:
        pass

plt.tick_params(labelsize=20)
plt.xlabel('step', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title('训练集损失函数', fontsize=25)

plt.legend(fontsize=20)

plt.savefig('图表/训练集损失函数.pdf', dpi=120, bbox_inches='tight')

plt.show()

# 绘制训练集准确率曲线
metrics = ['decode.acc_seg', 'aux.acc_seg']

plt.figure(figsize=(16, 8))

x = df_train['step']
for y in metrics:
    try:
        plt.plot(x, df_train[y], label=y, **get_line_arg())
    except:
        pass

plt.tick_params(labelsize=20)
plt.xlabel('step', fontsize=20)
plt.ylabel('Metrics', fontsize=20)
plt.title('训练集准确率', fontsize=25)

plt.legend(fontsize=20)

plt.savefig('图表/训练集准确率.pdf', dpi=120, bbox_inches='tight')

plt.show()

# 测试集评估指标折线
print(df_test.columns)

metrics = ['aAcc', 'mIoU', 'mAcc', 'mDice', 'mFscore', 'mPrecision', 'mRecall']

plt.figure(figsize=(16, 8))

x = df_test['step']
for y in metrics:
    try:
        plt.plot(x, df_test[y], label=y, **get_line_arg())
    except:
        pass

plt.tick_params(labelsize=20)
plt.ylim([0, 100])
plt.xlabel('step', fontsize=20)
plt.ylabel('Metrics', fontsize=20)
plt.title('测试集评估指标', fontsize=25)

plt.legend(fontsize=20)

plt.savefig('图表/测试集分类评估指标.pdf', dpi=120, bbox_inches='tight')

plt.show()

```

* 训练过程训练集各类别评估指标

```python
from matplotlib import colors as mcolors
import random
random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
linestyle = ['--', '-.', '-']

def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg

import pandas as pd
import matplotlib.pyplot as plt

# 类别列表
class_list = ['background', 'red', 'green', 'white', 'seed-black', 'seed-white']
# 日志文件路径
log_path = './work_dirs/ZihaoDataset-PSPNet/20230818_210528/20230818_210528.log'

with open(log_path, 'r') as f:
    logs = f.read()

## 定义正则表达式
import re
import numpy as np

def transform_table_line(raw):
    raw = list(map(lambda x: x.split('|'), raw))
    raw = list(map(
      lambda row: list(map(
          lambda col: float(col.strip()),
          row
      )),
      raw
    ))
    return raw

x = range(500, 40500, 500)

metrics_json = {}
for each_class in class_list: # 遍历每个类别
    re_pattern = r'\s+{}.*?\|(.*)?\|'.format(each_class) # 定义该类别的正则表达式
    metrics_json[each_class] = {}
    metrics_json[each_class]['re_pattern'] = re.compile(re_pattern)

# 匹配
for each_class in class_list: # 遍历每个类别
    find_string = re.findall(metrics_json[each_class]['re_pattern'], logs) # 粗匹配
    find_string = transform_table_line(find_string) # 精匹配
    metrics_json[each_class]['metrics'] = find_string

print(metrics_json.keys())
# dict_keys(['background', 'red', 'green', 'white', 'seed-black', 'seed-white'])

# 查看某一类别的评估指标 IoU Acc Dice Fscore Precision Recall
each_class = 'red'
each_class_metrics = np.array(metrics_json[each_class]['metrics'])

plt.figure(figsize=(16, 8))

for idx, each_metric in enumerate(['IoU', 'Acc', 'Dice', 'Fscore', 'Precision', 'Recall']):

    try:
        plt.plot(x, each_class_metrics[:,idx], label=each_metric, **get_line_arg())
    except:
        pass

plt.tick_params(labelsize=20)
plt.ylim([0, 100])
plt.xlabel('step', fontsize=20)
plt.ylabel('Metrics', fontsize=20)
plt.title('类别 {} 训练过程中，在测试集上的评估指标'.format(each_class), fontsize=25)

plt.legend(fontsize=20)

plt.savefig('图表/类别 {} 训练过程评估指标.pdf'.format(each_class), dpi=120, bbox_inches='tight')

plt.show()

# 注意x的元素个数，应和metrics_json[each_class]['metrics']元素个数一致，绘图才能成功

# 查看每个类别的评估指标 IoU Acc Dice Fscore Precision Recall
for each_class in class_list: # 遍历每个类别
    each_class_metrics = np.array(metrics_json[each_class]['metrics'])
    
    plt.figure(figsize=(16, 8))

    for idx, each_metric in enumerate(['IoU', 'Acc', 'Dice', 'Fscore', 'Precision', 'Recall']):

        try:
            plt.plot(x, each_class_metrics[:,idx], label=each_metric, **get_line_arg())
        except:
            pass

    plt.tick_params(labelsize=20)
    plt.ylim([0, 100])
    plt.xlabel('step', fontsize=20)
    plt.ylabel('Metrics', fontsize=20)
    plt.title('图表/类别 {} 训练过程中，在测试集上的评估指标'.format(each_class), fontsize=25)

    plt.legend(fontsize=20)

    # plt.savefig('类别 {} 训练过程评估指标.pdf'.format(each_class), dpi=120, bbox_inches='tight')

    plt.show()


```

* 测试集性能评估，采用命令行的方式进行：
```shell
python tools/test.py Zihao-Configs/ZihaoDataset_UNet_20230818.py ./work_dirs/ZihaoDataset-PSPNet/iter_40000.pth
```

* 获取参数量、计算量：
```python
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} --shape ${INPUT_SHAPE}
```
*注意运行上述命令时，需要先修改get-flops.py的87行为` outputs = get_model_complexity_info(model,inputs=data['inputs'],show_table=False,show_arch=False)`*

* 速度指标-FPS（至少需要200张图片）
```shell
python tools/analysis_tools/benchmark.py Zihao-Configs/ZihaoDataset_PSPNet_20230818.py ./work_dirs/ZihaoDataset-PSPNet/iter_40000.pth
```

#### MMSegmentation推理预测

```python
import numpy as np
import matplotlib.pyplot as plt

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2


# 模型 config 配置文件
config_file = 'Zihao-Configs/ZihaoDataset_KNet_20230818.py'

# 模型 checkpoint 权重文件
checkpoint_file = 'checkpoint/Zihao_KNet.pth'

# device = 'cpu'
device = 'cuda:0'

model = init_model(config_file, checkpoint_file, device=device)
# 载入测试集图像
img_path = 'Watermelon87_Semantic_Seg_Mask/img_dir/val/01bd15599c606aa801201794e1fa30.jpg'

img_bgr = cv2.imread(img_path)
result = inference_model(model, img_bgr)
print(result.keys()) # ['seg_logits', 'pred_sem_seg']

pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
print(np.unique(pred_mask)) # array([0, 1, 2, 3, 4, 5])

# 可视化语义分割预测结果-（和原图并排显示）
plt.figure(figsize=(14, 8))

plt.subplot(1,2,1)
plt.imshow(img_bgr[:,:,::-1])
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img_bgr[:,:,::-1])
plt.imshow(pred_mask, alpha=0.6) # alpha 高亮区域透明度，越小越接近原图
plt.axis('off')
plt.savefig('outputs/K1-2.jpg')
plt.show()

# 加上图例的方法
from mmseg.datasets import ZihaoDataset
import numpy as np
import mmcv 
from PIL import Image

# 获取类别名和调色板
classes = ZihaoDataset.METAINFO['classes']
palette = ZihaoDataset.METAINFO['palette']
opacity = 0.15 # 透明度，越大越接近原图

# 将分割图按调色板染色
# seg_map = result[0].astype('uint8')
seg_map = pred_mask.astype('uint8')
seg_img = Image.fromarray(seg_map).convert('P')
seg_img.putpalette(np.array(palette, dtype=np.uint8))

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.figure(figsize=(14, 8))
img_plot = ((np.array(seg_img.convert('RGB')))*(1-opacity) + mmcv.imread(img_path)*opacity) / 255
im = plt.imshow(img_plot)

# 为每一种颜色创建一个图例
patches = [mpatches.Patch(color=np.array(palette[i])/255., label=classes[i]) for i in range(len(classes))]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

plt.savefig('outputs/K1-6.jpg')
plt.show()
```

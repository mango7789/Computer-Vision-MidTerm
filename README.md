## 任务一：微调在ImageNet上预训练的卷积神经网络实现鸟类识别

### 文件结构

```txt
CUB-200-2011
├──Fine_Tuning_With_Pretrain        # 寻找最优学习率训练产生的tensorboard
├──Full_Train                       # 完整训练产生的tensorboard
├──Random_Init                      # 使用随机权重训练产生的tensorboard
├──Output                           # 运行结果，包含txt, log文件
    ├──best_accuracy_lr.txt         # 不同学习率得到的accuracy
    ├──best_accuracy_ep.txt         # 不同训练epoch得到的accuracy
    └──resent-sub.log               # 使用notebook文件训练得到的log日志
├──image                            # tensorboard可视化训练loss, accuracy的截图
├──data.py                          # 导入、预处理数据集
├──model.py                         # 定义在CUB数据集上的ResNet-18
├──solver.py                        # 求解器，封装训练测试函数
├──train.py                         # 封装训练parser                        
├──train.sh                         # 训练的shell脚本                        
├──test.py                          # 封装测试parser
├──resent-cub.ipynb                 # 训练、测试的notebook文件
└──requirements.txt                 # 依赖库                
```

### 模型权重

- [ResNet-18](https://drive.google.com/file/d/1fDSB7W71iAiA7-mWxaPpXXonUy0JKqrw/view?usp=sharing)

### 使用说明

- 安装依赖库
    ```bash
    # suppose you are currently in the root directory
    cd CUB-200-2011
    pip install -r requirements.txt 
    ```
- 下载模型权重文件，并将其放入`./Output`文件夹下
- 下载数据集，并将其放入`./data`文件夹下 (`./data/CUB-200-2011/...`)
  - 下载地址：https://data.caltech.edu/records/65de6-vp158/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
  - 解压缩
    ```bash
    tar -xzvf CUB_200_2011.tgz ./data
    ```
- 训练
  - 个性化训练 
    ```bash
    # example use
    python train.py --epochs 20 --ft_lr 0.0001 --fc_lr 0.001
    ```
  - 或直接运行shell脚本
    ```bash
    .\train.sh
    ``` 
- 测试
    ```bash
    # example use
    python test.py --path .\Output\resnet18_cub.pth
    ```
- 使用tensorboard可视化结果
    ```bash
    # choose one from them to execute
    tensorboard --logdir .\Fine_Tuning_With_Pretrain
    tensorboard --logdir .\Full_Train
    tensorboard --logdir .\Random_Init
    ``` 

> [!NOTE]
> 推荐直接使用notebook进行训练和测试

## 任务2：在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3

### 文件结构

```txt
Pascal VOC
├──configs
    ├──_base_                   # 和mmdetection提供的`/_base_`类似，仅保留相关文件
    └──pascal_voc               # 存放faster rcnn和yolov3的配置文件                
├──tools                    # 和mmdetection提供的`/tools`类似，仅保留相关文件
    ├──train.py                 # 训练模型
    └──test.py                                
├──demo                     # 存放用来进行推断的测试图片
    ├──in                       # 在原数据集中的测试图片，名称代表编号，来自VOC2007测试集
    └──out                      # 不在原数据集中的图片                     
├──img                      # inference结果图片，训练损失、准确率截图
    ├──in                       # 在数据集内图片的推断结果 
        ├──first/vis                  # faster_rcnn第一阶段proposal bbox
        └──second/vis                 # faster_rcnn第二阶段bbox
        └──first_second_stage.png     # 两个阶段的对比图
    ├──out                      # 数据集外图片的推断结果
        ├──faster_rcnn/vis            # faster_rcnn对数据集外图片推断的结果
        ├──yolov3/vis                 # yolov3对数据集外图片推断的结果
        └──faster_rcnn_yolov3.png     # 两个模型的对比图
    └──tensorboard              # tensorboard截图
├──work_dirs                # 训练产生的输出文件，包含日志，模型配置及权重
    ├──faster-rcnn              # faster-rcnn训练产生的文件
    └──yolov3                   # yolov3训练产生的文件
├──download.py              # 下载数据集                         
├──voc-mmdetection.ipynb    # notebook文件         
└──requirements.txt         # 依赖库      
```

### 模型权重
- [Faster R-CNN](https://drive.google.com/file/d/1ADEGGQ4bv6aeOwT7BD4aS3WpK6f5R-35/view?usp=drive_link)
- [YOLOv3](https://drive.google.com/file/d/1YM0HFjrWzOT8IiJHdItAy-PdtvYdk1as/view?usp=sharing)


### 使用说明

- 安装依赖库
    ```bash
    # suppose you are currently in the root directory
    cd Pascal VOC
    pip install -r requirements.txt 
    ```
- 下载模型权重文件
  - 将`Faster R-CNN`的模型权重文件放入`./work_dirs/faster-rcnn`文件夹下
  - 将`YOLOv3`的模型权重文件放入`./work_dirs/yolov3`文件夹下
- 下载数据集
  ```bash
  python download.py
  ``` 
- 训练
  ```bash
  python tools\train.py configs\pascal_voc\faster-rcnn.py      
  python tools\train.py configs\pascal_voc\yolov3.py        
  ```
- 推断
  - 查看`voc-mmdetection.ipynb`的第三个section
- 使用tensorboard可视化结果
    ```bash
    # choose one from them to execute
    tensorboard --logdir .\work_dirs\faster-rcnn\20240528_175108\vis_data\pascal_voc
    tensorboard --logdir .\work_dirs\yolov3\20240530_125156\vis_data\pascal_voc
    ```
> [!NOTE]
> 推荐使用notebook进行图片推断，可修改相应的图片路径
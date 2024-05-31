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

- https://drive.google.com/file/d/1fDSB7W71iAiA7-mWxaPpXXonUy0JKqrw/view?usp=sharing

### 使用说明

- 安装依赖库
    ```bash
    # suppose you are currently in the root directory
    cd CUB-200-2011
    pip install -r requirements.txt 
    ```
- 下载模型权重文件，并将其放入`./Output`文件夹下
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
> [!NOTE]
> 推荐直接使用notebook进行训练和测试，该notebook是在kaggle上进行训练 
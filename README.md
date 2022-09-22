# FastFlow-Paddle

## 目录

- [1. 简介]()
- [2. 数据集]()
- [3. 复现精度]()
- [4. 模型数据与环境]()
    - [4.1 目录介绍]()
    - [4.2 准备环境]()
    - [4.3 准备数据]()
- [5. 开始使用]()
    - [5.1 模型训练]()
    - [5.2 模型评估]()
    - [5.3 模型预测]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 模型信息]()

## 1. 简介
论文中提出了一种图像缺陷异常检测模型，可以不依赖于异常数据来检测未知的异常缺陷。具体来说，提出了使用2D normalizing flow的FastFLow，并使用它来估计概率分布。FastFLow可以作为plug-in模块，与任意的深度特征提取器(如ResNet和Vision Transformer)一起使用，用于无监督异常检测和定位。在训练阶段，FastFlow学习将输入的视觉特征转化为可处理的分布，并在测试阶段得到异常的似然（即概率）



**论文:** [FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows](https://arxiv.org/pdf/2111.07677.pdf)

**参考repo:** [anomalib](https://github.com/openvinotoolkit/anomalib/tree/main/anomalib/models/fastflow)

**aistudio体验教程:** [FastFlow_Paddle](https://aistudio.baidu.com/aistudio/projectdetail/4579846?contributionType=1)


## 2. 数据集

MVTec AD是MVtec公司提出的一个用于异常检测的数据集。与之前的异常检测数据集不同，该数据集模仿了工业实际生产场景，并且主要用于unsupervised anomaly detection。数据集为异常区域都提供了像素级标注，是一个全面的、包含多种物体、多种异常的数据集。数据集包含不同领域中的五种纹理以及十种物体，且训练集中只包含正常样本，测试集中包含正常样本与缺陷样本，因此需要使用无监督方法学习正常样本的特征表示，并用其检测缺陷样本。

数据集下载链接：[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 解压到data文件夹下


## 3. 复现精度

| FastFlow(ResNet18 )|   image-level AUC |  pixel-level AUC  |
|:-------------------|------------------:|------------------:|
| 论文               |               97.9 |             97.2 |
| 复现               |               98.0 |             97.2 |



## 4. 模型数据与环境

### 4.1 目录介绍

```
    |--images                         # 测试使用的样例图片，两张
    |--data                           # 训练和测试数据集
    |--lite_data                      # 自建立的小数据集，含有bottle 
    |--deploy                         # 预测部署相关
        |--export_model.py            # 导出模型
        |--infer.py                   # 部署预测
    |--configs                        # 配置
    |--models                         # 模型实现文件
    |--datasets                       # 数据集加载
    |--utils                          # 工具代码
    |--test_tipc                      # tipc代码
    |--predict.py                     # 预测代码
    |--eval.py                        # 评估代码
    |--train.py                       # 训练代码
    |--train.sh                       # 训练所有类别并进行测试
    |--README.md                      # 用户手册
```

### 4.2 准备环境

- 框架：
  - PaddlePaddle == 2.3.2
- 硬件：
  - GeForce RTX 3070
- 环境配置：
  - conda create -n fastflow python==3.7.0 (创建conda 环境)
  - conda activate fastflow
  - pip install paddlepaddle-gpu==2.3.2.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html (安装paddlepaddle)
  - 使用`pip install -r requirements.txt`安装其他依赖。


### 4.3 准备数据

- 全量数据训练：
  - 数据集下载链接：[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 解压到data文件夹下
  - data目录结构为：
    ```
    data
        └── bottle
            ├── ground_truth
            ├── test
            ├── train
        └── cable
            ├── ground_truth
            ├── test
            ├── train 
        └── ...
    ```
- 少量数据训练：
  - 无需下载数据集，直接使用lite_data里的数据
  
## 5. 开始使用
### 5.1 模型训练

- 全量数据训练：
  - `python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir exp -cat bottle`(训练单个类别)
  - `sh train.sh` （训练所有类别并进行测试，需要在脚本中指定exp_dir路径(dir_name=exp)和python环境(python=python3.7)）
- 少量数据训练：
  - `python train.py  -cfg ./configs/resnet18.yaml --data ./lite_data --exp_dir exp -cat bottle`
  
日志和模型训练权重保存在models文件下

可以将训练好的模型权重和日志[exp.zip 提取码：3ra1](https://pan.baidu.com/s/1EoDHZWbi8xsDVo6rWwqk2g) 解压放results，直接对模型评估和预测

### 5.2 模型评估(通过5.1完成训练后)

- 全量数据模型评估：`python eval.py  -cfg ./configs/resnet18.yaml --data ./data -cat all --exp_dir exp`
- 少量数据模型评估：`python eval.py  -cfg ./configs/resnet18.yaml --data ./lite_data -cat bottle --exp_dir exp/`


### 5.3 模型预测（需要预先完成5.1训练以及5.2的评估）

- 模型预测：`python predict.py -cfg ./configs/resnet18.yaml --category bottle --image_path images/bottle_good.png --exp_dir exp`

结果如下：
```
Normal - score:  0.614
可视化图在results/{exp_dir}/{category}下
```
- 基于推理引擎的模型预测：
```
python deploy/export_model.py --exp_dir exp --category bottle -cfg configs/resnet18.yaml --save_inference_dir ./results/inference

python deploy/infer.py  -cfg configs/resnet18.yaml --save_inference_dir ./results/inference --use_gpu True --image_path images/bottle_good.png
```
结果如下：
```
> python deploy/export_model.py --exp_dir exp --category bottle -cfg configs/resnet18.yaml --save_inference_dir ./results/inference
inference model has been saved into ./results/inference

> python deploy/infer.py  -cfg configs/resnet18.yaml --save_inference_dir ./results/inference --use_gpu True --image_path images/bottle_good.png
Normal - score:  0.614
```


## 6. 自动化测试脚本
- tipc 所有代码一键测试命令（少量数集）
```
bash test_tipc/test_train_inference_python.sh test_tipc/configs/fastflow/train_infer_python.txt lite_train_lite_infer 
```

结果日志在test_tipc/output/fastflow/lite_train_lite_infer目录下

## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 模型信息

| 信息 | 描述 |
| --- | --- |
| 作者 | Lele|
| 日期 | 2022年9月 |
| 框架版本 | PaddlePaddle==2.3.2 |
| 应用场景 | 异常检测 |
| 硬件支持 | GPU、CPU |

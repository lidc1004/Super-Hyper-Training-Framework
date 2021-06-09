# Super-Hyper Training Framework (PyTorch)



#### For English Version, Please refer the file : [**README-EN.md**](README-EN.md)



## 安装 

你需要首先安装相关的环境，请参考：[**requirements.txt**](requirements.txt)

推荐使用Anaconda建立独立的虚拟环境，并使用python >= 3.6.3；(本实验平台使用3.8.0)

对于Nvidia所提供的Apex半精度训练框架安装，请参考：[**https://github.com/NVIDIA/apex**](**https://github.com/NVIDIA/apex**)



## 数据准备

本项目提供Video Classification任务的用例；

数据集的地址索引可以在“Tools”文件夹下的args.py文件下进行修改；

对于不同的任务，数据集的构成也大有不同，因此在该文件中不做赘述；



## 框架代码组成

该框架由已经经过简化，由以下的文件夹以及文件组成：

```shell
├─Dataset
│
├─Models
│      model.py
│
├─Result
│      saved_model
|      SumWriter
│
├─Tools
|      args.py
|      init.py
|      utils.py
|      data.py
|
├─train.py
|
```

> train.py为模型训练主程式；
>
> Models文件夹放置的为你的深度学习模型；
>
> Result文件夹中存放的是你的训练结果或者是其它的生成文件；
>
> Tools文件夹中存放的是训练所需要的功能集；
>
> > args.py为参数配置文件；
> >
> > init.py为训练功能的初始化；
> >
> > utils.py为工具函数；
> >
> > data.py为数据处理主函数；



## 训练

该框架在args.py文件配置好参数后，只需要使用命令：

```shell
python -m torch.distributed.launch --nproc_per_node=n train.py
```

便可以启动分布式训练，其中n为你需要的进程数，每个进程都会独立地分配一张显卡；

**注意：**请务必在训练之前在配置文件中--enable_GPUs_id 写好你需要启动的显卡数量以及其ID

若只需要使用单卡或者多卡并行训练；只需要在配置中将--distributed设置为False即可；

之后使用命令：

```shell
python train.py
```

即可实现DP或者单卡训练；



## 半精度与全精度训练

该框架提供多种训练精度模式；

如果需要进行设置，请在args.py文件中更改--opt_level参数：

| 代号 | 模式                      |
| ---- | ------------------------- |
| O0   | 全精度训练（FP32）        |
| O1   | 混合精度训练（FP32+FP16） |
| O2   | 半精度训练（FP16）        |

大部分情况下我们推荐使用混合精度训练模式，即O1；



## 优化器以及损失函数

该框架仅提供“SGD，Adam”优化器供参考使用，如果有需要使用别的优化器，请在init.py文件下的optimizer_init（）函数中进行拓展；

```python
def optimizers(model, lr, betas):
    # you could define your own optimizers here
    SGD = optim.SGD(model.parameters(), lr=lr, momentum=betas[0], weight_decay=1e-2)
    Adam = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=1e-3)
	
    optimizer_dict = {"SGD":SGD, "Adam":Adam}

    return optimizer_dict
```
优化器与迭代器的代码将会在后续版本中添加并优化；



损失函数请在init.py文件下的init_criterion（）函数中进行拓展，由于不同的任务可能存在多个损失函数，因此我们将损失函数返回成列表以方便调用，在这个用例中只提供"BCE"和“CE”损失函数供参考；

```python
@staticmethod
def init_criterion(device, criterion_type):
    # you can defind your own criterion function here
    criterions = {"BCE":nn.BCEWithLogitsLoss(), "CE":nn.CrossEntropyLoss()}

    return [criterions[cr].to(device) for cr in criterion_type]
```

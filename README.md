# Super-Hyper Training Framework (PyTorch）



#### For Chinese Version, Please refer the file : [**README-中文.md**](README-中文.md)



## Install 

You need to build the relevant environment first, please refer to : [**requirements.txt**](requirements.txt)

It is recommended to use Anaconda to establish an independent virtual environment, and python > = 3.6.3; (3.8.0 is used for this experimental platform)

For the installation of APEX training framework provided by NVIDIA, please refer to : **[https://github.com/NVIDIA/apex](https://github.com/NVIDIA/apex)**



## Data Preparation

This project provides the use case of video classification task;

The address index of the dataset can be found in the **args.py**, where you could do the parameters modification;

For different tasks, the composition of data sets have significant different, so there is no repetition in this file;



## Frame Code Composition

The framework consists of the following simplified folders and files：

```shell
├─Dataset
│
├─Models
│      model.py
│
├─Result
│      saved_model
│      SumWriter
│
├─Tools
│      args.py
│      init.py
│      utils.py
│      data.py
│
├─train.py
│
```

> train.py -> which enable the training, also the main program；
>
> Models ->  where you store your models code；
>
> Result   -> here stores your training results or other generated files;
>
> Tools     -> the function set needed for training;
>
> > args.py -> parameter configuration file;
> >
> > init.py   -> Initialization of training function;
> >
> > utils.py ->  the function set needed for data loading;
> >
> > data.py -> the main function of data processing;



## Training

In this framework, after the parameters are configured in the file **args.py**, you only need to use the command:

```shell
python -m torch.distributed.launch --nproc_per_node=n train.py
```

Then you can start distributed training, where **n** is the number of processes you need, and each process will be assigned a graphics card independently;

**Note: ** Please set the number of graphics cards you need and their ID in parameter "--enable_GPUs_id" in the file **args.py** before training.



If you only need to use single card or multi card parallel training, just set -- distributed to false in the configuration;

Then use the command:

```shell
python train.py
```

DP or single GPU training can be realized;



## Semi-Precision and Full Precision Training

The framework provides a variety of training modes;

If you need to settings the modes, please change the parameter **'-- opt_ level'** in file **args.py** ;

| 代号 | 模式                        |
| ---- | --------------------------- |
| O0   | Full Precision（FP32）      |
| O1   | Semi-Precision（FP32+FP16） |
| O2   | Half-Precision（FP16）      |

In most cases, we recommend to use Semi-precision training mode, which is O1;



## Optimizer and Loss Function

The framework only provides "SGD, Adam" optimizer for reference. If you need to use another optimizers, you can defind them in function **optimizer_init()** in file **init.py**

```python
def optimizers(model, lr, betas):
    # you could define your own optimizers here
    SGD = optim.SGD(model.parameters(), lr=lr, momentum=betas[0], weight_decay=1e-2)
    Adam = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=1e-3)
	
    optimizer_dict = {"SGD":SGD, "Adam":Adam}

    return optimizer_dict
```
The code of optimizer and iterator will be added and optimized in subsequent versions;



The loss function could be added in function **init_criterion()** in file **init.py**. In this case, since different task might use mulitplu loss function, thus only "BCE" and "CE" loss functions are provided for reference;

```python
@staticmethod
def init_criterion(device, criterion_type):
    # you can defind your own criterion function here
    criterions = {"BCE":nn.BCEWithLogitsLoss(), "CE":nn.CrossEntropyLoss()}

    return [criterions[cr].to(device) for cr in criterion_type]
```

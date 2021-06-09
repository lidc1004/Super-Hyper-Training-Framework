import apex
import random
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from apex.parallel.LARC import LARC

class initialization():
    """docstring for initialization"""
    def __init__(self, local_rank= None):
        self.local_rank = None

    @staticmethod
    def init_params(enable_GPUs_id, distributed):
        if distributed:
            # FOR DISTRIBUTED:  Set the device according to local_rank.
            # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
            # environment variables, and requires that you use init_method=`env://`.
            torch.distributed.init_process_group(backend='nccl', 
                                                init_method='env://')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            torch.distributed.barrier()
            print("We would use distributed training") if local_rank == enable_GPUs_id[0] else None

        else:
            print("We would use default training")
            device = torch.device("cuda", enable_GPUs_id[0])
            local_rank = enable_GPUs_id[0]

        return device, local_rank


    @staticmethod
    def setup_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed) #cpu
        torch.cuda.manual_seed_all(seed) #gpu


    @staticmethod
    def load_pretrained_model(model, local_rank, model_path):
        if os.path.exists(model_path):
            pre = torch.load(model_path, map_location=torch.device('cpu'))
            model_dict = model.state_dict()
            pretrained_dict = {k.replace('module.', ''): v for k, v in pre.items() if k.replace('module.', '') in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            del pre, pretrained_dict, model_dict
            print("Loading Pretrained Model From : {} For Gpu : {} ".format(model_path, local_rank))
        
        else:
            print("We could not find the model in path --> ", model_path)
            print("We would not load the checkpoint or Pretrained Model --> ", local_rank)

        return model


    @staticmethod
    def to_GPU(model, device, local_rank ,mode="train"):
        if torch.cuda.is_available():
            print("Using GPU -> ", local_rank)
            return model.to(device)
        else:
            raise Exception("Cuda is Unavailable now, Please Check Your Device Setting")


    @staticmethod
    def init_criterion(device, criterion_type):
        # you can defind your own criterion function here
        criterions = {"BCE":nn.BCEWithLogitsLoss(), "CE":nn.CrossEntropyLoss()}

        return [criterions[cr].to(device) for cr in criterion_type]


    @staticmethod
    def optimizer_init(model, local_rank, optimizer_name, lr, step_size, gamma, betas=None):
        def optimizers(model, lr, betas):
            # you could define your own optimizers here
            SGD = optim.SGD(model.parameters(), lr=lr, momentum=betas[0], weight_decay=1e-2)
            Adam = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-2, weight_decay=1e-3)

            optimizer_dict = {"SGD":SGD, "Adam":Adam}
            
            return optimizer_dict
                                             

        if optimizer_name in ['SGD', 'Adam']:
            optimizer = optimizers(model, lr, betas)[optimizer_name]
        else:
            raise Exception("Only 'SGD' and 'Adam' are available, if you want to use other optimizer, please add it in funtion optimizer_init() in init.py")
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
        
        return optimizer, scheduler


    @staticmethod
    def amp_init(model, optimizer, local_rank, use_amp=True ,sync_bn=False, opt_level='O1'):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        if use_amp == True:
            # with apex syncbn we sync bn per group because it speeds up computation
            # compared to global syncbn
            model.to('cuda:'+str(local_rank))
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
            print("Using Automatic Mixed Precision Training")
        else:
            print("Using Normal-Precision Training")

        return model, optimizer
                

    @staticmethod
    def use_multi_GPUs(model, local_rank, enable_GPUs_id, distributed=True): 
        def Distributed_GPUs(model, local_rank):
            torch.distributed.barrier()
            print("Using Multi-Gpu, devices count is:{}".format(torch.cuda.device_count())) if local_rank == enable_GPUs_id[0] else None
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)
            return model
        
        def Parallel_GPUs(model, local_rank):
            model = nn.DataParallel(model, device_ids=enable_GPUs_id)
            print("Using Multi-Gpu, device_ids is: {}".format(enable_GPUs_id))
            return model

        def Single_GPU():
            return print("Using Single-Gpu, device_ids is: {}".format(enable_GPUs_id[0]))


        if distributed:# multi-Gpu
            model = Distributed_GPUs(model, local_rank)
        elif distributed == False and len(enable_GPUs_id) > 1:
            model = Parallel_GPUs(model, enable_GPUs_id)
        else:
            Single_GPU()

        return model
import os
import sys
import time
import json
import numpy as np
import apex
import math
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from prefetch_generator import BackgroundGenerator

from Tools.utils import *
from Tools.data import *
from Tools.init import initialization
from Tools.args import args

from Models.slowfastnet import resnet50

class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k, element in enumerate(self.batch):
                self.batch[k] = element.to(self.device, non_blocking=True)
            return self.batch

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


def _print(words, local_rank):
    print(words) if local_rank == args.enable_GPUs_id[0] else None


def train(model, train_dataloader, epoch, criterion, optimizer, local_rank, device):
    top1, top3 = topN_params_init()
    loss_record = []
    
    model.train()
    train_bar = tqdm(total=len(train_dataloader), ncols=80) if local_rank == args.enable_GPUs_id[0] else None

    prefetcher = DataPrefetcher(BackgroundGenerator(train_dataloader), device)
    batch = prefetcher.next()
    step = 0

    while batch is not None:
        step += 1
        if step >= len(train_dataloader):
            break
        batch = prefetcher.next()
        inputs = batch[0]
        class_labels = batch[1]

        outputs = model(inputs)

        loss = criterion[0](outputs, class_labels)
        loss_record.append(loss.item())

        top1, top3 = update_top1_3(inputs, outputs, class_labels, top1, top3)

        optimizer.zero_grad()

        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        torch.distributed.barrier() if args.distributed else None

        train_bar.update(1) if local_rank == args.enable_GPUs_id[0] else None

        if local_rank == args.enable_GPUs_id[0] and step%100 == 0:
            # measure accuracy and record loss
            print('-------------------------------------------------------')
            for params in optimizer.param_groups:
                print('lr: {lr:.7f}, batch_size: {batch_size}'.format(lr=params['lr'], \
                                                                        batch_size = args.batch_size))

            print_multi_quota(epoch, step-1, train_dataloader, loss_record, top1, top3)

    if local_rank == args.enable_GPUs_id[0]:
        # measure accuracy and record loss
        print('-------------------------------------------------------')
        for params in optimizer.param_groups:
            print('lr: {lr:.7f}, batch_size: {batch_size}'.format(lr=params['lr'], \
                                                                    batch_size =  args.batch_size))

        print_multi_quota(epoch, step-1, train_dataloader, loss_record, top1, top3)


def validation(model, val_dataloader, epoch, criterion, device):
    top1, top3 = topN_params_init()
    
    model.eval()
    TP, TN, FN, FP, loss_record = 0, 0, 0 ,0, []

    val_bar = tqdm(total=len(val_dataloader), ncols=80)

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(BackgroundGenerator(val_dataloader)):
            inputs = inputs.to(device)
            class_labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion[0](outputs, class_labels)
            loss_record.append(loss.item())

            top1, top3 = update_top1_3(inputs, outputs, class_labels, top1, top3)

            val_bar.update(1)

    # measure accuracy and record loss
    print('\n----------------Validation----------------')

    print_multi_quota(epoch, step, val_dataloader, loss_record, top1, top3)

    if epoch % 1 == 0 and epoch >= 0:
        checkpoint = os.path.join(args.model_save_dir,
                                    "epoch" + str(epoch) + '_loss_' + str(format(np.mean(loss_record[:step]), '.3f')) + '_pre_' + str(format(top1.avg, '.2f'))  + "%.pth")
        torch.save(model.state_dict(), checkpoint)


def main():
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(",".join(str(id) for id in args.enable_GPUs_id))

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    init = initialization()
    init.setup_seed(666)
    device, local_rank = init.init_params(args.enable_GPUs_id, args.distributed)

    _print("Using cudnn benchmark: {} ".format(args.cudnn_benchmark), local_rank)
    torch.backends.cudnn.benchmark = True if args.cudnn_benchmark and torch.cuda.is_available() else False
    # init the validation datasete

    val_dataloader = DataLoader(VideoDataset(args.val_dataset_path,
                                             resize_shape=args.resize_shape,
                                             mode='validation',
                                             clip_len=args.clip_len,
                                             crop_size=args.crop_size),
                                batch_size=args.batch_size, 
                                shuffle=True,
                                pin_memory=False,
                                num_workers=args.num_workers) if local_rank == args.enable_GPUs_id[0] else None

    torch.distributed.barrier() if args.distributed else None
    # load model
    _print("Loading Network: SpeedNet\n Total Epoch: 400\n", local_rank)
    model = resnet50(class_num = 400)
    # load pretrained model
    model = init.load_pretrained_model(model, local_rank, args.model_path) if args.load_checkpoint else model
    model = init.to_GPU(model, device, local_rank)
    # init the criterion to device
    criterion = init.init_criterion(device, args.criterion_type)
    # init the optimizer and scheduler
    optimizer, scheduler = init.optimizer_init(
                                               model, 
                                               local_rank, 
                                               args.optimizer, 
                                               args.lr, 
                                               args.op_scheduler[0], 
                                               args.op_scheduler[1], 
                                               args.betas)
    _print("optimizer parameters: \n   Method:{} \n   Step_size:{} \n   Gamma:{} \n" \
            .format(args.optimizer, args.op_scheduler[0], args.op_scheduler[1]), local_rank)

    # use mixed precision do the training
    model, optimizer = init.amp_init(model, optimizer, local_rank, args.use_amp, args.sync_bn, args.opt_level)
    # use multipul GPUs for training
    model = init.use_multi_GPUs(model, local_rank, args.enable_GPUs_id, args.distributed)

    if args.mode == 'validation' and local_rank == args.enable_GPUs_id[0]:
        validation(model, val_dataloader, epoch, criterion, device)
        sys.exit()

    if args.mode == 'train' or 'training':
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        for epoch in range(args.epoch_num):

            if epoch == 0:
                print("\nUsing default training -> Batch Size: {}, local_rank:{}".format(args.batch_size, local_rank))
                print("\n                       -> Learning Rate: {}, local_rank:{}".format(args.lr, local_rank))

            train_dataset = VideoDataset(args.train_dataset_path,
                                         local_rank = local_rank,
                                         enable_GPUs_num = len(args.enable_GPUs_id),
                                         distributed_load = args.distributed,
                                         resize_shape=args.resize_shape,
                                         mode='train',
                                         clip_len=args.clip_len,
                                         crop_size=args.crop_size)

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers,
                                          pin_memory=False,
                                          shuffle=True)

            # train_dataloader = Multigird_dataloader(epoch, local_rank)
            print("Loading Training Dataset : Gpu-Ids -> {}".format(local_rank))

            train(model, train_dataloader, epoch, criterion, optimizer, local_rank, device)
            scheduler.step()
            
            del train_dataloader

            if (epoch+1) % 1 == 0 and local_rank == args.enable_GPUs_id[0]:
                validation(model, val_dataloader, epoch, criterion, device)
                print('\n-------------------------------------------------------')
                print("Start A New Training Epoch : ")

if __name__ == '__main__':
    main()

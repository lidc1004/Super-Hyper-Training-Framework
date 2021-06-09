import os
from pathlib import Path

import random

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
import torch

from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):

    def __init__(self, directory_list, local_rank=0, enable_GPUs_num=0, distributed_load=False, resize_shape=[224, 224] , mode='train', clip_len=32, crop_size = 168):
        
        self.clip_len, self.crop_size, self.resize_shape = clip_len, crop_size, resize_shape
        self.mode = mode

        self.fnames, labels = [],[]
        # get the directory of the specified split
        for directory in directory_list:
            folder = Path(directory)
            print("Load dataset from folder : ", folder)
            for label in sorted(os.listdir(folder)):
                for fname in os.listdir(os.path.join(folder, label)) if mode=="train" else os.listdir(os.path.join(folder, label))[:10]:
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)

        random_list = list(zip(self.fnames, labels))
        random.shuffle(random_list)
        self.fnames[:], labels[:] = zip(*random_list)

        # self.fnames = self.fnames[:240]

        if mode == 'train' and distributed_load:
            single_num_ = len(self.fnames)//enable_GPUs_num
            self.fnames = self.fnames[local_rank*single_num_:((local_rank+1)*single_num_)]
            labels = labels[local_rank*single_num_:((local_rank+1)*single_num_)]

        self.transform = transforms.Compose([
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

                
    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classess
        self.index = index
        buffer = self.loadvideo()
        
        height_index = np.random.randint(buffer.shape[2] - self.crop_size)
        width_index = np.random.randint(buffer.shape[3] - self.crop_size)

        return buffer[:,:,height_index:height_index + self.crop_size, width_index:width_index + self.crop_size], self.label_array[index]


    def __len__(self):
        return len(self.fnames)


    def loadvideo(self):
        # initialize a VideoCapture object to read video data into a numpy array
        flip, flipCode = 1, random.choice([-1,0,1]) if np.random.random() < 0.5 else 0
        
        self.frame_count = 0
        while self.frame_count < self.min_len+2:
            video_stream = self.init_video()

        speed_rate = np.random.randint(1, 3) if self.frame_count > self.clip_len*2+2 else 1
        time_index = np.random.randint(self.frame_count - self.clip_len * speed_rate)

        start_idx, end_idx, final_idx = time_index, time_index+(self.clip_len*speed_rate), self.frame_count-1
        count, sample_count, retaining = 0, 0, True

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
        
        while (count <= end_idx and retaining):
            retaining, frame = video_stream.read()
            if count < start_idx:
                count += 1
                continue
            if count % speed_rate == speed_rate-1 and count >= start_idx and sample_count < self.clip_len:
                if flip:
                    frame = cv2.flip(frame, flipCode=flipCode)
                try:
                    buffer[sample_count] = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                except cv2.error as err:
                    continue
                sample_count += 1
            count += 1
        video_stream.release()

        return buffer.transpose((1, 0, 2, 3))


    def init_video(self):
        try:
            video_stream = cv2.VideoCapture(self.fnames[self.index])
        except RuntimeError:
            self.index = np.random.randint(self.__len__())
            video_stream = self.init_video()
        
        self.frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_count < self.clip_len*2+2:
            self.index = np.random.randint(self.__len__())
        
        return video_stream


if __name__ == '__main__':

    datapath = ['/root/kinetics-400/compress/train_256']
    
    from prefetch_generator import BackgroundGenerator
    
    dataset = VideoDataset(datapath, 
                            resize_shape=[224, 224],
                            mode='validation')
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=24, pin_memory=False)

    bar = tqdm(total=len(dataloader), ncols=80)
    '''
    prefetcher = DataPrefetcher(BackgroundGenerator(dataloader), 0)
    batch = prefetcher.next()
    iter_id = 0
    while batch is not None:
        iter_id += 1
        bar.update(1)
        if iter_id >= len(dataloader):
            break

        batch = prefetcher.next()
        print(batch[0].shape)
        print("label: ", batch[1])

    '''
    for step, (buffer, labels) in enumerate(BackgroundGenerator(dataloader)):
        print(buffer.squeeze(0).shape)
        print("label: ", labels)
        bar.update(1)
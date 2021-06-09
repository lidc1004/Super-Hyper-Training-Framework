import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_dataset_path', type=str, help='path to the video training dataset',
                    default=['/root/kinetics-400/compress/train_256'])
parser.add_argument('--val_dataset_path', type=str, help='path to the video validation dataset',
                    default=['/root/kinetics-400/compress/val_256'])


parser.add_argument('--record_save_dir', type=str, help = 'setting the model save dir',
                    default='./Result/record/')
parser.add_argument('--model_save_dir', type=str, help = 'setting the model save dir',
                    default='./Result/saved_model')
parser.add_argument('--log_dir', type=str, help='setting the model save dir',
                    default='./Result/SumWriter')

parser.add_argument('--mode', type=str, help='The Type of running mode',
                    default='train')
parser.add_argument('--resize_shape', type=list, help="The size of resize order, format is [int, int]",
                    default=[224, 224])
parser.add_argument('--clip_len', type=int, help="The length of single clip",
                    default=32)
parser.add_argument('--crop_size', type=int, help="The size of random crop",
                    default=168)
parser.add_argument('--jumpinterval', type=int, help="clip skipping",
                    default=1)

parser.add_argument('--use_gpu', type=bool, help = 'Whether use gpus',
                    default=True)
parser.add_argument('--distributed', type=bool, help = 'Whether use multipul gpus and distributed training',
                    default=True)
parser.add_argument('--load_checkpoint', type=bool, help = 'Whether load the checkpoint or the pretrained model',
                    default=False)
parser.add_argument('--model_path', type=str, help = 'the path you save the trained model',
                    default='./Result/saved_model/epoch2_loss_5.140_pre_6.17%.pth')
parser.add_argument('--opt_level', type=str, help = 'setting the optimization level, the first letter should be CAPS',
                    default='O1')
parser.add_argument('--local_rank', type=list, help = 'setting the optimization level, the first letter should be CAPS')
parser.add_argument('--enable_GPUs_id', type=list, help = 'gpu devices ids',
                    default=[0, 1, 2, 3])
parser.add_argument('--use_amp', type=bool, help = 'whether use amp training',
                    default=True)
parser.add_argument('--sync_bn', type=bool, help = 'Whether use sync_bn',
                    default=False)
parser.add_argument('--cudnn_benchmark', type=bool, help = 'Whether use the cudnn_benchmark',
                    default=True)
parser.add_argument('--criterion_type', type=list, help='setting the types of criterion',
                    default=['CE'])
parser.add_argument('--optimizer', type=str, help='setting the type of optimizer',
                    default='Adam')
parser.add_argument('--batch_size', type=int, help='setting the batch size',
                    default=16)
parser.add_argument('--lr', type=float, help='setting the learing rate',
                    default=1.25e-3)
parser.add_argument('--betas', type=tuple, help='setting the betas for optimizer',
                    default=(0.9, 0.999))
parser.add_argument('--op_scheduler', type=list, help='setting the parameters for scheduler, op_scheduler[0] is step_size, op_scheduler[2] is gamma',
                    default=[1, 0.9457])
parser.add_argument('--epoch_num', type=int, help = 'setting the epoch_num',
                    default=150)
parser.add_argument('--num_workers', type=int, help='setting the workers number',
                    default=8)

args = parser.parse_args()
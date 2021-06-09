import numpy as np


def print_multi_quota(epoch, step, dataset, loss_record, top1, top3):
    print('Epoch: [{0}][{1}/{2}]'.format(epoch, step+1, len(dataset)))
    print('loss: {loss:.5f}'.format(loss=np.mean(loss_record[:step])))
    print('Top-1 accuracy: {top1_acc:.2f}%, Top-3 accuracy: {top3_acc:.2f}%'.format(
            top1_acc=top1.avg,
            top3_acc=top3.avg))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topN_params_init():
    top1 = AverageMeter()
    top3 = AverageMeter()
    return top1, top3


def update_top1_3(inputs, outputs, labels, top1, top3):
    def Multi_Accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    prec1, prec3 = Multi_Accuracy(outputs.data, labels, topk=(1, 3))
    top1.update(prec1.item(), inputs.size(0))
    top3.update(prec3.item(), inputs.size(0))
    return top1, top3


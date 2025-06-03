import os
import time
from collections import OrderedDict

import numpy as np
import torch
import random
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lib.dataset import VideoDataset
# from lib.dataset_new import VideoDataset
from lib import slowfastnet
from tensorboardX import SummaryWriter
from lib import slowfastnet_three
from lib import slowfastnet_two
from lib import slowfastnet_three_xiugai
from lib import slowfastnet_ln
from lib import slowfastnet_original
from lib import slowfastnet_TtS
from modules import r3d, r3d_zf
from modules import r3d_v2
from lib import DynamicLRScheduler
from modules import X3D


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# def seed_everything(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     cudnn.deterministic = True
#     cudnn.benchmark = False


def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, (inputs, labels) in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (step + 1) % params['display'] == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                print('lr: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)
    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top5_acc_epoch', top5.avg, epoch)


def validation(model, val_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if (step + 1) % params['display'] == 0:
                print('----validation----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)

    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)

    return losses.avg, top1.avg, top5.avg

def load_pretrained_weights(model, pretrained_path):
    # 加载预训练权重
    original_weights = torch.load(pretrained_path, map_location="cpu")['state_dict']

    # 移除分类头相关权重
    keys_to_remove = [k for k in original_weights.keys()
                      if 'cls_head' in k or 'fc_cls' in k]
    for k in keys_to_remove:
        original_weights.pop(k)

    # 创建键名映射字典
    key_mapping = OrderedDict()

    # 1. 映射fast路径权重
    # fast_conv1 -> backbone.fast_path.conv1
    key_mapping['fast_conv1.weight'] = 'backbone.fast_path.conv1.conv.weight'
    key_mapping['fast_bn1.weight'] = 'backbone.fast_path.conv1.bn.weight'
    key_mapping['fast_bn1.bias'] = 'backbone.fast_path.conv1.bn.bias'
    key_mapping['fast_bn1.running_mean'] = 'backbone.fast_path.conv1.bn.running_mean'
    key_mapping['fast_bn1.running_var'] = 'backbone.fast_path.conv1.bn.running_var'
    key_mapping['fast_bn1.num_batches_tracked'] = 'backbone.fast_path.conv1.bn.num_batches_tracked'

    # 2. 映射slow路径权重
    # slow_conv1 -> backbone.slow_path.conv1
    key_mapping['slow_conv1.weight'] = 'backbone.slow_path.conv1.conv.weight'
    key_mapping['slow_bn1.weight'] = 'backbone.slow_path.conv1.bn.weight'
    key_mapping['slow_bn1.bias'] = 'backbone.slow_path.conv1.bn.bias'
    key_mapping['slow_bn1.running_mean'] = 'backbone.slow_path.conv1.bn.running_mean'
    key_mapping['slow_bn1.running_var'] = 'backbone.slow_path.conv1.bn.running_var'
    key_mapping['slow_bn1.num_batches_tracked'] = 'backbone.slow_path.conv1.bn.num_batches_tracked'

    # 3. 映射lateral连接权重
    key_mapping['lateral_p1.weight'] = 'backbone.slow_path.conv1_lateral.conv.weight'
    key_mapping['lateral_res2.weight'] = 'backbone.slow_path.layer1_lateral.conv.weight'
    key_mapping['lateral_res3.weight'] = 'backbone.slow_path.layer2_lateral.conv.weight'
    key_mapping['lateral_res4.weight'] = 'backbone.slow_path.layer3_lateral.conv.weight'

    # 4. 映射residual blocks权重
    # 定义residual blocks的映射关系
    res_blocks = {
        'res2': 'layer1',
        'res3': 'layer2',
        'res4': 'layer3',
        'res5': 'layer4'
    }

    for src_block, dst_block in res_blocks.items():
        # 处理fast路径
        for i in range(3 if src_block == 'res2' else 4 if src_block == 'res3' else 6 if src_block == 'res4' else 3):
            # conv1
            key_mapping[f'fast_{src_block}.{i}.conv1.weight'] = f'backbone.fast_path.{dst_block}.{i}.conv1.conv.weight'
            key_mapping[f'fast_{src_block}.{i}.bn1.weight'] = f'backbone.fast_path.{dst_block}.{i}.conv1.bn.weight'
            key_mapping[f'fast_{src_block}.{i}.bn1.bias'] = f'backbone.fast_path.{dst_block}.{i}.conv1.bn.bias'
            key_mapping[
                f'fast_{src_block}.{i}.bn1.running_mean'] = f'backbone.fast_path.{dst_block}.{i}.conv1.bn.running_mean'
            key_mapping[
                f'fast_{src_block}.{i}.bn1.running_var'] = f'backbone.fast_path.{dst_block}.{i}.conv1.bn.running_var'
            key_mapping[
                f'fast_{src_block}.{i}.bn1.num_batches_tracked'] = f'backbone.fast_path.{dst_block}.{i}.conv1.bn.num_batches_tracked'

            # conv2
            key_mapping[f'fast_{src_block}.{i}.conv2.weight'] = f'backbone.fast_path.{dst_block}.{i}.conv2.conv.weight'
            key_mapping[f'fast_{src_block}.{i}.bn2.weight'] = f'backbone.fast_path.{dst_block}.{i}.conv2.bn.weight'
            key_mapping[f'fast_{src_block}.{i}.bn2.bias'] = f'backbone.fast_path.{dst_block}.{i}.conv2.bn.bias'
            key_mapping[
                f'fast_{src_block}.{i}.bn2.running_mean'] = f'backbone.fast_path.{dst_block}.{i}.conv2.bn.running_mean'
            key_mapping[
                f'fast_{src_block}.{i}.bn2.running_var'] = f'backbone.fast_path.{dst_block}.{i}.conv2.bn.running_var'
            key_mapping[
                f'fast_{src_block}.{i}.bn2.num_batches_tracked'] = f'backbone.fast_path.{dst_block}.{i}.conv2.bn.num_batches_tracked'

            # conv3
            key_mapping[f'fast_{src_block}.{i}.conv3.weight'] = f'backbone.fast_path.{dst_block}.{i}.conv3.conv.weight'
            key_mapping[f'fast_{src_block}.{i}.bn3.weight'] = f'backbone.fast_path.{dst_block}.{i}.conv3.bn.weight'
            key_mapping[f'fast_{src_block}.{i}.bn3.bias'] = f'backbone.fast_path.{dst_block}.{i}.conv3.bn.bias'
            key_mapping[
                f'fast_{src_block}.{i}.bn3.running_mean'] = f'backbone.fast_path.{dst_block}.{i}.conv3.bn.running_mean'
            key_mapping[
                f'fast_{src_block}.{i}.bn3.running_var'] = f'backbone.fast_path.{dst_block}.{i}.conv3.bn.running_var'
            key_mapping[
                f'fast_{src_block}.{i}.bn3.num_batches_tracked'] = f'backbone.fast_path.{dst_block}.{i}.conv3.bn.num_batches_tracked'

            # downsample
            if i == 0:  # 只有第一个block有downsample
                key_mapping[
                    f'fast_{src_block}.{i}.downsample.0.weight'] = f'backbone.fast_path.{dst_block}.{i}.downsample.conv.weight'
                key_mapping[
                    f'fast_{src_block}.{i}.downsample.1.weight'] = f'backbone.fast_path.{dst_block}.{i}.downsample.bn.weight'
                key_mapping[
                    f'fast_{src_block}.{i}.downsample.1.bias'] = f'backbone.fast_path.{dst_block}.{i}.downsample.bn.bias'
                key_mapping[
                    f'fast_{src_block}.{i}.downsample.1.running_mean'] = f'backbone.fast_path.{dst_block}.{i}.downsample.bn.running_mean'
                key_mapping[
                    f'fast_{src_block}.{i}.downsample.1.running_var'] = f'backbone.fast_path.{dst_block}.{i}.downsample.bn.running_var'
                key_mapping[
                    f'fast_{src_block}.{i}.downsample.1.num_batches_tracked'] = f'backbone.fast_path.{dst_block}.{i}.downsample.bn.num_batches_tracked'

        # 处理slow路径
        for i in range(3 if src_block == 'res2' else 4 if src_block == 'res3' else 6 if src_block == 'res4' else 3):
            # conv1
            key_mapping[f'slow_{src_block}.{i}.conv1.weight'] = f'backbone.slow_path.{dst_block}.{i}.conv1.conv.weight'
            key_mapping[f'slow_{src_block}.{i}.bn1.weight'] = f'backbone.slow_path.{dst_block}.{i}.conv1.bn.weight'
            key_mapping[f'slow_{src_block}.{i}.bn1.bias'] = f'backbone.slow_path.{dst_block}.{i}.conv1.bn.bias'
            key_mapping[
                f'slow_{src_block}.{i}.bn1.running_mean'] = f'backbone.slow_path.{dst_block}.{i}.conv1.bn.running_mean'
            key_mapping[
                f'slow_{src_block}.{i}.bn1.running_var'] = f'backbone.slow_path.{dst_block}.{i}.conv1.bn.running_var'
            key_mapping[
                f'slow_{src_block}.{i}.bn1.num_batches_tracked'] = f'backbone.slow_path.{dst_block}.{i}.conv1.bn.num_batches_tracked'

            # conv2
            key_mapping[f'slow_{src_block}.{i}.conv2.weight'] = f'backbone.slow_path.{dst_block}.{i}.conv2.conv.weight'
            key_mapping[f'slow_{src_block}.{i}.bn2.weight'] = f'backbone.slow_path.{dst_block}.{i}.conv2.bn.weight'
            key_mapping[f'slow_{src_block}.{i}.bn2.bias'] = f'backbone.slow_path.{dst_block}.{i}.conv2.bn.bias'
            key_mapping[
                f'slow_{src_block}.{i}.bn2.running_mean'] = f'backbone.slow_path.{dst_block}.{i}.conv2.bn.running_mean'
            key_mapping[
                f'slow_{src_block}.{i}.bn2.running_var'] = f'backbone.slow_path.{dst_block}.{i}.conv2.bn.running_var'
            key_mapping[
                f'slow_{src_block}.{i}.bn2.num_batches_tracked'] = f'backbone.slow_path.{dst_block}.{i}.conv2.bn.num_batches_tracked'

            # conv3
            key_mapping[f'slow_{src_block}.{i}.conv3.weight'] = f'backbone.slow_path.{dst_block}.{i}.conv3.conv.weight'
            key_mapping[f'slow_{src_block}.{i}.bn3.weight'] = f'backbone.slow_path.{dst_block}.{i}.conv3.bn.weight'
            key_mapping[f'slow_{src_block}.{i}.bn3.bias'] = f'backbone.slow_path.{dst_block}.{i}.conv3.bn.bias'
            key_mapping[
                f'slow_{src_block}.{i}.bn3.running_mean'] = f'backbone.slow_path.{dst_block}.{i}.conv3.bn.running_mean'
            key_mapping[
                f'slow_{src_block}.{i}.bn3.running_var'] = f'backbone.slow_path.{dst_block}.{i}.conv3.bn.running_var'
            key_mapping[
                f'slow_{src_block}.{i}.bn3.num_batches_tracked'] = f'backbone.slow_path.{dst_block}.{i}.conv3.bn.num_batches_tracked'

            # downsample
            if i == 0:  # 只有第一个block有downsample
                key_mapping[
                    f'slow_{src_block}.{i}.downsample.0.weight'] = f'backbone.slow_path.{dst_block}.{i}.downsample.conv.weight'
                key_mapping[
                    f'slow_{src_block}.{i}.downsample.1.weight'] = f'backbone.slow_path.{dst_block}.{i}.downsample.bn.weight'
                key_mapping[
                    f'slow_{src_block}.{i}.downsample.1.bias'] = f'backbone.slow_path.{dst_block}.{i}.downsample.bn.bias'
                key_mapping[
                    f'slow_{src_block}.{i}.downsample.1.running_mean'] = f'backbone.slow_path.{dst_block}.{i}.downsample.bn.running_mean'
                key_mapping[
                    f'slow_{src_block}.{i}.downsample.1.running_var'] = f'backbone.slow_path.{dst_block}.{i}.downsample.bn.running_var'
                key_mapping[
                    f'slow_{src_block}.{i}.downsample.1.num_batches_tracked'] = f'backbone.slow_path.{dst_block}.{i}.downsample.bn.num_batches_tracked'

    # 5. 映射最后的fc层
    key_mapping['fc.weight'] = 'cls_head.fc_cls.weight'

    # 创建新的state_dict
    new_state_dict = OrderedDict()

    # 遍历模型的state_dict
    for key, value in model.state_dict().items():
        if key in key_mapping:
            # 如果键在映射表中，从预训练权重中获取对应的权重
            pretrained_key = key_mapping[key]
            if pretrained_key in original_weights:
                new_state_dict[key] = original_weights[pretrained_key]
                # print(f"Mapped: {key} -> {pretrained_key}")
            else:
                print(f"Warning: Pretrained key {pretrained_key} not found for model key {key}")
                new_state_dict[key] = value  # 使用随机初始化
        else:
            print(f"Warning: No mapping found for key {key}, using random initialization")
            new_state_dict[key] = value  # 使用随机初始化

    # 加载新的state_dict到模型
    model.load_state_dict(new_state_dict, strict=False)

    return model
def main():
    best_top1_acc = 0.0
    best_model_path = params['best_model_path']
    os.makedirs(best_model_path, exist_ok=True)
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("Loading dataset")
    train_dataloader = \
        DataLoader(
            VideoDataset(params['dataset'], mode='train', clip_len=params['clip_len'],
                         frame_sample_rate=params['frame_sample_rate']),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    val_dataloader = \
        DataLoader(
            VideoDataset(params['dataset'], mode='val', clip_len=params['clip_len'],
                         frame_sample_rate=params['frame_sample_rate']),
            batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

    print("load model")
    # model = slowfastnet_TtS.resnet50(class_num=params['num_classes'])
    model = slowfastnet_original.resnet50(class_num=params['num_classes'])
    # model = X3D.generate_model('S')
    # model = r3d_zf.generate_model(50)

    if params['pretrained'] is not None:
        load_pretrained_weights(model,params['pretrained'])
    #     # pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
    #     # try:
    #     #     model_dict = model.module.state_dict()
    #     # except AttributeError:
    #     #     model_dict = model.state_dict()
    #     # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     print("load pretrain model")
    #     # model_dict.update(pretrained_dict)
    #     # model.load_state_dict(model_dict)
    #     model_weights = model.state_dict()  # 获取模型的初始权重
    #
    #     # 加载预训练权重文件
    #     pretrained_weights = torch.load(params['pretrained'])  # 替换为你的预训练权重路径
    #     state_dict = pretrained_weights['state_dict']  # 提取 state_dict 部分
    #     # 去掉 module 前缀
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         new_key = key.replace("module.", "")
    #         if "fc." not in new_key:  # 跳过分类头
    #             new_state_dict[new_key] = value
    #
    #     # 加载处理后的权重到模型
    #     model.load_state_dict(new_state_dict, strict=False)  # strict=False 允许部分权重加载
    #     # num_ftrs = model.fc.in_features
    #     # model.fc = nn.Linear(num_ftrs, params['num_classes'])

    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'],
                          weight_decay=params['weight_decay'])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step'], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epoch_num'], )
    # scheduler = DynamicLRScheduler.DynamicLRScheduler(optimizer, patience=5, min_lr=1e-6)
    # scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
    #     optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=params['epoch_num'] * 0.05),
    #     optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epoch_num'] * 0.95)],
    #                                             milestones=[params['epoch_num'] * 0.05])
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


    model_save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    for epoch in range(params['epoch_num']):
        train(model, train_dataloader, epoch, criterion, optimizer, writer)
        if epoch % 2 == 0:
            loss_avg, top1_acc, top5_acc = validation(model, val_dataloader, epoch, criterion, optimizer, writer)
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_path = os.path.join(best_model_path, "best_model.pth")
                torch.save({'epoch': epoch,'state_dict': model.module.state_dict(), 'best_acc': best_top1_acc}, best_path)
                timestamp_path = os.path.join(model_save_dir, f"best_model_epoch{epoch}_acc{top1_acc:.2f}.pth")
                torch.save(model.module.state_dict(), timestamp_path)
            # scheduler.step(loss_avg)
        scheduler.step()
        # if epoch % 20 == 0:
        #     checkpoint = os.path.join(model_save_dir,
        #                               "clip_len_" + str(params['clip_len']) + "frame_sample_rate_" + str(
        #                                   params['frame_sample_rate']) + "_checkpoint_" + str(epoch) + ".pth.tar")
        #     torch.save(model.module.state_dict(), checkpoint)
    writer.close


if __name__ == '__main__':
    main()

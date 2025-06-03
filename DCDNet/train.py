import os
import time
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
from modules import X3D
from lib import DynamicLRScheduler


def load_pretrained_model(model, pretrained_path):
    """
    加载预训练模型参数到当前模型

    参数:
        model: 当前模型实例
        pretrained_path: 预训练模型文件路径

    返回:
        加载了预训练参数的模型
    """
    # 加载预训练权重
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    # 获取当前模型的参数字典
    model_dict = model.state_dict()

    # 1. 只保留backbone部分的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'backbone.' in k}

    # 2. 移除backbone.前缀
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}

    # 3. 创建详细的名称映射关系
    name_mapping = {
        # conv1部分
        'conv1_s.conv.weight': 'conv1_s.weight',
        'conv1_t.conv.weight': 'conv1_t.weight',
        'conv1_t.bn.weight': 'conv1_t.weight',  # 注意: 你的模型可能没有这个BN层

        # layer1部分
        'layer1.{}.conv1.conv.weight': 'layer1.{}.conv1.weight',
        'layer1.{}.conv1.bn.weight': 'layer1.{}.bn1.weight',
        'layer1.{}.conv1.bn.bias': 'layer1.{}.bn1.bias',
        'layer1.{}.conv2.conv.weight': 'layer1.{}.conv2.weight',
        'layer1.{}.conv2.bn.weight': 'layer1.{}.bn2.weight',
        'layer1.{}.conv2.bn.bias': 'layer1.{}.bn2.bias',
        'layer1.{}.conv3.conv.weight': 'layer1.{}.conv3.weight',
        'layer1.{}.conv3.bn.weight': 'layer1.{}.bn3.weight',
        'layer1.{}.conv3.bn.bias': 'layer1.{}.bn3.bias',
        'layer1.{}.se_module.fc1.weight': 'layer1.{}.fc1.weight',
        'layer1.{}.se_module.fc1.bias': 'layer1.{}.fc1.bias',
        'layer1.{}.se_module.fc2.weight': 'layer1.{}.fc2.weight',
        'layer1.{}.se_module.fc2.bias': 'layer1.{}.fc2.bias',
        'layer1.{}.downsample.conv.weight': 'layer1.{}.downsample.0.weight',
        'layer1.{}.downsample.bn.weight': 'layer1.{}.downsample.1.weight',
        'layer1.{}.downsample.bn.bias': 'layer1.{}.downsample.1.bias',

        # layer2-4部分 (模式相同)
        'layer{}.{}.conv1.conv.weight': 'layer{}.{}.conv1.weight',
        'layer{}.{}.conv1.bn.weight': 'layer{}.{}.bn1.weight',
        'layer{}.{}.conv1.bn.bias': 'layer{}.{}.bn1.bias',
        'layer{}.{}.conv2.conv.weight': 'layer{}.{}.conv2.weight',
        'layer{}.{}.conv2.bn.weight': 'layer{}.{}.bn2.weight',
        'layer{}.{}.conv2.bn.bias': 'layer{}.{}.bn2.bias',
        'layer{}.{}.conv3.conv.weight': 'layer{}.{}.conv3.weight',
        'layer{}.{}.conv3.bn.weight': 'layer{}.{}.bn3.weight',
        'layer{}.{}.conv3.bn.bias': 'layer{}.{}.bn3.bias',
        'layer{}.{}.se_module.fc1.weight': 'layer{}.{}.fc1.weight',
        'layer{}.{}.se_module.fc1.bias': 'layer{}.{}.fc1.bias',
        'layer{}.{}.se_module.fc2.weight': 'layer{}.{}.fc2.weight',
        'layer{}.{}.se_module.fc2.bias': 'layer{}.{}.fc2.bias',
        'layer{}.{}.downsample.conv.weight': 'layer{}.{}.downsample.0.weight',
        'layer{}.{}.downsample.bn.weight': 'layer{}.{}.downsample.1.weight',
        'layer{}.{}.downsample.bn.bias': 'layer{}.{}.downsample.1.bias',

        # conv5部分
        'conv5.conv.weight': 'conv5.weight',
        'conv5.bn.weight': 'bn5.weight',
        'conv5.bn.bias': 'bn5.bias',
    }

    # 4. 构建映射后的预训练字典
    mapped_pretrained_dict = {}

    for pretrained_key, pretrained_value in pretrained_dict.items():
        # 尝试匹配所有可能的映射规则
        matched = False

        # 检查是否有数字需要替换 (如layer1.0)
        if any(f'layer{i}.' in pretrained_key for i in range(1, 5)):
            # 处理带数字的层
            for template, new_template in name_mapping.items():
                if '.' in template and any(f'layer{i}.' in template for i in range(1, 5)):
                    # 提取层号和块号
                    parts = pretrained_key.split('.')
                    layer_num = parts[0][-1]  # 如 'layer1' -> '1'
                    block_num = parts[1]  # 如 '0'

                    old_key = template.format(layer_num, block_num)
                    if pretrained_key.startswith(old_key):
                        new_key = new_template.format(layer_num, block_num)
                        # 保留后缀 (如.running_mean等)
                        suffix = pretrained_key[len(old_key):]
                        new_key += suffix
                        mapped_pretrained_dict[new_key] = pretrained_value
                        matched = True
                        break
        else:
            # 处理不带数字的键
            for old, new in name_mapping.items():
                if '.' not in old and pretrained_key.startswith(old):
                    new_key = pretrained_key.replace(old, new)
                    mapped_pretrained_dict[new_key] = pretrained_value
                    matched = True
                    break

        if not matched:
            print(f"Warning: No mapping found for pretrained key: {pretrained_key}")

    # 5. 过滤形状不匹配的参数
    filtered_pretrained_dict = {k: v for k, v in mapped_pretrained_dict.items()
                                if k in model_dict and model_dict[k].shape == v.shape}

    # 6. 打印加载信息
    print("\nSuccessfully loaded parameters:")
    for k in filtered_pretrained_dict:
        print(f"\t{k}")

    print("\nMissing parameters:")
    for k in model_dict:
        if k not in filtered_pretrained_dict and 'num_batches_tracked' not in k and 'running_' not in k:
            print(f"\t{k}")

    print("\nShape mismatch parameters (pretrained vs current model):")
    for k, v in mapped_pretrained_dict.items():
        if k in model_dict and model_dict[k].shape != v.shape:
            print(f"\t{k}: {v.shape} vs {model_dict[k].shape}")

    # 7. 更新模型参数字典
    model_dict.update(filtered_pretrained_dict)

    # 8. 加载到模型中
    model.load_state_dict(model_dict, strict=False)

    return model
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
    model = r3d_v2.generate_model(50)
    # model = X3D.generate_model(x3d_version='S')
    # model = r3d_zf.generate_model(50)

    # if params['pretrained'] is not None:
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
    #     print("Loaded keys:", pretrained_weights.keys())  # 查看文件包含的键
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

    # if params['pretrained'] is not None:
    #     print("load pretrain model")
    #     pretrained_weights = torch.load(params['pretrained'])
    #     print("Loaded keys:", pretrained_weights.keys())  # 调试用
    #
    #     # 根据SlowFast格式调整
    #     if 'model_state' in pretrained_weights:  # SlowFast官方权重
    #         state_dict = pretrained_weights['model_state']
    #     elif 'state_dict' in pretrained_weights:  # 其他框架权重
    #         state_dict = pretrained_weights['state_dict']
    #     else:  # 直接是状态字典
    #         state_dict = pretrained_weights
    #
    #     # 统一处理键名
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         # 去除多GPU前缀和不需要的键
    #         key = key.replace("module.", "")
    #         if not key.startswith(("fc", "logits")):  # 跳过分类头
    #             new_state_dict[key] = value
    #
    #     # 加载权重
    #     model.load_state_dict(new_state_dict, strict=False)
    #
    #     # 替换分类头（必须步骤）
    #     if isinstance(model, nn.DataParallel):
    #         model.module.fc2 = nn.Linear(2048, params['num_classes']).cuda()
    #     else:
    #         model.fc2 = nn.Linear(2048, params['num_classes']).cuda()


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

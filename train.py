# Universally Slimmable Networks and Improved Training Techniques
import argparse
import datetime
import functools
import glob
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import models
from datasets.dataset import get_train_loader, get_val_loader, ArchLoader
from utils.utils import (
    AvgrageMeter,
    CrossEntropyLossSoft,
    accuracy,
    create_exp_dir,
    reduce_mean,
    save_checkpoint,
    mixup_criterion,
    mixup_accuracy,
    load_checkpoint,
)

print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser("ResNet20-cifar100")

parser.add_argument(
    "--local_rank", type=int, default=0, help="local rank for distributed training"
)
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.05, help="init learning rate"
)  # 0.8
parser.add_argument("--num_workers", type=int, default=3, help="num of workers")
parser.add_argument(
    "--model-type",
    type=str,
    default="slimmable",
    help="type of model(sample masked dynamic independent slimmable original)",
)

parser.add_argument(
    "--finetune", action="store_true", help="finetune model with distill"
)
parser.add_argument(
    "--distill", action="store_true", help="finetune model with track_200.json"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="training dataset cifar10 or cifar100",
)

# hyper parameter
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
parser.add_argument("--cutout", type=float, default=0, help="cutout rate")
parser.add_argument("--mixup", action="store_true", help="use mixup or not")
parser.add_argument("--mixup_alpha", type=float, default=1.0, help="alpha in mixup")
parser.add_argument(
    "--resume", type=str, default="", help="path of resume weights. (model-latest.th)"
)
parser.add_argument("--autoaug", action="store_true", help="use autoaugmentation")

parser.add_argument("--report_freq", type=float, default=2, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=200, help="num of training epochs")

parser.add_argument("--classes", type=int, default=10, help="number of classes")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing")
parser.add_argument(
    "--config", help="configuration file", type=str, default="configs/meta.yml"
)
parser.add_argument(
    "--save_dir", type=str, help="save exp floder name", default="exp1_sandwich"
)
args = parser.parse_args()

# process argparse & yaml
if not args.config:
    opt = vars(args)
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(args)
    args = opt
else:  # yaml priority is higher than args
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    args = argparse.Namespace(**opt)

args.exp_name = (
    args.save_dir
    + "_"
    + datetime.datetime.now().strftime("%mM_%dD_%HH")
    + "_"
    + "{:04d}".format(random.randint(0, 1000))
)

# 文件处理
if not os.path.exists(os.path.join("exp", args.exp_name)):
    os.makedirs(os.path.join("exp", args.exp_name))


# 日志文件
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)

fh = logging.FileHandler(os.path.join("exp", args.exp_name, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(args)

# 配置文件
with open(os.path.join("exp", args.exp_name, "config.yml"), "w") as f:
    yaml.dump(args, f)

# Tensorboard文件
writer = SummaryWriter(
    "exp/%s/runs/%s-%05d"
    % (args.exp_name, time.strftime("%m-%d", time.localtime()), random.randint(0, 100))
)

create_exp_dir(os.path.join("exp", args.exp_name), scripts_to_save=glob.glob("*.py"))


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    args.device = torch.device("cuda")
    args.nprocs = num_gpus
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    best_val_acc = -1
    logging.info("gpu device = %d" % args.gpu)

    model = models.build_model(args.model_type, num_classes=args.classes)

    model = model.cuda(args.gpu)

    if num_gpus > 1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

        args.world_size = torch.distributed.get_world_size()
        args.batch_size = args.batch_size // args.world_size

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    soft_criterion = CrossEntropyLossSoft()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 200, eta_min=0.0005
    )
    if args.resume != "":
        load_checkpoint(args.resume, model, optimizer=optimizer)

    if num_gpus > 1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

        args.world_size = torch.distributed.get_world_size()
        args.batch_size = args.batch_size // args.world_size
    # Prepare data
    train_loader = get_train_loader(
        args.batch_size, args.num_workers, clss=args.dataset
    )
    # 原来跟train batch size一样，现在修改小一点 ，
    val_loader = get_val_loader(args.batch_size, args.num_workers, clss=args.dataset)

    archloader = ArchLoader("data/track_200.json")

    for epoch in range(args.epochs):
        train(
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            model,
            archloader,
            criterion,
            soft_criterion,
            args,
            args.seed,
            epoch,
            writer,
        )

        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        scheduler.step()
        if (epoch + 1) % args.report_freq == 0:
            if args.model_type in ["dynamic", "masked", "slimmable"]:
                top1_val, objs_val = infer(
                    train_loader, val_loader, model, criterion, archloader, args, epoch
                )
            else:
                top1_val, objs_val = valid(
                    train_loader, val_loader, model, criterion, archloader, args, epoch
                )
            if best_val_acc < top1_val:
                # update
                best_val_acc = top1_val
                save_checkpoint(
                    {
                        "state_dict": model.state_dict(),
                        "prec": top1_val,
                        "last_epoch": epoch,
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch,
                    args.exp_name,
                    tag="best_",
                )
            if args.local_rank == 0:
                # model
                if writer is not None:
                    writer.add_scalar("Val/loss", objs_val, epoch)
                    writer.add_scalar("Val/acc1", top1_val, epoch)

                save_checkpoint(
                    {
                        "state_dict": model.state_dict(),
                        "prec": top1_val,
                        "last_epoch": epoch,
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch,
                    args.exp_name,
                )
    logging.info("best top1 acc in validation datasets: %.2f" % (best_val_acc))


def train(
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    model,
    archloader,
    criterion,
    soft_criterion,
    args,
    seed,
    epoch,
    writer=None,
):
    losses_, top1_ = AvgrageMeter(), AvgrageMeter()
    inplace_distillation = True

    model.train()
    widest = [
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        32,
        32,
        32,
        32,
        32,
        32,
        64,
        64,
        64,
        64,
        64,
        64,
        64,
    ]
    narrowest = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    train_loader = tqdm(train_dataloader)
    train_loader.set_description(
        "[%s%04d/%04d %s%f]"
        % ("Epoch:", epoch + 1, args.epochs, "lr:", scheduler.get_last_lr()[0])
    )

    for step, (image, target) in enumerate(train_loader):
        n = image.size(0)
        image = Variable(image, requires_grad=False).cuda(args.gpu, non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(args.gpu, non_blocking=True)

        if args.model_type in ["dynamic", "masked", "slimmable"]:
            # sandwich rule
            candidate_list = []
            candidate_list += [
                archloader.generate_spos_like_batch().tolist() for i in range(6)
            ]
            candidate_list += [narrowest]

            # archloader.generate_niu_fair_batch(step)
            # 全模型来一遍
            soft_target = model(image, widest)
            soft_loss = criterion(soft_target, target)
            soft_loss.backward()

            # 采样几个子网来一遍
            for arc in candidate_list:
                logits = model(image, arc)
                # loss = soft_criterion(logits, soft_target.cuda(
                #     args.gpu, non_blocking=True))
                if inplace_distillation:
                    T = 1  # temperature in knowledge distillation 2 10 20

                    soft_target = torch.nn.functional.softmax(
                        soft_target / T, dim=1
                    ).detach()

                    loss = 0.5 * torch.mean(
                        soft_criterion(logits, soft_target)
                    ) + 0.5 * criterion(logits, target)
                else:
                    loss = criterion(logits, target)

                loss.backward()

            prec1, _ = accuracy(logits, target, topk=(1, 5))
            losses_.update(loss.data.item(), n)
            top1_.update(prec1.data.item(), n)

        else:
            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()

            prec1, _ = accuracy(logits, target, topk=(1, 5))
            losses_.update(loss.data.item(), n)
            top1_.update(prec1.data.item(), n)

        if torch.cuda.device_count() > 1:
            torch.distributed.barrier()
            loss = reduce_mean(loss, args.nprocs)
            prec1 = reduce_mean(prec1, args.nprocs)

        optimizer.step()
        optimizer.zero_grad()

        postfix = {
            "train_loss": "%.6f" % (losses_.avg),
            "train_acc1": "%.6f" % top1_.avg,
        }

        train_loader.set_postfix(log=postfix)

        if args.local_rank == 0 and step % 10 == 0 and writer is not None:
            writer.add_scalar(
                "Train/loss",
                losses_.avg,
                step + len(train_dataloader) * epoch * args.batch_size,
            )
            writer.add_scalar(
                "Train/acc1",
                top1_.avg,
                step + len(train_dataloader) * epoch * args.batch_size,
            )
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        if step % args.report_freq == 0:
            logging.info(
                "{} |=> Train loss = {} Train acc = {}".format(
                    now, losses_.avg, top1_.avg
                )
            )


def infer(train_loader, val_loader, model, criterion, archloader, args, epoch):
    objs_, top1_ = AvgrageMeter(), AvgrageMeter()

    model.eval()
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    # [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64]
    # .generate_width_to_narrow(epoch, args.epochs)
    fair_arc_list = archloader.generate_spos_like_batch().tolist()

    logging.info("{} |=> Test rng = {}".format(now, fair_arc_list))  # 只测试最后一个模型

    # if args.model_type == "dynamic":
    #     # BN calibration
    #     retrain_bn(model, train_loader, fair_arc_list, device=0)

    with torch.no_grad():
        for step, (image, target) in enumerate(val_loader):
            t0 = time.time()
            datatime = time.time() - t0
            image = Variable(image, requires_grad=False).cuda(
                args.local_rank, non_blocking=True
            )
            target = Variable(target, requires_grad=False).cuda(
                args.local_rank, non_blocking=True
            )

            logits = model(image, fair_arc_list)
            loss = criterion(logits, target)

            top1, _ = accuracy(logits, target, topk=(1, 5))

            if torch.cuda.device_count() > 1:
                torch.distributed.barrier()
                loss = reduce_mean(loss, args.nprocs)
                top1 = reduce_mean(top1, image.size(0))
                top5 = reduce_mean(top5, image.size(0))

            n = image.size(0)
            objs_.update(loss.data.item(), n)
            top1_.update(top1.data.item(), n)

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        logging.info(
            "{} |=> valid: step={}, loss={:.2f}, val_acc1={:.2f}, datatime={:.2f}".format(
                now, step, objs_.avg, top1_.avg, datatime
            )
        )

    return top1_.avg, objs_.avg


def valid(train_loader, val_loader, model, criterion, archloader, args, epoch):
    objs_, top1_ = AvgrageMeter(), AvgrageMeter()

    model.eval()
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    with torch.no_grad():
        for step, (image, target) in enumerate(val_loader):
            t0 = time.time()
            datatime = time.time() - t0
            image = Variable(image, requires_grad=False).cuda(
                args.local_rank, non_blocking=True
            )
            target = Variable(target, requires_grad=False).cuda(
                args.local_rank, non_blocking=True
            )

            logits = model(image)
            loss = criterion(logits, target)

            top1, _ = accuracy(logits, target, topk=(1, 5))

            if torch.cuda.device_count() > 1:
                torch.distributed.barrier()
                loss = reduce_mean(loss, args.nprocs)
                top1 = reduce_mean(top1, image.size(0))
                top5 = reduce_mean(top5, image.size(0))

            n = image.size(0)
            objs_.update(loss.data.item(), n)
            top1_.update(top1.data.item(), n)

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        logging.info(
            "{} |=> valid: step={}, loss={:.2f}, val_acc1={:.2f}, datatime={:.2f}".format(
                now, step, objs_.avg, top1_.avg, datatime
            )
        )

    return top1_.avg, objs_.avg


if __name__ == "__main__":
    main()

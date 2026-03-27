import matplotlib
matplotlib.use('Agg')

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import v2

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

import math

import pandas as pd
import numpy as np
import wandb
import os
from tqdm.auto import tqdm
import copy

from pathlib import Path

from sklearn.model_selection import train_test_split


class N_Times_Transform:
    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, img):
        results = [self.transform(img) for _ in range(self.n)]
        result_tensor = torch.stack(results, dim=0)
        return result_tensor


class EMA_Model:
    def __init__(self, model, decay, rampup_steps):
        self.decay = decay
        self.rampup_steps = rampup_steps
        self.steps = 0
        self.model = model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def compile(self):
        self.ema_model = torch.compile(self.ema_model)

    @torch.no_grad()
    def step(self):
        self.steps += 1
        decay = self.decay if self.steps > self.rampup_steps else 0.0

        model_state_dict = self.model.state_dict()
        for key, value in self.ema_model.state_dict().items():
            if value.dtype.is_floating_point:
                value.copy_(value * decay + model_state_dict[key] * (1 - decay))
            else:
                value.copy_(model_state_dict[key])

    def state_dict(self):
        result = {
            'decay': self.decay,
            'rampup_steps': self.rampup_steps,
            'steps': self.steps,
            'ema_model_state_dict': self.ema_model.state_dict(),
        }
        return result

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.rampup_steps = state_dict['rampup_steps']
        self.steps = state_dict['steps']
        self.ema_model.load_state_dict(state_dict['ema_model_state_dict'])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_str, transform, labels_path = None, images_df_path=None):
        self.is_train = False
        self.labels_df = None
        if labels_path is not None:
            self.is_train = True
            self.labels_df = pd.read_csv(labels_path)

        path = Path(path_str)
        self.N = len([x for x in path.iterdir()])

        if images_df_path is None:
            self.images = [torch.tensor([]) for _ in range(self.N)]
            for idx in tqdm(range(self.N), dynamic_ncols=True, desc='Loading images', leave=False):
                img_name = '0' * (5 - len(str(idx))) + str(idx)
                if self.is_train:
                    img_name = 'trainval_' + img_name
                else:
                    img_name = 'test_' + img_name

                img_path_str = path_str + '/' + img_name + '.jpg'
                image = torchvision.io.decode_image(img_path_str, mode=torchvision.io.ImageReadMode.RGB)
                self.images[idx] = image
        else:
            self.images = torch.load(images_df_path, weights_only=False)

        self.transform = transform

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        label = -1
        if self.is_train and self.labels_df is not None:
            label = self.labels_df.iloc[idx, 1]

        image = self.transform(self.images[idx])
        return (image, label)

def set_subset_transform(subset, transform):
    new_subset = copy.copy(subset)
    new_subset.dataset = copy.copy(new_subset.dataset)
    new_subset.dataset.transform = transform
    return new_subset

def get_data(batch_size, transform_train, transform_test, use_DDP, drop_last):
    torch.manual_seed(0)
    np.random.seed(0)

    trainvalset = Dataset(path_str=f'{data_dir_path}/trainval', transform=transform_train, labels_path=f'{data_dir_path}/labels.csv', images_df_path=f'{data_dir_path}/trainval_df/trainval_images.pt')

    testset = Dataset(path_str=f'{data_dir_path}/test', transform=transform_test, images_df_path=f'{data_dir_path}/test_df/test_images.pt')

    if test_size != 0:
        train_idx, valid_idx = train_test_split(np.arange(len(trainvalset)), test_size=test_size,
                                            shuffle=True, random_state=0)
        trainset = torch.utils.data.Subset(trainvalset, train_idx)
        valset = torch.utils.data.Subset(trainvalset, valid_idx)

        valset = set_subset_transform(valset, transform_test)
    else:
        trainset = trainvalset

    if use_DDP:
        train_sampler = DistributedSampler(trainset, shuffle=True)
        if test_size != 0:
            val_sampler = DistributedSampler(valset, shuffle=False)
        test_sampler = DistributedSampler(testset, shuffle=False)
    else:
        train_sampler = None
        if test_size != 0:
            val_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                               num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=drop_last)
    if test_size != 0:
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                                num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler,
                                              num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    if test_size == 0:
        return train_loader, None, test_loader, train_sampler, None, test_sampler
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


N_CLASSES = 200
class CNN_baseline(nn.Module):
    def __init__(self, blocks_num):
        super().__init__()

        self.t = blocks_num

        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
        )
        self.seq2 = nn.Conv2d(3, 32, 1)

        self.blocks_seq1 = nn.ModuleList()
        self.blocks_seq2 = nn.ModuleList()
        for i in range(self.t - 1):
            seq1 = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
            )
            seq2 = nn.Conv2d(32, 32, 1)
            self.blocks_seq1.append(seq1)
            self.blocks_seq2.append(seq2)

        self.fin1 = nn.AvgPool2d(10)
        self.fin2 = nn.Linear(32 * 4 * 4, N_CLASSES)

    def forward(self, x):
        out = self.seq1(x) + self.seq2(x)
        out = torch.nn.functional.relu(out)

        for i in range(self.t - 1):
            out = self.blocks_seq1[i](out) + self.blocks_seq2[i](out)
            out = torch.nn.functional.relu(out)

        out = self.fin1(out)
        out = self.fin2(torch.flatten(out, start_dim=1))

        return out


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks_seq1 = nn.ModuleList()
        self.blocks_seq2 = nn.ModuleList()

        channels = [64, 128, 256, 512]
        sizes = [3, 4, 6, 3]

        self.initial = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )

        # 40x40 -> 20x20 -> 10x10 -> 5x5

        for block_group in range(len(sizes)):
            for block in range(sizes[block_group]):
                if block == 0 and block_group != 0:
                    seq1 = nn.Sequential(
                        nn.Conv2d(channels[block_group - 1], channels[block_group], 3, padding=1, stride=2, bias=False),
                        nn.BatchNorm2d(channels[block_group]),
                        nn.ReLU(),
                        nn.Conv2d(channels[block_group], channels[block_group], 3, padding=1, bias=False),
                        nn.BatchNorm2d(channels[block_group]),
                    )
                    seq2 = nn.Conv2d(channels[block_group - 1], channels[block_group], 1, stride=2, bias=False)
                    self.blocks_seq1.append(seq1)
                    self.blocks_seq2.append(seq2)
                else:
                    seq1 = nn.Sequential(
                        nn.Conv2d(channels[block_group], channels[block_group], 3, padding=1, bias=False),
                        nn.BatchNorm2d(channels[block_group]),
                        nn.ReLU(),
                        nn.Conv2d(channels[block_group], channels[block_group], 3, padding=1, bias=False),
                        nn.BatchNorm2d(channels[block_group]),
                    )
                    seq2 = nn.Conv2d(channels[block_group], channels[block_group], 1, bias=False)
                    self.blocks_seq1.append(seq1)
                    self.blocks_seq2.append(seq2)

        self.fin1 = nn.AdaptiveAvgPool2d(1)
        self.fin2 = nn.Linear(channels[-1] * 1 * 1, N_CLASSES)

    def forward(self, x):
        x = self.initial(x)

        for i in range(len(self.blocks_seq1)):
            x = self.blocks_seq1[i](x) + self.blocks_seq2[i](x)
            x = torch.nn.functional.relu(x)

        x = self.fin1(x)
        x = self.fin2(torch.flatten(x, start_dim=1))

        return x


class WideResNet(nn.Module):
    def __init__(self, n, k, dropout_rate=0.0):
        super().__init__()

        self.residual_blocks_base = nn.ModuleList()
        self.residual_blocks_skip = nn.ModuleList()

        channels = [16, 16 * k, 32 * k, 64 * k]
        sizes = [n] * 3

        #       initial       #1            #2            #3          AvgPool
        # 40x40x3 -> 40x40x16 -> 40x40x16*k -> 20x20x32*k -> 10x10x64*k -> 1x1x64*k

        self.initial = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)

        for block_group in range(len(sizes)):
            for block in range(sizes[block_group]):
                if block == 0:
                    stride = 2
                    if block_group == 0:
                        stride = 1
                    residual = nn.Sequential(
                        nn.BatchNorm2d(channels[block_group]),
                        nn.ReLU(),
                        nn.Conv2d(channels[block_group], channels[block_group + 1], 3, padding=1, stride=stride, bias=False),
                        nn.BatchNorm2d(channels[block_group + 1]),
                        nn.ReLU(),
                    )
                    if dropout_rate != 0:
                        residual.append(nn.Dropout2d(p=dropout_rate))
                    residual.append(nn.Conv2d(channels[block_group + 1], channels[block_group + 1], 3, padding=1, bias=False))

                    skip = nn.Conv2d(channels[block_group], channels[block_group + 1], 1, stride=stride, bias=False)
                    self.residual_blocks_base.append(residual)
                    self.residual_blocks_skip.append(skip)
                else:
                    residual = nn.Sequential(
                        nn.BatchNorm2d(channels[block_group + 1]),
                        nn.ReLU(),
                        nn.Conv2d(channels[block_group + 1], channels[block_group + 1], 3, padding=1, bias=False),
                        nn.BatchNorm2d(channels[block_group + 1]),
                        nn.ReLU(),
                    )
                    if dropout_rate != 0:
                        residual.append(nn.Dropout2d(p=dropout_rate))
                    residual.append(nn.Conv2d(channels[block_group + 1], channels[block_group + 1], 3, padding=1, bias=False))

                    skip = nn.Identity()
                    self.residual_blocks_base.append(residual)
                    self.residual_blocks_skip.append(skip)

        self.fin1 = nn.Sequential(
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fin2 = nn.Linear(channels[-1] * 1 * 1, N_CLASSES)

    def forward(self, x):
        x = self.initial(x)

        for i in range(len(self.residual_blocks_base)):
            x = self.residual_blocks_base[i](x) + self.residual_blocks_skip[i](x)

        x = self.fin1(x)
        x = self.fin2(torch.flatten(x, start_dim=1))

        return x


class Warmup_plus_Cosine(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, base_lr, ramp_up_steps, T_max, eta_min, last_epoch=-1):
        self.base_lr = base_lr
        self.T_max = T_max - ramp_up_steps
        self.eta_min = eta_min
        self.ramp_up_steps = ramp_up_steps
        super(Warmup_plus_Cosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch % (self.T_max + self.ramp_up_steps)
        if current_step < self.ramp_up_steps:
            return [base_lr * (current_step + 1) / self.ramp_up_steps for base_lr in self.base_lrs]
        else:
            T_cur = current_step - self.ramp_up_steps
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(T_cur * math.pi / self.T_max)) / 2 for base_lr in self.base_lrs]


def create_saves(saves_path):
    if os.path.isdir(saves_path):
        if not use_DDP or local_rank == 0:
            print("=== Saves folder is already created ===")
    else:
        if not use_DDP or local_rank == 0:
            print(f"=== Folder '{saves_path}' not found ===")
            print(f"=== Creating folder '{saves_path}' ===")
        os.mkdir(saves_path)
        if not use_DDP or local_rank == 0:
            print(f"=== Folder '{saves_path}' is created ===")

def clean_saves(saves_path):
    create_saves(saves_path)
    for filename in os.listdir(saves_path):
        file_path = os.path.join(saves_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
    print("=== Saves folder cleaned ===")


def validate(model, val_loader, use_float16=False):
    loss_sum, acc_sum = 0.0, 0.0
    count = 0

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation', unit='batch', dynamic_ncols=True, leave=False):
            data = data.to(device)
            target = target.to(device)
            tta_was_used = len(data.shape) != 4

            if tta_was_used:
                predicted_sum = torch.zeros(data.shape[0], N_CLASSES, device=device)
                for i in range(data.shape[1]):
                    if use_float16:
                        with torch.autocast(device_type=device_str, dtype=torch.float16):
                            predicted_sum += model(data[:, i])
                    else:
                        predicted_sum += model(data[:, i])
                predicted = predicted_sum / data.shape[1]
            else:
                if use_float16:
                    with torch.autocast(device_type=device_str, dtype=torch.float16):
                        predicted = model(data)
                else:
                    predicted = model(data)

            loss = F.cross_entropy(predicted, target)

            loss_sum += loss.item() * len(target)

            pred_classes = torch.argmax(predicted, dim=1)
            acc = (pred_classes == target).float()

            acc_sum += acc.sum().item()
            count += len(target)

    return loss_sum, acc_sum, count

def train_epoch(model, optimizer, train_loader, scheduler=None, train_batch_transform=None, ema_model=None, scaler=None):
    loss_sum, acc_sum = 0.0, 0.0
    count = 0

    model.train()

    for data, target in tqdm(train_loader, desc='Training Epoch', unit='batch', dynamic_ncols=True, leave=False):
        data = data.to(device)
        target = target.to(device)

        true_labels = target
        if train_batch_transform is not None:
            data, target = train_batch_transform(data, target)

        if scaler is None:
            predicted = model(data)
            loss = F.cross_entropy(predicted, target)

            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type=device_str, dtype=torch.float16):
                predicted = model(data)
                loss = F.cross_entropy(predicted, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad()

        loss_sum += loss.item() * len(true_labels)

        pred_classes = torch.argmax(predicted, dim=1)
        acc = (pred_classes == true_labels).float()
        acc_sum += acc.sum().item()

        count += len(true_labels)

        if ema_model is not None:
            ema_model.step()

        if scheduler is not None:
            scheduler.step()

    return loss_sum, acc_sum, count


def get_checkpoint(epoch, best_epoch, best_val_acc, best_ema_epoch, best_ema_val_acc, model, optimizer, scheduler=None, ema_model=None, swa_model=None, swa_scheduler=None, scaler=None):
    checkpoint = {
        'epoch': epoch,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'best_ema_epoch': best_ema_epoch,
        'best_ema_val_acc': best_ema_val_acc,
        'model_state_dict': model.state_dict() if not use_DDP else model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'ema_model_state_dict': ema_model.state_dict() if ema_model else None,
        'swa_model_state_dict': swa_model.state_dict() if swa_model else None,
        'swa_scheduler_state_dict': swa_scheduler.state_dict() if swa_scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
    }
    return checkpoint


def train_and_validate(model, optimizer, n_epochs, train_loader, val_loader, scheduler=None, train_batch_transform=None,
                       ema_model=None, swa_model=None, swa_start=None, swa_scheduler=None, checkpoint=None, use_float16=False,
                       use_DDP=None, train_sampler=None, val_sampler=None):

    scaler = None
    if use_float16:
        scaler = torch.GradScaler(init_scale=2**8)
        if not use_DDP or local_rank == 0:
            print('=== Using Mixed Precision ===')
    else:
        if not use_DDP or local_rank == 0:
            print('=== Using Standart Precision ===')

    start_epoch = 0
    if checkpoint is not None:
        if not use_DDP or local_rank == 0:
            print('=== Loading model... ===')
        start_epoch = checkpoint['epoch'] + 1
        if swa_start is None:
            if use_DDP:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        if swa_start is None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if swa_model is not None and 'swa_model_state_dict' in checkpoint and checkpoint['swa_model_state_dict'] is not None:
            swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
        if swa_scheduler is not None and 'swa_scheduler_state_dict' in checkpoint and checkpoint['swa_scheduler_state_dict'] is not None:
            swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state_dict'])
        if ema_model is not None:
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        if not use_DDP or local_rank == 0:
            print('=== Model loaded successfully ===')

    best_ema_val_acc = 0
    best_ema_epoch = start_epoch - 1

    best_val_acc = 0
    best_epoch = start_epoch - 1

    if not use_DDP or local_rank == 0:
        if checkpoint is None:
            wandb.init(
                project="DL1 Big Homework 1",
                config={
                    "epochs": n_epochs,
                    "batch_size": train_loader.batch_size,
                    "base_lr": base_lr,
                    "optimizer": optimizer.__class__.__name__
                },
                group=run_group,
                name=run_name,
                mode=wandb_mode,
            )
            clean_saves(saves_path)
        else:
            wandb.init(
                project="DL1 Big Homework 1",
                config={
                    "epochs": n_epochs,
                    "batch_size": train_loader.batch_size,
                    "base_lr": base_lr,
                    "optimizer": optimizer.__class__.__name__
                },
                id=run_id,
                group=run_group,
                name=run_name,
                resume="allow",
                mode=wandb_mode,
            )

            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']
            if 'best_ema_epoch' in checkpoint:
                best_ema_epoch = checkpoint['best_ema_epoch']

            if 'best_val_acc' in checkpoint:
                best_val_acc = checkpoint['best_val_acc']
            if 'best_ema_val_acc' in checkpoint:
                best_ema_val_acc = checkpoint['best_ema_val_acc']

            create_saves(saves_path)

    use_validation = val_loader is not None and len(val_loader) != 0

    for epoch in tqdm(range(start_epoch, n_epochs), desc='Total Training', unit='epoch', initial=start_epoch, total=n_epochs, dynamic_ncols=True, leave=True):
        if use_DDP and train_sampler is not None and val_sampler is not None:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        if swa_start is not None and epoch >= swa_start:
            train_loss_sum, train_acc_sum, train_count = train_epoch(model, optimizer, train_loader, ema_model=ema_model,
                                                train_batch_transform=train_batch_transform, scaler=scaler)
        else:
            train_loss_sum, train_acc_sum, train_count = train_epoch(model, optimizer, train_loader, ema_model=ema_model, scheduler=scheduler,
                                                train_batch_transform=train_batch_transform, scaler=scaler)

        if use_validation:
            val_loss_sum, val_acc_sum, val_count = validate(model, val_loader, use_float16=use_float16)
        else:
            val_loss_sum, val_acc_sum, val_count = -1, -1, -1

        ema_val_loss_sum, ema_val_acc_sum, ema_val_count, ema_val_loss, ema_val_acc = -1, -1, -1, -1, -1
        train_loss, train_acc, val_loss, val_acc = -1, -1, -1, -1

        if ema_model is not None and use_validation:
            ema_val_loss_sum, ema_val_acc_sum, ema_val_count = validate(ema_model.ema_model, val_loader, use_float16=use_float16)

        if use_DDP:
            if ema_model is not None:
                t = torch.tensor([train_loss_sum, train_acc_sum, train_count, val_loss_sum, val_acc_sum, val_count, ema_val_loss_sum, ema_val_acc_sum, ema_val_count], device=device)
            else:
                t = torch.tensor([train_loss_sum, train_acc_sum, train_count, val_loss_sum, val_acc_sum, val_count], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

            if local_rank == 0:
                train_loss = t[0].item() / t[2].item()
                train_acc = t[1].item() / t[2].item()
                val_loss = t[3].item() / t[5].item()
                val_acc = t[4].item() / t[5].item()
                if ema_model is not None:
                    ema_val_loss = t[6].item() / t[8].item()
                    ema_val_acc = t[7].item() / t[8].item()

                tqdm.write(f'Epoch: {epoch + 1}/{n_epochs}')
                tqdm.write(f' Train Loss, Train Acccuracy: {train_loss}, {train_acc}')
                if use_validation:
                    tqdm.write(f' Validation Loss, Validation Acccuracy: {val_loss}, {val_acc}')
                    if ema_model is not None:
                        tqdm.write(f' EMA Val Loss, EMA Val Acccuracy: {ema_val_loss}, {ema_val_acc}')
        elif not use_DDP:
            train_loss = train_loss_sum / train_count
            train_acc = train_acc_sum / train_count
            val_loss = val_loss_sum / val_count
            val_acc = val_acc_sum / val_count
            if ema_model is not None:
                ema_val_loss = ema_val_loss_sum / ema_val_count
                ema_val_acc = ema_val_acc_sum / ema_val_count

            tqdm.write(f'Epoch: {epoch + 1}/{n_epochs}')
            tqdm.write(f' Train Loss, Train Acccuracy: {train_loss}, {train_acc}')
            if use_validation:
                tqdm.write(f' Validation Loss, Validation Acccuracy: {val_loss}, {val_acc}')
                if ema_model is not None:
                    tqdm.write(f' EMA Val Loss, EMA Val Acccuracy: {ema_val_loss}, {ema_val_acc}')

        if not use_DDP or local_rank == 0:
            checkpoint = get_checkpoint(epoch, best_epoch, best_val_acc, best_ema_epoch, best_ema_val_acc, model, optimizer, scheduler, ema_model, swa_model, swa_scheduler, scaler)
            torch.save(checkpoint, f"{saves_path}/last_save.pth")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(checkpoint, f"{saves_path}/best_save.pth")
            if ema_model is not None and ema_val_acc > best_ema_val_acc:
                best_ema_val_acc = ema_val_acc
                best_ema_epoch = epoch
                torch.save(checkpoint, f"{saves_path}/best_ema_save.pth")
            if epoch % 10 == 0:
                torch.save(checkpoint, f"{saves_path}/regular_save_{epoch}.pth")

        if swa_start is not None and swa_model is not None and epoch >= swa_start:
            if use_DDP:
                swa_model.update_parameters(model.module)
            else:
                swa_model.update_parameters(model)
        if swa_start is not None and swa_scheduler is not None and epoch >= swa_start:
            swa_scheduler.step()

        grad_norm = nn.utils.get_total_norm(model.parameters())

        if not use_DDP or local_rank == 0:
            info_to_log = {
                "epoch": epoch,
                "train loss": train_loss,
                "train accuracy": train_acc,
                "validation loss": val_loss,
                "validation accuracy": val_acc,
                "last epoch with impovement": best_epoch,
                "learning rate": optimizer.param_groups[0]['lr'],
                "gradient norm": grad_norm.item()
            }
            if ema_model is not None:
                info_to_log["last epoch with EMA impovement"] = best_ema_epoch
                info_to_log["EMA validation loss"] = ema_val_loss
                info_to_log["EMA validation accuracy"] = ema_val_acc

            wandb.log(info_to_log)

    if swa_model is not None:
        update_bn(train_loader, swa_model, device)

    if not use_DDP or local_rank == 0:
        checkpoint = get_checkpoint(n_epochs-1, best_epoch, best_val_acc, best_ema_epoch, best_ema_val_acc, model, optimizer, scheduler, ema_model, swa_model, swa_scheduler, scaler)
        torch.save(checkpoint, f"{saves_path}/final_save_{n_epochs-1}.pth")
        wandb.finish()


def test_predict(model, test_loader):
    classes = []
    model.eval()

    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc='Testing', unit='batch', dynamic_ncols=True, leave=True):
            data = data.to(device)
            tta_was_used = len(data.shape) != 4

            if tta_was_used:
                predicted_sum = torch.zeros(data.shape[0], N_CLASSES, device=device)
                for i in range(data.shape[1]):
                    if use_float16:
                        with torch.autocast(device_type=device_str, dtype=torch.float16):
                            pred = model(data[:, i])
                    else:
                        pred = model(data[:, i])
                    predicted_sum += pred
                predicted = predicted_sum
            else:
                data = data.to(device)
                if use_float16:
                    with torch.autocast(device_type=device_str, dtype=torch.float16):
                        predicted = model(data)
                else:
                    predicted = model(data)

            pred_classes = torch.argmax(predicted, dim=1)
            for x in pred_classes.tolist():
                classes.append(x)

    return classes


def test_and_write_to_csv(model, test_loader):
    print('=== Submitting to csv file... ===')

    classes = test_predict(model, test_loader)

    ids = []
    for idx in range(len(classes)):
        img_id = 'test_' + '0' * (5 - len(str(idx))) + str(idx) + '.jpg'
        ids.append(img_id)

    classes = pd.Series(classes)
    ids = pd.Series(ids)
    result = pd.DataFrame(data={'Id': ids, 'Category': classes})

    path = f'{submission_dir}/submission.csv'
    result.to_csv(path, index=False)

    print('=== Submission file has been successfully saved ===')

def load_checkpoint(checkpoint_name):
    if len(checkpoint_name) >= 4 and checkpoint_name[-4:] == '.pth':
        checkpoint_name = checkpoint_name[:-4]
    
    if not use_DDP or local_rank == 0:
        print(f"=== Loading checkpoint '{checkpoint_dir}/{checkpoint_name}.pth'... ===")

    try:
        checkpoint = torch.load(f'{checkpoint_dir}/{checkpoint_name}.pth', weights_only=False)
    except Exception as e:
        print (f'=== ERROR Loading failed: {e} ===')
        exit(1)

    if not use_DDP or local_rank == 0:
        print("=== Checkpoint loaded successfully ===")
    return checkpoint

def load_model(model, checkpoint_name, use_ema=False, use_swa=False):
    checkpoint = load_checkpoint(checkpoint_name)
    if checkpoint is None:
        return

    if use_ema:
        ema_model = EMA_Model(model, 0, 0)
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        model = ema_model.ema_model
    elif use_swa:
        swa_model = AveragedModel(model)
        swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
        model = swa_model
    elif use_DDP:        
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_checkpoint_and_submit(model, checkpoint_name, use_ema=False, use_swa=False):
    model = load_model(model, checkpoint_name, use_swa=use_swa)
    if use_swa:
        print('=== Updating batch norm... ===')
        update_bn(train_loader, model, device)
        print('=== Batch norm updated ===')
    if use_compile:
        print('=== Compiling... ===')
        if use_compile:
            model = torch.compile(model)
        print('=== Compiled ===')

    test_and_write_to_csv(model, test_loader)


def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


data_dir_path = './bhw1'
saves_path = './saves'
checkpoint_dir = './previous_runs_best_saves'
submission_dir = '.'

test_size = 0.1

submit = False
checkpoint_name = 'WideResNet-28-10'

use_float16 = True
use_compile = True
drop_last = True
use_DDP = True

n_epochs = 200
batch_size = 64
base_lr = 0.05
min_lr = 0
weight_decay = 5e-4
# label_smoothing = 0  # Not Implemented

ema_decay = 0.999
use_swa = True

wandb_mode = 'offline'
run_id = None
run_name = None
run_group = 'Ungrouped runs'
previous_best_val_acc = 0
previous_best_epoch = None

if torch.cuda.device_count() <= 1 or not torch.cuda.is_available():
    use_DDP = False


if use_DDP:
    local_rank = setup()
    device = torch.device(f'cuda:{local_rank}')
    device_str = f'cuda:{local_rank}'
    print(f'=== Using cuda device {local_rank} ===')
else:
    local_rank = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'=== Using {device} device ===')

if not use_DDP or local_rank == 0:
    print("=== Loading data... ===")


transform = v2.Compose([
    v2.RandomCrop(40, padding=4),
    v2.RandomHorizontalFlip(p=0.5),
    v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5691, 0.5447, 0.4933), std=(0.2386, 0.2335, 0.2516)),
])

base_test_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5691, 0.5447, 0.4933), std=(0.2386, 0.2335, 0.2516)),
])

tta_transform = N_Times_Transform(
    v2.Compose([
        v2.RandomCrop(40, padding=4),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5691, 0.5447, 0.4933), std=(0.2386, 0.2335, 0.2516)),
    ]),
    n=50
)

train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = \
    get_data(batch_size=batch_size, transform_train=transform, transform_test=base_test_transform, use_DDP=use_DDP, drop_last=drop_last)

if not use_DDP or local_rank == 0:
    print("=== Data loaded successfully ===")


# model = ResNet34().to(device)
model = WideResNet(4, 10, dropout_rate=0.3).to(device) # WideResNet-28-10
# model = WideResNet(4, 8).to(device) # WideResNet-28-8

# ema_model = EMA_Model(model, ema_decay, n_epochs * len(train_loader) * 0.05)
ema_model = None

if use_DDP:
    if local_rank == 0:
        print(f"=== Using {torch.cuda.device_count()} GPUs ===")
    model = DDP(model, device_ids=[local_rank])
else:
    print(f"=== Using 1 GPU ===")

if use_compile:
    model = torch.compile(model)

if submit:
    load_checkpoint_and_submit(model, checkpoint_name, use_swa=use_swa)
    exit(0)

checkpoint = None
if checkpoint_name is not None:
    checkpoint = load_checkpoint(checkpoint_name)

if checkpoint:
    if use_DDP:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

decay, no_decay = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "bn" in name or "bias" in name or param.dim() == 1:
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = optim.SGD([
    {"params": decay, "weight_decay": weight_decay},
    {"params": no_decay, "weight_decay": 0.0}],
    lr=base_lr,
    momentum=0.9,
    nesterov=True
)

if checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = Warmup_plus_Cosine(optimizer, base_lr=base_lr, ramp_up_steps=10*len(train_loader), T_max=n_epochs*len(train_loader), eta_min=min_lr)


if use_swa:
    if use_DDP:
        swa_model = AveragedModel(model.module, device)
    else:
        swa_model = AveragedModel(model, device)
    swa_start = 161
    swa_lr = 0.005
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=swa_lr,
        anneal_strategy="cos",
        anneal_epochs=0
    )
else:
    swa_model = None
    swa_start = None
    swa_scheduler = None


train_batch_transform = v2.RandomChoice([v2.MixUp(num_classes=N_CLASSES), v2.CutMix(num_classes=N_CLASSES)])
train_and_validate(model, optimizer, n_epochs, train_loader, val_loader, scheduler, train_batch_transform, checkpoint=checkpoint,
                   ema_model=ema_model, swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler, use_float16=use_float16,
                   use_DDP=use_DDP, train_sampler=train_sampler, val_sampler=val_sampler)

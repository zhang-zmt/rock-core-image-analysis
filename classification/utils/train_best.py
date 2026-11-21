#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_resnet.py  (改进版)

• 若传 --data_dir，仅训练该目录（保持旧行为）
• 若传 --data_root（包含 binary / ore_subclass / nonore_subclass），
  按顺序训练 3 套模型，权重分别保存为:
      best_binary.pth / best_ore_sub.pth / best_nonore_sub.pth

python utils/train_best.py --data_root jb_train --arch resnet34 --min_scale 0.85 --brightness 0.35 --brightness_prob 0.5 --batch_size 4 --use_sampler --num_workers 2 --pin_memory --amp --out_dir jb_out
"""

import os, time, argparse, random
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms, models

# ---------- Dataset for manifest mode ----------
class ManifestDataset(Dataset):
    def __init__(self, manifest_path, label2idx, transform):
        self.items = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                path, lab = line.split('\t')
                if lab not in label2idx: continue
                self.items.append((path, label2idx[lab]))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = datasets.folder.default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, y

# ---------- CLI ----------
def get_args():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--data_dir',  help='单一数据集目录 (保持旧行为)')
    g.add_argument('--data_root', help='包含 binary/ ore_subclass/ nonore_subclass/ 的根目录')
    ap.add_argument('--save_path', default=None, help = '仅在 --data_dir 模式下生效，用于指定保存的权重文件名')
    ap.add_argument('--arch',       default='resnet101', choices=['resnet18','resnet34','resnet50','resnet101'])
    ap.add_argument('--epochs',     type=int, default=1)
    ap.add_argument('--binary_epochs', type=int, default=None,
                    help='epochs for binary stage (default=epochs)')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr',         type=float, default=1e-3)
    ap.add_argument('--momentum',   type=float, default=0.9)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--device',     default='cuda')
    ap.add_argument('--img_size',   type=int, default=224)
    ap.add_argument('--min_scale',  type=float, default=0.6,
                    help='min scale for RandomResizedCrop')
    ap.add_argument('--brightness', type=float, default=0.3,
                    help='ColorJitter brightness factor (0 to disable)')
    ap.add_argument('--brightness_prob', type=float, default=0.5,
                    help='probability to apply brightness jitter')
    ap.add_argument('--rotation',   type=float, default=0.0,
                    help='RandomRotation degrees (0 to disable)')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--pin_memory', action='store_true')
    ap.add_argument('--use_sampler', action='store_true',
                    help='use WeightedRandomSampler for class imbalance')
    ap.add_argument('--amp', action='store_true',
                    help='use mixed precision on CUDA')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default=None,
                    help='optional directory for saving weights')
    return ap.parse_args()

# ---------- 数据 ----------
def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_transforms(img_size, min_scale, brightness, brightness_prob, rotation):
    mean = [0.485,0.456,0.406]
    std = [0.229,0.224,0.225]
    min_scale = max(0.1, min(1.0, min_scale))
    train_aug = [
        transforms.RandomResizedCrop(img_size, scale=(min_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if rotation and rotation > 0:
        train_aug.append(transforms.RandomRotation(rotation))
    if brightness and brightness > 0:
        p = min(max(brightness_prob, 0.0), 1.0)
        jitter = transforms.ColorJitter(brightness=brightness)
        train_aug.append(transforms.RandomApply([jitter], p=p))
    train_aug += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    resize_size = int(round(img_size * 1.14))
    eval_aug = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(train_aug), transforms.Compose(eval_aug)

def build_loaders(root, bs, img_size, min_scale, brightness, brightness_prob, rotation,
                  num_workers, pin_memory, use_sampler):
    train_tf, eval_tf = build_transforms(img_size, min_scale, brightness, brightness_prob, rotation)
    tfm = {'train': train_tf, 'val': eval_tf, 'test': eval_tf}

    def has_manifest(dir_path):
        return os.path.isfile(os.path.join(dir_path, 'train.txt'))

    if has_manifest(root):
        # build label map from train manifest
        train_manifest = os.path.join(root, 'train.txt')
        labels = []
        with open(train_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                _, lab = line.strip().split('\t')
                labels.append(lab)
        classes = sorted(list(set(labels)))
        label2idx = {c:i for i,c in enumerate(classes)}
        sets = {
            split: ManifestDataset(os.path.join(root, f'{split}.txt'), label2idx, tfm[split])
            for split in ['train','val','test']
        }
    else:
        sets = {s: datasets.ImageFolder(os.path.join(root, s), tfm[s])
                for s in ['train','val','test']}
        classes = sets['train'].classes

    if use_sampler:
        if hasattr(sets['train'], 'samples'):
            targets = [y for _, y in sets['train'].samples]
        else:
            targets = [y for _, y in sets['train']]
        class_count = torch.bincount(torch.tensor(targets))
        weights = 1.0 / (class_count.float() + 1e-6)
        sample_weights = [weights[t] for t in targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(sets['train'], batch_size=bs, sampler=sampler,
                                  num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(sets['train'], batch_size=bs, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
    loaders = {
        'train': train_loader,
        'val': DataLoader(sets['val'], batch_size=bs, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory),
        'test': DataLoader(sets['test'], batch_size=bs, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory),
    }
    return loaders, classes

# ---------- 训练/评估 ----------
def run_epoch(model, loader, crit, opt, dev, train=True, scaler=None, use_amp=False):
    model.train() if train else model.eval()
    tot_loss = corr = 0
    with torch.set_grad_enabled(train):
        for X,y in tqdm(loader, leave=False, ncols=80,
                        desc='Train' if train else 'Eval'):
            X,y = X.to(dev), y.to(dev)
            if train:
                opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(X)
                loss = crit(out, y)
            if train:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            tot_loss += loss.item()*X.size(0)
            corr += (out.argmax(1)==y).sum().item()
    n = len(loader.dataset)
    return tot_loss/n, corr/n

def train_one(dir_path:Path, save_path:Path, args, device, scaler, use_amp,
              epochs=None, task_name=None):
    loaders, classes = build_loaders(
        dir_path,
        args.batch_size,
        args.img_size,
        args.min_scale,
        args.brightness,
        args.brightness_prob,
        args.rotation,
        args.num_workers,
        args.pin_memory,
        args.use_sampler,
    )
    n_cls = len(classes)
    num_epochs = epochs or args.epochs
    task_label = task_name if task_name else dir_path.name
    n_train = len(loaders['train'].dataset)
    n_val = len(loaders['val'].dataset)
    print(f'\nTraining: {task_label}  classes={n_cls} -> {classes}')
    print(f'  train_samples={n_train}  val_samples={n_val}  batch_size={args.batch_size}  epochs={num_epochs}')

    model = getattr(models, args.arch)(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, n_cls)
    model.to(device)

    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    best, best_ep = 0, 0
    for ep in range(1, num_epochs+1):
        t0 = time.time()
        lr_now = opt.param_groups[0]['lr']
        tr_l, tr_a = run_epoch(model, loaders['train'], crit, opt, device,
                               True, scaler, use_amp)
        va_l, va_a = run_epoch(model, loaders['val'],   crit, opt, device,
                               False, None, use_amp)
        sched.step()
        minutes = (time.time() - t0)/60
        print(f'[{task_label}] Ep{ep:02d}/{num_epochs} bs={args.batch_size} lr={lr_now:.4g} '
              f'Train L={tr_l:.4f} A={tr_a:.3f} | Val L={va_l:.4f} A={va_a:.3f} '
              f'({minutes:.1f} min)')
        if va_a > best:
            best, best_ep = va_a, ep
            torch.save(model.state_dict(), str(save_path))
            print(f'  -> Save best to {save_path}')

    # test
    model.load_state_dict(torch.load(str(save_path), map_location=device))
    te_l, te_a = run_epoch(model, loaders['test'], crit, opt, device,
                           False, None, use_amp)
    print(f'Best@Ep{best_ep}  TestAcc={te_a:.3f}')

# ---------- 主 ----------
def main():
    args = get_args()
    dev  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    use_amp = bool(args.amp and dev.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start = time.time()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    bin_epochs = args.binary_epochs or args.epochs

    if args.data_dir:       # 旧逻辑：只训练一个
        save_name = args.save_path if args.save_path else 'best.pth'
        save_path = Path(save_name)
        if out_dir:
            save_path = out_dir / save_path.name
        train_one(Path(args.data_dir), save_path, args, dev, scaler, use_amp,
                  epochs=args.epochs, task_name=Path(args.data_dir).name)
    else:                   # 新逻辑：顺序训练三套
        root = Path(args.data_root)
        todo = [
            ('binary',           'best_binary.pth'),
            ('ore_subclass',     'best_ore_sub.pth'),
            ('nonore_subclass',  'best_nonore_sub.pth')
        ]
        for sub, save in todo:
            train_dir = root / sub
            if not train_dir.exists():
                print(f'⚠ 目录 {train_dir} 不存在，跳过')
                continue
            save_path = Path(save)
            if out_dir:
                save_path = out_dir / save_path.name
            stage_epochs = bin_epochs if sub == 'binary' else args.epochs
            train_one(train_dir, save_path, args, dev, scaler, use_amp,
                      epochs=stage_epochs, task_name=sub)

    print(f'\n总耗时 {(time.time()-start)/60:.1f} 分钟')

if __name__ == '__main__':
    main()

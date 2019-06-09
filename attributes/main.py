import argparse
from collections import defaultdict, deque
import json
import hashlib
from pathlib import Path
from typing import List, Tuple

import json_log_plots
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision import transforms
import tqdm


N_ATTRIBUTES = 92
N_CATEGORIES = 13
DATA_ROOT = Path(__file__).parent.parent / 'data'


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('run_path')
    arg('--min-cls-count', type=int, default=20)
    arg('--workers', type=int, default=4)
    arg('--batch-size', type=int, default=32)
    arg('--epochs', type=int, default=100)
    arg('--lr', type=float, default=1e-5)
    arg('--size', default=(224, 336),
        type=lambda x: tuple(map(int, x.split('x'))),
        help='width, height, e.g. 224x336')
    arg('--action', choices=['train', 'valid', 'test'], default='train')
    args = parser.parse_args()

    run_path = Path(args.run_path)
    run_path.mkdir(exist_ok=True, parents=True)
    folds = json.loads((DATA_ROOT / 'folds-518.json').read_text())
    df = pd.read_pickle(DATA_ROOT / 'train.pkl')
    all_image_ids = [image_id for fold in folds for image_id in fold]
    df = df[df['ImageId'].isin(all_image_ids)]
    df['attributes'] = df['ClassId'].apply(lambda x: '_'.join(x.split('_')[1:]))
    df['category'] = df['ClassId'].apply(lambda x: x.split('_')[0])
    # drop annotation errors
    df = df[(df['attributes'] != '') &
            (df['category'].astype(int) < N_CATEGORIES)]

    value_counts = df['attributes'].value_counts()
    attrs_classes = sorted(
        value_counts[value_counts > args.min_cls_count].index.values)

    model_path = run_path / 'model.pth'
    if args.action == 'valid':
        state = torch.load(model_path)
        attrs_classes = state['attrs_classes']
        model = Model(attrs_classes)
        model.load_state_dict(state['state_dict'])
    else:
        model = Model(attrs_classes)

    valid_image_ids = folds[0]
    valid_index = df['ImageId'].isin(valid_image_ids)
    df_train, df_valid = df[~valid_index], df[valid_index]
    print(f'{len(df):,} items total, {len(df_train):,} in train, '
          f'{len(df_valid):,} in valid. {len(attrs_classes)} attrs classes')
    train_dataset = FashionDataset(
        df=df_train,
        attrs_classes=attrs_classes,
        image_root=DATA_ROOT / 'train',
        training=True,
        size=args.size,
        debug=True,
    )
    valid_dataset = FashionDataset(
        df=df_valid,
        attrs_classes=attrs_classes,
        image_root=DATA_ROOT / 'train',
        size=args.size,
        training=False,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.workers,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        num_workers=args.workers,
        shuffle=False,
        batch_size=args.batch_size,
    )

    device = torch.device('cuda')
    optimizer = None
    step = 0
    log_step = 50
    running_losses = defaultdict(lambda: deque(maxlen=log_step))

    def to_device(x, y):
        x = x.to(device)
        y = tuple(v.to(device) for v in y)
        return x, y

    def train_step(xy):
        x, y = to_device(*xy)
        optimizer.zero_grad()
        y_pred = model(x)
        loss, loss_dict = get_loss(y_pred, y)
        (loss * args.batch_size).backward()
        optimizer.step()
        for k, v in loss_dict.items():
            running_losses[k].append(float(v.item()))

    def run_validation():
        model.eval()
        metrics = defaultdict(list)
        with torch.no_grad():
            for xy in tqdm.tqdm(valid_loader, desc='validation'):
                x, y = to_device(*xy)
                y_pred = model(x)
                _, loss_dict = get_loss(y_pred, y)
                for k, v in loss_dict.items():
                    metrics[f'valid_{k}'].append(float(v.item()))
                _, pred_attrs_cls, pred_category = y_pred
                _, true_attrs_cls, true_category = y
                metrics['category_acc'].extend(
                    v.item() for v in (
                        pred_category.argmax(dim=1) == true_category))
                metrics['attrs_cls_acc'].extend(
                    v.item() for v in (
                        pred_attrs_cls.argmax(dim=1) == true_attrs_cls))
                other_cls = len(attrs_classes)
                metrics['attrs_cls_nonother_ratio'].extend(
                    pred.item() == true.item() for pred, true in zip(
                        pred_attrs_cls.argmax(dim=1), true_attrs_cls)
                    if pred.item() != other_cls)
                metrics['attrs_cls_nonother_acc'].extend(
                    v.item() for v in (
                        pred_attrs_cls.argmax(dim=1) == other_cls))
        model.train()
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        return metrics

    model.to(device)

    if args.action == 'valid':
        valid_metrics = run_validation()
        for k, v in valid_metrics.items():
            print(f'{k:<30} {v:.3f}')
        return

    for epoch in tqdm.trange(args.epochs, desc='epochs'):
        if epoch == 0:
            optimizer = Adam(model.head.parameters(), lr=args.lr)
        elif epoch == 1:
            optimizer = Adam(model.parameters(), lr=args.lr)
        for i, batch in enumerate(tqdm.tqdm(train_loader), 1):
            train_step(batch)
            step += 1
            if i % log_step == 0:
                json_log_plots.write_event(
                    run_path, step=args.batch_size * step,
                    **{k: np.mean(v) for k, v in running_losses.items()})
        torch.save({
            'state_dict': model.state_dict(),
            'attrs_classes': attrs_classes,
        }, model_path)
        valid_metrics = run_validation()
        json_log_plots.write_event(
            run_path, step=args.batch_size * step, **valid_metrics)
        # TODO validation


class Model(nn.Module):
    def __init__(self, attrs_classes: List[str]):
        super().__init__()
        self.attrs_classes = attrs_classes
        self.base = resnet50(pretrained=True)
        base_features = self.base.fc.in_features
        del self.base.fc
        self.head = Head(base_features, attrs_classes)

    def forward(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)

        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)

        x = base.avgpool(x)
        x = x.reshape(x.size(0), -1)

        return self.head(x)


class Head(nn.Module):
    def __init__(self, base_features, attrs_classes):
        super().__init__()
        # TODO check if an extra fc layer would help
        self.fc_attrs = nn.Linear(base_features, N_ATTRIBUTES)
        self.fc_attrs_cls = nn.Linear(base_features, len(attrs_classes) + 1)
        self.fc_category = nn.Linear(base_features, N_CATEGORIES)

    def forward(self, x):
        x_attrs = self.fc_attrs(x)
        x_attrs_cls = self.fc_attrs_cls(x)
        x_category = self.fc_category(x)
        return x_attrs, x_attrs_cls, x_category


def get_loss(y_pred, y_true):
    pred_attrs, pred_attrs_cls, pred_category = y_pred
    true_attrs, true_attrs_cls, true_category = y_true
    loss_attrs = F.binary_cross_entropy_with_logits(pred_attrs, true_attrs)
    loss_attrs_cls = F.cross_entropy(pred_attrs_cls, true_attrs_cls)
    loss_category = F.cross_entropy(pred_category, true_category)
    loss = loss_attrs + 2 * loss_attrs_cls + loss_category
    return loss, {
        'loss': loss,
        'loss_attrs': loss_attrs,
        'loss_attrs_cls': loss_attrs_cls,
        'loss_category': loss_category,
    }


class FashionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, attrs_classes: List[str],
                 image_root: Path, training: bool,
                 size: Tuple[int, int], debug=False):
        self.df = df
        self.attrs_classes = attrs_classes
        self.attrs_classes_idx = {x: idx for idx, x in enumerate(attrs_classes)}
        self.image_root = image_root
        self.training = training
        self.debug = debug
        self.size = size
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5 if self.training else 0),
            transforms.Resize((self.size[1], self.size[0])),
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = self.load_crop(item)
        # TODO flexible crop with small rotation if self.training
        image = self.pad_crop(image)
        attrs_cls_idx = self.attrs_classes_idx.get(
            item.attributes, len(self.attrs_classes) + 1)
        attributes_ohe = torch.zeros(N_ATTRIBUTES)
        for attr_id in map(int, item.attributes.split('_')):
            attributes_ohe[attr_id] = 1
        category_idx = int(item.category)
        image = self.transform(image)
        if self.debug:
            image.save(f'image_{item.ImageId}_{item.ClassId}.jpeg')
        targets = (attributes_ohe, attrs_cls_idx, category_idx)
        return self.to_tensor(image), targets

    def pad_crop(self, image):
        w, h = image.size
        target_ratio = self.size[1] / self.size[0]
        if h / w > target_ratio:
            new_w = int(round(h / target_ratio))
            new_h = h
        else:
            new_h = int(round(w * target_ratio))
            new_w = w
        assert new_w >= w and new_h >= h
        new_image = Image.new('RGB', (new_w, new_h), color='white')
        paste_box = (new_w - w) // 2, (new_h - h) // 2
        new_image.paste(image, paste_box)
        return new_image

    def load_crop(self, item):
        cached_path = (self.image_root / '_crop_cache' / 'v1' / (
            hashlib.md5((item.ImageId + item.EncodedPixels).encode('ascii'))
            .hexdigest() + '.jpeg'))
        if cached_path.exists():
            return Image.open(cached_path).convert('RGB')
        image = self.load_image(item.ImageId)
        x0, y0, x1, y1 = get_item_box(item, image.size)
        width = x1 - x0
        height = y1 - y0
        margin_w = 0.1 * width
        margin_h = 0.1 * height
        box = (max(0, int(round(x0 - margin_w))),
               max(0, int(round(y0 - margin_h))),
               min(image.size[0], int(round(x1 + margin_w))),
               min(image.size[1], int(round(y1 + margin_h))))
        image = image.crop(box)
        cached_path.parent.mkdir(exist_ok=True, parents=True)
        image.save(cached_path)
        return image

    def load_image(self, image_id: str) -> Image.Image:
        return Image.open(self.image_root / image_id).convert('RGB')


Box = Tuple[int, int, int, int]


def get_item_box(item, size) -> Box:
    width, height = size
    if hasattr(item, 'Width'):
        assert size == (item.Width, item.Height)
    pixel_list = list(map(int, item.EncodedPixels.split(' ')))
    xmax = ymax = 0
    xmin, ymin = width, height
    for i in range(0, len(pixel_list), 2):
        start_index = pixel_list[i] - 1
        index_len = pixel_list[i + 1] - 1
        if index_len == 0:
            continue
        end_index = start_index + index_len - 1  # inclusive
        x0 = start_index // height
        y0 = start_index % height
        xmin = min(x0, xmin)
        ymin = min(y0, ymin)
        x1 = end_index // height
        y1 = end_index % height
        xmax = max(x1, xmax)
        ymax = max(y1, ymax)
        if x0 != x1:
            ymin = 0
            ymax = height - 1
    return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    main()

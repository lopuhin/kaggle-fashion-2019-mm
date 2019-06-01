from collections import defaultdict
import json
from pathlib import Path
import pickle

import pandas as pd
from PIL import Image
import numpy as np
import tqdm

from mmdet.datasets.custom import CustomDataset


DATA_ROOT = Path(__file__).parent.parent / 'data'


class FashionDataset(CustomDataset):
    def get_ann_info(self, idx):
        info = self.img_infos[idx]
        ann = info['ann']
        masks = np.array([
            get_item_mask_box(rle, info['width'], info['height'])[0]
            for rle in ann['rle_masks']])
        return dict(ann, masks=masks)


def main():
    folds = json.loads((DATA_ROOT / 'folds.json').read_text())
    valid_image_ids = folds[0]
    df = load_train_df()
    valid_index = df['ImageId'].isin(valid_image_ids)
    df_train = df[~valid_index]
    df_valid = df[valid_index]
    df_test = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    with (DATA_ROOT / 'mm_train.pkl').open('wb') as f:
        pickle.dump(convert_train_ann(df_train), f)
    with (DATA_ROOT / 'mm_valid.pkl').open('wb') as f:
        pickle.dump(convert_train_ann(df_valid), f)
    with (DATA_ROOT / 'mm_test.pkl').open('wb') as f:
        pickle.dump(convert_test_ann(df_test), f)


def convert_train_ann(df):
    items_by_image_id = defaultdict(list)
    for x in df.itertuples():
        items_by_image_id[x.ImageId].append(x)
    annotations = []
    for image_id, items in tqdm.tqdm(sorted(items_by_image_id.items())):
        bboxes, masks, labels = [], [], []
        width = items[0].Width
        height = items[0].Height
        for i, item in enumerate(items):
            _, box = get_item_mask_box(item.EncodedPixels, width, height)
            bboxes.append(box)
            masks.append(item.EncodedPixels)
            label, *_ = map(int, str(item.ClassId).split('_'))
            labels.append(label + 1)  # background is 0
        annotations.append({
            'filename': image_id,
            'width': width,
            'height': height,
            'ann': {
                'bboxes': np.array(bboxes, dtype=np.float32),
                'labels': np.array(labels),
                'rle_masks': masks,
            },
        })
    return annotations


def convert_test_ann(df):
    annotations = []
    for item in tqdm.tqdm(df.itertuples(), total=len(df)):
        image = Image.open(DATA_ROOT / 'test' / item.ImageId)
        annotations.append({
            'filename': item.ImageId,
            'width': image.width,
            'height': image.height,
        })
    return annotations


def get_item_mask_box(rle, width, height):
    mask = np.zeros(width * height, dtype=np.uint8)
    pixel_list = list(map(int, rle.split(' ')))
    xmax = ymax = 0
    xmin, ymin = width, height
    for i in range(0, len(pixel_list), 2):
        start_index = pixel_list[i] - 1
        index_len = pixel_list[i + 1] - 1
        if index_len == 0:
            continue
        end_index = start_index + index_len - 1  # inclusive
        mask[start_index: end_index + 1] = 1
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
    mask = mask.reshape((height, width), order='F')
    return mask, (xmin, ymin, xmax, ymax)


def load_train_df(root: Path = DATA_ROOT) -> pd.DataFrame:
    pkl_path = root / 'train.pkl'
    if pkl_path.exists():
        df = pd.read_pickle(pkl_path)
    else:
        df = pd.read_csv(root / 'train.csv')
        df.to_pickle(pkl_path)
    bad_image_ids = [  # size in df does not match size in the image
        '2ab8c02ce17612733ddee218b4ce1fd1.jpg',
        'f4d6e71fbffc3e891e5009fef2c8bf6b.jpg',
    ]
    df = df[~df['ImageId'].isin(bad_image_ids)]
    return df


if __name__ == '__main__':
    main()

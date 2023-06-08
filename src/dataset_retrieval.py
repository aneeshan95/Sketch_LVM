import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]

class Sketchy(torch.utils.data.Dataset):

    def __init__(
        self, opts, transform, mode='train',
            instance_level=False, used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.instance_level = instance_level
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(
            self.opts.data_dir, 'Sketchy', 'sketch', 'tx_000000000000'
        ))
        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes))
            else:
                self.all_categories = unseen_classes

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(
                self.opts.data_dir, 'Sketchy', 'sketch', 'tx_000000000000', category, '*.png'
            )))
            self.all_photos_path[category] = glob.glob(os.path.join(
                self.opts.data_dir, 'EXTEND_image_sketchy', category, '*.jpg'
            ))

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]
        sk_path = filepath

        filepath, filename = os.path.split(filepath)
        category = os.path.split(filepath)[-1]

        if self.instance_level:
            img_path = os.path.join(
                self.opts.data_dir, 'Sketchy', 'photo', 'tx_000000000000', category, filename.split('-')[0]+'.jpg'
            )
            neg_path = np.random.choice(self.all_photos_path[category])

        else:
            img_path = np.random.choice(self.all_photos_path[category])
            neg_path = np.random.choice(self.all_photos_path[np.random.choice(self.all_categories)])

        sk_data = Image.open(sk_path).convert('RGB')
        img_data = Image.open(img_path).convert('RGB')
        neg_data = Image.open(neg_path).convert('RGB')

        sk_data = ImageOps.pad(sk_data, size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(img_data, size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(neg_data, size=(self.opts.max_size, self.opts.max_size))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)

    dataset_train = Sketchy(opts, dataset_transforms, mode='train',
        instance_level=opts.instance_level, return_orig=True)

    dataset_val = Sketchy(opts, dataset_transforms, mode='val',
        instance_level=opts.instance_level, used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1

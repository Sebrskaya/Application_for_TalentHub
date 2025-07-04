import os
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import random


class WeldDataset(Dataset):
    def __init__(self, dataset_path = "K:\\Diploma_new\\datasets\\al5083\\weld", is_train=True, resize=256, cropsize=224):
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # Загрузка данных
        self.x, self.y = self.load_dataset()

        # Трансформации
        self.transform = T.Compose([
            T.Resize(resize, Image.Resampling.LANCZOS),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.is_train:
            (img_path, defect_name, _), label = self.x[idx], self.y[idx]

            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            return img, label  # без маски
        
        else:
            (img_path, defect_name, mask_path), label = self.x[idx], self.y[idx]

            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
            else:
                #print(f"⚠️ Mask not found for: {img_path}")
                mask = Image.new('L', (self.resize, self.resize), 0)  # чёрная маска

            mask = T.Resize(self.resize, Image.Resampling.NEAREST)(mask)
            mask = T.CenterCrop(self.cropsize)(mask)
            mask = T.ToTensor()(mask)
            mask = (mask > 0.5).float()  # Бинаризация

        return img, label, mask, defect_name

    def load_dataset(self):
        x, y = [], []
        phase = 'train' if self.is_train else 'test'

        if self.is_train:
            # Ожидаем, что внутри train только изображения
            train_dir = os.path.join(self.dataset_path, 'train', 'sorted_images', 'good_weld')
            img_fpaths = [os.path.join(train_dir, f)
                        for f in os.listdir(train_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            x.extend([(f, 'good_weld', None) for f in img_fpaths])
            y.extend([0] * len(img_fpaths))  # метка 0 — нормальный шов

        else:
            test_img_root = os.path.join(self.dataset_path, 'test', 'Video_frames')
            mask_root = os.path.join(self.dataset_path, 'test', 'gt_video_frames')

            for defect_type in os.listdir(test_img_root):
                img_dir = os.path.join(test_img_root, defect_type)
                mask_dir = os.path.join(mask_root, defect_type)

                if not os.path.isdir(img_dir):
                    continue

                for f in os.listdir(img_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(img_dir, f)
                        mask_path = os.path.join(mask_dir, f.replace('.png', '_mask.png'))  # или другое соответствие
                        x.append((img_path, defect_type, mask_path))
                        if defect_type == 'good_weld':
                            y.append(0)  # 0 (нормальный)
                        else:
                            y.append(1)  # 1 (аномалия)
        return x, y


class WeldDatasetClass(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Классы и соответствующие им индексы
        self.class_names = {
            'good_weld': 0,
            'burn_through': 1,
            'contamination': 2,
            'lack_of_fusion': 3,
            'misalignment': 4,
            'lack_of_penetration': 5
        }

        self.classes = list(self.class_names.keys())

        # Сканируем папки классов
        self.samples = []
        for class_name, class_idx in self.class_names.items():
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_path, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import paddle.vision.transforms as transforms
import numpy as np
import cv2


class MVTecDataset(Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = np.array(image)

        image = self.image_transform(image)
        
        if self.is_train:
            return image, image_file
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = paddle.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                )
                target = np.array(target)
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    paddle.set_device('gpu:1')
    train_dataset = MVTecDataset(
        root='./mvtec-ad',
        category='bottle',
        input_size=256,
        is_train=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    for data, file in train_dataloader:
        print(data.shape)
        print(file)
        break
    
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from glob import glob

from PIL import Image
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import paddle.vision.transforms as transforms
import numpy as np
import cv2


class MVTecDataset(Dataset):
    '''
    MVTec AD dataset
    '''
    def __init__(self, root, category, input_size, is_train=True):
        #image transform
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:  ### images for training
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:         ### images for testing
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            
            ### transform for groundtruth mask
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        ### load image
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = np.array(image)

        ### image transform
        image = self.image_transform(image)
        
        
        if self.is_train: ### train
            return image, image_file
        else:             ### test
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

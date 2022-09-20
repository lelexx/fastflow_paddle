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


CHECKPOINT_DIR = "results"  ### save model params and logs in CHECKPOINT_DIR
device = 'gpu:0'

### all categories in MVTec AD dataset
MVTEC_CATEGORIES = [
    "toothbrush",
    "screw",
    "hazelnut",
    "transistor",
    "tile",
    "pill",
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "leather",
    "metal_nut",
    "wood",
    "zipper",
]

## encoder name
BACKBONE_RESNET18 = "resnet18"

### train batchsize 
TRAIN_BATCH_SIZE = 32
### test batch size
TEST_BATCH_SIZE = 40

### number of epoches
NUM_EPOCHS = 500

### learning rate
LR = 1e-3
### weight decay
WEIGHT_DECAY = 1e-5
###clip Norm
CLIP_NORM= 0.9


LOG_INTERVAL = 10
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 1

### early stopping
PIXEL_AUROC_RATIO = 0.5
IMAGE_AUROC_RATIO = 0.5
PATIENCE = 30
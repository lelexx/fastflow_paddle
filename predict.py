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


import argparse
import os
import datetime
import time
import yaml
from tqdm import tqdm
import constants as const
import dataset
import fastflow
import utils
import numpy as np
import cv2 
import paddle
import paddle.nn as nn
import random
import warnings
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
import paddle.vision.transforms as transforms
from PIL import Image

from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

paddle.set_device(const.device)

def build_model_paddle(config):
    '''
    create fastflow model
    '''
    model = fastflow.FastFlow(
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
        use_norm=config['use_norm'],
        momentum = config['momentum'],
        channels = config['channels'],
        scales = config['scales'],
        clamp = config['clamp'],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if not p.stop_gradient)
        )
    )

    return model



def predict(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model_paddle(config)
    model.eval()
    
    checkpoint_path =  os.path.join(const.CHECKPOINT_DIR, args.exp_dir, args.category, 'best.pdparams')
    checkpoint = paddle.load(checkpoint_path)
    
    image_transform = transforms.Compose(
        [
            transforms.Resize(config["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ### load pretrained checkpoint 
    model.set_dict(checkpoint["model_state_dict"])
    
    image = Image.open(args.image_path)
    image = np.array(image)
    image = image_transform(image).unsqueeze(0)
    
    ## forward
    with paddle.no_grad():
        ret = model(image)

    output = ret["anomaly_map"].detach().cpu()
    output = output.numpy()[0]
    
    ## post process
    image_score = np.max(output)
    if image_score > args.image_threshold:
        print('Anomaly - score:  {:.3f}'.format(image_score))
    else:
        print('Normal - score:  {:.3f}'.format(image_score))
    output = gaussian_filter(output, sigma=6)[0]
    predict_map = (output > args.pixel_threshold).astype(np.float32)
    
    ## save detection map in ./output/lele.jpg
    save_image = np.concatenate((output, predict_map), axis = 0) * 255
    cv2.imwrite('./output/lele.jpg', save_image.astype(np.uint8))
    

def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, default='configs/resnet18.yaml', help="path to config file"
    )
    parser.add_argument("--image_path", type=str, required=True, help="path to image")
    parser.add_argument("--image_threshold", type=float, default= 0.8, help="image_level threshold")
    parser.add_argument("--pixel_threshold", type=float, default= 0.8, help="pixel_level threshold")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        default='bottle',
        help="category name in mvtec",
    )
    parser.add_argument(
        "--exp_dir", type=str, help="exp_dir to load checkpoint"
    )

    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    predict(args)

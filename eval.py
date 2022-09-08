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


from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

paddle.set_device(const.device)

def build_test_data_loader_paddle(args, config):
    '''
    create test dataloader
    '''
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )

    return paddle.io.DataLoader(
        test_dataset,
        batch_size=const.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


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


def eval_once(dataloader, model, eval = True):
    '''
    function:eval on test dataset
    
    return:
        auroc_px: pixel_level auc
        auroc_sp: image_level auc
    '''
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    preds = []
    gts = []

    for data, targets in dataloader:
        targets = targets.cpu().numpy().astype(int)

        with paddle.no_grad():
            ret = model(data)

        outputs = ret["anomaly_map"].detach().cpu()
        outputs = outputs.numpy()

        preds.append(outputs)
        gts.append(targets)
    
    preds = np.concatenate(preds, axis = 0)
    gts = np.concatenate(gts, axis = 0)

    targets = gts

    outputs = preds
    for i in range(targets.shape[0]):
        gt_list_sp.append(np.max(targets[i]))
        pr_list_sp.append(np.max(outputs[i]))
        if eval:
            outputs[i] = gaussian_filter(outputs[i], sigma=6)

        gt_list_px.extend(targets[i].ravel())
        pr_list_px.extend(outputs[i].ravel())
        
    auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

    print("EVAL:  image-auc : {:.6f} pixel-auc:{:.6f}".format(auroc_sp, auroc_px))
    
    return auroc_px, auroc_sp


def evaluate(args):
    '''
    evaluate on test dataset
    '''
    config = yaml.safe_load(open(args.config, "r"))
    ### build model
    model = build_model_paddle(config)
    pixel_auroc_dict, image_auroc_dict = {}, {}
    pixel_auroc_mean, image_auroc_mean = 0, 0
    
    ## evaluate on categories
    for category in const.MVTEC_CATEGORIES if args.category == 'all' else [args.category]:
        args.category = category
        checkpoint_path_last = os.path.join(const.CHECKPOINT_DIR, args.exp_dir, category, 'last.pdparams')
        checkpoint_path_best = os.path.join(const.CHECKPOINT_DIR, args.exp_dir, category, 'best.pdparams')
        checkpoint = paddle.load(checkpoint_path_last if args.is_last else checkpoint_path_best)

        ## load pretrained checkpoint
        model.set_dict(checkpoint["model_state_dict"])
        
        ## buils testdataset
        test_dataloader = build_test_data_loader_paddle(args, config)
        ## eval
        pixel_auroc, image_auroc = eval_once(test_dataloader, model, eval=True)
        
        pixel_auroc_dict[category] = pixel_auroc
        image_auroc_dict[category] = image_auroc
        pixel_auroc_mean += pixel_auroc
        image_auroc_mean += image_auroc
    pixel_auroc_mean /= len(pixel_auroc_dict)
    image_auroc_mean /= len(image_auroc_dict)
    ### print eval results
    for category in pixel_auroc_dict.keys():
        print("{:15}  pixel auroc:{:.3f}   image auroc:{:.3f}".format(category, pixel_auroc_dict[category], image_auroc_dict[category]))
        
    print("{:15}  pixel auroc:{:.3f}   image auroc:{:.3f}".format('all', pixel_auroc_mean, image_auroc_mean))


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, default='configs/resnet18.yaml', help="path to config file"
    )
    parser.add_argument("--data", type=str, default='./data', help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES + ['all'],
        default='bottle',
        help="category name in mvtec",
    )
    parser.add_argument("--is_last", action="store_true", help="load last param")
    parser.add_argument(
        "--exp_dir", type=str, default = 'exp', help="path to load checkpoint"
    )

    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
